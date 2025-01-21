import logging
import os
import statistics
import time

import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state
from tqdm import tqdm, trange

import wandb
from locodiff.dataset import get_dataloaders
from locodiff.envs import MazeEnv
from locodiff.models.unet import ConditionalUnet1D, ValueUnet1D
from locodiff.policy import DiffusionPolicy
from locodiff.utils import ExponentialMovingAverage, InferenceContext, Normalizer

# A logger for this file
log = logging.getLogger(__name__)


class ClassifierRunner:
    def __init__(
        self, env: VecEnv | MazeEnv, agent_cfg, log_dir: str | None = None, device="cpu"
    ):
        self.env = env
        self.env.reset()
        self.cfg = agent_cfg
        self.device = device

        # classes
        self.train_loader, self.test_loader = get_dataloaders(**self.cfg.dataset)
        self.normalizer = Normalizer(self.train_loader, agent_cfg.scaling, device)
        # TODO: init model with hydra
        model = ConditionalUnet1D(**self.cfg.model)
        classifier = ValueUnet1D(**self.cfg.model)
        # model = DiffusionTransformer(**self.cfg.model)
        self.policy = DiffusionPolicy(
            model, self.normalizer, env, **self.cfg.policy, classifier=classifier
        )

        # ema
        self.ema_helper = ExponentialMovingAverage(
            self.policy.get_params(), self.cfg.ema_decay, self.cfg.device
        )
        self.use_ema = agent_cfg.use_ema

        # variables
        if isinstance(env, VecEnv):
            self.num_steps_per_env = int(
                self.cfg.episode_length
                / (self.env.cfg.decimation * self.env.cfg.sim.dt)  # type: ignore
            )
        elif isinstance(env, MazeEnv):
            self.num_steps_per_env = int(self.cfg.episode_length / 0.1)
        self.log_dir = log_dir
        self.current_learning_iteration = 0

        # logging
        if self.log_dir is not None:
            # initialize wandb
            wandb.init(project=self.cfg.wandb_project, dir=log_dir, config=self.cfg)
            # make model directory
            os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)  # type: ignore
            # save git diffs
            store_code_state(self.log_dir, [__file__])

    def learn(self):
        obs, _ = self.env.get_observations()
        obs = obs.to(self.device)
        self.policy.reset()
        self.train_mode()  # switch to train mode (for dropout for example)

        start_iter = self.current_learning_iteration
        tot_iter = int(start_iter + self.cfg.num_iters)
        generator = iter(self.train_loader)
        for it in trange(start_iter, tot_iter):
            start = time.time()

            # evaluation
            if it % self.cfg.eval_interval == 0:
                with InferenceContext(self):
                    test_mse = []
                    for batch in tqdm(self.test_loader, desc="Testing...", leave=False):
                        mse = self.policy.test_classifier(batch)
                        test_mse.append(mse)
                    test_mse = statistics.mean(test_mse)

            # training
            try:
                batch = next(generator)
            except StopIteration:
                generator = iter(self.train_loader)
                batch = next(generator)

            loss = self.policy.update_classifier(batch)
            self.ema_helper.update(self.policy.get_params())

            # logging
            self.current_learning_iteration = it
            if self.log_dir is not None and it % self.cfg.log_interval == 0:
                # timing
                stop = time.time()
                iter_time = stop - start

                self.log(locals())
                if it % self.cfg.sim_interval == 0:
                    self.save(os.path.join(self.log_dir, "models", "model.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, "models", "model.pt"))

    def log(self, locs: dict):
        # training
        wandb.log(
            {
                "Loss/loss": locs["loss"],
                "Perf/iter_time": locs["iter_time"] / self.cfg.log_interval,
            },
            step=locs["it"],
        )
        # evaluation
        if locs["it"] % self.cfg.eval_interval == 0:
            wandb.log(
                {"Loss/test_mse": locs["test_mse"]},
                step=locs["it"],
            )

    def save(self, path, infos=None):
        if self.use_ema:
            self.ema_helper.store(self.policy.parameters())
            self.ema_helper.copy_to(self.policy.parameters())

        saved_dict = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.policy.optimizer.state_dict(),
            "norm_state_dict": self.normalizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        if self.use_ema:
            self.ema_helper.restore(self.policy.parameters())

    def load(self, path):
        loaded_dict = torch.load(path)
        self.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.normalizer.load_state_dict(loaded_dict["norm_state_dict"])
        self.policy.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.policy.to(device)
        return self.policy.act

    def train_mode(self):
        self.policy.train()

    def eval_mode(self):
        self.policy.eval()
