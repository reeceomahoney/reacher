import logging
import os
import statistics
import time

import hydra
import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state
from tqdm import tqdm, trange

import wandb
from locodiff.dataset import get_dataloaders
from locodiff.envs import MazeEnv
from locodiff.policy import DiffusionPolicy
from locodiff.utils import ExponentialMovingAverage, Normalizer

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
        normalizer = Normalizer(self.train_loader, agent_cfg.scaling, device)
        model = hydra.utils.instantiate(self.cfg.model)
        classifier = hydra.utils.instantiate(self.cfg.classifier)
        self.policy = DiffusionPolicy(
            model, normalizer, env, **self.cfg.policy, classifier=classifier
        )

        # ema
        self.ema_helper = ExponentialMovingAverage(
            self.policy.parameters(), self.cfg.ema_decay, self.cfg.device
        )
        self.use_ema = agent_cfg.use_ema

        # variables
        if isinstance(env, VecEnv):
            self.num_steps_per_env = int(
                self.env.cfg.episode_length_s  # type: ignore
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
        self.policy.reset()
        self.train_mode()

        start_iter = self.current_learning_iteration
        tot_iter = int(start_iter + self.cfg.num_iters)
        generator = iter(self.train_loader)
        for it in trange(start_iter, tot_iter):
            start = time.time()

            # evaluation
            if it % self.cfg.eval_interval == 0:
                test_mse = []
                plot = True
                for batch in tqdm(self.test_loader, desc="Testing...", leave=False):
                    mse = self.policy.test_classifier(batch, plot)
                    plot = False
                    test_mse.append(mse)
                test_mse = statistics.mean(test_mse)

            # training
            try:
                batch = next(generator)
            except StopIteration:
                generator = iter(self.train_loader)
                batch = next(generator)

            loss = self.policy.update_classifier(batch)
            self.ema_helper.update(self.policy.parameters())

            # logging
            self.current_learning_iteration = it
            if self.log_dir is not None and it % self.cfg.log_interval == 0:
                # timing
                stop = time.time()
                iter_time = stop - start

                self.log(locals())
                if it % self.cfg.eval_interval == 0:
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
            "model_state_dict": self.policy.model.state_dict(),
            "optimizer_state_dict": self.policy.optimizer.state_dict(),
            "norm_state_dict": self.policy.normalizer.state_dict(),
            "classifier_state_dict": self.policy.classifier.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        if self.use_ema:
            self.ema_helper.restore(self.policy.parameters())

    def load(self, path):
        loaded_dict = torch.load(path)
        self.policy.model.load_state_dict(loaded_dict["model_state_dict"])
        self.policy.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        loaded_dict["norm_state_dict"].pop("r_min")
        loaded_dict["norm_state_dict"].pop("r_max")
        self.policy.normalizer.load_state_dict(loaded_dict["norm_state_dict"], strict=False)
        self.policy.classifier.load_state_dict(loaded_dict["classifier_state_dict"])
        # self.current_learning_iteration = loaded_dict["iter"]
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
