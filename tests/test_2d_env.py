from torch.utils.data import DataLoader

from locodiff.dataset import PDControlledParticleDataset

if __name__ == "__main__":
    # Create dataset with PD control
    dataset = PDControlledParticleDataset(
        num_samples=100,
        trajectory_length=64,
        grid_size=1.0,
        process_noise=0.1,
        measurement_noise=0.01,
        init_pos_var=0.01,
        kp=4.0,
        kd=2.0,
        dt=0.02,
        seed=42,
    )

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # Visualize a batch
    dataset.visualize_batch(100)

    # Iterate through the dataloader
    for batch in dataloader:
        # Access batch['trajectory], batch['start_corner'], etc.
        print(f"Batch trajectories shape: {batch['obs'].shape}")
        print(f"Batch start corners shape: {batch['action'].shape}")
        break  # Just show the first batch
