import numpy as np
import matplotlib.pyplot as plt
import argparse
from MockDiffusionModel import MockDiffusionModel

def euclidean_guidance_direction(state, target_pos):
    """
    Returns the vector pointing from state to target.
    """
    diff = target_pos - state
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return np.zeros_like(state)
    return diff / dist # Normalized direction

def get_distance_trace(path, target_pos):
    """Calculates Euclidean distance to target for each step in path."""
    distances = []
    for step in path:
        dist = np.linalg.norm(step - target_pos)
        distances.append(dist)
    return distances

def main():
    parser = argparse.ArgumentParser(description="Diffusion Model Comparison")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    args = parser.parse_args()

    # Setup
    start_pos = np.array([5.0, 5.0]) # Start far away (noise)
    target_pos = np.array([0.0, 0.0]) # Target
    
    # Initialize Pretrained Model (The physics/neural net)
    from PretrainedModel import PretrainedDiffusionModel
    pretrained_model = PretrainedDiffusionModel()
    
    # Initialize Samplers with the Shared Pretrained Model
    
    # 1. Baseline Sampler (MockDiffusionModel)
    model = MockDiffusionModel(pretrained_model, target_pos, n_steps=args.steps)
    
    print("Running Un-guided Sampling...")
    # Unguided
    model.set_initial_noise(start_pos)
    result_unguided = model.get_denoising_trajectory(guidance_func=None)
    unguided_path = result_unguided['trajectory']
    print(f"Unguided time: {result_unguided['time']:.4f}s")
    
    print("Running Guided Sampling...")
    # Guided
    guidance_fn = lambda x: euclidean_guidance_direction(x, target_pos)
    model.set_initial_noise(start_pos)
    result_guided = model.get_denoising_trajectory(guidance_func=guidance_fn, guidance_scale=0.2)
    guided_path = result_guided['trajectory']
    print(f"Guided time: {result_guided['time']:.4f}s")
    
    # Calculate Distance between Guided and Unguided samples at each step
    # unguided_path and guided_path should have the same shape (Steps, Dim)
    diffs = unguided_path - guided_path
    dists_between_samples = np.linalg.norm(diffs, axis=1)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    iterations = range(len(dists_between_samples))
    
    plt.plot(iterations, dists_between_samples, label='Distance (Guided vs Unguided)', linewidth=2, color='purple')
    
    plt.title('Euclidean Distance between Guided and Unguided Samples')
    plt.xlabel('Sampling Iteration')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    output_file = "distance_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    # --- CoDiG Test ---
    from CoDiGDiffusionModel import CoDiGDiffusionModel
    print("\nRunning CoDiG Sampling...")
    
    obstacles = [(2.5, 2.5, 1.0)] # Obstacle at (2.5, 2.5) with radius 1.0
    codig_model = CoDiGDiffusionModel(pretrained_model, target_pos, obstacles, n_steps=args.steps, codig_scale=0.1)
    
    codig_model.set_initial_noise(start_pos)
    # Run with target guidance AND obstacle avoidance
    result_codig = codig_model.get_denoising_trajectory(guidance_func=guidance_fn, guidance_scale=0.2)
    codig_path = result_codig['trajectory']
    print(f"CoDiG time: {result_codig['time']:.4f}s")
    
    # --- HardFlow Test ---
    from HardFlowDiffusionModel import HardFlowDiffusionModel
    print("\nRunning HardFlow Sampling...")
    
    hardflow_model = HardFlowDiffusionModel(pretrained_model, target_pos, obstacles, n_steps=args.steps)
    hardflow_model.set_initial_noise(start_pos)
    result_hardflow = hardflow_model.get_denoising_trajectory(guidance_func=guidance_fn, guidance_scale=0.2)
    hardflow_path = result_hardflow['trajectory']
    print(f"HardFlow time: {result_hardflow['time']:.4f}s")

    # --- CFG++ Test ---
    from CFGPlusPlusDiffusionModel import CFGPlusPlusDiffusionModel
    print("\nRunning CFG++ Sampling...")
    
    # High scale to demonstrate effect
    cfgpp_model = CFGPlusPlusDiffusionModel(pretrained_model, target_pos, n_steps=args.steps, cfg_scale=4.0)
    cfgpp_model.set_initial_noise(start_pos)
    result_cfgpp = cfgpp_model.get_denoising_trajectory() # Uses internal CFG scale
    cfgpp_path = result_cfgpp['trajectory']
    print(f"CFG++ time: {result_cfgpp['time']:.4f}s")

    # --- Langevin MCMC Test ---
    from LangevinMCMCDiffusionModel import LangevinMCMCDiffusionModel
    print("\nRunning Langevin MCMC Sampling...")
    
    # Using the obstacle and target interaction as the 'Verifier'
    langevin_model = LangevinMCMCDiffusionModel(pretrained_model, target_pos, obstacles, n_steps=args.steps, mcmc_steps_per_iter=10, step_size=0.05)
    langevin_model.set_initial_noise(start_pos)
    result_langevin = langevin_model.get_denoising_trajectory() 
    langevin_path = result_langevin['trajectory']
    print(f"Langevin MCMC time: {result_langevin['time']:.4f}s")

    # Plot Trajectories to see obstacle avoidance
    plt.figure(figsize=(10, 10))
    plt.plot(unguided_path[:, 0], unguided_path[:, 1], 'k--', label='Unguided', alpha=0.3)
    plt.plot(guided_path[:, 0], guided_path[:, 1], 'b-', label='Euclidean Guided')
    plt.plot(codig_path[:, 0], codig_path[:, 1], 'r-', label='CoDiG')
    plt.plot(hardflow_path[:, 0], hardflow_path[:, 1], 'g-', label='HardFlow')
    plt.plot(cfgpp_path[:, 0], cfgpp_path[:, 1], 'm-', label='CFG++')
    plt.plot(langevin_path[:, 0], langevin_path[:, 1], 'c-', label='Langevin MCMC', linewidth=2)
    
    # Draw Obstacle
    circle = plt.Circle((obstacles[0][0], obstacles[0][1]), obstacles[0][2], color='r', alpha=0.3)
    plt.gca().add_patch(circle)
    
    plt.scatter([start_pos[0]], [start_pos[1]], c='g', marker='o', label='Start')
    plt.scatter([target_pos[0]], [target_pos[1]], c='gold', marker='*', s=200, label='Target')
    
    plt.title('Trajectory Comparison: All Methods')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig("all_methods_comparison_v2.png")
    print("Comparison plot saved to all_methods_comparison_v2.png")


if __name__ == "__main__":
    main()
