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
    
    # Initialize model (without start_pos, as that's per-sample now)
    model = MockDiffusionModel(target_pos, n_steps=args.steps)
    
    print("Running Un-guided Sampling...")
    # Unguided
    model.set_initial_noise(start_pos)
    unguided_path = model.get_denoising_trajectory(guidance_func=None)
    
    print("Running Guided Sampling...")
    # Guided
    guidance_fn = lambda x: euclidean_guidance_direction(x, target_pos)
    model.set_initial_noise(start_pos)
    guided_path = model.get_denoising_trajectory(guidance_func=guidance_fn, guidance_scale=0.2)
    
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
    codig_model = CoDiGDiffusionModel(target_pos, obstacles, n_steps=args.steps, codig_scale=0.1)
    
    codig_model.set_initial_noise(start_pos)
    # Run with target guidance AND obstacle avoidance
    codig_path = codig_model.get_denoising_trajectory(guidance_func=guidance_fn, guidance_scale=0.2)
    
    # Plot Trajectories to see obstacle avoidance
    plt.figure(figsize=(8, 8))
    plt.plot(unguided_path[:, 0], unguided_path[:, 1], 'k--', label='Unguided')
    plt.plot(guided_path[:, 0], guided_path[:, 1], 'b-', label='Euclidean Guided')
    plt.plot(codig_path[:, 0], codig_path[:, 1], 'r-', label='CoDiG (Obstacle Avoidance)', linewidth=2)
    
    # Draw Obstacle
    circle = plt.Circle((obstacles[0][0], obstacles[0][1]), obstacles[0][2], color='r', alpha=0.3)
    plt.gca().add_patch(circle)
    
    plt.scatter([start_pos[0]], [start_pos[1]], c='g', marker='o', label='Start')
    plt.scatter([target_pos[0]], [target_pos[1]], c='gold', marker='*', s=200, label='Target')
    
    plt.title('Trajectory Comparison: CoDiG vs Baseline')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig("codig_trajectory.png")
    print("CoDiG trajectory plot saved to codig_trajectory.png")


if __name__ == "__main__":
    main()
