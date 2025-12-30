import numpy as np
import matplotlib.pyplot as plt
import argparse

class MockDiffusionModel:
    """
    A mock diffusion model for testing the pipeline.
    Simulates a reverse process from noise to a target (or random walk).
    """
    def __init__(self, start_pos, target_pos, n_steps=50):
        self.start_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)
        self.n_steps = n_steps

    def sample(self, initial_state, guidance_func=None, guidance_scale=0.1):
        """
        Simulates sampling.
        Returns a list of states (path).
        """
        current_state = np.array(initial_state)
        path = [current_state.copy()]
        
        # Simple linear interpolation + noise for mock behavior
        # In a real diffusion model, this would be the reverse SDE/ODE solver
        for t in range(self.n_steps):
            # Direction towards the "true" mean (mocking the model predicting x0 or eps)
            # Let's say the model naturally drifts towards the origin magnitude, but we want it to go to target_pos
            # For the mock, let's just say it drifts randomly if unguided, or towards target if guided?
            # Actually, usually diffusion models denoise. Let's assume the "clean" data manifold is around target_pos for this mock.
            
            noise = np.random.randn(*current_state.shape) * 0.1
            
            # Unconditional update (drift towards 0,0 for example, or just random walk)
            # Let's mock it as stepping towards 0 with some noise
            drift = -0.05 * current_state 
            
            # Apply guidance if provided
            guidance = np.zeros_like(current_state)
            if guidance_func is not None:
                # Gradient of the loss with respect to x
                # We want to minimize distance, so moves AGAINST the gradient of distance?
                # Usually guidance is +scale * grad(log p(y|x))
                # If we want to minimize Energy (Distance), we move -scale * grad(E)
                # Let's define guidance_func as returning the direction we WANT to move
                guidance = guidance_func(current_state) * guidance_scale
                
            next_state = current_state + drift + guidance + noise
            current_state = next_state
            path.append(current_state.copy())
            
        return np.array(path)

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
    
    model = MockDiffusionModel(start_pos, target_pos, n_steps=args.steps)
    
    print("Running Un-guided Sampling...")
    # Unguided: guidance_func is None
    unguided_path = model.sample(start_pos, guidance_func=None)
    
    print("Running Guided Sampling...")
    # Guided: Use euclidean direction
    # We pass a lambda adapting the signature if necessary, or just the function
    # The model.sample code expects guidance_func(state) -> vector
    guidance_fn = lambda x: euclidean_guidance_direction(x, target_pos)
    guided_path = model.sample(start_pos, guidance_func=guidance_fn, guidance_scale=0.2)
    
    # Calculate Distances
    unguided_dists = get_distance_trace(unguided_path, target_pos)
    guided_dists = get_distance_trace(guided_path, target_pos)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    iterations = range(len(unguided_dists))
    
    plt.plot(iterations, unguided_dists, label='Unguided', linewidth=2, linestyle='--')
    plt.plot(iterations, guided_dists, label='Guided (Euclidean)', linewidth=2)
    
    plt.title('Distance to Target vs Sampling Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Euclidean Distance to Target')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    output_file = "distance_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
