import numpy as np
from diffusion_interface import DiffusionModel

class MockDiffusionModel(DiffusionModel):
    """
    A mock diffusion model for testing the pipeline.
    Simulates a reverse process from noise to a target (or random walk).
    """
    def __init__(self, target_pos, n_steps=50):
        self.target_pos = np.array(target_pos)
        self.n_steps = n_steps
        self.current_state = None

    def set_initial_noise(self, noise: np.ndarray):
        """Sets the starting state."""
        self.current_state = np.array(noise)

    def get_denoising_trajectory(self, guidance_func=None, guidance_scale=0.1):
        """
        Simulates sampling from the stored initial state.
        Returns a dictionary with 'trajectory' and 'time'.
        """
        import time
        start_time = time.time()
        
        if self.current_state is None:
            raise ValueError("Initial noise not set. Call set_initial_noise() first.")
            
        current_state = self.current_state.copy()
        path = [current_state.copy()]
        
        for t in range(self.n_steps):
            noise = np.random.randn(*current_state.shape) * 0.1
            drift = -0.05 * current_state 
            
            guidance = np.zeros_like(current_state)
            if guidance_func is not None:
                guidance = guidance_func(current_state) * guidance_scale
                
            next_state = current_state + drift + guidance + noise
            current_state = next_state
            path.append(current_state.copy())
            
        end_time = time.time()
        return {'trajectory': np.array(path), 'time': end_time - start_time}
