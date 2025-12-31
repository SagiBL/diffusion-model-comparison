import numpy as np
from diffusion_interface import DiffusionModel

class MockDiffusionModel(DiffusionModel):
    """
    A basic sampler that uses the pretrained model's drift directly.
    """
    def __init__(self, pretrained_model, target_pos, n_steps=50):
        self.pretrained_model = pretrained_model
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
            
            # Use the shared pretrained model for the physics/drift
            drift = self.pretrained_model.predict_drift(current_state, t)
            
            guidance = np.zeros_like(current_state)
            if guidance_func is not None:
                guidance = guidance_func(current_state) * guidance_scale
                
            next_state = current_state + drift + guidance + noise
            current_state = next_state
            path.append(current_state.copy())
            
        end_time = time.time()
        return {'trajectory': np.array(path), 'time': end_time - start_time}
