import numpy as np

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
