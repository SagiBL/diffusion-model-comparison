import numpy as np
from diffusion_interface import DiffusionModel

class CoDiGDiffusionModel(DiffusionModel):
    """
    CoDiG (Constraint-Aware Diffusion Guidance) Model.
    
    This model incorporates a barrier function guidance during sampling to avoid obstacles.
    It follows the structure of the MockDiffusionModel for the underlying dynamics 
    but adds the specific CoDiG sampling logic.
    """
    def __init__(self, target_pos, obstacles, n_steps=50, codig_scale=1.0):
        """
        Args:
            target_pos (array): Destination.
            obstacles (list of tuples): List of (x, y, radius) for circular obstacles.
            n_steps (int): Sampling steps.
            codig_scale (float): Strength of the barrier guidance.
        """
        self.target_pos = np.array(target_pos)
        self.obstacles = obstacles
        self.n_steps = n_steps
        self.codig_scale = codig_scale
        self.current_state = None

    def set_initial_noise(self, noise: np.ndarray):
        self.current_state = np.array(noise)

    def _compute_barrier_gradient(self, state):
        """
        Computes the gradient of the barrier function sum( log( dist(x, obs) - radius ) )
        or similar repulsion.
        
        Using a simple repulsion: 1/dist^2 or similar when close.
        Let's use: if dist < safe_dist, grad = normal * (safe_dist - dist) * scale?
        
        Standard CoDiG uses Gradient of Barrier.
        Barrier B(x) -> infinity as x approaches boundary.
        L_barrier = sum( exp( - (dist - radius)/temperature ) ) ?
        
        Let's use a simple repulsion force for the mock:
        Force = \sum normalized_vec_away * (1 / (dist - radius + epsilon))
        """
        grad = np.zeros_like(state)
        for obs in self.obstacles:
            ox, oy, r = obs
            obs_center = np.array([ox, oy])
            
            diff = state - obs_center
            dist = np.linalg.norm(diff)
            
            # Distance to surface
            surface_dist = dist - r
            
            if surface_dist < 0.5: # activation threshold
                # Repulsion direction
                direction = diff / (dist + 1e-6)
                
                # Magnitude: increasing as we get closer
                # 1 / surface_dist
                # Avoid division by zero
                mag = 1.0 / (max(surface_dist, 0.01))
                
                grad += direction * mag
                
        return grad

    def get_denoising_trajectory(self, guidance_func=None, guidance_scale=0.1):
        import time
        start_time = time.time()
        
        if self.current_state is None:
            raise ValueError("Initial noise not set.")
            
        current_state = self.current_state.copy()
        path = [current_state.copy()]
        
        for t in range(self.n_steps):
            noise = np.random.randn(*current_state.shape) * 0.1
            
            # 1. Standard Reverse Drift (Mock dynamics towards target)
            drift = -0.05 * current_state 
            # (Note: In a real model this comes from the UNet prediction)
            
            # 2. External Guidance (e.g., Euclidean to target)
            ext_guidance = np.zeros_like(current_state)
            if guidance_func is not None:
                ext_guidance = guidance_func(current_state) * guidance_scale
                
            # 3. CoDiG (Constraint) Guidance
            # "Integrates barrier functions into the denoising process"
            barrier_grad = self._compute_barrier_gradient(current_state)
            codig_guidance = barrier_grad * self.codig_scale
            
            # Combine
            next_state = current_state + drift + ext_guidance + codig_guidance + noise
            current_state = next_state
            path.append(current_state.copy())
            
        end_time = time.time()
        return {'trajectory': np.array(path), 'time': end_time - start_time}
