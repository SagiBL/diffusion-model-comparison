import numpy as np
from diffusion_interface import DiffusionModel

class HardFlowDiffusionModel(DiffusionModel):
    """
    HardFlow Model (Mock Implementation).
    Uses the pretrained model for the initial rollout, then optimizes it.
    """
    def __init__(self, pretrained_model, target_pos, obstacles, n_steps=50):
        """
        Args:
            pretrained_model: The shared base model.
            target_pos (array): Destination.
            obstacles (list of tuples): List of (x, y, radius) for circular obstacles.
            n_steps (int): Sampling steps.
        """
        self.pretrained_model = pretrained_model
        self.target_pos = np.array(target_pos)
        self.obstacles = obstacles
        self.n_steps = n_steps
        self.current_state = None

    def set_initial_noise(self, noise: np.ndarray):
        self.current_state = np.array(noise)

    def _project_point_to_boundary(self, point, obs_center, radius, margin=0.05):
        """Projects a point to be outside the obstacle circle."""
        diff = point - obs_center
        dist = np.linalg.norm(diff)
        if dist < radius + margin:
            # Inside or too close
            if dist < 1e-6:
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            return obs_center + direction * (radius + margin)
        return point

    def get_denoising_trajectory(self, guidance_func=None, guidance_scale=0.1):
        import time
        start_time = time.time()
        
        if self.current_state is None:
            raise ValueError("Initial noise not set.")
            
        # 1. Initial Rollout (Unconstrained)
        # We simulate the base process first to get the initial "Reference Trajectory"
        current_state = self.current_state.copy()
        
        path_states = [current_state.copy()]
        for t in range(self.n_steps):
            noise = np.random.randn(*current_state.shape) * 0.1
            
            # Use shared pretrained model
            drift = self.pretrained_model.predict_drift(current_state, t)
            
            ext_guidance = np.zeros_like(current_state)
            if guidance_func is not None:
                ext_guidance = guidance_func(current_state) * guidance_scale
            
            # Standard update
            next_state = current_state + drift + ext_guidance + noise
            current_state = next_state
            path_states.append(current_state.copy())
            
        trajectory = np.array(path_states) # Shape (N+1, 2)
        
        # 2. Trajectory Optimization Loop (Hard Constraints)
        # Iteratively satisfying constraints while keeping shape
        n_opt_steps = 20
        alpha = 0.5 # Smoothing factor
        
        for _ in range(n_opt_steps):
            trajectory_new = trajectory.copy()
            violation_found = False
            
            # A. Constraint Projection
            for i in range(len(trajectory)):
                for obs in self.obstacles:
                    ox, oy, r = obs
                    trajectory_new[i] = self._project_point_to_boundary(
                        trajectory_new[i], np.array([ox, oy]), r
                    )
                    
            # B. Smoothness / Flow Consistency
            # We want x_t to be close to (x_{t-1} + x_{t+1})/2, or simply close to original reference.
            # But the reference was colliding.
            # Let's just smooth local points to avoid jagged "teleportation" out of obstacles.
            # Simple moving average for smoothness, but anchor start/end?
            # Start is fixed (x_T). End is free-ish (but aims for target).
            
            # In real HardFlow, we solve min ||x - x_ref|| s.t. x in SafeSet.
            # x_new is roughly that projection.
            
            # To prevent it from looking jagged, we can do a slight smoothing pass
            # except for the start point
            for i in range(1, len(trajectory) - 1):
                # Simple relaxation
                smoothed = 0.5 * trajectory_new[i] + 0.25 * trajectory_new[i-1] + 0.25 * trajectory_new[i+1]
                trajectory_new[i] = smoothed
                
            # Re-project after smoothing to ensure Hard Constraint is dominant
            for i in range(len(trajectory)):
                for obs in self.obstacles:
                    ox, oy, r = obs
                    trajectory_new[i] = self._project_point_to_boundary(
                        trajectory_new[i], np.array([ox, oy]), r
                    )
            
            trajectory = trajectory_new

        end_time = time.time()
        return {'trajectory': trajectory, 'time': end_time - start_time}
