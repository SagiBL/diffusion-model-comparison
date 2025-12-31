import numpy as np
from diffusion_interface import DiffusionModel

class LangevinMCMCDiffusionModel(DiffusionModel):
    """
    Inference-Time Scaling via Langevin MCMC with Verifier Gradient.
    
    Ref: "Inference-Time Scaling of Diffusion Models Through Classical Search"
    
    This sampler performs 'Local Search' at each timestep (or selected steps) using 
    Langevin MCMC to refine the sample towards high-verifier-score regions.
    
    Verifier: In this mock setting, the verifier is V(x) = - (DistanceToTarget + ObstaclePenalty).
    So Gradient(Verification) is Guidance towards target + Repulsion from obstacles.
    """
    def __init__(self, pretrained_model, target_pos, obstacles, n_steps=50, mcmc_steps_per_iter=5, step_size=0.01):
        """
        Args:
            pretrained_model: The shared base model.
            target_pos: Destination.
            obstacles: List of obstacles.
            n_steps: Diffusion steps.
            mcmc_steps_per_iter: Number of Langevin steps to run *after* each diffusion step.
            step_size: Step size for Langevin dynamics.
        """
        self.pretrained_model = pretrained_model
        self.target_pos = np.array(target_pos)
        self.obstacles = obstacles
        self.n_steps = n_steps
        self.mcmc_steps_per_iter = mcmc_steps_per_iter
        self.step_size = step_size
        self.current_state = None

    def set_initial_noise(self, noise: np.ndarray):
        self.current_state = np.array(noise)

    def _compute_verifier_gradient(self, state):
        """
        Computes gradient of the Verifier Function (Energy).
        Energy E(x) ~ Distance(x, target) + ObstacleRepulsion(x)
        We want to MINIMIZE Energy, so we move in -Grad(E).
        """
        # 1. Target Attraction (Minimize distance)
        diff = self.target_pos - state
        dist = np.linalg.norm(diff)
        if dist > 1e-4:
            target_grad = diff / dist # Direction towards target
        else:
            target_grad = np.zeros_like(state)
            
        # 2. Obstacle Repulsion (Maximize distance)
        obs_grad = np.zeros_like(state)
        for obs in self.obstacles:
            ox, oy, r = obs
            obs_center = np.array([ox, oy])
            diff_obs = state - obs_center
            dist_obs = np.linalg.norm(diff_obs)
            surface_dist = dist_obs - r
            
            # Simple 1/x repulsion
            if surface_dist < 0.5:
                # Push away
                direction = diff_obs / (dist_obs + 1e-6)
                mag = 1.0 / (max(surface_dist, 0.01))
                obs_grad += direction * mag

        return target_grad + obs_grad

    def get_denoising_trajectory(self, guidance_func=None, guidance_scale=0.1):
        import time
        start_time = time.time()
        
        if self.current_state is None:
            raise ValueError("Initial noise not set.")
            
        current_state = self.current_state.copy()
        path = [current_state.copy()]
        
        for t in range(self.n_steps):
            # A. Standard Reverse Diffusion Step
            # (Predict drift, add noise)
            noise_diff = np.random.randn(*current_state.shape) * 0.1
            drift = self.pretrained_model.predict_drift(current_state, t)
            
            # Standard conditional guidance (optional, can be part of verifier too)
            ext_guidance = np.zeros_like(current_state)
            if guidance_func is not None:
                ext_guidance = guidance_func(current_state) * guidance_scale
                
            current_state = current_state + drift + ext_guidance + noise_diff
            
            # B. Langevin MCMC Refinement (Local Search)
            # Run MCMC to improve the sample w.r.t the verifier (constraints/target)
            # x_{k+1} = x_k + eps * Grad(LogP(x)) + eps * Grad(Verifier(x)) + sqrt(2*eps)*z
            
            for k in range(self.mcmc_steps_per_iter):
                # Gradient of Prior (LogP): Approximate with drift/score from model
                # In diffusion, score ~ -eps/sigma. Our 'drift' is essentially the update direction.
                # Let's use the model's drift prediction at the current MCMC state as the prior gradient.
                prior_grad = self.pretrained_model.predict_drift(current_state, t)
                
                # Gradient of Verifier
                verifier_grad = self._compute_verifier_gradient(current_state)
                
                # Langevin Update
                mcmc_noise = np.random.randn(*current_state.shape)
                
                # Update
                # Note: step_size usually should be annealed with t, but fixed for mock.
                current_state = current_state + \
                                self.step_size * (prior_grad + verifier_grad) + \
                                np.sqrt(2 * self.step_size) * mcmc_noise * 0.1 # scaled noise
            
            path.append(current_state.copy())
            
        end_time = time.time()
        return {'trajectory': np.array(path), 'time': end_time - start_time}
