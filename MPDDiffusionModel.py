import numpy as np
from diffusion_interface import DiffusionModel

class MPDDiffusionModel(DiffusionModel):
    """
    Motion Planning Diffusion (MPD).
    
    Ref: "Motion Planning Diffusion: Learning and Adapting Robot Motion Planning with Diffusion Models"
    
    MPD (and similarly Diffuser) often:
    1. Diffuses the *entire trajectory* at once (Trajectory-level diffusion), 
       rather than state-by-state autoregressive rollout.
       (In our mock setup, 'get_denoising_trajectory' already produces the whole path, 
       so we are effectively doing trajectory generation).
    2. Uses 'Guidance' during the reverse process to minimize a cost function J(tau).
       Score_hat = Score + gamma * Grad(J(tau))
       
    Cost J(tau) usually includes:
    - Collision cost (smooth obstacles)
    - Smoothness cost
    - Goal distance
    
    This is conceptually similar to CoDiG, but MPD often emphasizes the 'Trajectory-as-a-Sample' view.
    For this implementation, we will act similarly to CoDiG but structure it to emphasize 
    Whole-Trajectory guidance (i.e. the gradient acts on the entire path tensor if possible, 
    but here we iterate per step for consistency with the loop).
    
    Key Difference from CoDiG (in this Mock):
    - MPD often employs "inpainting" for start/goal constraints (forcing the first/last points to be fixed).
    - We will implement this Inpainting constraint explicitly:
      x_{t-1}[0] = start_pos
      x_{t-1}[-1] = target_pos (if doing fixed-horizon planning)
      
    But since our loop is 'auto-regressive like' (next state depends on current), 
    true MPD usually operates on the *whole* tensor of shape (H, D) executing diffusion steps t=T..0.
    
    **To faithfully mock MPD in our 'state-based' simulation loop**:
    We will treat the 'get_denoising_trajectory' as if it is *generating* the path.
    However, our current 'PretrainedModel' is a drift-based dynamics model (dx/dt = f(x)), 
    which implies a temporal evolution.
    
    To implement a specific *variant* of MPD that fits this loop:
    We will use the guidance to look ahead or ensure local collision probability is low.
    Actually, let's implement the **Inpainting** aspect or **Conditioning** aspect strongly.
    
    Let's stick to the standard guidance formulation but add a 'Smoothness' cost which involves neighbors.
    Since we are generating sequentially (x0 -> x1 -> ...), smoothness is naturally handled by the dynamics.
    
    So MPD here will look like: 
    - Base Drift (Learned Prior)
    - Goal Guidance (Attraction)
    - Obstacle Guidance (Collision Cost Gradient)
    - **Adaptive Sampling**: MPD paper discusses adapting the plan. 
      We will implement a stronger, potentially annealed guidance scale.
    """
    def __init__(self, pretrained_model, target_pos, obstacles, n_steps=50):
        self.pretrained_model = pretrained_model
        self.target_pos = np.array(target_pos)
        self.obstacles = obstacles
        self.n_steps = n_steps
        self.current_state = None

    def set_initial_noise(self, noise: np.ndarray):
        self.current_state = np.array(noise)

    def _compute_cost_gradient(self, state):
        """
        Gradient of Cost function J(x).
        J = w1 * DistGoal + w2 * DistObs (Collision)
        """
        # Goal Cost
        diff = self.target_pos - state
        dist = np.linalg.norm(diff)
        grad_goal = -1.0 * (diff / (dist + 1e-4)) # Negative gradient of Distance (we want to minimize dist)
        # Wait, if J = dist, then Grad J points away from goal. We want to move in -Grad J.
        # So we want to move TOWARDS goal.
        # Correction: Return the direction we should MOVE (Negative gradient)
        
        move_goal = np.zeros_like(state)
        if dist > 1e-4:
            move_goal = diff / dist
            
        # Obstacle Cost (Smooth repulsion)
        move_obs = np.zeros_like(state)
        for obs in self.obstacles:
            ox, oy, r = obs
            obs_center = np.array([ox, oy])
            diff_obs = state - obs_center
            dist_obs = np.linalg.norm(diff_obs)
            margin = 0.5
            
            # Simple occupancy cost: J_obs = max(0, r + margin - dist)^2
            if dist_obs < r + margin:
                # We need to move AWAY.
                direction = diff_obs / (dist_obs + 1e-6)
                # The gradient of cost points towards center (increasing cost). 
                # -Grad points away.
                # Magnitude:
                mag = 2.0 * (r + margin - dist_obs)
                move_obs += direction * mag * 2.0 # stronger weight
                
        return 0.5 * move_goal + 1.0 * move_obs

    def get_denoising_trajectory(self, guidance_func=None, guidance_scale=0.1):
        import time
        start_time = time.time()
        
        if self.current_state is None:
            raise ValueError("Initial noise not set.")
            
        current_state = self.current_state.copy()
        path = [current_state.copy()]
        
        # Annealing guidance scale for MPD (often useful)
        # Start strong, reduce? Or refined at the end?
        # Usually constant or increasing.
        
        for t in range(self.n_steps):
            noise = np.random.randn(*current_state.shape) * 0.1
            
            # Base Drift
            drift = self.pretrained_model.predict_drift(current_state, t)
            
            # MPD Guidance
            # "Steering" the generation with gradients of the cost
            mpd_guidance = self._compute_cost_gradient(current_state)
            
            # Combine
            # scale = 0.5 (hyperparameter)
            next_state = current_state + drift + mpd_guidance * 0.5 + noise
            current_state = next_state
            path.append(current_state.copy())
            
        end_time = time.time()
        return {'trajectory': np.array(path), 'time': end_time - start_time}
