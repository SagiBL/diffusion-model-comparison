import numpy as np
from diffusion_interface import DiffusionModel
# Optional: import cvxopt or scipy if we wanted real QP, but for 2D single-integrator mock, 
# we can use analytical solutions or simple geometric projections.

class CoBLDiffusionModel(DiffusionModel):
    """
    CoBL-Diffusion: Conditional Robot Planning using CBF and CLF.
    
    This sampler modifies the diffusion transition to satisfy:
    1. Control Lyapunov Function (CLF): Stability towards target.
       V(x) = ||x - x_goal||^2
    2. Control Barrier Function (CBF): Safety (Constraint satisfaction).
       h(x) = dist(x, obstacle) - radius >= 0
    
    Method:
    At each step, we have a proposed 'drift' (from pretrained model) + noise.
    We treat this as a reference control 'u_ref'.
    We find the closest 'u' to 'u_ref' that satisfies:
      Grad(h) * u >= -gamma * h(x)  (CBF Constraint)
      Grad(V) * u <= -alpha * V(x)  (CLF Constraint - often softened or just used as guidance)
      
    For this implementation, we will act as a "Safety Filter" on the predicted drift.
    """
    def __init__(self, pretrained_model, target_pos, obstacles, n_steps=50, alpha=0.5, gamma=1.0):
        """
        Args:
            pretrained_model: Shared base model.
            target_pos: Destination.
            obstacles: List of (x,y,r).
            n_steps: Sampling steps.
            alpha: CLF gain.
            gamma: CBF gain.
        """
        self.pretrained_model = pretrained_model
        self.target_pos = np.array(target_pos)
        self.obstacles = obstacles
        self.n_steps = n_steps
        self.alpha = alpha
        self.gamma = gamma
        self.current_state = None

    def set_initial_noise(self, noise: np.ndarray):
        self.current_state = np.array(noise)

    def _get_clf_u(self, state):
        """
        Derives a guidance velocity from CLF.
        V = 0.5 * ||x - x_goal||^2
        dV/dx = x - x_goal
        We want dV/dt <= -alpha * V
        (dV/dx) * u <= -alpha * V
        
        A simple CLF-compliant velocity is just proportional control:
        u_clf = -alpha * (x - x_goal)
        """
        return -self.alpha * (state - self.target_pos)

    def _apply_cbf_filter(self, state, u_nominal):
        """
        Modifies u_nominal to satisfy CBF constraints.
        h(x) = ||x - obs||^2 - r^2 >= 0 (Using squared dist for smoother gradients)
        
        Constraint: L_f h(x) + L_g h(x) u >= -gamma * h(x)
        Here dynamics are simple single integrator: dx = u.
        So: (dh/dx) * u >= -gamma * h(x)
        
        If constraint violated, project u to the boundary.
        min ||u - u_nom||^2 s.t. A u >= b
        """
        u_safe = u_nominal.copy()
        
        for obs in self.obstacles:
            ox, oy, r = obs
            obs_center = np.array([ox, oy])
            
            # Formulate CBF
            # h(x) = ||x - c|| - r  (Distance based)
            diff = state - obs_center
            dist = np.linalg.norm(diff)
            h = dist - r
            
            if dist < 1e-6:
                grad_h = np.random.randn(2)
                grad_h /= np.linalg.norm(grad_h)
            else:
                grad_h = diff / dist
            
            # Safe set condition: grad_h * u >= -gamma * h
            b = -self.gamma * h
            constraint_val = np.dot(grad_h, u_safe)
            
            if constraint_val < b:
                # Violation! Project u_safe onto the half-plane.
                # Project u back along the normal (grad_h) until constraint is met.
                # u_new = u_old + lambda * grad_h
                # dot(grad_h, u_old + lambda * grad_h) = b
                # dot(grad_h, u_old) + lambda * ||grad_h||^2 = b
                # lambda = (b - dot(grad_h, u_old)) / ||grad_h||^2
                # Since grad_h is normalized, ||grad_h||^2 = 1
                
                lam = (b - constraint_val)
                u_safe = u_safe + lam * grad_h
                
        return u_safe

    def get_denoising_trajectory(self, guidance_func=None, guidance_scale=0.1):
        import time
        start_time = time.time()
        
        if self.current_state is None:
            raise ValueError("Initial noise not set.")
            
        current_state = self.current_state.copy()
        path = [current_state.copy()]
        
        for t in range(self.n_steps):
            noise = np.random.randn(*current_state.shape) * 0.1
            
            # 1. Base Model Drift
            drift_base = self.pretrained_model.predict_drift(current_state, t)
            
            # 2. CLF Guidance (Stability)
            # Combine base drift with CLF desire.
            # In CoBL, diffusion score is often combined with CLF.
            # We'll treat CLF as a strong guidance term acting as the 'nominal' control reference
            u_clf = self._get_clf_u(current_state)
            
            # Weighted interaction: Base drift vs CLF?
            # Let's say we want to follow base drift BUT bias heavily by CLF?
            # Or simpler: u_nominal = drift_base + u_clf
            u_nominal = drift_base + u_clf * 0.1 # Scale CLF contribution
            
            # 3. CBF Filter (Safety)
            # Project the nominal update (drift) to be safe
            u_safe = self._apply_cbf_filter(current_state, u_nominal)
            
            # Update
            next_state = current_state + u_safe + noise
            current_state = next_state
            path.append(current_state.copy())
            
        end_time = time.time()
        return {'trajectory': np.array(path), 'time': end_time - start_time}
