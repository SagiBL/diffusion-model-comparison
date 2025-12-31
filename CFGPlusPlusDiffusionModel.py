import numpy as np
from diffusion_interface import DiffusionModel

class CFGPlusPlusDiffusionModel(DiffusionModel):
    """
    CFG++ (Manifold-Constrained Classifier Free Guidance).
    
    Standard CFG: 
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
        
    CFG++ (Simplified Concept for Mock):
        CFG often pushes samples off-manifold (intensities too high/weird directions).
        CFG++ tries to keep the vector 'on-manifold'.
        
        For this 2D mock, 'off-manifold' might mean velocity too high or direction inconsistent.
        We will mock the 'manifold projection' by normalizing or rescaling the guidance 
        to match the expected norm of the unconditional signal, or similar geometric constraint.
    """
    def __init__(self, pretrained_model, target_pos, n_steps=50, cfg_scale=3.0):
        self.pretrained_model = pretrained_model
        self.target_pos = np.array(target_pos)
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.current_state = None

    def set_initial_noise(self, noise: np.ndarray):
        self.current_state = np.array(noise)

    def get_denoising_trajectory(self, guidance_func=None, guidance_scale=0.1):
        # Note: CFG usually doesn't need external 'guidance_func' (classifier guidance),
        # it uses internal 'cfg_scale' with conditional model.
        # But we keep signature for interface consistency.
        
        import time
        start_time = time.time()
        
        if self.current_state is None:
            raise ValueError("Initial noise not set.")
            
        current_state = self.current_state.copy()
        path = [current_state.copy()]
        
        for t in range(self.n_steps):
            noise = np.random.randn(*current_state.shape) * 0.1
            
            # 1. Unconditional Prediction
            drift_uncond = self.pretrained_model.predict_drift(current_state, t, target=None)
            
            # 2. Conditional Prediction
            drift_cond = self.pretrained_model.predict_drift(current_state, t, target=self.target_pos)
            
            # 3. Standard CFG Vector
            # w * (cond - uncond)
            guidance_vec = drift_cond - drift_uncond
            
            # Standard CFG Update: uncond + scale * guidance
            # drift_cfg = drift_uncond + self.cfg_scale * guidance_vec
            
            # 4. CFG++ Logic (Mock)
            # Problem: drift_cfg might be very large or point weirdly.
            # Fix: Rescale the combined vector to have similiar magnitude to drift_cond?
            # Or project?
            # A simple interpretation of "Manifold Constrained":
            # The direction is informed by CFG, but magnitude is constrained to reasonable physics.
            
            # Let's compute the standard, then normalize magnitude to be close to 'drift_cond' magnitude
            raw_update = drift_uncond + self.cfg_scale * guidance_vec
            
            # Norm constraint (Manifold constraint mock)
            ref_norm = np.linalg.norm(drift_cond)
            raw_norm = np.linalg.norm(raw_update)
            
            if raw_norm > 1e-6:
                # Rescale to match the expected 'conditional' speed (manifold speed)
                drift_cfgpp = raw_update * (ref_norm / raw_norm)
            else:
                drift_cfgpp = raw_update
            
            # Combine
            next_state = current_state + drift_cfgpp + noise
            current_state = next_state
            path.append(current_state.copy())
            
        end_time = time.time()
        return {'trajectory': np.array(path), 'time': end_time - start_time}
