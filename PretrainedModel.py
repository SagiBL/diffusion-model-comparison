import numpy as np

class PretrainedDiffusionModel:
    """
    Represents the underlying trained diffusion model (e.g., U-Net).
    It provides the learned gradient field / residual prediction used by samplers.
    """
    def __init__(self):
        # In a real scenario, this would load weights.
        pass

    def predict_drift(self, state, t):
        """
        Predicts the drift (or score/noise) at a given state and time step.
        
        Args:
            state (np.ndarray): Current state x_t.
            t (int): Current timestep.
            
        Returns:
            np.ndarray: The predicted drift vector.
        """
        # Mock dynamics: drifts towards origin (0,0)
        # This matches the "drift = -0.05 * current_state" logic we used before.
        return -0.05 * state
