from abc import ABC, abstractmethod
import numpy as np

class DiffusionModel(ABC):
    """
    Abstract Base Class for Diffusion Models.
    Enforces a consistent API for setting noise and retrieving trajectories.
    """
    
    @abstractmethod
    def set_initial_noise(self, noise: np.ndarray):
        """
        Sets the starting noise (x_T) for the reverse diffusion process.
        
        Args:
            noise (np.ndarray): The initial noise tensor/array.
        """
        pass

    @abstractmethod
    def get_denoising_trajectory(self, guidance_func=None, guidance_scale: float = 0.0) -> np.ndarray:
        """
        Runs the denoising process and returns the full trajectory.
        
        Args:
            guidance_func (callable, optional): Function for guidance (e.g., gradient of energy).
            guidance_scale (float, optional): Scalar to control guidance strength.
            
        Returns:
            np.ndarray: The sequence of states from x_T to x_0.
        """
        pass
