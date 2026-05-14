from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import os
from pathlib import Path


@dataclass
class Co3Config:
    """Configuration for Co3 pipeline."""

    # Prompt parameters
    prompt: str 
    prompt_orig: str = ""  # Original prompt for reference
    concept: str = ""
    negative_prompt: str = ""   
    # Basic parameters
    seeds: List[int] = field(default_factory=lambda: [1688])
    device: str = 'cuda:0'
    output_path: str = 'output'
    output_path_all: str = 'output'
    
    # Generation parameters
    use_cfgpp: bool = True
    guidance_scale: float = 0.8
    n_timesteps: int = 50

    # Model parameters
    sd_version: str = 'xl'  # choices: ['xl']

    # latent correction
    perform_latent_correction: bool = True
    latent_corrector_type: str = 'Co3Corrector'  # choices: ['Co3Corrector', 'none']
    corrector_algo: str = "co3-hybrid"  # choices: ["co3-resampling", "co3-corrector", "co3-hybrid"]
    num_latent_corrector_steps: int = 10
    num_resampling_steps: int = 4  # number of sum_0 resampling steps at the inital ts. 
    num_ts_to_correct: int = 1  # Number of timesteps to correct, used in 'same_t' algorithm
    lmda: float = 0.8  # guidance_scale eqv of the latentCorrector class
    modulate_comp_weights: bool = False  # whether to modulate the concept weights based on the distances
    beta: float = 0.9  # for modulating the concept weights based on distances, higher beta means more focus on closer concepts. It activates the "exp" method and only after t < T - 1

    seg_gpu: int = 1
    
    # Resolution and cropping
    crops_coords_top_left_h: int = 0
    crops_coords_top_left_w: int = 0
    resolution_h: int = 1024
    resolution_w: int = 1024

    def __post_init__(self):
        """Post-initialization validation and setup."""

        # fixes the "None":
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value == 'None':
                setattr(self, field_name, '')
        
        # Create output directories
        os.makedirs(self.output_path, exist_ok=True)
        if self.output_path_all:
            os.makedirs(self.output_path_all, exist_ok=True)
        
        # Validate SD version
        valid_versions = ['xl']
        if self.sd_version not in valid_versions:
            raise ValueError(f"sd_version must be one of {valid_versions}")
        
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Re-run validation
        self.__post_init__()

