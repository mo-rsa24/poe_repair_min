import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F



class Co3Corrector():
    def __init__(self, 
                #  num_corrector_steps: int,
                 lmda: float = 0.8,
                 step_size: float = 1.0,
                 use_cfg_corrector: bool = False,
                 modulate_comp_weights: bool = False,
                 verbose: bool = True
                ):
        """
        Args:
            num_corrector_steps (int): Number of steps for the Langevin correction.
            lmda (float): Scaling factor for the composed score. 0<= lmda <= 1 for CFG++ style.
            step_size (float): Step size for the Langevin correction.
            use_cfg_corrector (bool): Whether to use the CFG corrector.
        """
        
        # self.num_corrector_steps = num_corrector_steps
        self.lmda = lmda
        self.step_size = step_size
        self.use_cfg_corrector = use_cfg_corrector
        self.verbose = verbose
        self.modulate_comp_weights = modulate_comp_weights

    def _get_composed_score(self, noise_pred_multi, noise_pred_concepts, alpha_t, lmda=1.0):
        num_concepts = len(noise_pred_concepts)
        composed_score = 0
        
        composed_score = composed_score - num_concepts * noise_pred_multi
        for cc in range(num_concepts):
            composed_score = composed_score + noise_pred_concepts[cc]
        if self.use_cfg_corrector:
            composed_score = lmda * composed_score

        composed_score = -1. * (1/(1-alpha_t)**(0.5)) * composed_score
        return composed_score
    
    _warned_get_composed_noise = False
    def _get_composed_noise(self, x, noise_pred_multi, noise_pred_concepts, alpha_t, lmda=1.0):
        """  
            sum_zero composing.
        """

        num_concepts = len(noise_pred_concepts)
        composed_noise = 0

        composed_noise = composed_noise + num_concepts * noise_pred_multi
        for cc in range(num_concepts):
            composed_noise = composed_noise - noise_pred_concepts[cc]
        if self.use_cfg_corrector:
            composed_noise = lmda * composed_noise

        composed_noise = (x + (1-alpha_t).sqrt() * composed_noise) / (1-alpha_t).sqrt()
        return composed_noise



    def _get_sumzero_tweedie_composed_noise(self, x, noise_pred_multi, noise_pred_concepts, weights, alpha_t, lmda=1.0):
        """ 
            sum_zero composing.
        """
        assert len(weights) == len(noise_pred_multi) + len(noise_pred_concepts), f"weights length {len(weights)} does not match number of concepts {len(noise_pred_concepts)} and multi {len(noise_pred_multi)}"

        num_concepts = len(noise_pred_concepts)
        composed_noise = 0

        # composed_noise = composed_noise + num_concepts * noise_pred_multi
        composed_noise = composed_noise + weights[0] * noise_pred_multi
        for cc in range(num_concepts):
            composed_noise = composed_noise + weights[cc+1] * noise_pred_concepts[cc]
        if self.use_cfg_corrector:
            composed_noise = lmda * composed_noise

        composed_noise = (x + (1-alpha_t).sqrt() * composed_noise) / (1-alpha_t).sqrt()
        return composed_noise
    
    def _get_contrastive_noise(self, noise_pred_multi, noise_pred_concepts, alpha_t, lmda=1.0):
    

        num_concepts = len(noise_pred_concepts)
        composed_noise = 0

        composed_noise = composed_noise + num_concepts * noise_pred_multi
        for cc in range(num_concepts):
            composed_noise = composed_noise - noise_pred_concepts[cc]
        if self.use_cfg_corrector:
            composed_noise = lmda * composed_noise
        return composed_noise

    def _get_contrastive_tweedie(self, x, noise_pred_multi, noise_pred_uncond, noise_pred_concepts, weights, alpha_t, lmda=1.0):
        """ Order of weights should be [multi, concept1, concept2, ..., conceptN]
        """

        assert len(weights) == len(noise_pred_multi) + len(noise_pred_concepts), f"weights length {len(weights)} does not match number of concepts {len(noise_pred_concepts)} and multi {len(noise_pred_multi)}"


        num_concepts = len(noise_pred_concepts)
        noise_pred = noise_pred_uncond + lmda * (noise_pred_multi - noise_pred_uncond)
        tweedie_multi = x - (1 - alpha_t).sqrt() * noise_pred
 
        tweedie_multi = weights[0] * tweedie_multi 
        for i in range(num_concepts):
            noise_pred_single = noise_pred_uncond + lmda * (noise_pred_concepts[i:i+1] - noise_pred_uncond)
            tweedie_single = x - (1 - alpha_t).sqrt() * noise_pred_single
            tweedie_multi = tweedie_multi + weights[i+1] * tweedie_single  # concept weights start from index 1. And we are summing because weights are negative for concepts
        return tweedie_multi

    
    @staticmethod
    def _get_concept_weights_from_dists(
        dists: torch.Tensor,
        method: str = 'exp',    # 'exp' or 'invpow'
        beta: float = 1.0,     # for 'exp'
        p: float = 2.0,         # for 'invpow'
        eps: float = 1e-6,
        w_min_mag: float = 0.01,
        w_max_mag: float = 0.9,
    ) -> torch.Tensor:
        """
        Compute adaptive concept weights based on distances.
        
        Args:
            dists: 1D tensor of distances [K]
            method: 'exp' or 'invpow'
            beta: sensitivity for exp kernel
            p: exponent for invpow kernel
            eps: small constant for stability
            w_min_mag, w_max_mag: clipping range on |weights|
        
        Returns:
            weights: 1D tensor [K], negative, sum = -1
        """
        dists = dists.float()

        if method == 'exp':
            a = torch.exp(-beta * dists)
        elif method == 'invpow':
            a = 1.0 / (torch.pow(dists + eps, p))
        else:
            raise ValueError("method must be 'exp' or 'invpow'")

        if torch.all(a == 0):
            a = torch.ones_like(a)

        # Normalize to sum to -1
        w = -a / a.sum()
        return w

    def co3_resampling(self,
                            num_corrector_steps: int,
                           text_embeddings: torch.Tensor,
                            unet: Callable,
                            x: torch.Tensor,
                            unet_added_conditions: Dict[str, Any],
                            t: torch.Tensor,
                            at: torch.Tensor,
                            add_time_ids: torch.Tensor,
                            step_size: float = 0.01,
                            beta: float = 1.0   # for adaptive concept weights when method='exp', beta=0 corresponds to uniform weights
                            ):
        iteration = 0
        num_concepts = len(text_embeddings[2:])
        intermediates = None
        dists = []

        # Define composition weights here
        concept_weights = [-1.0/num_concepts]*num_concepts
        composition_weights = [1.0] + concept_weights #[-1.0/num_concepts]*num_concepts # \sum = 0 weights

        while iteration < num_corrector_steps:
            iteration += 1
            unet.zero_grad()
            latent_model_input = torch.cat([x] * (num_concepts + 2))  # Concatenate for uncond, multi, and concepts
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, added_cond_kwargs=unet_added_conditions)['sample']
            noise_pred_uncond, noise_pred_multi = noise_pred[:1], noise_pred[1:2]
            noise_pred_concepts = noise_pred[2:]
            for i in range(num_concepts):
                dists.append((noise_pred_concepts[i].flatten() - noise_pred_multi.flatten()).norm().item())

            if self.modulate_comp_weights:
                concept_dists = torch.tensor(dists)
                concept_weights = self._get_concept_weights_from_dists(concept_dists, method='exp', beta=beta)  # for method ="exp", beta=0 correspond to uniform weights
                composition_weights = [1.0] + concept_weights.tolist()
            # contrast concept directions
            composed_noise_lmda = self._get_sumzero_tweedie_composed_noise(x, noise_pred_multi, noise_pred_concepts, composition_weights, at, lmda=self.lmda)
            # intermediate latent 1 (= at.sqrt() * denoised_tweedie)
            denoised_tweedie_at = x - (1 - at).sqrt() * composed_noise_lmda
            # Manifold aware noise addition
            x_2 = denoised_tweedie_at + (1 - at).sqrt() * noise_pred_uncond

            x = x_2
            
            if self.verbose:
                print(f"Iteration: {iteration}/{num_corrector_steps}" # loss: {score.abs().mean().item():.4f}"
                        f"  max(composed_noise): {composed_noise_lmda.max().item():.4f} min(composed_noise): {composed_noise_lmda.min().item():.4f}"
                        f"  max(x): {x.max().item():.4f} min(x): {x.min().item():.4f}  step_size: {step_size:.4f}"
                        f"  dist: {[f'{s:.4f}' for s in dists]}"
                        f"  weights: {[f'{w:.4f}' for w in concept_weights]}"
                        f" beta: {beta}")
            dists = []
        return x, intermediates



    def co3_corrector(self,
                            num_corrector_steps: int,
                           text_embeddings: torch.Tensor,
                            unet: Callable,
                            x: torch.Tensor,
                            unet_added_conditions: Dict[str, Any],
                            t: torch.Tensor,
                            at: torch.Tensor,
                            add_time_ids: torch.Tensor,
                            step_size: float = 0.01,
                            beta: float = 1.0
                            ):
        iteration = 0
        num_concepts = len(text_embeddings[2:])
        intermediates = None
        dists = []
        
        # Define composition weights here
        concept_weights = [-1.0/num_concepts]*num_concepts
        composition_weights = [2.0] + concept_weights # \sum = 0 weights

        while iteration < num_corrector_steps:
            iteration += 1
            unet.zero_grad()
            latent_model_input = torch.cat([x] * (num_concepts + 2))  # Concatenate for uncond, multi, and concepts
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, added_cond_kwargs=unet_added_conditions)['sample']
            noise_pred_uncond, noise_pred_multi = noise_pred[:1], noise_pred[1:2]
            noise_pred_concepts = noise_pred[2:]
            for i in range(num_concepts):
                dists.append((noise_pred_concepts[i].flatten() - noise_pred_multi.flatten()).norm().item())


            if self.modulate_comp_weights:
                concept_dists = torch.tensor(dists)
                concept_weights = self._get_concept_weights_from_dists(concept_dists, method='exp', beta=beta) # for method ="exp", beta=0 correspond to uniform weights
                composition_weights = [2.0] + concept_weights.tolist()

            composed_tweedie = self._get_contrastive_tweedie(x, noise_pred_multi, noise_pred_uncond, noise_pred_concepts, composition_weights, at, lmda=self.lmda)
            r = (x  - (noise_pred_uncond + self.lmda * (noise_pred_multi - noise_pred_uncond)) * (1 - at).sqrt()).max() #at.sqrt() * x.max()
            
            #! normalized 
            composed_tweedie = composed_tweedie / composed_tweedie.max() * r
            x_2 = composed_tweedie + (1-at).sqrt() * noise_pred_uncond #noise_pred_multi
            x_2 = x_2 * (x.norm()/ x_2.norm())
            x = x_2
            
            if self.verbose:
                print(f"Iteration: {iteration}/{num_corrector_steps}" # loss: {score.abs().mean().item():.4f}"
                    f"  max(x_0_hat): {composed_tweedie.max().item():.4f} min(x_0_hat): {composed_tweedie.min().item():.4f}"
                    f"  max(x): {x.max().item():.4f} min(x): {x.min().item():.4f}  step_size: {step_size:.4f}"
                    f"  dist: {[f'{s:.4f}' for s in dists]}"
                    f"  weights: {[f'{w:.4f}' for w in concept_weights]}"
                    f" beta: {beta}")
            dists = []
        return x, intermediates