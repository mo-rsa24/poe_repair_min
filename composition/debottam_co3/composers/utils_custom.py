import torch
import os
import random
import numpy as np
import math
from einops import rearrange
import torch.nn.functional as F
import torchvision

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_time(model, t):
   
    down_res_dict = {1: [0, 1], 2: [0, 1]}
    down_transformers_len = {1:[2,2],2:[10,10]}
    up_transformers_len = {0:[10,10,10],1:[2,2,2]}
    up_res_dict = {0:[ 1, 2],1: [0, 1, 2]}
    up_res_dict_cross = {0:[ 0, 1, 2],1: [0, 1, 2]}
    mid_transformers_len=10
    for res in up_res_dict_cross:
        for block in up_res_dict_cross[res]:
            for idx in range(up_transformers_len[res][block]):
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[idx].attn2
                
                setattr(module, 't', t)
            
    for res in down_res_dict:
        for block in down_res_dict[res]:
            for idx in range(down_transformers_len[res][block]):
                module = model.unet.down_blocks[res].attentions[block].transformer_blocks[idx].attn2
                
                setattr(module, 't', t)

    
    for idx in range(mid_transformers_len):
        module = model.unet.mid_block.attentions[0].transformer_blocks[idx].attn2
        
        setattr(module, 't', t)

