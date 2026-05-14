import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from PIL import Image
import numpy as np
from composers.Co3 import Co3
from composers.utils_custom import seed_everything
import json
# import spacy
import stanza
import pyrallis
from composers.config import Co3Config


def get_result_path(opt, prefix=None)-> str:
    if prefix is None:
        prefix = f"./output/test/{opt.prompt}_{opt.seed}"
    if not prefix.endswith('/'):
        prefix = prefix + '/'
    if opt.num_ts_to_correct > 0:
        prefix += f"algo-{opt.corrector_algo}_"
    else:
        prefix += f"algo-sd"
    return prefix



@pyrallis.wrap()
def main(opt: Co3Config):

    root_output_path = get_result_path(opt, prefix=opt.output_path)
    sdcorr = Co3(opt)
    print(f"opt:{opt}")

    for seed in opt.seeds:
        opt.seed = seed
        prompt_text = opt.prompt_orig.replace(" ", "_")
        opt.output_dir =  os.path.join(root_output_path, f"{prompt_text}", f"seed-{opt.seed}") 
        os.makedirs(opt.output_dir, exist_ok=True)
        
        seed_everything(opt.seed)
        sdcorr.config = opt
        sdcorr.prepare_prompts(opt)
        sdcorr.prepare_embeds()
        img = sdcorr.run_sampling()

        save_path = os.path.join(opt.output_dir, f"{prompt_text}.png")
        img[0].save(save_path)

    

if __name__ == "__main__":
    main()