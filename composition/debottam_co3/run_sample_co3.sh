#!/usr/bin/env bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export CUDA_VISIBLE_DEVICES=0

PROMPT="turtle+mouse+a turtle and a mouse"

PROMPT_ORIG="a turtle and a mouse"


guidance_scale=0.8 



export RESULT_PATH="./outputs/" # replace spaces with underscores
python sample_co3.py \
    --guidance_scale $guidance_scale --n_timesteps 50 --prompt "$PROMPT" \
    --output_path $RESULT_PATH --output_path_all $RESULT_PATH --sd_version "xl" --resolution_h 1024 --resolution_w 1024 \
    --prompt_orig "$PROMPT_ORIG" --seeds [1688] \
    --negative_prompt '' \
    --num_ts_to_correct 6 \
    --num_latent_corrector_steps 5 \
    --num_resampling_steps 3 \
    --corrector_algo "co3-hybrid" \
    --modulate_comp_weights "True" \
    --beta 0.9 \
    --lmda 0.8 \


