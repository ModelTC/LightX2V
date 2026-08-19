#!/bin/bash

# Set paths first.
lightx2v_path=path/to/LightX2V
source_dir=path/to/MiniMax-H3/text_encoder
output_dir=path/to/MiniMax-H3_quantized/fp8

export CUDA_VISIBLE_DEVICES=0

# Convert the embedding and first 50 text-encoder layers used by MiniMax-H3.
python ${lightx2v_path}/tools/convert/converter.py \
--source ${source_dir} \
--output ${output_dir} \
--output_name minimax_h3_text_encoder_fp8_sgl \
--model_type h3_text_encoder \
--quantized \
--linear_type fp8 \
--device cuda:0 \
--single_file
