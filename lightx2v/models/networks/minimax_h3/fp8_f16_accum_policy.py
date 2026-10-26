from lightx2v.utils.fp8_f16_accum_checkpoint import validate_fp8_f16_accum_checkpoint as validate_checkpoint_metadata

FP8_F16_ACCUM_WEIGHT_QMAX = 14.0
DIT_FP8_F16_ACCUM_ACTIVATION_QMAX = 7.0
VIDEO_VAE_FP8_F16_ACCUM_ACTIVATION_QMAX = 14.0
FP8_F16_ACCUM_QUANTIZATION_PROFILE = "h3-fp8-f16-accum"
FP8_F16_ACCUM_PROJECTION_SUFFIXES = (
    ".attn.to_q",
    ".attn.to_k",
    ".attn.to_v",
    # The runtime-only fused projection must preserve the same accumulation
    # contract as the three checkpoint-backed projections it replaces.
    ".attn.to_qkv",
    ".attn.to_out.0",
    ".ff.net.0.proj",
    ".ff.net.2",
)


def validate_fp8_f16_accum_checkpoint(checkpoint_path):
    validate_checkpoint_metadata(
        checkpoint_path,
        quantization_profile=FP8_F16_ACCUM_QUANTIZATION_PROFILE,
        numeric_fields=(("weight_qmax", FP8_F16_ACCUM_WEIGHT_QMAX),),
        not_found_prefix="No safetensors weights found in FP8 checkpoint",
        profile_name="quantization profile",
        include_actual_profile=True,
    )
