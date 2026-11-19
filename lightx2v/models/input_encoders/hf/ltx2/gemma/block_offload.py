"""Synchronous, private CPU weight staging for the LTX Gemma 3 text path."""

from contextlib import contextmanager

import torch

from lightx2v.common.offload.module_adapter import ModuleCPUWeights, module_tensors


@contextmanager
def _staged_modules(modules, device):
    """Stage native modules with the shared adapter; retain CPU object identities."""
    states = {module: (ModuleCPUWeights(module), module_tensors(module)) for module in modules}
    active = set()

    def activate(module, _args):
        if module in active:
            raise RuntimeError("A staged text module cannot be entered recursively")
        active.add(module)
        weights, _ = states[module]
        weights.activate(weights.sources.keys(), device)

    def restore(module, *_args):
        if module not in active:
            return
        if device.type != "cpu":
            getattr(torch, device.type).current_stream(device).synchronize()
        # Tied weights must point back to their original CPU Parameter objects.
        for name, tensor in states[module][1].items():
            parent_name, _, leaf = name.rpartition(".")
            parent = module.get_submodule(parent_name)
            (parent._parameters if leaf in parent._parameters else parent._buffers)[leaf] = tensor
        active.remove(module)

    handles = []
    try:
        for module in states:
            handles.append(module.register_forward_pre_hook(activate))
            handles.append(module.register_forward_hook(restore, always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module in tuple(active):
            restore(module)


@torch.inference_mode()
def encode_with_block_offload(encoder, text, device, padding_side="left"):
    """Run Gemma layers, projection, then the two connectors in separate stages.

    CPU tensors are never pinned or copied back from the device. The normal HF
    backbone owns masking, RoPE, layer order, and hidden-state collection. This
    text-only path skips the unused vocabulary logits, vision tower, and KV cache.
    """
    gemma = encoder.model
    if gemma.config.model_type != "gemma3":
        raise ValueError("Gemma block offload currently supports the LTX-2/2.3 Gemma 3 text encoder only")
    if getattr(encoder, "_block_offload_active", False):
        raise RuntimeError("The text encoder already has an active staged request")
    device = torch.device(device)
    backbone = gemma.model
    language_model = backbone.language_model
    modules = [*language_model.layers, language_model.norm]
    pairs = encoder.tokenizer.tokenize_with_weights(text)["gemma"]
    input_ids = torch.tensor([[pair[0] for pair in pairs]], device="cpu")
    # Gemma maps an out-of-vocabulary image token to zero before embedding lookup.
    if gemma.config.image_token_id >= backbone.vocab_size:
        input_ids.masked_fill_(input_ids == gemma.config.image_token_id, 0)
    attention_mask = torch.tensor([[pair[1] for pair in pairs]], device=device)
    encoder._block_offload_active = True
    try:
        inputs_embeds = language_model.embed_tokens(input_ids).to(device)
        with _staged_modules(modules, device):
            outputs = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        with _staged_modules([encoder.feature_extractor], device):
            video_features, audio_features = encoder.feature_extractor(outputs.hidden_states, attention_mask, padding_side)
        del outputs
        additive_mask = encoder._convert_to_additive_mask(attention_mask, video_features.dtype)
        processor = encoder.embeddings_processor
        connectors = [processor.video_connector]
        if processor.audio_connector is not None:
            connectors.append(processor.audio_connector)
        with _staged_modules(connectors, device):
            return processor.create_embeddings(video_features, audio_features, additive_mask)
    finally:
        encoder._block_offload_active = False
