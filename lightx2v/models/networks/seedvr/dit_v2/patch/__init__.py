def get_na_patch_layers(patch_type="v1"):
    assert patch_type in ["v1"]
    if patch_type == "v1":
        from .patch_v1 import NaPatchIn, NaPatchOut
    return NaPatchIn, NaPatchOut
