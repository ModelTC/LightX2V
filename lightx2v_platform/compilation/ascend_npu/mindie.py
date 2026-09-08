from lightx2v_platform.registry_factory import PLATFORM_COMPILE_BACKEND_REGISTER


@PLATFORM_COMPILE_BACKEND_REGISTER("mindie")
def create_mindie_backend():
    """Create MindIE-SD lazily and only for an explicit Ascend request."""
    try:
        from mindiesd.compilation import MindieSDBackend

        return MindieSDBackend()
    except Exception as exc:
        raise RuntimeError("compile_backend='mindie' requires a working MindIE-SD installation on Ascend.") from exc
