import os

from lightx2v_platform import *

init_ai_device(os.getenv("AI_DEVICE", "cuda"))
from lightx2v_platform.base.global_var import AI_DEVICE  # noqa E402

if __name__ == "__main__":
    print(f"AI_DEVICE : {AI_DEVICE}")
    is_available = check_ai_device(AI_DEVICE)
    print(f"Device available: {is_available}")
