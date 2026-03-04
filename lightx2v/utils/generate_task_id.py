import random
import string
import time
from datetime import datetime


def generate_task_id():
    """
    Generate a random task ID in the format XXXX-XXXX-XXXX-XXXX-XXXX.
    Features:
    1. Does not modify the global random state.
    2. Each X is an uppercase letter or digit (0-9).
    3. Combines time factors to ensure high randomness.
    """
    # Save the current random state (does not affect external randomness)
    original_state = random.getstate()

    try:
        # Define character set (uppercase letters + digits)
        characters = string.ascii_uppercase + string.digits

        # Create an independent random instance
        local_random = random.Random(time.perf_counter_ns())

        # Generate 5 groups of 4-character random strings
        groups = []
        for _ in range(5):
            # Mix new time factor for each group
            time_mix = int(datetime.now().timestamp())
            local_random.seed(time_mix + local_random.getstate()[1][0] + time.perf_counter_ns())

            groups.append("".join(local_random.choices(characters, k=4)))

        return "-".join(groups)

    finally:
        # Restore the original random state
        random.setstate(original_state)


if __name__ == "__main__":
    # Set global random seed
    random.seed(42)

    # Test that external randomness is not affected
    print("External random number 1:", random.random())  # Always the same
    print("Task ID 1:", generate_task_id())  # Different each time
    print("External random number 1:", random.random())  # Always the same
    print("Task ID 1:", generate_task_id())  # Different each time
