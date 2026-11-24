__all__ = [
    "PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION",
]


PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION = """
    Summarize human motion only from the user text for representation: action categories, key body-part movements, order/transitions, trajectory/direction, posture; include style/emotion/speed only if present. Explicitly capture laterality (left/right) when mentioned; do not guess. If multiple actions are described, indicate the count of distinct actions (e.g., actions=3) and their order. Do not invent missing info. Keep one concise paragraph.
"""
