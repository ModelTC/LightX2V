WAN_NEGATIVE_PROMPT = (
    "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

LTX2_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, "
    "background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, "
    "color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, "
    "incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, "
    "silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, "
    "added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, "
    "inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, "
    "or AI artifacts."
)

LINGBOT_VIDEO_NEGATIVE_PROMPT = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", '
    '"jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", '
    '"invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", '
    '"cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", '
    '"signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], '
    '"temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", '
    '"morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", '
    '"frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", '
    '"deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'
)
