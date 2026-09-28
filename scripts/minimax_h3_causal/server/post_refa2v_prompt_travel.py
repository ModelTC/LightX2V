import base64
from pathlib import Path

import requests
from loguru import logger


def file_to_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


if __name__ == "__main__":
    url = "http://localhost:8000/v1/tasks/video/"
    # First handoff example: meinv_01_ablation40_six_key.
    prompt = """subject_definitions:
<Subject 1>: The young East Asian woman with long straight black hair, fair skin, a dark blue off-shoulder floral lace top, and a silver pendant necklace.

summary:
[reference generation + audio reuse] The young East Asian woman remains centered in a close vertical portrait with a direct gaze, speaking directly to the camera with a neutral and relaxed demeanor, natural head and body movement, occasional blinking, and clearly visible timed gestures from the action track. The background composition stays locked to <Picture 1>.

retention_analysis:
<Picture 1>: fully_preserved - The original composition, fixed viewpoint, a pale interior wall with subtle vertical panel texture, soft beauty-portrait realism, gentle frontal illumination, and cool dark-blue contrast, and all visible spatial, lighting, text, object, and color relationships are fully preserved.
<Subject 1>: fully_preserved - The young East Asian woman with long straight black hair, fair skin, a dark blue off-shoulder floral lace top, and a silver pendant necklace remains fully preserved, including the long hair, lace pattern, exposed shoulders, pendant, facial makeup, pale wall, and close framing, together with the established pose, placement, and direct-to-camera role.

detailed_description:
Soft beauty-portrait realism, gentle frontal illumination, and cool dark-blue contrast are preserved throughout the image-based video. [Shot 1] Static Shot. The camera position and lens remain still. <Subject 1> remains centered in a close vertical portrait with a direct gaze. The subject is preserved as the young East Asian woman with long straight black hair, fair skin, a dark blue off-shoulder floral lace top, and a silver pendant necklace. The established setting remains a pale interior wall with subtle vertical panel texture. All identity-defining facial features, body proportions, hairstyle, clothing, accessories, source-supported objects, and spatial relationships stay consistent with <Picture 1>. In particular, the long hair, lace pattern, exposed shoulders, pendant, facial makeup, pale wall, and close framing retain their original appearance, scale, placement, color, and material character.

<Subject 1> speaks directly toward the camera with a neutral and relaxed manner. Mouth movement follows the complete reused source speech naturally. The eyes remain generally attentive to the lens, allowing only small conversational gaze adjustments that do not redirect the performance toward an invented person or event. Occasional blinks appear spontaneous and understated. Gentle changes in head orientation and a slight, natural response through the shoulders and upper body keep the speaking performance alive while preserving the source pose and framing. Facial expression follows the cadence of the audio without creating an unsupported emotional change.

When an action-track cue is active, the named gesture is large and readable (hands may leave the rest pose and enter frame), then returns toward rest after the beat. Do not replace the cue with a tiny talking-head beat. The gesture, head movement, blinking, facial articulation, and subtle body response operate as one continuous performance rather than a sequence of disconnected motions. Existing clothing and accessories follow only the subject's small movements and do not transform, detach, or change design. Any visible hair motion remains minimal and physically consistent. Existing props stay exactly where they are unless the reference already shows them held, in which case the original grip and object relationship remain stable without a newly invented use.

The environment remains completely unchanged around the subject. The original background geometry, furniture, plants, displays, architecture, landscape features, equipment, legible text, decorative elements, lighting direction, shadows, reflections, depth of field, and color relationships are preserved wherever visible. Background people, when present, remain incidental parts of the reference setting and do not become new primary subjects or initiate new events. No new person, prop, graphic, weather change, room, or replacement background appears. Illumination remains steady, and no screen or sign changes content. There is no camera movement, lens change, crop change, reframing, cut, transition, handheld drift, or environmental transformation. The result remains a single continuous direct-to-camera speaking moment grounded in the exact visual character of <Picture 1>.

overall_soundscape:
The complete synchronized target soundtrack is kept unchanged as the final audio track, preserving the entire original signal with no added, removed, or redesigned sound.

non_diegetic_music:
N/A"""

    action_prompts = {
        "0.000": [
            "<Subject 1> tips her chin down a short distance and brings it back up once, a single clear nod aimed at the lens. Her eyes stay on the camera through the nod. Both hands remain out of the gesture. The feet stay planted and the torso does not shift, lean, or walk."
        ],
        "4.000": [
            "<Subject 1> crosses both wrists in front of her chest into a clear X and holds the crossed wrists toward the camera. Her elbows stay bent and her feet stay planted. The feet stay planted and the torso does not shift, lean, or walk."
        ],
        "8.000": [
            "<Subject 1> raises both shoulders toward her ears in one even shrug, holds the raised shoulders, then lets them drop back to the rest line. Her face stays neutral and her hands stay down. The feet stay planted and the torso does not shift, lean, or walk."
        ],
        "12.000": [
            "<Subject 1> lifts her right hand beside her cheek, palm open toward the lens, and swings that palm left and right several times in a clear wave. She then lowers the right hand. The left hand stays down. The feet stay planted and the torso does not shift, lean, or walk."
        ],
        "16.000": [
            "<Subject 1> brings both hands up to chest height, curves the fingers, and joins each thumb to its index finger so the two hands form one heart aimed at the camera. She holds the heart steady. The feet stay planted and the torso does not shift, lean, or walk."
        ],
        "20.000": [
            "<Subject 1> lifts both hands onto the top of her head, one palm over the other, elbows out to the sides, and holds that hands-on-head pose. She does not stand up or step. The feet stay planted and the torso does not shift, lean, or walk."
        ],
        "24.000": [
            "<Subject 1> lifts both hands to her face and lays the palms flat over both eyes, fingers together, so the eyes are fully hidden. She holds the cover. The mouth stays visible below the hands. The feet stay planted and the torso does not shift, lean, or walk."
        ],
        "28.000": [
            "<Subject 1> bends her right elbow and points the right index finger straight at the camera, the other fingers curled. She holds the point at chest-to-face height, then lowers the arm. The left hand stays down. The feet stay planted and the torso does not shift, lean, or walk."
        ],
        "32.000": [
            "<Subject 1> raises both hands to chest height and turns both thumbs up, the other fingers curled into loose fists. She holds the two thumbs toward the lens. The feet stay planted and the torso does not shift, lean, or walk."
        ],
        "36.000": [
            "<Subject 1> lowers both hands out of the gesture and lets her face return to the original rest expression in <Picture 1>. She holds that rest, with no new hand sign and no new expression. The feet stay planted and the torso does not shift, lean, or walk."
        ],
    }

    message = {
        "task": "refa2v",
        "prompt": prompt,
        # Encode client-local media so the server does not need these paths.
        "image_path": file_to_base64("/data/nvme1/zhangbilang/zoe-diffusion-h3-prompttravel/configs/handoff/h3_six_key_action/images/meinv_01.png"),
        "audio_path": file_to_base64("/data/nvme1/zhangbilang/zoe-diffusion-h3-prompttravel/configs/handoff/h3_six_key_action/audio/female_qingdao_62s.mp3"),
        "action_prompts": action_prompts,
        "seed": 42,
        "num_frames": 974,
        "size": [1376, 768],
        "save_result_path": "./minimax_h3_causal_prompt_travel_meinv_01_ablation40_six_key.mp4",
    }

    logger.info(f"Submitting refa2v request to {url}")
    response = requests.post(url, json=message)
    response.raise_for_status()
    logger.info(f"response: {response.json()}")
