from lightx2v.models.runners.request_args import COMMON, PROMPT, VIDEO, fields
from lightx2v.utils.request_spec import RequestSpec

SPECS = {
    "cosmos3": RequestSpec(
        {
            "t2i": COMMON | PROMPT | fields("size"),
            "t2v": VIDEO,
            "i2v": VIDEO | fields("image_path"),
            "t2av": VIDEO,
            "i2av": VIDEO | fields("image_path"),
            "i2va": COMMON
            | PROMPT
            | fields("action_mode", "action_path", "domain_name", "image_path", "policy_image", "policy_state", "save_action_path", "state_path", "size", "video_path", "view_point"),
            "v2av": COMMON | PROMPT | fields("action_mode", "action_path", "domain_name", "image_path", "save_action_path", "size", "video_path", "view_point"),
        },
    ),
}
