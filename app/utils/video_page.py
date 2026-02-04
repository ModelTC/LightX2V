import random

import gradio as gr
from utils.i18n import DEFAULT_LANG, t
from utils.model_choices import (
    get_clip_model_choices,
    get_dit_choices,
    get_high_noise_choices,
    get_low_noise_choices,
    get_t5_model_choices,
    get_vae_choices,
)
from utils.model_components import build_wan21_components, build_wan22_components
from utils.model_handlers import TOKENIZER_CONFIG, create_download_wrappers, create_update_status_wrappers
from utils.model_utils import HF_AVAILABLE, MS_AVAILABLE, check_model_exists, extract_model_name, is_fp8_supported_gpu

MAX_NUMPY_SEED = 2**32 - 1


def generate_random_seed():
    """生成随机种子"""
    return random.randint(0, MAX_NUMPY_SEED)


def generate_unique_filename(output_dir, is_image=False):
    """生成唯一文件名"""
    import os
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = ".png" if is_image else ".mp4"
    filename = f"{timestamp}{ext}"
    return os.path.join(output_dir, filename)


def build_video_page(
    model_path,
    output_dir,
    run_inference,
    lang=DEFAULT_LANG,
):
    """构建视频生成页面 - 重构简化版"""

    # 创建 update 和 download 包装函数
    update_funcs = create_update_status_wrappers()
    get_choices_funcs = {
        "get_dit_choices": get_dit_choices,
        "get_high_noise_choices": get_high_noise_choices,
        "get_low_noise_choices": get_low_noise_choices,
        "get_t5_model_choices": get_t5_model_choices,
        "get_clip_model_choices": get_clip_model_choices,
        "get_vae_choices": get_vae_choices,
    }
    download_funcs = create_download_wrappers(get_choices_funcs)

    # 判断模型是否是 distill 版本
    def is_distill_model(model_type, dit_path, high_noise_path):
        if model_type == "Wan2.1":
            check_name = dit_path.lower() if dit_path else ""
        else:
            check_name = high_noise_path.lower() if high_noise_path else ""
        return "4step" in check_name

    # 主布局：左右分栏
    with gr.Row():
        # 左侧：配置和输入区域
        with gr.Column(scale=5):
            # 模型配置区域
            with gr.Accordion(t("model_config", lang), open=True, elem_classes=["model-config"]):
                gr.Markdown(t("model_config_hint", lang))
                # FP8 支持提示
                if not is_fp8_supported_gpu():
                    gr.Markdown(t("fp8_not_supported", lang))

                model_path_input = gr.Textbox(value=model_path, visible=False)

                # 模型类型 + 任务类型 + 下载源
                with gr.Row():
                    model_type_input = gr.Radio(
                        label=t("model_type", lang),
                        choices=["Wan2.1", "Wan2.2"],
                        value="Wan2.1",
                        info=t("model_type_info", lang),
                        elem_classes=["horizontal-radio"],
                    )
                    task_type_input = gr.Radio(
                        label=t("task_type", lang),
                        choices=[("I2V", "i2v"), ("T2V", "t2v")],
                        value="i2v",
                        info=t("task_type_info", lang),
                        elem_classes=["horizontal-radio"],
                    )
                    download_source_input = gr.Radio(
                        label=t("download_source", lang),
                        choices=(["huggingface", "modelscope"] if (HF_AVAILABLE and MS_AVAILABLE) else (["huggingface"] if HF_AVAILABLE else ["modelscope"] if MS_AVAILABLE else [])),
                        value=("modelscope" if MS_AVAILABLE else ("huggingface" if HF_AVAILABLE else None)),
                        info=t("download_source_info", lang),
                        visible=HF_AVAILABLE or MS_AVAILABLE,
                        elem_classes=["horizontal-radio"],
                    )

                # wan2.1 和 wan2.2 组件
                wan21_components = build_wan21_components(model_path, model_path_input, model_type_input, task_type_input, download_source_input, update_funcs, download_funcs, lang)
                wan21_row = wan21_components["wan21_row"]
                dit_path_input = wan21_components["dit_path_input"]
                dit_download_btn = wan21_components["dit_download_btn"]
                wan21_use_lora_input = wan21_components["use_lora"]
                wan21_lora_path_input = wan21_components["lora_path_input"]
                wan21_lora_strength_input = wan21_components["lora_strength"]

                wan22_components = build_wan22_components(model_path, model_path_input, task_type_input, download_source_input, update_funcs, download_funcs, lang)
                wan22_row = wan22_components["wan22_row"]
                high_noise_path_input = wan22_components["high_noise_path_input"]
                low_noise_path_input = wan22_components["low_noise_path_input"]
                high_noise_download_btn = wan22_components["high_noise_download_btn"]
                low_noise_download_btn = wan22_components["low_noise_download_btn"]

                wan22_use_lora_input = wan22_components["use_lora"]
                wan22_high_noise_lora_path_input = wan22_components["high_noise_lora_path_input"]
                wan22_high_noise_lora_strength_input = wan22_components["high_noise_lora_strength"]
                wan22_low_noise_lora_path_input = wan22_components["low_noise_lora_path_input"]
                wan22_low_noise_lora_strength_input = wan22_components["low_noise_lora_strength"]

                # 文本编码器（模型 + Tokenizer）
                with gr.Row() as t5_row:
                    with gr.Column(scale=1):
                        t5_model_choices_init = get_t5_model_choices(model_path)
                        t5_path_input = gr.Dropdown(
                            label=t("text_encoder", lang),
                            choices=t5_model_choices_init,
                            value=t5_model_choices_init[0] if t5_model_choices_init else "",
                            allow_custom_value=True,
                        )
                        # 初始化时检查 T5 模型状态
                        t5_btn_visible = False
                        if t5_model_choices_init:
                            first_choice = t5_model_choices_init[0]
                            actual_name = extract_model_name(first_choice)
                            t5_exists = check_model_exists(model_path, actual_name)
                            t5_btn_visible = not t5_exists

                        t5_download_btn = gr.Button(t("download", lang), visible=t5_btn_visible, size="sm", variant="secondary")
                        t5_download_status = gr.Markdown("", visible=False)
                    with gr.Column(scale=1):
                        # 初始化时检查 T5 Tokenizer 状态
                        t5_tokenizer_config = TOKENIZER_CONFIG.get("t5")
                        t5_tokenizer_name = t5_tokenizer_config["name"] if t5_tokenizer_config else "google"
                        t5_tokenizer_exists = check_model_exists(model_path, t5_tokenizer_name) if t5_tokenizer_config else False
                        t5_tokenizer_status_text = f"{t5_tokenizer_name} ✅" if t5_tokenizer_exists else f"{t5_tokenizer_name} ❌"
                        t5_tokenizer_btn_visible = not t5_tokenizer_exists

                        t5_tokenizer_hint = gr.Dropdown(
                            label=t("text_encoder_tokenizer", lang),
                            choices=["google ✅", "google ❌"],
                            value=t5_tokenizer_status_text,
                            interactive=False,
                        )
                        t5_tokenizer_download_btn = gr.Button(t("download", lang), visible=t5_tokenizer_btn_visible, size="sm", variant="secondary")
                        t5_tokenizer_download_status = gr.Markdown("", visible=False)

                # 图像编码器（模型 + Tokenizer，条件显示）
                with gr.Row(visible=True) as clip_row:
                    with gr.Column(scale=1):
                        clip_model_choices_init = get_clip_model_choices(model_path)
                        clip_path_input = gr.Dropdown(
                            label=t("image_encoder", lang),
                            choices=clip_model_choices_init,
                            value=clip_model_choices_init[0] if clip_model_choices_init else "",
                            allow_custom_value=True,
                        )
                        # 初始化时检查 CLIP 模型状态
                        clip_btn_visible = False
                        if clip_model_choices_init:
                            first_choice = clip_model_choices_init[0]
                            actual_name = extract_model_name(first_choice)
                            clip_exists = check_model_exists(model_path, actual_name)
                            clip_btn_visible = not clip_exists

                        clip_download_btn = gr.Button(t("download", lang), visible=clip_btn_visible, size="sm", variant="secondary")
                        clip_download_status = gr.Markdown("", visible=False)
                    with gr.Column(scale=1):
                        # 初始化时检查 CLIP Tokenizer 状态
                        clip_tokenizer_config = TOKENIZER_CONFIG.get("clip")
                        clip_tokenizer_name = clip_tokenizer_config["name"] if clip_tokenizer_config else "xlm-roberta-large"
                        clip_tokenizer_exists = check_model_exists(model_path, clip_tokenizer_name) if clip_tokenizer_config else False
                        clip_tokenizer_status_text = f"{clip_tokenizer_name} ✅" if clip_tokenizer_exists else f"{clip_tokenizer_name} ❌"
                        clip_tokenizer_btn_visible = not clip_tokenizer_exists

                        clip_tokenizer_hint = gr.Dropdown(
                            label=t("image_encoder_tokenizer", lang),
                            choices=["xlm-roberta-large ✅", "xlm-roberta-large ❌"],
                            value=clip_tokenizer_status_text,
                            interactive=False,
                        )
                        clip_tokenizer_download_btn = gr.Button(t("download", lang), visible=clip_tokenizer_btn_visible, size="sm", variant="secondary")
                        clip_tokenizer_download_status = gr.Markdown("", visible=False)

                # VAE
                with gr.Row() as vae_row:
                    vae_choices_init = get_vae_choices(model_path)
                    vae_path_input = gr.Dropdown(
                        label=t("vae", lang),
                        choices=vae_choices_init,
                        value=(vae_choices_init[0] if vae_choices_init else ""),
                        allow_custom_value=True,
                        interactive=True,
                    )
                    # 初始化时检查 VAE 状态
                    vae_btn_visible = False
                    if vae_choices_init:
                        first_choice = vae_choices_init[0]
                        actual_name = extract_model_name(first_choice)
                        vae_exists = check_model_exists(model_path, actual_name)
                        vae_btn_visible = not vae_exists

                    vae_download_btn = gr.Button(t("download", lang), visible=vae_btn_visible, size="sm", variant="secondary")
                    vae_download_status = gr.Markdown("", visible=False)

                # 使用预创建的包装函数
                update_dit_status = update_funcs["update_dit_status"]
                update_high_noise_status = update_funcs["update_high_noise_status"]
                update_low_noise_status = update_funcs["update_low_noise_status"]
                update_t5_model_status = update_funcs["update_t5_model_status"]
                update_t5_tokenizer_status = update_funcs["update_t5_tokenizer_status"]
                update_clip_model_status = update_funcs["update_clip_model_status"]
                update_clip_tokenizer_status = update_funcs["update_clip_tokenizer_status"]
                update_vae_status = update_funcs["update_vae_status"]

                download_t5_model = download_funcs["download_t5_model"]
                download_t5_tokenizer = download_funcs["download_t5_tokenizer"]
                download_clip_model = download_funcs["download_clip_model"]
                download_clip_tokenizer = download_funcs["download_clip_tokenizer"]
                download_vae = download_funcs["download_vae"]

                # T5 模型路径变化时，只更新 T5 模型状态（不更新 Tokenizer）
                t5_path_input.change(
                    fn=update_t5_model_status,
                    inputs=[model_path_input, t5_path_input],
                    outputs=[t5_download_btn],
                )

                # CLIP 模型路径变化时，只更新 CLIP 模型状态（不更新 Tokenizer）
                clip_path_input.change(
                    fn=update_clip_model_status,
                    inputs=[model_path_input, clip_path_input],
                    outputs=[clip_download_btn],
                )

                # Tokenizer 独立检测（当 model_path 变化时，同时更新所有 Tokenizer）
                def update_all_tokenizers(model_path_val):
                    """更新所有 Tokenizer 状态"""
                    t5_tokenizer_result = update_t5_tokenizer_status(model_path_val)
                    clip_tokenizer_result = update_clip_tokenizer_status(model_path_val)
                    return (
                        t5_tokenizer_result[0],  # t5_tokenizer_hint
                        t5_tokenizer_result[1],  # t5_tokenizer_download_btn
                        clip_tokenizer_result[0],  # clip_tokenizer_hint
                        clip_tokenizer_result[1],  # clip_tokenizer_download_btn
                    )

                # 绑定 model_path 变化事件，更新所有 Tokenizer 状态
                model_path_input.change(
                    fn=update_all_tokenizers,
                    inputs=[model_path_input],
                    outputs=[
                        t5_tokenizer_hint,
                        t5_tokenizer_download_btn,
                        clip_tokenizer_hint,
                        clip_tokenizer_download_btn,
                    ],
                )

                vae_path_input.change(
                    fn=update_vae_status,
                    inputs=[model_path_input, vae_path_input],
                    outputs=[vae_download_btn],
                )

                # 绑定下载按钮事件
                t5_download_btn.click(
                    fn=download_t5_model,
                    inputs=[model_path_input, t5_path_input, download_source_input],
                    outputs=[t5_download_status, t5_download_btn, t5_path_input],
                )

                t5_tokenizer_download_btn.click(
                    fn=download_t5_tokenizer,
                    inputs=[model_path_input, download_source_input],
                    outputs=[
                        t5_tokenizer_download_status,
                        t5_tokenizer_hint,
                        t5_tokenizer_download_btn,
                    ],
                )

                clip_download_btn.click(
                    fn=download_clip_model,
                    inputs=[model_path_input, clip_path_input, download_source_input],
                    outputs=[clip_download_status, clip_download_btn, clip_path_input],
                )

                clip_tokenizer_download_btn.click(
                    fn=download_clip_tokenizer,
                    inputs=[model_path_input, download_source_input],
                    outputs=[
                        clip_tokenizer_download_status,
                        clip_tokenizer_hint,
                        clip_tokenizer_download_btn,
                    ],
                )

                vae_download_btn.click(
                    fn=download_vae,
                    inputs=[
                        model_path_input,
                        vae_path_input,
                        download_source_input,
                    ],
                    outputs=[
                        vae_download_status,
                        vae_download_btn,
                        vae_path_input,
                    ],
                )

                # 统一的模型组件更新函数
                def update_model_components(model_type, task_type, model_path_val):
                    """统一更新所有模型组件"""
                    show_clip = model_type == "Wan2.1" and task_type == "i2v"
                    show_image_input = task_type == "i2v"
                    is_wan21 = model_type == "Wan2.1"

                    # 获取模型选项
                    t5_choices = get_t5_model_choices(model_path_val)
                    vae_choices = get_vae_choices(model_path_val)
                    clip_choices = get_clip_model_choices(model_path_val) if show_clip else []

                    # 更新 Tokenizer 状态
                    t5_tokenizer_result = update_t5_tokenizer_status(model_path_val)
                    clip_tokenizer_result = update_clip_tokenizer_status(model_path_val)

                    # 更新模型下载按钮状态
                    t5_btn_update = update_t5_model_status(model_path_val, t5_choices[0] if t5_choices else "")
                    clip_btn_update = update_clip_model_status(model_path_val, clip_choices[0] if clip_choices else "") if show_clip else gr.update()
                    vae_btn_update = update_vae_status(model_path_val, vae_choices[0] if vae_choices else "")

                    if is_wan21:
                        dit_choices = get_dit_choices(model_path_val, "wan2.1", task_type)
                        # 更新 DIT 下载按钮状态
                        from utils.model_utils import extract_model_name

                        dit_btn_update = update_dit_status(model_path_val, extract_model_name(dit_choices[0]) if dit_choices else "", "wan2.1") if dit_choices else gr.update(visible=False)
                        return (
                            gr.update(visible=True),  # wan21_row
                            gr.update(visible=False),  # wan22_row
                            gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else "", visible=True),  # dit_path_input
                            gr.update(),  # high_noise_path_input
                            gr.update(),  # low_noise_path_input
                            gr.update(visible=show_clip),  # clip_row
                            gr.update(visible=True),  # vae_row
                            gr.update(visible=True),  # t5_row
                            gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),  # t5_path_input
                            gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else ""),  # clip_path_input
                            gr.update(choices=vae_choices, value=vae_choices[0] if vae_choices else ""),  # vae_path_input
                            gr.update(visible=show_image_input),  # image_input_row
                            gr.update(label=t("output_video_path", lang)),  # save_result_path
                            t5_tokenizer_result[0],  # t5_tokenizer_hint
                            t5_tokenizer_result[1],  # t5_tokenizer_download_btn
                            clip_tokenizer_result[0],  # clip_tokenizer_hint
                            clip_tokenizer_result[1],  # clip_tokenizer_download_btn
                            t5_btn_update,  # t5_download_btn
                            clip_btn_update,  # clip_download_btn
                            vae_btn_update,  # vae_download_btn
                            dit_btn_update,  # dit_download_btn
                            gr.update(),  # high_noise_download_btn
                            gr.update(),  # low_noise_download_btn
                        )
                    else:  # wan2.2
                        high_noise_choices = get_high_noise_choices(model_path_val, "wan2.2", task_type)
                        low_noise_choices = get_low_noise_choices(model_path_val, "wan2.2", task_type)
                        # 更新 high_noise 和 low_noise 下载按钮状态
                        from utils.model_utils import extract_model_name

                        high_noise_btn_update = (
                            update_high_noise_status(model_path_val, extract_model_name(high_noise_choices[0]) if high_noise_choices else "") if high_noise_choices else gr.update(visible=False)
                        )
                        low_noise_btn_update = (
                            update_low_noise_status(model_path_val, extract_model_name(low_noise_choices[0]) if low_noise_choices else "") if low_noise_choices else gr.update(visible=False)
                        )
                        return (
                            gr.update(visible=False),  # wan21_row
                            gr.update(visible=True),  # wan22_row
                            gr.update(visible=False),  # dit_path_input
                            gr.update(choices=high_noise_choices, value=high_noise_choices[0] if high_noise_choices else ""),  # high_noise_path_input
                            gr.update(choices=low_noise_choices, value=low_noise_choices[0] if low_noise_choices else ""),  # low_noise_path_input
                            gr.update(visible=show_clip),  # clip_row
                            gr.update(visible=True),  # vae_row
                            gr.update(visible=True),  # t5_row
                            gr.update(choices=t5_choices, value=t5_choices[0] if t5_choices else ""),  # t5_path_input
                            gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else ""),  # clip_path_input
                            gr.update(choices=vae_choices, value=vae_choices[0] if vae_choices else ""),  # vae_path_input
                            gr.update(visible=show_image_input),  # image_input_row
                            gr.update(label=t("output_video_path", lang)),  # save_result_path
                            t5_tokenizer_result[0],  # t5_tokenizer_hint
                            t5_tokenizer_result[1],  # t5_tokenizer_download_btn
                            clip_tokenizer_result[0],  # clip_tokenizer_hint
                            clip_tokenizer_result[1],  # clip_tokenizer_download_btn
                            t5_btn_update,  # t5_download_btn
                            clip_btn_update,  # clip_download_btn
                            vae_btn_update,  # vae_download_btn
                            gr.update(),  # dit_download_btn
                            high_noise_btn_update,  # high_noise_download_btn
                            low_noise_btn_update,  # low_noise_download_btn
                        )

            # 输入参数区域
            with gr.Accordion(t("input_params", lang), open=True, elem_classes=["input-params"]):
                # 图片输入（i2v 时显示）
                with gr.Column(visible=True) as image_input_row:
                    image_files = gr.File(
                        label=t("input_images", lang),
                        file_count="multiple",
                        file_types=["image"],
                        height=150,
                        interactive=True,
                    )
                    # 图片预览 Gallery
                    image_gallery = gr.Gallery(
                        label=t("uploaded_images_preview", lang),
                        columns=4,
                        rows=2,
                        height=200,
                        object_fit="contain",
                        show_label=True,
                    )
                    # 将多个文件路径转换为逗号分隔的字符串
                    image_path = gr.Textbox(
                        label=t("image_path", lang),
                        visible=False,
                    )

                    def update_image_path_and_gallery(files):
                        if files is None or len(files) == 0:
                            return "", []
                        # 提取文件路径
                        paths = [f.name if hasattr(f, "name") else f for f in files]
                        # 返回逗号分隔的路径和图片列表用于 Gallery 显示
                        return ",".join(paths), paths

                    image_files.change(
                        fn=update_image_path_and_gallery,
                        inputs=[image_files],
                        outputs=[image_path, image_gallery],
                    )

                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(
                            label=t("prompt", lang),
                            lines=3,
                            placeholder=t("prompt_placeholder", lang),
                            max_lines=5,
                        )
                    with gr.Column():
                        negative_prompt = gr.Textbox(
                            label=t("negative_prompt", lang),
                            lines=3,
                            placeholder=t("negative_prompt_placeholder", lang),
                            max_lines=5,
                            value="镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        )
                    with gr.Column(visible=True):
                        resolution = gr.Dropdown(
                            choices=["480p", "540p", "720p"],
                            value="480p",
                            label=t("max_resolution", lang),
                            info=t("max_resolution_info", lang),
                        )

                    with gr.Column(scale=9):
                        seed = gr.Slider(
                            label=t("random_seed", lang),
                            minimum=0,
                            maximum=MAX_NUMPY_SEED,
                            step=1,
                            value=generate_random_seed(),
                        )
                    with gr.Column():
                        default_dit = get_dit_choices(model_path, "wan2.1", "i2v")[0] if get_dit_choices(model_path, "wan2.1", "i2v") else ""
                        default_high_noise = get_high_noise_choices(model_path, "wan2.2", "i2v")[0] if get_high_noise_choices(model_path, "wan2.2", "i2v") else ""
                        default_is_distill = is_distill_model("wan2.1", default_dit, default_high_noise)
                        if default_is_distill:
                            infer_steps = gr.Slider(
                                label=t("infer_steps", lang),
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=4,
                                info=t("infer_steps_distill", lang),
                            )
                        else:
                            infer_steps = gr.Slider(
                                label=t("infer_steps", lang),
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=40,
                            )

                # 根据模型类别设置默认CFG
                default_cfg_scale = 1 if default_is_distill else 5

                with gr.Row():
                    sample_shift = gr.Slider(
                        label=t("sample_shift", lang),
                        value=5,
                        minimum=0,
                        maximum=10,
                        step=1,
                        info=t("sample_shift_info", lang),
                    )
                    cfg_scale = gr.Slider(
                        label=t("cfg_scale", lang),
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=default_cfg_scale,
                        info=t("cfg_scale_info", lang),
                    )

                # 统一的模型参数更新函数（推理步数和CFG）
                def update_model_params(model_type, dit_path, high_noise_path):
                    is_distill = is_distill_model(model_type, dit_path, high_noise_path)
                    return (
                        gr.update(value=4 if is_distill else 40),  # infer_steps
                        gr.update(value=1 if is_distill else 5),  # cfg_scale
                    )

                # 监听模型路径和类型变化
                inputs = [model_type_input, dit_path_input, high_noise_path_input]
                for trigger in [dit_path_input, high_noise_path_input, model_type_input]:
                    trigger.change(
                        fn=update_model_params,
                        inputs=inputs,
                        outputs=[infer_steps, cfg_scale],
                    )

                with gr.Row(visible=True):
                    fps = gr.Slider(
                        label=t("fps", lang),
                        minimum=8,
                        maximum=30,
                        step=1,
                        value=16,
                        info=t("fps_info", lang),
                    )
                    video_duration = gr.Slider(
                        label=t("video_duration", lang),
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        value=5.0,
                        info=t("video_duration_info", lang),
                    )

                save_result_path = gr.Textbox(
                    label=t("output_video_path", lang),
                    value=generate_unique_filename(output_dir),
                    info=t("output_video_path_info", lang),
                    visible=False,  # 隐藏输出路径，自动生成
                )

            # 模型类型和任务类型切换都使用同一个函数（在 image_input_row 和 save_result_path 定义后绑定）
            model_type_input.change(
                fn=update_model_components,
                inputs=[model_type_input, task_type_input, model_path_input],
                outputs=[
                    wan21_row,
                    wan22_row,
                    dit_path_input,
                    high_noise_path_input,
                    low_noise_path_input,
                    clip_row,
                    vae_row,
                    t5_row,
                    t5_path_input,
                    clip_path_input,
                    vae_path_input,
                    image_input_row,
                    save_result_path,
                    t5_tokenizer_hint,
                    t5_tokenizer_download_btn,
                    clip_tokenizer_hint,
                    clip_tokenizer_download_btn,
                    t5_download_btn,
                    clip_download_btn,
                    vae_download_btn,
                    dit_download_btn,
                    high_noise_download_btn,
                    low_noise_download_btn,
                ],
            )

            task_type_input.change(
                fn=update_model_components,
                inputs=[model_type_input, task_type_input, model_path_input],
                outputs=[
                    wan21_row,
                    wan22_row,
                    dit_path_input,
                    high_noise_path_input,
                    low_noise_path_input,
                    clip_row,
                    vae_row,
                    t5_row,
                    t5_path_input,
                    clip_path_input,
                    vae_path_input,
                    image_input_row,
                    save_result_path,
                    t5_tokenizer_hint,
                    t5_tokenizer_download_btn,
                    clip_tokenizer_hint,
                    clip_tokenizer_download_btn,
                    t5_download_btn,
                    clip_download_btn,
                    vae_download_btn,
                    dit_download_btn,
                    high_noise_download_btn,
                    low_noise_download_btn,
                ],
            )

        # 右侧：输出区域
        with gr.Column(scale=4):
            with gr.Accordion(t("output_result", lang), open=True, elem_classes=["output-video"]):
                output_video = gr.Video(
                    label="",
                    height=600,
                    autoplay=True,
                    show_label=False,
                    visible=True,
                )

                infer_btn = gr.Button(
                    t("generate_video", lang),
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"],
                )

    # 包装推理函数，视频页面只返回视频，只传递必要的参数
    def run_inference_wrapper(
        prompt_val,
        negative_prompt_val,
        save_result_path_val,
        infer_steps_val,
        video_duration_val,
        resolution_val,
        seed_val,
        sample_shift_val,
        cfg_scale_val,
        fps_val,
        model_path_val,
        model_type_val,
        task_type_val,
        dit_path_val,
        high_noise_path_val,
        low_noise_path_val,
        t5_path_val,
        clip_path_val,
        vae_path_val,
        image_path_val,
        wan21_use_lora_val,
        wan21_lora_path_val,
        wan21_lora_strength_val,
        wan22_use_lora_val,
        wan22_high_noise_lora_path_val,
        wan22_high_noise_lora_strength_val,
        wan22_low_noise_lora_path_val,
        wan22_low_noise_lora_strength_val,
    ):
        # 视频页面不需要 qwen_image 相关参数，使用 None
        # offload 和 rope 相关参数已在 run_inference 中通过 get_auto_config_dict 自动决定

        # 将视频时长（秒）转换为帧数
        num_frames_val = int(video_duration_val * fps_val) + 1

        # 提取所有路径参数中的模型名，移除状态符号（✅/❌）
        from utils.model_utils import extract_model_name

        dit_path_val = extract_model_name(dit_path_val) if dit_path_val else None
        high_noise_path_val = extract_model_name(high_noise_path_val) if high_noise_path_val else None
        low_noise_path_val = extract_model_name(low_noise_path_val) if low_noise_path_val else None
        t5_path_val = extract_model_name(t5_path_val) if t5_path_val else None
        clip_path_val = extract_model_name(clip_path_val) if clip_path_val else ""
        vae_path_val = extract_model_name(vae_path_val) if vae_path_val else None

        if model_type_val == "Wan2.1":
            use_lora = wan21_use_lora_val
            lora_path = wan21_lora_path_val
            lora_strength = wan21_lora_strength_val
            low_noise_lora_path = None
            low_noise_lora_strength = None
            high_noise_lora_path = None
            high_noise_lora_strength = None
        else:
            use_lora = wan22_use_lora_val
            high_noise_lora_path = wan22_high_noise_lora_path_val
            high_noise_lora_strength = wan22_high_noise_lora_strength_val
            low_noise_lora_path = wan22_low_noise_lora_path_val
            low_noise_lora_strength = wan22_low_noise_lora_strength_val
            lora_path = None
            lora_strength = None

        result = run_inference(
            prompt=prompt_val,
            negative_prompt=negative_prompt_val,
            save_result_path=save_result_path_val,
            infer_steps=infer_steps_val,
            num_frames=num_frames_val,
            resolution=resolution_val,
            seed=seed_val,
            sample_shift=sample_shift_val,
            cfg_scale=cfg_scale_val,  # enable_cfg 会根据 cfg_scale 在 run_inference 内部自动设置
            fps=fps_val,
            model_path_input=model_path_val,
            model_type_input=model_type_val,
            task_type_input=task_type_val,
            dit_path_input=dit_path_val,
            high_noise_path_input=high_noise_path_val,
            low_noise_path_input=low_noise_path_val,
            t5_path_input=t5_path_val,
            clip_path_input=clip_path_val,
            vae_path=vae_path_val,
            image_path=image_path_val,
            use_lora=use_lora,
            lora_path=lora_path,
            lora_strength=lora_strength,
            high_noise_lora_path=high_noise_lora_path,
            high_noise_lora_strength=high_noise_lora_strength,
            low_noise_lora_path=low_noise_lora_path,
            low_noise_lora_strength=low_noise_lora_strength,
        )
        return gr.update(value=result)

    infer_btn.click(
        fn=run_inference_wrapper,
        inputs=[
            prompt,
            negative_prompt,
            save_result_path,
            infer_steps,
            video_duration,
            resolution,
            seed,
            sample_shift,
            cfg_scale,
            fps,
            model_path_input,
            model_type_input,
            task_type_input,
            dit_path_input,
            high_noise_path_input,
            low_noise_path_input,
            t5_path_input,
            clip_path_input,
            vae_path_input,
            image_path,
            wan21_use_lora_input,
            wan21_lora_path_input,
            wan21_lora_strength_input,
            wan22_use_lora_input,
            wan22_high_noise_lora_path_input,
            wan22_high_noise_lora_strength_input,
            wan22_low_noise_lora_path_input,
            wan22_low_noise_lora_strength_input,
        ],
        outputs=[output_video],
    )
