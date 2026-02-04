import os
import random
from datetime import datetime

import gradio as gr
from utils.i18n import DEFAULT_LANG, t
from utils.model_components import (
    build_qwen_image_components,
    build_z_image_turbo_components,
)
from utils.model_utils import HF_AVAILABLE, MS_AVAILABLE, is_fp8_supported_gpu

MAX_NUMPY_SEED = 2**32 - 1


def generate_random_seed():
    """生成随机种子"""
    return random.randint(0, MAX_NUMPY_SEED)


def generate_unique_filename(output_dir, is_image=False):
    """生成唯一文件名"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = ".png" if is_image else ".mp4"
    filename = f"{timestamp}{extension}"
    return os.path.join(output_dir, filename)


def build_image_page(
    model_path,
    output_dir,
    run_inference,
    lang=DEFAULT_LANG,
):
    # 主布局：左右分栏
    with gr.Row():
        # 左侧：配置和输入区域
        with gr.Column(scale=5):
            # 模型配置区域
            with gr.Accordion(t("model_config", lang), open=True, elem_classes=["model-config"]):
                gr.Markdown(t("model_config_hint_image", lang))
                # FP8 支持提示
                if not is_fp8_supported_gpu():
                    gr.Markdown(t("fp8_not_supported", lang))

                model_path_input = gr.Textbox(value=model_path, visible=False)

                # 任务类型选择（前端显示大写，传递值小写）
                task_type_input = gr.Radio(
                    label=t("task_type", lang),
                    choices=[("I2I", "i2i"), ("T2I", "t2i")],
                    value="i2i",
                    info=t("task_type_info", lang),
                    elem_classes=["horizontal-radio"],
                )

                # 模型类型选择（根据任务类型动态更新）
                model_type_input = gr.Radio(
                    label=t("model_type", lang),
                    choices=["Qwen-Image-Edit-2511"],
                    value="Qwen-Image-Edit-2511",
                    info=t("model_type_info", lang),
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
                # 构建 Qwen 模型组件（默认显示）
                qwen_image_components = build_qwen_image_components(model_path, model_path_input, download_source_input, model_type_input, lang)
                qwen_image_dit_path_input = qwen_image_components["qwen_image_dit_path_input"]
                qwen_image_vae_path_input = qwen_image_components["qwen_image_vae_path_input"]
                qwen_image_scheduler_path_input = qwen_image_components["qwen_image_scheduler_path_input"]
                qwen25vl_encoder_path_input = qwen_image_components["qwen25vl_encoder_path_input"]
                qwen_image_dit_download_btn = qwen_image_components["qwen_image_dit_download_btn"]
                qwen_image_vae_download_btn = qwen_image_components["qwen_image_vae_download_btn"]
                qwen_image_scheduler_download_btn = qwen_image_components["qwen_image_scheduler_download_btn"]
                qwen25vl_encoder_download_btn = qwen_image_components["qwen25vl_encoder_download_btn"]
                qwen_image_use_lora_input = qwen_image_components["use_lora"]
                qwen_image_lora_path_input = qwen_image_components["lora_path_input"]
                qwen_image_lora_strength_input = qwen_image_components["lora_strength"]

                # 构建 Z-Image-Turbo 模型组件（默认隐藏）
                z_image_turbo_components = build_z_image_turbo_components(model_path, model_path_input, download_source_input, model_type_input, lang)
                z_image_turbo_dit_path_input = z_image_turbo_components["z_image_turbo_dit_path_input"]
                z_image_turbo_vae_path_input = z_image_turbo_components["z_image_turbo_vae_path_input"]
                z_image_turbo_scheduler_path_input = z_image_turbo_components["z_image_turbo_scheduler_path_input"]
                qwen3_encoder_path_input = z_image_turbo_components["qwen3_encoder_path_input"]
                z_image_turbo_dit_download_btn = z_image_turbo_components["z_image_turbo_dit_download_btn"]
                z_image_turbo_vae_download_btn = z_image_turbo_components["z_image_turbo_vae_download_btn"]
                z_image_turbo_scheduler_download_btn = z_image_turbo_components["z_image_turbo_scheduler_download_btn"]
                qwen3_encoder_download_btn = z_image_turbo_components["qwen3_encoder_download_btn"]
                z_image_turbo_use_lora_input = z_image_turbo_components["use_lora"]
                z_image_turbo_lora_path_input = z_image_turbo_components["lora_path_input"]
                z_image_turbo_lora_strength_input = z_image_turbo_components["lora_strength"]

                # 获取组件容器以便控制显示/隐藏
                qwen_components_group = qwen_image_components.get("components_group", None)
                z_image_turbo_components_group = z_image_turbo_components.get("components_group", None)

            with gr.Accordion(t("input_params", lang), open=True, elem_classes=["input-params"]):
                image_files = gr.File(
                    label=t("input_image", lang),
                    file_count="multiple",
                    file_types=["image"],
                    height=150,
                    interactive=True,
                )
                image_gallery = gr.Gallery(
                    label=t("image_preview", lang),
                    columns=4,
                    rows=2,
                    height=200,
                    object_fit="contain",
                    show_label=True,
                )
                image_path = gr.Textbox(label=t("image_path", lang), visible=False)

                aspect_ratio = gr.Dropdown(
                    label=t("aspect_ratio", lang),
                    choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                    value="1:1",
                    visible=True,
                    info=t("aspect_ratio_info", lang),
                )

                # 任务类型变化处理函数
                def on_task_type_change(task_type_val):
                    if task_type_val == "t2i":
                        # t2i 模式：显示 Qwen-Image-2512 和 Z-Image-Turbo，隐藏图片输入和宽高比
                        return (
                            gr.update(choices=["Z-Image-Turbo", "Qwen-Image-2512"], value="Z-Image-Turbo"),  # model_type_input
                            gr.update(visible=False),  # image_files
                            gr.update(visible=False),  # image_gallery
                            gr.update(value=""),  # image_path
                            gr.update(visible=False),  # aspect_ratio
                        )
                    else:
                        # i2i 模式：显示 Qwen-Image-Edit-2511，显示图片输入和宽高比
                        return (
                            gr.update(choices=["Qwen-Image-Edit-2511"], value="Qwen-Image-Edit-2511"),  # model_type_input
                            gr.update(visible=True),  # image_files
                            gr.update(visible=True),  # image_gallery
                            gr.update(),  # image_path (保持不变)
                            gr.update(visible=True),  # aspect_ratio
                        )

                # 绑定任务类型变化事件
                task_type_input.change(
                    fn=on_task_type_change,
                    inputs=[task_type_input],
                    outputs=[model_type_input, image_files, image_gallery, image_path, aspect_ratio],
                )

                # 模型类型变化处理函数
                def on_model_type_change(model_type_val, model_path_val):
                    # 控制组件组显示/隐藏
                    show_qwen = model_type_val in ["Qwen-Image-2512", "Qwen-Image-Edit-2511"]
                    show_z_image_turbo = model_type_val == "Z-Image-Turbo"

                    # 导入更新函数
                    from utils.model_handlers import update_model_status
                    from utils.model_utils import extract_model_name

                    # 更新模型选择
                    if model_type_val == "Qwen-Image-2512":
                        from utils.model_choices import get_qwen_image_2512_dit_choices, get_qwen_image_2512_scheduler_choices, get_qwen_image_2512_vae_choices

                        dit_choices = get_qwen_image_2512_dit_choices(model_path_val)
                        vae_choices = get_qwen_image_2512_vae_choices(model_path_val)
                        scheduler_choices = get_qwen_image_2512_scheduler_choices(model_path_val)

                        # 更新下载按钮状态
                        from utils.model_choices import get_qwen25vl_encoder_choices

                        qwen25vl_encoder_choices = get_qwen25vl_encoder_choices(model_path_val)
                        dit_btn_update = update_model_status(model_path_val, extract_model_name(dit_choices[0]) if dit_choices else "", "qwen_image_2512_dit")
                        vae_btn_update = update_model_status(model_path_val, extract_model_name(vae_choices[0]) if vae_choices else "", "qwen_image_2512_vae")
                        scheduler_btn_update = update_model_status(model_path_val, extract_model_name(scheduler_choices[0]) if scheduler_choices else "", "qwen_image_2512_scheduler")
                        qwen25vl_btn_update = update_model_status(model_path_val, extract_model_name(qwen25vl_encoder_choices[0]) if qwen25vl_encoder_choices else "", "qwen25vl_encoder")

                        return (
                            gr.update(visible=show_qwen),  # qwen_components_group
                            gr.update(visible=show_z_image_turbo),  # z_image_turbo_components_group
                            gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else ""),  # qwen_image_dit_path_input
                            gr.update(choices=vae_choices, value=vae_choices[0] if vae_choices else ""),  # qwen_image_vae_path_input
                            gr.update(choices=scheduler_choices, value=scheduler_choices[0] if scheduler_choices else ""),  # qwen_image_scheduler_path_input
                            gr.update(),  # z_image_turbo_dit_path_input (保持不变)
                            gr.update(),  # z_image_turbo_vae_path_input (保持不变)
                            gr.update(),  # z_image_turbo_scheduler_path_input (保持不变)
                            gr.update(),  # qwen3_encoder_path_input (保持不变)
                            gr.update(),  # aspect_ratio (保持不变)
                            dit_btn_update,  # qwen_image_dit_download_btn
                            vae_btn_update,  # qwen_image_vae_download_btn
                            scheduler_btn_update,  # qwen_image_scheduler_download_btn
                            qwen25vl_btn_update,  # qwen25vl_encoder_download_btn
                            gr.update(),  # z_image_turbo_dit_download_btn
                            gr.update(),  # z_image_turbo_vae_download_btn
                            gr.update(),  # z_image_turbo_scheduler_download_btn
                            gr.update(),  # qwen3_encoder_download_btn
                        )
                    elif model_type_val == "Z-Image-Turbo":
                        from utils.model_choices import get_qwen3_encoder_choices, get_z_image_turbo_dit_choices, get_z_image_turbo_scheduler_choices, get_z_image_turbo_vae_choices

                        dit_choices = get_z_image_turbo_dit_choices(model_path_val)
                        vae_choices = get_z_image_turbo_vae_choices(model_path_val)
                        scheduler_choices = get_z_image_turbo_scheduler_choices(model_path_val)
                        qwen3_encoder_choices = get_qwen3_encoder_choices(model_path_val)

                        # 更新下载按钮状态
                        dit_btn_update = update_model_status(model_path_val, extract_model_name(dit_choices[0]) if dit_choices else "", "z_image_turbo_dit")
                        vae_btn_update = update_model_status(model_path_val, extract_model_name(vae_choices[0]) if vae_choices else "", "z_image_turbo_vae")
                        scheduler_btn_update = update_model_status(model_path_val, extract_model_name(scheduler_choices[0]) if scheduler_choices else "", "z_image_turbo_scheduler")
                        qwen3_btn_update = update_model_status(model_path_val, extract_model_name(qwen3_encoder_choices[0]) if qwen3_encoder_choices else "", "qwen3_encoder")

                        return (
                            gr.update(visible=show_qwen),  # qwen_components_group
                            gr.update(visible=show_z_image_turbo),  # z_image_turbo_components_group
                            gr.update(),  # qwen_image_dit_path_input (保持不变)
                            gr.update(),  # qwen_image_vae_path_input (保持不变)
                            gr.update(),  # qwen_image_scheduler_path_input (保持不变)
                            gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else ""),  # z_image_turbo_dit_path_input
                            gr.update(choices=vae_choices, value=vae_choices[0] if vae_choices else ""),  # z_image_turbo_vae_path_input
                            gr.update(choices=scheduler_choices, value=scheduler_choices[0] if scheduler_choices else ""),  # z_image_turbo_scheduler_path_input
                            gr.update(choices=qwen3_encoder_choices, value=qwen3_encoder_choices[0] if qwen3_encoder_choices else ""),  # qwen3_encoder_path_input
                            gr.update(visible=True),  # aspect_ratio
                            gr.update(),  # qwen_image_dit_download_btn
                            gr.update(),  # qwen_image_vae_download_btn
                            gr.update(),  # qwen_image_scheduler_download_btn
                            gr.update(),  # qwen25vl_encoder_download_btn
                            dit_btn_update,  # z_image_turbo_dit_download_btn
                            vae_btn_update,  # z_image_turbo_vae_download_btn
                            scheduler_btn_update,  # z_image_turbo_scheduler_download_btn
                            qwen3_btn_update,  # qwen3_encoder_download_btn
                        )
                    else:  # Qwen-Image-Edit-2511
                        from utils.model_choices import get_qwen_image_dit_choices, get_qwen_image_scheduler_choices, get_qwen_image_vae_choices

                        dit_choices = get_qwen_image_dit_choices(model_path_val)
                        vae_choices = get_qwen_image_vae_choices(model_path_val)
                        scheduler_choices = get_qwen_image_scheduler_choices(model_path_val)

                        # 更新下载按钮状态
                        from utils.model_choices import get_qwen25vl_encoder_choices

                        qwen25vl_encoder_choices = get_qwen25vl_encoder_choices(model_path_val)
                        dit_btn_update = update_model_status(model_path_val, extract_model_name(dit_choices[0]) if dit_choices else "", "qwen_image_dit")
                        vae_btn_update = update_model_status(model_path_val, extract_model_name(vae_choices[0]) if vae_choices else "", "qwen_image_vae")
                        scheduler_btn_update = update_model_status(model_path_val, extract_model_name(scheduler_choices[0]) if scheduler_choices else "", "qwen_image_scheduler")
                        qwen25vl_btn_update = update_model_status(model_path_val, extract_model_name(qwen25vl_encoder_choices[0]) if qwen25vl_encoder_choices else "", "qwen25vl_encoder")

                        return (
                            gr.update(visible=show_qwen),  # qwen_components_group
                            gr.update(visible=show_z_image_turbo),  # z_image_turbo_components_group
                            gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else ""),  # qwen_image_dit_path_input
                            gr.update(choices=vae_choices, value=vae_choices[0] if vae_choices else ""),  # qwen_image_vae_path_input
                            gr.update(choices=scheduler_choices, value=scheduler_choices[0] if scheduler_choices else ""),  # qwen_image_scheduler_path_input
                            gr.update(),  # z_image_turbo_dit_path_input (保持不变)
                            gr.update(),  # z_image_turbo_vae_path_input (保持不变)
                            gr.update(),  # z_image_turbo_scheduler_path_input (保持不变)
                            gr.update(),  # qwen3_encoder_path_input (保持不变)
                            gr.update(),  # aspect_ratio
                            dit_btn_update,  # qwen_image_dit_download_btn
                            vae_btn_update,  # qwen_image_vae_download_btn
                            scheduler_btn_update,  # qwen_image_scheduler_download_btn
                            qwen25vl_btn_update,  # qwen25vl_encoder_download_btn
                            gr.update(),  # z_image_turbo_dit_download_btn
                            gr.update(),  # z_image_turbo_vae_download_btn
                            gr.update(),  # z_image_turbo_scheduler_download_btn
                            gr.update(),  # qwen3_encoder_download_btn
                        )

                # 绑定模型类型变化事件
                model_type_input.change(
                    fn=on_model_type_change,
                    inputs=[model_type_input, model_path_input],
                    outputs=[
                        qwen_components_group,
                        z_image_turbo_components_group,
                        qwen_image_dit_path_input,
                        qwen_image_vae_path_input,
                        qwen_image_scheduler_path_input,
                        z_image_turbo_dit_path_input,
                        z_image_turbo_vae_path_input,
                        z_image_turbo_scheduler_path_input,
                        qwen3_encoder_path_input,
                        aspect_ratio,
                        qwen_image_dit_download_btn,
                        qwen_image_vae_download_btn,
                        qwen_image_scheduler_download_btn,
                        qwen25vl_encoder_download_btn,
                        z_image_turbo_dit_download_btn,
                        z_image_turbo_vae_download_btn,
                        z_image_turbo_scheduler_download_btn,
                        qwen3_encoder_download_btn,
                    ],
                )

                def update_image_path_and_gallery(files):
                    if files is None or len(files) == 0:
                        return "", []
                    paths = [f.name if hasattr(f, "name") else f for f in files]
                    return ",".join(paths), paths

                image_files.change(
                    fn=update_image_path_and_gallery,
                    inputs=[image_files],
                    outputs=[image_path, image_gallery],
                )

                with gr.Row():
                    prompt = gr.Textbox(
                        label=t("prompt", lang),
                        lines=3,
                        placeholder=t("prompt_placeholder", lang),
                        max_lines=5,
                    )
                    negative_prompt = gr.Textbox(
                        label=t("negative_prompt", lang),
                        lines=3,
                        placeholder=t("negative_prompt_placeholder", lang),
                        max_lines=5,
                        value="",
                    )

                with gr.Row():
                    seed = gr.Slider(
                        label=t("random_seed", lang),
                        minimum=0,
                        maximum=MAX_NUMPY_SEED,
                        step=1,
                        value=generate_random_seed(),
                    )
                    infer_steps = gr.Slider(
                        label=t("infer_steps", lang),
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=8,
                        info=t("infer_steps_image_info", lang),
                    )
                    # aspect_ratio 已在上面定义，这里不需要重复定义
                    cfg_scale = gr.Slider(
                        label=t("cfg_scale", lang),
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1,
                        info=t("cfg_scale_info", lang),
                    )

                # 模型类型变化时更新 cfg_scale 和 infer_steps（Z-Image-Turbo 默认值）
                def on_model_type_change_for_defaults(model_type_val):
                    if model_type_val == "Z-Image-Turbo":
                        return (
                            gr.update(value=9),  # infer_steps
                            gr.update(value=1),  # cfg_scale
                        )
                    else:
                        return (
                            gr.update(),  # infer_steps (保持不变)
                            gr.update(),  # cfg_scale (保持不变)
                        )

                # 绑定模型类型变化事件，更新默认值
                model_type_input.change(
                    fn=on_model_type_change_for_defaults,
                    inputs=[model_type_input],
                    outputs=[infer_steps, cfg_scale],
                )

                save_result_path = gr.Textbox(
                    label=t("output_image_path", lang),
                    value=generate_unique_filename(output_dir, is_image=True),
                    info=t("output_image_path_info", lang),
                    visible=False,
                )

        # 右侧：输出区域
        with gr.Column(scale=4):
            with gr.Accordion(t("output_result", lang), open=True, elem_classes=["output-video"]):
                output_image = gr.Image(
                    label=t("output_image", lang),
                    height=600,
                    show_label=False,
                )
                infer_btn = gr.Button(
                    t("generate_image", lang),
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"],
                )

        def run_inference_wrapper(
            prompt_val,
            negative_prompt_val,
            save_result_path_val,
            infer_steps_val,
            seed_val,
            cfg_scale_val,
            model_path_val,
            model_type_val,
            task_type_val,
            image_path_val,
            qwen_image_dit_path_val,
            qwen_image_vae_path_val,
            qwen_image_scheduler_path_val,
            qwen25vl_encoder_path_val,
            z_image_turbo_dit_path_val,
            z_image_turbo_vae_path_val,
            z_image_turbo_scheduler_path_val,
            qwen3_encoder_path_val,
            aspect_ratio_val,
            qwen_image_use_lora_val,
            qwen_image_lora_path_val,
            qwen_image_lora_strength_val,
            z_image_turbo_use_lora_val,
            z_image_turbo_lora_path_val,
            z_image_turbo_lora_strength_val,
        ):
            # 根据模型类型传递不同的参数
            if model_type_val == "Z-Image-Turbo":
                result = run_inference(
                    prompt=prompt_val,
                    negative_prompt=negative_prompt_val,
                    save_result_path=save_result_path_val,
                    infer_steps=infer_steps_val,
                    seed=seed_val,
                    cfg_scale=cfg_scale_val,
                    model_path_input=model_path_val,
                    model_type_input=model_type_val,
                    task_type_input=task_type_val,
                    image_path=None,  # Z-Image-Turbo 只支持 t2i，不需要图片
                    qwen_image_dit_path_input=None,
                    qwen_image_vae_path_input=None,
                    qwen_image_scheduler_path_input=None,
                    qwen25vl_encoder_path_input=None,
                    z_image_dit_path_input=z_image_turbo_dit_path_val,
                    z_image_vae_path_input=z_image_turbo_vae_path_val,
                    z_image_scheduler_path_input=z_image_turbo_scheduler_path_val,
                    qwen3_encoder_path_input=qwen3_encoder_path_val,
                    aspect_ratio=aspect_ratio_val,  # Z-Image-Turbo 需要 aspect_ratio
                    use_lora=z_image_turbo_use_lora_val,
                    lora_path=z_image_turbo_lora_path_val,
                    lora_strength=z_image_turbo_lora_strength_val,
                )
            else:
                result = run_inference(
                    prompt=prompt_val,
                    negative_prompt=negative_prompt_val,
                    save_result_path=save_result_path_val,
                    infer_steps=infer_steps_val,
                    seed=seed_val,
                    cfg_scale=cfg_scale_val,
                    model_path_input=model_path_val,
                    model_type_input=model_type_val,
                    task_type_input=task_type_val,
                    image_path=image_path_val,
                    qwen_image_dit_path_input=qwen_image_dit_path_val,
                    qwen_image_vae_path_input=qwen_image_vae_path_val,
                    qwen_image_scheduler_path_input=qwen_image_scheduler_path_val,
                    qwen25vl_encoder_path_input=qwen25vl_encoder_path_val,
                    z_image_dit_path_input=None,
                    z_image_vae_path_input=None,
                    z_image_scheduler_path_input=None,
                    qwen3_encoder_path_input=None,
                    aspect_ratio=aspect_ratio_val,
                    use_lora=qwen_image_use_lora_val,
                    lora_path=qwen_image_lora_path_val,
                    lora_strength=qwen_image_lora_strength_val,
                )
            return gr.update(value=result)

        infer_btn.click(
            fn=run_inference_wrapper,
            inputs=[
                prompt,
                negative_prompt,
                save_result_path,
                infer_steps,
                seed,
                cfg_scale,
                model_path_input,
                model_type_input,
                task_type_input,
                image_path,
                qwen_image_dit_path_input,
                qwen_image_vae_path_input,
                qwen_image_scheduler_path_input,
                qwen25vl_encoder_path_input,
                z_image_turbo_dit_path_input,
                z_image_turbo_vae_path_input,
                z_image_turbo_scheduler_path_input,
                qwen3_encoder_path_input,
                aspect_ratio,
                qwen_image_use_lora_input,
                qwen_image_lora_path_input,
                qwen_image_lora_strength_input,
                z_image_turbo_use_lora_input,
                z_image_turbo_lora_path_input,
                z_image_turbo_lora_strength_input,
            ],
            outputs=[output_image],
        )
