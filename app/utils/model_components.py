"""模型组件构建模块
为不同的模型类型创建 UI 组件
这些函数需要在 Gradio 的 with 块内调用
"""

import os

import gradio as gr
from utils.i18n import t
from utils.model_choices import (
    get_dit_choices,
    get_high_noise_choices,
    get_low_noise_choices,
    get_qwen3_encoder_choices,
    get_qwen25vl_encoder_choices,
    get_qwen_image_2512_dit_choices,
    get_qwen_image_2512_scheduler_choices,
    get_qwen_image_2512_vae_choices,
    get_qwen_image_dit_choices,
    get_qwen_image_scheduler_choices,
    get_qwen_image_vae_choices,
    get_z_image_turbo_dit_choices,
    get_z_image_turbo_scheduler_choices,
    get_z_image_turbo_vae_choices,
)
from utils.model_handlers import (
    download_model_handler,
    update_model_status,
)


def get_lora_choices(model_path):
    """获取 LoRA 模型可选项，从 model_path/loras 目录检索

    Args:
        model_path: 模型根路径

    Returns:
        list: LoRA 文件列表，如果没有则返回 [""]
    """
    loras_dir = os.path.join(model_path, "loras")
    if not os.path.exists(loras_dir):
        return [""]

    lora_files = []
    # 支持常见的 LoRA 文件格式
    lora_extensions = [".safetensors", ".pt", ".pth", ".ckpt"]

    for item in os.listdir(loras_dir):
        item_path = os.path.join(loras_dir, item)
        if os.path.isfile(item_path):
            # 检查是否是 LoRA 文件
            if any(item.lower().endswith(ext) for ext in lora_extensions):
                lora_files.append(item)

    # 按文件名排序
    lora_files.sort()

    return lora_files if lora_files else [""]


def build_wan21_components(model_path, model_path_input, model_type_input, task_type_input, download_source_input, update_funcs, download_funcs, lang="zh"):
    """构建 wan2.1 模型相关组件
    必须在 Gradio 的 with 块内调用

    Returns:
        dict: 包含所有 wan2.1 相关组件的字典
    """
    # wan2.1：Diffusion模型
    with gr.Column(elem_classes=["diffusion-model-group"]) as wan21_row:
        with gr.Row():
            with gr.Column(scale=5):
                dit_choices_init = get_dit_choices(model_path, "wan2.1", "i2v")
                dit_path_input = gr.Dropdown(
                    label="🎨 Diffusion模型",
                    choices=dit_choices_init,
                    value=dit_choices_init[0] if dit_choices_init else "",
                    allow_custom_value=True,
                    visible=True,
                )
            with gr.Column(scale=1, min_width=150):
                # 初始化时检查模型状态，确定下载按钮的初始可见性
                from utils.model_utils import check_model_exists, extract_model_name

                dit_btn_visible = False
                if dit_choices_init:
                    first_choice = dit_choices_init[0]
                    actual_name = extract_model_name(first_choice)
                    dit_exists = check_model_exists(model_path, actual_name)
                    dit_btn_visible = not dit_exists

                dit_download_btn = gr.Button("📥 下载", visible=dit_btn_visible, size="sm", variant="secondary")
        dit_download_status = gr.Markdown("", visible=False)

        lora_choices_init = get_lora_choices(model_path)
        with gr.Row():
            with gr.Column(scale=1):
                use_lora = gr.Checkbox(
                    label=t("use_lora", lang),
                    value=False,
                )
                lora_path_input = gr.Dropdown(
                    label=t("lora", lang),
                    choices=lora_choices_init,
                    value=lora_choices_init[0] if lora_choices_init and lora_choices_init[0] else "",
                    allow_custom_value=True,
                    visible=False,
                    info=t("lora_info", lang),
                )
                lora_strength = gr.Slider(
                    label=t("lora_strength", lang),
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0,
                    visible=False,
                    info=t("lora_strength_info", lang),
                )

    # 绑定事件
    update_dit_status = update_funcs["update_dit_status"]
    download_dit_model = download_funcs["download_dit_model"]

    dit_path_input.change(
        fn=lambda mp, mn, mt: update_dit_status(mp, mn, mt),
        inputs=[model_path_input, dit_path_input, model_type_input],
        outputs=[dit_download_btn],
    )

    dit_download_btn.click(
        fn=download_dit_model,
        inputs=[model_path_input, dit_path_input, model_type_input, task_type_input, download_source_input],
        outputs=[dit_download_status, dit_download_btn, dit_path_input],
    )

    # LoRA 开关变化时显示/隐藏相关组件
    def on_use_lora_change(use_lora_val):
        return (
            gr.update(visible=use_lora_val),  # lora_path_input
            gr.update(visible=use_lora_val),  # lora_strength
        )

    use_lora.change(
        fn=on_use_lora_change,
        inputs=[use_lora],
        outputs=[lora_path_input, lora_strength],
    )

    # 当 model_path 变化时更新 LoRA 选择
    def update_lora_choices(model_path_val):
        lora_choices = get_lora_choices(model_path_val)
        return gr.update(choices=lora_choices, value=lora_choices[0] if lora_choices and lora_choices[0] else "")

    model_path_input.change(
        fn=update_lora_choices,
        inputs=[model_path_input],
        outputs=[lora_path_input],
    )

    return {
        "wan21_row": wan21_row,
        "dit_path_input": dit_path_input,
        "dit_download_btn": dit_download_btn,
        "dit_download_status": dit_download_status,
        "use_lora": use_lora,
        "lora_path_input": lora_path_input,
        "lora_strength": lora_strength,
    }


def build_wan22_components(model_path, model_path_input, task_type_input, download_source_input, update_funcs, download_funcs, lang="zh"):
    """构建 wan2.2 模型相关组件
    必须在 Gradio 的 with 块内调用

    Returns:
        dict: 包含所有 wan2.2 相关组件的字典
    """
    # wan2.2 专用：高噪模型 + 低噪模型
    with gr.Row(visible=False, elem_classes=["wan22-row"]) as wan22_row:
        with gr.Column(scale=1):
            high_noise_choices_init = get_high_noise_choices(model_path, "wan2.2", "i2v")
            high_noise_path_input = gr.Dropdown(
                label="🔊 高噪模型",
                choices=high_noise_choices_init,
                value=high_noise_choices_init[0] if high_noise_choices_init else "",
                allow_custom_value=True,
            )
            # 初始化时检查模型状态
            from utils.model_utils import check_model_exists, extract_model_name

            high_noise_btn_visible = False
            if high_noise_choices_init:
                first_choice = high_noise_choices_init[0]
                actual_name = extract_model_name(first_choice)
                high_noise_exists = check_model_exists(model_path, actual_name)
                high_noise_btn_visible = not high_noise_exists

            high_noise_download_btn = gr.Button("📥 下载", visible=high_noise_btn_visible, size="sm", variant="secondary")
            high_noise_download_status = gr.Markdown("", visible=False)
        with gr.Column(scale=1):
            low_noise_choices_init = get_low_noise_choices(model_path, "wan2.2", "i2v")
            low_noise_path_input = gr.Dropdown(
                label="🔇 低噪模型",
                choices=low_noise_choices_init,
                value=low_noise_choices_init[0] if low_noise_choices_init else "",
                allow_custom_value=True,
            )
            # 初始化时检查模型状态
            low_noise_btn_visible = False
            if low_noise_choices_init:
                first_choice = low_noise_choices_init[0]
                actual_name = extract_model_name(first_choice)
                low_noise_exists = check_model_exists(model_path, actual_name)
                low_noise_btn_visible = not low_noise_exists

            low_noise_download_btn = gr.Button("📥 下载", visible=low_noise_btn_visible, size="sm", variant="secondary")
            low_noise_download_status = gr.Markdown("", visible=False)

        # LoRA 组件（Wan2.2 需要为 high_noise 和 low_noise 分别配置）
        lora_choices_init = get_lora_choices(model_path)
        with gr.Row():
            with gr.Column(scale=1):
                use_lora = gr.Checkbox(
                    label=t("use_lora", lang),
                    value=False,
                )
                # High Noise LoRA
                high_noise_lora_path_input = gr.Dropdown(
                    label=t("high_noise_lora", lang),
                    choices=lora_choices_init,
                    value=lora_choices_init[0] if lora_choices_init and lora_choices_init[0] else "",
                    allow_custom_value=True,
                    visible=False,
                    info=t("high_noise_lora_info", lang),
                )
                high_noise_lora_strength = gr.Slider(
                    label=t("high_noise_lora_strength", lang),
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0,
                    visible=False,
                    info=t("high_noise_lora_strength_info", lang),
                )
                # Low Noise LoRA
                low_noise_lora_path_input = gr.Dropdown(
                    label=t("low_noise_lora", lang),
                    choices=lora_choices_init,
                    value=lora_choices_init[0] if lora_choices_init and lora_choices_init[0] else "",
                    allow_custom_value=True,
                    visible=False,
                    info=t("low_noise_lora_info", lang),
                )
                low_noise_lora_strength = gr.Slider(
                    label=t("low_noise_lora_strength", lang),
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0,
                    visible=False,
                    info=t("low_noise_lora_strength_info", lang),
                )

    # 绑定事件
    update_high_noise_status = update_funcs["update_high_noise_status"]
    update_low_noise_status = update_funcs["update_low_noise_status"]
    download_high_noise_model = download_funcs["download_high_noise_model"]
    download_low_noise_model = download_funcs["download_low_noise_model"]

    high_noise_path_input.change(
        fn=update_high_noise_status,
        inputs=[model_path_input, high_noise_path_input],
        outputs=[high_noise_download_btn],
    )

    low_noise_path_input.change(
        fn=update_low_noise_status,
        inputs=[model_path_input, low_noise_path_input],
        outputs=[low_noise_download_btn],
    )

    high_noise_download_btn.click(
        fn=download_high_noise_model,
        inputs=[model_path_input, high_noise_path_input, task_type_input, download_source_input],
        outputs=[high_noise_download_status, high_noise_download_btn, high_noise_path_input],
    )

    low_noise_download_btn.click(
        fn=download_low_noise_model,
        inputs=[model_path_input, low_noise_path_input, task_type_input, download_source_input],
        outputs=[low_noise_download_status, low_noise_download_btn, low_noise_path_input],
    )

    # LoRA 开关变化时显示/隐藏相关组件
    def on_use_lora_change(use_lora_val):
        return (
            gr.update(visible=use_lora_val),  # high_noise_lora_path_input
            gr.update(visible=use_lora_val),  # high_noise_lora_strength
            gr.update(visible=use_lora_val),  # low_noise_lora_path_input
            gr.update(visible=use_lora_val),  # low_noise_lora_strength
        )

    use_lora.change(
        fn=on_use_lora_change,
        inputs=[use_lora],
        outputs=[high_noise_lora_path_input, high_noise_lora_strength, low_noise_lora_path_input, low_noise_lora_strength],
    )

    # 当 model_path 变化时更新 LoRA 选择
    def update_lora_choices(model_path_val):
        lora_choices = get_lora_choices(model_path_val)
        return (
            gr.update(choices=lora_choices, value=lora_choices[0] if lora_choices and lora_choices[0] else ""),  # high_noise_lora_path_input
            gr.update(choices=lora_choices, value=lora_choices[0] if lora_choices and lora_choices[0] else ""),  # low_noise_lora_path_input
        )

    model_path_input.change(
        fn=update_lora_choices,
        inputs=[model_path_input],
        outputs=[high_noise_lora_path_input, low_noise_lora_path_input],
    )

    return {
        "wan22_row": wan22_row,
        "high_noise_path_input": high_noise_path_input,
        "high_noise_download_btn": high_noise_download_btn,
        "high_noise_download_status": high_noise_download_status,
        "low_noise_path_input": low_noise_path_input,
        "low_noise_download_btn": low_noise_download_btn,
        "low_noise_download_status": low_noise_download_status,
        "use_lora": use_lora,
        "high_noise_lora_path_input": high_noise_lora_path_input,
        "high_noise_lora_strength": high_noise_lora_strength,
        "low_noise_lora_path_input": low_noise_lora_path_input,
        "low_noise_lora_strength": low_noise_lora_strength,
    }


def build_qwen_image_components(model_path, model_path_input, download_source_input, model_type_input, lang="zh"):
    """构建 Qwen-Image-Edit-2511/Qwen-Image-2512 模型相关组件（用于图片页面）
    必须在 Gradio 的 with 块内调用

    Args:
        model_path: 模型路径
        model_path_input: 模型路径输入组件
        download_source_input: 下载源输入组件
        model_type_input: 模型类型输入组件（用于动态更新）
        lang: 语言代码，默认为 "zh"

    Returns:
        dict: 包含所有 Qwen-Image 相关组件的字典
    """

    # 根据模型类型选择函数
    def get_dit_choices_func(model_path_val):
        model_type_val = model_type_input.value if hasattr(model_type_input, "value") else "Qwen-Image-Edit-2511"
        if model_type_val == "Qwen-Image-2512":
            return get_qwen_image_2512_dit_choices(model_path_val)
        else:
            return get_qwen_image_dit_choices(model_path_val)

    def get_vae_choices_func(model_path_val):
        model_type_val = model_type_input.value if hasattr(model_type_input, "value") else "Qwen-Image-Edit-2511"
        if model_type_val == "Qwen-Image-2512":
            return get_qwen_image_2512_vae_choices(model_path_val)
        else:
            return get_qwen_image_vae_choices(model_path_val)

    def get_scheduler_choices_func(model_path_val):
        model_type_val = model_type_input.value if hasattr(model_type_input, "value") else "Qwen-Image-Edit-2511"
        if model_type_val == "Qwen-Image-2512":
            return get_qwen_image_2512_scheduler_choices(model_path_val)
        else:
            return get_qwen_image_scheduler_choices(model_path_val)

    qwen_image_dit_choices_init = get_dit_choices_func(model_path)
    qwen_image_vae_choices_init = get_vae_choices_func(model_path)
    qwen_image_scheduler_choices_init = get_scheduler_choices_func(model_path)

    # 初始化时检查模型状态，确定下载按钮的初始可见性
    def get_initial_download_btn_visibility(choices, model_path_val, model_type_val, base_category):
        if not choices:
            return False
        first_choice = choices[0]
        from utils.model_utils import check_model_exists, extract_model_name

        actual_name = extract_model_name(first_choice)
        exists = check_model_exists(model_path_val, actual_name)
        return not exists

    initial_model_type = model_type_input.value if hasattr(model_type_input, "value") else "Qwen-Image-Edit-2511"
    dit_btn_visible = get_initial_download_btn_visibility(qwen_image_dit_choices_init, model_path, initial_model_type, "dit")
    vae_btn_visible = get_initial_download_btn_visibility(qwen_image_vae_choices_init, model_path, initial_model_type, "vae")
    scheduler_btn_visible = get_initial_download_btn_visibility(qwen_image_scheduler_choices_init, model_path, initial_model_type, "scheduler")

    # Qwen-Image 模型配置
    with gr.Column(elem_classes=["diffusion-model-group"]) as qwen_components_group:
        # Diffusion 模型
        with gr.Row():
            with gr.Column(scale=5):
                qwen_image_dit_path_input = gr.Dropdown(
                    label=t("diffusion_model", lang),
                    choices=qwen_image_dit_choices_init,
                    value=qwen_image_dit_choices_init[0] if qwen_image_dit_choices_init else "",
                    allow_custom_value=True,
                )
            with gr.Column(scale=1, min_width=150):
                qwen_image_dit_download_btn = gr.Button(t("download", lang), visible=dit_btn_visible, size="sm", variant="secondary")
        qwen_image_dit_download_status = gr.Markdown("", visible=False)

        # VAE 和 Scheduler
        with gr.Row():
            with gr.Column(scale=1):
                qwen_image_vae_path_input = gr.Dropdown(
                    label=t("vae", lang),
                    choices=qwen_image_vae_choices_init,
                    value=qwen_image_vae_choices_init[0] if qwen_image_vae_choices_init else "",
                    allow_custom_value=True,
                )
                qwen_image_vae_download_btn = gr.Button(t("download", lang), visible=vae_btn_visible, size="sm", variant="secondary")
                qwen_image_vae_download_status = gr.Markdown("", visible=False)
            with gr.Column(scale=1):
                qwen_image_scheduler_path_input = gr.Dropdown(
                    label=t("scheduler", lang),
                    choices=qwen_image_scheduler_choices_init,
                    value=qwen_image_scheduler_choices_init[0] if qwen_image_scheduler_choices_init else "",
                    allow_custom_value=True,
                )
                qwen_image_scheduler_download_btn = gr.Button(t("download", lang), visible=scheduler_btn_visible, size="sm", variant="secondary")
                qwen_image_scheduler_download_status = gr.Markdown("", visible=False)

        # Qwen25-VL 编码器
        with gr.Row():
            with gr.Column(scale=1):
                qwen25vl_encoder_choices_init = get_qwen25vl_encoder_choices(model_path)
                qwen25vl_encoder_path_input = gr.Dropdown(
                    label=t("qwen25vl_encoder", lang),
                    choices=qwen25vl_encoder_choices_init,
                    value=qwen25vl_encoder_choices_init[0] if qwen25vl_encoder_choices_init else "",
                    allow_custom_value=True,
                )
                # 初始化时检查模型状态
                from utils.model_utils import check_model_exists, extract_model_name

                qwen25vl_btn_visible = False
                if qwen25vl_encoder_choices_init:
                    first_choice = qwen25vl_encoder_choices_init[0]
                    actual_name = extract_model_name(first_choice)
                    qwen25vl_exists = check_model_exists(model_path, actual_name)
                    qwen25vl_btn_visible = not qwen25vl_exists

                qwen25vl_encoder_download_btn = gr.Button(t("download", lang), visible=qwen25vl_btn_visible, size="sm", variant="secondary")
                qwen25vl_encoder_download_status = gr.Markdown("", visible=False)

        # LoRA 组件
        lora_choices_init = get_lora_choices(model_path)
        with gr.Row():
            with gr.Column(scale=1):
                use_lora = gr.Checkbox(
                    label=t("use_lora", lang),
                    value=False,
                )
                lora_path_input = gr.Dropdown(
                    label=t("lora", lang),
                    choices=lora_choices_init,
                    value=lora_choices_init[0] if lora_choices_init and lora_choices_init[0] else "",
                    allow_custom_value=True,
                    visible=False,
                    info=t("lora_info", lang),
                )
                lora_strength = gr.Slider(
                    label=t("lora_strength", lang),
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0,
                    visible=False,
                    info=t("lora_strength_info", lang),
                )

        # LoRA 开关变化时显示/隐藏相关组件
        def on_use_lora_change(use_lora_val):
            return (
                gr.update(visible=use_lora_val),  # lora_path_input
                gr.update(visible=use_lora_val),  # lora_strength
            )

        use_lora.change(
            fn=on_use_lora_change,
            inputs=[use_lora],
            outputs=[lora_path_input, lora_strength],
        )

        # 当 model_path 变化时更新 LoRA 选择
        def update_lora_choices(model_path_val):
            lora_choices = get_lora_choices(model_path_val)
            return gr.update(choices=lora_choices, value=lora_choices[0] if lora_choices and lora_choices[0] else "")

        model_path_input.change(
            fn=update_lora_choices,
            inputs=[model_path_input],
            outputs=[lora_path_input],
        )

    # 模型类型变化时更新选择
    def update_choices_on_model_type_change(model_type_val, model_path_val):
        if model_type_val == "Qwen-Image-2512":
            dit_choices = get_qwen_image_2512_dit_choices(model_path_val)
            vae_choices = get_qwen_image_2512_vae_choices(model_path_val)
            scheduler_choices = get_qwen_image_2512_scheduler_choices(model_path_val)
        else:
            dit_choices = get_qwen_image_dit_choices(model_path_val)
            vae_choices = get_qwen_image_vae_choices(model_path_val)
            scheduler_choices = get_qwen_image_scheduler_choices(model_path_val)
        return (
            gr.update(choices=dit_choices, value=dit_choices[0] if dit_choices else ""),
            gr.update(choices=vae_choices, value=vae_choices[0] if vae_choices else ""),
            gr.update(choices=scheduler_choices, value=scheduler_choices[0] if scheduler_choices else ""),
        )

    # 绑定下载按钮事件
    def download_qwen_image_dit(model_path_val, model_name, model_type_val, download_source_val, progress=gr.Progress()):
        # 从模型名称中提取实际名称
        from utils.model_utils import extract_model_name

        actual_name = extract_model_name(model_name)
        actual_name_lower = actual_name.lower()

        # 如果模型名称包含 "qwen_image_2512"，自动使用 qwen_image_2512_dit category
        # 否则根据 model_type_val 判断
        if "qwen_image_2512" in actual_name_lower:
            category = "qwen_image_2512_dit"
            get_choices = get_qwen_image_2512_dit_choices
            model_type_val = "Qwen-Image-2512"  # 确保使用正确的模型类型
        elif model_type_val == "Qwen-Image-2512":
            category = "qwen_image_2512_dit"
            get_choices = get_qwen_image_2512_dit_choices
        else:
            category = "qwen_image_dit"
            get_choices = get_qwen_image_dit_choices

        return download_model_handler(model_path_val, model_name, category, download_source_val, get_choices_func=get_choices, model_type_val=model_type_val, progress=progress)

    def download_qwen_image_vae(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        model_type_val = model_type_input.value if hasattr(model_type_input, "value") else "Qwen-Image-Edit-2511"
        category = "qwen_image_2512_vae" if model_type_val == "Qwen-Image-2512" else "qwen_image_vae"
        get_choices = get_qwen_image_2512_vae_choices if model_type_val == "Qwen-Image-2512" else get_qwen_image_vae_choices
        return download_model_handler(model_path_val, model_name, category, download_source_val, get_choices_func=get_choices, progress=progress)

    def download_qwen_image_scheduler(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        model_type_val = model_type_input.value if hasattr(model_type_input, "value") else "Qwen-Image-Edit-2511"
        category = "qwen_image_2512_scheduler" if model_type_val == "Qwen-Image-2512" else "qwen_image_scheduler"
        get_choices = get_qwen_image_2512_scheduler_choices if model_type_val == "Qwen-Image-2512" else get_qwen_image_scheduler_choices
        return download_model_handler(model_path_val, model_name, category, download_source_val, get_choices_func=get_choices, progress=progress)

    def download_qwen25vl_encoder(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "qwen25vl_encoder", download_source_val, get_choices_func=get_qwen25vl_encoder_choices, progress=progress)

    # 绑定状态更新和下载事件
    def get_model_category(model_type_val, base_category):
        if model_type_val == "Qwen-Image-2512":
            return f"qwen_image_2512_{base_category}"
        return f"qwen_image_{base_category}"

    def update_dit_status(model_path_val, model_name, model_type_val):
        category = get_model_category(model_type_val, "dit")
        return update_model_status(model_path_val, model_name, category)

    def update_vae_status(model_path_val, model_name, model_type_val):
        category = get_model_category(model_type_val, "vae")
        return update_model_status(model_path_val, model_name, category)

    def update_scheduler_status(model_path_val, model_name, model_type_val):
        category = get_model_category(model_type_val, "scheduler")
        return update_model_status(model_path_val, model_name, category)

    qwen_image_dit_path_input.change(
        fn=lambda mp, mn, mt: update_dit_status(mp, mn, mt),
        inputs=[model_path_input, qwen_image_dit_path_input, model_type_input],
        outputs=[qwen_image_dit_download_btn],
    )

    qwen_image_vae_path_input.change(
        fn=lambda mp, mn, mt: update_vae_status(mp, mn, mt),
        inputs=[model_path_input, qwen_image_vae_path_input, model_type_input],
        outputs=[qwen_image_vae_download_btn],
    )

    qwen_image_scheduler_path_input.change(
        fn=lambda mp, mn, mt: update_scheduler_status(mp, mn, mt),
        inputs=[model_path_input, qwen_image_scheduler_path_input, model_type_input],
        outputs=[qwen_image_scheduler_download_btn],
    )

    qwen25vl_encoder_path_input.change(
        fn=lambda mp, mn: update_model_status(mp, mn, "qwen25vl_encoder"),
        inputs=[model_path_input, qwen25vl_encoder_path_input],
        outputs=[qwen25vl_encoder_download_btn],
    )

    qwen_image_dit_download_btn.click(
        fn=download_qwen_image_dit,
        inputs=[model_path_input, qwen_image_dit_path_input, model_type_input, download_source_input],
        outputs=[qwen_image_dit_download_status, qwen_image_dit_download_btn, qwen_image_dit_path_input],
    )

    qwen_image_vae_download_btn.click(
        fn=download_qwen_image_vae,
        inputs=[model_path_input, qwen_image_vae_path_input, download_source_input],
        outputs=[qwen_image_vae_download_status, qwen_image_vae_download_btn, qwen_image_vae_path_input],
    )

    qwen_image_scheduler_download_btn.click(
        fn=download_qwen_image_scheduler,
        inputs=[model_path_input, qwen_image_scheduler_path_input, download_source_input],
        outputs=[qwen_image_scheduler_download_status, qwen_image_scheduler_download_btn, qwen_image_scheduler_path_input],
    )

    qwen25vl_encoder_download_btn.click(
        fn=download_qwen25vl_encoder,
        inputs=[model_path_input, qwen25vl_encoder_path_input, download_source_input],
        outputs=[qwen25vl_encoder_download_status, qwen25vl_encoder_download_btn, qwen25vl_encoder_path_input],
    )

    return {
        "qwen_image_dit_path_input": qwen_image_dit_path_input,
        "qwen_image_dit_download_btn": qwen_image_dit_download_btn,
        "qwen_image_dit_download_status": qwen_image_dit_download_status,
        "qwen_image_vae_path_input": qwen_image_vae_path_input,
        "qwen_image_vae_download_btn": qwen_image_vae_download_btn,
        "qwen_image_vae_download_status": qwen_image_vae_download_status,
        "qwen_image_scheduler_path_input": qwen_image_scheduler_path_input,
        "qwen_image_scheduler_download_btn": qwen_image_scheduler_download_btn,
        "qwen_image_scheduler_download_status": qwen_image_scheduler_download_status,
        "qwen25vl_encoder_path_input": qwen25vl_encoder_path_input,
        "qwen25vl_encoder_download_btn": qwen25vl_encoder_download_btn,
        "qwen25vl_encoder_download_status": qwen25vl_encoder_download_status,
        "use_lora": use_lora,
        "lora_path_input": lora_path_input,
        "lora_strength": lora_strength,
        "components_group": qwen_components_group,
    }


def build_z_image_turbo_components(model_path, model_path_input, download_source_input, model_type_input, lang="zh"):
    """构建 Z-Image-Turbo 模型相关组件（用于图片页面）
    必须在 Gradio 的 with 块内调用

    Args:
        model_path: 模型路径
        model_path_input: 模型路径输入组件
        download_source_input: 下载源输入组件
        model_type_input: 模型类型输入组件（用于动态更新）
        lang: 语言代码，默认为 "zh"

    Returns:
        dict: 包含所有 Z-Image-Turbo 相关组件的字典
    """

    # 初始化选择
    z_image_turbo_dit_choices_init = get_z_image_turbo_dit_choices(model_path)
    z_image_turbo_vae_choices_init = get_z_image_turbo_vae_choices(model_path)
    z_image_turbo_scheduler_choices_init = get_z_image_turbo_scheduler_choices(model_path)
    qwen3_encoder_choices_init = get_qwen3_encoder_choices(model_path)

    # Z-Image-Turbo 模型配置
    with gr.Column(elem_classes=["diffusion-model-group"], visible=False) as z_image_turbo_components_group:
        # Diffusion 模型
        with gr.Row():
            with gr.Column(scale=5):
                z_image_turbo_dit_path_input = gr.Dropdown(
                    label=t("diffusion_model", lang),
                    choices=z_image_turbo_dit_choices_init,
                    value=z_image_turbo_dit_choices_init[0] if z_image_turbo_dit_choices_init else "",
                    allow_custom_value=True,
                )
            with gr.Column(scale=1, min_width=150):
                # 计算初始按钮可见性
                import os

                initial_dit_btn_visible = False
                if z_image_turbo_dit_choices_init:
                    initial_dit_value = z_image_turbo_dit_choices_init[0]
                    from utils.model_utils import check_model_exists, extract_model_name

                    actual_name = extract_model_name(initial_dit_value)
                    initial_dit_btn_visible = not check_model_exists(model_path, actual_name)

                z_image_turbo_dit_download_btn = gr.Button(t("download", lang), visible=initial_dit_btn_visible, size="sm", variant="secondary")
        z_image_turbo_dit_download_status = gr.Markdown("", visible=False)

        # VAE 和 Scheduler
        with gr.Row():
            with gr.Column(scale=1):
                z_image_turbo_vae_path_input = gr.Dropdown(
                    label=t("vae", lang),
                    choices=z_image_turbo_vae_choices_init,
                    value=z_image_turbo_vae_choices_init[0] if z_image_turbo_vae_choices_init else "",
                    allow_custom_value=True,
                )
                # 计算初始按钮可见性
                initial_vae_btn_visible = False
                if z_image_turbo_vae_choices_init:
                    initial_vae_value = z_image_turbo_vae_choices_init[0]
                    from utils.model_utils import check_model_exists, extract_model_name

                    actual_name = extract_model_name(initial_vae_value)
                    initial_vae_btn_visible = not check_model_exists(model_path, actual_name)

                z_image_turbo_vae_download_btn = gr.Button(t("download", lang), visible=initial_vae_btn_visible, size="sm", variant="secondary")
                z_image_turbo_vae_download_status = gr.Markdown("", visible=False)
            with gr.Column(scale=1):
                z_image_turbo_scheduler_path_input = gr.Dropdown(
                    label=t("scheduler", lang),
                    choices=z_image_turbo_scheduler_choices_init,
                    value=z_image_turbo_scheduler_choices_init[0] if z_image_turbo_scheduler_choices_init else "",
                    allow_custom_value=True,
                )
                # 计算初始按钮可见性
                initial_scheduler_btn_visible = False
                if z_image_turbo_scheduler_choices_init:
                    initial_scheduler_value = z_image_turbo_scheduler_choices_init[0]
                    from utils.model_utils import check_model_exists, extract_model_name

                    actual_name = extract_model_name(initial_scheduler_value)
                    initial_scheduler_btn_visible = not check_model_exists(model_path, actual_name)

                z_image_turbo_scheduler_download_btn = gr.Button(t("download", lang), visible=initial_scheduler_btn_visible, size="sm", variant="secondary")
                z_image_turbo_scheduler_download_status = gr.Markdown("", visible=False)

        # Qwen3 编码器
        with gr.Row():
            with gr.Column(scale=1):
                qwen3_encoder_path_input = gr.Dropdown(
                    label=t("qwen3_encoder", lang),
                    choices=qwen3_encoder_choices_init,
                    value=qwen3_encoder_choices_init[0] if qwen3_encoder_choices_init else "",
                    allow_custom_value=True,
                )
                # 计算初始按钮可见性：检查整个仓库目录是否存在
                import os

                initial_qwen3_btn_visible = False
                if qwen3_encoder_choices_init:
                    repo_id = "JunHowie/Qwen3-4B-GPTQ-Int4"
                    repo_name = repo_id.split("/")[-1]  # "Qwen3-4B-GPTQ-Int4"
                    repo_path = os.path.join(model_path, repo_name)
                    initial_qwen3_btn_visible = not (os.path.exists(repo_path) and os.path.isdir(repo_path))

                qwen3_encoder_download_btn = gr.Button(t("download", lang), visible=initial_qwen3_btn_visible, size="sm", variant="secondary")
                qwen3_encoder_download_status = gr.Markdown("", visible=False)

        # LoRA 组件
        lora_choices_init = get_lora_choices(model_path)
        with gr.Row():
            with gr.Column(scale=1):
                use_lora = gr.Checkbox(
                    label=t("use_lora", lang),
                    value=False,
                )
                lora_path_input = gr.Dropdown(
                    label=t("lora", lang),
                    choices=lora_choices_init,
                    value=lora_choices_init[0] if lora_choices_init and lora_choices_init[0] else "",
                    allow_custom_value=True,
                    visible=False,
                    info=t("lora_info", lang),
                )
                lora_strength = gr.Slider(
                    label=t("lora_strength", lang),
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0,
                    visible=False,
                    info=t("lora_strength_info", lang),
                )

    # 绑定下载按钮事件
    def download_z_image_turbo_dit(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "z_image_turbo_dit", download_source_val, get_choices_func=get_z_image_turbo_dit_choices, progress=progress)

    def download_z_image_turbo_vae(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "z_image_turbo_vae", download_source_val, get_choices_func=get_z_image_turbo_vae_choices, progress=progress)

    def download_z_image_turbo_scheduler(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "z_image_turbo_scheduler", download_source_val, get_choices_func=get_z_image_turbo_scheduler_choices, progress=progress)

    def download_qwen3_encoder(model_path_val, model_name, download_source_val, progress=gr.Progress()):
        return download_model_handler(model_path_val, model_name, "qwen3_encoder", download_source_val, get_choices_func=get_qwen3_encoder_choices, progress=progress)

    # 绑定状态更新和下载事件
    def update_dit_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "z_image_turbo_dit")

    def update_vae_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "z_image_turbo_vae")

    def update_scheduler_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "z_image_turbo_scheduler")

    def update_qwen3_encoder_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "qwen3_encoder")

    z_image_turbo_dit_path_input.change(
        fn=update_dit_status,
        inputs=[model_path_input, z_image_turbo_dit_path_input],
        outputs=[z_image_turbo_dit_download_btn],
    )

    z_image_turbo_vae_path_input.change(
        fn=update_vae_status,
        inputs=[model_path_input, z_image_turbo_vae_path_input],
        outputs=[z_image_turbo_vae_download_btn],
    )

    z_image_turbo_scheduler_path_input.change(
        fn=update_scheduler_status,
        inputs=[model_path_input, z_image_turbo_scheduler_path_input],
        outputs=[z_image_turbo_scheduler_download_btn],
    )

    # Qwen3 编码器状态更新：检查整个仓库目录是否存在
    def update_qwen3_encoder_status(model_path_val, model_name):
        return update_model_status(model_path_val, model_name, "qwen3_encoder")

    qwen3_encoder_path_input.change(
        fn=update_qwen3_encoder_status,
        inputs=[model_path_input, qwen3_encoder_path_input],
        outputs=[qwen3_encoder_download_btn],
    )

    z_image_turbo_dit_download_btn.click(
        fn=download_z_image_turbo_dit,
        inputs=[model_path_input, z_image_turbo_dit_path_input, download_source_input],
        outputs=[z_image_turbo_dit_download_status, z_image_turbo_dit_download_btn, z_image_turbo_dit_path_input],
    )

    z_image_turbo_vae_download_btn.click(
        fn=download_z_image_turbo_vae,
        inputs=[model_path_input, z_image_turbo_vae_path_input, download_source_input],
        outputs=[z_image_turbo_vae_download_status, z_image_turbo_vae_download_btn, z_image_turbo_vae_path_input],
    )

    z_image_turbo_scheduler_download_btn.click(
        fn=download_z_image_turbo_scheduler,
        inputs=[model_path_input, z_image_turbo_scheduler_path_input, download_source_input],
        outputs=[z_image_turbo_scheduler_download_status, z_image_turbo_scheduler_download_btn, z_image_turbo_scheduler_path_input],
    )

    qwen3_encoder_download_btn.click(
        fn=download_qwen3_encoder,
        inputs=[model_path_input, qwen3_encoder_path_input, download_source_input],
        outputs=[qwen3_encoder_download_status, qwen3_encoder_download_btn, qwen3_encoder_path_input],
    )

    # LoRA 开关变化时显示/隐藏相关组件
    def on_use_lora_change(use_lora_val):
        return (
            gr.update(visible=use_lora_val),  # lora_path_input
            gr.update(visible=use_lora_val),  # lora_strength
        )

    use_lora.change(
        fn=on_use_lora_change,
        inputs=[use_lora],
        outputs=[lora_path_input, lora_strength],
    )

    # 当 model_path 变化时更新 LoRA 选择
    def update_lora_choices(model_path_val):
        lora_choices = get_lora_choices(model_path_val)
        return gr.update(choices=lora_choices, value=lora_choices[0] if lora_choices and lora_choices[0] else "")

    model_path_input.change(
        fn=update_lora_choices,
        inputs=[model_path_input],
        outputs=[lora_path_input],
    )

    return {
        "z_image_turbo_dit_path_input": z_image_turbo_dit_path_input,
        "z_image_turbo_dit_download_btn": z_image_turbo_dit_download_btn,
        "z_image_turbo_dit_download_status": z_image_turbo_dit_download_status,
        "z_image_turbo_vae_path_input": z_image_turbo_vae_path_input,
        "z_image_turbo_vae_download_btn": z_image_turbo_vae_download_btn,
        "z_image_turbo_vae_download_status": z_image_turbo_vae_download_status,
        "z_image_turbo_scheduler_path_input": z_image_turbo_scheduler_path_input,
        "z_image_turbo_scheduler_download_btn": z_image_turbo_scheduler_download_btn,
        "z_image_turbo_scheduler_download_status": z_image_turbo_scheduler_download_status,
        "qwen3_encoder_path_input": qwen3_encoder_path_input,
        "qwen3_encoder_download_btn": qwen3_encoder_download_btn,
        "qwen3_encoder_download_status": qwen3_encoder_download_status,
        "use_lora": use_lora,
        "lora_path_input": lora_path_input,
        "lora_strength": lora_strength,
        "components_group": z_image_turbo_components_group,
    }
