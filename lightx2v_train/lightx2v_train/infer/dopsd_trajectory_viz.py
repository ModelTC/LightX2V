from PIL import Image, ImageDraw, ImageFont


def _fit_height(image, height):
    if image.height == height:
        return image
    width = max(1, int(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def save_student_teacher_trajectory_grid(student_step_images, teacher_step_images, save_path):
    if len(student_step_images) != len(teacher_step_images):
        raise ValueError("student and teacher trajectory lengths must match")

    pad = 12
    header_h = 32
    row_label_w = 56
    font = ImageFont.load_default()
    num_steps = len(student_step_images)
    if num_steps == 0:
        return

    row_h = max(img.height for img in student_step_images + teacher_step_images)
    student_cols = [_fit_height(img.convert("RGB"), row_h) for img in student_step_images]
    teacher_cols = [_fit_height(img.convert("RGB"), row_h) for img in teacher_step_images]

    panel_w = student_cols[0].width + pad + teacher_cols[0].width
    canvas_h = header_h + num_steps * (row_h + pad) + pad
    canvas_w = row_label_w + pad + panel_w + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_y = 8
    draw.text((row_label_w + pad + 8, header_y), "Student", fill=(0, 0, 0), font=font)
    draw.text((row_label_w + pad + student_cols[0].width + pad + 8, header_y), "Teacher", fill=(0, 0, 0), font=font)

    y = header_h
    for step_idx, (student_img, teacher_img) in enumerate(zip(student_cols, teacher_cols)):
        draw.text((8, y + (row_h - 10) // 2), f"t{step_idx}", fill=(0, 0, 0), font=font)
        x_student = row_label_w + pad
        canvas.paste(student_img, (x_student, y))
        x_teacher = x_student + student_img.width + pad
        canvas.paste(teacher_img, (x_teacher, y))
        y += row_h + pad

    save_path = str(save_path)
    canvas.save(save_path)
