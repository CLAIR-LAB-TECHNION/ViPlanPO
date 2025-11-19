import os
import re
import shutil


def get_img_output_dir(work_mode, instance_id, scene_id, task) -> str:
    img_output_dir = os.path.join(f'img/{work_mode}', f'{task}_{scene_id}_{instance_id}')
    if os.path.exists(img_output_dir):
        shutil.rmtree(img_output_dir)
    os.makedirs(img_output_dir, exist_ok=True)
    return img_output_dir


def sanitize_filename_component(text):
    sanitized = re.sub(r'[^a-zA-Z0-9_-]+', '_', text.lower())
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized or 'question'


def save_vlm_question_images(questions, image, img_log_info, check_type, logger):
    if not img_log_info:
        raise KeyError('Missing img_log_info')
        return

    output_dir = img_log_info.get('img_output_dir')
    if not output_dir:
        raise KeyError('Missing img_log_info.output_dir')
        return

    problem_name = os.path.splitext(os.path.basename(img_log_info.get('problem_file', 'problem')))[0]
    scene_id = img_log_info.get('scene_id', 'scene')
    instance_id = img_log_info.get('instance_id', 'instance')
    counter = img_log_info.get('image_counter', 0)

    for question_text, _ in questions.values():
        counter += 1
        safe_question = sanitize_filename_component(question_text)
        filename = f"{problem_name}_{scene_id}_{instance_id}_{check_type or 'query'}_{counter:04d}_{safe_question}.png"
        filepath = os.path.join(output_dir, filename)
        try:
            image.save(filepath)
        except Exception as exc:
            logger.warning(f"Failed to save VLM image to {filepath}: {exc}")

    img_log_info['image_counter'] = counter
