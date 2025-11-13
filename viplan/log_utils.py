import os
import shutil


def get_img_output_dir(work_mode, instance_id, scene_id, task) -> str:
    img_output_dir = os.path.join(f'img/{work_mode}', f'{task}_{scene_id}_{instance_id}')
    if os.path.exists(img_output_dir):
        shutil.rmtree(img_output_dir)
    os.makedirs(img_output_dir, exist_ok=True)
    return img_output_dir
