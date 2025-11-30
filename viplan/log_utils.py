import json
import logging
import os
import re
import shutil
from logging.handlers import RotatingFileHandler


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


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record):
        json_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            # "name": record.name,
            # "message": record.getMessage(), # Already in msg
            # "filename": record.filename,
            "lineno": record.lineno,
            # "process": record.process,
            # "threadName": record.threadName
        }

        for key, value in record.__dict__.items():
            if value is None:
                continue
            if key.startswith('_'):
                continue
            if key in ['name', 'levelname', 'pathname', 'lineno',
                       'funcName', 'created', 'asctime', 'msecs',
                       'relativeCreated', 'thread', 'threadName',
                       'processName', 'process', 'message', 'module',
                       'exc_info', 'exc_text', 'stack_info', 'extra']:
                continue
            json_record[key] = value
        # Add extra fields passed via the 'extra' parameter
        # json_record.update(record.__dict__.get('extra', {}

        if record.exc_info:
            json_record['exc_info'] = self.formatException(record.exc_info)

        return json.dumps(json_record)


def get_task_logger(out_dir : os.PathLike, unique_id: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Log file configuration
    log_file_path = os.path.join(out_dir, f'execution_{unique_id}.log')

    max_bytes = 1024 * 1024  # 1 MB in bytes
    backup_count = 5  # Keep 5 backup files (app_size_rotated.log.1, .2, etc.)

    handler = RotatingFileHandler(
        filename=log_file_path,
        mode='a',
        maxBytes=max_bytes,
        backupCount=backup_count
    )

    # Optional: Customize the suffix for rotated files to include the date clearly
    handler.suffix = "%Y-%m-%d"

    # 3. Apply the JsonFormatter to the handler
    # The datefmt is required by the JsonFormatter's formatTime method
    formatter = JsonFormatter(datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    # 4. Add the handler to the logger
    logger.addHandler(handler)

    # Optional: Add a StreamHandler for console output as well
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # console_handler.setFormatter(console_formatter)
    # logger.addHandler(console_handler)

    return logger
