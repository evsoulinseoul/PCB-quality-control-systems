import argparse
import time
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "pcb_defect_yolov8_40epochs" / "weights" / "best.pt"
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "test_images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "batch_predictions"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def detect_defects_in_folder(input_dir: Path, output_dir: Path, model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Папка с изображениями не найдена: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not image_files:
        print(f"В папке нет изображений: {input_dir}")
        return

    model = YOLO(str(model_path))
    start_time = time.time()

    for image_path in image_files:
        results = model(str(image_path))
        for result in results:
            result.save(filename=str(output_dir / image_path.name))
        print(f"Обработано: {image_path.name}")

    total_time = time.time() - start_time
    print(f"Все изображения обработаны за {total_time:.2f} секунд.")
    print(f"Результаты сохранены в: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пакетное обнаружение дефектов на изображениях печатных плат.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help="Папка с исходными изображениями.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Папка для сохранения размеченных изображений.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Путь к весам YOLO-модели.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_defects_in_folder(args.input, args.output, args.model)
