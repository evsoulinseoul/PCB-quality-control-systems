import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = BASE_DIR / "data" / "yolo_dataset" / "data.yaml"
DEFAULT_PROJECT_DIR = BASE_DIR / "models"
DEFAULT_RUN_NAME = "pcb_defect_yolov8_40epochs"


def train_model(data_yaml: Path, project_dir: Path, run_name: str, epochs: int, batch: int, imgsz: int) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Файл data.yaml не найден: {data_yaml}")

    model = YOLO("yolov8l.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=10,
        perspective=0.0005,
        mosaic=1.0,
        mixup=0.2,
        project=str(project_dir),
        name=run_name,
        augment=True,
        hsv_h=0.1,
        hsv_s=0.5,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        device=device,
    )

    metrics = model.val()
    print("Метрики:", metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение YOLOv8 для обнаружения дефектов печатных плат.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Путь к data.yaml YOLO-датасета.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_DIR, help="Папка для сохранения результата обучения.")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME, help="Имя запуска обучения.")
    parser.add_argument("--epochs", type=int, default=40, help="Количество эпох обучения.")
    parser.add_argument("--batch", type=int, default=16, help="Размер batch.")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер входного изображения.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.data, args.project, args.name, args.epochs, args.batch, args.imgsz)
