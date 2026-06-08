import argparse
import glob
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "data" / "PCB_DATASET"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "yolo_dataset"

CLASSES = ["mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"]
CLASS_TO_ID = {name.lower(): i for i, name in enumerate(CLASSES)}


def convert_annotation(xml_file: Path, txt_file: Path) -> None:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    with txt_file.open("w", encoding="utf-8") as out_file:
        for obj in root.iter("object"):
            cls_name = obj.find("name").text.lower()
            if cls_name not in CLASS_TO_ID:
                continue

            xml_box = obj.find("bndbox")
            x_min = int(xml_box.find("xmin").text)
            x_max = int(xml_box.find("xmax").text)
            y_min = int(xml_box.find("ymin").text)
            y_max = int(xml_box.find("ymax").text)

            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h

            out_file.write(f"{CLASS_TO_ID[cls_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def process_data(data_list: list[tuple[Path, Path]], split: str, output_dir: Path) -> None:
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for img_path, ann_path in data_list:
        file_name = img_path.name
        txt_name = img_path.with_suffix(".txt").name
        shutil.copy(img_path, image_dir / file_name)
        convert_annotation(ann_path, label_dir / txt_name)


def write_data_yaml(output_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASSES))
    content = f"""path: {output_dir.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n"""
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def convert_dataset(dataset_dir: Path, output_dir: Path, test_size: float, random_state: int) -> None:
    annotations_dir = dataset_dir / "Annotations"
    images_dir = dataset_dir / "images"

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Папка с XML-аннотациями не найдена: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Папка с изображениями не найдена: {images_dir}")

    all_xmls = [Path(p) for p in glob.glob(str(annotations_dir / "**" / "*.xml"), recursive=True)]
    all_data = []

    for ann_path in all_xmls:
        img_filename = ann_path.with_suffix(".jpg").name
        for cls in CLASSES:
            img_path = images_dir / cls / img_filename
            if img_path.exists():
                all_data.append((img_path, ann_path))
                break

    if not all_data:
        raise RuntimeError("Не найдено ни одной пары изображение-аннотация.")

    train_data, val_data = train_test_split(all_data, test_size=test_size, random_state=random_state)
    process_data(train_data, "train", output_dir)
    process_data(val_data, "val", output_dir)
    write_data_yaml(output_dir)

    print(f"Готово. Train: {len(train_data)}, val: {len(val_data)}")
    print(f"YOLO-датасет сохранён в: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Конвертация PCB_DATASET из XML-разметки в YOLO-формат.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR, help="Папка исходного PCB_DATASET.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Папка для YOLO-датасета.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Доля валидационной выборки.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed для разбиения train/val.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_dataset(args.dataset, args.output, args.test_size, args.random_state)
