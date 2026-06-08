# PCB Quality Control System

Проект предназначен для автоматического контроля качества печатных плат по изображениям. В основе используется YOLOv8-модель, которая находит дефекты на плате, сохраняет изображения с разметкой и позволяет просматривать/уточнять найденные дефекты через настольное PyQt-приложение.

## Что находится в проекте

`app.py` — основное desktop-приложение. Через интерфейс выбирается папка с изображениями, запускается анализ, просматриваются найденные дефекты, вручную добавляются или удаляются боксы, затем сохраняется обновлённая база изображений и аннотаций.

`batch_predict.py` — консольный запуск модели на папке изображений. Нужен, если требуется быстро обработать набор файлов без графического интерфейса.

`convert_dataset.py` — конвертация исходного датасета `PCB_DATASET` из XML-разметки в YOLO-формат. На выходе создаются папки `images/train`, `images/val`, `labels/train`, `labels/val` и файл `data.yaml`.

`train_model.py` — обучение YOLOv8-модели на подготовленном датасете. По умолчанию ожидает `data/yolo_dataset/data.yaml` и сохраняет результат в `models/pcb_defect_yolov8_40epochs`.

`models/pcb_defect_yolov8_40epochs/` — сохранённый результат обучения: веса модели, графики, confusion matrix, метрики и служебный `args.yaml`.

`data/` — рабочая папка для исходных данных. В неё можно положить исходный `PCB_DATASET`, тестовые изображения и YOLO-датасет после конвертации.

`outputs/` — папка для результатов запуска приложения и пакетного предсказания.

## Логика работы

Сначала исходный датасет с XML-аннотациями приводится к YOLO-формату через `convert_dataset.py`. Затем модель обучается через `train_model.py`. После обучения веса `best.pt` используются в `app.py` или `batch_predict.py` для поиска дефектов на новых изображениях.

Классы дефектов:

1. `mouse_bite`
2. `spur`
3. `missing_hole`
4. `short`
5. `open_circuit`
6. `spurious_copper`

Все пути в коде сделаны относительными от корня проекта, поэтому проект можно переносить между компьютерами без ручной замены локальных путей конкретного компьютера.

## Установка

Рекомендуется использовать виртуальное окружение.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

## Как воспроизвести полный пайплайн

### 1. Подготовить исходный датасет

Положить исходный датасет в папку:

```text
data/PCB_DATASET/
```

Ожидаемая структура:

```text
data/PCB_DATASET/
├── Annotations/
└── images/
    ├── mouse_bite/
    ├── spur/
    ├── missing_hole/
    ├── short/
    ├── open_circuit/
    └── spurious_copper/
```

### 2. Сконвертировать датасет в YOLO-формат

```bash
python convert_dataset.py
```

Результат появится в:

```text
data/yolo_dataset/
```

Можно указать свои папки:

```bash
python convert_dataset.py --dataset data/PCB_DATASET --output data/yolo_dataset
```

### 3. Обучить модель

```bash
python train_model.py
```

По умолчанию используется:

```text
data/yolo_dataset/data.yaml
```

Результат обучения сохраняется в:

```text
models/pcb_defect_yolov8_40epochs/
```

Для изменения параметров:

```bash
python train_model.py --data data/yolo_dataset/data.yaml --epochs 40 --batch 16 --imgsz 640
```

### 4. Запустить пакетное предсказание без интерфейса

Положить изображения в:

```text
data/test_images/
```

Запустить:

```bash
python batch_predict.py
```

Результаты будут сохранены в:

```text
outputs/batch_predictions/
```

Можно указать свои пути:

```bash
python batch_predict.py --input data/test_images --output outputs/batch_predictions --model models/pcb_defect_yolov8_40epochs/weights/best.pt
```

### 5. Запустить графическое приложение

```bash
python app.py
```

В приложении нужно выбрать папку с изображениями, нажать «Начать анализ», затем при необходимости открыть изображение двойным кликом и вручную исправить список дефектов.

## Основные файлы модели

`models/pcb_defect_yolov8_40epochs/weights/best.pt` — лучшие веса модели для запуска предсказаний.

`models/pcb_defect_yolov8_40epochs/weights/last.pt` — веса с последней эпохи обучения.

`models/pcb_defect_yolov8_40epochs/results.csv` — численные метрики обучения по эпохам.

`models/pcb_defect_yolov8_40epochs/results.png` — графики обучения.

`models/pcb_defect_yolov8_40epochs/confusion_matrix.png` — матрица ошибок.

## Примечания

Если CUDA недоступна, обучение и предсказание будут выполняться на CPU. Для обучения это будет существенно медленнее. Для запуска уже обученной модели через `app.py` или `batch_predict.py` GPU не обязателен, но ускоряет обработку изображений.
