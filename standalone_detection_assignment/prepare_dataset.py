from __future__ import annotations

import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlretrieve

TARGET_CLASS_NAMES = ["person", "bottle", "cell phone"]
TARGET_IMAGES_PER_CLASS = 30
MIN_SINGLE_IMAGES_PER_CLASS = 5
TRAIN_RATIO = 0.8
SEED = 42


def ensure_annotations(base_dir: Path) -> Path:
    downloads_dir = base_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    zip_path = downloads_dir / "annotations_trainval2017.zip"
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    if not zip_path.exists():
        print(f"Downloading annotations to {zip_path} ...")
        urlretrieve(url, zip_path)

    extracted_json = downloads_dir / "annotations" / "annotations" / "instances_val2017.json"
    if not extracted_json.exists():
        print("Extracting instances_val2017.json ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("annotations/instances_val2017.json", downloads_dir / "annotations")

    return extracted_json


def load_coco_annotations(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_image_ids(data: dict) -> tuple[list[int], dict[int, list[dict]], dict[int, dict], dict[int, str]]:
    name_to_id = {category["name"]: category["id"] for category in data["categories"]}
    id_to_name = {value: key for key, value in name_to_id.items()}
    target_ids = [name_to_id[name] for name in TARGET_CLASS_NAMES]

    images = {image["id"]: image for image in data["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        if annotation["category_id"] in target_ids:
            annotations_by_image[annotation["image_id"]].append(annotation)

    single_candidates = {class_id: [] for class_id in target_ids}
    multi_candidates: list[int] = []

    for image_id, annotations in annotations_by_image.items():
        image_classes = {annotation["category_id"] for annotation in annotations}
        if len(image_classes) == 1:
            class_id = next(iter(image_classes))
            single_candidates[class_id].append(image_id)
        else:
            multi_candidates.append(image_id)

    random.seed(SEED)
    for class_id in target_ids:
        random.shuffle(single_candidates[class_id])
    random.shuffle(multi_candidates)

    selected_image_ids: list[int] = []
    selected_set: set[int] = set()
    class_presence = Counter()

    for class_id in target_ids:
        picked = 0
        for image_id in single_candidates[class_id]:
            if image_id in selected_set:
                continue
            selected_image_ids.append(image_id)
            selected_set.add(image_id)
            class_presence[class_id] += 1
            picked += 1
            if picked >= MIN_SINGLE_IMAGES_PER_CLASS:
                break

    all_candidates = multi_candidates + [
        image_id for class_id in target_ids for image_id in single_candidates[class_id]
    ]

    def image_classes(image_id: int) -> set[int]:
        return {annotation["category_id"] for annotation in annotations_by_image[image_id]}

    while not all(class_presence[class_id] >= TARGET_IMAGES_PER_CLASS for class_id in target_ids):
        best_image_id = None
        best_score = None

        for image_id in all_candidates:
            if image_id in selected_set:
                continue

            classes = image_classes(image_id)
            gain = sum(class_presence[class_id] < TARGET_IMAGES_PER_CLASS for class_id in classes)
            if gain == 0:
                continue

            underfill = sum(max(0, TARGET_IMAGES_PER_CLASS - class_presence[class_id]) for class_id in classes)
            score = (gain, len(classes) > 1, underfill, len(annotations_by_image[image_id]))
            if best_score is None or score > best_score:
                best_score = score
                best_image_id = image_id

        if best_image_id is None:
            break

        selected_image_ids.append(best_image_id)
        selected_set.add(best_image_id)
        for class_id in image_classes(best_image_id):
            if class_presence[class_id] < TARGET_IMAGES_PER_CLASS:
                class_presence[class_id] += 1

    print("Selected images:", len(selected_image_ids))
    for class_id in target_ids:
        print(f"  {id_to_name[class_id]}: {class_presence[class_id]} images")

    single_count = 0
    multi_count = 0
    for image_id in selected_image_ids:
        if len(image_classes(image_id)) == 1:
            single_count += 1
        else:
            multi_count += 1
    print(f"Single-object images: {single_count}")
    print(f"Multi-object images: {multi_count}")

    remap = {class_id: index for index, class_id in enumerate(target_ids)}
    return selected_image_ids, annotations_by_image, images, {old_id: TARGET_CLASS_NAMES[new_id] for old_id, new_id in remap.items()}


def split_dataset(
    selected_image_ids: list[int], annotations_by_image: dict[int, list[dict]], target_name_map: dict[int, str]
) -> tuple[list[int], list[int]]:
    random.seed(SEED)

    def image_classes(image_id: int) -> set[int]:
        return {annotation["category_id"] for annotation in annotations_by_image[image_id]}

    remaining_ids: list[int] = []
    val_ids: list[int] = []
    class_val_counts = Counter()

    shuffled = selected_image_ids[:]
    random.shuffle(shuffled)
    target_val_size = max(1, round(len(selected_image_ids) * (1 - TRAIN_RATIO)))

    for image_id in shuffled:
        classes = image_classes(image_id)
        should_add_to_val = len(val_ids) < target_val_size and any(class_val_counts[class_id] < 3 for class_id in classes)
        if should_add_to_val:
            val_ids.append(image_id)
            for class_id in classes:
                class_val_counts[class_id] += 1
        else:
            remaining_ids.append(image_id)

    for image_id in remaining_ids:
        if len(val_ids) < target_val_size:
            val_ids.append(image_id)
        else:
            break

    train_ids = [image_id for image_id in selected_image_ids if image_id not in set(val_ids)]

    print("Train images:", len(train_ids))
    print("Val images:", len(val_ids))
    for class_id, class_name in target_name_map.items():
        print(f"  Val coverage for {class_name}: {class_val_counts[class_id]} images")

    return train_ids, val_ids


def write_yolo_dataset(
    base_dir: Path,
    selected_image_ids: list[int],
    train_ids: list[int],
    val_ids: list[int],
    annotations_by_image: dict[int, list[dict]],
    images: dict[int, dict],
) -> Path:
    dataset_root = base_dir / "dataset"
    image_dirs = {
        "train": dataset_root / "images" / "train",
        "val": dataset_root / "images" / "val",
    }
    label_dirs = {
        "train": dataset_root / "labels" / "train",
        "val": dataset_root / "labels" / "val",
    }

    for path in [*image_dirs.values(), *label_dirs.values()]:
        path.mkdir(parents=True, exist_ok=True)

    downloads_dir = base_dir / "downloads" / "val2017"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    name_to_id = {name: index for index, name in enumerate(TARGET_CLASS_NAMES)}
    coco_name_to_id = {1: 0, 44: 1, 77: 2}

    split_lookup = {image_id: "train" for image_id in train_ids}
    split_lookup.update({image_id: "val" for image_id in val_ids})

    for image_id in selected_image_ids:
        image_info = images[image_id]
        file_name = image_info["file_name"]
        local_image_path = downloads_dir / file_name
        if not local_image_path.exists():
            image_url = f"http://images.cocodataset.org/val2017/{file_name}"
            print("Downloading image:", file_name)
            urlretrieve(image_url, local_image_path)

        split_name = split_lookup[image_id]
        target_image_path = image_dirs[split_name] / file_name
        if not target_image_path.exists():
            target_image_path.write_bytes(local_image_path.read_bytes())

        label_path = label_dirs[split_name] / f"{Path(file_name).stem}.txt"
        width = float(image_info["width"])
        height = float(image_info["height"])

        lines: list[str] = []
        for annotation in annotations_by_image[image_id]:
            x_min, y_min, box_width, box_height = annotation["bbox"]
            x_center = (x_min + box_width / 2) / width
            y_center = (y_min + box_height / 2) / height
            norm_width = box_width / width
            norm_height = box_height / height
            class_index = coco_name_to_id[annotation["category_id"]]
            lines.append(
                f"{class_index} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            )

        label_path.write_text("\n".join(lines), encoding="utf-8")

    yaml_path = dataset_root / "coco3_subset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
                "  1: bottle",
                "  2: cell phone",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    annotation_path = ensure_annotations(base_dir)
    data = load_coco_annotations(annotation_path)
    selected_image_ids, annotations_by_image, images, target_name_map = select_image_ids(data)
    train_ids, val_ids = split_dataset(selected_image_ids, annotations_by_image, target_name_map)
    yaml_path = write_yolo_dataset(base_dir, selected_image_ids, train_ids, val_ids, annotations_by_image, images)
    print("Dataset YAML:", yaml_path)


if __name__ == "__main__":
    main()
