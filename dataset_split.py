import os
import random
import shutil
import logging
from log_file import setup_logging
logger = setup_logging("dataset_split")
class DatasetSplitter:
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        source_dir: str,
        dest_dir: str,
        train_count: int,
        val_count: int,
        test_count: int,
        logger: logging.Logger
    ):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.train_count = train_count
        self.val_count = val_count
        self.test_count = test_count
        self.logger = logger.getChild("dataset_splitter")

        random.seed(42)

        self.split_classes = {
            "train": set(),
            "val": set(),
            "test": set()
        }

    def create_output_dirs(self):
        for split in ["train", "val", "test"]:
            path = os.path.join(self.dest_dir, split)
            os.makedirs(path, exist_ok=True)
            self.logger.info(f"Created directory: {path}")

    def get_images(self, class_path):
        return [
            img for img in os.listdir(class_path)
            if img.lower().endswith(self.IMAGE_EXTENSIONS)
        ]

    def split_class(self, food_class):
        class_path = os.path.join(self.source_dir, food_class)

        if not os.path.isdir(class_path):
            return

        images = self.get_images(class_path)
        required = self.train_count + self.val_count + self.test_count

        if len(images) < required:
            raise ValueError(
                f"Not enough images in '{food_class}' "
                f"(found {len(images)}, required {required})"
            )

        random.shuffle(images)

        splits = {
            "train": images[:self.train_count],
            "val": images[self.train_count:self.train_count + self.val_count],
            "test": images[self.train_count + self.val_count:required]
        }

        for split, split_imgs in splits.items():
            split_class_dir = os.path.join(self.dest_dir, split, food_class)
            os.makedirs(split_class_dir, exist_ok=True)

            for img in split_imgs:
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(split_class_dir, img)
                )

            if split_imgs:
                self.split_classes[split].add(food_class)

        self.logger.info(
            f"{food_class} → "
            f"Train={len(splits['train'])}, "
            f"Val={len(splits['val'])}, "
            f"Test={len(splits['test'])}"
        )

    def run(self):
        try:
            self.logger.info("Dataset splitting started")
            self.create_output_dirs()

            for food_class in os.listdir(self.source_dir):
                self.split_class(food_class)

            self.logger.info("Dataset split completed successfully")

            for split, classes in self.split_classes.items():
                self.logger.info(
                    f"Total classes in {split}: {len(classes)}"
                )

        except Exception:
            self.logger.exception("Dataset splitting failed")
            raise
if __name__ == "__main__":
    splitter = DatasetSplitter(
        source_dir="Food Classification dataset",
        dest_dir="data_split",
        train_count=200,
        val_count=50,
        test_count=10,
        logger=logger
    )
    splitter.run()
