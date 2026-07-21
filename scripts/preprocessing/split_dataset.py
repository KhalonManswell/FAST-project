from pathlib import Path
from sklearn.model_selection import train_test_split 
import shutil


SOURCE_DIR = Path("data/processed/selected")
OUTPUT_DIR = Path("data/processed")

train_ratio = 0.70
val_ratio = 0.15

for folder in SOURCE_DIR.iterdir():
    if not folder.is_dir():
        continue

    images = list(folder.glob("*.png"))

    train_images, temp = train_test_split(images, test_size=0.3, random_state=21)
    test_images, val_images = train_test_split(temp, test_size=0.5, random_state=21)

    splits = {
        "train": train_images,
        "test": test_images,
        "val": val_images
    }

    for split, split_images in splits.items():
        destination = OUTPUT_DIR / split / folder.name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy(image_path, destination / image_path.name)



