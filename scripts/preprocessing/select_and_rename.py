from pathlib import Path
import csv
import random
import shutil

SOURCE_DIR = Path("data/original/unas_dataset")
OUTPUT_DIR = Path("data/processed")
CSV_DIR = Path("data/metadata/renaming_log.csv")

SELECTED_CLASSES = [
    "N35",
    "M17",
    "S29",
    "X1",
    "G43",
    "G17",
    "D21",
    "I9",
    "V31",
    "E34",
]

NUM_IMAGES = 120
random.seed(21)

with open(CSV_DIR, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "class",
        "original_filename",
        "new_filename"
    ])


    for class_name in SELECTED_CLASSES:
        source_class = SOURCE_DIR / class_name
        output_class = OUTPUT_DIR / class_name

        output_class.mkdir(parents=True, exist_ok=True)

        images = list(source_class.glob("*.png"))
        selected_images = random.sample(images, NUM_IMAGES)

        #Renaming the image files to follow the standard Gardiner convention
        for number, image_path in enumerate(selected_images, start=1):
            new_name = f"{class_name}_{number:03d}.png"
            new_path = output_class / new_name

            shutil.copy(image_path, new_path)

            writer.writerow([
                    class_name,
                    image_path.name,
                    new_name
                ])


print("Images selected and renaming. Changes logged")