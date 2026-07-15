from pathlib import Path

import pandas as pd
import shutil

DATA_PATHS = ["data/raw/egyptian_hieroglyphics_dataset/_annotations.csv",
              "data/raw/egyptian_hieroglyphics_dataset/_annotations 2.csv",
              "data/raw/egyptian_hieroglyphics_dataset/_annotations 3.csv"]



for path in DATA_PATHS:
    csv = pd.read_csv(path)

    for _, row in csv.iterrows():
        image = row['filename']
        class_name = row['class']

        source = Path("data/raw/egyptian_hieroglyphics_dataset") / image
        destination = Path("data/raw_by_class/") / class_name

        destination.mkdir(parents = True, exist_ok = True)

        shutil.copy(source, destination/image)



