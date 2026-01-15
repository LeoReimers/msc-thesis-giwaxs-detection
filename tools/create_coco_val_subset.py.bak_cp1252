# -*- coding: utf-8 -*-
import json
import os
import random

# Angepasste Pfade relativ zu DINO/tools/
ANNOTATIONS_PATH = "../data/coco/annotations/instances_val2017.json"
OUTPUT_PATH = "../data/coco/annotations/instances_val2017_subset.json"
NUM_IMAGES = 50

# Lade Originaldaten
with open(ANNOTATIONS_PATH, "r") as f:
    data = json.load(f)

# Wähle zufällig 50 Bild-IDs
image_ids = [img["id"] for img in data["images"]]
selected_ids = set(random.sample(image_ids, NUM_IMAGES))

# Filtere zugehörige Bilder und Annotationen
subset_images = [img for img in data["images"] if img["id"] in selected_ids]
subset_annotations = [ann for ann in data["annotations"] if ann["image_id"] in selected_ids]

# Kopiere Kategorien
subset_data = {
    "images": subset_images,
    "annotations": subset_annotations,
    "categories": data["categories"]
}

# Speichern
with open(OUTPUT_PATH, "w") as f:
    json.dump(subset_data, f)

print(f"? Subset gespeichert: {len(subset_images)} Bilder, {len(subset_annotations)} Annotationen ? {OUTPUT_PATH}")
