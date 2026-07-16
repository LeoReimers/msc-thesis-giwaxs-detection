# -*- coding: utf-8 -*-
import json
import shutil
import os

# Parameter
NUM_IMAGES = 100

# Pfade
ann_file = 'data/coco/annotations/instances_train2017.json'
source_image_dir = 'data/coco/train2017_full'
output_dir = 'data/coco/train2017'

# Zielordner erstellen (falls nicht vorhanden)
os.makedirs(output_dir, exist_ok=True)

# Lade die JSON-Datei
with open(ann_file, 'r') as f:
    data = json.load(f)

# Wähle die ersten N Bilder aus
selected_images = data['images'][:NUM_IMAGES]
selected_ids = {img['id'] for img in selected_images}

# Filtere Annotationen für die ausgewählten Bilder
filtered_annotations = [ann for ann in data['annotations'] if ann['image_id'] in selected_ids]

# Baue neue JSON-Struktur
new_data = {
    'info': data.get('info', {}),
    'licenses': data.get('licenses', []),
    'images': selected_images,
    'annotations': filtered_annotations,
    'categories': data.get('categories', [])
}

# Speichere die reduzierte JSON-Datei
subset_ann_file = 'data/coco/annotations/instances_train2017_subset.json'
with open(subset_ann_file, 'w') as f:
    json.dump(new_data, f)

# Kopiere die zugehörigen Bilder
print(f'?? Kopiere {len(selected_images)} Bilder nach {output_dir}')
copied = 0
for img in selected_images:
    fname = img['file_name']
    src = os.path.join(source_image_dir, fname)
    dst = os.path.join(output_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1
    else:
        print(f'?? Bild nicht gefunden: {fname}')

print(f'? Fertig: {copied} Bilder erfolgreich kopiert.')
