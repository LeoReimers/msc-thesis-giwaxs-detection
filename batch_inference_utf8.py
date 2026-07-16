# -*- coding: utf-8 -*-

import torch
import argparse
from PIL import Image, ImageDraw, ImageFont
import os

from models.dino.dino import build_dino, PostProcess
from util.slconfig import SLConfig
from util import box_ops
from util.coco_classes import coco_classes
import torchvision.transforms as T

def visualize_pil(image, outputs, save_path, score_threshold=0.3):
    draw = ImageDraw.Draw(image)
    boxes = outputs['boxes']
    scores = outputs['scores']
    labels = outputs['labels']

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=50)
    except:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        box = box.tolist()
        label_id = label.item()
        label_name = coco_classes[label_id] if label_id < len(coco_classes) else str(label_id)
        text = f"{label_name} ({score:.2f})"

        # TextgrÃ¶Ãe ermitteln
        if font is not None:
            try:
                text_bbox = font.getbbox(text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_width, text_height = font.getsize(text)

            text_bg = [box[0], box[1] - text_height, box[0] + text_width, box[1]]
            draw.rectangle(text_bg, fill="red")

        draw.rectangle(box, outline="red", width=4)
        draw.text((box[0], box[1] - text_height), text, fill="white", font=font)

    image.save(save_path)

def prepare_image(image_path):
    transform = T.Compose([
        T.Resize((800, 1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0).cuda()

def load_model(checkpoint_path):
    config_path = os.path.join(os.path.dirname(checkpoint_path), 'config_args_all.json')
    args = SLConfig.fromfile(config_path)
    args.device = 'cuda'
    model, _, _ = build_dino(args)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    model.eval().cuda()
    return model

def main(args):
    model = load_model(args.resume)
    postprocessor = PostProcess()

    image_files = sorted([
        f for f in os.listdir(args.img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"INFO: {len(image_files)} Bilder gefunden in: {args.img_dir}")

    for idx, fname in enumerate(image_files):
        image_path = os.path.join(args.img_dir, fname)
        save_path = os.path.join(args.out_dir, fname)

        try:
            image, img_tensor = prepare_image(image_path)

            with torch.no_grad():
                outputs = model(img_tensor)

            results = postprocessor(outputs, torch.tensor([[img_tensor.shape[2], img_tensor.shape[3]]]).cuda())
            visualize_pil(image, results[0], save_path)
            print(f"[{idx+1}/{len(image_files)}] â Gespeichert: {save_path}")

        except Exception as e:
            print(f"[{idx+1}/{len(image_files)}] FEHLER bei Bild {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Ordner mit Eingabebildern")
    parser.add_argument("--resume", type=str, required=True, help="Pfad zum Checkpoint")
    parser.add_argument("--out_dir", type=str, required=True, help="Zielordner fÃ¼r Ausgaben")
    args = parser.parse_args()
    main(args)
