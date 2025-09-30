import torch
import argparse
from PIL import Image, ImageDraw
import os

from models.dino.dino import build_dino, PostProcess
from util.slconfig import SLConfig
from util import box_ops

import torchvision.transforms as T

def visualize_pil(image, outputs, save_path, score_threshold=0.3):
    draw = ImageDraw.Draw(image)
    boxes = outputs['boxes']
    scores = outputs['scores']
    labels = outputs['labels']

    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        box = box.tolist()
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{label.item()} ({score:.2f})", fill="red")

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
    image, img_tensor = prepare_image(args.image_path)

    with torch.no_grad():
        outputs = model(img_tensor)

    postprocessor = PostProcess()
    results = postprocessor(outputs, torch.tensor([[img_tensor.shape[2], img_tensor.shape[3]]]).cuda())

    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)

    visualize_pil(image, results[0], args.save_path)
    print(f"✔️ Ergebnis gespeichert unter: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="output.jpg")
    args = parser.parse_args()
    main(args)
