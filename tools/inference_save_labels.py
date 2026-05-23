import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from engine.core import YAMLConfig

def process_images(model, device, img_size, vit_backbone, img_paths, output_label_dir, det_thresh=0.45):
    transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if vit_backbone else T.Lambda(lambda x: x)
    ])

    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir, exist_ok=True)

    for img_path in tqdm(img_paths, desc="Running Inference"):
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(im_pil).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(im_data, orig_size)
        
        labels, boxes, scores = output
        
        boxes_np = boxes[0].cpu().numpy()
        scores_np = scores[0].cpu().numpy()
        labels_np = labels[0].cpu().numpy()

        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(output_label_dir, label_name)
        
        with open(label_path, "w") as f:
            for i in range(len(boxes_np)):
                if scores_np[i] < det_thresh:
                    continue
                
                x1, y1, x2, y2 = boxes_np[i]
                cls = int(labels_np[i])
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                nw = (x2 - x1) / w
                nh = (y2 - y1) / h
                
                f.write(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

def main(args):
    print(f"Loading config from {args.config}...")
    sys.stdout.flush()
    cfg = YAMLConfig(args.config, resume=args.resume)
    print("Config loaded.")
    sys.stdout.flush()

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    model.eval()

    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)

    # Process dataset
    dataset_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    images_root = os.path.join(dataset_dir, 'images')
    if not os.path.exists(images_root):
        print(f"Error: {images_root} not found.")
        return

    # Look for train/val/test splits
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(images_root, split)
        if not os.path.exists(split_dir):
            continue
            
        print(f"Processing split: {split}")
        img_paths = []
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_paths.append(os.path.join(root, file))
        
        print(f"Found {len(img_paths)} images in {split}")
        sys.stdout.flush()
        if not img_paths:
            continue
            
        output_split_dir = os.path.join(output_dir, split)
        process_images(model, device, img_size, vit_backbone, img_paths, output_split_dir, args.thresh)

    print(f"Inference completed. Labels saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to dataset")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output directory")
    parser.add_argument('-t', '--thresh', type=float, default=0.45, help="Detection threshold")
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(args)
