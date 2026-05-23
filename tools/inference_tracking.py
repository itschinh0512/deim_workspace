import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import argparse
import glob
import re
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from engine.core import YAMLConfig

sys.path.append('/home/locth/omni2rect_DEIM/OC_SORT')
from trackers.ocsort_tracker.ocsort import OCSort

def get_video_prefix(filename):
    # Extract the video prefix from filename like 11_shot1_fisheye_fisheye_00002475_normal_00005100.jpg
    # Remove the trailing _\d+_normal_\d+.jpg
    match = re.search(r'(.+)_\d+_normal_\d+\.jpg', filename)
    if match:
        return match.group(1)
    # Fallback to something else
    parts = filename.split('_')
    if len(parts) > 3:
        return '_'.join(parts[:-3])
    return 'unknown_video'

def draw(images, labels, boxes, scores, track_ids, thrh=0.45):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        # Track IDs already matches the filtered scores from OC_SORT, 
        # but since OC_SORT does its own filtering, let's just use what OC_SORT returned.
        # Wait, the draw function in DEIMv2 filters based on scores.
        # But if we use track_ids, they come from OC_SORT output.
        pass

def get_color(idx):
    idx = int(idx) * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def process_sequence(model, device, img_size, vit_backbone, img_paths, output_video_path=None, output_label_dir=None, det_thresh=0.45):
    # Initialize tracker for the sequence
    tracker = OCSort(det_thresh=det_thresh, iou_threshold=0.3, use_byte=False)

    transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if vit_backbone else T.Lambda(lambda x: x)
    ])

    if not img_paths:
        return

    # Read first image to get dimensions
    first_im = Image.open(img_paths[0]).convert('RGB')
    orig_w, orig_h = first_im.size

    # Video writer (optional)
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10 # arbitrary fps for sequence of images
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_w, orig_h))

    if output_label_dir:
        os.makedirs(output_label_dir, exist_ok=True)

    # Metadata for the sequence
    sequence_metadata = {
        "sub_sequence": os.path.basename(output_video_path) if output_video_path else "unknown",
        "tracks": []
    }
    track_data = {} # tid -> list of frames

    for img_path in tqdm(img_paths, desc=f"Processing {os.path.basename(output_video_path) if output_video_path else 'sequence'}"):
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(im_pil).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(im_data, orig_size)
        
        labels, boxes, scores = output
        
        # Format for OC_SORT: [[x1,y1,x2,y2,score], ...]
        boxes_np = boxes[0].cpu().numpy()
        scores_np = scores[0].cpu().numpy()
        labels_np = labels[0].cpu().numpy()

        # Combine
        if len(boxes_np) > 0:
            dets = np.concatenate((boxes_np, scores_np[:, None]), axis=1)
        else:
            dets = np.empty((0, 5))

        # Update tracker
        # img_info is [h, w], img_size is the tracker scale thing. 
        # The tracker expects dets to be unscaled (original image size), and img_size to be the size we want to scale to?
        # Looking at OC_SORT code:
        # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        # bboxes /= scale
        # If we pass img_size = [h, w], scale will be 1, which is what we want since DEIM already outputs original coords.
        online_targets = tracker.update(dets, [h, w], [h, w])
        
        # online_targets is [x1, y1, x2, y2, track_id]
        
        # Save labels
        if output_label_dir:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(output_label_dir, label_name)
            
            with open(label_path, "w") as f:
                for t in online_targets:
                    x1, y1, x2, y2 = t[:4]
                    tid = int(t[4])
                    
                    if tid not in track_data:
                        track_data[tid] = []
                    track_data[tid].append(img_name)

                    # Convert to normalized OBB format (8 points) for compatibility with the annotation tool
                    # Points: (x1,y1), (x2,y1), (x2,y2), (x1,y2)
                    nx1, ny1 = x1 / w, y1 / h
                    nx2, ny2 = x2 / w, y1 / h
                    nx3, ny3 = x2 / w, y2 / h
                    nx4, ny4 = x1 / w, y2 / h
                    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 track_id
                    # Using class_id 0 as default (license plate)
                    f.write(f"0 {nx1:.6f} {ny1:.6f} {nx2:.6f} {ny2:.6f} {nx3:.6f} {ny3:.6f} {nx4:.6f} {ny4:.6f} {tid}\n")

        # Draw on image (only if video output is requested)
        if out:
            draw_im = ImageDraw.Draw(im_pil)
            for t in online_targets:
                tlwh = t[:4]
                tid = int(t[4])
                color = get_color(tid)
                
                draw_im.rectangle(list(tlwh), outline=color, width=3)
                draw_im.text((tlwh[0], tlwh[1]-10), text=f"ID: {tid}", fill=color)

            # Write to video
            frame = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
            out.write(frame)

    if out:
        out.release()

    # Save metadata JSON
    if output_label_dir and track_data:
        meta_dir = os.path.join(os.path.dirname(output_label_dir), "metadata")
        os.makedirs(meta_dir, exist_ok=True)
        
        split = os.path.basename(output_label_dir)
        seq_name = sequence_metadata["sub_sequence"].replace("_tracked.mp4", "")
        meta_path = os.path.join(meta_dir, f"{split}_{seq_name}.json")
        
        for tid, frames in track_data.items():
            sequence_metadata["tracks"].append({
                "track_id": tid,
                "num_frames": len(frames),
                "frames": frames,
                "best_crop_frame": frames[len(frames)//2] # Pick middle frame as thumbnail
            })
            
        with open(meta_path, "w") as f:
            json.dump(sequence_metadata, f, indent=2)


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

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

    images_dir = os.path.join(dataset_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} not found.")
        return

    scenes = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    
    for scene in scenes:
        scene_dir = os.path.join(images_dir, scene)
        
        # Get all images in this scene using os.walk
        img_paths = []
        for root, _, files in os.walk(scene_dir):
            for file in files:
                if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                    img_paths.append(os.path.join(root, file))
        img_paths.sort()
        
        if not img_paths:
            print(f"No images found in {scene_dir}.")
            continue
            
        # Group by segment directory
        sequences = {}
        for p in img_paths:
            seg_dir = os.path.dirname(p)
            if seg_dir not in sequences:
                sequences[seg_dir] = []
            sequences[seg_dir].append(p)
        
        print(f"Found {len(sequences)} segments in {scene}.")
        
        scene_out_dir = os.path.join(output_dir, scene)
        os.makedirs(scene_out_dir, exist_ok=True)

        # Determine split (default to train, or extract from path if possible)
        # Assuming path like .../images/train/scene...
        split = "train"
        if "val" in images_dir:
            split = "val"
        elif "test" in images_dir:
            split = "test"

        label_split_dir = os.path.join(output_dir, "labels_tracked", split)
        os.makedirs(label_split_dir, exist_ok=True)

        for seg_dir, seq_paths in sequences.items():
            rel_path = os.path.relpath(seg_dir, scene_dir)
            seq_name = rel_path.replace(os.sep, '_')
            print(f"Processing segment {seq_name} with {len(seq_paths)} frames...")
            # We can still output video if desired, but focus is now on labels
            out_vid = os.path.join(scene_out_dir, f"{seq_name}_tracked.mp4")
            process_sequence(model, device, img_size, vit_backbone, seq_paths, out_vid, label_split_dir)

    print("All tracking tasks completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to dataset_fisheye")
    parser.add_argument('-o', '--output', type=str, default='tracking_outputs', help="Output directory")
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(args)
