"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import argparse
import os
import subprocess
import sys
import time

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import torch
import torchvision.transforms as transforms
import tqdm
from fiftyone import ViewField as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from engine.core import YAMLConfig

label_map = {
    0: 'license_plate',
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
DEFAULT_IMG_DIR = os.path.join(PROJECT_ROOT, 'dataset_fisheye', 'images', 'val')
DEFAULT_ANN_FILE = os.path.join(PROJECT_ROOT, 'dataset_fisheye', 'annotations', 'instances_val.json')


def kill_existing_mongod():
    try:
        result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE)
        processes = result.stdout.decode('utf-8').splitlines()

        for process in processes:
            if 'mongod' in process and '--dbpath' in process:
                # find mongod PID
                pid = int(process.split()[1])
                print(f"Killing existing mongod process with PID: {pid}")
                # kill mongod session
                os.kill(pid, 9)
    except Exception as e:
        print(f"Error occurred while killing mongod: {e}")

kill_existing_mongod()

class CustomModel(fom.Model):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.eval().cuda()
        self.postprocessor = cfg.postprocessor.eval().cuda()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640)),  # Resize to the size expected by your model
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def media_type(self):
        return "image"

    @property
    def has_logits(self):
        return False

    @property
    def has_embeddings(self):
        return False

    @property
    def ragged_batches(self):
        return False

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self):
        return True

    @preprocess.setter
    def preprocess(self, value):
        pass

    def _convert_predictions(self, predictions):
        class_labels, bboxes, scores = predictions[0]['labels'], predictions[0]['boxes'], predictions[0]['scores']

        detections = []
        for label, bbox, score in zip(class_labels, bboxes, scores):
            detection = fol.Detection(
                label=label_map.get(label.item(), str(label.item())),
                bounding_box=[
                    bbox[0] / 640,  # Normalized coordinates
                    bbox[1] / 640,
                    (bbox[2] - bbox[0]) / 640,
                    (bbox[3] - bbox[1]) / 640
                ],
                confidence=score
            )
            detections.append(detection)

        return fol.Detections(detections=detections)

    def predict(self, image):
        image = Image.fromarray(image).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).cuda()
        outputs = self.model(image_tensor)
        orig_target_sizes = torch.tensor([[640, 640]]).cuda()
        predictions = self.postprocessor(outputs, orig_target_sizes)
        return self._convert_predictions(predictions)

    def predict_all(self, images):
        image_tensors = []
        for image in images:
            image = Image.fromarray(image)
            image_tensor = self.transform(image)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors).cuda()
        outputs = self.model(image_tensors)
        orig_target_sizes = torch.tensor([[640, 640] for image in images]).cuda()
        predictions = self.postprocessor(outputs, orig_target_sizes)
        converted_predictions = [self._convert_predictions(pred) for pred in predictions]

        # Ensure the output is a list of lists of Detections
        return converted_predictions

def filter_by_predictions5_confidence(predictions_view, confidence_threshold=0.3):
    for j, sample in tqdm.tqdm(enumerate(predictions_view), total=len(predictions_view)):
        has_modified = False
        for i, detection in enumerate(sample["predictions0"].detections):

            if "original_confidence" not in detection:
                detection["original_confidence"] = detection["confidence"]

            if (detection["confidence"] <= confidence_threshold and sample["predictions5"].detections[i]["confidence"] >= confidence_threshold) or \
               (detection["confidence"] >= confidence_threshold and sample["predictions5"].detections[i]["confidence"] <= confidence_threshold):

                sample["predictions0"].detections[i]["confidence"] = sample["predictions5"].detections[i]["confidence"]
                has_modified = True
        if has_modified:
            sample.save()


def restore_confidence(predictions_view):
    for j, sample in tqdm.tqdm(enumerate(predictions_view), total=len(predictions_view)):
        for i, detection in enumerate(sample["predictions0"].detections):
            if "original_confidence" in detection:
                detection["confidence"] = detection["original_confidence"]
        sample.save()

def fast_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = w1 * h1
    boxBArea = w2 * h2
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def assign_iou_diff(predictions_view):
    for sample in predictions_view:
        ious_0 = [detection.eval0_iou if 'eval0_iou' in detection else None for detection in sample["predictions0"].detections]
        ious_5 = [detection.eval5_iou if 'eval5_iou' in detection else None for detection in sample["predictions5"].detections]
        bbox_0 = [detection.bounding_box for detection in sample["predictions0"].detections]
        bbox_5 = [detection.bounding_box for detection in sample["predictions5"].detections]
        # iou_diffs = [abs(iou_5 - iou_0) if iou_0 is not None and iou_5 is not None else -1 for iou_0, iou_5 in zip(ious_0, ious_5)]
        iou_inter = [fast_iou(b0, b5) for b0, b5 in zip(bbox_0, bbox_5)]
        iou_diffs = [abs(iou_5 - iou_0) if iou_0 is not None and iou_5 is not None and iou_inter > 0.5 else -1 for iou_0, iou_5, iou_inter in zip(ious_0, ious_5, iou_inter)]

        for detection, iou_diff in zip(sample["predictions0"].detections, iou_diffs):
            detection["iou_diff"] = iou_diff
        for detection, iou_diff in zip(sample["predictions5"].detections, iou_diffs):
            detection["iou_diff"] = iou_diff
        # for detection, iou_diff in zip(sample["predictions100"].detections, iou_diffs):
        #     detection["iou_diff"] = iou_diff
        sample.save()

def resolve_gt_field(dataset, requested_field):
    field_schema = dataset.get_field_schema()
    if requested_field:
        if requested_field in field_schema:
            return requested_field
        raise ValueError(
            "Ground-truth field '{}' not found. Available fields: {}".format(
                requested_field, ", ".join(sorted(field_schema.keys()))
            )
        )
    if "ground_truth" in field_schema:
        return "ground_truth"
    detection_fields = [
        name
        for name, field in field_schema.items()
        if getattr(field, "document_type", None) == fol.Detections
    ]
    if len(detection_fields) == 1:
        return detection_fields[0]
    if len(detection_fields) > 1:
        raise ValueError(
            "Multiple detection fields found: {}. Use --gt-field to choose one.".format(
                ", ".join(sorted(detection_fields))
            )
        )
    return None

def get_or_create_dataset(args):
    if args.dataset_name:
        existing = args.dataset_name in fo.list_datasets()
        if existing and args.overwrite_dataset:
            fo.delete_dataset(args.dataset_name)
            existing = False
        if existing:
            return fo.load_dataset(args.dataset_name)
    return fo.Dataset.from_dir(
        dataset_dir=os.path.dirname(args.ann_file),
        dataset_type=fo.types.COCODetectionDataset,
        data_path=args.img_folder,
        labels_path=args.ann_file,
        name=args.dataset_name,
    )

def main(args):
    try:
        if os.path.exists("saved_predictions_view") and os.path.exists("saved_filtered_view"):
            print("Loading saved predictions and filtered views...")
            predictions_view = fo.Dataset.from_dir(
                dataset_dir="saved_predictions_view",
                dataset_type=fo.types.FiftyOneDataset
            ).view()
            filtered_view = fo.Dataset.from_dir(
                dataset_dir="saved_filtered_view",
                dataset_type=fo.types.FiftyOneDataset
            ).view()
            predictions_view.dataset.persistent = True
            session = fo.launch_app(predictions_view.dataset, port=args.port)
        else:
            dataset = get_or_create_dataset(args)

            dataset.persistent = True

            session = fo.launch_app(dataset, port=args.port)
            cfg = YAMLConfig(args.config, resume=args.resume)
            if 'HGNetv2' in cfg.yaml_cfg:
                cfg.yaml_cfg['HGNetv2']['pretrained'] = False
            if args.resume:
                checkpoint = torch.load(args.resume, map_location='cpu')
                if 'ema' in checkpoint:
                    state = checkpoint['ema']['module']
                else:
                    state = checkpoint['model']
            else:
                raise AttributeError('only support resume to load model.state_dict by now.')

            # NOTE load train mode state -> convert to deploy mode
            cfg.model.load_state_dict(state)
            gt_field = resolve_gt_field(dataset, args.gt_field)
            if args.limit is not None:
                predictions_view = dataset.take(args.limit, seed=51)
            else:
                predictions_view = dataset.view()

            model = CustomModel(cfg)
            L = model.model.decoder.decoder.eval_idx
            # Apply models and save predictions in different label fields
            for i in [L]:
                model.model.decoder.decoder.eval_idx = i
                label_field = "predictions{:d}".format(i)
                predictions_view.apply_model(model, label_field=label_field)

            # filter_by_predictions5_confidence(predictions_view, confidence_threshold=0.3)
            if gt_field is None:
                print("Skipping evaluation: no ground-truth detection field found")
            else:
                for i in [L]:
                    label_field = "predictions{:d}".format(i)
                    predictions_view = predictions_view.filter_labels(label_field, F("confidence") > 0.5, only_matches=False)
                    eval_key = "eval{:d}".format(i)
                    _ = predictions_view.evaluate_detections(
                        label_field,
                        gt_field=gt_field,
                        eval_key=eval_key,
                        compute_mAP=True,
                    )

            # assign_iou_diff(predictions_view)

            # filtered_view = predictions_view.filter_labels("predictions0", F("iou_diff") > 0.05, only_matches=True)
            # filtered_view = filtered_view.filter_labels("predictions5", F("iou_diff") > 0.05, only_matches=True)
            # restore_confidence(filtered_view)

            predictions_view.export(
                export_dir="saved_predictions_view",
                dataset_type=fo.types.FiftyOneDataset
            )
            # filtered_view.export(
            #     export_dir="saved_filtered_view",
            #     dataset_type=fo.types.FiftyOneDataset
            # )

        # Display the filtered view
        session.view = predictions_view

        # Keep the session open
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Shutting down session")
        if 'session' in locals():
            session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--port', '-p', type=int)
    parser.add_argument('--img-folder', type=str, default=DEFAULT_IMG_DIR)
    parser.add_argument('--ann-file', type=str, default=DEFAULT_ANN_FILE)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--gt-field', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default='fisheye-detection')
    parser.add_argument('--overwrite-dataset', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
