#!/usr/bin/env python3
"""
Offline Vehicle-Assisted License Plate Tracker

Reads pre-computed labels from disk:
  - LP: merged_labels (OBB: class x1..x8)
  - Vehicle: vehicle_labels (YOLO: class xc yc w h)

Outputs tracked LP labels (OBB + track_id) and metadata JSON.
Treats all splits as a monolithic dataset.
"""
import os
import re
import json
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ===== Constants =====
SEGMENT_GAP = 100
LP_IOU_THRESH = 0.15        # Low because LPs are small
VEH_IOU_THRESH = 0.25
MAX_AGE_NORMAL = 50          # Frame index gap
MAX_AGE_ASSISTED = 500       # With vehicle parent alive
MAX_AGE_OCCLUDED = 800       # Actively occluded by another vehicle
LP_VEH_EXPAND = 0.4          # Expand vehicle bbox for LP association
LP_VEH_PROXIMITY = 0.15      # Fallback proximity threshold (normalized dist)

# ===== Utility =====

def obb_to_aabb(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    return np.array([min(xs), min(ys), max(xs), max(ys)])

def yolo_to_aabb(yolo):
    xc, yc, w, h = yolo
    return np.array([xc - w/2, yc - h/2, xc + w/2, yc + h/2])

def aabb_center(a):
    return np.array([(a[0]+a[2])/2, (a[1]+a[3])/2])

def aabb_area(a):
    return max(0, a[2]-a[0]) * max(0, a[3]-a[1])

def compute_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = aabb_area(a) + aabb_area(b) - inter
    return inter / union if union > 0 else 0

def overlap_ratio(small, big):
    """Fraction of small_box covered by big_box."""
    ix1 = max(small[0], big[0]); iy1 = max(small[1], big[1])
    ix2 = min(small[2], big[2]); iy2 = min(small[3], big[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area = aabb_area(small)
    return inter / area if area > 0 else 0

def expand_aabb(a, factor):
    cx, cy = aabb_center(a)
    w = (a[2]-a[0]) * (1+factor)
    h = (a[3]-a[1]) * (1+factor)
    return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])

def dist_centers(a, b):
    return np.linalg.norm(aabb_center(a) - aabb_center(b))

# ===== Label Parsing =====

def parse_obb(path):
    boxes = []
    if not os.path.exists(path): return boxes
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 9:
                boxes.append(np.array([float(x) for x in p[1:9]]))
    return boxes

def parse_yolo(path):
    boxes = []
    if not os.path.exists(path): return boxes
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                boxes.append([float(x) for x in p[1:5]])
    return boxes

# ===== Segment Building =====

def parse_frame_info(filename):
    name = filename.replace('.jpg','').replace('.png','').replace('.txt','')
    m = re.match(r'(.+_fisheye)_(\d+)_normal_(\d+)', name)
    if m: return m.group(1), int(m.group(2))
    return None, None

def build_segments(filenames, max_gap=SEGMENT_GAP):
    groups = defaultdict(list)
    for f in filenames:
        prefix, idx = parse_frame_info(f)
        if prefix is not None:
            groups[prefix].append((idx, f))

    segments = []
    for prefix, frames in groups.items():
        frames.sort()
        current = [frames[0]]
        for i in range(1, len(frames)):
            if frames[i][0] - frames[i-1][0] > max_gap:
                segments.append({'prefix': prefix, 'frames': current})
                current = []
            current.append(frames[i])
        if current:
            segments.append({'prefix': prefix, 'frames': current})
    return segments

# ===== Track =====

class Track:
    def __init__(self, tid, aabb, obb, frame_idx, frame_name):
        self.id = tid
        self.aabb = aabb
        self.obb = obb
        self.velocity = np.array([0.0, 0.0])
        self.last_frame_idx = frame_idx
        self.frames = {frame_name: obb}
        self.state = 'active'
        self.age = 0  
        self.parent_veh = None  
        self.occluded_by_other = False

    def predict(self, gap):
        cx, cy = aabb_center(self.aabb)
        damping = min(1.0, 10.0 / max(1, gap))
        pcx = cx + self.velocity[0] * gap * damping
        pcy = cy + self.velocity[1] * gap * damping
        w = self.aabb[2] - self.aabb[0]
        h = self.aabb[3] - self.aabb[1]
        return np.array([pcx-w/2, pcy-h/2, pcx+w/2, pcy+h/2])

    def update(self, aabb, obb, frame_idx, frame_name):
        gap = max(1, frame_idx - self.last_frame_idx)
        old_c = aabb_center(self.aabb)
        new_c = aabb_center(aabb)
        new_v = (new_c - old_c) / gap
        self.velocity = 0.4 * new_v + 0.6 * self.velocity
        self.aabb = aabb
        self.obb = obb
        self.last_frame_idx = frame_idx
        self.frames[frame_name] = obb
        self.state = 'active'
        self.age = 0
        self.occluded_by_other = False

    def mark_missed(self, gap):
        self.age += gap

    @property
    def max_age(self):
        if self.occluded_by_other: return MAX_AGE_OCCLUDED
        if self.parent_veh and self.parent_veh.state == 'active': return MAX_AGE_ASSISTED
        return MAX_AGE_NORMAL

    @property
    def is_lost(self):
        return self.age > self.max_age

# ===== Matching =====

def greedy_match(cost_matrix, threshold):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    pairs = []
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            if cost_matrix[i,j] < threshold:
                pairs.append((cost_matrix[i,j], i, j))
    pairs.sort()
    ur, uc = set(), set()
    matched = []
    for _, i, j in pairs:
        if i not in ur and j not in uc:
            matched.append((i,j)); ur.add(i); uc.add(j)
    return matched, [i for i in range(cost_matrix.shape[0]) if i not in ur], \
           [j for j in range(cost_matrix.shape[1]) if j not in uc]

# ===== Dual Tracker =====

# Distance thresholds (normalized image coords)
LP_DIST_TIGHT = 0.03       # First pass: tight match
LP_DIST_RELAXED = 0.08     # Second pass: relaxed match
VEH_DIST_TIGHT = 0.06
VEH_DIST_RELAXED = 0.12

class DualTracker:
    def __init__(self):
        self.lp_tracks = []
        self.veh_tracks = []
        self.next_lp_id = 1
        self.next_veh_id = 1
        self.prev_frame_idx = None

    def _associate_lp_vehicle(self, lp_aabbs, veh_aabbs):
        assoc = {}
        for li, lp in enumerate(lp_aabbs):
            lp_c = aabb_center(lp)
            best_vi, best_dist = None, LP_VEH_PROXIMITY
            for vi, veh in enumerate(veh_aabbs):
                expanded = expand_aabb(veh, LP_VEH_EXPAND)
                if expanded[0] <= lp_c[0] <= expanded[2] and expanded[1] <= lp_c[1] <= expanded[3]:
                    d = dist_centers(lp, veh)
                    if d < best_dist:
                        best_dist = d; best_vi = vi
            if best_vi is not None:
                assoc[li] = best_vi
            else:
                for vi, veh in enumerate(veh_aabbs):
                    d = dist_centers(lp, veh)
                    if d < best_dist:
                        best_dist = d; best_vi = vi
                if best_vi is not None:
                    assoc[li] = best_vi
        return assoc

    def _check_occlusion(self, track, veh_aabbs, parent_veh_aabb):
        pred = track.predict(track.age if track.age > 0 else 1)
        for veh in veh_aabbs:
            if parent_veh_aabb is not None and compute_iou(veh, parent_veh_aabb) > 0.5:
                continue  
            if overlap_ratio(pred, veh) > 0.3:
                return True
        return False

    def _match_by_distance(self, tracks, det_aabbs, frame_gap, dist_thresh):
        """Match tracks to detections using center distance.
        
        Adapts threshold based on frame gap — larger gaps allow more drift.
        Returns matched pairs, unmatched track indices, unmatched det indices.
        """
        if not tracks or not det_aabbs:
            return [], list(range(len(tracks))), list(range(len(det_aabbs)))

        n, m = len(tracks), len(det_aabbs)
        cost = np.full((n, m), 1e6)
        
        # Scale threshold with frame gap (objects can move more in larger gaps)
        gap_factor = min(max(1.0, frame_gap / 5.0), 5.0)
        effective_thresh = dist_thresh * gap_factor

        for i, t in enumerate(tracks):
            pred = t.predict(frame_gap)
            pred_c = aabb_center(pred)
            for j, d in enumerate(det_aabbs):
                det_c = aabb_center(d)
                dist = np.linalg.norm(pred_c - det_c)
                if dist < effective_thresh:
                    cost[i, j] = dist

        return greedy_match(cost, effective_thresh)

    def _two_stage_match(self, tracks, det_aabbs, frame_gap, tight_thresh, relaxed_thresh):
        """Two-stage matching: tight first, then relaxed for remainders."""
        # Stage 1: tight match
        matched, unmatched_t, unmatched_d = self._match_by_distance(
            tracks, det_aabbs, frame_gap, tight_thresh)
        
        if not unmatched_t or not unmatched_d:
            return matched, unmatched_t, unmatched_d
        
        # Stage 2: relaxed match on remainders
        remaining_tracks = [tracks[i] for i in unmatched_t]
        remaining_dets = [det_aabbs[j] for j in unmatched_d]
        
        matched2, still_unmatched_t, still_unmatched_d = self._match_by_distance(
            remaining_tracks, remaining_dets, frame_gap, relaxed_thresh)
        
        # Map back to original indices
        for ti2, di2 in matched2:
            matched.append((unmatched_t[ti2], unmatched_d[di2]))
        
        final_unmatched_t = [unmatched_t[i] for i in still_unmatched_t]
        final_unmatched_d = [unmatched_d[j] for j in still_unmatched_d]
        
        return matched, final_unmatched_t, final_unmatched_d

    def process_frame(self, lp_obbs, veh_yolos, frame_idx, frame_name):
        gap = frame_idx - self.prev_frame_idx if self.prev_frame_idx is not None else 1
        gap = max(1, gap)
        self.prev_frame_idx = frame_idx

        lp_aabbs = [obb_to_aabb(o) for o in lp_obbs]
        veh_aabbs = [yolo_to_aabb(v) for v in veh_yolos]

        # --- Vehicle tracking (two-stage distance match) ---
        matched_v, unmatched_vt, unmatched_vd = self._two_stage_match(
            self.veh_tracks, veh_aabbs, gap, VEH_DIST_TIGHT, VEH_DIST_RELAXED)
        for ti, di in matched_v:
            self.veh_tracks[ti].update(veh_aabbs[di], None, frame_idx, frame_name)
        for ti in unmatched_vt:
            self.veh_tracks[ti].mark_missed(gap)
        for di in unmatched_vd:
            t = Track(self.next_veh_id, veh_aabbs[di], None, frame_idx, frame_name)
            self.next_veh_id += 1
            self.veh_tracks.append(t)
        self.veh_tracks = [t for t in self.veh_tracks if not t.is_lost]

        # --- LP-Vehicle association ---
        active_veh_aabbs = [t.aabb for t in self.veh_tracks]
        lp_veh_assoc = self._associate_lp_vehicle(lp_aabbs, active_veh_aabbs)

        # --- LP tracking (two-stage distance match) ---
        matched_l, unmatched_lt, unmatched_ld = self._two_stage_match(
            self.lp_tracks, lp_aabbs, gap, LP_DIST_TIGHT, LP_DIST_RELAXED)

        for ti, di in matched_l:
            self.lp_tracks[ti].update(lp_aabbs[di], lp_obbs[di], frame_idx, frame_name)
            if di in lp_veh_assoc:
                self.lp_tracks[ti].parent_veh = self.veh_tracks[lp_veh_assoc[di]]

        for ti in unmatched_lt:
            t = self.lp_tracks[ti]
            parent_aabb = t.parent_veh.aabb if t.parent_veh and t.parent_veh.state == 'active' else None
            if self._check_occlusion(t, active_veh_aabbs, parent_aabb):
                t.occluded_by_other = True
            t.mark_missed(gap)

        for di in unmatched_ld:
            t = Track(self.next_lp_id, lp_aabbs[di], lp_obbs[di], frame_idx, frame_name)
            self.next_lp_id += 1
            if di in lp_veh_assoc:
                t.parent_veh = self.veh_tracks[lp_veh_assoc[di]]
            self.lp_tracks.append(t)

        self.lp_tracks = [t for t in self.lp_tracks if not t.is_lost]

    def get_active_lp_tracks(self):
        return [t for t in self.lp_tracks if t.state == 'active']



def process_segment(segment, lp_dir, veh_dir, output_dir, file_to_split):
    tracker = DualTracker()

    for frame_idx, frame_name in segment['frames']:
        split = file_to_split[frame_name]
        lbl_name = frame_name.replace('.jpg','.txt').replace('.png','.txt')
        
        lp_obbs = parse_obb(os.path.join(lp_dir, split, lbl_name))
        veh_yolos = parse_yolo(os.path.join(veh_dir, split, lbl_name))
        tracker.process_frame(lp_obbs, veh_yolos, frame_idx, frame_name)

        active = tracker.get_active_lp_tracks()
        out_path = os.path.join(output_dir, split, lbl_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with open(out_path, 'w') as f:
            for t in active:
                if frame_name in t.frames:
                    obb = t.frames[frame_name]
                    coords = ' '.join(f'{c:.6f}' for c in obb)
                    f.write(f'0 {coords} {t.id}\n')

    return tracker.lp_tracks


def main():
    parser = argparse.ArgumentParser(description='Offline Vehicle-Assisted LP Tracker (Monolithic)')
    parser.add_argument('--lp-dir', default='/home/locth/omni2rect_DEIM/Fisheye_dataset_fisheye/merged_labels')
    parser.add_argument('--veh-dir', default='/home/locth/omni2rect_DEIM/Fisheye_dataset_fisheye/vehicle_labels')
    parser.add_argument('--img-dir', default='/home/locth/omni2rect_DEIM/Fisheye_dataset_fisheye/images')
    parser.add_argument('--output', default='/home/locth/omni2rect_DEIM/Fisheye_dataset_fisheye/labels_tracked_v2')
    args = parser.parse_args()

    all_filenames = []
    file_to_split = {}
    
    for split in ['train', 'val', 'test']:
        img_split = os.path.join(args.img_dir, split)
        if not os.path.exists(img_split):
            continue
        
        files = [f for f in os.listdir(img_split) if f.endswith(('.jpg','.png'))]
        for f in files:
            all_filenames.append(f)
            file_to_split[f] = split

    all_filenames.sort()
    segments = build_segments(all_filenames)
    print(f"Total images: {len(all_filenames)}, Total segments: {len(segments)}")

    meta_dir = os.path.join(args.output, 'metadata')
    os.makedirs(meta_dir, exist_ok=True)

    for seg_i, seg in enumerate(tqdm(segments, desc='Tracking Monolithic Dataset')):
        prefix = seg['prefix']
        seg_name = f'{prefix}_seg{seg_i}'
        tracks = process_segment(seg, args.lp_dir, args.veh_dir, args.output, file_to_split)

        # Determine split for metadata (use first frame's split)
        first_frame = seg['frames'][0][1]
        split = file_to_split[first_frame]

        meta = {
            'sub_sequence': seg_name,
            'prefix': prefix,
            'num_frames': len(seg['frames']),
            'tracks': []
        }
        for t in tracks:
            if len(t.frames) >= 2:  
                meta['tracks'].append({
                    'track_id': t.id,
                    'num_frames': len(t.frames),
                    'frames': sorted(t.frames.keys()),
                    'best_crop_frame': sorted(t.frames.keys())[len(t.frames)//2]
                })
        meta_path = os.path.join(meta_dir, f'{split}_{seg_name}.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    print(f'Tracking complete. Output: {args.output}')

if __name__ == '__main__':
    main()
