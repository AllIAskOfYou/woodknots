import cv2
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from typing import Any, Dict

from mltools.matching.volume import VoxelGrid


def load_sfm_data(sfm_path: str | Path) -> dict:
    sfm_path = Path(sfm_path)
    if not sfm_path.exists():
        raise FileNotFoundError(f"SfM file not found at: {sfm_path}")

    with open(sfm_path, "r") as f:
        return json.load(f)

class MeshroomCamerasParser:
    def __init__(self, cameras_path: str):
        self.cameras_path = cameras_path
        self.data = load_sfm_data(cameras_path)

    def _parse_intrinsics(self):
        intrinsics = {}
        for item in self.data["intrinsics"]:
            id = item["intrinsicId"]
            pp = item.get("principalPoint", [0.0, 0.0])
            intrinsics[id] = {
                "width": int(item["width"]),
                "height": int(item["height"]),
                "sensor_width": float(item["sensorWidth"]),
                "sensor_height": float(item["sensorHeight"]),
                "focal_length": float(item["focalLength"]),
                # Meshroom stores the principal point as an (x, y) offset from
                # the image centre in its own y-down convention; the renderer
                # is y-up, the same flip _parse_poses applies to the poses
                # (diag(1, -1, -1)). So x carries over and y negates. This
                # previously rotated the offset instead, (x, y) -> (y, -x),
                # which put the principal point a few pixels off both axes.
                "cx": float(pp[0]),
                "cy": -float(pp[1]),
                "distortion_params": np.array(item["distortionParams"], dtype=np.float32)
            }
        return intrinsics
    
    def _parse_poses(self):
        poses = {}
        for item in self.data["poses"]:
            id = item["poseId"]
            transform = item["pose"]["transform"]
            R = np.array(transform["rotation"], dtype=np.float32)
            C = np.array(transform["center"], dtype=np.float32)
            rotation_matrix = np.array([
                [R[0], -R[1], -R[2]],
                [-R[3], R[4], R[5]],
                [-R[6], R[7], R[8]]
            ], dtype=np.float32)
            position = np.array([C[0], -C[1], -C[2]], dtype=np.float32)
            poses[id] = {
                "rotation_matrix": rotation_matrix,
                "position": position
            }
        return poses

    def parse(self):
        intrinsics = self._parse_intrinsics()
        poses = self._parse_poses()
        
        cameras = {}
        for view in self.data["views"]:
            intrinsic = intrinsics.get(view.get("intrinsicId"))
            pose = poses.get(view.get("poseId"))
            if intrinsic is None or pose is None:
                continue
            cameras[view["frameId"]] = {
                "position": pose["position"],
                "rotation_matrix": pose["rotation_matrix"],
                "focal_length": intrinsic["focal_length"],
                "sensor_width": intrinsic["sensor_width"],
                "sensor_height": intrinsic["sensor_height"],
                "width": intrinsic["width"],
                "height": intrinsic["height"],
                "cx": intrinsic["cx"],
                "cy": intrinsic["cy"],
                "distortion_params": intrinsic["distortion_params"]
            }
        return cameras


def parse_coco_annotations(coco_json_path: str | Path) -> list[dict[str, Any]]:
    if not Path(coco_json_path).exists():
        raise FileNotFoundError(f"COCO JSON file not found at: {coco_json_path}")
    coco = COCO(coco_json_path)
    # load all annotations of wood knots and with valid knot id
    annotations = [ann for ann in coco.anns.values() if ann['category_id'] == 2 and ann['attributes']['id'] != -1]
    annotations_parsed = dict()
    for ann in annotations:
        ann_id = ann['id']
        image_id = ann['image_id']
        image_name = coco.loadImgs(ann['image_id'])[0]['file_name']
        knot_id = ann['attributes']['id']
        segmentation = ann['segmentation']
        # normalize segmentation coordinates to [0, 1] range based on image dimensions
        img_info = coco.loadImgs(image_id)[0]
        width, height = img_info['width'], img_info['height']
        is_portrait = height > width
        # print(f"Image {image_name} dimensions: width={width}, height={height}, is_portrait={is_portrait}")
        seg_polys = []
        for seg in segmentation:
            poly = np.array(seg).reshape((-1, 2))
            poly[:, 0] /= width
            poly[:, 1] /= height
            if is_portrait:
                # rotate the polygon coordinates by 90 degrees clockwise
                poly = np.column_stack((poly[:, 1], 1 - poly[:, 0]))
                print(f"Rotated polygon for image {image_name} (portrait mode): {poly}")
            seg_polys.append(poly)
        annotations_parsed[ann_id] = {
            'image_name': image_name,
            'knot_id': int(knot_id),
            'segmentation': seg_polys
        }
    return annotations_parsed


def correct_annotations(annotations: dict, corrections: list):
    for correction in corrections:
        id1, id2 = correction
        knot_id1 = annotations[id1]['knot_id']
        knot_id2 = annotations[id2]['knot_id']
        for ann in annotations.values():
            if ann['knot_id'] == knot_id2:
                ann['knot_id'] = knot_id1


def construct_voxel_grid(annotations: list[dict], projections_dir: str, voxel_size: float = 0.1, save_masks: bool = False) -> VoxelGrid:
    image_ids = set([Path(ann['image_name']).stem for ann in annotations.values()])
    projections = {}
    for image_id in image_ids:
        projection_file = Path(projections_dir) / f"{image_id}.npy"
        if projection_file.exists():
            projections[image_id] = np.load(projection_file)
        else:
            print(f"Warning: Projection file {projection_file} does not exist.")
            projections[image_id] = None

    vg = VoxelGrid(voxel_size=voxel_size)
    for ann_id, ann in list(annotations.items()):
        image_id = Path(ann['image_name']).stem
        projection = projections[image_id]
        if projection is None:
            del annotations[ann_id]
            continue
        pos = projection[..., :3]
        valid_mask = projection[..., 3] > 0.5
        h, w = projection.shape[:2]
        seg_polys = ann['segmentation'] # in normalized coordinates
        mask = np.zeros((h, w), dtype=np.uint8)
        for seg in seg_polys:
            seg[:, 0] *= w
            seg[:, 1] *= h
            poly = np.round(seg).astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
        if save_masks:
            ann['mask'] = mask
        # print(f"Segmentation mask for ann_id {ann['ann_id']} has {np.sum(mask)} pixels.")
        combined_mask = valid_mask & (mask > 0)
        points = pos[combined_mask]
        if points.size == 0:
            print(f"Warning: No valid points found for annotation {ann_id} in image {image_id}.")
            continue
        # print(ann['ann_id'], ann['image_name'], points.shape, ann['image_id'])

        vg.add_volume(points, volume_id=ann_id, group_id=int(image_id))
    return vg

def confusion_matrix(annotations: dict[dict], structures: list[dict], return_pairs: bool = False) -> np.ndarray:
    # Create a mapping from annotation_id to structure_id
    ann_to_structure = {}
    for str_id, structure in enumerate(structures):
        for ann_id in structure:
            ann_to_structure[ann_id] = str_id

    # iterate over all pairs of annotations and count true positives, false positives, true negatives, and false negatives
    anns = []
    for ann_id, ann in annotations.items():
        ann['ann_id'] = ann_id
        anns.append(ann)

    tps, fps, tns, fns = [], [], [], []

    for i in range(len(anns)):
        for j in range(i + 1, len(anns)):
            ann_i = anns[i]
            ann_j = anns[j]
            ann_str_i = ann_to_structure.get(ann_i['ann_id'])
            ann_str_j = ann_to_structure.get(ann_j['ann_id'])
            same_knot = ann_i['knot_id'] == ann_j['knot_id']
            same_structure = (
                ann_str_i is not None and ann_str_j is not None
                and ann_str_i == ann_str_j
            )
            if same_knot and same_structure:
                tps.append((ann_i['ann_id'], ann_j['ann_id']))
            elif not same_knot and same_structure:
                fps.append((ann_i['ann_id'], ann_j['ann_id']))
            elif same_knot and not same_structure:
                fns.append((ann_i['ann_id'], ann_j['ann_id']))
            else:
                tns.append((ann_i['ann_id'], ann_j['ann_id']))
    if return_pairs:
        return tps, fps, fns, tns 
    return len(tps), len(fps), len(fns), len(tns)