import argparse
import mmcv
import os
import torch
import warnings
import tempfile
import torch.distributed as dist
import shutil
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type

import importlib
import time
import os.path as osp
from copy import deepcopy

from nuscenes.nuscenes import NuScenes
from PIL import Image

from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from matplotlib import rcParams
import matplotlib.pyplot as plt
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = 'render.png',
        extra_info: bool = False) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes = []
    select_cams = []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)
            # We found an image that matches. Let's abort.
    # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
    #                      'Try using e.g. BoxVisibility.ANY.'
    # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    num_cam = len(all_bboxes)

    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    select_cams = [sample_record['data'][cam] for cam in select_cams]
    print('bbox in cams:', select_cams)
    # Plot LIDAR view.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    # Plot CAMERA view.
    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(lidar_points),
                                  '# radar points: {0:>4}'.format(radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

        plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)

def get_sample_data(sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    selected_anntokens=None,
                    use_flat_vehicle_coordinates: bool = False):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def lidiar_render(sample_token, data,out_path=None):
    bbox_gt_list = []
    bbox_pred_list = []
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        try:
            bbox_gt_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=category_to_detection_name(content['category_name']),
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=''))
        except:
            pass

    bbox_anns = data['results'][sample_token]
    for content in bbox_anns:
        bbox_pred_list.append(DetectionBox(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content['attribute_name']))
    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)
    print('green is ground truth')
    print('blue is the predited result')
    visualize_sample(nusc, sample_token, gt_annotations, pred_annotations, savepath=out_path+'_bev')


def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
     'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
     'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
     'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
     'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
     'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
     'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
     'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    #print(category_name)
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def render_sample_data(
        sample_token: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
      ) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    lidiar_render(sample_token, pred_data, out_path=out_path)
    sample = nusc.get('sample', sample_token)
    # sample = data['results'][sample_token_list[0]][0]
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
    if ax is None:
        _, ax = plt.subplots(4, 3, figsize=(24, 18))
    j = 0
    for ind, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]

        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        # Load boxes and image.
        boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']), name=record['detection_name'], token='predicted') 
                 for record in pred_data['results'][sample_token] if record['detection_score'] > 0.2]

        data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token, box_vis_level=box_vis_level, pred_anns=boxes)
        
        _, boxes_gt, _ = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)
        
        if ind == 3:
            j += 1
        ind = ind % 3
        data = Image.open(data_path)

        # Show image.
        ax[j, ind].imshow(data)
        ax[j + 2, ind].imshow(data)

        # Show boxes.
        if with_anns:
            for box in boxes_pred:
                c = np.array(get_color(box.name)) / 255.0
                box.render(ax[j, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))
            for box in boxes_gt:
                c = np.array(get_color(box.name)) / 255.0
                box.render(ax[j + 2, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Limit visible range.
        ax[j, ind].set_xlim(0, data.size[0])
        ax[j, ind].set_ylim(data.size[1], 0)
        ax[j + 2, ind].set_xlim(0, data.size[0])
        ax[j + 2, ind].set_ylim(data.size[1], 0)

        ax[j, ind].axis('off')
        ax[j, ind].set_title('PRED: {} {labels_type}'.format(
            sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax[j, ind].set_aspect('equal')

        ax[j + 2, ind].axis('off')
        ax[j + 2, ind].set_title('GT:{} {labels_type}'.format(
            sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax[j + 2, ind].set_aspect('equal')

    if out_path is not None:
        plt.savefig(out_path+'_camera', bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()
    plt.close()
    
def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic

def get_quaternion_from_euler(e):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    roll = e[0]
    pitch = e[1]
    yaw = e[2]

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

def euler_from_quaternion(q):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    import math
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = math.atan2(t0, t1) / np.pi * 180 # degrees
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    # pitch_y = math.asin(t2) / np.pi * 180
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z) 
    # yaw_z = math.atan2(t3, t4) / np.pi * 180
    yaw_z = math.atan2(t3, t4)

    return [roll_x, pitch_y, yaw_z] # in radian

rank, world_size = get_dist_info()

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '2904'

# initialize the process group
dist.init_process_group("gloo", rank=rank, world_size=world_size)

config = './projects/configs/bevformer/bevformer_base.py'
ckpt = '../BEVFormer/ckpts/bevformer_r101_dcn_24ep.pth'

device = 'cuda:0'

cfg = Config.fromfile(config)

plugin_dir = cfg.plugin_dir
_module_dir = os.path.dirname(plugin_dir)
_module_dir = _module_dir.split('/')
_module_path = _module_dir[0]

for m in _module_dir[1:]:
    _module_path = _module_path + '.' + m
print(_module_path)
plg_lib = importlib.import_module(_module_path)

# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

cfg.model.pretrained = None

samples_per_gpu = 1
cfg.data.test.test_mode = True
samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)

# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, ckpt, map_location='cpu')

model.CLASSES = checkpoint['meta']['CLASSES']
model.PALETTE = checkpoint['meta']['PALETTE']

# Built model to single CUDA device
if device is not 'cpu':
    torch.cuda.set_device(device)
model.to(device)
model.eval()

dataset = build_dataset(cfg.data.test)

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    # dist=False,
    dist=True,
    shuffle=False,
    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
)

nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

# Change Data Input

"""Dataset 1. FOV (120, 120), Resolution 1000x1000"""
# tangent_patch_size = (1000, 1000)
# cur_frame = 'frame_0180.jpg'

# cur_filenames = ['./data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/3_front/'+cur_frame, 
#                  './data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/4_front_right/'+cur_frame, 
#                  './data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/2_front_left/'+cur_frame, 
#                  './data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/0_back/'+cur_frame,
#                  './data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/1_back_left/'+cur_frame, 
#                  './data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/5_back_right/'+cur_frame]

# tangent_cams = ['./data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/2_front_left/'+cur_frame, 
#                 './data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/3_front/'+cur_frame, 
#                 './data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/4_front_right/'+cur_frame, 
#                 './data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/1_back_left/'+cur_frame,
#                 './data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/0_back/'+cur_frame, 
#                 './data/simple_road_outdoor_oct_23/tangent_patches_120fov_1000x1000/5_back_right/'+cur_frame]

# tangent_intrinsics = [[[474.43093925,0.,491.31570912],[0.,467.96789827,496.66222459],[0.,0.,1.]], # 2_front_left
#                       [[461.48071582,0.,485.52727853],[0.,456.69616035,506.9048758],[0.,0.,1.]], # 3_front
#                       [[648.91137487,0.,491.25631262],[0.,466.39633743,503.55409406],[0.,0.,1.]], # 4_front_right
#                       [[379.97547954,0.,474.94269824],[0.,399.33335026,448.14651339],[0.,0.,1.]], # 1_back_left
#                       [[473.91432417,0.,502.09118734],[0.,470.17266741,493.76922778], [0.,0.,1.]], # 0_back
#                       [[476.58560896,0.,489.02737013],[0.,474.43221528,497.28835466],[0.,0.,1.]]] # 5_back_right

"""Dataset 2. FOV (70, 70), Resolution 800x800"""
# tangent_patch_size = (800, 800)
# cur_frame = 'frame_0180.jpg'

# # Input
# cur_filenames = ['./data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/3_front/'+cur_frame, 
#                  './data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/4_front_right/'+cur_frame, 
#                  './data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/2_front_left/'+cur_frame, 
#                  './data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/0_back/'+cur_frame,
#                  './data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/1_back_left/'+cur_frame, 
#                  './data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/5_back_right/'+cur_frame]
# # Visualization
# tangent_cams = ['./data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/2_front_left/'+cur_frame, 
#                 './data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/3_front/'+cur_frame, 
#                 './data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/4_front_right/'+cur_frame, 
#                 './data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/1_back_left/'+cur_frame,
#                 './data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/0_back/'+cur_frame, 
#                 './data/simple_road_outdoor_oct_23/tangent_patches_70fov_800x800/5_back_right/'+cur_frame]

# tangent_intrinsics = [[[656.24536467,0.,375.85853477],[0.,650.02567023,387.47207595],[0.,0.,1.]], # 2_front_left
#                       [[642.05847106,0.,386.48649006],[0.,634.39687303,389.69815667],[0.,0.,1.]], # 3_front
#                       [[495.57800473,0.,406.87866641],[0.,601.676778,302.86860227],[0.,0.,1.]], # 4_front_right
#                       [[518.16766045,0.,376.45885035],[0.,545.74779584,316.64273454],[0.,0.,1.]], # 1_back_left
#                       [[662.32537306,0.,397.04486038],[0.,656.71136718,398.44495815], [0.,0.,1.]], # 0_back
#                       [[641.31968424,0.,394.54743594],[0.,638.77832524,390.13875586],[0.,0.,1.]]] # 5_back_right

"""Dataset 2. FOV (70, 70), Resolution 800x800"""
tangent_patch_size = (1600, 900)
cur_frame = 'frame_0001.jpg'

# Input
cur_filenames = ['./data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/3_front/'+cur_frame, 
                 './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/4_front_right/'+cur_frame, 
                 './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/2_front_left/'+cur_frame, 
                 './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/0_back/'+cur_frame,
                 './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/1_back_left/'+cur_frame, 
                 './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/5_back_right/'+cur_frame]
# Visualization
tangent_cams = ['./data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/2_front_left/'+cur_frame, 
                './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/3_front/'+cur_frame, 
                './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/4_front_right/'+cur_frame, 
                './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/1_back_left/'+cur_frame,
                './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/0_back/'+cur_frame, 
                './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/5_back_right/'+cur_frame]

tangent_intrinsics_for_vis = [[[1326.72632, 0.0, 789.918136], [0.0, 1313.8785, 447.051964], [0.0, 0.0, 1.0]], # 2_front_left
                      [[1343.45019, 0.0, 820.183159], [0.0, 1280.23476, 442.850375], [0.0, 0.0, 1.0]], # 3_front
                      [[1318.58226, 0.0, 748.797979], [0.0, 1307.51676, 433.683573], [0.0, 0.0, 1.0]], # 4_front_right
                      [[1329.78802, 0.0, 794.247861], [0.0, 1318.65546, 422.083681], [0.0, 0.0, 1.0]], # 1_back_left
                      [[1318.10344, 0.0, 760.164664], [0.0, 1307.00893, 433.459504], [0.0, 0.0, 1.0]], # 0_back
                      [[1342.27544, 0.0, 790.251605], [0.0, 1326.28658, 452.853747], [0.0, 0.0, 1.0]]] # 5_back_right

tangent_intrinsics = {'CAM_FRONT': [[1343.45019, 0.0, 820.183159], [0.0, 1280.23476, 442.850375], [0.0, 0.0, 1.0]],
                      'CAM_FRONT_RIGHT': [[1318.58226, 0.0, 748.797979], [0.0, 1307.51676, 433.683573], [0.0, 0.0, 1.0]],
                      'CAM_FRONT_LEFT': [[1326.72632, 0.0, 789.918136], [0.0, 1313.8785, 447.051964], [0.0, 0.0, 1.0]],
                      'CAM_BACK': [[1318.10344, 0.0, 760.164664], [0.0, 1307.00893, 433.459504], [0.0, 0.0, 1.0]],
                      'CAM_BACK_LEFT': [[1329.78802, 0.0, 794.247861], [0.0, 1318.65546, 422.083681], [0.0, 0.0, 1.0]],
                      'CAM_BACK_RIGHT': [[1342.27544, 0.0, 790.251605], [0.0, 1326.28658, 452.853747], [0.0, 0.0, 1.0]]}

"""sample road scene video"""
# cur_frame = '0000630.png'
# cur_filenames = ['./data/insta360_sample_video/raw_data/tangent_images/3_front/'+cur_frame, './data/insta360_sample_video/raw_data/tangent_images/4_front_right/'+cur_frame, 
#              './data/insta360_sample_video/raw_data/tangent_images/2_front_left/'+cur_frame, './data/insta360_sample_video/raw_data/tangent_images/0_back/'+cur_frame,
#             './data/insta360_sample_video/raw_data/tangent_images/1_back_left/'+cur_frame, './data/insta360_sample_video/raw_data/tangent_images/5_back_right/'+cur_frame]

# tangent_cams = ['./data/insta360_sample_video/raw_data/tangent_images/2_front_left/'+cur_frame, './data/insta360_sample_video/raw_data/tangent_images/3_front/'+cur_frame, 
#              './data/insta360_sample_video/raw_data/tangent_images/4_front_right/'+cur_frame, './data/insta360_sample_video/raw_data/tangent_images/1_back_left/'+cur_frame,
#             './data/insta360_sample_video/raw_data/tangent_images/0_back/'+cur_frame, './data/insta360_sample_video/raw_data/tangent_images/5_back_right/'+cur_frame]


"""lidar2img 변경"""
train_nusc_infos = []
val_nusc_infos = []
frame_idx = 0

sample = nusc.sample[0]

lidar_token = sample['data']['LIDAR_TOP']
sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

mmcv.check_file_exist(lidar_path)

info = {
    'lidar_path': lidar_path,
    'token': sample['token'],
    'prev': sample['prev'],
    'next': sample['next'],
    'frame_idx': frame_idx,  # temporal related info
    'sweeps': [],
    'cams': dict(),
    'scene_token': sample['scene_token'],  # temporal related info
    'lidar2ego_translation': cs_record['translation'],
    'lidar2ego_rotation': cs_record['rotation'],
    'ego2global_translation': pose_record['translation'],
    'ego2global_rotation': pose_record['rotation'],
    'timestamp': sample['timestamp'],
}

if sample['next'] == '':
        frame_idx = 0
else:
    frame_idx += 1

l2e_r = info['lidar2ego_rotation']
l2e_t = info['lidar2ego_translation']
e2g_r = info['ego2global_rotation']
e2g_t = info['ego2global_translation']
l2e_r_mat = Quaternion(l2e_r).rotation_matrix # 3x3
e2g_r_mat = Quaternion(e2g_r).rotation_matrix

# obtain 6 image's information per frame
camera_types = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

temp_sensor2lidar_rotations = []
temp_sensor2ego_rotations = []
temp_sensor2ego_translations = []

for cam in camera_types:
    cam_token = sample['data'][cam]
    cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)

    # cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
    sd_rec = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    cam_info = {
        'data_path': data_path,
        'type': cam_token,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = cam_info['sensor2ego_rotation']
    l2e_t_s = cam_info['sensor2ego_translation']
    
    """ego2global (pose) 변경"""
    e2g_r_s = cam_info['ego2global_rotation']
    e2g_t_s = cam_info['ego2global_translation']
    # print(e2g_r_s)
    # print(e2g_t_s)
    # radians = euler_from_quaternion(Quaternion(e2g_r_s).inverse)
    # print([euler / np.pi * 180 for euler in radians])

    # obtain the RT from sensor to Top LiDAR
    # cam->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    
    temp_sensor2ego_translations.append(l2e_t_s)
    temp_sensor2ego_rotations.append(l2e_r_s_mat)
    
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    cam_info['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    cam_info['sensor2lidar_translation'] = T

    temp_sensor2lidar_rotations.append(cam_info['sensor2lidar_rotation'])
    
    cam_info.update(cam_intrinsic=cam_intrinsic)
    info['cams'].update({cam: cam_info})
    
"""Sensor2lidar rotation degrees 확인"""
for mat in temp_sensor2lidar_rotations:
    rot_q = Quaternion(matrix=mat)
    eulers = euler_from_quaternion(Quaternion(rot_q))
    # print("matrix", mat)
    # print("quaternion", rot_q)
    # print("euler in radian", eulers)
    # print("euler in degree", [euler / np.pi * 180 for euler in eulers])
    # print()

"""sensor2lidar rotation degree를 insta360에 맞게 변경"""
sensor2lidar_euler_degrees = [[-88.89452145820958, -0.34991764317283675, 0.0], # FRONT
                              [-89.73833720738743, 1.5337073678165962, -60.0], # FRONT_RIGHT
                              [-89.2717432442176, -1.221029004801636, 60.0], # FRONT_LEFT
                              [-90.4448247670561, 0.552239989680819, 180.0], # BACK
                              [-91.67728722938543, -1.4284019448799024, 120.0], # BACK_LFET
                              [-91.13862482181912, 2.0544464501438036, -120.0]] # BACK_RIGHT

# degree를 r matrix로 변경해야함
sensor2lidar_rotations = [] # tcam2egocam_rotations
for sensor2lidar_degrees in sensor2lidar_euler_degrees: # sensor2lidar == tangent cam 2 ego cam in Insta360
    sensor2lidar_radians = [degree * np.pi / 180 for degree in sensor2lidar_degrees]
    sensor2lidar_q = Quaternion(get_quaternion_from_euler(sensor2lidar_radians))
    sensor2lidar_r_mat = sensor2lidar_q.rotation_matrix
    sensor2lidar_rotations.append(sensor2lidar_r_mat)
    
    # 확인
    # radians = euler_from_quaternion(sensor2lidar_q)
    # print([euler / np.pi * 180 for euler in radians])

i = 0
lidar2img_rts = []
for cam_type, cam_info in info['cams'].items():
    # obtain lidar to image transformation matrix
    
    """rotation insta360에 맞게 변경"""
    # lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    lidar2cam_r = np.linalg.inv(sensor2lidar_rotations[i])
    
    """translation insta360에 맞게 변경"""
    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
    # lidar2cam_t = [0.0, 0.0, 1.59] @ lidar2cam_r.T
    
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    
    """cam intrinsic insta360에 맞게 변경"""
    # intrinsic = cam_info['cam_intrinsic']
    intrinsic = np.array(tangent_intrinsics[cam_type])
    
    viewpad = np.eye(4)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    lidar2img_rts.append(lidar2img_rt)
    i = i + 1
    
"""2. sensor2ego rotations 변경"""

"""Sensor2ego rotation degrees 확인"""
for mat in temp_sensor2ego_rotations:
    rot_q = Quaternion(matrix=mat)
    eulers = euler_from_quaternion(Quaternion(rot_q))
    # print("matrix", mat)
    # print("quaternion", rot_q)
    # print("euler in radian", eulers)
    # print([euler / np.pi * 180 for euler in eulers])
    # print()
    
camera_types = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

"""sensor2ego rotation degree를 insta360에 맞게 변경"""
sensor2ego_rotation_degrees = [[-90.32322642770005, -0.046127194838589326, -90.0], # FRONT 기준으로
                               [-90.7820235885, 0.5188438566959973, -150.0], # FRONT RIGHT
                               [-89.85977500319999, 0.12143609391200436, -30.0], # FRONT LEFT
                               [-89.0405962694, 0.22919685786400154, 90.0], # BACK
                               [-90.91736319750001, -0.21518275753700122, 30.0], # BACK_LFET
                               [-90.93206677999999, 0.6190947610589966, -210.0]] # BACK_RIGHT
i = 0
sensor2ego_rotations = {} # tcam2egocam_rotations
for sensor2ego_degrees in sensor2ego_rotation_degrees: # sensor2ego == tangent cam 2 ego cam in Insta360
    sensor2ego_radians = [degree * np.pi / 180 for degree in sensor2ego_degrees]
    sensor2ego_q = Quaternion(get_quaternion_from_euler(sensor2ego_radians))
    sensor2ego_r_mat = sensor2ego_q.rotation_matrix
    
    sensor2ego_rotations[camera_types[i]] = sensor2ego_r_mat
    i = i + 1


    
    
    
    
    
    

images = glob.glob('./data/simple_road_outdoor_oct_23/erp/*.jpg')
images = sorted(images)

for image in images:
    
    cur_frame = os.path.basename(image)
    print(cur_frame)
    
    # Input
    cur_filenames = ['./data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/3_front/'+cur_frame, 
                     './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/4_front_right/'+cur_frame, 
                     './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/2_front_left/'+cur_frame, 
                     './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/0_back/'+cur_frame,
                     './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/1_back_left/'+cur_frame, 
                     './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/5_back_right/'+cur_frame]
    # Visualization
    tangent_cams = ['./data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/2_front_left/'+cur_frame, 
                    './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/3_front/'+cur_frame, 
                    './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/4_front_right/'+cur_frame, 
                    './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/1_back_left/'+cur_frame,
                    './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/0_back/'+cur_frame, 
                    './data/simple_road_outdoor_oct_23/tangent_patches_70x39fov_1600x900/5_back_right/'+cur_frame]
    
    data_iterator = iter(data_loader)
    model.prev_frame_info['prev_bev'] = None # 첫 inference로 세팅

    data = next(data_iterator)
    data['img_metas'] = data['img_metas'][0].data
    data['img'] = data['img'][0].data

    results = {'img_filename': cur_filenames}

    test_pipeline = cfg.data.test.pipeline
    test_pipeline[3]['img_scale'] = tangent_patch_size
    test_pipeline = Compose(test_pipeline)

    cdata = test_pipeline(results)
    cdata['img_metas'] = cdata['img_metas'][0].data
    cdata['img'] = cdata['img'][0].data

    # Data loader에서 얻은 하나의 data를 custom data로 변경

    # img_metas 변경
    data['img_metas'][0][0]['filename'] = cdata['img_metas']['filename']
    data['img_metas'][0][0]['ori_shape'] = cdata['img_metas']['ori_shape']
    data['img_metas'][0][0]['img_shape'] = cdata['img_metas']['img_shape']
    data['img_metas'][0][0]['pad_shape'] = cdata['img_metas']['pad_shape']
    data['img_metas'][0][0]['scale_factor'] = cdata['img_metas']['scale_factor']
    data['img_metas'][0][0]['flip'] = cdata['img_metas']['flip']
    data['img_metas'][0][0]['pcd_horizontal_flip'] = cdata['img_metas']['pcd_horizontal_flip']
    data['img_metas'][0][0]['pcd_vertical_flip'] = cdata['img_metas']['pcd_vertical_flip']
    data['img_metas'][0][0]['img_norm_cfg'] = cdata['img_metas']['img_norm_cfg']
    data['img_metas'][0][0]['pcd_scale_factor'] = cdata['img_metas']['pcd_scale_factor']

    # img tensors 변경
    data['img'] = [cdata['img'][None, :]]
    
    # lidar2img 변경 (array 순서: front, front_right, front_left, back, back_left, back_right)
    # data['img_metas'][0][0]['lidar2img'] = ego2patches
    data['img_metas'][0][0]['lidar2img'] = lidar2img_rts

    """@@@@@@@@ TODO @@@@@@@@"""
    # TODO: pose 및 frame sequence (prev, next) 관련
    
    
    
    
    
    
    
    """Can_bus 변경 임시"""
    rotation = Quaternion([0.999514282, -0.009492505, -0.029651314, 0.001394546])

    """nuscenes: x가 차 전진 방향, y는 차량 좌우 방향, z는 차 위아래 방향"""
    # translation = [-0.008222580, -0.006377218, 0.040763527] # 원래
    translation = [0.040763527, -0.008222580, -0.006377218]

    data['img_metas'][0][0]['can_bus'][:3] = translation
    data['img_metas'][0][0]['can_bus'][3:7] = rotation

    patch_angle = quaternion_yaw(rotation) / np.pi * 180 # we get the yaw angle of ego car
    if patch_angle < 0:
        patch_angle += 360
    data['img_metas'][0][0]['can_bus'][-2] = patch_angle / 180 * np.pi # this angle is kept unchanged.
    data['img_metas'][0][0]['can_bus'][-1] = patch_angle # this angle is used to compute the detal of adjacent timestamps.
    
    
    
    
    
    

    bbox_results = []
    mask_results = []
    rank, world_size = get_dist_info()
    have_mask = False

    with torch.no_grad():

        data['img'][0] = data['img'][0].to(device)
        result = model(return_loss=False, rescale=True, **data)

        # encode mask results
        if isinstance(result, dict):
            if 'bbox_results' in result.keys():
                bbox_result = result['bbox_results']
                batch_size = len(result['bbox_results'])
                bbox_results.extend(bbox_result)
            if 'mask_results' in result.keys() and result['mask_results'] is not None:
                mask_result = custom_encode_mask_results(result['mask_results'])
                mask_results.extend(mask_result)
                have_mask = True
        else:
            batch_size = len(result)
            bbox_results.extend(result)

    # Collect results
    MAX_LEN = 512
    size = len(dataset)

    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8, device=device)
    if rank == 0:
        mmcv.mkdir_or_exist('.dist_test')
        tmpdir = tempfile.mkdtemp(dir='.dist_test')
        tmpdir = torch.tensor(
            bytearray(tmpdir.encode()), dtype=torch.uint8, device=device)
        dir_tensor[:len(tmpdir)] = tmpdir

    dist.broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()

    # dump the part result to the dir
    mmcv.dump(bbox_results, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    new_bbox_results = []

    # collect all parts
    if rank == 0:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))

        # sort the results
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:
            new_bbox_results.extend(list(res))
        # the dataloader may pad some samples
        new_bbox_results = new_bbox_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)

    # return
    outputs = {'bbox_results': new_bbox_results, 'mask_results': None}

    
    
    
    
    
    
    
    
    
    
    # Write Results into File
    mmcv.dump(outputs['bbox_results'], './output.pkl')
    jsonfile_prefix = osp.join('simple_road_scene_results', config.split('/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
    result_files = dict()

    # results_ = [out['pts_bbox'] for out in outputs['bbox_results']]
    # tmp_file_ = osp.join(jsonfile_prefix, name)
    # for name in outputs['bbox_results'][0]:
    #     results_ = [out[name] for out in outputs['bbox_results']]
    #     tmp_file_ = osp.join(jsonfile_prefix, name)
    #     result_files.update({name: dataset._format_bbox(results_, tmp_file_)})

    # 변경
    """CUSTOM FORMAT BBOX START"""

    results = [out['pts_bbox'] for out in outputs['bbox_results']]
    jsonfile_prefix_tmp = osp.join(jsonfile_prefix, 'pts_bbox')

    nusc_annos = {}
    mapped_class_names = dataset.CLASSES

    print('Start to convert detection format...')
    for sample_id, detection in enumerate(mmcv.track_iter_progress(results)):
        annos = []

        """START FUNC output_to_nusc_box(det, with_velocity)"""
        # boxes = output_to_nusc_box(det, True)
        box3d = detection['boxes_3d']
        scores = detection['scores_3d'].numpy()
        labels = detection['labels_3d'].numpy()

        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        box_yaw = box3d.yaw.numpy()

        # our LiDAR coordinate system -> nuScenes box coordinate system
        nus_box_dims = box_dims[:, [1, 0, 2]]

        boxes_lidar = []
        for i in range(len(box3d)):
            quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            if dataset.with_velocity:
                velocity = (*box3d.tensor[i, 7:9], 0.0)
            else:
                velocity = (0, 0, 0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (
            # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            boxes_lidar.append(box)
        """END FUNC output_to_nusc_box(det, with_velocity)"""

        sample_token = dataset.data_infos[sample_id]['token']

        """START FUNC lidar_nusc_box_to_global(): Convert the box from ego to global coordinate """
        # boxes = lidar_nusc_box_to_global(dataset.data_infos[sample_id], boxes, mapped_class_names, dataset.eval_detection_configs, dataset.eval_version)
        info = dataset.data_infos[sample_id]
        classes = mapped_class_names
        eval_configs = dataset.eval_detection_configs
        eval_version = dataset.eval_version

        boxes = []
        for box in boxes_lidar:

            # Move box to ego vehicle coord system
            box.rotate(Quaternion(info['lidar2ego_rotation']))
            box.translate(np.array(info['lidar2ego_translation']))

            # lidar2ego_radians = [degree * np.pi / 180 for degree in [0.0, 0.0, -89.8835]]
            # lidar2ego_q = Quaternion(get_quaternion_from_euler(lidar2ego_radians))
            # box.rotate(lidar2ego_q)
            # box.translate(np.array(info['lidar2ego_translation']))

            # print(np.array(info['lidar2ego_translation']))
            # radians = euler_from_quaternion(lidar2ego_q)
            # print([euler / np.pi * 180 for euler in radians])

            # filter det in ego.
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue

            # Move box to global coord system
            # box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
            # box.translate(np.array(info['ego2global_translation']))

            """Ego pose 변경"""
            rotation = [0.999514282, -0.009492505, -0.029651314, 0.001394546]
            translation = [0.040763527, -0.008222580, -0.006377218]
            box.rotate(pyquaternion.Quaternion(rotation))
            box.translate(np.array(translation))

            boxes.append(box)

        """END FUNC lidar_nusc_box_to_global()"""

        for i, box in enumerate(boxes):
            name = mapped_class_names[box.label]
            if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                if name in [
                        'car',
                        'construction_vehicle',
                        'bus',
                        'truck',
                        'trailer',
                ]:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = NuScenesDataset.DefaultAttribute[name]
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = NuScenesDataset.DefaultAttribute[name]

            nusc_anno = dict(
                sample_token=sample_token,
                translation=box.center.tolist(),
                size=box.wlh.tolist(),
                rotation=box.orientation.elements.tolist(),
                velocity=box.velocity[:2].tolist(),
                detection_name=name,
                detection_score=box.score,
                attribute_name=attr)
            annos.append(nusc_anno)
        nusc_annos[sample_token] = annos
    nusc_submissions = {
        'meta': dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=True),
        'results': nusc_annos,
    }

    mmcv.mkdir_or_exist(jsonfile_prefix_tmp)
    res_path = osp.join(jsonfile_prefix_tmp, 'results_nusc.json')
    print('Results writes to', res_path)
    mmcv.dump(nusc_submissions, res_path)
    """CUSTOM FORMAT BBOX END"""

    result_files.update({name: res_path})
    bevformer_results = mmcv.load(jsonfile_prefix+'/pts_bbox/results_nusc.json')

    
    
    
    
    
    
    
    
    
    
    
    
    
    # render_annotation('7603b030b42a4b1caa8c443ccc1a7d52')
    sample_token_list = list(bevformer_results['results'].keys())
    # for id in range(0, 1):

    pred_data=bevformer_results

    id = 0
    sample_token=sample_token_list[id]
    out_path=sample_token_list[id]
    use_flat_vehicle_coordinates = False
    score_threshold = 0.3
    with_anns = True
    box_vis_level = BoxVisibility.ANY
    axes_limit = 40
    ax=None
    nsweeps = 1
    underlay_map = True
    show_lidarseg = False
    show_lidarseg_legend = False
    filter_lidarseg_labels=None
    lidarseg_preds_bin_path = None
    verbose = True
    show_panoptic = False

    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]

    sample = nusc.get('sample', sample_token)

    if ax is None:
        _, ax = plt.subplots(2, 3, figsize=(24, 10))
        # _, ax = plt.subplots(4, 3, figsize=(24, 18))

    j = 0
    for ind, cam in enumerate(cams):
        print("@@@@@@@@@@@ CAM: ", cam)
        sample_data_token = sample['data'][cam]

        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        # Load boxes and image.
        boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']), name=record['detection_name'], token='predicted') 
                 for record in pred_data['results'][sample_token] if record['detection_score'] > score_threshold]

        # print("Predicted boxes", boxes)

        """
        get predicted data - START

        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """
        # Retrieve sensor & pose records
        sd_record = nusc.get('sample_data', sample_data_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        data_path = nusc.get_sample_data_path(sample_data_token)
        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])

        # nusc => tangent cam으로 변경
        data_path = tangent_cams[ind]
        camera_intrinsic = np.array(tangent_intrinsics_for_vis[ind])
        imsize = tangent_patch_size

        # print("data path", data_path)
        # print("cam instrinsic", camera_intrinsic)
        # print("img size", imsize)

        # Make list of Box objects including coord system transforms.
        boxes_pred = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                # print("Current box before translation", box)

                """기존 nuscenes"""
                """
                # Move box to ego vehicle coord system. (pose record: ego2global)
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                # print("global2ego", -np.array(pose_record['translation']), Quaternion(pose_record['rotation']).inverse)
                # print("Current box after moving to ego coord system", box)

                #  Move box to sensor coord system. (cs_record: sensor2ego)
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

                # print("ego2sensor", -np.array(cs_record['translation']), Quaternion(cs_record['rotation']).inverse)
                # print("Current box after moving to camera coord system", box)
                # print()
                """

                """insta360 방안 1"""
                # Move box to ego vehicle coord system. (pose record: ego2global)
                # ORB_SLAM이 ego2global인지 확인 (+/- 바꿔보기)

                # box.translate(-np.array(translation))
                # box.rotate(rotation.inverse)

                #  Move box to sensor coord system. 
                # nuscenes에서는 cs_record: sensor2ego, insta360에서는 egocam2tangentcam으로 R 만 존재
                # box.rotate(front2cam_rotvecs[cam])

                # print("insta360")
                # print("global2ego", -np.array([-0.062425584, -0.002085830, 0.344779789]), Quaternion([0.998749077, -0.013452108, -0.043297615, 0.021088535]).inverse)
                # print()

                """insta360 방안 2-1. nuscenes에서 sensor2ego rotation 변경"""
                """
                # Move box to ego vehicle coord system. (pose record: ego2global)
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system. (cs_record: sensor2ego)
                box.translate(-np.array(cs_record['translation']))
                # box.rotate(Quaternion(cs_record['rotation']).inverse)
                box.rotate(Quaternion(matrix=sensor2ego_rotations[cam]).inverse)
                """

                """insta360 방안 2-2. nuscenes에서 sensor2ego rotation 및 translation 변경"""

                """ego2global (pose) 변경"""
                # 1. Move box to ego vehicle coord system. (pose record: ego2global)
                # box.translate(-np.array(pose_record['translation']))
                # box.rotate(Quaternion(pose_record['rotation']).inverse)

                # ORB-SLAM3 (ego2global인지 global2ego인지?)
                rotation = [0.999514282, -0.009492505, -0.029651314, 0.001394546]
                translation = [0.040763527, -0.008222580, -0.006377218]

                box.translate(-np.array(translation))
                box.rotate(Quaternion(rotation).inverse)

                # print(np.array(pose_record['translation']))
                # print(Quaternion(pose_record['rotation']))
                # radians = euler_from_quaternion(Quaternion(rotation).inverse)
                # print([euler / np.pi * 180 for euler in radians])

                # 2. Move box to sensor coord system. (cs_record: sensor2ego)
                box.translate(-np.array(cs_record['translation']))
                # box.translate(-np.array([0.0, 0.0, 0.0]))

                # box.rotate(Quaternion(cs_record['rotation']).inverse)
                box.rotate(Quaternion(matrix=sensor2ego_rotations[cam]).inverse)


            if not box_in_image(box, camera_intrinsic, imsize, vis_level=box_vis_level):
                continue

            boxes_pred.append(box)

        """get predicted data - END"""

        print("Boxes in camera viewport", boxes_pred)
        print()

        if ind == 3:
            j += 1
        ind = ind % 3
        img_data = Image.open(data_path)

        # Show image.
        ax[j, ind].imshow(img_data)

        # Show boxes.
        if with_anns:
            for box in boxes_pred:
                c = np.array(get_color(box.name)) / 255.0
                box.render(ax[j, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Limit visible range.
        ax[j, ind].set_xlim(0, img_data.size[0])
        ax[j, ind].set_ylim(img_data.size[1], 0)

        ax[j, ind].axis('off')
        ax[j, ind].set_title('PRED: {} {labels_type}'.format(
            sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax[j, ind].set_aspect('equal')


    plt.savefig('./pred_results/' + cur_frame, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()
    plt.close()

    
    
    
    
    
    
    
    
    

# Save to video
os.system("ffmpeg -i pred_results/frame_%04d.jpg -vcodec libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 24 -y -an video.mp4")