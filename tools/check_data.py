'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-15 15:14:22
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import mmengine
import os
from tqdm import tqdm
import os.path as osp
import numpy as np
from collections import defaultdict, OrderedDict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def stat_data_wo_occ_path():
    """Because the original data has some missing occ_gt_final_path, we need to remove them.
    """
    for split in ['train', 'val']:
        val_pkl_fp = f"data/openscene-v1.1/openscene_mini_{split}.pkl"
        val_meta = mmengine.load(val_pkl_fp)
        print("split:", split, type(val_meta), len(val_meta))

        ## check the occ
        new_val_infos = []
        for info in val_meta:
            occ_gt_path = info['occ_gt_final_path']
            if occ_gt_path is None:
                continue
            new_val_infos.append(info)
        print(len(new_val_infos))



def check_occ():
    val_pkl_fp = "data/openscene-v1.1/openscene_mini_val.pkl"
    val_meta = mmengine.load(val_pkl_fp)
    print(type(val_meta), len(val_meta))

    ## check the occ
    element = val_meta[0]

    occ_gt_path = element['occ_gt_final_path']
    occ_gt = np.load(occ_gt_path)
    print(occ_gt.shape)
    print(np.unique(occ_gt[:, 1]))

    non_empty = occ_gt.shape[0]
    empty_num = 200 * 200 * 16 - non_empty
    print(f"Empty number: {empty_num}, non-empty number: {non_empty}")

    ## check all categories
    all_categories = set()
    for entry in tqdm(val_meta):
        occ_gt_path = entry['occ_gt_final_path']
        if occ_gt_path is None:
            continue

        occ_gt = np.load(occ_gt_path)
        category = np.unique(occ_gt[:, 1])
        all_categories.update(category.tolist())

    print(all_categories)


def check_private_wm_data():
    from projects.mmdet3d_plugin.datasets.pipelines.nuplan_loading import PointCloud

    private_test_fp = "data/openscene-v1.1/meta_datas/private_test_wm/private_test_wm.pkl"
    data_infos = mmengine.load(private_test_fp)
    print(type(data_infos), len(data_infos))

    entry = data_infos[0]
    print(entry.keys())

    lidar_path = entry['lidar_path']
    print(lidar_path)

    data_root = "data/openscene-v1.1/sensor_blobs/private_test_wm"
    
    front_cam_path_list = []
    for info in data_infos:
        lidar_path = info['lidar_path']
        occ_gt_path = info['occ_gt_final_path']

        pts_filename = osp.join(data_root, lidar_path)
        pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T

        front_cam_path = info['cams']['CAM_F0']['data_path']
        front_cam_path_list.append(front_cam_path)

        # for cam_type, cam_info in info['cams'].items():
        #     cam_info['data_path'] = osp.join(data_root, cam_info['data_path'])
        #     print(cam_info['data_path'])
    
    print(front_cam_path_list)


def rewrite_vidar_pred_pc():
    """We resave the vidar predicted point cloud by using the inside masking.
    """
    from projects.mmdet3d_plugin.datasets.pipelines.nuplan_loading import PointCloud
    from projects.mmdet3d_plugin.bevformer.utils import e2e_predictor_utils
    
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    pkl_fp = "data/openscene-v1.1/openscene_mini_val_v2.pkl"

    data_infos = mmengine.load(pkl_fp)
    print(type(data_infos), len(data_infos))
    print(data_infos[0].keys())

    our_data_root = "results/vidar_pred_pc"
    data_root = "data/openscene-v1.1/sensor_blobs/mini"

    save_root = "results/vidar_pred_pc_new"
    for idx, info in tqdm(enumerate(data_infos)):
        lidar_path = info['lidar_path']
        pts_filename = osp.join(data_root, lidar_path)
        pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T

        our_pc = osp.join(our_data_root, lidar_path + ".npz")
        our_pc = np.load(our_pc)['arr_0']

        gt_inside_mask = e2e_predictor_utils.get_inside_mask(our_pc, point_cloud_range)
        our_pc = our_pc[gt_inside_mask]
        
        save_path = os.path.join(save_root, lidar_path)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(save_path, our_pc)



def check_vidar_pred_pc():
    """Save the groud truth point cloud and the predicted point cloud for comparison.
    """
    from projects.mmdet3d_plugin.datasets.pipelines.nuplan_loading import PointCloud
    from projects.mmdet3d_plugin.bevformer.utils import e2e_predictor_utils
    
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    pkl_fp = "data/openscene-v1.1/openscene_mini_train.pkl"
    pkl_fp = "data/openscene-v1.1/openscene_mini_train_v2.pkl"
    # pkl_fp = "data/openscene-v1.1/openscene_mini_val_v2.pkl"

    data_infos = mmengine.load(pkl_fp)
    print(type(data_infos), len(data_infos))
    print(data_infos[0].keys())

    our_data_root = "results/vidar_pred_pc"
    our_data_root = "/data1/cyx/OpenScene/results/vidar_pred_pc_train"
    data_root = "data/openscene-v1.1/sensor_blobs/mini"
    for idx, info in enumerate(data_infos):
        lidar_path = info['lidar_path']
        pts_filename = osp.join(data_root, lidar_path)
        pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T

        our_pc = osp.join(our_data_root, lidar_path + ".npz")
        our_pc = np.load(our_pc)['arr_0']

        gt_inside_mask = e2e_predictor_utils.get_inside_mask(our_pc, point_cloud_range)
        our_pc = our_pc[gt_inside_mask]

        save_dir = "results/vidar_pred_compare_train"
        os.makedirs(save_dir, exist_ok=True)
        
        np.savetxt(f"{save_dir}/{idx:03d}_pc_gt.xyz", pc[:, :3])
        np.savetxt(f"{save_dir}/{idx:03d}_pc_ours.xyz", our_pc[:, :3])
        
        if idx > 10:
            break


def check_data_length():
    pkl_fp = "data/openscene-v1.1/openscene_mini_train.pkl"
    pkl_fp = "data/openscene-v1.1/openscene_mini_train_v2.pkl"
    # pkl_fp = "data/openscene-v1.1/openscene_mini_val_v2.pkl"

    data_infos = mmengine.load(pkl_fp)
    print(type(data_infos), len(data_infos))
    print(data_infos[0].keys())

    # convert the list of dict to dict
    data_infos_dict = defaultdict(list)
    for d in data_infos:
        data_infos_dict[d["scene_name"]].append(d)

    data_infos_dict = dict(data_infos_dict)
    print(len(data_infos_dict))

    num_frames_each_scene = [len(_scene) for _scene in data_infos_dict.values()]
    print(min(num_frames_each_scene), max(num_frames_each_scene))
    print(sorted(num_frames_each_scene))

    filtered_scene_names = []
    for key, value in data_infos_dict.items():
        # filter the scenes with less than 8 frames
        if len(value) < 8:
            continue
        filtered_scene_names.append(key)

    # only keep the first 1/4 for acceleration
    partial_filtered_scene_names = filtered_scene_names[:len(filtered_scene_names) // 4]
    filtered_data_infos = []
    for key in partial_filtered_scene_names:
        filtered_data_infos.extend(data_infos_dict[key])
    print(len(filtered_data_infos))


if __name__ == "__main__":
    check_vidar_pred_pc()
    # rewrite_vidar_pred_pc()
    exit()
    

    # check_data_length()
    # check_private_wm_data()



# private_test_fp = "data/openscene-v1.1/meta_datas/private_test_wm/private_test_wm.pkl"
# private_test_fp = "data/openscene-v1.1/openscene_mini_train.pkl"

# private_test_meta = mmengine.load(private_test_fp)
# print(type(private_test_meta), len(private_test_meta))
# print(private_test_meta[0].keys())
# exit()


# private_test_fp = "data/openscene-v1.1/meta_datas/private_test_wm/private_test_wm.pkl"
# private_test_meta = mmengine.load(private_test_fp)
# print(type(private_test_meta), len(private_test_meta))
# print(private_test_meta[0].keys())

# _meta = private_test_meta[0]
# print(_meta['frame_idx'])

# if 'pts_filename' in private_test_meta[0]:
#     print(private_test_meta[0]['pts_filename'])
# else:
#     print('No pts_filename in the meta file.')