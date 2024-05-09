'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-23 15:04:44
Email: haimingzhang@link.cuhk.edu.cn
Description: The XWorld occupancy world model.
'''
#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import mmcv
import os
import torch
from torch import nn
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
import copy
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.bevformer.dense_heads.mimo_modules import preprocess
from .bevformer import BEVFormer
from mmdet3d.models import builder
from ..utils import e2e_predictor_utils, eval_utils
from .vidar_eval import occ_to_voxel


@DETECTORS.register_module()
class ViDARXWorld(BEVFormer):
    def __init__(self,
                 history_len,
                 # Future predictions.
                 future_pred_head,
                 future_pred_frame_num,  # number of future prediction frames.
                 test_future_frame_num,  # number of future prediction frames when testing.

                 # BEV configurations.
                 point_cloud_range,
                 bev_h,
                 bev_w,

                 # Augmentations.
                 # A1. randomly drop current image (to enhance temporal feature.)
                 random_drop_image_rate=0.0,
                 # A2. add noise to previous_bev_queue.
                 random_drop_prev_rate=0.0,
                 random_drop_prev_start_idx=1,
                 random_drop_prev_end_idx=None,
                 # A3. grid mask augmentation.
                 grid_mask_image=True,
                 grid_mask_backbone_feat=False,
                 grid_mask_fpn_feat=False,
                 grid_mask_prev=False,
                 grid_mask_cfg=dict(
                     use_h=True,
                     use_w=True,
                     rotate=1,
                     offset=False,
                     ratio=0.5,
                     mode=1,
                     prob=0.7
                 ),

                 # Supervision.
                 supervise_all_future=True,

                 # Visualize point cloud.
                 _viz_pcd_flag=False,
                 _viz_pcd_path='dbg/pred_pcd',  # root/{prefix}

                 # Test server submission.
                 _submission=False,  # Flags for submission.
                 _submission_path='submission/model',  # root/{prefix}
                
                 # XWorld parameters
                 expansion=8,
                 num_classes=12,
                 patch_size=2,

                 *args,
                 **kwargs,):

        super().__init__(*args, **kwargs)
        self.future_pred_head = builder.build_head(future_pred_head)
        self.future_pred_frame_num = future_pred_frame_num
        self.test_future_frame_num = test_future_frame_num
        # if not predict any future,
        #  then only predict current frame.
        self.only_train_cur_frame = (future_pred_frame_num == 0)

        self.history_len = history_len

        self.point_cloud_range = point_cloud_range
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.expansion = expansion
        self.class_embeds = nn.Embedding(num_classes, expansion)
        self.patch_size = patch_size

        out_dim = 32
        self.predicter = nn.Sequential(
                nn.Linear(expansion, out_dim*2),
                nn.Softplus(),
                nn.Linear(out_dim*2, 1),
            )

        # Augmentations.
        self.random_drop_image_rate = random_drop_image_rate
        self.random_drop_prev_rate = random_drop_prev_rate
        self.random_drop_prev_start_idx = random_drop_prev_start_idx
        self.random_drop_prev_end_idx = random_drop_prev_end_idx

        # Grid mask.
        self.grid_mask_image = grid_mask_image
        self.grid_mask_backbone_feat = grid_mask_backbone_feat
        self.grid_mask_fpn_feat = grid_mask_fpn_feat
        self.grid_mask_prev = grid_mask_prev
        self.grid_mask = GridMask(**grid_mask_cfg)

        # Training configurations.
        # randomly sample one future for loss computation?
        self.supervise_all_future = supervise_all_future

        self._viz_pcd_flag = _viz_pcd_flag
        self._viz_pcd_path = _viz_pcd_path
        self._submission = _submission
        self._submission_path = _submission_path

        # remove useless parameters.
        del self.future_pred_head.transformer
        del self.future_pred_head.bev_embedding
        del self.future_pred_head.prev_frame_embedding
        del self.future_pred_head.can_bus_mlp
        del self.future_pred_head.positional_encoding

    ############# Align coordinates between reference (current frame) to other frames. #############
    def _get_history_ref_to_previous_transform(self, tensor, num_frames, img_metas_list):
        """Get transformation matrix from reference frame to all previous frames.

        Args:
            tensor: to convert {ref_to_prev_transform} to device and dtype.
            num_frames: total num of available history frames.
            img_metas_list: a list of batch_size items.
                In each item, there is {num_prev_frames} img_meta for transformation alignment.

        Return:
            ref_to_history_list (torch.Tensor): with shape as [bs, num_prev_frames, 4, 4]
        """
        ref_to_history_list = []
        for img_metas in img_metas_list:
            cur_ref_to_prev = [img_metas[i]['ref_lidar_to_cur_lidar'] for i in range(num_frames)]
            ref_to_history_list.append(cur_ref_to_prev)
        ref_to_history_list = tensor.new_tensor(np.array(ref_to_history_list))
        return ref_to_history_list

    def _align_bev_coordnates(self, frame_idx, ref_to_history_list, img_metas):
        """Align the bev_coordinates of frame_idx to each of history_frames.

        Args:
            frame_idx: the index of target frame.
            ref_to_history_list (torch.Tensor): a tensor with shape as [bs, num_prev_frames, 4, 4]
                indicating the transformation metric from reference to each history frames.
            img_metas: a list of batch_size items.
                In each item, there is one img_meta (reference frame)
                whose {future2ref_lidar_transform} & {ref2future_lidar_transform} are for
                transformation alignment.
        """
        bs, num_frame = ref_to_history_list.shape[:2]

        # 1. get future2ref and ref2future_matrix of frame_idx.
        future2ref = [img_meta['future2ref_lidar_transform'][frame_idx] for img_meta in img_metas]  # b, 4, 4
        future2ref = ref_to_history_list.new_tensor(np.array(future2ref))  # bs, 4, 4

        ref2future = [img_meta['ref2future_lidar_transform'][frame_idx] for img_meta in img_metas]  # b, 4, 4
        ref2future = ref_to_history_list.new_tensor(np.array(ref2future))  # bs, 4, 4

        # 2. compute the transformation matrix from current frame to all previous frames.
        future2ref = future2ref.unsqueeze(1).repeat(1, num_frame, 1, 1).contiguous()
        future_to_history_list = torch.matmul(future2ref, ref_to_history_list)

        # 3. compute coordinates of future frame.
        bev_grids = e2e_predictor_utils.get_bev_grids(
            self.bev_h, self.bev_w, bs * num_frame)  # bs * num_frame, bev_h, bev_w, 2 (x, y)
        bev_grids = bev_grids.view(bs, num_frame, -1, 2)
        bev_coords = e2e_predictor_utils.bev_grids_to_coordinates(
            bev_grids, self.point_cloud_range)

        # 4. align target coordinates of future frame to each of previous frames.
        aligned_bev_coords = torch.cat([
            bev_coords, torch.ones_like(bev_coords[..., :2])], -1)  # b, num_frame, h*w, 4
        aligned_bev_coords = torch.matmul(aligned_bev_coords, future_to_history_list)
        aligned_bev_coords = aligned_bev_coords[..., :2]  # b, num_frame, h*w, 2
        aligned_bev_grids, _ = e2e_predictor_utils.bev_coords_to_grids(
            aligned_bev_coords, self.bev_h, self.bev_w, self.point_cloud_range)
        aligned_bev_grids = (aligned_bev_grids + 1) / 2.  # range of [0, 1]
        # b, h*w, num_frame, 2
        aligned_bev_grids = aligned_bev_grids.permute(0, 2, 1, 3).contiguous()

        # 5. get target bev_grids at target future frame.
        tgt_grids = bev_grids[:, -1].contiguous()
        return tgt_grids, aligned_bev_grids, ref2future

    def preprocess(self, x):
        # x: bs, F, H, W, D
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x) # bs, F, H, W, D, c

        x = x.reshape(bs, F, H, W, D * self.expansion).permute(0, 1, 4, 2, 3)

        x = preprocess.reshape_patch(x, patch_size=self.patch_size)
        return x

    def post_process(self, x):
        x = preprocess.reshape_patch_back(x, self.patch_size)

        x = rearrange(x, 'b f (d c) h w -> b f h w d c', c=self.expansion)
        logits = self.predicter(x)
        return logits
    
    def transform_inputs(self, inputs, type='grid_sample'):
        if type == 'grid_sample':
            # grid sample the original occupancy to the target pc range.
            # generate normalized grid
            device = batched_input_occs.device
            x_size = 200
            y_size = 200
            z_size = 16
            x = torch.linspace(-1.0, 1.0, x_size).view(-1, 1, 1).repeat(1, y_size, z_size).to(device)
            y = torch.linspace(-1.0, 1.0, y_size).view(1, -1, 1).repeat(x_size, 1, z_size).to(device)
            z = torch.linspace(-1.0, 1.0, z_size).view(1, 1, -1).repeat(x_size, y_size, 1).to(device)
            grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)

            grid[..., 0] = grid[..., 0] * (51.2 / 50)
            grid[..., 1] = grid[..., 1] * (51.2 / 50)
            # grid[..., 2] = grid[..., 2] - (4 / 16.0)

            # add flow to grid
            _batched_input_occs = rearrange(batched_input_occs, 'b f h w d -> (b f) () h w d')
            _batched_input_occs = _batched_input_occs + 1

            bs = _batched_input_occs.shape[0]
            grid = grid.unsqueeze(0).expand(bs, -1, -1, -1, -1)
            _batched_input_occs = F.grid_sample(_batched_input_occs.float(), 
                                                grid.flip(-1).float(), 
                                                mode='nearest', 
                                                padding_mode='zeros',
                                                align_corners=True)

            batched_input_occs = rearrange(_batched_input_occs.long(), '(b f) () h w d -> b f h w d', f=num_frames)
            batched_input_occs[batched_input_occs == 0] = 12
            batched_input_occs = batched_input_occs - 1
        elif type == 'binary':
            ## convert the occupancy to binary occupancy
            batched_input_occs[batched_input_occs != 11] = 0
            batched_input_occs[batched_input_occs == 11] = 1
        else:
            raise ValueError(f"Unknown input transform type: {type}")
        return batched_input_occs
    
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      input_occs=None,
                      img_metas=None,
                      img=None,
                      gt_points=None,
                      img_depth=None,
                      img_mask=None,
                      **kwargs,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            gt_points (torch.Tensor optional): groundtruth point clouds for future
                frames with shape (x, x, x). Defaults to None.
                The 0-th frame represents current frame for reference.
        Returns:
            dict: Losses of different branches.
        """
        num_frames = self.history_len

        ## convert the raw occ_gts into to occupancy
        batched_input_occs = []
        for bs in range(len(input_occs)):
            next_occs = []
            cur_occ = input_occs[bs]
            for _occ in cur_occ:
                next_occs.append(occ_to_voxel(_occ))
            batched_input_occs.append(torch.stack(next_occs, 0))
        batched_input_occs = torch.stack(batched_input_occs, 0)  # (bs, F, H, W, D)

        ## convert the occupancy to binary occupancy
        # batched_input_occs[batched_input_occs != 11] = 0
        # batched_input_occs[batched_input_occs == 11] = 1

        # grid sample the original occupancy to the target pc range.
        # generate normalized grid
        device = batched_input_occs.device
        x_size = 200
        y_size = 200
        z_size = 16
        x = torch.linspace(-1.0, 1.0, x_size).view(-1, 1, 1).repeat(1, y_size, z_size).to(device)
        y = torch.linspace(-1.0, 1.0, y_size).view(1, -1, 1).repeat(x_size, 1, z_size).to(device)
        z = torch.linspace(-1.0, 1.0, z_size).view(1, 1, -1).repeat(x_size, y_size, 1).to(device)
        grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)

        grid[..., 0] = grid[..., 0] * (51.2 / 50)
        grid[..., 1] = grid[..., 1] * (51.2 / 50)
        grid[..., 2] = grid[..., 2] - (4 / 16.0)

        # add flow to grid
        _batched_input_occs = rearrange(batched_input_occs, 'b f h w d -> (b f) () h w d')
        _batched_input_occs = _batched_input_occs + 1

        bs = _batched_input_occs.shape[0]
        grid = grid.unsqueeze(0).expand(bs, -1, -1, -1, -1)
        _batched_input_occs = F.grid_sample(_batched_input_occs.float(), 
                                            grid.flip(-1).float(), 
                                            mode='nearest', 
                                            padding_mode='zeros',
                                            align_corners=True)

        batched_input_occs = rearrange(_batched_input_occs.long(), '(b f) () h w d -> b f h w d', f=num_frames)
        batched_input_occs[batched_input_occs == 0] = 12
        batched_input_occs = batched_input_occs - 1

        # Preprocess the historical occupancy
        x = self.preprocess(batched_input_occs)

        # C2. Check whether the frame has previous frames.
        prev_bev_exists_list = []
        prev_img_metas = copy.deepcopy(img_metas)

        img_metas = [each[num_frames-1] for each in img_metas]

        assert len(prev_img_metas) == 1, 'Only supports bs=1 for now.'
        for prev_img_meta in prev_img_metas:  # Loop batch.
            max_key = len(prev_img_meta) - 1
            prev_bev_exists = True
            for k in range(max_key, -1, -1):
                each = prev_img_meta[k]
                prev_bev_exists_list.append(prev_bev_exists)
                prev_bev_exists = prev_bev_exists and each['prev_bev_exists']
        prev_bev_exists_list = np.array(prev_bev_exists_list)[::-1]

        valid_frames = [0]

        if self.supervise_all_future:
            valid_frames.extend(list(range(self.future_pred_frame_num)))
        else:  # randomly select one future frame for computing loss to save memory cost.
            train_frame = np.random.choice(np.arange(self.future_pred_frame_num), 1)[0]
            valid_frames.append(train_frame)

        # forecasting the future occupancy
        next_bev_preds = self.future_pred_head.forward_head(x)
        next_bev_preds = self.post_process(next_bev_preds)  # (bs, F, X, Y, Z, c)
        next_bev_preds = rearrange(next_bev_preds, 'b f x y z c -> b f c y x z')
        next_bev_preds = rearrange(next_bev_preds, 'b f c y x z -> (b f) c () () (y x) z')
        # next_bev_preds = rearrange(next_bev_preds, 'b f x y z c -> (b f) c () () (x y) z')

        # next_bev_preds = rearrange(next_bev_preds, 'b f h w d c -> b f (h w d) c')
        pred_dict = {
            'next_bev_preds': next_bev_preds,
            'valid_frames': valid_frames,
            'full_prev_bev_exists': prev_bev_exists_list.all(),
            'prev_bev_exists_list': prev_bev_exists_list,
        }

        # 5. Compute loss for point cloud predictions.
        start_idx = 0
        losses = dict()
        loss_dict = self.future_pred_head.loss(
            pred_dict, gt_points, start_idx,
            tgt_bev_h=self.bev_h, tgt_bev_w=self.bev_w,
            tgt_pc_range=self.point_cloud_range,
            pred_frame_num=self.future_pred_frame_num+1,
            img_metas=img_metas)
        losses.update(loss_dict)
        return losses

    def forward_test(self, 
                     img_metas, 
                     img=None,
                     gt_points=None, 
                     input_occs=None,
                     **kwargs):
        """has similar implementation with train forward."""
        num_frames = self.history_len

        self.eval()

        ## convert the raw occ_gts into to occupancy
        batched_input_occs = []
        for bs in range(len(input_occs)):
            next_occs = []
            cur_occ = input_occs[bs]
            for _occ in cur_occ:
                next_occs.append(occ_to_voxel(_occ))
            batched_input_occs.append(torch.stack(next_occs, 0))
        batched_input_occs = torch.stack(batched_input_occs, 0)  # (bs, F, H, W, D)

        ## convert the occupancy to binary occupancy
        # batched_input_occs[batched_input_occs != 11] = 0
        # batched_input_occs[batched_input_occs == 11] = 1

        # grid sample the original occupancy to the target pc range.
        # generate normalized grid
        device = batched_input_occs.device
        x_size = 200
        y_size = 200
        z_size = 16
        x = torch.linspace(-1.0, 1.0, x_size).view(-1, 1, 1).repeat(1, y_size, z_size).to(device)
        y = torch.linspace(-1.0, 1.0, y_size).view(1, -1, 1).repeat(x_size, 1, z_size).to(device)
        z = torch.linspace(-1.0, 1.0, z_size).view(1, 1, -1).repeat(x_size, y_size, 1).to(device)
        grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)

        grid[..., 0] = grid[..., 0] * (51.2 / 50)
        grid[..., 1] = grid[..., 1] * (51.2 / 50)
        grid[..., 2] = grid[..., 2] - (4 / 16.0)

        # add flow to grid
        _batched_input_occs = rearrange(batched_input_occs, 'b f h w d -> (b f) () h w d')
        _batched_input_occs = _batched_input_occs + 1

        bs = _batched_input_occs.shape[0]
        grid = grid.unsqueeze(0).expand(bs, -1, -1, -1, -1)
        _batched_input_occs = F.grid_sample(_batched_input_occs.float(), 
                                            grid.flip(-1).float(), 
                                            mode='nearest', 
                                            padding_mode='zeros',
                                            align_corners=True)

        batched_input_occs = rearrange(_batched_input_occs.long(), '(b f) () h w d -> b f h w d', f=num_frames)
        batched_input_occs[batched_input_occs == 0] = 12
        batched_input_occs = batched_input_occs - 1

        # Preprocess the historical occupancy
        x = self.preprocess(batched_input_occs)

        next_bev_feats, valid_frames = [], []

        # 3. predict future BEV.
        valid_frames.extend(list(range(self.test_future_frame_num)))
        img_metas = [each[num_frames - 1] for each in img_metas]

        # forecasting the future occupancy
        next_bev_preds = self.future_pred_head.forward_head(x)
        next_bev_preds = self.post_process(next_bev_preds)  # (bs, F, X, Y, Z, c)
        next_bev_preds = rearrange(next_bev_preds, 'b f x y z c -> b f c y x z')
        next_bev_preds = rearrange(next_bev_preds, 'b f c y x z -> (b f) c () () (y x) z')
        # next_bev_preds = rearrange(next_bev_preds, 'b f x y z c -> (b f) c () () (x y) z')

        pred_dict = {
            'next_bev_features': next_bev_feats,
            'next_bev_preds': next_bev_preds,
            'valid_frames': valid_frames,
        }

        # decode results and compute some statistic results if needed.
        start_idx = 0
        decode_dict = self.future_pred_head.get_point_cloud_prediction(
            pred_dict, gt_points, start_idx,
            tgt_bev_h=self.bev_h, tgt_bev_w=self.bev_w,
            tgt_pc_range=self.point_cloud_range, img_metas=img_metas)

        # convert decode_dict to quantitative statistics.
        pred_pcds = decode_dict['pred_pcds']
        gt_pcds = decode_dict['gt_pcds']
        scene_origin = decode_dict['origin']

        pred_frame_num = len(pred_pcds[0])
        ret_dict = dict()
        for frame_idx in range(pred_frame_num):
            count = 0
            frame_name = frame_idx + start_idx
            ret_dict[f'frame.{frame_name}'] = dict(
                count=0,
                chamfer_distance=0,
                l1_error=0,
                absrel_error=0,
            )
            for bs in range(len(pred_pcds)):
                pred_pcd = pred_pcds[bs][frame_idx]
                gt_pcd = gt_pcds[bs][frame_idx]

                ret_dict[f'frame.{frame_name}']['chamfer_distance'] += (
                    e2e_predictor_utils.compute_chamfer_distance_inner(
                        pred_pcd, gt_pcd, self.point_cloud_range).item())

                l1_error, absrel_error = eval_utils.compute_ray_errors(
                    pred_pcd.cpu().numpy(), gt_pcd.cpu().numpy(),
                    scene_origin[bs, frame_idx].cpu().numpy(), scene_origin.device)
                ret_dict[f'frame.{frame_name}']['l1_error'] += l1_error
                ret_dict[f'frame.{frame_name}']['absrel_error'] += absrel_error

                if self._viz_pcd_flag:
                    cur_name = img_metas[bs]['sample_idx']
                    out_path = f'{self._viz_pcd_path}_{cur_name}_{frame_name}.png'
                    gt_inside_mask = e2e_predictor_utils.get_inside_mask(gt_pcd, self.point_cloud_range)
                    gt_pcd_inside = gt_pcd[gt_inside_mask]
                    pred_pcd_inside = pred_pcd[gt_inside_mask]
                    root_path = '/'.join(out_path.split('/')[:-1])
                    mmcv.mkdir_or_exist(root_path)
                    self._viz_pcd(
                        pred_pcd_inside.cpu().numpy(),
                        scene_origin[bs, frame_idx].cpu().numpy()[None, :],
                        output_path=out_path,
                        gt_pcd=gt_pcd_inside.cpu().numpy()
                    )

                if self._submission and frame_idx > 0:
                    # ViDAR additionally predict the current frame as 0-th index.
                    #   So, we need to ignore the 0-th index by default.
                    self._save_prediction(pred_pcd, img_metas[bs], frame_idx)

                count += 1
            ret_dict[f'frame.{frame_name}']['count'] = count

        if self._viz_pcd_flag:
            print('==== Visualize predicted point clouds done!! End the program. ====')
            print(f'==== The visualized point clouds are stored at {out_path} ====')
        return [ret_dict]

    def _save_prediction(self, pred_pcd, img_meta, frame_idx):
        """ Save prediction.

        The filename is <index>-<future-id>.txt
        In each line of the file: pred_depth
        """
        base_name = img_meta['sample_idx']
        base_name = f'{base_name}_{frame_idx}.txt'
        mmcv.mkdir_or_exist(self._submission_path)
        base_name = os.path.join(self._submission_path, base_name)

        r_depth = torch.sqrt((pred_pcd ** 2).sum(1)).cpu().numpy()

        with open(base_name, 'w') as f:
            for d in r_depth:
                f.write('%f\n' % (d))

    def _viz_pcd(self, pred_pcd, pred_ctr,  output_path, gt_pcd=None):
        """Visualize predicted future point cloud."""
        color_map = np.array([
            [0, 0, 230], [219, 112, 147], [255, 0, 0]
        ])
        pred_label = np.ones_like(pred_pcd)[:, 0].astype(np.int) * 0
        if gt_pcd is not None:
            gt_label = np.ones_like(gt_pcd)[:, 0].astype(np.int)

            pred_label = np.concatenate([pred_label, gt_label], 0)
            pred_pcd = np.concatenate([pred_pcd, gt_pcd], 0)

        e2e_predictor_utils._dbg_draw_pc_function(
            pred_pcd, pred_label, color_map, output_path=output_path,
            ctr=pred_ctr, ctr_labels=np.zeros_like(pred_ctr)[:, 0].astype(np.int)
        )
