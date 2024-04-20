import os
import glob
import time
import cv2
import tqdm
import imageio
import tensorboardX
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from models.sd_update import StableDiffusion
from models.network_3dgaussain import GSNetwork
from models.provider import MiniCam
from models.loss import ssim, l1_loss
from rich.console import Console
from models.provider import SampleViewsDataset, RandomRGBSampleViewsDataset
from PIL import Image
import shutil


class Trainer_Refinement(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model_coarse: GSNetwork,  # network
                 model_initial: GSNetwork,
                 guidance: StableDiffusion,
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 ema_decay=None,  # if use EMA, set the decay
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=5,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=True,  # whether to call scheduler.step() after every train step
                 ):
        self.near = 0.001
        self.far = 10.0
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        if isinstance(model_coarse, torch.nn.Module):
            model_coarse.to(self.device)

        if isinstance(model_initial, torch.nn.Module):
            model_initial.to(self.device)

        self.model = model_coarse
        self.model_initial = model_initial

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
        self.img2mse = lambda x, y: torch.mean((x - y) ** 2)

        self.guidance = guidance
        if self.guidance is not None:
            self.prepare_text_embeddings()
        else:
            self.text_global = None

        self.ema = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        if self.opt.bg_color is not None:
            self.bg_color = torch.tensor(self.opt.bg_color).cuda()
        else:
            self.bg_color = None

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.ply_path = os.path.join(self.workspace, 'ply')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        if isinstance(model_coarse, torch.nn.Module):
            self.log(f'[INFO] #parameters: {sum([p.numel() for p in model_coarse.parameters() if p.requires_grad])}')
        self.log(self.opt)

        self.editing_points_mask = None
        self.bbox_mask = None

        if self.epoch == 0:
            print('loading pre mesh net')
            self.load_checkpoint(checkpoint=self.opt.coarse_gs_path)
            self.load_init_checkpoint(checkpoint=self.opt.initial_gs_path)

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        print('generate mask')

        mesh = o3d.io.read_triangle_mesh(self.opt.bbox_path)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.bounding_box = o3d.t.geometry.RaycastingScene()
        _ = self.bounding_box.add_triangles(mesh)

        # if self.ori_mask is None:
        self.bbox_mask = torch.LongTensor(
            self.inside_check(self.model_initial.get_xyz.data.detach().cpu().numpy())).to(self.device)

        if self.editing_points_mask is None:
            ori_num = 320600
            new_num = self.model.get_xyz.data.shape[0]
            mask = torch.cat([torch.zeros(ori_num), torch.ones(new_num - ori_num)]).cuda()
            self.editing_points_mask = mask
        print('mask points: {}, all points: {}'.format(self.editing_points_mask.sum().item(),
                                                       self.editing_points_mask.size()[0]))

        self.update_ori_parameter()

        print('active_sh_degree: {}'.format(self.model.active_sh_degree))

        self.sh_size = self.model._features_rest.data.size()[1]

    def update_ori_parameter(self):
        self.ori_xyz = self.model._xyz.clone().data
        self.ori_features_dc = self.model._features_dc.clone().data
        self.ori_features_rest = self.model._features_rest.clone().data
        self.ori_opacity = self.model._opacity.clone().data
        self.ori_scaling = self.model._scaling.clone().data
        self.ori_rotation = self.model._rotation.clone().data

    def inside_check(self, points):

        query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        occupancy = self.bounding_box.compute_occupancy(query_point)
        mask = occupancy.numpy()

        return mask

    def oob_remove(self):

        remove_mask = (self.editing_points_mask > 0) * (1 - torch.LongTensor(
            self.inside_check(self.model.get_xyz.data.detach().cpu().numpy())).to(self.device))

        # remove_mask += (self.mask > 0) * (self.model.get_opacity < self.opt.min_opacity).squeeze()
        prune_mask = remove_mask > 0

        self.model.prune_points(prune_mask)
        self.editing_points_mask = self.editing_points_mask[~prune_mask]
        self.update_ori_parameter()
        torch.cuda.empty_cache()
        print('mask points: {}, all points: {}'.format(self.editing_points_mask.sum().item(),
                                                       self.editing_points_mask.size()[0]))

    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.opt.text_global is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_global = None
            return

        self.text_global = self.guidance.get_text_embeds([self.opt.text_global], [self.opt.negative])

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    def train_step(self, data):

        rgbs, mask, _, h, w, R, T, fx, fy, pose, index = data
        rgbs = rgbs.to(self.device)
        mask = mask.to(self.device)

        h = h[0]
        w = w[0]
        R = R[0]
        T = T[0]

        fx = fx[0]
        fy = fy[0]
        pose = pose[0]

        cur_cam = MiniCam(
            R,
            T,
            w,
            h,
            fy,
            fx,
            pose,
            self.near,
            self.far
        )

        outputs = self.model.render(cur_cam, bg_color=self.bg_color)

        pred_rgb = outputs['image']
        alpha = outputs['alpha'][0]

        # loss_c = F.mse_loss(outputs['image'].permute(1, 2, 0).reshape(-1, 3), rgbs.reshape(-1, 3), reduction='mean')
        loss_c = l1_loss(outputs['image'].unsqueeze(0), rgbs.permute(2, 0, 1).reshape(-1, 3, h, w))
        loss_s = 1.0 - ssim(outputs['image'].unsqueeze(0), rgbs.permute(2, 0, 1).reshape(-1, 3, h, w))

        loss = (1.0 - self.opt.lambda_dssim) * loss_c + self.opt.lambda_dssim * loss_s

        loss_dict = {
            'loss_c': loss_c.item(),
            'loss_s': loss_s.item(),

        }

        return pred_rgb, loss, loss_dict, outputs

    def eval_step(self, data, p_mask=None):
        # rgbs, mask, h, w, R, T, fx, fy, index = data
        rgbs, mask, _, h, w, R, T, fx, fy, pose, index = data

        h = h[0]
        w = w[0]
        R = R[0]
        T = T[0]

        fx = fx[0]
        fy = fy[0]
        pose = pose[0]

        cur_cam = MiniCam(
            R,
            T,
            w,
            h,
            fy,
            fx,
            pose,
            self.near,
            self.far
        )

        B = 1

        if p_mask is not None:
            outputs = self.model.render(cur_cam, invert_bg_color=True, mask=p_mask)
        else:
            outputs = self.model.render(cur_cam, bg_color=self.bg_color)

        pred_rgb = outputs['image'].permute(1, 2, 0)  # .reshape(B, H, W, 3)

        all_pred_rgb = outputs['image'].permute(1, 2, 0).reshape(B, h, w, 3)
        all_pred_depth = outputs['depth'].reshape(B, h, w)
        all_pred_depth /= torch.max(all_pred_depth)
        all_pred_normal = all_pred_rgb

        return all_pred_rgb, all_pred_depth, all_pred_normal

    def test_step(self, data, model, point_mask=None):

        rgbs, mask, _, h, w, R, T, fx, fy, pose, index = data

        h = h[0]
        w = w[0]
        R = R[0]
        T = T[0]

        fx = fx[0]
        fy = fy[0]
        pose = pose[0]

        cur_cam = MiniCam(
            R,
            T,
            w,
            h,
            fy,
            fx,
            pose,
            self.near,
            self.far
        )

        B = 1
        outputs = model.render(cur_cam, mask=point_mask, bg_color=self.bg_color)

        all_pred_rgb = outputs['image'].permute(1, 2, 0).reshape(B, h, w, 3)
        all_pred_depth = outputs['depth'].reshape(B, h, w)
        all_pred_depth /= torch.max(all_pred_depth)

        all_pred_alpha = outputs['alpha'].reshape(B, h, w)

        return all_pred_rgb, all_pred_depth, all_pred_alpha

    def save_mesh(self, save_path=None, resolution=128):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        mesh = self.model.extract_mesh(save_path, self.opt.density_thresh)
        mesh.write_ply(os.path.join(save_path, 'mesh.ply')
                       )

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    @torch.no_grad()
    def getGaussianKernel(self, ksize, sigma=0):
        if sigma <= 0:
            # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        center = ksize // 2
        xs = (np.arange(ksize, dtype=np.float32) - center)  # 元素与矩阵中心的横向距离
        kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))  # 计算一维卷积核
        # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
        kernel = kernel1d[..., None] @ kernel1d[None, ...]
        kernel = torch.from_numpy(kernel)
        kernel = kernel / kernel.sum()  # 归一化
        return kernel

    def GaussianBlur(self, batch_img, ksize, sigma=0):
        kernel = self.getGaussianKernel(ksize, sigma).to(self.device)  # 生成权重
        B, C, H, W = batch_img.shape  # C：图像通道数，group convolution 要用到
        # 生成 group convolution 的卷积核
        kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
        pad = (ksize - 1) // 2  # 保持卷积前后图像尺寸不变
        # mode=relfect 更适合计算边缘像素的权重
        batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
        weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,
                                stride=1, padding=0, groups=C)
        return weighted_pix

    def update_dataset(self, t):

        print('update dataset with t :{}'.format(t))
        test_loader = SampleViewsDataset(self.opt, device=self.device, type='test').dataloader()

        save_path = os.path.join(self.workspace, 'refine_views')
        mask_path = os.path.join(self.workspace, 'mask')

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        shutil.rmtree(save_path)
        shutil.rmtree(mask_path)

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(test_loader)):

                poses = data[-2][0]
                phi = data[0]
                theta = data[1]
                radius = data[2]

                out_name = f'{theta}_{phi}_{radius}.png'

                if self.opt.editing_type == 0:
                    ori_image, _, _ = self.test_step(data, self.model_initial)
                else:
                    ori_image, _, _ = self.test_step(data, self.model_initial,
                                                     point_mask=1 - self.bbox_mask)

                preds, preds_depth, dir = self.test_step(data, self.model)

                _, _, mask = self.test_step(data, self.model, point_mask=self.editing_points_mask)

                mask[mask > 0.5] = 1
                mask = mask[0].detach().cpu().numpy()

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                pred = Image.fromarray(pred)

                ori_image = ori_image[0].detach().cpu().numpy()
                ori_image = (ori_image * 255).astype(np.uint8)

                poses = torch.from_numpy(poses).float().to(self.device)

                image = self.guidance.pipeline(self.opt.text_global, image=pred, strength=t,
                                               guidance_scale=self.opt.im_guidance_scale,
                                               class_labels=poses.view(1, -1)).images[0]

                image = np.array(image)

                mask_png = (mask * 255).astype(np.uint8)
                mask_png = Image.fromarray(mask_png)
                mask_png.save(os.path.join(mask_path, out_name))

                blend_img = image * mask[:, :, np.newaxis] + ori_image * (1 - mask[:, :, np.newaxis])
                blend_img = Image.fromarray(blend_img.astype(np.uint8))
                blend_img.save(os.path.join(save_path, out_name))

        train_loader = RandomRGBSampleViewsDataset(self.opt, save_path, device=self.device, type='test').dataloader()

        return train_loader

    def train(self, valid_loader, max_epochs):

        train_loader = self.update_dataset(self.opt.intermediate_time)

        self.valid_loader = valid_loader
        self.evaluate_one_epoch(valid_loader)
        self.model.xyz_gradient_accum = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")
        self.model.denom = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):

            self.epoch = epoch

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint()

            self.train_one_epoch(train_loader)
            self.oob_remove()

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True, if_gui=False):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                preds, preds_depth, dir = self.test_step(data, self.model, if_gui=if_gui)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)

                    cv2.imwrite(os.path.join(save_path, f'{i:03d}.png'),
                                cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=10, quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8,
                             macro_block_size=1)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.model.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        if isinstance(self.model, torch.nn.Module):
            self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            self.local_step += 1
            self.global_step += 1

            self.model.update_learning_rate(self.global_step)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if self.global_step % 1000 == 0:
                self.model.oneupSHdegree()

            pred_rgbs, loss, loss_dict, outputs = self.train_step(data)

            # loss = c_loss
            loss.backward()

            with torch.no_grad():
                visibility_filter = outputs['visibility_filter']
                radii = outputs['radii']
                viewspace_point_tensor = outputs['viewspace_points']

                if visibility_filter.size()[0] != self.model.max_radii2D.size()[0]:
                    self.model.max_radii2D[self.editing_points_mask > 0][visibility_filter] = torch.max(
                        self.model.max_radii2D[self.editing_points_mask > 0][visibility_filter],
                        radii[visibility_filter])
                    self.model.add_densification_stats(viewspace_point_tensor, visibility_filter,
                                                       mask=self.editing_points_mask)
                else:
                    self.model.max_radii2D[visibility_filter] = torch.max(
                        self.model.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.global_step % self.opt.densification_interval == 0:
                    # size_threshold = 1 if self.global_step > self.opt.opacity_reset_interval else None

                    self.editing_points_mask = self.model.densify_and_prune_with_mask(self.opt.densify_grad_threshold,
                                                                                      min_opacity=self.opt.min_opacity,
                                                                                      extent=self.opt.extent,
                                                                                      max_screen_size=self.opt.max_screen_size,
                                                                                      old_mask=self.editing_points_mask)
                    self.update_ori_parameter()
                    print('mask points: {}, all points: {}'.format(self.editing_points_mask.sum().item(),
                                                                   self.editing_points_mask.size()[0]))
                    # print('update_ori_parameter')

                if self.global_step % self.opt.opacity_reset_interval == 0:
                    self.model.reset_opacity()

                self.model.optimizer.step()
                self.model.optimizer.zero_grad(set_to_none=True)

                self.model._xyz.data = (1 - self.editing_points_mask.unsqueeze(1).repeat(1, 3)) * self.ori_xyz + \
                                       self.editing_points_mask.unsqueeze(1).repeat(1, 3) * self.model._xyz.data

                self.model._features_dc.data = (1 - self.editing_points_mask.reshape(-1, 1, 1).repeat(1, 1,
                                                                                                      3)) * self.ori_features_dc + \
                                               self.editing_points_mask.reshape(-1, 1, 1).repeat(1, 1,
                                                                                                 3) * self.model._features_dc.data
                self.model._features_rest.data = (1 - self.editing_points_mask.reshape(-1, 1, 1).repeat(1, self.sh_size,
                                                                                                        3)) * self.ori_features_rest + \
                                                 self.editing_points_mask.reshape(-1, 1, 1).repeat(1, self.sh_size,
                                                                                                   3) * self.model._features_rest.data

                self.model._opacity.data = (1 - self.editing_points_mask.reshape(-1, 1)) * self.ori_opacity \
                                           + self.editing_points_mask.reshape(-1, 1) * self.model._opacity.data
                self.model._scaling.data = (1 - self.editing_points_mask.reshape(-1, 1).repeat(1,
                                                                                               3)) * self.ori_scaling + \
                                           self.editing_points_mask.reshape(-1, 1).repeat(1,
                                                                                          3) * self.model._scaling.data
                self.model._rotation.data = (1 - self.editing_points_mask.reshape(-1, 1).repeat(1,
                                                                                                4)) * self.ori_rotation + \
                                            self.editing_points_mask.reshape(-1, 1).repeat(1,
                                                                                           4) * self.model._rotation.data

            loss_val = loss_dict['loss_c']
            total_loss += loss_val

            if self.local_rank == 0:

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.model.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    print_str = f""
                    for k, v in loss_dict.items():
                        print_str += "{} : {:.4f}, ".format(k, v)

                    pbar.set_description(
                        print_str + f" ({total_loss / self.local_step:.4f}), lr={self.model.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        self.log(f"==> Finished Epoch {self.epoch}. average_loss {average_loss}")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        # self.model.eval()
        # self.model.train()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        all_preds = []
        all_preds_depth = []
        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                preds, preds_depth, preds_normal = self.eval_step(data)

                loss_val = 0
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation',
                                                   f'{name}_{self.local_step:04d}_depth.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        all_preds = np.stack(all_preds, axis=0)
        all_preds_depth = np.stack(all_preds_depth, axis=0)

        imageio.mimwrite(os.path.join(self.workspace, 'validation', f'{name}_rgb.mp4'), all_preds, fps=24, quality=8,
                         macro_block_size=1)
        imageio.mimwrite(os.path.join(self.workspace, 'validation', f'{name}_depth.mp4'), all_preds_depth, fps=24,
                         quality=8,
                         macro_block_size=1)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished. loss: ({total_loss / self.local_step:.4f})")

    def save_checkpoint(self, name=None):
        file_path = f"{name}.pth"

        self.stats["checkpoints"].append(file_path)

        if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
            old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
        state = {'epoch': self.epoch, 'global_step': self.global_step,
                 'model': self.model.capture(), 'mask': self.editing_points_mask}
        torch.save(state, os.path.join(self.ckpt_path, file_path))

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(checkpoint_dict['model'])
        if 'mask' in checkpoint_dict.keys():
            self.editing_points_mask = checkpoint_dict['mask']
        self.log("[INFO] loaded model.")
        self.log("[INFO] update optimizer.")
        return

    def load_init_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        self.model_initial.load_state_dict(checkpoint_dict['model'])

        self.log("[INFO] loaded model.")
        self.log("[INFO] update optimizer.")
        return