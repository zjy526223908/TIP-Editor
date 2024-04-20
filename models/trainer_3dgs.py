import os
import glob
import sys
import time
import cv2
import tqdm
import imageio
import tensorboardX
import numpy as np
import scipy.spatial
import torch
import pymeshlab
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from models.network_3dgaussain import GSNetwork
from models.provider import MiniCam
from models.loss import ssim, l1_loss
from rich.console import Console


class Trainer_3DGS(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model: GSNetwork,  # network
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
                 max_keep_ckpt=3,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
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

        if isinstance(model, torch.nn.Module):
            model.to(self.device)

        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model


        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)

        if self.opt.bg_color is not None:
            self.bg_color = torch.tensor(self.opt.bg_color).cuda()
        else:
            self.bg_color = None


        self.ema = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.editing_points_mask = None
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

        self.log(self.opt)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        if isinstance(model, torch.nn.Module):
            self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)



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

        rgbs, mask, h, w, R, T, fx, fy, pose, index  = data

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

        outputs = self.model.render(cur_cam)

        loss_c = l1_loss(outputs['image'].unsqueeze(0), rgbs.permute(0, 2, 1).reshape(-1, 3, h, w))
        loss_s = 1.0 - ssim(outputs['image'].unsqueeze(0), rgbs.permute(0, 2, 1).reshape(-1, 3, h, w))

        loss = (1.0 - self.opt.lambda_dssim) * loss_c + self.opt.lambda_dssim * loss_s

        loss_dict = {
            'loss_c': loss_c.item(),
            'loss_s': loss_s.item(),
        }


        return outputs['image'].permute(1, 2, 0), loss, loss_dict ,outputs


    def eval_step(self, data):
        # rgbs, mask, h, w, R, T, fx, fy, index = data
        rgbs, mask, h, w, R, T, fx, fy, pose, index = data

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
        outputs = self.model.render(cur_cam)

        pred_rgb = outputs['image'].permute(1, 2, 0)#.reshape(B, H, W, 3)

        loss = self.img2mse(pred_rgb.reshape(-1, 3), rgbs.reshape(-1, 3))


        all_pred_rgb = outputs['image'].permute(1, 2, 0).reshape(B, h, w, 3)
        all_pred_depth = outputs['depth'].reshape(B, h, w)
        all_pred_depth /= torch.max(all_pred_depth)
        all_pred_normal = all_pred_rgb

        return all_pred_rgb, all_pred_depth, all_pred_normal, loss

    def test_step(self, data,  perturb=False, if_gui=False, invert_bg_color=True, p_mask=None):

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

        # bg_color = torch.tensor([0.1, 0.2, 0.6]).cuda()

        B = 1

        if p_mask is not None:
            outputs = self.model.render(cur_cam, invert_bg_color=invert_bg_color, mask=p_mask) # invert_bg_color=True,
        else:
            outputs = self.model.render(cur_cam, bg_color=self.bg_color)


        all_pred_rgb = outputs['image'].permute(1, 2, 0).reshape(B, h, w, 3)
        all_pred_depth = outputs['depth'].reshape(B, h, w)
        all_pred_depth /= torch.max(all_pred_depth)

        return all_pred_rgb, all_pred_depth, dir


    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        self.model.xyz_gradient_accum = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")
        self.model.denom = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()


        for epoch in range(self.epoch + 1, max_epochs + 1):

            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=True, best=False)
                # self.model.save_ply(os.path.join(self.workspace, 'ply', '{}.ply'.format(str(epoch))))

        self.save_checkpoint(full=True, best=False)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def inside_test(self, points, vertices):
        deln = scipy.spatial.Delaunay(vertices)
        mask = deln.find_simplex(points) + 1
        mask[mask > 0] = 1
        return mask


    def oob_remove(self):



        remove_mask = (self.model.get_opacity < self.opt.min_opacity).squeeze()

        if self.editing_points_mask is not None:
            remove_mask = remove_mask * self.editing_points_mask

        prune_mask = remove_mask>0

        if self.editing_points_mask is not None:
            self.editing_points_mask = self.editing_points_mask[~prune_mask]

        self.model.prune_points(prune_mask)




        torch.cuda.empty_cache()

    def sample_views(self, loader, save_path=None):
        os.makedirs(os.path.join(save_path,'rgb'), exist_ok=True)
        pose_dict = {}
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(loader)):
                poses = data[-2][0]
                phi = int(data[0])
                theta = int(data[1])
                radius = data[2]
                out_name = f'{radius}_{theta}_{phi}.png'
                preds, preds_depth, dir = self.test_step(data)  # , invert_bg_color=True, p_mask=trainer.mask
                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path,'rgb', out_name), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                pose_dict[out_name] = poses
        np.save(os.path.join(save_path, 'pose_dict.npy'), pose_dict)


    def sample_mask_views(self, loader, save_path=None):
        os.makedirs(os.path.join(save_path,'mask'), exist_ok=True)

        # add 3d bounding box

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(self.opt.bounding_box_path)
        ms.meshing_surface_subdivision_midpoint(iterations=4)
        pymesh = ms.current_mesh()
        xyz = pymesh.vertex_matrix()

        from models.sh_utils import SH2RGB
        from models.network_3dgaussain import BasicPointCloud
        from PIL import Image

        num_pts = xyz.shape[0]
        shs = np.ones((num_pts, 3))
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )
        self.model.add_from_pcd(pcd, 1)

        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(loader)):
                phi = int(data[0])
                theta = int(data[1])
                radius = data[2]
                out_name = f'{radius}_{theta}_{phi}.png'
                preds, preds_depth, dir = self.test_step(data, if_gui=True)
                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                pred = Image.fromarray(pred).convert('L')
                pred.save(os.path.join(save_path, 'mask', out_name))


    def test(self, loader, save_path=None, name=None, write_video=True, if_gui=False):

        self.model.active_sh_degree = 0

        print(self.model.active_sh_degree)

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
                preds, preds_depth, dir = self.test_step(data, if_gui=if_gui)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)

                    # cv2.imwrite(os.path.join(save_path, f'{i:03d}.png'),
                    #             cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    # cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb_{dir.item()}.png'),
                    #             cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    # cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8,
                             macro_block_size=1)

        if self.editing_points_mask is not None:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            if write_video:
                all_preds = []
                all_preds_depth = []
            with torch.no_grad():
                for i, data in enumerate(loader):
                    preds, preds_depth, dir = self.test_step(data, if_gui=if_gui, p_mask=self.editing_points_mask)
                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)
                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    if write_video:
                        all_preds.append(pred)
                        all_preds_depth.append(pred_depth)
                    pbar.update(loader.batch_size)

            if write_video:
                all_preds = np.stack(all_preds, axis=0)
                imageio.mimwrite(os.path.join(save_path, f'{name}_rgb_local.mp4'), all_preds, fps=25, quality=8,
                                 macro_block_size=1)
                imageio.mimwrite(os.path.join(save_path, f'{name}_depth_local.mp4'), all_preds_depth, fps=25, quality=8,
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
            if  self.global_step % 1000 == 0:
                self.model.oneupSHdegree()


            pred_rgbs, loss,  loss_dict, outputs = self.train_step(data)


            # loss = c_loss
            loss.backward()

            with torch.no_grad():
                visibility_filter = outputs['visibility_filter']
                radii = outputs['radii']
                viewspace_point_tensor = outputs['viewspace_points']
                self.model.max_radii2D[visibility_filter] = torch.max(
                    self.model.max_radii2D[visibility_filter], radii[visibility_filter])
                self.model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.global_step > self.opt.densify_from_iter and self.global_step % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.global_step > self.opt.opacity_reset_interval else None
                    self.model.densify_and_prune(self.opt.densify_grad_threshold,
                         min_opacity=self.opt.min_opacity, extent=loader.dataset.radius, max_screen_size=size_threshold)

                if self.global_step % self.opt.opacity_reset_interval == 0:
                    self.model.reset_opacity()

                self.model.optimizer.step()
                self.model.optimizer.zero_grad(set_to_none = True)

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
                        print_str+f" ({total_loss / self.local_step:.4f}), lr={self.model.optimizer.param_groups[0]['lr']:.6f}")
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

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1
                # if self.local_step == 5:
                #     break

                preds, preds_depth, preds_normal, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_normal_list = [torch.zeros_like(preds_normal).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_normal_list, preds_normal)
                    preds_normal = torch.cat(preds_normal_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in
                                        range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0 :
                    if self.local_step % 10 == 0:
                        # save image
                        save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                        save_path_normal = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_normal.png')
                        save_path_depth = os.path.join(self.workspace, 'validation',
                                                       f'{name}_{self.local_step:04d}_depth.png')

                        # self.log(f"==> Saving validation image to {save_path}")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        pred = preds[0].detach().cpu().numpy()
                        pred = (pred * 255).astype(np.uint8)

                        preds_normal = preds_normal[0].detach().cpu().numpy()
                        preds_normal = ((preds_normal/2+0.5) * 255).astype(np.uint8)

                        pred_depth = preds_depth[0].detach().cpu().numpy()
                        pred_depth = (pred_depth * 255).astype(np.uint8)

                        cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                        # cv2.imwrite(save_path_normal, cv2.cvtColor(preds_normal, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

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

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'
        file_path = f"{name}.pth"

        self.stats["checkpoints"].append(file_path)

        if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
            old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

        state = {'epoch': self.epoch, 'global_step': self.global_step,
                 'model': self.model.capture()}

        if self.editing_points_mask is not None:
            state.update({'mask': self.editing_points_mask})

        torch.save(state, os.path.join(self.ckpt_path, file_path))

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return
        print(checkpoint)
        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(checkpoint_dict['model'])
        if 'mask' in checkpoint_dict.keys():
            self.editing_points_mask = checkpoint_dict['mask']
        else:
            self.editing_points_mask = None
        self.log("[INFO] loaded model.")
        self.log("[INFO] update optimizer.")

        return

