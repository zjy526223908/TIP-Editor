import torch
import numpy as np
import argparse
import os
from models.provider import SphericalSamplingDataset
from models.network_3dgaussain import GSNetwork
from models.trainer_refinement import Trainer_Refinement
from models.utils import seed_everything

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_global', default=None, help="global text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_vedio', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--coarse_gs_path', type=str, default=None)
    parser.add_argument('--initial_gs_path', type=str, default=None)
    parser.add_argument('--bbox_path', type=str, default=None)
    parser.add_argument('--sd_path', type=str, default='/mnt/d/project/sd_pre/textual_inversion_doll_unet_200')
    parser.add_argument('--sd_img_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sd_max_step', type=float, default=0.98, help="sd_max_step")
    parser.add_argument('--im_guidance_scale', type=float, default=5.0, help="initial learning rate")

    ### training options
    parser.add_argument('--sh_degree', type=int, default=0)
    parser.add_argument('--bbox_size_factor', type=float, default=1., help="size factor of 3d bounding box")
    parser.add_argument('--start_gamma', type=float, default=0.99, help="initial gamma value")
    parser.add_argument('--end_gamma', type=float, default=0.5, help="end gamma value")
    parser.add_argument('--points_times', type=int, default=1, help="repeat editing points x times")
    parser.add_argument('--position_lr_init', type=float, default=0.00016, help="initial learning rate")
    parser.add_argument('--position_lr_final', type=float, default=0.0000016, help="initial learning rate")
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--position_lr_max_steps', type=float, default=30_000, help="initial learning rate")
    parser.add_argument('--feature_lr', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.1, help="initial learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.005, help="initial learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.005, help="initial learning rate")
    parser.add_argument('--percent_dense', type=float, default=0.1, help="initial learning rate")
    parser.add_argument('--lambda_dssim', type=float, default=0.2, help="initial learning rate")
    parser.add_argument('--densification_interval', type=float, default=100)
    parser.add_argument('--opacity_reset_interval', type=float, default=3000)
    parser.add_argument('--densify_from_iter', type=float, default=500)
    parser.add_argument('--densify_until_iter', type=float, default=15_000)
    parser.add_argument('--densify_grad_threshold', type=float, default=0.0002)
    parser.add_argument('--min_opacity', type=float, default=0.001)
    parser.add_argument('--max_screen_size', type=float, default=1.0)
    parser.add_argument('--max_scale_size', type=float, default=0.05)
    parser.add_argument('--extent', type=float, default=2)
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--guidance_scale', type=float, default=10.0)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--bg_color', type=float, nargs='+', default=None)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    parser.add_argument("--editing_type", type=int, default=0, help="0:add new object, 1:edit existing object")
    parser.add_argument("--reset_points", type=bool, default=False,
                        help="If reset color and size of the editing points")

    ### dataset options
    parser.add_argument("--pose_sample_strategy", type=str, default='uniform',
                        help='input data directory')
    parser.add_argument('--batch_size', type=int, default=1, help="GUI width")
    parser.add_argument('--num_work', type=int, default=4, help="GUI width")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.4, 1.6],
                        help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[65, 65], help="training camera fovy range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-90, 90], help="training camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[60, 90], help="training camera fovy range")
    parser.add_argument('--fovy', type=float, default=50)
    parser.add_argument('--radius_list', type=float, nargs='*', default=[1.0], help="training camera fovy range")
    parser.add_argument('--phi_list', type=float, nargs='*', default=[-180, 180], help="training camera fovy range")
    parser.add_argument('--theta_list', type=float, nargs='*', default=[0, 90], help="training camera fovy range")
    parser.add_argument('--intermediate_time', type=float, default=0.05)

    opt = parser.parse_args()

    if opt.seed is not None:
        seed_everything(opt.seed)

    os.makedirs(opt.workspace, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:
        guidance = None  # no need to load guidance model at test

        trainer = Trainer_Refinement('df', opt, GSNetwork(opt, device), GSNetwork(opt, device), device=device,
                                     workspace=opt.workspace,
                                     fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.save_vedio:
            test_loader = SphericalSamplingDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=50).dataloader()
            trainer.test(test_loader)

    else:
        from models.sd_update import StableDiffusion

        guidance = StableDiffusion(opt, device, sd_path=opt.sd_path)

        trainer = Trainer_Refinement('df', opt, GSNetwork(opt, device), GSNetwork(opt, device), guidance,
                                     device=device, workspace=opt.workspace,
                                     fp16=opt.fp16, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval)

        valid_loader = SphericalSamplingDataset(opt, device=device, type='test', size=120).dataloader()

        max_epoch = np.ceil(opt.iters / (len(opt.radius_list) * len(opt.theta_list) * len(opt.phi_list))).astype(np.int32)
        print('max_epoch : {}'.format(max_epoch))

        trainer.train(valid_loader, max_epoch)
