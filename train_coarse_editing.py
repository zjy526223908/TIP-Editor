import torch
import numpy as np
import argparse

from models.provider import SphericalSamplingDataset

from models.trainer_coarse_editing import Trainer_SDS
from models.utils import seed_everything

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_global', default=None, help="global text prompt")
    parser.add_argument('--text_local', default=None, help="local text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_video', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--bbox_path', type=str, default=None)
    parser.add_argument('--sd_path', type=str, default='./res_gaussion/colmap_doll/content_personalization')
    parser.add_argument('--sd_img_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sd_max_step_start', type=float, default=0.75, help="sd_max_step")
    parser.add_argument('--sd_max_step_end', type=float, default=0.25, help="sd_max_step")

    ### training options
    parser.add_argument('--sh_degree', type=int, default=0)
    parser.add_argument('--bbox_size_factor', type=float, default=1., help="size factor of 3d bounding box")
    parser.add_argument('--start_gamma', type=float, default=0.99, help="initial gamma value")
    parser.add_argument('--end_gamma', type=float, default=0.5, help="end gamma value")
    parser.add_argument('--points_times', type=int, default=1, help="repeat editing points x times")
    parser.add_argument('--position_lr_init', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--position_lr_final', type=float, default=0.00002, help="initial learning rate")
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--position_lr_max_steps', type=float, default=30_000, help="initial learning rate")
    parser.add_argument('--feature_lr', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.1, help="initial learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.005, help="initial learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.005, help="initial learning rate")
    parser.add_argument('--percent_dense', type=float, default=0.1, help="initial learning rate")
    parser.add_argument('--densification_interval', type=float, default=250)
    parser.add_argument('--opacity_reset_interval', type=float, default=30000)
    parser.add_argument('--densify_from_iter', type=float, default=500)
    parser.add_argument('--densify_until_iter', type=float, default=15_000)
    parser.add_argument('--densify_grad_threshold', type=float, default=5)
    parser.add_argument('--min_opacity', type=float, default=0.001)
    parser.add_argument('--max_screen_size', type=float, default=1.0)
    parser.add_argument('--max_scale_size', type=float, default=0.05)
    parser.add_argument('--extent', type=float, default=0.5)
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--guidance_scale', type=float, default=10.0)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--bg_color', type=float, nargs='+', default=None)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    parser.add_argument("--editing_type", type=int, default=0, help="0:add new object, 1:edit existing object")
    parser.add_argument("--reset_points", type=bool, default=False, help="If reset color and size of the editing points")

    ### dataset options
    parser.add_argument("--pose_sample_strategy", type=str, default='uniform',
                        help='input data directory')
    parser.add_argument("--R_path", type=str, default=None,
                        help='input data directory')
    parser.add_argument('--batch_size', type=int, default=1, help="GUI width")
    parser.add_argument('--num_work', type=int, default=4, help="GUI width")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.4, 1.6],
                        help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[65, 65], help="training camera fovy range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-90, 90], help="training camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[60, 90], help="training camera fovy range")

    opt = parser.parse_args()

    if opt.seed is not None:
        seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.network_3dgaussain import GSNetwork

    model = GSNetwork(opt, device)

    if opt.test:
        guidance = None  # no need to load guidance model at test

        trainer = Trainer_SDS('df', opt, model, guidance, opt, device=device, workspace=opt.workspace,
                              fp16=opt.fp16,
                              use_checkpoint=opt.ckpt)

        if opt.save_video:
            test_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.sample_R_path, type='test', H=512,
                                                   W=512, size=250).dataloader()
            trainer.test(test_loader)
            sys.exit()

    else:

        from models.sd import StableDiffusion

        guidance = StableDiffusion(opt, device, sd_path=opt.sd_path)

        train_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.R_path, size=100 * opt.batch_size).dataloader()

        trainer = Trainer_SDS('df', opt, model, guidance, device=device, workspace=opt.workspace,
                              fp16=opt.fp16, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval)

        valid_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512,
                                                size=120).dataloader()

        max_epoch = np.ceil(opt.iters / 100).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)
