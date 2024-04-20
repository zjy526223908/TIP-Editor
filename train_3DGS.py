import argparse
import torch
import os
import sys
import numpy as np
from models.utils import seed_everything
from models.provider import SceneDataset, SphericalSamplingDataset, SampleViewsDataset
from models.trainer_3dgs import Trainer_3DGS


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=50, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### sampling options
    parser.add_argument('--sample', action='store_true', help="sample views mode")
    parser.add_argument('--radius_list', type=float, nargs='*', default=[0.2, 0.4])
    parser.add_argument('--fovy', type=float, default=50)
    parser.add_argument('--phi_list', type=float, nargs='*', default=[-180, 180])
    parser.add_argument('--theta_list', type=float, nargs='*', default=[0, 90])
    parser.add_argument('--bounding_box_path', type=str, default=None)


    ### training options
    parser.add_argument('--iters', type=int, default=30_000, help="training iters")
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--sh_degree', type=int, default=0)
    parser.add_argument('--position_lr_init', type=float, default= 0.00016)
    parser.add_argument('--position_lr_final', type=float, default= 0.0000016)
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01)
    parser.add_argument('--position_lr_max_steps', type=float, default=30_000)
    parser.add_argument('--feature_lr', type=float, default=0.0025)
    parser.add_argument('--opacity_lr', type=float, default=0.05)
    parser.add_argument('--scaling_lr', type=float, default=0.005)
    parser.add_argument('--rotation_lr', type=float, default=0.001)
    parser.add_argument('--percent_dense', type=float, default=0.01)
    parser.add_argument('--lambda_dssim', type=float, default=0.2)
    parser.add_argument('--densification_interval', type=float, default=100)
    parser.add_argument('--opacity_reset_interval', type=float, default=3000)
    parser.add_argument('--densify_from_iter', type=float, default=500)
    parser.add_argument('--densify_until_iter', type=float, default=15_000)
    parser.add_argument('--densify_grad_threshold', type=float, default=0.0002)
    parser.add_argument('--min_opacity', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--ckpt', type=str, default='latest')


    ### dataset options
    parser.add_argument("--data_path", type=str, default='/mnt/d/dataset/data_DTU/dtu_scan105/',
                        help='input data directory')
    parser.add_argument("--if_data_cuda", action='store_false')
    parser.add_argument("--data_type", type=str, default='dtu', help='input data')
    parser.add_argument('--initial_points', type=str, default=None)
    parser.add_argument('--bg_color', type=float, nargs='+', default=None)
    parser.add_argument("--R_path", type=str, default=None, help='input data directory')
    parser.add_argument("--sample_R_path", type=str, default=None, help='input data directory')
    parser.add_argument('--batch_size', type=int, default=1, help="GUI width")
    parser.add_argument('--batch_rays', type=int, default=512, help="GUI width")
    parser.add_argument('--train_resolution_level', type=float, default=1, help="GUI width")
    parser.add_argument('--eval_resolution_level', type=float, default=4, help="GUI width")
    parser.add_argument('--num_work', type=int, default=0, help="GUI width")
    parser.add_argument('--train_batch_type', type=str, default='image')
    parser.add_argument('--val_batch_type', type=str, default='image')
    parser.add_argument('--radius_range', type=float, nargs='*', default=[0.15, 0.15], help="test camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[50, 70], help="test camera fovy range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="test camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[60, 90], help="test camera fovy range")



    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.network_3dgaussain import GSNetwork


    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GSNetwork(opt, device)


    if isinstance(model, torch.nn.Module):
        print(model)

    if opt.test:
        guidance = None # no need to load guidance model at test
        trainer = Trainer_3DGS('df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.save_video:
            test_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.sample_R_path, type='test', H=512, W=512, size=250).dataloader()
            trainer.test(test_loader)
            sys.exit()
        elif opt.sample:
            test_loader = SampleViewsDataset(opt, device=device, R_path=opt.R_path, type='test', H=512, W=512).dataloader()
            trainer.sample_views(test_loader, os.path.join(opt.workspace, 'sample_views'))
            trainer.sample_mask_views(test_loader, os.path.join(opt.workspace, 'sample_views'))
            sys.exit()
        else:
            test_loader = SceneDataset(opt, device=device, R_path=opt.R_path, type='val').dataloader()
            trainer.test(test_loader, if_gui=False)
    else:

        train_loader = SceneDataset(opt, device=device, R_path=opt.R_path, type='train').dataloader()
        valid_loader = SceneDataset(opt, device=device, R_path=opt.R_path, type='val').dataloader()

        model.initialize_from_mesh(opt.initial_points, opt.R_path, 1)
        model.training_setup(opt)

        trainer = Trainer_3DGS('df', opt, model, device=device, workspace=opt.workspace, ema_decay=None, fp16=opt.fp16,
               use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

        print('max_epoch : {}'.format(max_epoch))

        trainer.train(train_loader, valid_loader, max_epoch)

