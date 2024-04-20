import os.path

import torch
import argparse
from models.sh_utils import eval_sh
import numpy as np
from plyfile import PlyData, PlyElement


parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--pth_path',type=str, required=True)
args = parser.parse_args()

model_path = args.pth_path

model = torch.load(model_path)['model']


max_sh_degree = model[0]
xyz = model[1]
features_dc = model[2]
features_rest = model[3]

features = torch.cat((features_dc, features_rest), dim=1)
shs_view = torch.cat((features_dc, features_rest), dim=1).transpose(1, 2)


camera_center = torch.tensor([-1.3660,  0.5176,  1.3660]).cuda()

dir_pp = xyz - camera_center.repeat(
    features.shape[0], 1
)
dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
sh2rgb = eval_sh(
    max_sh_degree, shs_view, dir_pp_normalized
)
colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)


xyz =xyz.detach().cpu().numpy()
colors_precomp = colors_precomp * 255
colors_precomp = colors_precomp.detach().cpu().numpy().astype(int)

points = xyz
points = [(points[i,0], points[i,1], points[i,2], colors_precomp[i,0], colors_precomp[i,1], colors_precomp[i,2], ) for i in range(points.shape[0])]



vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])



PlyData([el]).write('/'.join(model_path.split('/')[:-2]+['colored_scene_points.ply']))

