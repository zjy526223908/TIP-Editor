import sys
import os
import argparse
import trimesh
import numpy as np
from colmap_wrapper import run_colmap, load_colmap_data
parser = argparse.ArgumentParser()
parser.add_argument('--match_type', type=str,
                    default='exhaustive_matcher', help='type of matcher used.  Valid options: \
					exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
parser.add_argument('scenedir', type=str,
                    help='input scene directory')
args = parser.parse_args()

if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
    print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
    sys.exit()

if __name__ == '__main__':
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(args.scenedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(args.scenedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print('Need to run COLMAP')
        run_colmap(args.scenedir, args.match_type)
    else:
        print('Don\'t need to run COLMAP')

    print('Post-colmap')

    poses, pts3d, perm = load_colmap_data(args.scenedir)

    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                sys.eixt()
            cams[ind - 1] = 1
        vis_arr.append(cams)

    pts = np.stack(pts_arr, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(args.scenedir, 'sparse_points.ply'))
