import random
import torch
import math
import os
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from models.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, \
    read_extrinsics_text, read_intrinsics_text, qvec2rotmat

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front) & (phis > (2 * np.pi - front))] = 0
    res[(phis >= front) & (phis < (np.pi - front))] = 1
    res[(phis >= (np.pi - front)) & (phis < (np.pi + front))] = 2

    res[(phis >= (np.pi + front)) & (phis <= (2 * np.pi - front))] = 3
    # override by thetas

    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def circle_poses(device, radius=1.25, theta=60, phi=0, angle_overhead=30, angle_front=60):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = - safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).to(device)

    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)

    return poses, dirs

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def get_center_and_diag(cam_centers):
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, R, T, width, height, fovy, fovx, pose, znear, zfar):
        # c2w (pose) should be in NeRF convention.


        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0

        w2c = np.linalg.inv(pose)
        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(getWorld2View2(R.numpy(), T.numpy(), trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]



class SphericalSamplingDataset:
    def __init__(self, opt, device, R_path=None, type='train', H=512, W=512, size=100):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range
        self.phi_range = opt.phi_range
        self.theta_range = opt.theta_range
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.R_path = R_path

    def collate(self, index):
        # circle pose
        H_list = []
        W_list = []
        R_list = []
        T_list = []
        fov_list = []
        poses_list = []

        for index_i in index:
            if self.training:

                if self.opt.pose_sample_strategy == 'triangular':
                    phi = random.triangular(self.phi_range[0], self.phi_range[1],
                                            0.5 * (self.phi_range[0] + self.phi_range[1]))
                    theta = random.triangular(self.theta_range[0], self.theta_range[1],
                                              0.5 * (self.phi_range[0] + self.phi_range[1]))
                    radius = random.triangular(self.radius_range[0], self.radius_range[1],
                                               0.5 * (self.phi_range[0] + self.phi_range[1]))
                else:
                    phi = random.uniform(self.phi_range[0], self.phi_range[1])
                    theta = random.uniform(self.theta_range[0], self.theta_range[1])
                    radius = random.uniform(self.radius_range[0], self.radius_range[1])

                poses, dirs = circle_poses(self.device, radius=radius, theta=theta, phi=phi)

                # random focal
                fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]

            else:
                # circle pose
                phi = self.phi_range[0] + (index_i / self.size) * (self.phi_range[1] - self.phi_range[0])
                theta = 0.5 * (self.theta_range[0] + self.theta_range[1])
                radius = self.radius_range[0] + (index_i / self.size) * (self.radius_range[1] - self.radius_range[0])

                poses, dirs = circle_poses('cpu', radius=radius, theta=theta, phi=phi)

                # fixed focal
                fov = (self.fovy_range[1] + self.fovy_range[0]) / 2

            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            fov = focal2fov(focal, self.H )

            poses = poses[0]
            poses[:3, 1:3] *= -1
            poses = poses.cpu().numpy()
            w2c = np.linalg.inv(poses)

            w2c[0:3, -1] *= -1
            w2c[1:3, 0:3] *= -1

            if self.R_path:
                c2w = np.linalg.inv(w2c)
                R = np.load(self.R_path)
                R = np.linalg.inv(R)
                poses = R @ c2w
                w2c = np.linalg.inv(poses)

            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]


            H_list.append(self.H)
            W_list.append(self.W)
            R_list.append(torch.from_numpy(R))
            T_list.append(torch.from_numpy(T))
            fov_list.append(fov)
            poses_list.append(poses)


        return 0, 0, 0,H_list, W_list, R_list, T_list, fov_list, fov_list, poses_list, index



    def dataloader(self):
        if self.training:
            print(self.opt.batch_size)
            loader = DataLoader(list(range(self.size)), batch_size=self.opt.batch_size, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        else:
            loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate,
                                shuffle=self.training,  num_workers=0)

        return loader

class SceneDataset:
    def __init__(self, opt, device, R_path=None, type='train'):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range

        self.training = self.type in ['train', 'all']

        if self.training:
            resolution_level = self.opt.train_resolution_level
        else:
            resolution_level = self.opt.eval_resolution_level


        self.dataset = COLMAP(data_dir=self.opt.data_path, if_data_cuda=self.opt.if_data_cuda,
                            R_path=R_path, resolution_level=resolution_level, split='train',
                            batch_type=self.opt.train_batch_type, factor=1)


    def dataloader(self):
        if self.training:
            loader = DataLoader(self.dataset, shuffle=True, num_workers=self.opt.num_work,
                                batch_size=self.opt.batch_size, pin_memory=False)
        else:
            loader = DataLoader(self.dataset, shuffle=False, num_workers=0,
                                batch_size=1, pin_memory=False)

        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader

class BaseDataset(Dataset):
    """BaseDataset Base Class."""

    def __init__(self, data_dir, split, if_data_cuda=True, batch_type='all_images', factor=0):
        super(BaseDataset, self).__init__()
        self.near = 2
        self.far = 6
        self.split = split
        self.data_dir = data_dir

        self.batch_type = batch_type
        self.images = None
        self.rays = None
        self.if_data_cuda = if_data_cuda
        self.it = -1
        self.n_examples = 1
        self.factor = factor

    def _flatten(self, x):
        # Always flatten out the height x width dimensions
        x = [y.reshape([-1, y.shape[-1]]) for y in x]
        if self.batch_type == 'all_images':
            # If global batching, also concatenate all data into one list
            out_size = [len(x)] + list(x[0].shape)
            x = np.concatenate(x, axis=0).reshape(out_size)
            if self.if_data_cuda:
                x = torch.tensor(x).cuda()
            else:
                x = torch.tensor(x)
        else:
            if self.if_data_cuda:
                x = [torch.tensor(y).cuda() for y in x]
            else:
                x = [torch.tensor(y) for y in x]
        return x

    def _train_init(self):
        """Initialize training."""

        self._load_renderings()
        self._generate_rays()

        if self.split == 'train':
            self.images = self._flatten(self.images)
            self.masks = self._flatten(self.masks)
            self.origins = self._flatten(self.origins)
            self.directions = self._flatten(self.directions)

            # self.rays = namedtuple_map(self._flatten, self.rays)

    def _val_init(self):
        self._load_renderings()
        self._generate_rays()

        self.images = self._flatten(self.images)
        self.masks = self._flatten(self.masks)
        self.origins = self._flatten(self.origins)
        self.directions = self._flatten(self.directions)

    def _generate_rays(self):
        """Generating rays for all images."""
        raise ValueError('Implement in different dataset.')

    def _load_renderings(self):
        raise ValueError('Implement in different dataset.')

    def __len__(self):

        return self.n_images

    def __getitem__(self, index):

        if self.split == 'val':
            index = (self.it + 1) % self.n_examples
            self.it += 1
        # rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
        # index = torch.tensor(index)

        return self.images[index], self.masks[index], self.origins[index], self.directions[index], self.H[index], \
        self.W[index]

class COLMAP(BaseDataset):
    """Blender Dataset."""

    def __init__(self, data_dir, if_data_cuda=True, R_path=None,
                 resolution_level=1, split='train', batch_type='all_images', factor=0):
        super(COLMAP, self).__init__(data_dir, split, if_data_cuda, batch_type, factor)
        self.resolution_level = resolution_level
        self.near = 0.01
        self.far = 5.0
        self.if_data_cuda = if_data_cuda
        self.R_path = R_path
        print(self.if_data_cuda, batch_type)

        self._train_init()

    def _load_renderings(self):
        """Load images from disk."""
        self.images = []
        self.H = []
        self.W = []

        self.pose_all = []
        self.fx = []
        self.fy = []
        self.R = []
        self.T = []

        try:
            cameras_extrinsic_file = os.path.join(self.data_dir, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.data_dir, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.data_dir, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.data_dir, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        for idx, key in enumerate(cam_extrinsics):
            import sys
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
            sys.stdout.flush()
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]

            height = intr.height
            width = intr.width

            height = int(height / self.resolution_level)
            width = int(width / self.resolution_level)


            uid = intr.id
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
            t = T.reshape([3, 1])
            w2c = np.concatenate([np.concatenate([qvec2rotmat(extr.qvec), t], 1), bottom], 0)
            c2w = np.linalg.inv(w2c)

            if self.R_path:
                R = np.load(self.R_path)
                c2w = R @ c2w
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

            self.pose_all.append(torch.tensor(c2w).float())

            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]/ self.resolution_level
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]/ self.resolution_level
                focal_length_y = intr.params[1]/ self.resolution_level
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            image_path = os.path.join(self.data_dir,'images' ,os.path.basename(extr.name))

            image = Image.open(image_path)
            im_data = np.array(image.convert("RGB"))
            im_data = cv2.resize(im_data, (
                int(im_data.shape[1] / self.resolution_level), int(im_data.shape[0] / self.resolution_level)),
                                 interpolation=cv2.INTER_AREA)

            norm_data = im_data / 255.0

            self.fx.append(FovX)
            self.fy.append(FovY)
            self.R.append(R)
            self.T.append(T)

            self.images.append(torch.tensor(norm_data).float())
            self.H.append(int(image.size[1] / self.resolution_level))
            self.W.append(int(image.size[0] / self.resolution_level))

        print(self.W[0], self.H[0])
        self.masks = self.images

        self.n_images = len(self.images)
        self.pose_all = torch.stack(self.pose_all)
        self.intrinsics_all =  self.pose_all # [n_images, 4, 4]
        self.intrinsics_all_inv =  self.pose_all# [n_images, 4, 4]


        cam_centers = []
        for R, T in zip(self.R, self.T):
            W2C = getWorld2View2(R, T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        center, diagonal = get_center_and_diag(cam_centers)
        self.radius = diagonal * 1.1
        self.translate = -center


    def __getitem__(self, index):

        return self.images[index], self.masks[index], self.H[index],  self.W[index], \
            self.R[index],  self.T[index],  self.fx[index],  self.fy[index], self.pose_all[index], index


    def _train_init(self):
        """Initialize training."""

        self._load_renderings()

        if self.split == 'train':
            self.images = self._flatten(self.images)
            self.masks = self._flatten(self.masks)

    def _val_init(self):
        self._load_renderings()

        self.images = self._flatten(self.images)
        self.masks = self._flatten(self.masks)

class SampleViewsDataset:
    def __init__(self, opt, device, R_path=None, type='train', H=512, W=512):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.fovy = opt.fovy
        self.phi_list = opt.phi_list
        self.theta_list = opt.theta_list
        self.radius_list = opt.radius_list

        self.size = len(self.theta_list) * len(self.phi_list) * len(self.radius_list)

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.R_path = R_path

    def collate(self, index):
        radius = self.radius_list[int(index[0] / (len(self.phi_list) * len(self.theta_list)))]
        tmp_count = index[0] % (len(self.phi_list) * len(self.theta_list))
        phi = self.phi_list[tmp_count % (len(self.phi_list))]
        theta = self.theta_list[int(tmp_count / len(self.phi_list))]

        poses, dirs = circle_poses('cpu', radius=radius, theta=theta, phi=phi)

        # fixed focal
        fov = self.fovy
        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        fov = focal2fov(focal, self.H )

        poses = poses[0]
        poses[:3, 1:3] *= -1
        poses = poses.cpu().numpy()
        w2c = np.linalg.inv(poses)

        w2c[0:3, -1] *= -1
        w2c[1:3, 0:3] *= -1

        if self.R_path:
            c2w = np.linalg.inv(w2c)
            R = np.load(self.R_path)
            R = np.linalg.inv(R)
            poses = R @ c2w
            w2c = np.linalg.inv(poses)

        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        return phi, theta, radius, [self.H], [self.W], [torch.from_numpy(R)], [torch.from_numpy(T)], [fov], [fov], [poses], index


    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        return loader


class RandomRGBSampleViewsDataset:
    def __init__(self, opt, image_root, device, R_path=None, type='train', H=512, W=512):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range
        self.phi_list = opt.phi_list
        self.theta_list = opt.theta_list
        self.image_root = image_root

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.R_path = R_path


        self.name_list = os.listdir(self.image_root)

        self.size = len(self.name_list)

    def collate(self, index):

        name = self.name_list[index[0]]
        # print(name)
        theta, phi, radius = name.replace('.png', '').split('_')

        phi = float(phi)
        theta = float(theta)
        radius = float(radius)
        poses, dirs = circle_poses('cpu', radius=radius, theta=theta, phi=phi)

        # fixed focal
        fov = (self.fovy_range[1] + self.fovy_range[0]) / 2
        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        fov = focal2fov(focal, self.H )

        poses = poses[0]
        poses[:3, 1:3] *= -1
        poses = poses.cpu().numpy()
        w2c = np.linalg.inv(poses)

        w2c[0:3, -1] *= -1
        w2c[1:3, 0:3] *= -1

        if self.R_path:
            c2w = np.linalg.inv(w2c)
            R = np.load(self.R_path)
            R = np.linalg.inv(R)
            poses = R @ c2w
            w2c = np.linalg.inv(poses)

        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]



        image = Image.open(os.path.join(self.image_root, name)).convert('RGB')
        im_data = np.array(image)
        norm_data = im_data / 255.0
        norm_data = torch.tensor(norm_data).float()

        mask = Image.open(os.path.join(self.image_root.replace('refine_views', 'mask'), name)).convert('L')
        mask = np.array(mask)
        mask = mask / 255.0
        mask = torch.tensor(mask).float()

        return norm_data, mask, 0, [self.H], [self.W], [torch.from_numpy(R)], [torch.from_numpy(T)], [fov], [fov], [poses], index

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        return loader
