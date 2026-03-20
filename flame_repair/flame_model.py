import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path


class FLAMELayer(nn.Module):

    def __init__(self, flame_model_path: str, n_shape: int = 300, n_exp: int = 100):
        super().__init__()
        with open(flame_model_path, "rb") as f:
            flame_model = pickle.load(f, encoding="latin1")

        def to_np(x):
            return np.array(x) if not isinstance(x, np.ndarray) else x

        self.dtype = torch.float32
        self.register_buffer("v_template", torch.tensor(to_np(flame_model["v_template"]), dtype=self.dtype))
        self.register_buffer("faces", torch.tensor(to_np(flame_model["f"]).astype(np.int64), dtype=torch.long))

        shapedirs = torch.tensor(to_np(flame_model["shapedirs"]), dtype=self.dtype)
        self.register_buffer("shapedirs_shape", shapedirs[:, :, :n_shape])
        self.register_buffer("shapedirs_exp", shapedirs[:, :, 300 : 300 + n_exp])

        self.register_buffer("J_regressor", torch.tensor(np.array(flame_model["J_regressor"].todense()), dtype=self.dtype))

        posedirs = to_np(flame_model["posedirs"])
        num_pose_basis = posedirs.shape[-1]
        posedirs = torch.tensor(posedirs, dtype=self.dtype)
        self.register_buffer("posedirs", posedirs.reshape(-1, num_pose_basis))

        parents = to_np(flame_model["kintree_table"])[0].astype(np.int64)
        parents[0] = -1
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))

        lbs_weights = torch.tensor(to_np(flame_model["weights"]), dtype=self.dtype)
        self.register_buffer("lbs_weights", lbs_weights)

        self.n_shape = n_shape
        self.n_exp = n_exp

    def forward(self, shape_params, expression_params, pose_params):
        batch_size = shape_params.shape[0]
        v = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        v = v + torch.einsum("bl,mkl->bmk", shape_params, self.shapedirs_shape)
        v = v + torch.einsum("bl,mkl->bmk", expression_params, self.shapedirs_exp)

        J = torch.einsum("jv,bvk->bjk", self.J_regressor, v)

        rot_mats = batch_rodrigues(pose_params.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)

        pose_feature = (rot_mats[:, 1:, :, :] - torch.eye(3, device=v.device).unsqueeze(0).unsqueeze(0)).reshape(batch_size, -1)
        v = v + torch.matmul(pose_feature, self.posedirs.T).reshape(batch_size, -1, 3)

        v_posed = lbs(v, J, rot_mats, self.parents, self.lbs_weights)
        return v_posed


def batch_rodrigues(rot_vecs):
    batch_size = rot_vecs.shape[0]
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    rx, ry, rz = rot_dir[:, 0:1], rot_dir[:, 1:2], rot_dir[:, 2:3]
    zeros = torch.zeros_like(rx)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).reshape(-1, 3, 3)

    eye = torch.eye(3, device=rot_vecs.device).unsqueeze(0)
    rot_mat = eye + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def lbs(v, J, rot_mats, parents, weights):
    batch_size = v.shape[0]
    num_joints = J.shape[1]

    transforms = torch.zeros(batch_size, num_joints, 4, 4, device=v.device, dtype=v.dtype)

    for i in range(num_joints):
        rot = rot_mats[:, i]
        t = J[:, i].unsqueeze(-1)
        pad = torch.zeros(batch_size, 1, 4, device=v.device, dtype=v.dtype)
        pad[:, :, 3] = 1
        transform = torch.cat([torch.cat([rot, t], dim=-1), pad], dim=1)

        if parents[i] < 0:
            transforms[:, i] = transform
        else:
            transforms[:, i] = torch.bmm(transforms[:, parents[i]].clone(), transform)

    posed_joints = transforms[:, :, :3, 3].clone()
    joint_transforms = transforms.clone()
    joint_rest = torch.cat([J, torch.zeros(batch_size, num_joints, 1, device=v.device, dtype=v.dtype)], dim=-1)
    joint_rest = joint_rest.unsqueeze(-1)
    joint_transforms[:, :, :, 3:] = joint_transforms[:, :, :, 3:] - torch.matmul(joint_transforms, joint_rest)

    T = torch.einsum("bvj,bjmn->bvmn", weights, joint_transforms)
    v_homo = torch.cat([v, torch.ones(batch_size, v.shape[1], 1, device=v.device, dtype=v.dtype)], dim=-1)
    v_posed = torch.matmul(T, v_homo.unsqueeze(-1))[:, :, :3, 0]
    return v_posed
