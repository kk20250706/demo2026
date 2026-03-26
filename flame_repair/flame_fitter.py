import torch
import torch.nn as nn
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm
from .flame_model import FLAMELayer


class FLAMEFitter:

    def __init__(self, flame_model_path: str, device: str = "auto", n_shape: int = 100, n_exp: int = 50):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[FLAME] Using device: {self.device}")

        self.n_shape = n_shape
        self.n_exp = n_exp
        self.flame = FLAMELayer(flame_model_path, n_shape=n_shape, n_exp=n_exp).to(self.device)
        self.flame.eval()

    def fit(
        self,
        target_mesh: trimesh.Trimesh,
        n_iters: int = 500,
        lr: float = 0.01,
        w_chamfer: float = 1.0,
        w_landmark: float = 0.0,
        w_reg_shape: float = 1e-4,
        w_reg_exp: float = 1e-3,
        w_reg_pose: float = 1e-3,
    ) -> dict:
        target_verts = torch.tensor(target_mesh.vertices, dtype=torch.float32, device=self.device)

        center = target_verts.mean(dim=0)
        scale = (target_verts - center).norm(dim=1).max()
        target_verts_norm = (target_verts - center) / scale

        template_verts = self.flame.v_template.clone()
        t_center = template_verts.mean(dim=0)
        t_scale = (template_verts - t_center).norm(dim=1).max()

        shape_params = nn.Parameter(torch.zeros(1, self.n_shape, device=self.device))
        exp_params = nn.Parameter(torch.zeros(1, self.n_exp, device=self.device))
        pose_params = nn.Parameter(torch.zeros(1, 15, device=self.device))  # 5 joints * 3
        global_trans = nn.Parameter(torch.zeros(1, 3, device=self.device))
        global_scale = nn.Parameter(torch.ones(1, 1, device=self.device))

        optimizer = torch.optim.Adam([
            {"params": [shape_params], "lr": lr},
            {"params": [exp_params], "lr": lr},
            {"params": [pose_params], "lr": lr * 0.5},
            {"params": [global_trans, global_scale], "lr": lr * 2},
        ])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

        print(f"[FLAME] Fitting to target mesh ({len(target_mesh.vertices)} vertices)...")

        best_loss = float("inf")
        best_params = None

        for i in tqdm(range(n_iters), desc="FLAME Fitting"):
            optimizer.zero_grad()
            pred_verts = self.flame(shape_params, exp_params, pose_params)
            pred_verts_norm = (pred_verts[0] - t_center) / t_scale
            pred_verts_transformed = pred_verts_norm * global_scale + global_trans

            loss_chamfer = chamfer_distance_kdtree(pred_verts_transformed, target_verts_norm)
            loss_reg_shape = (shape_params ** 2).mean()
            loss_reg_exp = (exp_params ** 2).mean()
            loss_reg_pose = (pose_params ** 2).mean()

            loss = (
                w_chamfer * loss_chamfer
                + w_reg_shape * loss_reg_shape
                + w_reg_exp * loss_reg_exp
                + w_reg_pose * loss_reg_pose
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {
                    "shape": shape_params.detach().clone(),
                    "exp": exp_params.detach().clone(),
                    "pose": pose_params.detach().clone(),
                    "trans": global_trans.detach().clone(),
                    "scale": global_scale.detach().clone(),
                }

            if (i + 1) % 100 == 0:
                print(f"  [iter {i+1}] loss={loss.item():.6f} chamfer={loss_chamfer.item():.6f}")

        with torch.no_grad():
            final_verts = self.flame(best_params["shape"], best_params["exp"], best_params["pose"])
            final_verts_norm = (final_verts[0] - t_center) / t_scale
            final_verts_out = final_verts_norm * best_params["scale"] + best_params["trans"]
            final_verts_out = final_verts_out * scale + center

        faces_np = self.flame.faces.cpu().numpy()
        verts_np = final_verts_out.cpu().numpy()

        fitted_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        print(f"[FLAME] Fitting complete. Best loss: {best_loss:.6f}")

        return {
            "mesh": fitted_mesh,
            "params": {k: v.cpu().numpy() for k, v in best_params.items()},
            "loss": best_loss,
        }


def chamfer_distance_kdtree(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 3:
        pred = pred[0]
    if target.dim() == 3:
        target = target[0]

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    tree_target = cKDTree(target_np)
    dists_p2t, idx_p2t = tree_target.query(pred_np)

    tree_pred = cKDTree(pred_np)
    dists_t2p, idx_t2p = tree_pred.query(target_np)

    closest_target = target[torch.tensor(idx_p2t, dtype=torch.long, device=pred.device)]
    loss_p2t = ((pred - closest_target) ** 2).sum(dim=-1).mean()

    closest_pred = pred[torch.tensor(idx_t2p, dtype=torch.long, device=target.device)]
    loss_t2p = ((target - closest_pred) ** 2).sum(dim=-1).mean()

    return loss_p2t + loss_t2p
