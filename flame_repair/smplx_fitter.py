import torch
import torch.nn as nn
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm
from pathlib import Path


class SMPLXFitter:

    REQUIRED_FILES = [
        "SMPLX_NEUTRAL.npz",
        "SMPLX_MALE.npz",
        "SMPLX_FEMALE.npz",
    ]

    def __init__(self, model_dir: str, gender: str = "neutral", device: str = "auto"):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[SMPLX] Using device: {self.device}")

        model_dir = Path(model_dir)
        self._check_model_files(model_dir)

        try:
            import smplx
        except ImportError:
            raise ImportError(
                "smplx not installed. Run: pip install smplx\n"
                "Then download models from https://smpl-x.is.tue.mpg.de/"
            )

        self.model = smplx.create(
            str(model_dir),
            model_type="smplx",
            gender=gender,
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
            num_expression_coeffs=10,
        ).to(self.device)
        self.model.eval()

    def _check_model_files(self, model_dir: Path):
        missing = [f for f in self.REQUIRED_FILES if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"SMPL-X model files missing in {model_dir}:\n  " + "\n  ".join(missing) +
                "\nDownload from: https://smpl-x.is.tue.mpg.de/download.php"
            )

    def fit(
        self,
        target_mesh: trimesh.Trimesh,
        n_iters: int = 300,
        lr: float = 0.01,
    ) -> dict:
        target_verts_np = np.asarray(target_mesh.vertices, dtype=np.float64)

        center = target_verts_np.mean(axis=0)
        scale = float(np.linalg.norm(target_verts_np - center, axis=1).max())
        target_norm = (target_verts_np - center) / scale

        target_t = torch.tensor(target_norm, dtype=torch.float32, device=self.device)

        global_orient = nn.Parameter(torch.zeros(1, 3, device=self.device))
        transl = nn.Parameter(torch.zeros(1, 3, device=self.device))
        global_scale = nn.Parameter(torch.ones(1, 1, device=self.device))
        body_pose = nn.Parameter(torch.zeros(1, 63, device=self.device))
        betas = nn.Parameter(torch.zeros(1, 10, device=self.device))
        expression = nn.Parameter(torch.zeros(1, 10, device=self.device))

        stage_configs = [
            {"params": [global_orient, transl, global_scale], "iters": n_iters // 3},
            {"params": [global_orient, transl, global_scale, body_pose], "iters": n_iters // 3},
            {"params": [global_orient, transl, global_scale, body_pose, betas, expression], "iters": n_iters - 2 * (n_iters // 3)},
        ]

        best_loss = float("inf")
        best_verts = None

        for stage_idx, stage in enumerate(stage_configs):
            optimizer = torch.optim.Adam(stage["params"], lr=lr if stage_idx < 2 else lr * 0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stage["iters"] // 2, gamma=0.5)

            print(f"[SMPLX] Stage {stage_idx + 1}/3 ({stage['iters']} iters)...")
            for i in tqdm(range(stage["iters"]), desc=f"SMPLX Stage {stage_idx+1}", leave=False):
                optimizer.zero_grad()

                output = self.model(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    expression=expression,
                    return_verts=True,
                )
                pred_verts = output.vertices[0]

                t_center = pred_verts.mean(dim=0)
                t_scale = (pred_verts - t_center).norm(dim=1).max()
                pred_norm = (pred_verts - t_center) / (t_scale + 1e-8)
                pred_transformed = pred_norm * global_scale + transl

                loss_chamfer = _chamfer_loss(pred_transformed, target_t)
                loss_reg = (
                    1e-3 * (body_pose ** 2).mean()
                    + 1e-4 * (betas ** 2).mean()
                    + 1e-3 * (expression ** 2).mean()
                )
                loss = loss_chamfer + loss_reg
                loss.backward()
                optimizer.step()
                scheduler.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_verts = pred_transformed.detach().clone()

            if (stage_idx + 1) % 1 == 0:
                print(f"  [Stage {stage_idx+1}] loss={best_loss:.6f}")

        with torch.no_grad():
            final_verts_np = best_verts.cpu().numpy()
            final_verts_np = final_verts_np * scale + center

        faces_np = self.model.faces.astype(np.int64)
        fitted_mesh = trimesh.Trimesh(vertices=final_verts_np, faces=faces_np, process=False)
        fitted_mesh.fix_normals()

        print(f"[SMPLX] Fitting complete. Best loss: {best_loss:.6f}")
        return {"mesh": fitted_mesh, "loss": best_loss}


def _chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    tree_t = cKDTree(target_np)
    dists_p2t, idx_p2t = tree_t.query(pred_np)

    tree_p = cKDTree(pred_np)
    dists_t2p, idx_t2p = tree_p.query(target_np)

    closest_t = target[torch.tensor(idx_p2t, dtype=torch.long, device=pred.device)]
    loss_p2t = ((pred - closest_t) ** 2).sum(dim=-1).mean()

    closest_p = pred[torch.tensor(idx_t2p, dtype=torch.long, device=target.device)]
    loss_t2p = ((target - closest_p) ** 2).sum(dim=-1).mean()

    return loss_p2t + loss_t2p
