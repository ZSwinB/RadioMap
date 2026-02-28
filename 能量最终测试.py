# ====== Python 标准库 ======
import os
import numpy as np
import random
# ====== PyTorch 核心 ======
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
import matplotlib.pyplot as plt
# ====== log ======
import sys
from datetime import datetime
# ====== 图像读取 ======
from PIL import Image

# ====== 进度条======
from tqdm import tqdm

ROOT = "/root/RM/data"
SCAFFOLD_ROOT = "/root/RM/data/scaffoldfull"
UPWIND_ROOT = "/dev/shm/upwindmap"
DIR_ROOT  = r"/dev/shm/directionmap"
# =========================================================
# Dataset
# =========================================================
class RMDataset(Dataset):
    def __init__(self, index_file):
        self.index = np.load(index_file)

        

    def __len__(self):
        return len(self.index)

    def _load_png(self, path):
        img = Image.open(path).convert("F")
        img = torch.from_numpy(np.array(img)).float()
        img = img / 255.0
        return img

    def _load_npy(self, path):
        arr = np.load(path)
        return torch.from_numpy(arr).float()

    def __getitem__(self, i):
        scene_id, tx_id = self.index[i]

        geo = self._load_npy(f"{ROOT}/geo_npy/{scene_id}.npy")
        tx  = self._load_npy(f"{ROOT}/antenna_npy/{scene_id}_{tx_id}.npy")
        rss = self._load_npy(f"{ROOT}/RSS_npy/{scene_id}_{tx_id}.npy")

        v = np.load(f"{ROOT}/distance_vector/{scene_id}_{tx_id}.npy")
        v = torch.tensor(v).float()

        dist = np.load(f"/root/RM/pred_dist/{scene_id}_{tx_id}.npy")
        dist = torch.tensor(dist).float()

        special = np.load(f"{SCAFFOLD_ROOT}/{scene_id}_{tx_id}.npy")
        special = torch.tensor(special).float()

        lap_S = np.load(f"/dev/shm/lap_S/{scene_id}_{tx_id}.npy")
        lap_S = torch.tensor(lap_S).float()
        '''
        # ===== 读取 upwind log 数据 =====
        up_data = np.load(f"{UPWIND_ROOT}/{scene_id}_{tx_id}.npz")
        up = torch.from_numpy(up_data["up"]).long()
        beta = torch.from_numpy(up_data["beta"]).float()
        '''
        # ===== 读取 时间梯度 =====
        nx = torch.from_numpy(
            np.load(f"{DIR_ROOT}_npy/{scene_id}_{tx_id}_nx.npy")
        ).float()

        ny = torch.from_numpy(
            np.load(f"{DIR_ROOT}_npy/{scene_id}_{tx_id}_ny.npy")
        ).float()

        #, nx.squeeze(0), ny.squeeze(0)
        x = torch.stack([geo, tx, dist, nx.squeeze(0), ny.squeeze(0)], dim=0)

        return x, rss, v, special, lap_S


# =========================================================
# 空洞卷积
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1,)):
        super().__init__()

        blocks = []
        for d in dilations:
            padding = d

            blocks.append(
                nn.Conv2d(in_ch, out_ch, 3,
                          padding=padding,
                          dilation=d,
                          bias=False)
            )
            blocks.append(nn.BatchNorm2d(out_ch))
            blocks.append(nn.ReLU(inplace=True))

            blocks.append(
                nn.Conv2d(out_ch, out_ch, 3,
                          padding=padding,
                          dilation=d,
                          bias=False)
            )
            blocks.append(nn.BatchNorm2d(out_ch))
            blocks.append(nn.ReLU(inplace=True))

            in_ch = out_ch

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

# =========================================================
# 残差链接块
# =========================================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn1 = nn.GroupNorm(8, out_ch)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.GroupNorm(8, out_ch)

        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.proj = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResStack(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1,)):
        super().__init__()

        blocks = []
        for i, d in enumerate(dilations):
            if i == 0:
                blocks.append(ResBlock(in_ch, out_ch, dilation=d))
            else:
                blocks.append(ResBlock(out_ch, out_ch, dilation=d))

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)




# =========================================================
# 网络
# =========================================================
class UNet(nn.Module):
    def __init__(self, in_ch=5, base_ch=32):
        super().__init__()
        # 小尺度残差分支
        self.local_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 1)
        )

        # 可学习权重
        self.alpha = torch.tensor(0,requires_grad=False)

        self.down1 = DoubleConv(in_ch, base_ch, (1,))
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base_ch, base_ch*2, (1,))
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base_ch*2, base_ch*4, (1,2))
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(base_ch*4, base_ch*8, (1,2))
        self.pool4 = nn.MaxPool2d(2)

        self.mid = DoubleConv(base_ch*8, base_ch*16, (1,2,4))

        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, 2)
        self.conv4 = DoubleConv(base_ch*16, base_ch*8, (1,))

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.conv3 = DoubleConv(base_ch*8, base_ch*4, (1,))

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.conv2 = DoubleConv(base_ch*4, base_ch*2, (1,))

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.conv1 = DoubleConv(base_ch*2, base_ch, (1,))

        self.out = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        m = self.mid(self.pool4(d4))

        u4 = self.conv4(torch.cat([self.up4(m), d4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))

        main_out = self.out(u1)

        local_out = self.local_head(d1)

        return main_out 


# =========================================================
# 中心差分方向损失（只在 normal 上）
# =========================================================
class CosineDirectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, U, v, special):

        normal = (special == 0)

        # ===== 中心差分 =====
        Tx = (U[:, :, 2:] - U[:, :, :-2]) * 0.5
        Ty = (U[:, 2:, :] - U[:, :-2, :]) * 0.5

        # 对齐尺寸 (B,H-2,W-2)
        Tx = Tx[:, 1:-1, :]
        Ty = Ty[:, :, 1:-1]

        vx = v[:, 0, 1:-1, 1:-1]
        vy = v[:, 1, 1:-1, 1:-1]

        normal_c = normal[:, 1:-1, 1:-1]

        dot = Tx * vx + Ty * vy
        alpha = 0.07
        loss_map = -dot + alpha * (Tx**2 + Ty**2)

        if normal_c.any():
            return loss_map[normal_c].mean()
        else:
            return torch.tensor(0.0, device=U.device)
# =========================================================
# transport损失（只在 normal 上）
# =========================================================
class TransportLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, U, v, lap_S, special):

        normal = (special == 0)

        # ===== ∇U（中心差分）=====
        Ux = (U[:, :, 2:] - U[:, :, :-2]) * 0.5
        Uy = (U[:, 2:, :] - U[:, :-2, :]) * 0.5

        Ux = Ux[:, 1:-1, :]
        Uy = Uy[:, :, 1:-1]

        # ===== 对齐 v =====
        vx = v[:, 0, 1:-1, 1:-1]
        vy = v[:, 1, 1:-1, 1:-1]

        lap = lap_S[:, 1:-1, 1:-1]

        normal_c = normal[:, 1:-1, 1:-1]

        residual = Ux * vx + Uy * vy + lap

        if normal_c.any():
            return (residual[normal_c] ** 2).mean()
        else:
            return torch.tensor(0.0, device=U.device)

# =========================================================
# 因果损失（只在 normal 上）
# =========================================================

class CausalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u_pred, up, beta, special):

        B, H, W = u_pred.shape

        u_flat = u_pred.view(B, -1)
        up_flat = up.view(B, -1)
        special_flat = special.view(B, -1)

        valid = (up_flat >= 0) & (special_flat < 0.5)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=u_pred.device)

        # gather 上游
        u_up = u_flat.gather(1, up_flat.clamp(min=0))

        u_cur = u_flat[valid]
        u_up  = u_up[valid]

        # 只惩罚逆流
        loss = torch.relu(u_cur - u_up).pow(2)



        return loss.mean()

# =========================================================
# sobolev
# =========================================================

def gradient_loss(pred, gt, normal_mask):
    # pred, gt: (B, 1, H, W)
    # normal_mask: (B, 1, H, W)

    # 中心差分
    dx_p = pred[:, :, :, 2:] - pred[:, :, :, :-2]
    dx_g = gt[:, :, :, 2:] - gt[:, :, :, :-2]

    dy_p = pred[:, :, 2:, :] - pred[:, :, :-2, :]
    dy_g = gt[:, :, 2:, :] - gt[:, :, :-2, :]

    # mask 对齐（去掉边界两圈）
    mask_x = normal_mask[:, :, :, 1:-1]
    mask_y = normal_mask[:, :, 1:-1, :]

    loss_x = ((dx_p - dx_g).pow(2) * mask_x).sum() / (mask_x.sum() + 1e-6)
    loss_y = ((dy_p - dy_g).pow(2) * mask_y).sum() / (mask_y.sum() + 1e-6)

    return 0.5 * (loss_x + loss_y)
# =========================================================
# Validation
# =========================================================
def validate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total = 0.0

    with torch.no_grad():
        for x, rss, v, special, lap_S in loader:

            x = x.to(device)
            rss = rss.to(device).unsqueeze(1)

            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, rss)

            total += loss.item()

    return total / len(loader)

# =========================================================
# Visualization
# =========================================================
def visualize(model, sample, device, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    x, rss, v, special, lap_S = sample

    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    pred = out[0,0].cpu().numpy()
    gt   = rss.numpy()

    pred = (pred * 255).clip(0,255).astype("uint8")
    gt   = (gt * 255).clip(0,255).astype("uint8")

    fig, axs = plt.subplots(1,2,figsize=(8,4))
    axs[0].imshow(gt, cmap="gray")
    axs[1].imshow(pred, cmap="gray")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch:04d}.svg", dpi=200)
    plt.close()


# =========================================================
# Training
# =========================================================
def train(model, train_loader, val_loader, device, epochs, vis_sample):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.MSELoss()

    best_val = float("inf")
    no_improve_count = 0
    patience = 4
    min_lr = 1e-6
    lr_decay = 0.5

    for epoch in range(1, epochs + 1):

        model.train()
        total = 0.0

        log_grad = (epoch % 3 == 0)

        for batch_idx, (x, rss, v, special, lap_S) in enumerate(tqdm(train_loader)):

            x = x.to(device)
            rss = rss.to(device)
            special = special.to(device)

            optimizer.zero_grad()

            # =========================
            # Forward (AMP)
            # =========================
            with torch.cuda.amp.autocast():

                out = model(x)

                mse = criterion(out, rss.unsqueeze(1))

                normal_mask = (1.0 - special).unsqueeze(1)
                grad_loss = gradient_loss(out, rss.unsqueeze(1), normal_mask)

                loss = mse + 0 * grad_loss

            # =====================================================
            # 梯度统计（每3个epoch，第一个batch）
            # =====================================================
            if log_grad and batch_idx == 0:

                # ---------- MSE 梯度 ----------
                optimizer.zero_grad()
                scaler.scale(mse).backward(retain_graph=True)

                mse_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        mse_grad_norm += p.grad.detach().norm().item()

                # ---------- Grad Loss 梯度 ----------
                optimizer.zero_grad()
                scaler.scale(grad_loss).backward(retain_graph=True)

                grad_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_grad_norm += p.grad.detach().norm().item()

                print(f"[GradStat][Epoch {epoch}] "
                      f"MSE_grad = {mse_grad_norm:.4e} | "
                      f"Grad_grad = {grad_grad_norm:.4e}")

                optimizer.zero_grad()

            # =========================
            # 正常反向传播
            # =========================
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total += loss.item()

        print(f"Epoch {epoch:03d} | Loss {total/len(train_loader):.6f}")
        print("alpha =", model.alpha.item())

        # =========================
        # Validation
        # =========================
        val = validate(model, val_loader, device)
        print(f"Val MSE {val:.5e}")

        current_lr = optimizer.param_groups[0]['lr']

        if val < best_val:
            best_val = val
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            new_lr = current_lr * lr_decay

            if new_lr < min_lr:
                print("Learning rate reached minimum. Stopping training.")
                break

            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            print(f">>> LR reduced to {new_lr:.2e}")
            no_improve_count = 0

        visualize(model, vis_sample, device, epoch, "./最终vis")

# =========================================================
# Main
# =========================================================
def main():

    # ======================================================
    # 1. Config
    # ======================================================
    ROOT = "/root/RM/data"
    index_file = f"{ROOT}/minors/index.npy"

    batch_size = 8
    epochs = 100
    lr = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Training Configuration")
    print(f"ROOT       : {ROOT}")
    print(f"Index file : {index_file}")
    print(f"Batch size : {batch_size}")
    print(f"Epochs     : {epochs}")
    print(f"LR         : {lr}")
    print(f"Device     : {device}")
    print("=" * 60)
    print("现在是全数据，但是加了两个通道")
    print("=" * 60)

    # ======================================================
    # 2. Dataset & Scene-level Split
    # ======================================================
    dataset = RMDataset(index_file)
    index_array = dataset.index

    # 固定可视化样本
    VIS_IDX = 1234
    vis_sample = dataset[VIS_IDX]
    vis_scene = index_array[VIS_IDX][0]

    # 所有 scene
    all_scenes = list(set(index_array[:, 0]))

    # 去掉 VIS 场景（必须进 val）
    remaining_scenes = [s for s in all_scenes if s != vis_scene]

    random.shuffle(remaining_scenes)

    n_total_scene = len(all_scenes)
    n_val_scene = int(0.2 * n_total_scene)

    # 已经占 1 个 vis_scene
    n_extra_scene = n_val_scene - 1

    val_scene = set([vis_scene] + remaining_scenes[:n_extra_scene])
    train_scene = set(remaining_scenes[n_extra_scene:])

    # 根据 scene 生成 sample 索引
    train_indices = [
        i for i, (s, _) in enumerate(index_array)
        if s in train_scene
    ]

    val_indices = [
        i for i, (s, _) in enumerate(index_array)
        if s in val_scene
    ]

    # 检查是否泄漏
    train_scenes = set(index_array[i][0] for i in train_indices)
    val_scenes   = set(index_array[i][0] for i in val_indices)
    intersection = train_scenes & val_scenes

    print("Scene intersection:", intersection)
    print("Number of overlapping scenes:", len(intersection))
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples  : {len(val_indices)}")

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set   = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ======================================================
    # 3. Model
    # ======================================================
    model = UNet(in_ch=5, base_ch=32).to(device)
    print(f"Model: {model.__class__.__name__}")

    # ======================================================
    # 4. Training (Single Stage)
    # ======================================================
    train(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        vis_sample
    )

    torch.save(model.state_dict(), "RadioNet_final.pt")
    print("Saved: RadioNet_final.pt")

    print("\nTraining finished.")
if __name__ == "__main__":
    main()