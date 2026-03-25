# ====== Python 标准库 ======
import os
import numpy as np
import random
import math
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
from pytorch_msssim import ssim
# ====== 进度条======
from tqdm import tqdm

ROOT = "/root/RM/data"
SCAFFOLD_ROOT = "/root/RM/data/scaffoldfull"

#DIR_ROOT  = r"/dev/shm/directionmap"
DIST_ROOT =r"/root/RM/data/distance_map_DPM"
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

        # ===== geo / tx =====
        geo_build = self._load_npy(f"{ROOT}/geo_npy/{scene_id}.npy")
        geo_car   = self._load_npy(f"{ROOT}/cars_npy/{scene_id}.npy")
        geo = geo_build
        # geo = torch.maximum(geo_build, geo_car)

        tx  = self._load_npy(f"{ROOT}/antenna_npy/{scene_id}_{tx_id}.npy")
        rss = self._load_npy(f"{ROOT}/RSS_DPM/{scene_id}_{tx_id}.npy")

        # ===== dist =====
        dist_data = np.load(f"/dev/shm/distance_map_DPM/{scene_id}_{tx_id}.npz")
        dist = dist_data["dist_map"]  # (H,W,C)

        num_channels = 1   # ← 在这里控制通道数

        if dist.ndim == 3:
            dist = dist[..., :num_channels]          # (H,W,C)
            dist = np.transpose(dist, (2, 0, 1))     # (C,H,W)
        else:
            dist = dist[None, ...]                   # (1,H,W)

        dist = torch.from_numpy(dist.astype(np.float32))  # (C,H,W)
        # ===== 找 TX 坐标（稳定版）=====
        coords = torch.nonzero(tx)

        H, W = tx.shape

        if coords.numel() == 0:
            # 没找到 TX
            dist_euc = torch.zeros((H, W), dtype=torch.float32)
            tx_rss_map = torch.zeros((H, W), dtype=torch.float32)
        else:
            ty, tx_pos = coords[0]
            ty = ty.item()
            tx_pos = tx_pos.item()

            # ===== 欧几里得距离 =====
            yy = torch.arange(H).view(H, 1).expand(H, W)
            xx = torch.arange(W).view(1, W).expand(H, W)

            dist_euc = torch.sqrt((yy - ty)**2 + (xx - tx_pos)**2)
            dist_euc = dist_euc / math.sqrt(H**2 + W**2)

            # ===== TX点RSS通道（单点）=====
            tx_rss_val = rss[ty, tx_pos]

            tx_rss_map = torch.zeros((H, W), dtype=torch.float32)
            tx_rss_map[ty, tx_pos] = tx_rss_val
        # ===== special =====
        special = np.load(f"{SCAFFOLD_ROOT}/{scene_id}_{tx_id}.npy")
        if special.ndim == 3:
            special = special[:, :, 0]
        special = torch.from_numpy(special).float()

        # ===== 统一成 channel-first =====
        geo = geo.unsqueeze(0)   # (1,H,W)
        tx  = tx.unsqueeze(0)    # (1,H,W)
        dist_euc = dist_euc.unsqueeze(0)  # (1,H,W)
        tx_rss_map = tx_rss_map.unsqueeze(0)
        # ===== 拼接 =====
        x = torch.cat([geo, tx], dim=0)

        return x, rss, special

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
# 残差链接块(没用)
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
    def __init__(self, in_ch=2, base_ch=32):
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
# Validation
# =========================================================
def validate(model, loader, device):

    model.eval()

    err_sum = 0.0
    signal_sum = 0.0
    pixel_count = 0
    ssim_total = 0.0
    batch_count = 0

    with torch.no_grad():

        for x, rss,  special in loader:

            x = x.to(device)
            rss = rss.to(device).unsqueeze(1)

            with torch.amp.autocast("cuda"):
                out = model(x)

            # ----- error -----
            err = (out - rss) ** 2
            err_sum += err.sum().item()
            signal_sum += (rss ** 2).sum().item()
            pixel_count += err.numel()

            # ----- SSIM (avoid AMP dtype issue) -----
            ssim_val = ssim(
                out.float(),
                rss.float(),
                data_range=1.0,
                size_average=True
            )

            ssim_total += ssim_val.item()
            batch_count += 1

    # ----- metrics -----

    mse = err_sum / pixel_count
    nmse = err_sum / signal_sum
    rmse = math.sqrt(mse)

    psnr = 10 * math.log10(1.0 / mse)

    ssim_mean = ssim_total / batch_count

    return nmse, rmse, ssim_mean, psnr
# =========================================================
# Visualization
# =========================================================
def visualize(model, sample, device, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    x, rss,  special = sample

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
    patience = 3
    min_lr = 1e-5
    lr_decay = 0.5

    for epoch in range(1, epochs + 1):

        model.train()
        total = 0.0

        log_grad = (epoch % 500 == 0)

        for batch_idx, (x, rss,  special) in enumerate(tqdm(train_loader, mininterval=2.0)):

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


                loss = mse 
            
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
        # Validation (every 3 epochs)
        # =========================
        if epoch % 1 == 0:

            nmse, rmse, ssim_val, psnr = validate(model, val_loader, device)

            print(
                f"[VAL] Epoch {epoch:03d} | "
                f"NMSE {nmse:.4e} | "
                f"RMSE {rmse:.4e} | "
                f"SSIM {ssim_val:.4f} | "
                f"PSNR {psnr:.2f}"
            )

            current_lr = optimizer.param_groups[0]['lr']

            if rmse < best_val:
                best_val = rmse
                no_improve_count = 0
                '''
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_nmse": nmse,
                    "val_rmse": rmse,
                    "val_ssim": ssim_val,
                    "val_psnr": psnr
                }, "RadiomapNetbest_model.pt")
                '''
            else:
                no_improve_count += 1
                print(f">>> no_improve_count {no_improve_count}")

            if no_improve_count >= patience:
                new_lr = current_lr * lr_decay

                if new_lr < min_lr:
                    print("Learning rate reached minimum. Stopping training.")
                    break

                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

                print(f">>> LR reduced to {new_lr:.2e}")
                no_improve_count = 0


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
    epochs = 300
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
    print("全数据，而且严格验证集不可见,消融实验，baseline")
    print("=" * 60)


    # ======================================================
    # 2. Dataset & Scene-level Split (Fixed Split + Subsample)
    # ======================================================

    dataset = RMDataset(index_file)
    index_array = dataset.index

    # 固定可视化样本
    VIS_IDX = 525
    vis_sample = dataset[VIS_IDX]
    vis_scene = index_array[VIS_IDX][0]

    print(f"VIS scene: {vis_scene}")

    # ------------------------------------------------------
    # 使用固定 split（替换随机划分）
    # ------------------------------------------------------
    split_file = f"{ROOT}/minors/index_split.npy"

    if not os.path.exists(split_file):
        raise RuntimeError(
            f"Split file not found: {split_file}"
        )

    split_dict = np.load(split_file, allow_pickle=True).item()

    train_scene = set(split_dict["train_scene"])
    val_scene   = set(split_dict["val_scene"])

    # 保证 VIS 在 val
    if vis_scene not in val_scene:
        train_scene.discard(vis_scene)
        val_scene.add(vis_scene)

    # ------------------------------------------------------
    # 根据 scene 生成 sample 索引
    # ------------------------------------------------------
    train_indices = [
        i for i, (s, _) in enumerate(index_array)
        if s in train_scene
    ]

    val_indices = [
        i for i, (s, _) in enumerate(index_array)
        if s in val_scene
    ]

    # ------------------------------------------------------
    # 检查是否有 scene 泄漏
    # ------------------------------------------------------
    train_scenes = set(index_array[i][0] for i in train_indices)
    val_scenes   = set(index_array[i][0] for i in val_indices)

    intersection = train_scenes & val_scenes

    print("Scene intersection:", intersection)
    print("Number of overlapping scenes:", len(intersection))
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples  : {len(val_indices)}")

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise RuntimeError("Train/Val 划分结果为空")

    # ======================================================
    # 2.5 Subsample training data (scene-balanced)
    # ======================================================

    train_ratio = 1
    np.random.seed(323)

    train_indices_set = set(train_indices)
    train_indices_balanced = []

    for s in train_scene:

        # 当前 scene 的所有 sample
        scene_ids = [
            i for i, (scene, _) in enumerate(index_array)
            if scene == s
        ]

        # 只保留 train 内的
        scene_ids = [i for i in scene_ids if i in train_indices_set]

        if len(scene_ids) == 0:
            continue

        n = max(1, int(len(scene_ids) * train_ratio))

        selected = np.random.choice(scene_ids, size=n, replace=False)

        train_indices_balanced.extend(selected)

    train_indices = np.array(train_indices_balanced)

    print("After scene-balanced subsampling:")
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples  : {len(val_indices)}")

    # ======================================================
    # DataLoader
    # ======================================================

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
    model = UNet(in_ch=2, base_ch=32).to(device)
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