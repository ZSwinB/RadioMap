import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

ROOT = "/root/RM/data"
DIST_ROOT = "/root/RM/distancemap"

H = 256
W = 256
MAX_DIST = 1000.0


# =========================================================
# Dataset（只做IO）
# =========================================================
class RMDataset(Dataset):
    def __init__(self, index_file, packed_root):
        self.index = np.load(index_file)
        self.packed_root = packed_root

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        scene_id, tx_id = self.index[i]
        path = os.path.join(
            self.packed_root,
            f"{scene_id}_{tx_id}.pt"
        )
        data = torch.load(path, map_location="cpu")

        return (
            data["x"],
            data["dist"],
            data["ty"],
            data["tx_pos"],
            data["special_mask"],
        )


# =========================================================
# UNet
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

            # 第二个 dilation 分支之后 in_ch 要变成 out_ch
            in_ch = out_ch

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# =========================================================
# Dilation-Only 4-Level UNet
# =========================================================
class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()

        # ================= Encoder =================
        self.down1 = DoubleConv(in_ch, base_ch, dilations=(1,))
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base_ch, base_ch*2, dilations=(1,))
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base_ch*2, base_ch*4, dilations=(1,2))
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(base_ch*4, base_ch*8, dilations=(1,2))
        self.pool4 = nn.MaxPool2d(2)

        # ================= Bottleneck =================
        self.mid = DoubleConv(base_ch*8, base_ch*16, dilations=(1,2,4))

        # ==================================================
        # ================= LOW Decoder =====================
        # ==================================================

        self.up4_l = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, 2)
        self.conv4_l = DoubleConv(base_ch*16, base_ch*8, dilations=(1,2))

        self.up3_l = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.conv3_l = DoubleConv(base_ch*8, base_ch*4, dilations=(1,2))

        self.up2_l = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.conv2_l = DoubleConv(base_ch*4, base_ch*2, dilations=(1,))

        self.up1_l = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.conv1_l = DoubleConv(base_ch*2, base_ch, dilations=(1,))

        self.out_l = nn.Sequential(
            nn.Conv2d(base_ch, 1, 1),
            nn.Softplus()
        )

        # ==================================================
        # ================= HIGH Decoder ====================
        # ==================================================

        self.up4_h = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, 2)
        self.conv4_h = DoubleConv(base_ch*16, base_ch*8, dilations=(1,))

        self.up3_h = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.conv3_h = DoubleConv(base_ch*8, base_ch*4, dilations=(1,))

        self.up2_h = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.conv2_h = DoubleConv(base_ch*4, base_ch*2, dilations=(1,))

        self.up1_h = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.conv1_h = DoubleConv(base_ch*2, base_ch, dilations=(1,))

        self.out_h = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):

        # ========== Encoder ==========
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        m = self.mid(self.pool4(d4))

        # ========== LOW ==========
        u4l = self.up4_l(m)
        u4l = self.conv4_l(torch.cat([u4l, d4], dim=1))

        u3l = self.up3_l(u4l)
        u3l = self.conv3_l(torch.cat([u3l, d3], dim=1))

        u2l = self.up2_l(u3l)
        u2l = self.conv2_l(torch.cat([u2l, d2], dim=1))

        u1l = self.up1_l(u2l)
        u1l = self.conv1_l(torch.cat([u1l, d1], dim=1))

        delta_low = self.out_l(u1l)

        # ========== HIGH ==========
        u4h = self.up4_h(m)
        u4h = self.conv4_h(torch.cat([u4h, d4], dim=1))

        u3h = self.up3_h(u4h)
        u3h = self.conv3_h(torch.cat([u3h, d3], dim=1))

        u2h = self.up2_h(u3h)
        u2h = self.conv2_h(torch.cat([u2h, d2], dim=1))

        u1h = self.up1_h(u2h)
        u1h = self.conv1_h(torch.cat([u1h, d1], dim=1))

        delta_high = self.out_h(u1h)

        return delta_low, delta_high


# =========================================================
#  loss
# =========================================================
def loss_anchor(T_pred, dist_euclid, los_mask):

    return (
        ((T_pred - dist_euclid)**2) * los_mask
    ).sum() / (los_mask.sum() + 1e-8)


def loss_floor(grad_norm):

    return torch.relu(1.0 - grad_norm).pow(2).mean()

def loss_shape(grad_norm, special_mask, grad_gt_norm, valid_mask, max_dist):

    target_normal = 1.0 / max_dist

    target = torch.ones_like(grad_norm) * target_normal
    target = target * (1.0 - special_mask) + grad_gt_norm * special_mask

    return (
        (((grad_norm - target)/target+1e-8)**2) * valid_mask
    ).sum() / (valid_mask.sum() + 1e-8)


def loss_shape_normal(grad_norm, normal_mask, target_normal):
    return (
        (((grad_norm - target_normal) / (target_normal + 1e-8))**2)
        * normal_mask
    ).sum() / (normal_mask.sum() + 1e-8)
'''
def loss_shape_special(grad_norm, grad_gt_norm, special_mask):
    return (
        (((grad_norm - grad_gt_norm) / (grad_gt_norm + 1e-8))**2)
        * special_mask
    ).sum() / (special_mask.sum() + 1e-8)
'''
def loss_shape_special(
    grad_norm,
    special_mask,
    grad_gt_norm,
    alpha=1.0,     # 控制空间阈值强度
):
    """
    grad_norm:        |∇T|  (B,1,H,W)
    special_mask:     special 区域 (0/1)
    grad_gt_norm:     GT 梯度模长 (用于结构阈值)
    """

    eps = 1e-8

    # 1️⃣ 结构阈值（空间相关）
    # 在 special 上阈值随 GT 梯度增大
    threshold = 1.0 + alpha * grad_gt_norm.detach()

    # 2️⃣ 单侧 barrier
    penalty = torch.relu(threshold - grad_norm)

    # 3️⃣ 只在 special 区域生效
    weighted = penalty * special_mask

    return (weighted**2).sum() / (special_mask.sum() + eps)





def loss_fold(dx, dy, special_mask):
    eps = 1e-8

    norm = torch.sqrt(dx**2 + dy**2 + eps)
    ux = dx / norm
    uy = dy / norm

    loss = 0.0
    count = 0.0

    # 横向
    cos_x = (
        ux[:, :, :, 1:] * ux[:, :, :, :-1] +
        uy[:, :, :, 1:] * uy[:, :, :, :-1]
    )
    mask_x = special_mask[:, :, :, :-1]

    loss += (cos_x**2 * mask_x).sum()
    count += mask_x.sum()

    # 纵向
    cos_y = (
        ux[:, :, 1:, :] * ux[:, :, :-1, :] +
        uy[:, :, 1:, :] * uy[:, :, :-1, :]
    )
    mask_y = special_mask[:, :, :-1, :]

    loss += (cos_y**2 * mask_y).sum()
    count += mask_y.sum()

    return loss / (count + eps)


# =========================================================
# Direction Alignment Energy Loss
# =========================================================
def loss_align_energy(
    T_pred,          # (B,1,H,W)
    v_gt,            # (B,2,H,W)
    los_mask,        # (B,1,H,W)
    wall_mask,       # (B,1,H,W)
    special_mask,    # (B,1,H,W)
    alpha=1.0,
):
    """
    L = - v · ∇T + alpha * ||∇T||^2

    仅在以下区域计算：
        - special_mask == 1
        - 距离 LOS >= 2 像素
        - 距离墙 >= 2 像素

    梯度使用 mask-aware 差分（不跨界）
    """

    eps = 1e-8

    # =====================================================
    # 1️⃣ 构造有效区域 valid_mask
    # =====================================================

    # ---- 距离 LOS >= 2 ----
    los_d1 = F.max_pool2d(los_mask, 3, 1, 1)
    los_d2 = F.max_pool2d(los_d1, 3, 1, 1)
    safe_from_los = 1.0 - los_d2

    # ---- 距离墙 >= 2 ----
    wall_d1 = F.max_pool2d(wall_mask, 3, 1, 1)
    wall_d2 = F.max_pool2d(wall_d1, 3, 1, 1)
    safe_from_wall = 1.0 - wall_d2

    # ---- 最终有效区域 ----
    valid_mask = special_mask * safe_from_los * safe_from_wall
    valid = (valid_mask > 0)

    # =====================================================
    # 2️⃣ mask-aware 差分
    # =====================================================

    dx = torch.zeros_like(T_pred)
    dy = torch.zeros_like(T_pred)

    # ---------- x 方向 ----------
    left_valid  = torch.zeros_like(valid)
    right_valid = torch.zeros_like(valid)

    left_valid[:, :, :, 1:]  = valid[:, :, :, :-1]
    right_valid[:, :, :, :-1] = valid[:, :, :, 1:]

    both_x = left_valid & right_valid

    # 中心差分
    center_mask_x = both_x[:, :, :, 1:-1]
    dx[:, :, :, 1:-1][center_mask_x] = (
        T_pred[:, :, :, 2:][center_mask_x] -
        T_pred[:, :, :, :-2][center_mask_x]
    ) / 2.0

    # 单边差分
    only_left = left_valid & (~right_valid)
    dx[only_left] = (
        T_pred[only_left] -
        torch.roll(T_pred, 1, dims=3)[only_left]
    )

    only_right = right_valid & (~left_valid)
    dx[only_right] = (
        torch.roll(T_pred, -1, dims=3)[only_right] -
        T_pred[only_right]
    )

    # ---------- y 方向 ----------
    up_valid   = torch.zeros_like(valid)
    down_valid = torch.zeros_like(valid)

    up_valid[:, :, 1:, :]   = valid[:, :, :-1, :]
    down_valid[:, :, :-1, :] = valid[:, :, 1:, :]

    both_y = up_valid & down_valid

    center_mask_y = both_y[:, :, 1:-1, :]
    dy[:, :, 1:-1, :][center_mask_y] = (
        T_pred[:, :, 2:, :][center_mask_y] -
        T_pred[:, :, :-2, :][center_mask_y]
    ) / 2.0

    only_up = up_valid & (~down_valid)
    dy[only_up] = (
        T_pred[only_up] -
        torch.roll(T_pred, 1, dims=2)[only_up]
    )

    only_down = down_valid & (~up_valid)
    dy[only_down] = (
        torch.roll(T_pred, -1, dims=2)[only_down] -
        T_pred[only_down]
    )

    # =====================================================
    # 3️⃣ 能量项
    # =====================================================

    vx = v_gt[:, 0:1]
    vy = v_gt[:, 1:2]

    dot = vx * dx + vy * dy
    grad_sq = dx**2 + dy**2

    energy = -dot + alpha * grad_sq

    # =====================================================
    # 4️⃣ 只在 valid_mask 内平均
    # =====================================================

    loss = (energy * valid_mask).sum() / (valid_mask.sum() + eps)

    return loss


# =========================================================
# 从 GT 时间场生成方向场 v_gt
# =========================================================
def compute_v_gt(T_gt, wall_mask):
    """
    从 GT 时间场生成梯度场（不单位化）
    差分规则与预测端一致（不跨墙）
    """

    dx = torch.zeros_like(T_gt)
    dy = torch.zeros_like(T_gt)

    valid = (wall_mask == 0)

    # ---------- x ----------
    left_valid  = torch.zeros_like(valid)
    right_valid = torch.zeros_like(valid)

    left_valid[:, :, :, 1:]  = valid[:, :, :, :-1]
    right_valid[:, :, :, :-1] = valid[:, :, :, 1:]

    both_x = left_valid & right_valid

    center_mask_x = both_x[:, :, :, 1:-1]
    dx[:, :, :, 1:-1][center_mask_x] = (
        T_gt[:, :, :, 2:][center_mask_x] -
        T_gt[:, :, :, :-2][center_mask_x]
    ) / 2.0

    only_left = left_valid & (~right_valid)
    dx[only_left] = (
        T_gt[only_left] -
        torch.roll(T_gt, 1, dims=3)[only_left]
    )

    only_right = right_valid & (~left_valid)
    dx[only_right] = (
        torch.roll(T_gt, -1, dims=3)[only_right] -
        T_gt[only_right]
    )

    # ---------- y ----------
    up_valid   = torch.zeros_like(valid)
    down_valid = torch.zeros_like(valid)

    up_valid[:, :, 1:, :]   = valid[:, :, :-1, :]
    down_valid[:, :, :-1, :] = valid[:, :, 1:, :]

    both_y = up_valid & down_valid

    center_mask_y = both_y[:, :, 1:-1, :]
    dy[:, :, 1:-1, :][center_mask_y] = (
        T_gt[:, :, 2:, :][center_mask_y] -
        T_gt[:, :, :-2, :][center_mask_y]
    ) / 2.0

    only_up = up_valid & (~down_valid)
    dy[only_up] = (
        T_gt[only_up] -
        torch.roll(T_gt, 1, dims=2)[only_up]
    )

    only_down = down_valid & (~up_valid)
    dy[only_down] = (
        torch.roll(T_gt, -1, dims=2)[only_down] -
        T_gt[only_down]
    )

    v_gt = torch.cat([dx, dy], dim=1)

    return v_gt

# =========================================================
# 脚手架中心差分
# =========================================================

def eikonal_loss_center(T, special_mask, MAX_DIST):
    """
    T: (B,1,H,W)
    special_mask: (B,1,H,W)
    """

    # ---- 中心差分（去掉一圈边界） ----
    Tx = (T[:, :, :, 2:] - T[:, :, :, :-2]) * 0.5
    Ty = (T[:, :, 2:, :] - T[:, :, :-2, :]) * 0.5

    # 对齐尺寸
    Tx = Tx[:, :, 1:-1, :]
    Ty = Ty[:, :, :, 1:-1]

    # 梯度模长
    grad_norm = torch.sqrt(Tx**2 + Ty**2 + 1e-8)

    # ---- 正确的目标梯度尺度 ----
    c = 1.0 / MAX_DIST   # 因为 T = d / MAX_DIST

    # Hamilton L1 残差
    residual = torch.abs(grad_norm - c)

    # ---- 只在 normal 区域施加 ----
    normal_mask = 1.0 - special_mask
    normal_mask = normal_mask[:, :, 1:-1, 1:-1]

    loss = (residual * normal_mask).mean()

    return loss

# =========================================================
# val
# =========================================================
def validate(model, val_loader, device, yy, xx):

    model.eval()

    mse_total = 0.0
    mse_special_total = 0.0
    mse_normal_total = 0.0

    count = 0

    with torch.no_grad():

        for x, dist_real, ty, tx_pos, special_mask in val_loader:

            x = x.to(device)
            dist_real = dist_real.to(device)
            special_mask = special_mask.to(device)

            ty = ty.to(device).view(-1,1,1)
            tx_pos = tx_pos.to(device).view(-1,1,1)

            dist_euclid = torch.sqrt(
                (yy - ty)**2 + (xx - tx_pos)**2
            ) / MAX_DIST

            # ===== 双分支（和 train 完全一致）=====
            delta_low, delta_high = model(x)

            T_low = dist_euclid.unsqueeze(1) + delta_low
            T_pred = T_low + special_mask * delta_high

            diff = (T_pred - dist_real) ** 2

            # ===== 全局 MSE =====
            mse_total += diff.mean().item()

            # ===== special MSE =====
            mse_special = (
                (diff * special_mask).sum()
                / (special_mask.sum() + 1e-8)
            )
            mse_special_total += mse_special.item()

            # ===== normal MSE =====
            normal_mask = 1.0 - special_mask
            mse_normal = (
                (diff * normal_mask).sum()
                / (normal_mask.sum() + 1e-8)
            )
            mse_normal_total += mse_normal.item()

            count += 1

    return (
        mse_total * (MAX_DIST ** 2) / count,
        mse_normal_total * (MAX_DIST ** 2) / count,
        mse_special_total * (MAX_DIST ** 2) / count,
    )



# =========================================================
# 训练
# =========================================================
def train(model, train_loader, val_loader, device, epochs, vis_sample):

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 55, 65],
        gamma=0.1
    )

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    yy = yy.unsqueeze(0)
    xx = xx.unsqueeze(0)
    print("special区域的权重50")
    for epoch in range(1, epochs+1):

        target_lambda = 10
        lambda_eik = target_lambda

        model.train()
        total = 0.0

        low_energy_sum = 0.0
        high_energy_sum = 0.0
        norm_count = 0

        for x, dist_real, ty, tx_pos, special_mask in tqdm(
                train_loader,
                total=len(train_loader),
                desc="Training",
                ncols=100
        ):

            with torch.cuda.amp.autocast():

                x = x.to(device)
                dist_real = dist_real.to(device)
                special_mask = special_mask.to(device)

                ty = ty.to(device).view(-1,1,1)
                tx_pos = tx_pos.to(device).view(-1,1,1)

                dist_euclid = torch.sqrt(
                    (yy - ty)**2 + (xx - tx_pos)**2
                ) / MAX_DIST

                # ================= 双分支 =================
                delta_low, delta_high = model(x)

                # ===== 分支能量统计（无梯度）=====
                with torch.no_grad():
                    low_energy_sum += (delta_low**2).mean().item()
                    high_energy_sum += (delta_high**2).mean().item()
                    norm_count += 1

                T_low = dist_euclid.unsqueeze(1) + delta_low
                T_pred = T_low + special_mask * delta_high

                # ================= MSE =================
                diff = (T_pred - dist_real) ** 2
                normal_mask = 1.0 - special_mask

                loss_normal = (
                    (diff * normal_mask).sum()
                    / (normal_mask.sum() + 1e-8)
                )

                special_weight = 50

                loss_special = (
                    (diff * special_mask).sum()
                    / (special_mask.sum() + 1e-8)
                )

                # ================= 刷脚手架 =================
                T_pred = special_mask * dist_real + \
                         (1 - special_mask) * T_pred

                # ================= eik =================
                T_normal_only = T_low

                T_normal_only_scaf = (
                    special_mask * dist_real +
                    (1 - special_mask) * T_normal_only
                )

                loss_eik = eikonal_loss_center(
                    T_normal_only_scaf,
                    special_mask,
                    MAX_DIST
                )

                loss_num = (
                    loss_normal
                    + special_weight * loss_special
                    + lambda_eik * loss_eik
                )

            optimizer.zero_grad()
            scaler.scale(loss_num).backward()
            scaler.step(optimizer)
            scaler.update()

            total += loss_num.item()

        avg_loss = total / len(train_loader)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {avg_loss:.6f} | "
            f"low_energy {low_energy_sum/norm_count:.6f} | "
            f"high_energy {high_energy_sum/norm_count:.6f}"
        )

        visualize(
            model, vis_sample, device,
            yy, xx, epoch,
            save_dir="/root/RM/timevis",
            mask_dir="/root/RM/maskvis"
        )

        val_mse, val_normal_mse, val_special_mse = validate(
            model, val_loader, device, yy, xx
        )

        print(
            f"Val MSE {val_mse:.6f} | "
            f"Normal {val_normal_mse:.6f} | "
            f"Special {val_special_mse:.6f}"
        )

        scheduler.step()









# =========================================================
# 可视化（恢复物理单位）
# =========================================================
def visualize(model, vis_sample, device, yy, xx, epoch,
              save_dir="/root/RM/timevis",
              mask_dir="/root/RM/maskvis"):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    model.eval()

    x, dist_real, ty, tx_pos, special_mask = vis_sample

    # ===== 加 batch 维 =====
    x = x.unsqueeze(0).to(device)
    dist_real = dist_real.unsqueeze(0).to(device)
    special_mask = special_mask.unsqueeze(0).to(device)

    ty = ty.unsqueeze(0).to(device).view(-1,1,1)
    tx_pos = tx_pos.unsqueeze(0).to(device).view(-1,1,1)

    with torch.no_grad():
        delta_s, delta_h = model(x)

        # 如果模型输出是 (B,H,W)，补通道
        if delta_s.dim() == 3:
            delta_s = delta_s.unsqueeze(1)
        if delta_h.dim() == 3:
            delta_h = delta_h.unsqueeze(1)

        pred_delta = delta_s + special_mask.unsqueeze(1) * delta_h

    dist_euclid = torch.sqrt(
        (yy - ty)**2 + (xx - tx_pos)**2
    ) / MAX_DIST

    # 如果 dist_euclid 是 (1,H,W)，补通道
    if dist_euclid.dim() == 3:
        dist_euclid = dist_euclid.unsqueeze(1)

    pred_real = (dist_euclid + pred_delta) * MAX_DIST

    # ===== 统一 squeeze 成 (H,W) =====
    pred_real = pred_real.squeeze().cpu()
    dist_real = (dist_real * MAX_DIST).squeeze().cpu()
    special_mask_img = special_mask.squeeze().cpu()

    # ===== 画 GT / Pred =====
    fig, axs = plt.subplots(1,2, figsize=(10,5))

    axs[0].imshow(dist_real, cmap="viridis")
    axs[0].set_title("GT")

    axs[1].imshow(pred_real, cmap="viridis")
    axs[1].set_title("Pred")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch:04d}.svg", dpi=200)
    plt.close()

    # ===== 画 mask =====
    plt.figure(figsize=(5,5))
    plt.imshow(special_mask_img, cmap="gray")
    plt.title("Special Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{mask_dir}/epoch_{epoch:04d}.svg", dpi=200)
    plt.close()

# =========================================================
# 主函数
# =========================================================
def main():

    index_file = f"{ROOT}/minors/minor2_index.npy"
    dataset = RMDataset(
        index_file,
        packed_root="/dev/shm/packed_dataset"
    )
    index_array = dataset.index

    # ===============================
    # 1️⃣ 固定可视化样本
    # ===============================
    VIS_IDX = 525
    vis_sample = dataset[VIS_IDX]
    vis_scene = index_array[VIS_IDX][0]

    print("Fixed VIS scene:", vis_scene)

    # ===============================
    # 2️⃣ Scene-level 划分
    # ===============================
    all_scenes = list(set(index_array[:, 0]))
    remaining_scenes = [s for s in all_scenes if s != vis_scene]

    np.random.shuffle(remaining_scenes)

    n_total_scene = len(all_scenes)
    n_val_scene = int(0.2 * n_total_scene)
    n_extra_scene = n_val_scene - 1

    val_scene = set([vis_scene] + remaining_scenes[:n_extra_scene])
    train_scene = set(remaining_scenes[n_extra_scene:])

    train_indices = [
        i for i, (s, _) in enumerate(index_array)
        if s in train_scene
    ]

    val_indices = [
        i for i, (s, _) in enumerate(index_array)
        if s in val_scene
    ]

    print("Train scenes:", len(train_scene))
    print("Val scenes:", len(val_scene))
    print("Train samples:", len(train_indices))
    print("Val samples:", len(val_indices))

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set   = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=8
    )

    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        num_workers=8
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet().to(device)

    # ===============================
    # 3️⃣ 训练（传入固定 vis_sample）
    # ===============================
    train(model, train_loader, val_loader, device, epochs=100, vis_sample=vis_sample)



if __name__ == "__main__":
    print("双分支，一个平滑，一个切换线，可以看到脚手架。")
    main()
