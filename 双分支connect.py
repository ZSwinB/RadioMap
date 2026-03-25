import os
import torch
import numpy as np
from tqdm import tqdm

# ===============================
# 路径配置
# ===============================
ROOT = "/root/RM/data"
INDEX_FILE = f"{ROOT}/minors/index.npy"
PACKED_ROOT = "/dev/shm/packed_dataset"

MODEL_PATH = "./timemodel_best.pth"
SAVE_ROOT = "/root/RM/pred_dist"

H = 256
W = 256
MAX_DIST = 1000.0

os.makedirs(SAVE_ROOT, exist_ok=True)

# ===============================
# UNet 定义（与你训练一致）
# ===============================
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1,)):
        super().__init__()
        blocks = []
        for d in dilations:
            padding = d
            blocks.append(nn.Conv2d(in_ch, out_ch, 3, padding=padding, dilation=d, bias=False))
            blocks.append(nn.BatchNorm2d(out_ch))
            blocks.append(nn.SiLU())
            blocks.append(nn.Conv2d(out_ch, out_ch, 3, padding=padding, dilation=d, bias=False))
            blocks.append(nn.BatchNorm2d(out_ch))
            blocks.append(nn.SiLU())
            in_ch = out_ch
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()

        self.down1 = DoubleConv(in_ch, base_ch, (1,))
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base_ch, base_ch*2, (1,))
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base_ch*2, base_ch*4, (1,2))
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(base_ch*4, base_ch*8, (1,2))
        self.pool4 = nn.MaxPool2d(2)

        self.mid = DoubleConv(base_ch*8, base_ch*16, (1,2,4))

        # low branch
        self.up4_l = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, 2)
        self.conv4_l = DoubleConv(base_ch*16, base_ch*8, (1,2))

        self.up3_l = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.conv3_l = DoubleConv(base_ch*8, base_ch*4, (1,2))

        self.up2_l = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.conv2_l = DoubleConv(base_ch*4, base_ch*2, (1,))

        self.up1_l = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.conv1_l = DoubleConv(base_ch*2, base_ch, (1,))

        self.out_l = nn.Sequential(
            nn.Conv2d(base_ch, 1, 1),
            nn.Softplus()
        )

        # high branch
        self.up4_h = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, 2)
        self.conv4_h = DoubleConv(base_ch*16, base_ch*8, (1,))

        self.up3_h = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.conv3_h = DoubleConv(base_ch*8, base_ch*4, (1,))

        self.up2_h = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.conv2_h = DoubleConv(base_ch*4, base_ch*2, (1,))

        self.up1_h = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.conv1_h = DoubleConv(base_ch*2, base_ch, (1,))

        self.out_h = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        m = self.mid(self.pool4(d4))

        # low
        u4l = self.up4_l(m)
        u4l = self.conv4_l(torch.cat([u4l, d4], dim=1))

        u3l = self.up3_l(u4l)
        u3l = self.conv3_l(torch.cat([u3l, d3], dim=1))

        u2l = self.up2_l(u3l)
        u2l = self.conv2_l(torch.cat([u2l, d2], dim=1))

        u1l = self.up1_l(u2l)
        u1l = self.conv1_l(torch.cat([u1l, d1], dim=1))

        delta_low = self.out_l(u1l)

        # high
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

# ===============================
# 加载模型
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ===============================
# 网格
# ===============================
yy, xx = torch.meshgrid(
    torch.arange(H, device=device),
    torch.arange(W, device=device),
    indexing="ij"
)
yy = yy.unsqueeze(0)
xx = xx.unsqueeze(0)

# ===============================
# 读取 index
# ===============================
index_array = np.load(INDEX_FILE)

# ===============================
# forward
# ===============================
with torch.no_grad():

    for scene_id, tx_id in tqdm(index_array):

        path = os.path.join(
            PACKED_ROOT,
            f"{scene_id}_{tx_id}.pt"
        )

        data = torch.load(path, map_location=device)

        x = data["x"].float().unsqueeze(0).to(device)
        ty = data["ty"].float().view(1,1,1).to(device)
        tx_pos = data["tx_pos"].float().view(1,1,1).to(device)
        special_mask = data["special_mask"].float().unsqueeze(0).to(device)

        delta_low, delta_high = model(x)

        # 欧几里得基底（归一化）
        dist_euclid = torch.sqrt(
            (yy - ty)**2 + (xx - tx_pos)**2
        ) / MAX_DIST

        # 归一化时间场
        T_low = dist_euclid.unsqueeze(1) + delta_low
        T_pred = T_low + special_mask.unsqueeze(1) * delta_high

        # ===============================
        # 转为物理尺度（关键）
        # ===============================
        T_real = T_pred * MAX_DIST

        dist_np = T_real.squeeze().cpu().numpy().astype(np.float32)

        save_path = os.path.join(
            SAVE_ROOT,
            f"{scene_id}_{tx_id}.npy"
        )

        np.save(save_path, dist_np)

print("All physical-scale dist generated.")