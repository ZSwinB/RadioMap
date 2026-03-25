import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = "/root/RM/data"

GEO_ROOT = os.path.join(ROOT, "geo")
CAR_ROOT = os.path.join(ROOT, "cars")
ANT_ROOT = os.path.join(ROOT, "antenna")

LOS_ROOT = "/root/RM/los_mask"
DIST_ROOT = "/root/RM/data/distance_map_DPM"
SCAF_ROOT = "/root/RM/data/scaffoldfull"

INDEX_FILE = os.path.join(ROOT, "minors/indexfull.npy")

OUT_DIR = "/dev/shm/packed_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_DIST = 1000.0

from PIL import Image
import numpy as np

def check_png(path):
    img = Image.open(path)
    arr = np.array(img)
    print("unique values:", np.unique(arr))


def load_png(path):
    img = Image.open(path).convert("F")
    img = np.array(img, dtype=np.float32) / 255.0
    return img


def load_mask_png(path):
    img = Image.open(path)
    img = np.array(img, dtype=np.uint8)
    img = (img > 0).astype(np.uint8)
    return img

def main():

    index = np.load(INDEX_FILE)
    print("Total samples:", len(index))

    for scene_id, tx_id in tqdm(index):

        scene_id = str(scene_id)
        tx_id = str(tx_id)

        out_path = os.path.join(OUT_DIR, f"{scene_id}_{tx_id}.pt")

        if os.path.exists(out_path):
            continue

        try:

            # =========================
            # geo + cars
            # =========================
            geo_build = load_mask_png(f"{GEO_ROOT}/{scene_id}.png")
            geo_car   = load_mask_png(f"{CAR_ROOT}/{scene_id}.png")

            # geo = np.maximum(geo_build, geo_car)
            geo =geo_build
            # =========================
            # antenna
            # =========================
            # =========================
            # antenna（分路径读取）
            # =========================
            if int(tx_id) >= 80:
                tx = np.load(f"/root/RM/temp/{scene_id}_{tx_id}.npy").astype(np.uint8)
            else:
                tx = load_mask_png(f"{ANT_ROOT}/{scene_id}_{tx_id}.png")

            coords = np.nonzero(tx)

            if len(coords[0]) == 0:
                continue

            ty = float(coords[0][0])
            tx_pos = float(coords[1][0])

            # =========================
            # LOS mask
            # =========================
            los = np.load(
                f"{LOS_ROOT}/{scene_id}_{tx_id}.npy"
            ).astype(np.uint8)

            # =========================
            # scaffold (slice 0)
            # =========================
            special = np.load(
                f"{SCAF_ROOT}/{scene_id}_{tx_id}.npy"
            )

            if special.ndim == 3:
                special = special[:, :, 0]

            special = special.astype(np.uint8)

            # =========================
            # distancemap (npz + slice 0)
            # =========================
            dist_data = np.load(
                f"{DIST_ROOT}/{scene_id}_{tx_id}.npz"
            )

            dist_real = dist_data["dist_map"]

            if dist_real.ndim == 3:
                dist_real = dist_real[:, :, 0]

            dist_real = dist_real.astype(np.float32)

            dist_real /= MAX_DIST

            # =========================
            # stack input
            # =========================
            x = np.stack([geo, tx, los], axis=0)

            data = {
                "x": torch.from_numpy(x),
                "dist": torch.from_numpy(dist_real).unsqueeze(0),
                "ty": torch.tensor(ty),
                "tx_pos": torch.tensor(tx_pos),
                "special_mask": torch.from_numpy(special).unsqueeze(0),
            }

            tmp_path = out_path + ".tmp"

            torch.save(data, tmp_path)
            os.replace(tmp_path, out_path)

        except Exception as e:
            print(f"Skip {scene_id}_{tx_id} due to error:", e)
            continue

    print("Packing finished.")


if __name__ == "__main__":
    main()