import numpy as np
import os
from scipy.ndimage import label
from multiprocessing import Pool, cpu_count


# ==========================
# 路径
# ==========================
INPUT_ROOT = "/root/RM/data/distance_map_DPM"
OUTPUT_ROOT = "/root/RM/data/scaffoldfull"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ==========================
# 参数
# ==========================
epsilon = 0.08
component_threshold = 60


# ==========================
# 单个 scene
# ==========================
def process_scene(scene_id):

    print(f"Start Scene {scene_id}")

    for frame_id in range(104):   # 0~80

        input_path = os.path.join(INPUT_ROOT, f"{scene_id}_{frame_id}.npz")

        if not os.path.exists(input_path):
            continue

        data = np.load(input_path)
        dist_map = data["dist_map"]

        # ==========================
        # 统一取单层
        # ==========================
        if dist_map.ndim == 3:
            T = dist_map[:, :, 0].astype(np.float32)
        else:
            T = dist_map.astype(np.float32)

        H, W = T.shape

        # ==========================
        # 梯度
        # ==========================
        dx = np.zeros_like(T)
        dy = np.zeros_like(T)

        dx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / 2.0
        dy[1:-1, :] = (T[2:, :] - T[:-2, :]) / 2.0

        grad_norm = np.sqrt(dx * dx + dy * dy)

        # ==========================
        # 结构区域
        # ==========================
        structure_mask = (grad_norm > (1.0 + epsilon))

        # ==========================
        # 连通域过滤
        # ==========================
        labeled, num = label(structure_mask, structure=np.ones((3, 3)))

        clean_mask = np.zeros_like(structure_mask)

        for i in range(1, num + 1):
            comp = (labeled == i)
            if np.sum(comp) >= component_threshold:
                clean_mask |= comp

        final_mask = clean_mask.astype(np.uint8)

        # ==========================
        # 保存
        # ==========================
        save_path = os.path.join(OUTPUT_ROOT, f"{scene_id}_{frame_id}.npy")
        np.save(save_path, final_mask)

    print(f"Finish Scene {scene_id}")


# ==========================
# 主函数
# ==========================
if __name__ == "__main__":

    num_workers = max(cpu_count() - 1, 1)
    print("Using workers:", num_workers)

    with Pool(num_workers) as pool:
        pool.map(process_scene, range(0, 701))   # 0~700

    print("All scenes done.")