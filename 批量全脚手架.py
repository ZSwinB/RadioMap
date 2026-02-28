import numpy as np
import os
import csv
from scipy.ndimage import label
from multiprocessing import Pool, cpu_count

# ==========================
# 路径
# ==========================
T_root = r"G:\RMdata\distancemap"
wall_root = r"G:\RM\feature_store\numerical_data\geo"
save_root = r"G:\RMdata\scaffoldfull"
log_root = r"G:\RM\feature_views\time"

os.makedirs(save_root, exist_ok=True)
os.makedirs(log_root, exist_ok=True)

# ==========================
# 参数
# ==========================
epsilon = 0.08
component_threshold = 60

# ==========================
# 单个 scene 处理函数
# ==========================
def process_scene(scene_id):

    print(f"Start Scene {scene_id}")

    wall_path = os.path.join(wall_root, f"{scene_id}.npy")
    if not os.path.exists(wall_path):
        return []

    wall_mask = np.load(wall_path)
    free_pixels = int(np.sum(wall_mask == 0))

    scene_rows = []

    for frame_id in range(80):

        T_path = os.path.join(T_root, f"{scene_id}_{frame_id}.npy")
        if not os.path.exists(T_path):
            continue

        T = np.load(T_path)

        # 梯度
        dx = np.zeros_like(T)
        dy = np.zeros_like(T)

        dx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / 2.0
        dy[1:-1, :] = (T[2:, :] - T[:-2, :]) / 2.0

        grad_norm = np.sqrt(dx**2 + dy**2)
        structure_mask = (grad_norm > (1.0 + epsilon))

        raw_count = int(np.sum(structure_mask))

        # 删除小连通域
        structure_8 = np.ones((3,3), dtype=int)
        labeled_struct, num_struct = label(structure_mask, structure=structure_8)

        clean_mask = np.zeros_like(structure_mask)

        for i in range(1, num_struct + 1):
            component = (labeled_struct == i)
            if np.sum(component) >= component_threshold:
                clean_mask |= component

        final_mask = clean_mask
        final_count = int(np.sum(final_mask))

        # 保存
        save_path = os.path.join(save_root, f"{scene_id}_{frame_id}.npy")
        np.save(save_path, final_mask.astype(np.uint8))

        scene_rows.append([
            f"{scene_id}_{frame_id}",
            free_pixels,
            raw_count,
            final_count
        ])

    print(f"Finish Scene {scene_id}")
    return scene_rows


# ==========================
# 主函数
# ==========================
if __name__ == "__main__":

    num_workers = max(cpu_count() - 1, 1)
    print("Using workers:", num_workers)

    with Pool(num_workers) as pool:
        results = pool.map(process_scene, range(701))

    # 扁平化
    all_log_rows = [row for scene_rows in results for row in scene_rows]

    # 保存 log
    csv_all_path = os.path.join(log_root, "log_all_scenes.csv")
    txt_all_path = os.path.join(log_root, "log_all_scenes.txt")

    with open(csv_all_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_frame",
            "free_pixels",
            "raw_structure_count",
            "final_count_after_component_filter"
        ])
        writer.writerows(all_log_rows)

    with open(txt_all_path, "w") as f:
        f.write("scene_frame | free_pixels | raw_struct | final_after\n")
        f.write("-" * 60 + "\n")
        for row in all_log_rows:
            f.write(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}\n")

    print("All scenes done.")
