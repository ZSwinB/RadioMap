import cv2
import numpy as np
import os

base_dir = r"G:\RadioMapSeer\png\cars"

total_components = 0
total_pixels = 0

for i in range(701):  # 0 到 700
    image_path = os.path.join(base_dir, f"{i}.png")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"跳过无法读取的文件: {image_path}")
        continue

    # 只统计像素值为255的部分
    binary = (img == 255).astype(np.uint8)

    # 连通域分析（8连通）
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    num_components = num_labels - 1  # 去掉背景

    # 当前图像中255像素总数
    pixel_count = np.sum(binary)

    total_components += num_components
    total_pixels += pixel_count

if total_components == 0:
    print("没有检测到任何连通块")
else:
    avg_pixels_per_component = total_pixels / total_components
    print(f"总连通块数量: {total_components}")
    print(f"总255像素数量: {total_pixels}")
    print(f"平均每个连通块的像素数: {avg_pixels_per_component}")