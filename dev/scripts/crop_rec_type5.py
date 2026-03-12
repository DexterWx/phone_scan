#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import cv2
from pathlib import Path


def crop_images_from_results(input_folder: str, output_folder: str):
    """
    从结果文件夹中提取 rec_type=5 的区域并切图

    Args:
        input_folder: 包含子文件夹的输入目录（每个子文件夹有 result.json 和 output.jpg）
        output_folder: 切图输出目录
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有子文件夹
    input_path = Path(input_folder)
    subfolders = [f for f in input_path.iterdir() if f.is_dir()]

    if not subfolders:
        print(f"⚠️  No subfolders found in {input_folder}")
        return

    print(f"📁 Found {len(subfolders)} subfolders")

    total_crops = 0
    processed_folders = 0

    # 遍历每个子文件夹
    for subfolder in sorted(subfolders):
        result_json_path = subfolder / "result.json"
        image_path = subfolder / "output.jpg"

        # 检查必要文件是否存在
        if not result_json_path.exists():
            print(f"⚠️  Skipping {subfolder.name}: result.json not found")
            continue

        if not image_path.exists():
            print(f"⚠️  Skipping {subfolder.name}: output.jpg not found")
            continue

        try:
            # 读取 JSON 结果
            with open(result_json_path, 'r', encoding='utf-8') as f:
                result = json.load(f)

            # 检查是否有识别结果
            if result.get('code') != 0:
                print(f"⚠️  Skipping {subfolder.name}: recognition failed")
                continue

            rec_results = result.get('rec_results', [])
            if not rec_results:
                continue

            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Failed to read image: {image_path}")
                continue

            # 查找 rec_type=5 的项
            crops_in_folder = 0
            for idx_rec_item, rec_item in enumerate(rec_results):
                if rec_item.get('rec_type') != 5:
                    continue

                rec_options = rec_item.get('rec_options', [])

                # 遍历所有 coordinate 并切图
                for idx, option in enumerate(rec_options):
                    coordinate = option.get('coordinate')
                    if not coordinate:
                        continue

                    x = coordinate.get('x', 0)
                    y = coordinate.get('y', 0)
                    w = coordinate.get('w', 0)
                    h = coordinate.get('h', 0)

                    # 确保坐标在图片范围内
                    img_h, img_w = image.shape[:2]
                    x = max(0, min(x, img_w))
                    y = max(0, min(y, img_h))
                    w = max(0, min(w, img_w - x))
                    h = max(0, min(h, img_h - y))

                    if w <= 0 or h <= 0:
                        print(f"  ⚠️  Invalid coordinate: x={x}, y={y}, w={w}, h={h}")
                        continue

                    # 切图
                    cropped = image[y:y+h, x:x+w]

                    # 生成输出文件名
                    output_filename = f"{subfolder.name}_crop_{idx_rec_item:02d}_{idx:02d}.jpg"
                    output_path = os.path.join(output_folder, output_filename)

                    # 保存切图
                    cv2.imwrite(output_path, cropped)
                    crops_in_folder += 1
                    total_crops += 1

            if crops_in_folder > 0:
                print(f"✅ {subfolder.name}: extracted {crops_in_folder} crops")
                processed_folders += 1

        except Exception as e:
            print(f"❌ Error processing {subfolder.name}: {e}")
            continue

    print(f"\n🎉 Done! Processed {processed_folders} folders, extracted {total_crops} crops")
    print(f"📂 Output folder: {output_folder}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python crop_rec_type5.py <input_folder> <output_folder>")
        print("  input_folder: folder containing subfolders with result.json and output.jpg")
        print("  output_folder: folder to save cropped images")
        print("\nExample:")
        print("  python crop_rec_type5.py output_results cropped_images")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' does not exist")
        sys.exit(1)

    try:
        crop_images_from_results(input_folder, output_folder)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
