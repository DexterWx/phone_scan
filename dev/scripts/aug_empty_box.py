#!/usr/bin/env python3
"""对1_a4开头的图片进行随机高斯模糊增强"""

import os
import cv2
import numpy as np
import argparse
import random
from pathlib import Path


# 模糊参数上下限
KSIZE_MIN, KSIZE_MAX = 5, 9
SIGMA_MIN, SIGMA_MAX = 3, 7


def make_blur_name(filename, suffix="_blur"):
    """生成模糊后的文件名，去掉重复的.jpg，加上blur后缀"""
    # 1_a4_xxx.jpg_954_593.jpg -> 1_a4_xxx_954_593_blur.jpg
    name = filename.replace(".jpg_", "_").replace(".png_", "_")
    stem, ext = os.path.splitext(name)
    return f"{stem}{suffix}{ext}"


def gaussian_blur(img, ksize, sigma):
    """高斯模糊"""
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return blurred


def random_gaussian_blur(img):
    """随机高斯模糊，在上下限范围内随机"""
    # 随机选择模糊核大小 (必须是奇数)
    ksize_options = list(range(KSIZE_MIN, KSIZE_MAX + 1, 2))  # 奇数
    ksize = random.choice(ksize_options)
    sigma = random.uniform(SIGMA_MIN, SIGMA_MAX)

    blurred = gaussian_blur(img, ksize, sigma)
    return blurred, ksize, sigma


def process_images(input_dir, output_dir, preview=False):
    """处理1_a4开头的图片"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有1_a4开头的文件
    files = sorted([f for f in input_path.iterdir() if f.name.startswith("1_a4")])

    if not files:
        print("未找到1_a4开头的文件")
        return

    print(f"找到 {len(files)} 个1_a4开头的文件")

    # preview模式：用第一张图片展示最强和最弱模糊效果
    if preview:
        f = files[0]
        img = cv2.imread(str(f))
        if img is None:
            print(f"无法读取: {f.name}")
            return

        print("Preview模式：展示模糊最强和最弱效果")

        # 最弱模糊
        blurred_min = gaussian_blur(img, KSIZE_MIN, SIGMA_MIN)
        name_min = make_blur_name(f.name, "_blur_min")
        cv2.imwrite(str(output_path / name_min), blurred_min)
        print(f"最弱模糊: {name_min} -> ksize={KSIZE_MIN}, sigma={SIGMA_MIN}")

        # 最强模糊
        blurred_max = gaussian_blur(img, KSIZE_MAX, SIGMA_MAX)
        name_max = make_blur_name(f.name, "_blur_max")
        cv2.imwrite(str(output_path / name_max), blurred_max)
        print(f"最强模糊: {name_max} -> ksize={KSIZE_MAX}, sigma={SIGMA_MAX}")
        return

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"无法读取: {f.name}")
            continue

        blurred, ksize, sigma = random_gaussian_blur(img)

        # 保存
        out_name = make_blur_name(f.name)
        cv2.imwrite(str(output_path / out_name), blurred)
        print(f"处理: {out_name} -> ksize={ksize}, sigma={sigma:.2f}")

    print(f"\n输出目录: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对1_a4图片进行高斯模糊增强")
    parser.add_argument("--input", default="/Users/xu.wang/Downloads/crop_images",
                        help="输入目录")
    parser.add_argument("--output", default="/Users/xu.wang/Downloads/crop_images_blur",
                        help="输出目录")
    parser.add_argument("--preview", action="store_true",
                        help="预览模式，只处理一张图片")

    args = parser.parse_args()
    process_images(args.input, args.output, args.preview)
