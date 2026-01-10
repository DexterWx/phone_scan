#!/usr/bin/env python3
"""
清洗标注数据脚本
将多个子目录中的图片整合到一个输出目录中，并按规则重命名：
- 前缀：子目录名称
- 尺寸类型：宽>高为a3，否则为a4
- 编号：从1开始的数字
最终格式：{size_type}_{name}_{number}.jpg
"""

import os
import shutil
from pathlib import Path
from PIL import Image, ImageOps


def get_image_size(image_path: str) -> tuple[int, int]:
    """
    获取图片尺寸 (width, height)
    ⚠️ 已修复：正确处理 JPG 的 EXIF Orientation
    """
    with Image.open(image_path) as img:
        # 关键修复点：应用 EXIF 旋转
        img = ImageOps.exif_transpose(img)
        return img.size


def get_size_type(width: int, height: int) -> str:
    """根据宽高判断 a3 / a4（基于视觉方向）"""
    return "a3" if width > height else "a4"


def clean_mark_data(input_dir: str, output_dir: str):
    """
    清洗标注数据

    Args:
        input_dir: 输入目录，包含多个子目录
        output_dir: 输出目录，所有图片将放在这里
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    total_count = 0

    # 遍历所有子目录
    for subdir in sorted(input_path.iterdir()):
        if not subdir.is_dir():
            continue

        # 跳过隐藏目录
        if subdir.name.startswith('.'):
            continue

        prefix = subdir.name
        counter = 1

        print(f"处理目录: {prefix}")

        # 遍历子目录中的所有图片
        for img_file in sorted(subdir.iterdir()):
            if not img_file.is_file():
                continue

            # 检查是否是图片文件
            if img_file.suffix.lower() not in image_extensions:
                continue

            try:
                # 获取图片尺寸（已修复 EXIF 方向问题）
                width, height = get_image_size(str(img_file))
                size_type = get_size_type(width, height)

                # 生成新文件名
                new_name = f"{size_type}_{prefix}_{counter}.jpg"
                new_path = output_path / new_name

                # 复制文件（不改变你原来的行为）
                shutil.copy2(img_file, new_path)

                print(f"  {img_file.name} -> {new_name} ({width}x{height})")

                counter += 1
                total_count += 1

            except Exception as e:
                print(f"  错误处理 {img_file.name}: {e}")

    print(f"\n完成! 共处理 {total_count} 张图片")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="清洗标注数据")
    parser.add_argument(
        "-i", "--input",
        default="/Users/xu.wang/Downloads/mark_data",
        help="输入目录路径"
    )
    parser.add_argument(
        "-o", "--output",
        default="/Users/xu.wang/Downloads/mark_data_cleaned_v2",
        help="输出目录路径"
    )

    args = parser.parse_args()

    clean_mark_data(args.input, args.output)
