#!/usr/bin/env python3
"""图片缩放脚本"""

import argparse
import cv2
import sys
from pathlib import Path


def resize_image(input_path: str, output_path: str, scale: float) -> None:
    """
    缩放图片

    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        scale: 缩放比例 (0.5 = 缩小一半, 2.0 = 放大两倍)
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误: 无法读取图片 {input_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, resized)
    print(f"已保存: {output_path} ({w}x{h} -> {new_w}x{new_h})")


def main():
    parser = argparse.ArgumentParser(description="图片缩放脚本")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("output", help="输出图片路径")
    parser.add_argument("scale", type=float, help="缩放比例 (例如 0.5, 2.0)")

    args = parser.parse_args()
    resize_image(args.input, args.output, args.scale)


if __name__ == "__main__":
    main()
