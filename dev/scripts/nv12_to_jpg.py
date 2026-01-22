#!/usr/bin/env python3
"""
NV12 转 JPG 工具脚本
支持单个文件转换或批量转换目录下的所有 NV12 文件
"""

import sys
import re
from pathlib import Path
import cv2
import numpy as np


def parse_yuv_filename(filename):
    """从文件名解析宽高: name_WIDTHxHEIGHT.nv12"""
    match = re.search(r'_(\d+)x(\d+)\.nv(?:12|21)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def nv12_to_bgr(nv12_data, width, height):
    """将 NV12 数据转换为 BGR 图像"""
    # NV12 格式: Y 平面 + UV 交错平面
    y_size = width * height

    # 提取 Y 和 UV 平面
    y_plane = np.frombuffer(nv12_data[:y_size], dtype=np.uint8).reshape((height, width))
    uv_plane = np.frombuffer(nv12_data[y_size:], dtype=np.uint8).reshape((height // 2, width // 2, 2))

    # 上采样 UV 平面
    uv_upsampled = cv2.resize(uv_plane, (width, height), interpolation=cv2.INTER_LINEAR)

    # 合并 YUV 通道
    yuv = np.zeros((height, width, 3), dtype=np.uint8)
    yuv[:, :, 0] = y_plane
    yuv[:, :, 1] = uv_upsampled[:, :, 0]  # U
    yuv[:, :, 2] = uv_upsampled[:, :, 1]  # V

    # YUV 转 BGR
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return bgr


def convert_nv12_to_jpg(input_path, output_path, width=None, height=None):
    """转换单个 NV12 文件为 JPG"""
    input_path = Path(input_path)

    # 尝试从文件名解析宽高
    if width is None or height is None:
        dims = parse_yuv_filename(input_path.name)
        if dims:
            width, height = dims
        else:
            print(f"错误: 无法确定图像尺寸，请通过参数指定或使用格式: name_WIDTHxHEIGHT.nv12")
            return False

    # 验证文件大小
    expected_size = width * height * 3 // 2
    file_size = input_path.stat().st_size
    if file_size != expected_size:
        print(f"警告: 文件大小不匹配")
        print(f"  期望: {expected_size} 字节 ({width}x{height})")
        print(f"  实际: {file_size} 字节")
        return False

    # 读取 NV12 数据
    with open(input_path, 'rb') as f:
        nv12_data = f.read()

    # 转换为 BGR
    bgr = nv12_to_bgr(nv12_data, width, height)

    # 保存为 JPG
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"✓ {input_path.name} -> {output_path.name} ({width}x{height})")
    return True


def batch_convert(input_dir, output_dir, width=None, height=None):
    """批量转换目录下的所有 NV12 文件"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nv12_files = list(input_dir.glob("*.nv12")) + list(input_dir.glob("*.NV12"))

    if not nv12_files:
        print(f"未找到 NV12 文件: {input_dir}")
        return

    print(f"找到 {len(nv12_files)} 个 NV12 文件\n")

    success_count = 0
    for nv12_file in nv12_files:
        output_file = output_dir / f"{nv12_file.stem}.jpg"
        if convert_nv12_to_jpg(nv12_file, output_file, width, height):
            success_count += 1

    print(f"\n完成: {success_count}/{len(nv12_files)} 个文件转换成功")


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  单个文件: python nv12_to_jpg.py input.nv12 [output.jpg] [width] [height]")
        print("  批量转换: python nv12_to_jpg.py input_dir/ output_dir/ [width] [height]")
        print("\n示例:")
        print("  python nv12_to_jpg.py frame_1920x1080.nv12")
        print("  python nv12_to_jpg.py frame.nv12 output.jpg 1920 1080")
        print("  python nv12_to_jpg.py ./nv12_files/ ./jpg_files/ 3840 2700")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # 解析宽高参数
    width = int(sys.argv[-2]) if len(sys.argv) >= 4 and sys.argv[-2].isdigit() else None
    height = int(sys.argv[-1]) if len(sys.argv) >= 3 and sys.argv[-1].isdigit() else None

    if input_path.is_file():
        # 单个文件转换
        if len(sys.argv) >= 3 and not sys.argv[2].isdigit():
            output_path = sys.argv[2]
        else:
            output_path = input_path.with_suffix('.jpg')

        convert_nv12_to_jpg(input_path, output_path, width, height)

    elif input_path.is_dir():
        # 批量转换
        if len(sys.argv) >= 3 and not sys.argv[2].isdigit():
            output_dir = sys.argv[2]
        else:
            output_dir = input_path / "jpg_output"

        batch_convert(input_path, output_dir, width, height)

    else:
        print(f"错误: 路径不存在: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
