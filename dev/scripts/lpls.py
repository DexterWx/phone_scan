#!/usr/bin/env python3
"""
图像清晰度检测工具 (Laplacian + Tenengrad)
用法: python lpls.py <image_path_or_folder>
返回: 双指标清晰度评分及综合排名（值越大越清晰）
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import time


def read_image_gray(image_path):
    """读取图像并转为灰度图，截取中心区域"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 截取中心区域 (1/4 到 3/4)
    h, w = image.shape[:2]
    x1, y1 = w // 4, h // 4
    x2, y2 = w * 3 // 4, h * 3 // 4
    center = image[y1:y2, x1:x2]

    if len(center.shape) == 3:
        return cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    return center


def compute_laplacian_variance(gray):
    """
    计算图像的Laplacian方差（模糊度指标）
    值越大越清晰
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return laplacian.var()


def compute_tenengrad(gray):
    """
    计算Tenengrad梯度（清晰度指标）
    基于Sobel梯度幅值的平方均值，值越大越清晰
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = gx**2 + gy**2
    return np.mean(gradient_magnitude)


def process_single_image(image_path):
    """处理单张图片，返回双指标结果"""
    try:
        st_time = time.time()
        gray = read_image_gray(image_path)
        laplacian = compute_laplacian_variance(gray)
        tenengrad = compute_tenengrad(gray)
        elapsed = time.time() - st_time
        return {
            'path': image_path,
            'laplacian': laplacian,
            'tenengrad': tenengrad,
            'time': elapsed,
            'success': True
        }
    except Exception as e:
        return {
            'path': image_path,
            'laplacian': 0,
            'tenengrad': 0,
            'time': 0,
            'success': False,
            'error': str(e)
        }


def process_folder(folder_path):
    """处理文件夹中的所有图片"""
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.heic', '.heif'}

    # 收集所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"错误: 文件夹 {folder_path} 中没有找到图片文件")
        return []

    print(f"找到 {len(image_files)} 张图片")
    print("=" * 90)

    results = []
    for img_path in sorted(image_files):
        result = process_single_image(img_path)
        results.append(result)

    # 计算综合排名（排名相加，越小越好）
    successful = [r for r in results if r['success']]
    if successful:
        # 按 Laplacian 排序，获取排名
        sorted_by_lap = sorted(enumerate(successful), key=lambda x: x[1]['laplacian'], reverse=True)
        lap_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(sorted_by_lap)}

        # 按 Tenengrad 排序，获取排名
        sorted_by_ten = sorted(enumerate(successful), key=lambda x: x[1]['tenengrad'], reverse=True)
        ten_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(sorted_by_ten)}

        # 计算综合排名分数
        for idx, r in enumerate(successful):
            r['combined_score'] = lap_ranks[idx] + ten_ranks[idx]
            r['lap_rank'] = lap_ranks[idx]
            r['ten_rank'] = ten_ranks[idx]

        # 按综合分数排序
        successful.sort(key=lambda x: x['combined_score'])
        for rank, r in enumerate(successful, 1):
            r['final_rank'] = rank

    # 输出结果
    print(f"{'序号':>4} {'图片名':40s} | {'Laplacian':>10} | {'Tenengrad':>12} | {'综合排名':>8}")
    print("-" * 90)
    for r in successful:
        print(f"{r['final_rank']:>4} {r['path'].name:40s} | {r['laplacian']:>10.2f} | {r['tenengrad']:>12.2f} | {r['final_rank']:>8}")

    return results


def print_statistics(results):
    """打印统计信息"""
    successful_results = [r for r in results if r['success']]

    if not successful_results:
        print("\n没有成功处理的图片")
        return

    laplacians = [r['laplacian'] for r in successful_results]
    tenengrads = [r['tenengrad'] for r in successful_results]
    times = [r['time'] for r in successful_results]

    print("\n" + "=" * 90)
    print("统计信息:")
    print("-" * 90)
    print(f"总图片数:     {len(results)}")
    print(f"成功处理:     {len(successful_results)}")
    print(f"处理失败:     {len(results) - len(successful_results)}")
    print()
    print(f"Laplacian 统计:")
    print(f"  最小值:     {min(laplacians):.2f}")
    print(f"  最大值:     {max(laplacians):.2f}")
    print(f"  平均值:     {np.mean(laplacians):.2f}")
    print()
    print(f"Tenengrad 统计:")
    print(f"  最小值:     {min(tenengrads):.2f}")
    print(f"  最大值:     {max(tenengrads):.2f}")
    print(f"  平均值:     {np.mean(tenengrads):.2f}")
    print()
    print(f"平均处理时间: {np.mean(times)*1000:.2f}ms")
    print("=" * 90)


def main():
    if len(sys.argv) < 2:
        print("用法: python lpls.py <image_path_or_folder>")
        print("示例: python lpls.py ../test_data/cards/test1.jpg")
        print("示例: python lpls.py ../test_data/cards/")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"错误: 路径不存在: {input_path}")
        sys.exit(1)

    try:
        if input_path.is_file():
            # 处理单张图片
            result = process_single_image(input_path)
            if result['success']:
                print(f"图像路径:   {result['path']}")
                print(f"处理时间:   {result['time']*1000:.2f}ms")
                print(f"Laplacian:  {result['laplacian']:.2f}")
                print(f"Tenengrad:  {result['tenengrad']:.2f}")
            else:
                print(f"错误: {result['error']}")
                sys.exit(1)

        elif input_path.is_dir():
            # 处理文件夹
            results = process_folder(input_path)
            if results:
                print_statistics(results)

        else:
            print(f"错误: {input_path} 不是有效的文件或文件夹")
            sys.exit(1)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()