#!/usr/bin/env python3
"""
NV12 批量推理测试脚本
从文件夹读取图片，转换为 NV12 格式，调用 inference_batch 进行识别
"""

import ctypes
import os
import sys
import json
from pathlib import Path

# 抑制 macOS objc 重复类警告（cv2 与 rust opencv 冲突）
_stderr_fd = os.dup(2)
os.close(2)
_devnull = os.open(os.devnull, os.O_RDWR)
import cv2
import numpy as np
os.dup2(_stderr_fd, 2)
os.close(_stderr_fd)
os.close(_devnull)

# ============== 参数配置区 ==============
IMG_DIR = "/Users/xu.wang/Downloads/test_batch"      # 图片文件夹路径
MARK_JSON = "dev/test_data/cards/13601/test.json"  # 模板 JSON 路径
ROTATION = 0                               # 旋转角度: 0, 90, 180, 270
# ========================================


def find_library():
    """查找动态库路径"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # 优先查找 release 版本
    lib_paths = [
        project_root / "target" / "release" / "libphone_scan.dylib",
        project_root / "target" / "release" / "libphone_scan.so",
        project_root / "target" / "debug" / "libphone_scan.dylib",
        project_root / "target" / "debug" / "libphone_scan.so",
    ]

    for path in lib_paths:
        if path.exists():
            return str(path)

    return None


def load_library():
    """加载动态库"""
    lib_path = find_library()
    if lib_path is None:
        print("错误: 未找到动态库，请先编译项目:")
        print("  cargo build --release")
        sys.exit(1)

    print(f"加载库: {lib_path}")
    lib = ctypes.CDLL(lib_path)

    # 设置函数签名
    lib.initialize_paper.argtypes = [ctypes.c_char_p]
    lib.initialize_paper.restype = ctypes.c_char_p

    lib.inference_batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # images
        ctypes.POINTER(ctypes.c_uint32), # widths
        ctypes.POINTER(ctypes.c_uint32), # heights
        ctypes.POINTER(ctypes.c_uint8),  # rotations
        ctypes.POINTER(ctypes.c_uint32), # lens
        ctypes.c_uint32,                 # count
    ]
    lib.inference_batch.restype = ctypes.c_char_p

    lib.destroy_engine.argtypes = []
    lib.destroy_engine.restype = None

    lib.free_string.argtypes = [ctypes.c_char_p]
    lib.free_string.restype = None

    return lib


def bgr_to_nv12(bgr_image):
    """将 BGR 图像转换为 NV12 格式"""
    height, width = bgr_image.shape[:2]

    # BGR -> YUV (I420)
    yuv_i420 = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV_I420)

    # I420 的 U 和 V 平面是分开的，NV12 的 UV 是交错的
    # I420: Y + U + V
    # NV12: Y + UV (interleaved)

    y_size = width * height
    uv_size = y_size // 4

    y_plane = yuv_i420[:height, :].flatten()
    u_plane = yuv_i420[height:height + height // 4, :].flatten()
    v_plane = yuv_i420[height + height // 4:, :].flatten()

    # 交错 UV
    uv_interleaved = np.empty(uv_size * 2, dtype=np.uint8)
    uv_interleaved[0::2] = u_plane
    uv_interleaved[1::2] = v_plane

    # 拼接 Y + UV
    nv12_data = np.concatenate([y_plane, uv_interleaved])

    return nv12_data.tobytes(), width, height


def load_images_as_nv12(img_dir, rotation=0):
    """从文件夹加载图片并转换为 NV12 格式"""
    images_data = []
    widths = []
    heights = []
    rotations = []
    lens = []
    image_paths = []

    # 支持的图片格式
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    # 读取文件夹中的所有图片
    img_path = Path(img_dir)
    if not img_path.exists():
        print(f"错误: 文件夹不存在: {img_dir}")
        return None

    for file_path in sorted(img_path.iterdir()):
        if file_path.suffix.lower() in supported_exts:
            print(f"  读取: {file_path}")
            bgr_image = cv2.imread(str(file_path))
            if bgr_image is None:
                print(f"    警告: 无法读取图片，跳过")
                continue

            # 转换为 NV12
            nv12_data, width, height = bgr_to_nv12(bgr_image)

            images_data.append(nv12_data)
            widths.append(width)
            heights.append(height)
            rotations.append(rotation)
            lens.append(len(nv12_data))
            image_paths.append(str(file_path))

            print(f"    尺寸: {width}x{height}, NV12 大小: {len(nv12_data)} bytes")

    if not images_data:
        print("错误: 没有找到有效的图片")
        return None

    return {
        'images_data': images_data,
        'widths': widths,
        'heights': heights,
        'rotations': rotations,
        'lens': lens,
        'paths': image_paths,
    }


def main():
    import time

    if not IMG_DIR:
        print("请在脚本顶部配置参数:")
        print("  IMG_DIR   - 图片文件夹路径")
        print("  MARK_JSON - 模板 JSON 路径")
        sys.exit(1)

    # 加载库
    lib = load_library()

    # 初始化引擎
    print(f"\n初始化引擎: {MARK_JSON}")
    with open(MARK_JSON, 'r') as f:
        mark_json = f.read()

    init_result = lib.initialize_paper(mark_json.encode('utf-8'))
    init_info = json.loads(init_result.decode('utf-8'))

    if init_info.get('code', 1) != 0:
        print(f"初始化失败: {init_info.get('message', 'unknown error')}")
        sys.exit(1)

    print(f"初始化成功: {init_info.get('message')}")

    # 加载图片并转换为 NV12
    print(f"\n加载图片: {IMG_DIR}")
    data = load_images_as_nv12(IMG_DIR, ROTATION)
    if data is None:
        lib.destroy_engine()
        sys.exit(1)

    count = len(data['images_data'])
    print(f"\n共加载 {count} 张图片")

    # 拼接所有图片数据
    all_images_data = b''.join(data['images_data'])

    # 准备 ctypes 参数
    images_array = (ctypes.c_uint8 * len(all_images_data)).from_buffer_copy(all_images_data)
    widths_array = (ctypes.c_uint32 * count)(*data['widths'])
    heights_array = (ctypes.c_uint32 * count)(*data['heights'])
    rotations_array = (ctypes.c_uint8 * count)(*data['rotations'])
    lens_array = (ctypes.c_uint32 * count)(*data['lens'])

    # 调用 inference_batch
    print("\n开始推理...")
    start_time = time.time()

    result = lib.inference_batch(
        ctypes.cast(images_array, ctypes.POINTER(ctypes.c_uint8)),
        ctypes.cast(widths_array, ctypes.POINTER(ctypes.c_uint32)),
        ctypes.cast(heights_array, ctypes.POINTER(ctypes.c_uint32)),
        ctypes.cast(rotations_array, ctypes.POINTER(ctypes.c_uint8)),
        ctypes.cast(lens_array, ctypes.POINTER(ctypes.c_uint32)),
        count,
    )

    elapsed_time = time.time() - start_time

    # 解析结果
    result_json = json.loads(result.decode('utf-8'))

    print("\n" + "=" * 50)
    print("推理结果:")
    print(f"  耗时: {elapsed_time * 1000:.2f} ms")
    print(f"  code: {result_json.get('code')}")
    print(f"  message: {result_json.get('message')}")
    print(f"  page_number: {result_json.get('page_number')}")

    rec_results = result_json.get('rec_results', [])
    print(f"  识别项数: {len(rec_results)}")
    print("=" * 50)

    # 保存完整结果
    output_path = "dev/test_data/out/batch_nv12_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存到: {output_path}")

    # 销毁引擎
    lib.destroy_engine()
    print("\n引擎已销毁")


if __name__ == "__main__":
    main()
