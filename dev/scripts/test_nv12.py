#!/usr/bin/env python3
"""
NV12/NV21 批量推理测试脚本
从文件夹读取图片（支持 jpg/png 或原始 nv12/nv21 文件），调用 inference_batch 进行识别

原始 YUV 文件宽高:
  1. 优先使用下方配置的 YUV_WIDTH/YUV_HEIGHT
  2. 如果未配置，则从文件名解析: name_WIDTHxHEIGHT.nv12 或 name_WIDTHxHEIGHT.nv21
"""

import ctypes
import os
import sys
import json
import re
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
IMG_DIR = "/Users/xu.wang/Downloads/Batch_20260110_121610_787"      # 图片文件夹路径
MARK_JSON = "dev/test_data/cards/13601/test.json"  # 模板 JSON 路径
ROTATION = 0                            # 旋转角度: 0, 90, 180, 270
OUTPUT_IMAGE = "dev/test_data/out/batch_nv12_render.png"  # 输出渲染图片路径

# NV12/NV21 原始文件的宽高配置（如果文件名中没有宽高信息则使用此配置）
YUV_WIDTH = 2560                        # YUV 图像宽度
YUV_HEIGHT = 1440                       # YUV 图像高度
# ========================================


# 定义 InferenceBatchResult 结构体
class InferenceBatchResult(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.POINTER(ctypes.c_char)),  # 裸指针，避免 Python 自动管理内存
        ("image_data", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
    ]


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

    lib.inference_batch_and_return_rgb.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # images
        ctypes.POINTER(ctypes.c_uint32), # widths
        ctypes.POINTER(ctypes.c_uint32), # heights
        ctypes.POINTER(ctypes.c_uint8),  # rotations
        ctypes.POINTER(ctypes.c_uint32), # lens
        ctypes.c_uint32,                 # count
    ]
    lib.inference_batch_and_return_rgb.restype = InferenceBatchResult

    lib.destroy_engine.argtypes = []
    lib.destroy_engine.restype = None

    lib.free_string.argtypes = [ctypes.POINTER(ctypes.c_char)]  # 裸指针
    lib.free_string.restype = None

    lib.free_image_data.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # image_data
        ctypes.c_uint32,                 # width
        ctypes.c_uint32,                 # height
    ]
    lib.free_image_data.restype = None

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


def nv21_to_nv12(nv21_data, width, height):
    """将 NV21 数据转换为 NV12 格式（交换 UV 顺序）"""
    # NV21: Y + VU (V 在前，U 在后)
    # NV12: Y + UV (U 在前，V 在后)
    y_size = width * height

    nv12_data = bytearray(nv21_data)

    # 交换 UV 平面的顺序
    for i in range(y_size, len(nv12_data), 2):
        if i + 1 < len(nv12_data):
            nv12_data[i], nv12_data[i + 1] = nv12_data[i + 1], nv12_data[i]

    return bytes(nv12_data)


def parse_yuv_filename(filename):
    """
    从文件名中解析宽高信息
    支持格式: name_WIDTHxHEIGHT.nv12 或 name_WIDTHxHEIGHT.nv21
    例如: frame_1920x1080.nv12, image_1280x720.nv21
    返回: (width, height) 或 None
    """
    # 匹配 _数字x数字 模式
    match = re.search(r'_(\d+)x(\d+)\.nv(?:12|21)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def load_raw_yuv_file(file_path):
    """
    加载原始 NV12/NV21 文件
    宽高优先级: 1. 配置的 YUV_WIDTH/YUV_HEIGHT  2. 从文件名解析
    返回: (nv12_data, width, height) 或 None
    """
    file_path = Path(file_path)
    filename = file_path.name.lower()

    # 优先使用配置的宽高，否则从文件名解析
    if YUV_WIDTH > 0 and YUV_HEIGHT > 0:
        width, height = YUV_WIDTH, YUV_HEIGHT
    else:
        dims = parse_yuv_filename(filename)
        if dims is None:
            print(f"    警告: 未配置 YUV_WIDTH/YUV_HEIGHT，且无法从文件名解析宽高")
            print(f"    请在脚本头部配置 YUV_WIDTH 和 YUV_HEIGHT，或使用格式: name_WIDTHxHEIGHT.nv21")
            return None
        width, height = dims

    expected_size = width * height * 3 // 2

    # 读取文件
    with open(file_path, 'rb') as f:
        raw_data = f.read()

    if len(raw_data) != expected_size:
        print(f"    警告: 文件大小不匹配，期望 {expected_size} 字节 ({width}x{height})，实际 {len(raw_data)} 字节")
        return None

    # 如果是 NV21，转换为 NV12
    if filename.endswith('.nv21'):
        nv12_data = nv21_to_nv12(raw_data, width, height)
        return nv12_data, width, height
    else:
        return raw_data, width, height


def load_images_as_nv12(img_dir, rotation=0):
    """从文件夹加载图片并转换为 NV12 格式，支持 jpg/png 和原始 nv12/nv21 文件"""
    images_data = []
    widths = []
    heights = []
    rotations = []
    lens = []
    image_paths = []

    # 支持的格式
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    yuv_exts = {'.nv12', '.nv21'}

    # 读取文件夹中的所有文件
    img_path = Path(img_dir)
    if not img_path.exists():
        print(f"错误: 文件夹不存在: {img_dir}")
        return None

    for file_path in sorted(img_path.iterdir()):
        suffix = file_path.suffix.lower()

        if suffix in image_exts:
            # 处理图片文件
            print(f"  读取图片: {file_path}")
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

        elif suffix in yuv_exts:
            # 处理原始 NV12/NV21 文件
            print(f"  读取 YUV: {file_path}")
            result = load_raw_yuv_file(file_path)
            if result is None:
                continue

            nv12_data, width, height = result

            images_data.append(nv12_data)
            widths.append(width)
            heights.append(height)
            rotations.append(rotation)
            lens.append(len(nv12_data))
            image_paths.append(str(file_path))

            fmt = "NV21->NV12" if suffix == '.nv21' else "NV12"
            print(f"    尺寸: {width}x{height}, {fmt} 大小: {len(nv12_data)} bytes")

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

    # 调用 inference_batch_and_return_rgb
    print("\n开始推理...")
    start_time = time.time()

    result = lib.inference_batch_and_return_rgb(
        ctypes.cast(images_array, ctypes.POINTER(ctypes.c_uint8)),
        ctypes.cast(widths_array, ctypes.POINTER(ctypes.c_uint32)),
        ctypes.cast(heights_array, ctypes.POINTER(ctypes.c_uint32)),
        ctypes.cast(rotations_array, ctypes.POINTER(ctypes.c_uint8)),
        ctypes.cast(lens_array, ctypes.POINTER(ctypes.c_uint32)),
        count,
    )

    elapsed_time = time.time() - start_time

    # 解析 JSON 结果（使用 ctypes.string_at 从裸指针读取字符串）
    json_str = ctypes.string_at(result.json).decode('utf-8')
    result_json = json.loads(json_str)

    print("\n" + "=" * 50)
    print("推理结果:")
    print(f"  耗时: {elapsed_time * 1000:.2f} ms")
    print(f"  code: {result_json.get('code')}")
    print(f"  message: {result_json.get('message')}")
    print(f"  page_number: {result_json.get('page_number')}")

    rec_results = result_json.get('rec_results', [])
    print(f"  识别项数: {len(rec_results)}")
    print(f"  图片尺寸: {result.width}x{result.height}")
    print("=" * 50)

    # 保存完整 JSON 结果
    output_json_path = "dev/test_data/out/batch_nv12_result.json"
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 结果已保存到: {output_json_path}")

    # 保存渲染图片
    if result.width > 0 and result.height > 0 and result.image_data:
        # 将 RGB 数据转换为 numpy 数组
        img_size = result.width * result.height * 3
        rgb_data = np.ctypeslib.as_array(result.image_data, shape=(img_size,))
        rgb_image = rgb_data.reshape((result.height, result.width, 3))

        # RGB -> BGR (OpenCV 格式)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # 保存图片
        os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
        cv2.imwrite(OUTPUT_IMAGE, bgr_image)
        print(f"渲染图片已保存到: {OUTPUT_IMAGE}")

    # 释放内存
    lib.free_string(result.json)
    print("\n已释放 JSON 内存")
    if result.image_data:
        lib.free_image_data(result.image_data, result.width, result.height)
        print("\n已释放图片内存")

    # 销毁引擎
    lib.destroy_engine()
    print("引擎已销毁")


if __name__ == "__main__":
    main()
