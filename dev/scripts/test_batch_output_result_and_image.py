#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import ctypes
import platform
import time
import json
import cv2
import numpy as np
from ctypes import c_char, c_char_p, c_uint8, c_uint32, POINTER, c_size_t, Structure
from pathlib import Path

# 尝试导入 HEIF 支持
try:
    from pillow_heif import register_heif_opener
    from PIL import Image
    HEIF_SUPPORT = True
    register_heif_opener()
except ImportError:
    HEIF_SUPPORT = False


# 定义 InferenceBatchResult 结构体
class InferenceBatchResult(Structure):
    _fields_ = [
        ("json", POINTER(c_char)),      # JSON 字符串指针
        ("image_data", POINTER(c_uint8)),  # RGB 图片数据指针
        ("width", c_uint32),            # 图片宽度
        ("height", c_uint32),           # 图片高度
    ]


def load_library(library_path: str):
    """加载动态库"""
    if not os.path.exists(library_path):
        raise FileNotFoundError(f"Library file '{library_path}' does not exist.")

    try:
        if platform.system().lower() == "windows":
            os.add_dll_directory(r"E:\app\opencv\4.12\opencv\build\x64\vc16\bin")
        lib = ctypes.CDLL(library_path)
        return lib
    except OSError as e:
        raise RuntimeError(f"Failed to load library: {e}")


def setup_function_signatures(lib):
    """设置函数签名"""
    # initialize_paper
    lib.initialize_paper.argtypes = [c_char_p]
    lib.initialize_paper.restype = POINTER(c_char)

    # initialize
    lib.initialize.argtypes = [c_char_p]
    lib.initialize.restype = POINTER(c_char)

    # inference_paper_and_return_rgb
    lib.inference_paper_and_return_rgb.argtypes = [
        POINTER(c_uint8),  # data_ptr
        c_size_t,          # data_len
    ]
    lib.inference_paper_and_return_rgb.restype = InferenceBatchResult

    # free_string
    lib.free_string.argtypes = [POINTER(c_char)]
    lib.free_string.restype = None

    # free_image_data
    lib.free_image_data.argtypes = [POINTER(c_uint8)]
    lib.free_image_data.restype = None

    # destroy_engine
    lib.destroy_engine.argtypes = []
    lib.destroy_engine.restype = None


def read_file_safely(file_path: str) -> str:
    """安全读取文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_image_with_opencv(image_file: str, output_format: str = '.jpg', quality: int = 95) -> bytes:
    """使用 OpenCV 读取图片并转换为字节数据"""
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file '{image_file}' does not exist.")

    file_ext = os.path.splitext(image_file)[1].lower()
    is_heic = file_ext in ['.heic', '.heif']

    img = None

    # 处理 HEIC/HEIF 格式
    if is_heic:
        if not HEIF_SUPPORT:
            raise RuntimeError(f"HEIC/HEIF format not supported. Please install pillow-heif: pip install pillow-heif")

        try:
            pil_img = Image.open(image_file)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            img_array = np.array(pil_img)
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"Failed to load HEIC image '{image_file}': {e}")
    else:
        img = cv2.imread(image_file)
        if img is None:
            try:
                pil_img = Image.open(image_file)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                img_array = np.array(pil_img)
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception as pil_error:
                raise RuntimeError(f"Failed to load image '{image_file}': {pil_error}")

    # 设置编码参数
    encode_params = []
    if output_format.lower() == '.jpg':
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif output_format.lower() == '.png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]

    # 将图片编码为指定格式的字节数据
    success, encoded_img = cv2.imencode(output_format, img, encode_params)
    if not success:
        raise RuntimeError(f"Failed to encode image '{image_file}' to {output_format}")

    return encoded_img.tobytes()


def rgb_to_bgr_image(rgb_data: np.ndarray, width: int, height: int) -> np.ndarray:
    """将 RGB 数据转换为 BGR 图片（OpenCV 格式）"""
    # 重塑为 (height, width, 3)
    img_rgb = rgb_data.reshape((height, width, 3))
    # 转换为 BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def process_images_in_folder(lib, input_folder: str, output_folder: str):
    """处理文件夹中的所有图片"""
    # 获取所有支持的图片文件
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.heic', '.heif', '.webp']
    image_files = []

    for ext in supported_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))

    # 去重（Windows 上 glob 不区分大小写，会导致重复）
    image_files = list(set(image_files))
    # 排序以保证处理顺序一致
    image_files.sort()

    if not image_files:
        print(f"⚠️  No images found in {input_folder}")
        return

    print(f"📁 Found {len(image_files)} images in {input_folder}")

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 处理每张图片
    for idx, image_file in enumerate(image_files):
        image_name = image_file.stem
        print(f"\n[{idx + 1}/{len(image_files)}] Processing: {image_file.name}")

        try:
            # 读取图片
            image_data = read_image_with_opencv(str(image_file))
            image_array = (c_uint8 * len(image_data)).from_buffer_copy(image_data)

            # 调用推理接口
            start_time = time.time()
            result = lib.inference_paper_and_return_rgb(
                image_array,
                len(image_data)
            )
            elapsed_time = time.time() - start_time

            # 解析 JSON 结果
            if not result.json:
                print(f"  ❌ Inference returned null JSON pointer")
                continue

            json_str = ctypes.string_at(result.json).decode('utf-8')
            json_result = json.loads(json_str)

            print(f"  ✅ Inference completed in {elapsed_time:.3f}s")
            print(f"  📊 Result code: {json_result.get('code')}, message: {json_result.get('message')}")

            # 创建输出子文件夹
            output_subfolder = os.path.join(output_folder, f"{idx + 1:04d}_{image_name}")
            os.makedirs(output_subfolder, exist_ok=True)

            # 保存 JSON 结果
            json_output_path = os.path.join(output_subfolder, "result.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, ensure_ascii=False, indent=2)
            print(f"  💾 JSON saved to: {json_output_path}")

            # 保存 RGB 图片
            if result.image_data and result.width > 0 and result.height > 0:
                # 将 C 指针转换为 numpy 数组
                image_size = result.width * result.height * 3
                rgb_array = np.ctypeslib.as_array(result.image_data, shape=(image_size,))

                # 转换为 BGR 图片
                bgr_image = rgb_to_bgr_image(rgb_array, result.width, result.height)

                # 保存图片
                image_output_path = os.path.join(output_subfolder, "output.jpg")
                cv2.imwrite(image_output_path, bgr_image)
                print(f"  🖼️  Image saved to: {image_output_path} ({result.width}x{result.height})")

                # 释放图片数据
                lib.free_image_data(result.image_data)
            else:
                print(f"  ⚠️  No image data returned")

            # 释放 JSON 字符串
            lib.free_string(result.json)

        except Exception as e:
            print(f"  ❌ Error processing {image_file.name}: {e}")
            continue


def main():
    if len(sys.argv) != 6:
        print("Usage: python test_batch_output_result_and_image.py <library_path> <init_json_file> <input_folder> <output_folder> <use_paper>")
        print("  library_path: path to the dynamic library (.dll/.so/.dylib)")
        print("  init_json_file: path to the initialization JSON file")
        print("  input_folder: folder containing input images")
        print("  output_folder: folder to save results (will create subfolders)")
        print("  use_paper: 'paper' or 'single'")
        print("\nExample:")
        print("  python test_batch_output_result_and_image.py target/release/phone_scan.dll config.json input_images output_results paper")
        sys.exit(1)

    library_path = sys.argv[1]
    init_json_file = sys.argv[2]
    input_folder = sys.argv[3]
    output_folder = sys.argv[4]
    use_paper = sys.argv[5].lower() == 'paper'

    try:
        # 加载动态库
        print("📚 Loading library...")
        lib = load_library(library_path)
        setup_function_signatures(lib)

        # 初始化引擎
        print("🔧 Initializing engine...")
        init_json_content = read_file_safely(init_json_file)
        if use_paper:
            init_result_ptr = lib.initialize_paper(init_json_content.encode('utf-8'))
        else:
            init_result_ptr = lib.initialize(init_json_content.encode('utf-8'))

        if not init_result_ptr:
            raise RuntimeError("Initialize returned null pointer")

        init_result_str = ctypes.string_at(init_result_ptr).decode('utf-8')
        init_result = json.loads(init_result_str)
        print(f"📋 Initialization result: {init_result}")
        lib.free_string(init_result_ptr)

        if init_result.get('code') != 0:
            raise RuntimeError(f"Initialization failed: {init_result.get('message')}")

        # 处理图片
        print(f"\n🚀 Starting batch processing...")
        process_images_in_folder(lib, input_folder, output_folder)

        # 销毁引擎
        print(f"\n🧹 Destroying engine...")
        lib.destroy_engine()
        print(f"✅ All done!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
