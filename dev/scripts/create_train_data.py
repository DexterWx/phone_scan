#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import ctypes
import platform
import json
import cv2
import numpy as np
from ctypes import c_char, c_char_p, c_uint8, POINTER, c_size_t

# 尝试导入 HEIF 支持
try:
    from pillow_heif import register_heif_opener
    from PIL import Image
    HEIF_SUPPORT = True
    register_heif_opener()
except ImportError:
    HEIF_SUPPORT = False


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
    # initialize
    lib.initialize_paper.argtypes = [c_char_p]
    lib.initialize_paper.restype = POINTER(c_char)

    # create_train_data(data_ptr: *const u8, data_len: usize, out_dir: *mut c_char, file_name: *mut c_char) -> *mut c_char
    lib.create_train_data.argtypes = [POINTER(c_uint8), c_size_t, c_char_p, c_char_p]
    lib.create_train_data.restype = POINTER(c_char)

    # free_string
    lib.free_string.argtypes = [POINTER(c_char)]
    lib.free_string.restype = None

    # destroy_engine
    lib.destroy_engine.argtypes = None
    lib.destroy_engine.restype = None


def read_file_safely(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_image_with_opencv(image_file: str, output_format: str = '.jpg', quality: int = 100) -> bytes:
    """使用 OpenCV 读取图片并转换为字节数据"""
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file '{image_file}' does not exist.")

    file_ext = os.path.splitext(image_file)[1].lower()
    is_heic = file_ext in ['.heic', '.heif']

    img = None

    if is_heic:
        if not HEIF_SUPPORT:
            raise RuntimeError(f"HEIC/HEIF format not supported. Please install pillow-heif")

        pil_img = Image.open(image_file)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        img_array = np.array(pil_img)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image_file)
        if img is None:
            raise RuntimeError(f"Failed to load image '{image_file}'")

    encode_params = []
    if output_format.lower() == '.jpg':
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif output_format.lower() == '.png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]

    success, encoded_img = cv2.imencode(output_format, img, encode_params)
    if not success:
        raise RuntimeError(f"Failed to encode image '{image_file}' to {output_format}")

    return encoded_img.tobytes()


def get_image_files(image_dir: str) -> list:
    """获取目录下所有图片文件"""
    supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.heic', '.heif']
    image_files = []

    for filename in os.listdir(image_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_exts:
            image_files.append(os.path.join(image_dir, filename))

    return sorted(image_files)


def main():
    if len(sys.argv) != 5:
        print("Usage: python create_train_data.py <library_path> <init_json_file> <image_dir> <out_dir>")
        print("Example: python create_train_data.py target/aarch64-apple-darwin/release/libphone_scan.dylib config.json /path/to/images /path/to/output")
        sys.exit(1)

    library_path = sys.argv[1]
    init_json_file = sys.argv[2]
    image_dir = sys.argv[3]
    out_dir = sys.argv[4]

    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)

    try:
        lib = load_library(library_path)
        setup_function_signatures(lib)

        # 初始化引擎
        init_json_content = read_file_safely(init_json_file)
        init_result_ptr = lib.initialize_paper(init_json_content.encode('utf-8'))

        if not init_result_ptr:
            raise RuntimeError("Initialize returned null pointer")

        init_result_str = ctypes.string_at(init_result_ptr).decode('utf-8')
        init_result = json.loads(init_result_str)
        print(f"📋 Initialization result: {init_result}")
        lib.free_string(init_result_ptr)

        if init_result.get('code') != 0:
            raise RuntimeError(f"Initialization failed: {init_result.get('message')}")

        # 获取所有图片
        image_files = get_image_files(image_dir)
        print(f"📁 Found {len(image_files)} images in {image_dir}")

        # 处理每张图片
        success_count = 0
        fail_count = 0
        failed_files = []

        for i, image_path in enumerate(image_files):
            filename = os.path.basename(image_path)
            # 去掉扩展名作为 file_name
            file_name = os.path.splitext(filename)[0]

            try:
                image_data = read_image_with_opencv(image_path, output_format='.bmp')
                image_array = (c_uint8 * len(image_data)).from_buffer_copy(image_data)

                result_ptr = lib.create_train_data(
                    image_array,
                    len(image_data),
                    out_dir.encode('utf-8'),
                    file_name.encode('utf-8')
                )

                if not result_ptr:
                    print(f"❌ [{i+1}/{len(image_files)}] {filename}: null pointer returned")
                    failed_files.append(image_path)
                    fail_count += 1
                    continue

                result_str = ctypes.string_at(result_ptr).decode('utf-8')
                result = json.loads(result_str)
                lib.free_string(result_ptr)

                if result.get('code') == 0:
                    print(f"✅ [{i+1}/{len(image_files)}] {filename}: success")
                    success_count += 1
                else:
                    print(f"❌ [{i+1}/{len(image_files)}] {filename}: {result.get('message')}")
                    failed_files.append(image_path)
                    fail_count += 1

            except Exception as e:
                print(f"❌ [{i+1}/{len(image_files)}] {filename}: {e}")
                failed_files.append(image_path)
                fail_count += 1

        print(f"\n📊 Summary: {success_count} success, {fail_count} failed, total {len(image_files)}")

        if failed_files:
            print(f"\n❌ Failed files ({len(failed_files)}):")
            for f in failed_files:
                print(f"  {f}")

        lib.destroy_engine()
        print("✅ Engine destroyed")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
