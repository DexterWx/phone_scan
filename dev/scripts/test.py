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
    """设置函数签名（适配 CString::into_raw()）"""
    # Rust: pub extern "C" fn initialize(config_json: *const c_char) -> *mut c_char
    lib.initialize.argtypes = [c_char_p]
    lib.initialize.restype = POINTER(c_char)  # ✅ 改成裸指针

    # Rust: pub extern "C" fn inference(image_data: *const u8, len: usize) -> *mut c_char
    lib.inference.argtypes = [POINTER(c_uint8), c_size_t]
    lib.inference.restype = POINTER(c_char)  # ✅ 改成裸指针

    # Rust: pub extern "C" fn free_string(s: *mut c_char)
    lib.free_string.argtypes = [POINTER(c_char)]  # ✅ 改成裸指针
    lib.free_string.restype = None


def read_file_safely(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_image_with_opencv(image_file: str, output_format: str = '.jpg', quality: int = 95, resize_width: int = None) -> bytes:
    """使用 OpenCV 读取图片并转换为字节数据，支持 HEIC/HEIF 格式
    
    Args:
        image_file: 图片文件路径
        output_format: 输出格式 ('.jpg', '.png', '.bmp')
        quality: JPEG 质量 (1-100, 仅对 JPEG 有效)
        resize_width: 调整图片宽度 (None 表示保持原尺寸)
    
    Returns:
        编码后的图片字节数据
    """
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file '{image_file}' does not exist.")
    
    # 检查文件扩展名
    file_ext = os.path.splitext(image_file)[1].lower()
    is_heic = file_ext in ['.heic', '.heif']
    
    img = None
    
    # 处理 HEIC/HEIF 格式
    if is_heic:
        if not HEIF_SUPPORT:
            raise RuntimeError(f"HEIC/HEIF format not supported. Please install pillow-heif: pip install pillow-heif")
        
        try:
            # 使用 PIL 读取 HEIC 图片
            pil_img = Image.open(image_file)
            # 转换为 RGB 模式（如果需要）
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # 转换为 OpenCV 格式 (BGR)
            img_array = np.array(pil_img)
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HEIC image '{image_file}': {e}")
    
    else:
        # 使用 OpenCV 读取其他格式图片
        img = cv2.imread(image_file)
        if img is None:
            # 尝试使用 PIL 作为备用方法
            try:
                pil_img = Image.open(image_file)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                img_array = np.array(pil_img)
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception as pil_error:
                raise RuntimeError(f"Failed to load image '{image_file}' with both OpenCV and PIL. OpenCV error: imread returned None, PIL error: {pil_error}")
        else:
            pass  # OpenCV 成功读取图片
    
    # 图片预处理
    if resize_width is not None:
        img = preprocess_image(img, resize_width=resize_width)
    
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


def preprocess_image(img: np.ndarray, resize_width: int = None, resize_height: int = None) -> np.ndarray:
    """图片预处理函数
    
    Args:
        img: OpenCV 图片数组
        resize_width: 目标宽度 (None 表示保持比例)
        resize_height: 目标高度 (None 表示保持比例)
    
    Returns:
        预处理后的图片数组
    """
    processed_img = img.copy()
    
    # 调整图片大小
    if resize_width is not None or resize_height is not None:
        if resize_width is not None and resize_height is not None:
            # 指定具体尺寸
            processed_img = cv2.resize(processed_img, (resize_width, resize_height))
        elif resize_width is not None:
            # 按宽度等比例缩放
            height, width = processed_img.shape[:2]
            new_height = int(height * resize_width / width)
            processed_img = cv2.resize(processed_img, (resize_width, new_height))
        elif resize_height is not None:
            # 按高度等比例缩放
            height, width = processed_img.shape[:2]
            new_width = int(width * resize_height / height)
            processed_img = cv2.resize(processed_img, (new_width, resize_height))
        
    
    return processed_img


def get_platform_specific_example():
    system = platform.system().lower()
    if system == "windows":
        return "python test.py target/release/phone_scan.dll dev/test_data/cards/1/test.json dev/test_data/cards/1/test.jpg"
    elif system == "darwin":
        return "python test.py target/aarch64-apple-darwin/release/libphone_scan.dylib dev/test_data/cards/1/test.json dev/test_data/cards/1/test.jpg"
    else:
        return "python test.py target/release/libphone_scan.so dev/test_data/cards/1/test.json dev/test_data/cards/1/test.jpg"


def main():
    if len(sys.argv) < 4 or len(sys.argv) > 7:
        print("Usage: python test.py <library_path> <init_json_file> <image_file> [output_format] [quality] [resize_width]")
        print("  image_file: supports .jpg, .png, .bmp, .heic, .heif")
        print("  output_format: .jpg, .png, .bmp (default: .jpg)")
        print("  quality: JPEG quality 1-100 (default: 95)")
        print("  resize_width: resize image width (default: keep original)")
        print(f"Example: {get_platform_specific_example()}")
        print("  HEIC Example: python test.py target/release/phone_scan.dll config.json image.heic .jpg 95 1000")
        sys.exit(1)

    library_path = sys.argv[1]
    init_json_file = sys.argv[2]
    image_file = sys.argv[3]
    
    # 可选参数
    output_format = sys.argv[4] if len(sys.argv) > 4 else '.jpg'
    quality = int(sys.argv[5]) if len(sys.argv) > 5 else 95
    resize_width = int(sys.argv[6]) if len(sys.argv) > 6 else None

    try:
        lib = load_library(library_path)
        setup_function_signatures(lib)

        # 初始化
        init_json_content = read_file_safely(init_json_file)
        init_result_ptr = lib.initialize(init_json_content.encode('utf-8'))

        if not init_result_ptr:
            raise RuntimeError("Initialize returned null pointer")

        init_result_str = ctypes.string_at(init_result_ptr).decode('utf-8')
        init_result = json.loads(init_result_str)
        print(f"📋 Initialization result: {init_result}")
        lib.free_string(init_result_ptr)  # ✅ 必须释放

        if init_result.get('code') != 0:
            raise RuntimeError(f"Initialization failed: {init_result.get('message')}")

        # 推理
        image_data = read_image_with_opencv(image_file, output_format, quality, resize_width)
        image_array = (c_uint8 * len(image_data)).from_buffer_copy(image_data)

        start_time = time.time()
        infer_result_ptr = lib.inference(image_array, len(image_data))
        print(f"✅ Inference completed in {time.time() - start_time:.3f}s")

        if not infer_result_ptr:
            raise RuntimeError("Inference returned null pointer")

        infer_result_str = ctypes.string_at(infer_result_ptr).decode('utf-8')
        infer_result = json.loads(infer_result_str)
        print(f"🎯 Inference result: {infer_result['code']}")
        lib.free_string(infer_result_ptr)  # ✅ 必须释放

        if infer_result.get('code') != 0:
            raise RuntimeError(f"Inference failed: {infer_result.get('message')}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
