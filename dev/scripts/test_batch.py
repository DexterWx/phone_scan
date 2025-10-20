#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import glob
import json
import time
import ctypes
from pathlib import Path
from ctypes import c_uint8
from test import load_library, setup_function_signatures, read_file_safely, read_image_with_opencv


def get_output_format_from_filename(image_path: str) -> str:
    return '.bmp'
    """根据文件名自动判断输出格式"""
    file_ext = os.path.splitext(image_path)[1].lower()
    
    # HEIC/HEIF 格式转换为 JPEG
    if file_ext in ['.heic', '.heif']:
        return '.jpg'
    
    # 其他格式保持原格式
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        return file_ext
    
    # 默认使用 JPEG
    return '.jpg'


def get_image_files(directory: str) -> list:
    """获取目录下所有图片文件，包括 HEIC/HEIF 格式"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.heic', '*.heif']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern))
        # 也搜索大写扩展名（Windows 上 glob 是大小写不敏感的，但为了保险起见）
        pattern = os.path.join(directory, ext.upper())
        image_files.extend(glob.glob(pattern))
    
    # 去重并排序
    return sorted(list(set(image_files)))


def safe_image_path(image_path: str) -> str:
    """处理中文文件名路径问题"""
    try:
        # 尝试使用原始路径
        if os.path.exists(image_path):
            return image_path
        
        # 如果原始路径失败，尝试不同的编码方式
        import locale
        system_encoding = locale.getpreferredencoding()
        
        # 尝试使用系统默认编码
        try:
            encoded_path = image_path.encode(system_encoding)
            decoded_path = encoded_path.decode(system_encoding)
            if os.path.exists(decoded_path):
                return decoded_path
        except:
            pass
        
        # 尝试使用 UTF-8 编码
        try:
            encoded_path = image_path.encode('utf-8')
            decoded_path = encoded_path.decode('utf-8')
            if os.path.exists(decoded_path):
                return decoded_path
        except:
            pass
        
        # 如果都失败了，返回原始路径
        return image_path
        
    except Exception:
        return image_path


def process_single_image(lib, image_path: str, quality: int = 95, resize_width: int = None) -> dict:
    """处理单张图片，支持 HEIC/HEIF 格式，自动判断输出格式，增强错误处理"""
    try:
        # 处理中文文件名路径问题
        safe_path = safe_image_path(image_path)
        
        # 检查文件是否存在
        if not os.path.exists(safe_path):
            return {"error": f"File does not exist: {image_path} (safe_path: {safe_path})"}
        
        # 检查文件大小
        file_size = os.path.getsize(safe_path)
        if file_size == 0:
            return {"error": f"File is empty: {image_path}"}
        
        # 根据文件名自动判断输出格式
        output_format = get_output_format_from_filename(image_path)
        
        # 使用 OpenCV 读取图片（支持 HEIC/HEIF）
        image_data = read_image_with_opencv(safe_path, output_format, quality, resize_width)
        image_array = (c_uint8 * len(image_data)).from_buffer_copy(image_data)
        
        # 调用推理
        start_time = time.time()
        infer_result_ptr = lib.inference(image_array, len(image_data))
        elapsed_time = time.time() - start_time
        
        if not infer_result_ptr:
            return {"error": "Inference returned null pointer"}
        
        # 获取结果
        infer_result_str = ctypes.string_at(infer_result_ptr).decode('utf-8')
        infer_result = json.loads(infer_result_str)
        
        # 释放内存
        lib.free_string(infer_result_ptr)
        
        return {"result": infer_result, "elapsed_time": elapsed_time}
        
    except Exception as e:
        return {"error": str(e)}


def process_directory(lib, input_dir: str, output_dir: str, subdir_name: str, quality: int = 95, resize_width: int = None):
    """处理单个子目录"""
    # 获取所有图片文件
    image_files = get_image_files(input_dir)
    
    if not image_files:
        return 0, 0, []
    
    # 创建输出目录
    subdir_output = os.path.join(output_dir, subdir_name)
    os.makedirs(subdir_output, exist_ok=True)
    
    # 处理每张图片
    success_count = 0
    error_count = 0
    error_images = []  # 记录错误的图片路径和错误信息
    
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        image_stem = os.path.splitext(image_name)[0]
        
        result = process_single_image(lib, image_path, quality, resize_width)
        
        # 保存结果
        output_filename = f"{subdir_name}_{image_stem}.json"
        output_path = os.path.join(subdir_output, output_filename)
        
        # 添加元数据
        result_with_meta = {
            "image_path": image_path,
            "image_name": image_name,
            "processing_time": result['elapsed_time'],
            "timestamp": time.time(),
            "result": result['result']
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_with_meta, f, ensure_ascii=False, indent=2)
            
            if "error" in result:
                error_count += 1
                error_images.append({
                    "path": image_path,
                    "name": image_name,
                    "error": result['error']
                })
            elif result.get('code') == 1:
                error_count += 1
                error_images.append({
                    "path": image_path,
                    "name": image_name,
                    "error": result.get('message', 'Unknown error')
                })
            else:
                success_count += 1
                
        except Exception as e:
            error_count += 1
            error_images.append({
                "path": image_path,
                "name": image_name,
                "error": f"Failed to save result: {e}"
            })
    
    return success_count, error_count, error_images


def main():
    """主函数"""
    if len(sys.argv) < 5 or len(sys.argv) > 7:
        print("Usage: python test_batch.py <library_path> <init_json_file> <input_directory> <output_directory> [quality] [resize_width]")
        print("  image_file: supports .jpg, .png, .bmp, .heic, .heif (format auto-detected)")
        print("  quality: JPEG quality 1-100 (default: 95)")
        print("  resize_width: resize image width (default: keep original)")
        print("Example: python test_batch.py target/release/phone_scan.dll dev/test_data/cards/1/test.json D:/download/photo_images D:/output")
        print("  With options: python test_batch.py target/release/phone_scan.dll config.json D:/photos D:/output 90 1000")
        print("\nDirectory structure:")
        print("  input_directory/")
        print("    ├── 1/")
        print("    │   ├── image1.jpg")
        print("    │   ├── image2.png")
        print("    │   └── image3.heic")
        print("    ├── 2/")
        print("    │   └── image4.jpg")
        print("    └── ...")
        sys.exit(1)

    library_path = sys.argv[1]
    init_json_file = sys.argv[2]
    input_directory = sys.argv[3]
    output_directory = sys.argv[4]
    
    # 可选参数
    quality = int(sys.argv[5]) if len(sys.argv) > 5 else 95
    resize_width = int(sys.argv[6]) if len(sys.argv) > 6 else None

    try:
        # 检查输入目录
        if not os.path.exists(input_directory):
            raise FileNotFoundError(f"Input directory '{input_directory}' does not exist.")
        
        # 创建输出目录
        os.makedirs(output_directory, exist_ok=True)
        
        # 加载库和初始化引擎
        lib = load_library(library_path)
        setup_function_signatures(lib)
        
        init_json_content = read_file_safely(init_json_file)
        init_result_ptr = lib.initialize(init_json_content.encode('utf-8'))
        
        if not init_result_ptr:
            raise RuntimeError("Initialize returned null pointer")
        
        init_result_str = ctypes.string_at(init_result_ptr).decode('utf-8')
        init_result = json.loads(init_result_str)
        lib.free_string(init_result_ptr)
        
        if init_result.get('code') != 0:
            raise RuntimeError(f"Initialization failed: {init_result.get('message')}")
        
        # 获取所有数字子目录
        subdirs = []
        for item in os.listdir(input_directory):
            item_path = os.path.join(input_directory, item)
            if os.path.isdir(item_path) and item.isdigit():
                subdirs.append(item)
        
        subdirs.sort(key=int)  # 按数字排序
        
        if not subdirs:
            print(f"⚠️  No numeric subdirectories found in {input_directory}")
            sys.exit(1)
        
        # 处理每个子目录
        total_success = 0
        total_errors = 0
        all_error_images = []  # 收集所有错误的图片路径
        start_time = time.time()
        
        for subdir in subdirs:
            input_subdir = os.path.join(input_directory, subdir)
            success, errors, error_images = process_directory(lib, input_subdir, output_directory, subdir, quality, resize_width)
            total_success += success
            total_errors += errors
            all_error_images.extend(error_images)
        
        total_time = time.time() - start_time
        
        # 输出总结
        if all_error_images:
            print(f"\n❌ Failed images ({len(all_error_images)} total):")
            for i, error_info in enumerate(all_error_images, 1):
                print(f"   {i:2d}. {error_info['name']}")
                print(f"       Path: {error_info['path']}")
                print(f"       Error: {error_info['error']}")
                print()
        else:
            print(f"\n🎉 All images processed successfully!")
        
        print(f"📊 Total results: ✅ {total_success} success, ❌ {total_errors} errors")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
