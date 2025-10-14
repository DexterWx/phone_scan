#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import ctypes
from ctypes import c_char_p, c_void_p, c_int, c_uint8, POINTER, c_size_t


def main():
    # 检查命令行参数
    if len(sys.argv) != 4:
        print("Usage: python test.py <library_path> <init_json_file> <image_file>")
        print("Example: python test.py target/aarch64-apple-darwin/release/libphone_scan.dylib dev/test_data/cards/1/test.json dev/test_data/cards/1/test.jpg")
        sys.exit(1)

    # 获取命令行参数
    library_path = sys.argv[1]
    init_json_file = sys.argv[2]
    image_file = sys.argv[3]

    # 检查文件是否存在
    if not os.path.exists(init_json_file):
        print(f"Error: Initialization JSON file '{init_json_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(image_file):
        print(f"Error: Image file '{image_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(library_path):
        print(f"Error: Library file '{library_path}' does not exist.")
        sys.exit(1)

    # 加载动态库
    try:
        lib = ctypes.CDLL(library_path)
    except OSError as e:
        print(f"Error loading library: {e}")
        sys.exit(1)

    # 设置initialize函数签名
    # initialize(mark_ptr: *const c_char) -> *mut c_char
    lib.initialize.argtypes = [c_char_p]
    lib.initialize.restype = c_char_p

    # 设置inference函数签名
    # inference(data_ptr: *const u8, data_len: usize) -> *mut c_char
    lib.inference.argtypes = [POINTER(c_uint8), c_size_t]
    lib.inference.restype = c_char_p

    # 设置free_string函数签名
    # free_string(s: *mut c_char)
    lib.free_string.argtypes = [c_char_p]
    lib.free_string.restype = None

    # 读取初始化JSON文件
    try:
        with open(init_json_file, 'r', encoding='utf-8') as f:
            init_json_content = f.read()
    except Exception as e:
        print(f"Error reading initialization JSON file: {e}")
        sys.exit(1)

    # 调用initialize函数
    print("Initializing engine...")
    init_result_ptr = lib.initialize(init_json_content.encode('utf-8'))
    
    # 注意：这里我们先复制字符串内容，然后再释放内存
    if init_result_ptr:
        # 直接从指针获取内容，不创建额外的c_char_p对象
        init_result_bytes = ctypes.string_at(init_result_ptr)
        init_result_str = init_result_bytes.decode('utf-8')
        print(f"Initialization result: {init_result_str}")
        # 释放字符串内存
        lib.free_string(init_result_ptr)
    else:
        print("Error: initialize function returned null pointer")
        sys.exit(1)

    # 读取图片文件
    try:
        with open(image_file, 'rb') as f:
            image_data = f.read()
    except Exception as e:
        print(f"Error reading image file: {e}")
        sys.exit(1)

    # 将图片数据转换为C兼容格式
    image_array = (c_uint8 * len(image_data)).from_buffer_copy(image_data)

    # 调用inference函数
    print("Running inference...")
    infer_result_ptr = lib.inference(image_array, len(image_data))
    
    # 注意：这里我们也先复制字符串内容，然后再释放内存
    if infer_result_ptr:
        # 直接从指针获取内容，不创建额外的c_char_p对象
        infer_result_bytes = ctypes.string_at(infer_result_ptr)
        infer_result_str = infer_result_bytes.decode('utf-8')
        print(f"Inference result: {infer_result_str}")
        # 释放字符串内存
        lib.free_string(infer_result_ptr)
    else:
        print("Error: inference function returned null pointer")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()