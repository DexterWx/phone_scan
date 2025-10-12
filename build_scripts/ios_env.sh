#!/bin/bash

# iOS OpenCV 环境配置脚本
# 用于配置在 M1 Mac 上编译的 iOS OpenCV 静态库

echo "========================================"
echo "Setting up iOS OpenCV Environment"
echo "========================================"

# 设置您编译的 OpenCV iOS 路径
export OPENCV_DIR_IOS="/Users/xu.wang/workspace/gitlab/opencv/ios_build/build/build-arm64-iphoneos"
export OPENCV_DIR="$OPENCV_DIR_IOS/install"

# OpenCV 库配置
export OPENCV_LINK_LIBS="opencv_world"
export OPENCV_LINK_PATHS="$OPENCV_DIR_IOS/install/lib"
export OPENCV_INCLUDE_PATHS="$OPENCV_DIR_IOS/install/include"

# 第三方库路径
export OPENCV_LINK_PATHS_THIRDPARTY="$OPENCV_DIR_IOS/install/lib/3rdparty"

# iOS 目标配置
export IOS_TARGET_ARCH="arm64"
export IOS_TARGET_VERSION="18.5"

# Rust 目标
export RUST_TARGET="aarch64-apple-ios"

# 设置 OpenCV 版本和构建类型
export OPENCV_VERSION="4"
export OPENCV_BUILD_TYPE="Release"

# 设置 clang 路径（iOS 工具链）
export LIBCLANG_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib"

# 设置 iOS SDK 路径
export IOS_SDK_PATH="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk"

echo ""
echo "Environment variables set:"
echo "OPENCV_DIR_IOS = $OPENCV_DIR_IOS"
echo "OPENCV_DIR = $OPENCV_DIR"
echo "OPENCV_LINK_LIBS = $OPENCV_LINK_LIBS"
echo "OPENCV_LINK_PATHS = $OPENCV_LINK_PATHS"
echo "OPENCV_INCLUDE_PATHS = $OPENCV_INCLUDE_PATHS"
echo "OPENCV_VERSION = $OPENCV_VERSION"
echo "OPENCV_BUILD_TYPE = $OPENCV_BUILD_TYPE"
echo "IOS_TARGET_ARCH = $IOS_TARGET_ARCH"
echo "IOS_TARGET_VERSION = $IOS_TARGET_VERSION"
echo "RUST_TARGET = $RUST_TARGET"
echo "LIBCLANG_PATH = $LIBCLANG_PATH"
echo "IOS_SDK_PATH = $IOS_SDK_PATH"
echo ""

# 验证 OpenCV 安装
echo "Checking iOS OpenCV installation..."
if [ -d "$OPENCV_DIR_IOS" ]; then
    echo "✓ iOS OpenCV directory found"
    
    # 检查 include 目录
    if [ -d "$OPENCV_DIR_IOS/install/include" ]; then
        echo "✓ OpenCV include directory found"
        if [ -f "$OPENCV_DIR_IOS/install/include/opencv2/opencv.hpp" ]; then
            echo "✓ OpenCV headers found"
        else
            echo "✗ OpenCV headers not found"
        fi
    else
        echo "✗ OpenCV include directory not found"
    fi
    
    # 检查 lib 目录
    if [ -d "$OPENCV_DIR_IOS/install/lib" ]; then
        echo "✓ OpenCV lib directory found"
        # 检查静态库文件
        if ls "$OPENCV_DIR_IOS/install/lib"/libopencv_world*.a 1> /dev/null 2>&1; then
            echo "✓ OpenCV static libraries found"
            echo "Available libraries:"
            ls -la "$OPENCV_DIR_IOS/install/lib"/libopencv_world*.a
        else
            echo "✗ OpenCV static libraries not found"
            echo "Contents of lib directory:"
            ls -la "$OPENCV_DIR_IOS/install/lib/"
        fi
    else
        echo "✗ OpenCV lib directory not found"
    fi
else
    echo "✗ iOS OpenCV directory not found at $OPENCV_DIR_IOS"
    echo "Please check your OPENCV_DIR_IOS path"
fi

# 验证 Xcode 工具链
echo ""
echo "Checking Xcode toolchain..."
if [ -d "$LIBCLANG_PATH" ]; then
    echo "✓ Xcode clang library found"
else
    echo "✗ Xcode clang library not found"
fi

if [ -d "$IOS_SDK_PATH" ]; then
    echo "✓ iOS SDK found"
else
    echo "✗ iOS SDK not found"
fi

echo ""
echo "========================================"
echo "iOS OpenCV Environment Setup Complete"
echo "========================================"
echo ""
echo "You can now run:"
echo "  cargo build --release --target aarch64-apple-ios"
echo ""
echo "The generated static library will be at:"
echo "  target/aarch64-apple-ios/release/libphone_scan.a"
echo ""
