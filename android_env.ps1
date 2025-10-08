# Android NDK Environment Configuration (PowerShell版本)
# 这个脚本设置Android开发环境变量

Write-Host "========================================" -ForegroundColor Green
Write-Host "Setting up Android Development Environment" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 设置Android NDK路径 (请根据你的实际安装路径修改)
$env:ANDROID_NDK_HOME = "E:\app\androidstudio\tmp\sdk\ndk\29.0.14206865"
$env:ANDROID_HOME = "E:\app\androidstudio\tmp\sdk"

# 设置Android OpenCV路径 (请根据你的实际OpenCV Android SDK路径修改)
$env:OPENCV_DIR_ANDROID = "D:\download\opencv-4.12.0-android-sdk\OpenCV-android-sdk\sdk"

# 添加Android工具链到PATH
$env:PATH = "$env:PATH;$env:ANDROID_NDK_HOME\toolchains\llvm\prebuilt\windows-x86_64\bin"

# 设置C++标准库路径
$env:CXXFLAGS = "-I$env:ANDROID_NDK_HOME\toolchains\llvm\prebuilt\windows-x86_64\include\c++\v1"

# 设置目标架构
$env:ANDROID_TARGET_ARCH = "arm64-v8a"
$env:ANDROID_TARGET_API = "21"

# Rust目标
$env:RUST_TARGET = "aarch64-linux-android"

# 设置OpenCV环境变量（opencv crate需要的）
$env:OPENCV_LINK_LIBS = "opencv_java4"
$env:OPENCV_LINK_PATHS = "$env:OPENCV_DIR_ANDROID\native\libs\$env:ANDROID_TARGET_ARCH"
$env:OPENCV_INCLUDE_PATHS = "$env:OPENCV_DIR_ANDROID\native\jni\include"
$env:OPENCV_DIR = "$env:OPENCV_DIR_ANDROID"

Write-Host ""
Write-Host "Environment variables set:" -ForegroundColor Yellow
Write-Host "ANDROID_NDK_HOME = $env:ANDROID_NDK_HOME"
Write-Host "ANDROID_HOME = $env:ANDROID_HOME"
Write-Host "OPENCV_DIR_ANDROID = $env:OPENCV_DIR_ANDROID"
Write-Host "OPENCV_LINK_LIBS = $env:OPENCV_LINK_LIBS"
Write-Host "OPENCV_LINK_PATHS = $env:OPENCV_LINK_PATHS"
Write-Host "OPENCV_INCLUDE_PATHS = $env:OPENCV_INCLUDE_PATHS"
Write-Host "OPENCV_DIR = $env:OPENCV_DIR"
Write-Host "ANDROID_TARGET_ARCH = $env:ANDROID_TARGET_ARCH"
Write-Host "ANDROID_TARGET_API = $env:ANDROID_TARGET_API"
Write-Host "RUST_TARGET = $env:RUST_TARGET"
Write-Host ""

# 验证Android NDK安装
Write-Host "Checking Android NDK installation..." -ForegroundColor Cyan
if (Test-Path "$env:ANDROID_NDK_HOME\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android21-clang++.exe") {
    Write-Host "✓ Android NDK toolchain found" -ForegroundColor Green
} else {
    Write-Host "✗ Android NDK toolchain not found" -ForegroundColor Red
    Write-Host "Please check your ANDROID_NDK_HOME path" -ForegroundColor Red
}

# 验证OpenCV Android SDK
Write-Host "Checking OpenCV Android SDK..." -ForegroundColor Cyan
if (Test-Path "$env:OPENCV_DIR_ANDROID") {
    Write-Host "✓ OpenCV Android SDK directory found" -ForegroundColor Green
} else {
    Write-Host "✗ OpenCV Android SDK directory not found at $env:OPENCV_DIR_ANDROID" -ForegroundColor Red
    Write-Host "Please check your OPENCV_DIR_ANDROID path" -ForegroundColor Red
}

# 检查OpenCV库文件
$opencv_lib_path = "$env:OPENCV_DIR_ANDROID\native\libs\$env:ANDROID_TARGET_ARCH"
if (Test-Path $opencv_lib_path) {
    Write-Host "✓ OpenCV native libraries found for $env:ANDROID_TARGET_ARCH" -ForegroundColor Green
} else {
    Write-Host "✗ OpenCV native libraries not found for $env:ANDROID_TARGET_ARCH" -ForegroundColor Red
    Write-Host "Expected path: $opencv_lib_path" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Android Development Environment Setup Complete" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run:" -ForegroundColor Yellow
Write-Host "  .\build_android.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To build for Android targets." -ForegroundColor Yellow
Write-Host ""
