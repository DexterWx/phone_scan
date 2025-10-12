# Windows OpenCV Environment Configuration (PowerShell�汾)
# ����ű�����OpenCV��Windows�ϵĻ�������

Write-Host "========================================" -ForegroundColor Green
Write-Host "Setting up Windows OpenCV Environment" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# ����OpenCV��Ŀ¼
$env:OPENCV_DIR = "E:\app\opencv\4.12\opencv\build\"

# ����OpenCV������
$env:OPENCV_LINK_LIBS = "opencv_world4120"

# ����OpenCV��·��
$env:OPENCV_LINK_PATHS = "E:\app\opencv\4.12\opencv\build\x64\vc16\lib"

# ����OpenCV����·��
$env:OPENCV_INCLUDE_PATHS = "E:\app\opencv\4.12\opencv\build\include"

# ����OpenCV binĿ¼
$env:OPENCV_BIN = "E:\app\opencv\4.12\opencv\build\x64\vc16\bin"

# ���OpenCV bin��PATH
$env:PATH = "$env:PATH;$env:OPENCV_BIN"

# ���ö����OpenCV��������
$env:OPENCV_VERSION = "4120"
$env:OPENCV_BUILD_TYPE = "Release"

# ����LLVM/Clang·��
$env:LIBCLANG_PATH = "E:\app\llvm\install\LLVM\bin"

# ���LLVM bin��PATH
$env:PATH = "$env:PATH;E:\app\llvm\install\LLVM\bin"

Write-Host ""
Write-Host "Environment variables set:" -ForegroundColor Yellow
Write-Host "OPENCV_DIR = $env:OPENCV_DIR"
Write-Host "OPENCV_LINK_LIBS = $env:OPENCV_LINK_LIBS"
Write-Host "OPENCV_LINK_PATHS = $env:OPENCV_LINK_PATHS"
Write-Host "OPENCV_INCLUDE_PATHS = $env:OPENCV_INCLUDE_PATHS"
Write-Host "OPENCV_BIN = $env:OPENCV_BIN"
Write-Host "OPENCV_VERSION = $env:OPENCV_VERSION"
Write-Host "OPENCV_BUILD_TYPE = $env:OPENCV_BUILD_TYPE"
Write-Host "LIBCLANG_PATH = $env:LIBCLANG_PATH"
Write-Host ""

# ��֤OpenCV��װ
Write-Host "Checking OpenCV installation..." -ForegroundColor Cyan
if (Test-Path "$env:OPENCV_DIR\include\opencv2\opencv.hpp") {
    Write-Host "? OpenCV headers found" -ForegroundColor Green
} else {
    Write-Host "? OpenCV headers not found at $env:OPENCV_DIR\include\opencv2\opencv.hpp" -ForegroundColor Red
    Write-Host "Please check your OPENCV_DIR path" -ForegroundColor Red
}

if (Test-Path "$env:OPENCV_LINK_PATHS\opencv_world4120.lib") {
    Write-Host "? OpenCV library found" -ForegroundColor Green
} else {
    Write-Host "? OpenCV library not found at $env:OPENCV_LINK_PATHS\opencv_world4120.lib" -ForegroundColor Red
    Write-Host "Please check your OPENCV_LINK_PATHS and OPENCV_LINK_LIBS" -ForegroundColor Red
}

if (Test-Path "$env:OPENCV_BIN\opencv_world4120.dll") {
    Write-Host "? OpenCV DLL found" -ForegroundColor Green
} else {
    Write-Host "? OpenCV DLL not found at $env:OPENCV_BIN\opencv_world4120.dll" -ForegroundColor Red
    Write-Host "Please check your OPENCV_BIN path" -ForegroundColor Red
}

# ��֤LLVM��װ
Write-Host "Checking LLVM installation..." -ForegroundColor Cyan
if (Test-Path "$env:LIBCLANG_PATH\libclang.dll") {
    Write-Host "? LLVM libclang.dll found" -ForegroundColor Green
} else {
    Write-Host "? LLVM libclang.dll not found at $env:LIBCLANG_PATH\libclang.dll" -ForegroundColor Red
    Write-Host "Please check your LIBCLANG_PATH" -ForegroundColor Red
}

if (Test-Path "$env:LIBCLANG_PATH\clang.exe") {
    Write-Host "? LLVM clang.exe found" -ForegroundColor Green
} else {
    Write-Host "? LLVM clang.exe not found at $env:LIBCLANG_PATH\clang.exe" -ForegroundColor Red
    Write-Host "Please check your LIBCLANG_PATH" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Windows OpenCV Environment Setup Complete" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run:" -ForegroundColor Yellow
Write-Host "  cargo test" -ForegroundColor White
Write-Host ""
Write-Host "To test the OpenCV integration." -ForegroundColor Yellow
Write-Host ""
