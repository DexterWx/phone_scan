# 构建脚本集合

这个目录包含了用于不同平台的构建脚本。

## Android构建

### Windows
```batch
build_android.bat
```

### Linux/macOS
```bash
chmod +x build_android.sh
./build_android.sh
```

## iOS构建 (仅macOS)

```bash
chmod +x build_ios.sh
./build_ios.sh
```

## 环境设置

### Android环境变量
编辑 `android_env.bat` (Windows) 或设置环境变量 (Linux/macOS):

```bash
export ANDROID_NDK_HOME=/path/to/your/ndk
export OPENCV_DIR=/path/to/your/opencv
```

### 安装Rust目标

```bash
# Android
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add i686-linux-android
rustup target add x86_64-linux-android

# iOS (仅macOS)
rustup target add aarch64-apple-ios
rustup target add x86_64-apple-ios
```

## 输出

- Android: `target/*/release/libphone_scan.so`
- iOS: `target/*/release/libphone_scan.a`
- iOS通用库: `dist/ios/libphone_scan.a`
