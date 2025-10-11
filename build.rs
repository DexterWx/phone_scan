fn main() {
    // 只在 iOS 目标上执行
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "ios" {
        // 获取 OpenCV 路径
        let opencv_dir = std::env::var("OPENCV_DIR_IOS").unwrap_or_else(|_| {
            "/Users/xu.wang/workspace/gitlab/opencv/ios_build/build/build-arm64-iphoneos".to_string()
        });
        
        let install_dir = format!("{}/install", opencv_dir);
        let lib_dir = format!("{}/lib", install_dir);
        let thirdparty_dir = format!("{}/lib/3rdparty", install_dir);
        let include_dir = format!("{}/include", install_dir);
        
        // 添加库搜索路径
        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:rustc-link-search=native={}", thirdparty_dir);
        
        // 添加包含路径
        println!("cargo:rustc-link-arg=-I{}", include_dir);
        
        // 链接 OpenCV 主库
        println!("cargo:rustc-link-lib=static=opencv_world");
        
        // 链接第三方库
        println!("cargo:rustc-link-lib=static=libjpeg-turbo");
        println!("cargo:rustc-link-lib=static=libpng");
        println!("cargo:rustc-link-lib=static=libwebp");
        println!("cargo:rustc-link-lib=static=zlib");
        
        // iOS 系统框架
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=UIKit");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        println!("cargo:rustc-link-lib=framework=CoreMedia");
        println!("cargo:rustc-link-lib=framework=AVFoundation");
        
        // 强制重新链接
        println!("cargo:rerun-if-changed={}", lib_dir);
        println!("cargo:rerun-if-changed={}", thirdparty_dir);
    }
}