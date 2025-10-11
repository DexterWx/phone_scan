pub mod myutils;
pub mod models;
pub mod recognize;

// 添加这行导入
use myutils::test::generate_triangle_image_internal;
use std::ffi::{CString, c_char};

/// 生成一个黑色三角形在白色背景上的图片，返回base64编码
#[no_mangle]
pub extern "C" fn generate_triangle_image() -> *mut c_char {
    match generate_triangle_image_internal() {
        Ok(base64_string) => {
            let c_string = CString::new(base64_string).unwrap();
            c_string.into_raw()
        }
        Err(_) => {
            let error_string = CString::new("Error generating image").unwrap();
            error_string.into_raw()
        }
    }
}

/// 释放C字符串内存
#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use opencv::imgcodecs::imwrite;
    use crate::myutils::test::create_triangle_image;

    use super::*;

    #[test]
    fn test_generate_triangle_image() {
        // 创建测试数据目录
        let test_dir = "dev/test_data";
        if !Path::new(test_dir).exists() {
            fs::create_dir_all(test_dir).expect("无法创建测试目录");
        }
        
        // 生成图片并获取base64
        let result = generate_triangle_image_internal();
        assert!(result.is_ok());
        let base64 = result.unwrap();
        assert!(!base64.is_empty());
        
        println!("Generated base64 length: {}", base64.len());
        println!("Generated base64 (first 100 chars): {}", &base64[..100.min(base64.len())]);
        
        // 在测试中保存图片文件用于验证
        let output_path = format!("{}/triangle_test.png", test_dir);
        let img = create_triangle_image().unwrap();
        
        // 保存图片文件
        imwrite(&output_path, &img, &opencv::core::Vector::<i32>::new()).unwrap();
        println!("图片已保存到: {}", output_path);
        
        // 验证文件是否真的被创建
        assert!(Path::new(&output_path).exists(), "图片文件未被创建");
    }
}