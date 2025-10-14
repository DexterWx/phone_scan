pub mod myutils;
pub mod models;
pub mod recognize;
pub mod config;

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
    use opencv::imgcodecs::{imread, imwrite};
    use crate::myutils::myjson::to_json;
    use crate::myutils::test::create_triangle_image;
    use crate::recognize::engine;
    use anyhow::Result;

    use super::*;

    #[test]
    fn test_demo() -> Result<()> {
        let scan_id = "1";
        let scan_path = format!("dev/test_data/cards/{scan_id}/test.json");
        let img_path = format!("dev/test_data/cards/{scan_id}/test.jpg");
        let image = imread(&img_path, opencv::imgcodecs::IMREAD_COLOR)?;

        let scan_string = fs::read_to_string(scan_path)?;

        let engine = engine::RecEngine::new(&scan_string)?;
        let res = engine.inference(&image)?;

        fs::write(format!("dev/test_data/out/{scan_id}.json"), to_json(&res)?)?;

        Ok(())
    }

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


pub mod build {
    use std::ffi::{c_char, CString};
    use crate::{models::{InitInfo, MobileOutput}, myutils::myjson::{c_to_mat, c_to_string, to_json}, recognize::engine::RecEngine};

    static mut ENGINE: Option<RecEngine> = None;
    
    #[no_mangle]
    pub extern "C" fn initialize(mark_ptr: *const c_char) -> *mut c_char{
        let mark_str = c_to_string(mark_ptr);

        let engine = RecEngine::new(&mark_str);
        
        let mut res = InitInfo {
            code: 0,
            message: "初始化成功".to_string(),
        };
        
        if engine.is_err() {
            res.code = 1;
            res.message = engine.err().unwrap().to_string();
            return CString::new(to_json(&res).unwrap()).unwrap().into_raw()
        }

        // 初始化引擎
        unsafe {
            ENGINE = Some(engine.unwrap());
        }

        return CString::new(to_json(&res).unwrap()).unwrap().into_raw()
    }


    #[no_mangle]
    pub extern "C" fn inference(data_ptr: *const u8, data_len: usize) -> *mut c_char {
        
        let mut failed_output = MobileOutput {
            code: 1,
            message: "failed".to_string(),
            rec_results: vec![],
        };

        unsafe {
            if ENGINE.is_none() {
                failed_output.message = "请先初始化引擎".to_string();
                return CString::new(to_json(&failed_output).unwrap()).unwrap().into_raw();
            }
        }
        
        let image = c_to_mat(data_ptr, data_len);
        if image.is_err() {
            failed_output.message = image.err().unwrap().to_string();
            return CString::new(to_json(&failed_output).unwrap()).unwrap().into_raw();
        }

        unsafe {
            let engine = ENGINE.as_ref().unwrap();
            let success_output = engine.inference(&image.unwrap());
            if success_output.is_err() {
                failed_output.message = success_output.err().unwrap().to_string();
                return CString::new(to_json(&failed_output).unwrap()).unwrap().into_raw();
            }
            return CString::new(to_json(&success_output.unwrap()).unwrap()).unwrap().into_raw();
        }
    }
}