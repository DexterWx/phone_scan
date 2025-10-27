pub mod myutils;
pub mod models;
pub mod recognize;
pub mod config;

#[cfg(test)]
mod tests {
    use std::fs;
    use opencv::imgcodecs::imread;
    use crate::myutils::myjson::to_json;
    use crate::recognize::engine;
    use anyhow::Result;

    use super::*;

    #[test]
    fn test_demo() -> Result<()> {
        let scan_id = "2";
        let scan_path = format!("dev/test_data/cards/{scan_id}/test.json");
        let img_path = format!("dev/test_data/cards/{scan_id}/test.jpg");
        let image = imread(&img_path, opencv::imgcodecs::IMREAD_COLOR)?;

        let scan_string = fs::read_to_string(scan_path)?;

        let engine = engine::RecEngine::new(&scan_string)?;
        let res = engine.inference(&image)?;

        fs::write(format!("dev/test_data/out/{scan_id}.json"), to_json(&res)?)?;

        Ok(())
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

    /// 释放C字符串内存
    #[no_mangle]
    pub extern "C" fn free_string(s: *mut c_char) {
        if !s.is_null() {
            unsafe {
                let _cstring = CString::from_raw(s);
            }
        }
    }
}