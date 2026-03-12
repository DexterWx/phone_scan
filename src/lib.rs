pub mod myutils;
pub mod models;
pub mod recognize;
pub mod config;

#[cfg(test)]
mod tests {
    use std::fs;
    use opencv::core::MatTraitConst;
    use opencv::imgcodecs::imread;
    use crate::myutils::myjson::to_json;
    use crate::myutils::image::{calc_laplacian_variance, select_clearest_image_owned, read_images_from_dir};
    use crate::recognize::engine;
    use anyhow::Result;

    use super::*;

    #[test]
    fn test_demo() -> Result<()> {
        let scan_id = "13412";
        let scan_path = format!("dev/test_data/cards/{scan_id}/test.json");
        let img_path = format!("dev/test_data/cards/{scan_id}/test.jpg");
        let image = imread(&img_path, opencv::imgcodecs::IMREAD_COLOR)?;

        let scan_string = fs::read_to_string(scan_path)?;

        let engine = engine::RecEngine::new_single(&scan_string)?;
        let res = engine.inference_single(&image)?;

        fs::write(format!("dev/test_data/out/{scan_id}.json"), to_json(&res)?)?;

        Ok(())
    }

    #[test]
    fn test_paper() -> Result<()> {
        let scan_id = "13587";
        let scan_path = format!("dev/test_data/cards/{scan_id}/test.json");
        let img_path = format!("dev/test_data/cards/{scan_id}/test.jpg");
        let image = imread(&img_path, opencv::imgcodecs::IMREAD_COLOR)?;

        let scan_string = fs::read_to_string(scan_path)?;

        let engine = engine::RecEngine::new_paper(&scan_string)?;
        let (res, _rgb) = engine.inference_paper(&image)?;

        fs::write(format!("dev/test_data/out/{scan_id}.json"), to_json(&res)?)?;

        Ok(())
    }

    /// 批量推理测试: 模拟 inference_batch 接口的完整流程
    /// 读取图片 -> 转 NV12 -> 解码并选择最清晰 -> 识别
    #[test]
    fn test_batch_inference() -> Result<()> {
        use crate::myutils::image::{encode_nv12, decode_nv12_batch_and_select_clearest};

        let BATCH_SCAN_ID = "13603";
        let BATCH_IMG_DIR = "/Users/xu.wang/Downloads/1.9case77ti3";
        let ROTATION: u8 = 0; // 测试旋转角度
        let scan_path = format!("dev/test_data/cards/{}/test.json", BATCH_SCAN_ID);
        let engine = engine::RecEngine::new_paper(&fs::read_to_string(&scan_path)?)?;

        let (images, paths) = read_images_from_dir(BATCH_IMG_DIR)?;
        if images.is_empty() {
            println!("警告: 文件夹 {} 中没有找到图片", BATCH_IMG_DIR);
            return Ok(());
        }

        println!("找到 {} 张图片，转换为 NV12 格式...", images.len());

        // 1. 将所有图片转成 NV12 格式，模拟移动端数据
        let mut all_nv12_data: Vec<u8> = Vec::new();
        let mut widths: Vec<u32> = Vec::new();
        let mut heights: Vec<u32> = Vec::new();
        let mut rotations: Vec<u8> = Vec::new();
        let mut lens: Vec<u32> = Vec::new();

        for (idx, (image, path)) in images.iter().zip(paths.iter()).enumerate() {
            let (nv12_data, width, height) = encode_nv12(image)?;
            println!("  [{}] {} - {}x{}, NV12 size: {} bytes",
                idx, path, width, height, nv12_data.len());

            lens.push(nv12_data.len() as u32);
            widths.push(width as u32);
            heights.push(height as u32);
            rotations.push(ROTATION);
            all_nv12_data.extend(nv12_data);
        }

        println!("\n总 NV12 数据: {} bytes", all_nv12_data.len());

        // 2. 使用 decode_nv12_batch_and_select_clearest 解码并选择最清晰
        let (clearest_image, _image_index) = decode_nv12_batch_and_select_clearest(
            &all_nv12_data,
            &widths,
            &heights,
            &rotations,
            &lens,
        )?;

        println!("选择最清晰的图片: {}x{}", clearest_image.cols(), clearest_image.rows());

        // 3. 进行识别
        let (res, _rgb) = engine.inference_paper(&clearest_image)?;
        let output_path = format!("dev/test_data/out/batch_{}.json", BATCH_SCAN_ID);
        fs::write(&output_path, to_json(&res)?)?;
        println!("识别结果已保存到: {}", output_path);

        Ok(())
    }

    #[test]
    fn test_crop() -> Result<()> {
        let scan_id = "13588";
        let scan_path = format!("dev/test_data/cards/{scan_id}/test.json");
        let img_path = format!("dev/test_data/cards/{scan_id}/test.jpg");
        let image = imread(&img_path, opencv::imgcodecs::IMREAD_COLOR)?;

        let scan_string = fs::read_to_string(scan_path)?;

        let engine = engine::RecEngine::new_paper(&scan_string)?;
        let _ = engine.make_vx_data(
            &image,
            &"dev/test_data/mark_test".to_string(),
            &"A3_wangxu_1".to_string()
        )?;
        Ok(())
    }

    #[test]
    fn test_vx() -> Result<()> {
        let scan_id = "13603";
        let scan_path = format!("dev/test_data/cards/{scan_id}/test.json");
        let img_path = format!("/Users/xu.wang/workspace/gitlab/phone_scan/dev/test_data/debug/sub_images/sub_0_6.jpg");
        let image = imread(&img_path, opencv::imgcodecs::IMREAD_GRAYSCALE)?;

        let scan_string = fs::read_to_string(scan_path)?;

        let engine = engine::RecEngine::new_paper(&scan_string)?;
        let (res, conf) = engine.rec_vx_module.infer_tiny_cnn(&image)?;
        println!("识别结果: {} {}", res, conf);
        Ok(())
    }

}


pub mod build {
    use std::ffi::{c_char, CString};
    use opencv::core::Mat;
    use crate::myutils::image::decode_nv12_batch_and_select_clearest;
    use crate::{models::{InitInfo, MobileOutput}, myutils::myjson::{c_to_mat, c_to_string, mat_to_c, to_json}, recognize::engine::RecEngine};
    static mut ENGINE: Option<RecEngine> = None;

    /// FFI 返回结构体，包含 JSON 和 RGB 图片数据
    #[repr(C)]
    pub struct InferenceBatchResult {
        /// JSON 字符串指针
        pub json: *mut c_char,
        /// RGB 图片数据指针（3通道）
        pub image_data: *mut u8,
        /// 图片宽度
        pub width: u32,
        /// 图片高度
        pub height: u32,
    }
    
    #[no_mangle]
    pub extern "C" fn initialize(mark_ptr: *const c_char) -> *mut c_char{
        let mark_str = c_to_string(mark_ptr);

        let engine = RecEngine::new_single(&mark_str);
        
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
    pub extern "C" fn initialize_paper(mark_ptr: *const c_char) -> *mut c_char{

        let mark_str = c_to_string(mark_ptr);

        let mut res = InitInfo {
            code: 0,
            message: "初始化成功".to_string(),
        };

        let engine = RecEngine::new_paper(&mark_str);

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
            page_number: 0,
            image_index: 0,
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
            let success_output = engine.inference_single(&image.unwrap());
            if success_output.is_err() {
                failed_output.message = success_output.err().unwrap().to_string();
                return CString::new(to_json(&failed_output).unwrap()).unwrap().into_raw();
            }
            return CString::new(to_json(&success_output.unwrap()).unwrap()).unwrap().into_raw();
        }
    }

    #[no_mangle]
    pub extern "C" fn inference_paper(data_ptr: *const u8, data_len: usize) -> *mut c_char {
        let mut failed_output = MobileOutput {
            code: 1,
            message: "failed".to_string(),
            page_number: 0,
            image_index: 0,
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
            let success_output = engine.inference_paper(&image.unwrap());
            if success_output.is_err() {
                failed_output.message = success_output.err().unwrap().to_string();
                return CString::new(to_json(&failed_output).unwrap()).unwrap().into_raw();
            }
            let (output, _rgb) = success_output.unwrap();
            return CString::new(to_json(&output).unwrap()).unwrap().into_raw();
        }
    }

    /// 批量推理接口
    /// 从多张 NV12 图片中选择最清晰的一张进行识别
    ///
    /// 参数:
    /// - images: 所有图片拼接后的连续内存首地址 (NV12 格式)
    /// - widths: 宽度数组指针
    /// - heights: 高度数组指针
    /// - rotations: 旋转角度数组指针 (0, 90, 180, 270)
    /// - lens: 每张图片的字节长度数组指针
    /// - count: 图片数量
    ///
    /// 返回: JSON 字符串 (MobileOutput)
    #[no_mangle]
    pub extern "C" fn inference_batch(
        images: *const u8,
        widths: *const u32,
        heights: *const u32,
        rotations: *const u8,
        lens: *const u32,
        count: u32,
    ) -> *mut c_char {

        let mut failed_output = MobileOutput {
            code: 1,
            message: "failed".to_string(),
            page_number: 0,
            image_index: 0,
            rec_results: vec![],
        };

        // 检查引擎是否初始化
        unsafe {
            if ENGINE.is_none() {
                failed_output.message = "请先初始化引擎".to_string();
                return CString::new(to_json(&failed_output).unwrap()).unwrap().into_raw();
            }
        }

        // 检查图片数量
        if count == 0 {
            failed_output.message = "图片数量为 0".to_string();
            return CString::new(to_json(&failed_output).unwrap()).unwrap().into_raw();
        }

        // 将指针转换为切片
        let lens_slice = unsafe { std::slice::from_raw_parts(lens, count as usize) };
        let total_len: usize = lens_slice.iter().map(|&x| x as usize).sum();
        let images_slice = unsafe { std::slice::from_raw_parts(images, total_len) };
        let widths_slice = unsafe { std::slice::from_raw_parts(widths, count as usize) };
        let heights_slice = unsafe { std::slice::from_raw_parts(heights, count as usize) };
        let rotations_slice = unsafe { std::slice::from_raw_parts(rotations, count as usize) };

        // 解码并选择最清晰图片
        let (clearest_image, image_index) = match decode_nv12_batch_and_select_clearest(
            images_slice,
            widths_slice,
            heights_slice,
            rotations_slice,
            lens_slice,
        ) {
            Ok(result) => result,
            Err(e) => {
                failed_output.message = e.to_string();
                return CString::new(to_json(&failed_output).unwrap()).unwrap().into_raw();
            }
        };

        // 使用最清晰的图片进行识别
        unsafe {
            let engine = ENGINE.as_ref().unwrap();
            match engine.inference_paper(&clearest_image) {
                Ok((mut output, _rgb)) => {
                    output.image_index = image_index;
                    CString::new(to_json(&output).unwrap()).unwrap().into_raw()
                },
                Err(e) => {
                    failed_output.message = e.to_string();
                    CString::new(to_json(&failed_output).unwrap()).unwrap().into_raw()
                }
            }
        }
    }

    /// 批量推理接口 V2 - 返回 JSON 和 RGB 图片数据
    /// 从多张 NV12 图片中选择最清晰的一张进行识别
    ///
    /// 参数:
    /// - images: 所有图片拼接后的连续内存首地址 (NV12 格式)
    /// - widths: 宽度数组指针
    /// - heights: 高度数组指针
    /// - rotations: 旋转角度数组指针 (0, 90, 180, 270)
    /// - lens: 每张图片的字节长度数组指针
    /// - count: 图片数量
    ///
    /// 返回: InferenceBatchResult 包含 JSON 和 RGB 图片数据
    #[no_mangle]
    pub extern "C" fn inference_batch_and_return_rgb(
        images: *const u8,
        widths: *const u32,
        heights: *const u32,
        rotations: *const u8,
        lens: *const u32,
        count: u32,
    ) -> InferenceBatchResult {

        let mut failed_output = MobileOutput {
            code: 1,
            message: "failed".to_string(),
            page_number: 0,
            image_index: 0,
            rec_results: vec![],
        };

        // 辅助函数：创建失败结果
        let make_failed_result = |output: &MobileOutput| -> InferenceBatchResult {
            InferenceBatchResult {
                json: CString::new(to_json(output).unwrap()).unwrap().into_raw(),
                image_data: std::ptr::null_mut(),
                width: 0,
                height: 0,
            }
        };

        // 检查引擎是否初始化
        unsafe {
            if ENGINE.is_none() {
                failed_output.message = "请先初始化引擎".to_string();
                return make_failed_result(&failed_output);
            }
        }

        // 检查图片数量
        if count == 0 {
            failed_output.message = "图片数量为 0".to_string();
            return make_failed_result(&failed_output);
        }

        // 将指针转换为切片
        let lens_slice = unsafe { std::slice::from_raw_parts(lens, count as usize) };
        let total_len: usize = lens_slice.iter().map(|&x| x as usize).sum();
        let images_slice = unsafe { std::slice::from_raw_parts(images, total_len) };
        let widths_slice = unsafe { std::slice::from_raw_parts(widths, count as usize) };
        let heights_slice = unsafe { std::slice::from_raw_parts(heights, count as usize) };
        let rotations_slice = unsafe { std::slice::from_raw_parts(rotations, count as usize) };

        // 解码并选择最清晰图片
        let (clearest_image, image_index) = match decode_nv12_batch_and_select_clearest(
            images_slice,
            widths_slice,
            heights_slice,
            rotations_slice,
            lens_slice,
        ) {
            Ok(result) => result,
            Err(e) => {
                failed_output.message = e.to_string();
                return make_failed_result(&failed_output);
            }
        };

        // 使用最清晰的图片进行识别
        unsafe {
            let engine = ENGINE.as_ref().unwrap();
            match engine.inference_paper(&clearest_image) {
                Ok((mut output, rgb)) => {
                    output.image_index = image_index;
                    // 转换 RGB 图片数据
                    match mat_to_c(&rgb) {
                        Ok((image_data, width, height)) => {
                            InferenceBatchResult {
                                json: CString::new(to_json(&output).unwrap()).unwrap().into_raw(),
                                image_data,
                                width,
                                height,
                            }
                        }
                        Err(e) => {
                            failed_output.message = format!("图片转换失败: {}", e);
                            make_failed_result(&failed_output)
                        }
                    }
                }
                Err(e) => {
                    failed_output.message = e.to_string();
                    make_failed_result(&failed_output)
                }
            }
        }
    }

    /// 销毁引擎，释放资源
    #[no_mangle]
    pub extern "C" fn destroy_engine() {
        unsafe {
            ENGINE = None; // 将引擎设置为None，触发Drop释放资源
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

    /// 释放 RGB 图片数据内存
    #[no_mangle]
    pub extern "C" fn free_image_data(image_data: *mut u8, width: u32, height: u32) {
        if !image_data.is_null() && width > 0 && height > 0 {
            let len = (width * height * 3) as usize; // RGB 3 通道
            unsafe {
                let layout = std::alloc::Layout::from_size_align(len, 1).unwrap();
                std::alloc::dealloc(image_data, layout);
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn create_train_data(data_ptr: *const u8, data_len: usize, out_dir: *mut c_char, file_name: *mut c_char) -> *mut c_char {
        let mut failed_output = MobileOutput {
            code: 1,
            message: "failed".to_string(),
            page_number: 0,
            image_index: 0,
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
            let success_output = engine.make_vx_data(&image.unwrap(), &c_to_string(out_dir), &c_to_string(file_name));
            if success_output.is_err() {
                failed_output.message = success_output.err().unwrap().to_string();
                return CString::new(to_json(&failed_output).unwrap()).unwrap().into_raw();
            }
            return CString::new(to_json(&success_output.unwrap()).unwrap()).unwrap().into_raw();
        }
    }

}