use anyhow::{Result, Context};
use opencv::{core::{Mat, MatTraitConst, AlgorithmHint}, imgcodecs::{imdecode, IMREAD_COLOR}, imgproc, prelude::MatTraitConstManual};
use serde_json;
use std::{any::type_name, ffi::{c_char, CStr}};

pub fn from_json<T>(json_str: &str) -> Result<T>
where
    T: for<'de> serde::Deserialize<'de>,
{
    serde_json::from_str(json_str)
        .context(format!("{} 反序列化失败", type_name::<T>()))
}

pub fn to_json<T>(value: &T) -> Result<String>
where
    T: serde::Serialize,
{
    serde_json::to_string(value)
        .context(format!("{} 序列化失败", type_name::<T>()))
}

pub fn c_to_string(input_c: *const c_char) -> String {
    // 将 C 字符串指针转换为 Rust 字符串
    let c_str = unsafe { CStr::from_ptr(input_c) };
    let mark_str = c_str.to_string_lossy().into_owned();
    return mark_str;
}

pub fn c_to_mat(data_ptr: *const u8, data_len: usize) -> Result<Mat> {
    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
    // 直接使用切片，避免内存拷贝
    let img = imdecode(&data, IMREAD_COLOR)?;
    Ok(img)
}

/// 将 BGR Mat 转换为 RGB 格式，返回数据指针、宽度和高度
/// 调用方负责使用 free_image_data 释放内存
pub fn mat_to_c(image: &Mat) -> Result<(*mut u8, u32, u32)> {
    let mut rgb = Mat::default();
    imgproc::cvt_color(image, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    let rows = rgb.rows() as u32;
    let cols = rgb.cols() as u32;
    let data = rgb.data_bytes()?;

    // 分配内存并复制数据
    let len = data.len();
    let ptr = unsafe {
        let layout = std::alloc::Layout::from_size_align(len, 1)
            .context("内存布局错误")?;
        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            anyhow::bail!("内存分配失败");
        }
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, len);
        ptr
    };

    Ok((ptr, cols, rows))
}
