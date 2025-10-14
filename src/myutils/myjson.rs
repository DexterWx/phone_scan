use anyhow::{Result, Context};
use opencv::{core::{Mat, Vector}, imgcodecs::{imdecode, IMREAD_COLOR}};
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
    let data_vec = Vector::<u8>::from_iter(data.iter().cloned());
    
    let img = imdecode(&data_vec, IMREAD_COLOR)?;
    Ok(img)
}
