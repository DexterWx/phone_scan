use opencv::{
    core::Mat,
    imgcodecs::{imdecode, IMREAD_COLOR},
    prelude::*,
};
use base64::{Engine as _, engine::general_purpose};
use anyhow::{Result, Context};

pub fn read_image(base64: &str) -> Result<Mat> {
    // 解码 base64 字符串为字节数组
    let image_data = general_purpose::STANDARD
        .decode(base64)
        .context("Base64 解码失败")?;
    
    // 将字节数组转换为 OpenCV 的 Vector<u8>
    let image_vector = opencv::core::Vector::<u8>::from_slice(&image_data);
    
    // 使用 imdecode 将字节数据解码为 Mat 对象
    let mat = imdecode(&image_vector, IMREAD_COLOR)
        .context("字节流 解码失败")?;
    
    // 检查图片是否为空
    if mat.empty() {
        anyhow::bail!("解码成功，但图片为空");
    }
    
    Ok(mat)
}