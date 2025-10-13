use opencv::{
    core::{Mat, Size, Point2i, AlgorithmHint},
    imgcodecs::{imdecode, IMREAD_COLOR},
    imgproc,
    prelude::*,
};
use base64::{Engine as _, engine::general_purpose};
use anyhow::{Result, Context};
use crate::models::ProcessedImage;
use crate::config::{BoundaryDetectionConfig, ImageProcessingConfig};

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

/// 图片预处理：灰度化、高斯模糊、二值化、形态学操作
pub fn process_image(image: &Mat) -> Result<ProcessedImage> {
    // 1. 灰度化
    let mut gray = Mat::default();
    imgproc::cvt_color(image, &mut gray, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // 2. 高斯模糊
    let mut blur = Mat::default();
    let kernel_size = Size::new(ImageProcessingConfig::GAUSSIAN_KERNEL_SIZE, ImageProcessingConfig::GAUSSIAN_KERNEL_SIZE);
    imgproc::gaussian_blur(&gray, &mut blur, kernel_size, ImageProcessingConfig::GAUSSIAN_SIGMA, ImageProcessingConfig::GAUSSIAN_SIGMA, opencv::core::BORDER_DEFAULT, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // 3. 自适应阈值二值化
    let mut thresh = Mat::default();
    imgproc::adaptive_threshold(
        &blur,
        &mut thresh,
        255.0,
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
        imgproc::THRESH_BINARY_INV,
        BoundaryDetectionConfig::BLOCK_SIZE,
        BoundaryDetectionConfig::C as f64,
    )?;

    // 4. 形态学闭操作
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(BoundaryDetectionConfig::MORPH_KERNEL, BoundaryDetectionConfig::MORPH_KERNEL),
        Point2i::new(-1, -1),
    )?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &thresh,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point2i::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    )?;

    Ok(ProcessedImage {
        gray,
        thresh,
        closed,
    })
}