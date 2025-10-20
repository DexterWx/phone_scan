use opencv::{
    core::{AlgorithmHint, Mat, Point2i, Point2f, Size, Vector},
    imgcodecs::{imdecode, imread, IMREAD_COLOR},
    imgproc,
    prelude::*,
};
use base64::{Engine as _, engine::general_purpose};
use anyhow::{Result, Context};
use crate::models::{ProcessedImage, Coordinate, Quad};
use crate::config::{LocationConfig, ImageProcessingConfig};

pub fn read_image(input: &String) -> Result<Mat> {
    // 判断输入是文件路径还是base64字符串
    // 根据需求：长度超过200认为是base64，否则是路径
    if input.len() > 200 {
        // 处理base64字符串
        let image_data = general_purpose::STANDARD
            .decode(input)
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
    } else {
        // 处理文件路径
        let mat = imread(input, IMREAD_COLOR)
            .context("读取图片文件失败")?;
        
        // 检查图片是否为空
        if mat.empty() {
            anyhow::bail!("读取成功，但图片为空");
        }
        
        Ok(mat)
    }
}

pub fn resize_image(image: &Mat, target_width: i32) -> Result<Mat> {
    let mut resized = Mat::default();
    let scale = target_width as f64 / image.cols() as f64;
    imgproc::resize(image, &mut resized, Size::new(target_width, -1), scale, scale, imgproc::INTER_LINEAR)?;
    Ok(resized)
}

/// 图片预处理：灰度化、高斯模糊、二值化、形态学操作
pub fn process_image(image: &Mat) -> Result<ProcessedImage> {
    // 0. 图片统一到宽度
    let resized = resize_image(image, LocationConfig::TARGET_WIDTH)?;

    // 1. 灰度化
    let mut gray = Mat::default();
    imgproc::cvt_color(&resized, &mut gray, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

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
        LocationConfig::BLOCK_SIZE,
        LocationConfig::C as f64,
    )?;

    // 4. 形态学闭操作
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(LocationConfig::MORPH_KERNEL, LocationConfig::MORPH_KERNEL),
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

/// 计算透视变换矩阵
/// detected_quad: 检测到的四边形（实际图片中的四边形）
/// target_rect: 目标矩形区域（xywh格式）
pub fn get_perspective_transform_matrix(
    detected_quad: &Quad,
    target_rect: &Coordinate,
) -> Result<Mat> {
    // 将检测到的点转换为OpenCV格式
    let src_points = get_points_from_quad(detected_quad);
    let target_points = get_points_from_coordinate(target_rect);

    // 计算透视变换矩阵
    let transform_matrix = imgproc::get_perspective_transform(&src_points, &target_points, 0)
        .context("计算透视变换矩阵失败")?;

    Ok(transform_matrix)
}

/// 将四边形转换为OpenCV格式
pub fn get_points_from_quad(quad: &Quad) -> Vector<Point2f> {
    // 将检测到的点转换为OpenCV格式
    let points = Vector::<Point2f>::from_slice(&[
        Point2f::new(quad.points[0].x as f32, quad.points[0].y as f32),
        Point2f::new(quad.points[1].x as f32, quad.points[1].y as f32),
        Point2f::new(quad.points[2].x as f32, quad.points[2].y as f32),
        Point2f::new(quad.points[3].x as f32, quad.points[3].y as f32),
    ]);
    points
}

pub fn get_points_from_coordinate(coordinate: &Coordinate) -> Vector<Point2f> {
    let points = Vector::<Point2f>::from_slice(&[
        Point2f::new(coordinate.x as f32, coordinate.y as f32),                                    // 左上角
        Point2f::new((coordinate.x + coordinate.w) as f32, coordinate.y as f32),                 // 右上角
        Point2f::new((coordinate.x + coordinate.w) as f32, (coordinate.y + coordinate.h) as f32), // 右下角
        Point2f::new(coordinate.x as f32, (coordinate.y + coordinate.h) as f32),                 // 左下角
    ]);
    points
}


/// 透视变换
pub fn pers_trans_image(
    processed_image: &ProcessedImage,
    transform_matrix: &Mat,
    target_w: i32,
    target_h: i32
) -> Result<ProcessedImage> {
    // 对所有图像应用透视变换
    let mut gray_warped = Mat::default();
    imgproc::warp_perspective(
        &processed_image.gray,
        &mut gray_warped,
        &transform_matrix,
        Size::new(target_w, target_h),
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    ).context("应用透视变换到灰度图失败")?;

    let mut thresh_warped = Mat::default();
    imgproc::warp_perspective(
        &processed_image.thresh,
        &mut thresh_warped,
        &transform_matrix,
        Size::new(target_w, target_h),
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    ).context("应用透视变换到二值图失败")?;

    let mut closed_warped = Mat::default();
    imgproc::warp_perspective(
        &processed_image.closed,
        &mut closed_warped,
        &transform_matrix,
        Size::new(target_w, target_h),
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    ).context("应用透视变换到形态学处理图失败")?;

    Ok(ProcessedImage {
        gray: gray_warped,
        thresh: thresh_warped,
        closed: closed_warped,
    })
}


/// 计算积分图
pub fn integral_image(image: &Mat) -> Result<Mat> {
    // 检查输入图像是否为空
    if image.empty() {
        anyhow::bail!("输入图像为空");
    }
    
    // 创建输出积分图
    let mut integral = Mat::default();
    
    // 使用OpenCV内置的积分图函数（简化版本，只需要3个参数）
    imgproc::integral(
        image,
        &mut integral,
        -1  // sdepth: 积分图的数据深度（-1表示与输入图像相同）
    ).context("计算积分图失败")?;
    
    Ok(integral)
}


