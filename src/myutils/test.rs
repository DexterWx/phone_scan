use opencv::{
    core::{Point, Scalar, Vector},
    imgcodecs::imencode,
    imgproc::{fill_poly, LINE_8},
    prelude::*,
    Result,
};
use base64::{Engine as _, engine::general_purpose};

// 辅助函数：生成图片Mat对象
pub fn create_triangle_image() -> opencv::Result<opencv::core::Mat> {
    let mut img = opencv::core::Mat::new_rows_cols_with_default(400, 400, opencv::core::CV_8UC3, opencv::core::Scalar::new(255.0, 255.0, 255.0, 0.0))?;
    let triangle_points = opencv::core::Vector::<opencv::core::Point>::from_slice(&[
        opencv::core::Point::new(200, 100),  // 顶点
        opencv::core::Point::new(100, 300),  // 左下角
        opencv::core::Point::new(300, 300),  // 右下角
    ]);
    opencv::imgproc::fill_poly(&mut img, &triangle_points, opencv::core::Scalar::new(0.0, 0.0, 0.0, 0.0), opencv::imgproc::LINE_8, 0, opencv::core::Point::new(0, 0))?;
    Ok(img)
}


pub fn generate_triangle_image_internal() -> Result<String> {
    // 创建白色背景图片 (400x400)
    let mut img = Mat::new_rows_cols_with_default(400, 400, opencv::core::CV_8UC3, Scalar::new(255.0, 255.0, 255.0, 0.0))?;
    
    // 定义三角形的三个顶点 (黑色三角形)
    let triangle_points = Vector::<Point>::from_slice(&[
        Point::new(200, 100),  // 顶点
        Point::new(100, 300),  // 左下角
        Point::new(300, 300),  // 右下角
    ]);
    
    // 绘制黑色三角形
    fill_poly(&mut img, &triangle_points, Scalar::new(0.0, 0.0, 0.0, 0.0), LINE_8, 0, Point::new(0, 0))?;
    
    // 编码为PNG格式
    let mut buffer = Vector::<u8>::new();
    imencode(".png", &img, &mut buffer, &Vector::<i32>::new())?;
    
    // 转换为base64 (使用新的API)
    let base64_string = general_purpose::STANDARD.encode(buffer.as_slice());
    
    Ok(base64_string)
}
