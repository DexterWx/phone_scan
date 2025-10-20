use anyhow::Result;
use opencv::{
    core::{Mat, Point, Rect, Scalar, Vector},
    imgproc::{circle, fill_poly, line, rectangle},
    prelude::*,
};
use crate::models::{AssistLocation, Coordinate, MobileOutput, Quad};

/// 渲染模式
#[derive(Debug, Clone, Copy)]
pub enum RenderMode {
    /// 实心矩形
    Filled,
    /// 空心矩形
    Hollow,
    /// 只渲染4个角点
    Corners,
}

/// 渲染坐标到图像上
pub fn render_coordinate(
    image: &mut Mat,
    coord: &Coordinate,
    mode: Option<RenderMode>,
    color: Option<Scalar>,
    thickness: Option<i32>,
) -> Result<()> {
    let mode = mode.unwrap_or(RenderMode::Corners);
    let color = color.unwrap_or(Colors::red());
    let thickness = thickness.unwrap_or(2);
    let rect = Rect::new(coord.x, coord.y, coord.w, coord.h);
    
    match mode {
        RenderMode::Filled => {
            rectangle(
                image,
                rect,
                color,
                -1, // -1 表示填充
                8,
                0,
            )?;
        }
        RenderMode::Hollow => {
            rectangle(
                image,
                rect,
                color,
                thickness,
                8,
                0,
            )?;
        }
        RenderMode::Corners => {
            let corners = [
                Point::new(coord.x, coord.y),                           // 左上
                Point::new(coord.x + coord.w, coord.y),                 // 右上
                Point::new(coord.x + coord.w, coord.y + coord.h),       // 右下
                Point::new(coord.x, coord.y + coord.h),                 // 左下
            ];
            
            for corner in &corners {
                circle(
                    image,
                    *corner,
                    thickness * 2, // 角点半径
                    color,
                    -1,
                    8,
                    0,
                )?;
            }
        }
    }
    
    Ok(())
}

/// 渲染四边形到图像上
pub fn render_quad(
    image: &mut Mat,
    quad: &Quad,
    mode: Option<RenderMode>,
    color: Option<Scalar>,
    thickness: Option<i32>,
) -> Result<()> {
    let mode = mode.unwrap_or(RenderMode::Hollow);
    let color = color.unwrap_or(Colors::green());
    let thickness = thickness.unwrap_or(2);
    
    match mode {
        RenderMode::Filled => {
            // 创建点向量用于填充
            let points_vec = quad.points.iter()
                .map(|p| Point::new(p.x, p.y))
                .collect::<Vec<_>>();
            let points = Vector::<Point>::from_slice(&points_vec);
            let mut points_vector = Vector::<Vector<Point>>::new();
            points_vector.push(points);
            
            // 填充四边形
            fill_poly(
                image,
                &points_vector,
                color,
                8,
                0,
                Point::default(),
            )?;
        }
        RenderMode::Hollow => {
            // 绘制四条边
            for i in 0..4 {
                let start = &quad.points[i];
                let end = &quad.points[(i + 1) % 4];
                
                line(
                    image,
                    Point::new(start.x, start.y),
                    Point::new(end.x, end.y),
                    color,
                    thickness,
                    8,
                    0,
                )?;
            }
        }
        RenderMode::Corners => {
            // 只绘制四个角点
            for point in &quad.points {
                circle(
                    image,
                    Point::new(point.x, point.y),
                    thickness * 2,
                    color,
                    -1,
                    8,
                    0,
                )?;
            }
        }
    }
    
    Ok(())
}

/// 渲染多个坐标
pub fn render_coordinates(
    image: &mut Mat,
    coords: &[Coordinate],
    mode: Option<RenderMode>,
    color: Option<Scalar>,
    thickness: Option<i32>,
) -> Result<()> {
    for coord in coords {
        render_coordinate(image, coord, mode, color, thickness)?;
    }
    Ok(())
}

/// 渲染MobileOutput结构
pub fn render_output(
    image: &mut Mat,
    mobile_output: &MobileOutput,
    assist_location: &AssistLocation,
    mode: Option<RenderMode>,
    color: Option<Scalar>,
    thickness: Option<i32>,
    scale: Option<f64>, // 添加缩放参数
) -> Result<()> { 
    let mode = mode.unwrap_or(RenderMode::Hollow);
    let color = color.unwrap_or(Colors::red());
    let thickness = thickness.unwrap_or(2);
    let scale = scale.unwrap_or(1.0); // 默认不缩放

    // 如果需要缩放图像本身，则进行缩放
    if scale != 1.0 {
        let new_width = (image.cols() as f64 * scale) as i32;
        let new_height = (image.rows() as f64 * scale) as i32;
        let mut resized_image = Mat::default();
        
        opencv::imgproc::resize(
            image,
            &mut resized_image,
            opencv::core::Size::new(new_width, new_height),
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;
        
        // 将调整大小后的图像复制回原图像
        resized_image.copy_to(image)?;
    }

    // 遍历所有识别结果
    for rec_result in &mobile_output.rec_results {
        // 遍历所有填涂项和对应的结果
        for (index, fill_item) in rec_result.fill_items.iter().enumerate() {
            // 根据缩放调整坐标
            let scaled_coord = Coordinate {
                x: (fill_item.coordinate.x as f64 * scale) as i32,
                y: (fill_item.coordinate.y as f64 * scale) as i32,
                w: (fill_item.coordinate.w as f64 * scale) as i32,
                h: (fill_item.coordinate.h as f64 * scale) as i32,
            };
            
            // 只有在rec_result为true时才绘制矩形框
            if index < rec_result.rec_result.len() && rec_result.rec_result[index] {
                // 渲染选中选项的坐标框
                render_coordinate(image, &scaled_coord, Some(mode), Some(color), Some(thickness))?;
            }
            
            // 在选项上方渲染填涂率数字（保留两位小数）
            let text_x = (fill_item.coordinate.x as f64 * scale) as i32;
            let text_y = (fill_item.coordinate.y as f64 * scale - 5.0) as i32; // 在框上方一点的位置
            
            // 格式化填涂率，保留两位小数
            let fill_rate_text = format!("{:.2}", fill_item.fill_rate);
            
            // 使用OpenCV的put_text函数渲染文本
            opencv::imgproc::put_text(
                image,
                &fill_rate_text,
                Point::new(text_x, text_y),
                opencv::imgproc::FONT_HERSHEY_SIMPLEX,
                0.3,  // 根据缩放调整字体大小
                Colors::blue(),
                1,
                opencv::imgproc::LINE_8,
                false
            )?;
        }
    }

    for assist_location in assist_location.left.iter() {
        let scaled_coord = Coordinate {
            x: (assist_location.x as f64 * scale) as i32,
            y: (assist_location.y as f64 * scale) as i32,
            w: (assist_location.w as f64 * scale) as i32,
            h: (assist_location.h as f64 * scale) as i32,
        };
        render_coordinate(image, &scaled_coord, Some(mode), Some(color), Some(1))?;
    }
    for assist_location in assist_location.right.iter() {
        let scaled_coord = Coordinate {
            x: (assist_location.x as f64 * scale) as i32,
            y: (assist_location.y as f64 * scale) as i32,
            w: (assist_location.w as f64 * scale) as i32,
            h: (assist_location.h as f64 * scale) as i32,
        };
        render_coordinate(image, &scaled_coord, Some(mode), Some(color), Some(1))?;
    }

    Ok(())
}


/// 渲染辅助定位点
pub fn render_assist_location(
    image: &mut Mat,
    assist_location: &AssistLocation,
    mode: Option<RenderMode>,
    color: Option<Scalar>,
    thickness: Option<i32>,
) -> Result<()> {
    let mode = mode.unwrap_or(RenderMode::Hollow);
    let color = color.unwrap_or(Colors::red());
    let thickness = thickness.unwrap_or(1);

    for assist_location in assist_location.left.iter() {
        render_coordinate(image, &assist_location, Some(mode), Some(color), Some(thickness))?;
    }
    for assist_location in assist_location.right.iter() {
        render_coordinate(image, &assist_location, Some(mode), Some(color), Some(thickness))?;
    }
    Ok(())
}

/// 预设颜色
pub struct Colors;
impl Colors {
    pub fn red() -> Scalar { Scalar::new(0.0, 0.0, 255.0, 0.0) }
    pub fn green() -> Scalar { Scalar::new(0.0, 255.0, 0.0, 0.0) }
    pub fn blue() -> Scalar { Scalar::new(255.0, 0.0, 0.0, 0.0) }
    pub fn yellow() -> Scalar { Scalar::new(0.0, 255.0, 255.0, 0.0) }
    pub fn white() -> Scalar { Scalar::new(255.0, 255.0, 255.0, 0.0) }
    pub fn black() -> Scalar { Scalar::new(0.0, 0.0, 0.0, 0.0) }
    pub fn orange() -> Scalar { Scalar::new(0.0, 165.0, 255.0, 0.0) }
}