use anyhow::Result;
use opencv::{
    core::{Mat, Point, Rect, Scalar, Vector},
    imgproc::{circle, fill_poly, line, rectangle},
    prelude::*,
};
use crate::models::{Coordinate, Quad};

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

/// 预设颜色
pub struct Colors;
impl Colors {
    pub fn red() -> Scalar { Scalar::new(0.0, 0.0, 255.0, 0.0) }
    pub fn green() -> Scalar { Scalar::new(0.0, 255.0, 0.0, 0.0) }
    pub fn blue() -> Scalar { Scalar::new(255.0, 0.0, 0.0, 0.0) }
    pub fn yellow() -> Scalar { Scalar::new(0.0, 255.0, 255.0, 0.0) }
    pub fn white() -> Scalar { Scalar::new(255.0, 255.0, 255.0, 0.0) }
    pub fn black() -> Scalar { Scalar::new(0.0, 0.0, 0.0, 0.0) }
}