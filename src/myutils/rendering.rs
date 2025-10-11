use anyhow::Result;
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size},
    imgproc::{rectangle, circle, line},
    prelude::*,
};
use crate::models::Coordinate;

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
