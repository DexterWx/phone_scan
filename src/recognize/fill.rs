use anyhow::{Ok, Result};
use opencv::core::{Mat, MatTraitConst};
use crate::models::{Coordinate, Mark, ProcessedImage};

pub struct RecFillModule;

impl RecFillModule {

    pub fn new() -> Self {
        Self
    }

    pub fn infer(&self, process_image: &ProcessedImage, mark: &Mark) {
        
    }

    pub fn all_options_fill_rate_otsu(&self, image: &Mat, mark: &Mark) -> Result<f64> {
        let coordinates: Vec<&crate::models::Coordinate> = mark.rec_items
            .iter()
            .flat_map(|rec_item| &rec_item.sub_options)
            .collect();
        
        let fill_rates = coordinates.iter().map(|&coord| {
            self.calculate_fill_rate(image, coord)
        }).collect::<Result<Vec<f64>>>()?;


        Ok(0.0)
    }

    /// 计算指定区域的填涂率（白色像素占比）
    /// 使用积分图加速计算，积分图可能比原图大1个像素
    pub fn calculate_fill_rate(&self, integral_image: &Mat, coordinate: &Coordinate) -> Result<f64> {
        // 获取积分图尺寸
        let integral_rows = integral_image.rows();
        let integral_cols = integral_image.cols();
        
        // 检查坐标是否有效
        if coordinate.x < 0 || coordinate.y < 0 || 
           coordinate.x + coordinate.w > integral_cols - 1 || 
           coordinate.y + coordinate.h > integral_rows - 1 {
            anyhow::bail!("坐标超出积分图范围");
        }
        
        // 由于积分图比原图大1像素，需要调整坐标
        // 积分图的(0,0)对应原图的(-1,-1)位置
        let x1 = coordinate.x as i32 + 1; // 左上角x坐标
        let y1 = coordinate.y as i32 + 1; // 左上角y坐标
        let x2 = x1 + coordinate.w as i32 - 1; // 右下角x坐标
        let y2 = y1 + coordinate.h as i32 - 1; // 右下角y坐标
        
        // 从积分图获取四个角的值
        let a = integral_image.at_2d::<f64>(y1 - 1, x1 - 1)?; // 左上角上方
        let b = integral_image.at_2d::<f64>(y1 - 1, x2)?;     // 右上角上方
        let c = integral_image.at_2d::<f64>(y2, x1 - 1)?;     // 左下角左侧
        let d = integral_image.at_2d::<f64>(y2, x2)?;         // 右下角
        
        // 使用积分图计算区域和
        let sum = d - b - c + a;
        
        // 计算区域面积
        let area = coordinate.w as f64 * coordinate.h as f64;
        
        // 计算白色像素占比（填涂率）
        // 由于二值图中白色为255，黑色为0，所以需要将和除以255得到白色像素数量
        let white_pixels = sum / 255.0;
        let fill_rate = white_pixels / area;
        
        Ok(fill_rate)
    }
}