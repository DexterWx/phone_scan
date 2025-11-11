use anyhow::{Ok, Result};
use opencv::core::{Mat, MatTraitConst};
use crate::config::FillConfig;
use crate::models::{Coordinate, MobileOutput, ProcessedImage, RecType};
use crate::models::FillItem;

pub struct RecFillModule;

impl RecFillModule {

    pub fn new() -> Self {
        Self
    }

    pub fn infer(&self, process_image: &ProcessedImage, mobile_output: &mut MobileOutput) -> Result<()> {
        // 1. 计算积分图
        let integral_image = crate::myutils::image::integral_image(&process_image.thresh)?;

        // 2. 计算所有选项的填涂率和otsu值
        self.calculate_all_fill_rate(&integral_image, mobile_output)?;
        let fill_rates = mobile_output.rec_results.iter()
            .flat_map(|rec_result| rec_result.fill_items.iter().map(|item| item.fill_rate))
            .collect::<Vec<f64>>();
        let (mut thresh, _) = crate::myutils::math::otsu_threshold(&fill_rates);
        thresh = thresh.max(FillConfig::FILL_RATE_MIN);
        thresh = (thresh * 100.0).round() / 100.0;

        #[cfg(debug_assertions)]
        {
            println!("填涂率阈值: {:.4}", thresh);
        }

        // // 3. 单选识别
        // self.set_single_fill(mobile_output, thresh)?;
        // // 4. 多选识别
        // self.set_multi_fill(mobile_output, thresh)?;
        self.set_default_fill(mobile_output, thresh)?;
        

        Ok(())
        
    }

    pub fn set_multi_fill(&self, mobile_output: &mut MobileOutput, thresh: f64) -> Result<()> {
        for rec_result in mobile_output.rec_results.iter_mut() {
            if rec_result.rec_tpye != RecType::MultipleChoice {
                continue;
            }
            let fill_items = &mut rec_result.fill_items;
            for (index,fill_item) in fill_items.iter_mut().enumerate() {
                if fill_item.fill_rate > thresh {
                    rec_result.rec_result[index] = true;
                } else {
                    rec_result.rec_result[index] = false;
                }
            }
        }
        
        Ok(())
    }

    pub fn set_default_fill(&self, mobile_output: &mut MobileOutput, thresh: f64) -> Result<()> {
        for rec_result in mobile_output.rec_results.iter_mut() {
            if rec_result.rec_tpye != RecType::MultipleChoice && rec_result.rec_tpye != RecType::SingleChoice{
                continue;
            }
            let fill_items = &mut rec_result.fill_items;
            for (index,fill_item) in fill_items.iter_mut().enumerate() {
                if fill_item.fill_rate > thresh {
                    rec_result.rec_result[index] = true;
                } else {
                    rec_result.rec_result[index] = false;
                }
            }
        }
        
        Ok(())
    }

    pub fn set_single_fill(&self, mobile_output: &mut MobileOutput, thresh: f64) -> Result<()> {
        for rec_result in mobile_output.rec_results.iter_mut() {
            if rec_result.rec_tpye != RecType::SingleChoice {
                continue;
            }
            
            // 找到填涂率最高的选项
            let mut max_fill_rate = 0.0;
            let mut max_index = None;
            
            for (index, fill_item) in rec_result.fill_items.iter().enumerate() {
                if fill_item.fill_rate > max_fill_rate {
                    max_fill_rate = fill_item.fill_rate;
                    max_index = Some(index);
                }
            }
            
            // 如果找到了最大填涂率且大于阈值，则标记为选中
            if let Some(index) = max_index {
                if max_fill_rate > thresh {
                    rec_result.rec_result[index] = true;
                }
            }
        }
        
        Ok(())
    }

    pub fn calculate_all_fill_rate(&self, integral_image: &Mat, mobile_output: &mut MobileOutput) -> Result<()> {
        for rec_result in mobile_output.rec_results.iter_mut() {
            let fill_items = &mut rec_result.fill_items;
            for fill_item in fill_items.iter_mut() {
                let fill_rate = calculate_fill_rate(integral_image, &mut fill_item.coordinate)?;
                fill_item.fill_rate = fill_rate;
            }
        }

        Ok(())
    }

    fn _calculate_max_fill_rate(&self, integral_image: &Mat, coordinate: &mut Coordinate) -> Result<f64> {
        let mut max_fill_rate = 0.0;
        for move_i in -1..=1 {
            for move_j in 0..=2 {
                let new_coordinate = Coordinate {
                    x: coordinate.x + move_i,
                    y: coordinate.y + move_j,
                    w: coordinate.w,
                    h: coordinate.h,
                };
                let fill_rate = calculate_fill_rate(integral_image, &new_coordinate)?;
                if fill_rate > max_fill_rate {
                    max_fill_rate = fill_rate;
                    coordinate.x = new_coordinate.x;
                    coordinate.y = new_coordinate.y;
                }
            }
        }
        Ok(max_fill_rate)
    }

    fn _fix_fill_coordinates_with_otsu(&self, integral_image: &Mat, fill_items: &mut Vec<FillItem>) -> Result<()> {
        /// 以当前coordinate为中心的3*3范围内
        /// 在这9中情况下，分别计算所有coordinate的填涂率
        /// 每一种情况都用这些填涂率计算一个最大类间方差
        /// 找到方差最大的一种情况，将当前coordinate的坐标修正为这种情况的坐标
        /// 并且修改所有coordinate的坐标
        
        for move_i in -1..=1 {
            for move_j in -1..=1 {
                let mut fill_rates = Vec::new();
                let mut max_variance = 0.0;
                let mut new_coors = Vec::new();
                for fill_item in fill_items.iter_mut() {
                    let new_coordinate = Coordinate {
                        x: fill_item.coordinate.x + move_i,
                        y: fill_item.coordinate.y + move_j,
                        w: fill_item.coordinate.w,
                        h: fill_item.coordinate.h,
                    };
                    let fill_rate = calculate_fill_rate(integral_image, &new_coordinate)?;
                    fill_rates.push(fill_rate);
                    new_coors.push(new_coordinate);
                }
                let (_thresh, variance) = crate::myutils::math::otsu_threshold(&fill_rates);
                if variance > max_variance {
                    max_variance = variance;
                    for (index, fill_item) in fill_items.iter_mut().enumerate() {
                        fill_item.coordinate = new_coors[index].clone();
                    }
                }
            }
        }
        Ok(())
    }
}


/// 计算指定区域的填涂率（白色像素占比）
pub fn calculate_fill_rate(integral_image: &Mat, coordinate: &Coordinate) -> Result<f64> {
    // 获取积分图尺寸
    let integral_rows = integral_image.rows();
    let integral_cols = integral_image.cols();
    
    // 检查坐标是否有效
    if coordinate.x < 0 || coordinate.y < 0 || 
        coordinate.x + coordinate.w > integral_cols - 1 || 
        coordinate.y + coordinate.h > integral_rows - 1 {
        anyhow::bail!("坐标超出积分图范围");
    }
    
    let x1 = coordinate.x as i32; // 左上角x坐标
    let y1 = coordinate.y as i32; // 左上角y坐标
    let x2 = x1 + coordinate.w as i32; // 右下角x坐标
    let y2 = y1 + coordinate.h as i32; // 右下角y坐标
    
    // 从积分图获取四个角的值
    // 积分图通常是i32类型，需要解引用后再转换为f64进行计算
    let a = *integral_image.at_2d::<i32>(y1, x1)? as f64; // 左上角上方
    let b = *integral_image.at_2d::<i32>(y1, x2)? as f64;     // 右上角上方
    let c = *integral_image.at_2d::<i32>(y2, x1)? as f64;     // 左下角左侧
    let d = *integral_image.at_2d::<i32>(y2, x2)? as f64;         // 右下角
    
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