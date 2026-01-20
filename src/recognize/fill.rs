use anyhow::{Ok, Result};
use opencv::core::{Mat, MatTraitConst};
use crate::config::FillConfig;
use crate::models::{Coordinate, MobileOutput, ProcessedImage, RecType};
use crate::models::RecOption;
use crate::myutils::image::sum_pixel;

pub struct RecFillModule;

impl RecFillModule {

    pub fn new() -> Self {
        Self
    }

    pub fn infer<T: FillConfig>(&self, process_image: &ProcessedImage, mobile_output: &mut MobileOutput, ) -> Result<()> {
        // 1. 计算积分图
        let integral_image = crate::myutils::image::integral_image(&process_image.thresh)?;

        // 2. 计算所有选项的填涂率和otsu值
        self.refine_all_fill_coordinate::<T>(&integral_image, mobile_output)?;
        self.calculate_all_fill_rate(&integral_image, mobile_output)?;
        let fill_rates = mobile_output.rec_results.iter()
            .filter(|rec_result| [RecType::SingleChoice, RecType::MultipleChoice].contains(&rec_result.rec_type))
            .flat_map(|rec_result| rec_result.rec_options.iter().map(|item| item.fill_rate))
            .collect::<Vec<f64>>();
        let (mut thresh, _) = crate::myutils::math::otsu_threshold(&fill_rates);
        thresh = thresh.max(T::fill_rate_min());
        thresh = (thresh * 100.0).ceil() / 100.0;
        
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
            if rec_result.rec_type != RecType::MultipleChoice {
                continue;
            }
            let fill_items = &mut rec_result.rec_options;
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
            if ![RecType::SingleChoice, RecType::MultipleChoice].contains(&rec_result.rec_type) {
                continue;
            }
            let fill_items = &mut rec_result.rec_options;
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
            if rec_result.rec_type != RecType::SingleChoice {
                continue;
            }
            
            // 找到填涂率最高的选项
            let mut max_fill_rate = 0.0;
            let mut max_index = None;
            
            for (index, fill_item) in rec_result.rec_options.iter().enumerate() {
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
            if ![RecType::SingleChoice, RecType::MultipleChoice].contains(&rec_result.rec_type) {
                continue;
            }
            let fill_items = &mut rec_result.rec_options;
            for fill_item in fill_items.iter_mut() {
                let fill_rate = calculate_fill_rate(integral_image, &mut fill_item.coordinate)?;
                fill_item.fill_rate = fill_rate;
            }
        }

        Ok(())
    }

    pub fn refine_all_fill_coordinate<T: FillConfig>(&self, integral_image: &Mat, mobile_output: &mut MobileOutput) -> Result<()> {
        for rec_result in mobile_output.rec_results.iter_mut() {
            if ![RecType::SingleChoice, RecType::MultipleChoice].contains(&rec_result.rec_type) {
                continue;
            }
            let res = self.refine_items_fill_coordinate::<T>(integral_image, &mut rec_result.rec_options);
            if res.is_err() {
                continue;
            }
        }

        Ok(())
    }

    /// 通过Otsu最大类间方差优化坐标位置
    /// 在以当前坐标为中心的4x4范围内(-2到2)寻找使所有选项填涂率方差最大的位置
    fn refine_items_fill_coordinate<T: FillConfig>(&self, integral_image: &Mat, fill_items: &mut Vec<RecOption>) -> Result<()> {
        if fill_items.is_empty() {
            return Ok(());
        }

        let mut max_variance = 0.0;
        let mut best_coordinates: Vec<Coordinate> = Vec::new();

        // 在-2到2的范围内搜索最优坐标偏移
        'outer: for dx in -T::refine_coor_range() ..= T::refine_coor_range() {
            for dy in -T::refine_coor_range() ..= T::refine_coor_range(){
                let mut fill_rates = Vec::new();
                let mut temp_coordinates = Vec::new();
                
                // 计算所有选项在这个偏移下的填涂率
                for fill_item in fill_items.iter() {
                    let new_coordinate = Coordinate {
                        x: fill_item.coordinate.x + dx,
                        y: fill_item.coordinate.y + dy,
                        w: fill_item.coordinate.w,
                        h: fill_item.coordinate.h,
                    };
                    
                    // 计算填涂率并处理可能的错误
                    let fill_rate_result = calculate_fill_rate(integral_image, &new_coordinate)?;
                    fill_rates.push(fill_rate_result);
                    temp_coordinates.push(new_coordinate);
                }
                // 如果所有fill_rate都大于0.8，结束搜索
                if fill_rates.iter().all(|&rate| rate > 0.8) {
                    best_coordinates = temp_coordinates;
                    max_variance = f64::MAX;
                    break 'outer;
                }
                
                let (_, variance) = crate::myutils::math::otsu_threshold(&fill_rates);
                // 更新最优坐标（如果方差更大）
                if variance > max_variance {
                    max_variance = variance;
                    best_coordinates = temp_coordinates;
                }
            }
        }

        // 如果找到了更好的坐标，则更新坐标
        if max_variance > 0.0 {
            for (i, fill_item) in fill_items.iter_mut().enumerate() {
                fill_item.coordinate = best_coordinates[i].clone();
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
    
    let sum = sum_pixel(integral_image, coordinate)?;
    
    // 计算区域面积
    let area = coordinate.w as f64 * coordinate.h as f64;
    
    // 计算白色像素占比（填涂率）
    // 由于二值图中白色为255，黑色为0，所以需要将和除以255得到白色像素数量
    let white_pixels = sum / 255.0;
    let fill_rate = white_pixels / area;
    
    Ok(fill_rate)
}