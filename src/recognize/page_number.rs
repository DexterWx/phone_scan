use anyhow::Result;
use opencv::core::Mat;

use crate::{config::CommonConfig, models::{Coordinate, ProcessedImage}, myutils::image::integral_image, recognize::fill::{self, calculate_fill_rate}};

pub struct PageNumberModule;

impl PageNumberModule {
    pub fn new() -> Self {
        Self
    }

    pub fn infer(&self, image: &ProcessedImage, coors: &Vec<Coordinate>) -> Result<usize> {
        let integral_image = integral_image(&image.thresh)?;
        let refine_coors = self.refine_page_number_coor(&integral_image, coors)?;
        #[cfg(debug_assertions)]
        {
            use crate::myutils::rendering::{RenderMode, render_coordinates};

            let mut render_image = image.rgb.clone();
            let _ = render_coordinates(&mut render_image, &refine_coors, Some(RenderMode::Hollow), None, None);
            let debug_path = format!("dev/test_data/debug/z_page_number.jpg");
            opencv::imgcodecs::imwrite(&debug_path, &render_image, &opencv::core::Vector::new())?;
        }
        let mut binary_str = String::new();
        for (index,coor) in refine_coors.iter().enumerate() {
            let fill_rate = calculate_fill_rate(&integral_image, coor)?;
            if index == 0 {continue}
            if fill_rate >= CommonConfig::PAGE_NUMBER_FILL_RATE {
                binary_str.push_str("1");
            } else {
                binary_str.push_str("0");
            }
        }

        match u32::from_str_radix(&binary_str, 2) {
            Ok(decimal) => {
                if decimal == 0 {
                    anyhow::bail!("页码点异常");
                }
                return Ok(decimal as usize);
            },
            Err(e) => {
                anyhow::bail!("页码转换失败: {}", e);
            }
        }
    }

    fn refine_page_number_coor(&self, integral_image: &Mat, coors: &Vec<Coordinate>) -> Result<Vec<Coordinate>> {
        let mut refined_coors = coors.clone();
        let mut max_var = 0.0;
        for move_y in -CommonConfig::PAGE_NUMBER_EXTEND_SIZE..CommonConfig::PAGE_NUMBER_EXTEND_SIZE {
            for move_x in -CommonConfig::PAGE_NUMBER_EXTEND_SIZE..CommonConfig::PAGE_NUMBER_EXTEND_SIZE {
                let mut fill_rates = Vec::new();
                let mut tmp_coors = Vec::new();
                for coor in coors.iter() {
                    let new_coor = Coordinate {
                        x: coor.x + move_x,
                        y: coor.y + move_y,
                        w: coor.w,
                        h: coor.h,
                    };
                    let fill_rate = fill::calculate_fill_rate(integral_image, &new_coor)?;
                    fill_rates.push(fill_rate);
                    tmp_coors.push(new_coor);
                }
                if fill_rates[0] < CommonConfig::PAGE_NUMBER_FILL_RATE {
                    continue;
                }
                let (_, variance) = crate::myutils::math::otsu_threshold(&fill_rates);
                if variance > max_var {
                    refined_coors = tmp_coors;
                    max_var = variance;
                }
            }
        }
        Ok(refined_coors)
    }
}

    