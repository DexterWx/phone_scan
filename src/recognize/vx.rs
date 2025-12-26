use crate::{config::{VxConfig, VxPageConfig}, models::{MobileOutput, ProcessedImage, RecResult, RecType, TopologyFeatures}, myutils::image::{crop_image, det_red_lab, extract_topology_features, preprocess_vx_line, zhang_suen_thinning}};
use anyhow::{Ok, Result};
use opencv::{core::Mat, prelude::MatTraitConst};

static mut COUNT: usize = 0;

pub struct RecVxModule;

impl RecVxModule {
    pub fn new() -> Self {
        Self
    }

    pub fn infer(&self, process_image: &ProcessedImage, mobile_output: &mut MobileOutput) -> Result<()> {
        for rec_result in mobile_output.rec_results.iter_mut() {
            if rec_result.rec_type != RecType::Vx {
                continue;
            }
            self.rec_options(process_image, rec_result)?;
            self.refine_options(rec_result)?;
        }
        self.set_vx(mobile_output)?;

        Ok(())
    }

    fn rec_options(&self, process_image: &ProcessedImage, options: &mut RecResult) -> Result<()> {
        
        for rec_option in options.rec_options.iter_mut() {
            let coor = &rec_option.coordinate;
            let sub_image = crop_image(&process_image.rgb, coor)?;
            let topology = self.extract_topology_from_rgb::<VxPageConfig>(&sub_image)?;
            
            let vx_res = self.rec_vx(&topology)?;
            rec_option.vx = vx_res;
            rec_option.topology = Some(topology);
        }
        Ok(())
    }

    fn refine_options(&self, options: &mut RecResult) -> Result<()> {
        // 第一步，把误判为单线的case改成非单线
        let _ = self.line_to_other(options);
        // 第二部，把误判为非单线的case改成单线
        let _ = self.other_to_line(options);
        Ok(())
    }

    fn line_to_other(&self, options: &mut RecResult) -> Result<()> {
        // 1. 如果只有一个选项，直接返回，不需要修改。
        if options.rec_options.len() <= 1 {
            return Ok(());
        }
        // 2. 如果有多个选项是单线
        // 对比每个option点数最多的连通域的曲率分数，最小的不变，其他的改为非单线。

        // 先计算出最小曲率分数
        let mut min_score = f64::MAX;
        for option in options.rec_options.iter_mut() {
            if !option.vx {
                continue;
            }
            if option.topology.is_none() {
                continue;
            }
            // 找到点数最多的连通域
            let connect = option.topology.as_ref().unwrap().connects.iter()
                .max_by_key(|c| c.points_count);
            if connect.is_none() {
                continue;   
            }
            let connect = connect.unwrap();
            let score = connect.curvature_score;
            if score < min_score {
                min_score = score;
            }
        }

        // 然后把分数大于最小分数的改为非单线
        for option in options.rec_options.iter_mut() {
            if !option.vx {
                continue;
            }
            if option.topology.is_none() {
                continue;
            }
            // 找到点数最多的连通域
            let connect = option.topology.as_ref().unwrap().connects.iter()
                .max_by_key(|c| c.points_count);
            if connect.is_none() {
                continue;   
            }
            let connect = connect.unwrap();
            let score = connect.curvature_score;
            if score > min_score {
                option.vx = false;
            }
        }


        Ok(())
    }

    fn other_to_line(&self, options: &mut RecResult) -> Result<()> {
        // 1. 如果只有一个选项，直接返回，不需要修改。
        if options.rec_options.len() <= 1 {
            return Ok(());
        }
        // 2. 如果没有单线
        // 有且只有一个option的点数量超过20
        // 将这个唯一的case改为单线
        let mut candidate_index= 0;
        let mut more20count = 0;
        for (index, option) in options.rec_options.iter_mut().enumerate() {
            if option.vx {
                continue;
            }
            if option.topology.is_none() {
                continue;
            }
            let topology = option.topology.as_ref().unwrap();
            let sum_points: usize = topology.connects.iter().map(|c| c.points_count).sum();
            if sum_points >= 20 {
                more20count += 1;
                candidate_index = index;
            }
        }
        if more20count == 1 {
            options.rec_options[candidate_index].vx = true;
        }

        Ok(())
    }

    fn rec_vx(&self, topology: &TopologyFeatures) -> Result<bool> {
        let is_single = self.is_single_line_from_features::<VxPageConfig>(topology);
        Ok(is_single)
    }

    fn extract_topology_from_rgb<T: VxConfig>(&self, rgb: &Mat) -> Result<TopologyFeatures> {
        let red = &det_red_lab(rgb)?;
        // 步骤 1：快速预筛选
        if !self.quick_filter::<T>(red)? {
            return Ok(TopologyFeatures::default());
        }

        // 步骤 2：预处理
        let preprocessed = preprocess_vx_line::<T>(red)?;

        // 步骤 3：骨架化
        let skeleton = zhang_suen_thinning(&preprocessed)?;

        // 步骤 4：提取拓扑特征
        let features = extract_topology_features(&skeleton)?;
        #[cfg(debug_assertions)] {
            unsafe {
                // if branch_count == 0 && end_count <= 2 {
                if true {
                    opencv::imgcodecs::imwrite(
                        format!("dev/test_data/debug/sk_{:?}_rgb.jpg",
                            COUNT
                        ).as_str(),
                        rgb, &opencv::core::Vector::new()
                    )?;
                    opencv::imgcodecs::imwrite(
                        format!("dev/test_data/debug/sk_{:?}_red.jpg",
                            COUNT
                        ).as_str(),
                        &red, &opencv::core::Vector::new()
                    )?;

                    let connect_count = features.connects.len();
                    let branch_count = features.connects.iter().filter(|c| c.has_branch).count();
                    let end_count = features.connects.iter().map(|c| c.end_points).sum::<usize>();
                    let connect = features.connects.iter().max_by_key(|c| c.points_count);
                    let score = connect.map(|c| c.curvature_score).unwrap_or(-1.0);
                    let sum_points = features.connects.iter().map(|c| c.points_count).sum::<usize>();
                    opencv::imgcodecs::imwrite(
                        format!(
                            "dev/test_data/debug/sk_{:?}_info_{connect_count:?}_{branch_count:?}_{end_count:?}_{score:?}_{sum_points:?}.jpg",
                            COUNT,
                        ).as_str(),
                        &skeleton, &opencv::core::Vector::new()
                    )?;
                    COUNT+=1;
                }
            }

        }
        
        Ok(features)
    }

    /// 快速预筛选（基于填涂率）
    /// 输入图像：红色线条为白色（255），背景为黑色（0）
    fn quick_filter<T: VxConfig>(&self, binary_image: &Mat) -> Result<bool> {
        let total_pixels = (binary_image.rows() * binary_image.cols()) as f64;
        // count_non_zero 统计非零像素（白色=255），即红色线条
        let red_pixels = opencv::core::count_non_zero(binary_image)? as f64;
        let fill_ratio = red_pixels / total_pixels;

        // 填涂率过低 → 无效/空白
        if fill_ratio < T::fill_ratio_min() {
            return Ok(false);
        }

        // 填涂率过高 → 乱划/涂抹
        if fill_ratio > T::fill_ratio_max() {
            return Ok(false);
        }

        // 需要继续分析
        Ok(true)
    }

    /// 基于拓扑特征判断是否为单线
    fn is_single_line_from_features<T: VxConfig>(&self, features: &TopologyFeatures) -> bool {
        if features.connects.len() == 0 {
            return false;
        }
        // 端点数检查
        let total_end_points = features.connects.iter().map(|c| c.end_points).sum::<usize>();
        if total_end_points > T::max_end_points() {
            return false;
        }
        for connect in features.connects.iter() {
            // 有分支点
            if connect.has_branch {
                return false;
            }
        }
        // 通过所有规则，判定为单线
        true
    }

    fn set_vx(&self, mobile_output: &mut MobileOutput) -> Result<()> {
        for rec_result in mobile_output.rec_results.iter_mut() {
            if rec_result.rec_type != RecType::Vx {
                continue;
            }
            for (index, rec_option) in rec_result.rec_options.iter_mut().enumerate() {
                if rec_option.vx {
                    rec_result.rec_result[index] = true;
                } else {
                    rec_result.rec_result[index] = false;
                }
            }
        }
        Ok(())
    }
}
