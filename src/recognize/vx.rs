use crate::{config::{VxConfig, VxPageConfig}, models::{MobileOutput, ProcessedImage, RecResult, RecType, TopologyFeatures}, myutils::image::{crop_image, det_red, extract_topology_features, preprocess_vx_line, refine_skeleton, zhang_suen_thinning}};
use anyhow::{Ok, Result};
use opencv::{core::Mat, prelude::MatTraitConst};

// static mut COUNT: usize = 0;

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
            // self.refine_options(rec_result)?;
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
            rec_option.topology = Some(topology)
        }
        Ok(())
    }

    fn refine_options(&self, options: &mut RecResult) -> Result<()> {
        // 1. 如果所有的vx都是false
        if options.rec_options.iter().all(|x| !x.vx) {
            for rec_option in options.rec_options.iter_mut() {
                if rec_option.topology.is_none() {
                    continue
                }
                let topology = rec_option.topology.as_ref().unwrap();
                if topology.end_points == 4 && topology.branch_points == 0 {
                    rec_option.vx = true;
                }
            }
        }
        
        Ok(())
    }

    fn rec_vx(&self, topology: &TopologyFeatures) -> Result<bool> {
        let is_single = self.is_single_line_from_features::<VxPageConfig>(topology);
        Ok(is_single)
    }

    fn extract_topology_from_rgb<T: VxConfig>(&self, rgb: &Mat) -> Result<TopologyFeatures> {
        let image = &det_red(rgb)?;
        // unsafe {
        //     opencv::imgcodecs::imwrite(format!("dev/test_data/debug/sk_{:?}_red.jpg", COUNT).as_str(), &image, &opencv::core::Vector::new())?;
        // }
        // 步骤 1：快速预筛选
        if !self.quick_filter::<T>(image)? {
            return Ok(TopologyFeatures::default());
        }

        // 步骤 2：预处理
        let preprocessed = preprocess_vx_line::<T>(image)?;

        // 步骤 3：骨架化
        let skeleton = zhang_suen_thinning(&preprocessed)?;

        // 步骤 3.5：骨架精简（新增）
        let skeleton = refine_skeleton(&skeleton)?;

        // 步骤 4：提取拓扑特征
        let features = extract_topology_features(&skeleton)?;

        // unsafe {
        //     opencv::imgcodecs::imwrite(
        //         format!(
        //             "dev/test_data/debug/sk_{:?}_{:?}_{:?}.jpg",
        //             COUNT,features.branch_points,features.end_points
        //         ).as_str(),
        //         &skeleton, &opencv::core::Vector::new()
        //     )?;
        //     COUNT+=1;
        // }

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
        // 规则 1：没有分支点（排除叉和多线）
        if features.branch_points > T::max_branch_points() {
            return false;
        }

        // 规则 2：端点数在合理范围内
        if features.end_points < T::min_end_points() {
            return false;
        }

        if features.end_points > T::max_end_points() {
            return false;
        }

        // 规则 3：如果骨架太少，可能是噪点
        if features.total_pixels < 5 {
            return false;
        }

        // 规则 4：如果像素点过多，可能V笔过多
        if features.total_pixels > features.image_width {
            return false;
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
