use crate::models::Coordinate;
use crate::models::AssistLocation;
use crate::models::ProcessedImage;
use crate::myutils::image::merge_coordinates;
use crate::config::AssistLocationConfig;
use anyhow::Result;
use opencv::core::Mat;
use opencv::core::MatTraitConst;
use opencv::{
    core::{Rect, Vector},
    imgproc::{contour_area, find_contours, bounding_rect, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE},
};

pub struct AssistLocationModule;

impl AssistLocationModule {
    pub fn new() -> Self {
        Self
    }

    pub fn infer(&self, processed_image: &ProcessedImage, assist_location: &AssistLocation) -> Result<AssistLocation> {
        let left_area = merge_coordinates(&assist_location.left, AssistLocationConfig::ASSIST_AREA_EXTEND_SIZE);
        let right_area = merge_coordinates(&assist_location.right, AssistLocationConfig::ASSIST_AREA_EXTEND_SIZE);
        let left_src_assist = Self::find_assist_location(&processed_image.closed, &left_area)?;
        let right_src_assist = Self::find_assist_location(&processed_image.closed, &right_area)?;
        
        if left_src_assist.len() != right_src_assist.len() {
            anyhow::bail!("辅助定位点数量不匹配，左侧找到{}个，右侧找到{}个", left_src_assist.len(), right_src_assist.len());
        }

        if left_src_assist.len() != assist_location.left.len() {
            anyhow::bail!("辅助定位点数量异常");
        }

        Ok(
            AssistLocation {
                left: left_src_assist,
                right: right_src_assist 
            }
        )
    }

    // pub fn align_assist_location(&self, processed_image: &ProcessedImage, coordinates: &Vec<Coordinate>) -> Result<AssistLocation> {
        
    //     todo!()
    // }

    /// 在闭图上寻找辅助定位点
    pub fn find_assist_location(closed: &Mat, coordinate: &Coordinate) -> Result<Vec<Coordinate>> {
        // 创建感兴趣区域ROI
        let roi_rect = Rect::new(
            coordinate.x.max(0),
            coordinate.y.max(0),
            coordinate.w.min(closed.cols() - coordinate.x.max(0)),
            coordinate.h.min(closed.rows() - coordinate.y.max(0))
        );
        
        // 提取ROI区域
        let roi = Mat::roi(closed, roi_rect)?;
        
        // 查找轮廓
        let mut contours = Vector::<Vector<opencv::core::Point2i>>::new();
        find_contours(
            &roi,
            &mut contours,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
            opencv::core::Point2i::new(0, 0),
        )?;
        
        let mut assist_points = Vec::new();
        let integral_image = crate::myutils::image::integral_image(&roi.clone_pointee())?;
        // 遍历所有轮廓
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = contour_area(&contour, false)?;
            
            // 计算轮廓的边界矩形
            let bounding_rect = bounding_rect(&contour)?;
            
            // 检查区域是否接近6*6的正方形
            // 允许一定误差，比如5-7像素范围内
            let width = bounding_rect.width;
            let height = bounding_rect.height;
            
            if width < AssistLocationConfig::ASSIST_POINT_MIN_SIZE {continue;}
            if width > AssistLocationConfig::ASSIST_POINT_MAX_SIZE {continue;}
            if height < AssistLocationConfig::ASSIST_POINT_MIN_SIZE {continue;}
            if height > AssistLocationConfig::ASSIST_POINT_MAX_SIZE {continue;}
            if (width - height).abs() > AssistLocationConfig::ASSIST_POINT_WHDIFF_MAX {continue;}
            if area < AssistLocationConfig::ASSIST_POINT_MIN_AREA {continue;}
            if area > AssistLocationConfig::ASSIST_POINT_MAX_AREA {continue;}
            let fill_rate = crate::recognize::fill::calculate_fill_rate(
                &integral_image,
                &Coordinate {
                    x: bounding_rect.x+1,
                    y: bounding_rect.y+1,
                    w: bounding_rect.width-2,
                    h: bounding_rect.height-2,
                }
            )?;
            if fill_rate < AssistLocationConfig::ASSIST_POINT_MIN_FILL_RATIO {continue;}

            assist_points.push(Coordinate {
                x: bounding_rect.x + coordinate.x,
                y: bounding_rect.y + coordinate.y,
                w: bounding_rect.width,
                h: bounding_rect.height,
            });

        }

        assist_points.sort_by(|a, b| a.y.cmp(&b.y));
        
        Ok(assist_points)
    }

}