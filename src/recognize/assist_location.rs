use crate::config::AssistLocationConfig;
use crate::config::AssistLocationPageConfig;
use crate::models::Coordinate;
use crate::models::AssistLocation;
use crate::models::ProcessedImage;
use crate::myutils::image::merge_coordinates;
use anyhow::Ok;
use anyhow::Result;
use opencv::core::Mat;
use opencv::core::MatTraitConst;
use opencv::{
    core::{Rect, Vector},
    imgproc::{find_contours, bounding_rect, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE},
};

pub struct AssistLocationModule;

impl AssistLocationModule {
    pub fn new() -> Self {
        Self
    }

    pub fn infer_single<T: AssistLocationConfig>(
        &self, processed_image: &ProcessedImage,
        assist_location: &AssistLocation
    ) -> Result<AssistLocation> {
        let left_area = merge_coordinates(&assist_location.left, T::assist_area_extend_size_w(), T::assist_area_extend_size_h());
        let right_area = merge_coordinates(&assist_location.right, T::assist_area_extend_size_w(), T::assist_area_extend_size_h());
        let left_src_assist = Self::find_assist_location::<T>(&processed_image.closed, &left_area)?;
        let right_src_assist = Self::find_assist_location::<T>(&processed_image.closed, &right_area)?;

        // let mut rgb = processed_image.rgb.clone();
        // render_coordinates(&mut rgb, &left_src_assist, Some(crate::myutils::rendering::RenderMode::Hollow), None, None);
        // render_coordinates(&mut rgb, &right_src_assist, Some(crate::myutils::rendering::RenderMode::Hollow), None, None);
        // opencv::imgcodecs::imwrite("dev/test_data/debug/assist_location_found.jpg", &rgb, &Vector::<i32>::new()).unwrap();
        if left_src_assist.len() != right_src_assist.len() {
            anyhow::bail!("辅助定位点数量不匹配，左侧找到{}个，右侧找到{}个", left_src_assist.len(), right_src_assist.len());
        }

        if left_src_assist.len() != assist_location.left.len() {
            anyhow::bail!("辅助定位点数量异常: {:?}_{}", left_src_assist.len(), assist_location.left.len(),);
        }

        Ok(
            AssistLocation {
                left: left_src_assist,
                right: right_src_assist 
            }
        )
    }

    pub fn infer_paper(&self, processed_image: &ProcessedImage, assist_location: &AssistLocation) -> Result<AssistLocation> {
        let mut assist_locations = Vec::new();
        let split_locations = assist_location.split();
        for single_location in split_locations {
            let real_single_location = self.infer_single::<AssistLocationPageConfig>(processed_image, &single_location)?;
            assist_locations.push(real_single_location);
        }
        let res = AssistLocation::merge(&assist_locations);
        Ok(res)
    }

    /// 在闭图上寻找辅助定位点
    pub fn find_assist_location<T: AssistLocationConfig>(closed: &Mat, coordinate: &Coordinate) -> Result<Vec<Coordinate>> {
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
            // let area = contour_area(&contour, false)?;
            
            // 计算轮廓的边界矩形
            let bounding_rect = bounding_rect(&contour)?;
            // 检查区域是否接近6*6的正方形
            // 允许一定误差，比如5-7像素范围内
            let width = bounding_rect.width;
            let height = bounding_rect.height;
            let area = (width * height) as f64;
            
            if width < T::assist_point_min_size() {continue;}
            if width > T::assist_point_max_size() {continue;}
            if height < T::assist_point_min_size() {continue;}
            if height > T::assist_point_max_size() {continue;}
            if (width - height).abs() > T::assist_point_whdiff_max() {continue;}
            if area < T::assist_point_min_area() {continue;}
            if area > T::assist_point_max_area() {continue;}
            let fill_rate = crate::recognize::fill::calculate_fill_rate(
                &integral_image,
                &Coordinate {
                    x: bounding_rect.x+1,
                    y: bounding_rect.y+1,
                    w: bounding_rect.width-2,
                    h: bounding_rect.height-2,
                }
            )?;
            if fill_rate < T::assist_point_min_fill_ratio() {
                continue;
            }
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