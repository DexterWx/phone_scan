use crate::config::AssistLocationConfig;
use crate::config::AssistLocationPageConfig;
use crate::models::Coordinate;
use crate::models::AssistLocation;
use crate::models::ProcessedImage;
use crate::myutils::image::merge_coordinates;
use crate::myutils::rendering::render_coordinates;
use crate::recognize::align::find_extra_indices_cos;
use crate::recognize::align::find_missing_indices_cos;
use crate::recognize::align::{extract_y_centers, find_missing_indices, find_extra_indices, filter_by_extra_indices};
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
        assist_location: &mut AssistLocation
    ) -> Result<AssistLocation> {
        let left_area = merge_coordinates(&assist_location.left, T::assist_area_extend_size_w(), T::assist_area_extend_size_h());
        let right_area = merge_coordinates(&assist_location.right, T::assist_area_extend_size_w(), T::assist_area_extend_size_h());
        let left_src_assist = Self::find_assist_location::<T>(&processed_image.closed, &left_area)?;
        let right_src_assist = Self::find_assist_location::<T>(&processed_image.closed, &right_area)?;
        #[cfg(debug_assertions)]
        {
            println!("辅助定位点寻找结果，左侧找到{}个，右侧找到{}个", left_src_assist.len(), right_src_assist.len());
            let mut rgb = processed_image.rgb.clone();
            let _ = render_coordinates(&mut rgb, &left_src_assist, Some(crate::myutils::rendering::RenderMode::Hollow), None, None);
            let _ = render_coordinates(&mut rgb, &right_src_assist, Some(crate::myutils::rendering::RenderMode::Hollow), None, None);
            let out_path = format!("dev/test_data/debug/assist_location_found_{:?}.jpg", assist_location.left[0].x);
            opencv::imgcodecs::imwrite(&out_path, &rgb, &Vector::<i32>::new()).unwrap();
        }

        // 允许的最大多检/漏检数量
        const MAX_DIFF: usize = 2;

        let expected_count = assist_location.left.len(); // 左右标注数量应该相同
        let detected_left_count = left_src_assist.len();
        let detected_right_count = right_src_assist.len();

        let left_match = detected_left_count == expected_count;
        let right_match = detected_right_count == expected_count;

        // 如果两列都完全匹配，直接返回
        if left_match && right_match {
            return Ok(AssistLocation {
                left: left_src_assist,
                right: right_src_assist,
            });
        }

        // 必须有一列完全匹配才能进入对齐逻辑
        if !left_match && !right_match {
            anyhow::bail!(
                "左右两列都不匹配，无法对齐。左侧检测{}个/期望{}个，右侧检测{}个/期望{}个",
                detected_left_count, expected_count,
                detected_right_count, expected_count
            );
        }

        // 检查不匹配的那列差异是否在允许范围内
        let diff = if !left_match {
            (detected_left_count as i32 - expected_count as i32).abs() as usize
        } else {
            (detected_right_count as i32 - expected_count as i32).abs() as usize
        };

        if diff > MAX_DIFF {
            anyhow::bail!(
                "辅助定位点数量差异过大（最大允许{}），左侧检测{}个/期望{}个，右侧检测{}个/期望{}个",
                MAX_DIFF,
                detected_left_count, expected_count,
                detected_right_count, expected_count
            );
        }

        // 对不匹配的列进行处理
        if left_match {
            // 左侧是完整列，右侧需要处理
            let left_y = extract_y_centers(&left_src_assist);
            let right_y = extract_y_centers(&right_src_assist);

            if detected_right_count < expected_count {
                // 右侧漏检：找出缺失的标注点索引，从右侧标注中删除
                let missing_indices = find_missing_indices_cos(&left_y, &right_y);
                #[cfg(debug_assertions)]
                {
                    println!("右侧漏检，缺失索引: {:?}", missing_indices);
                }

                // 从后往前删除标注数据中的右侧点
                let mut indices_to_remove = missing_indices.clone();
                indices_to_remove.sort_by(|a, b| b.cmp(a));
                for idx in indices_to_remove {
                    assist_location.right.remove(idx);
                }

                Ok(AssistLocation {
                    left: left_src_assist,
                    right: right_src_assist,
                })
            } else {
                // 右侧多检：找出多余的检测点索引，过滤掉
                let extra_indices = find_extra_indices_cos(&left_y, &right_y);
                #[cfg(debug_assertions)]
                {
                    println!("右侧多检，多余索引: {:?}", extra_indices);
                }

                let final_right = filter_by_extra_indices(&right_src_assist, &extra_indices);

                Ok(AssistLocation {
                    left: left_src_assist,
                    right: final_right,
                })
            }
        } else {
            // 右侧是完整列，左侧需要处理
            let right_y = extract_y_centers(&right_src_assist);
            let left_y = extract_y_centers(&left_src_assist);

            if detected_left_count < expected_count {
                // 左侧漏检：找出缺失的标注点索引，从左侧标注中删除
                let missing_indices = find_missing_indices_cos(&right_y, &left_y);
                #[cfg(debug_assertions)]
                {
                    println!("左侧漏检，缺失索引: {:?}", missing_indices);
                }

                // 从后往前删除标注数据中的左侧点
                let mut indices_to_remove = missing_indices.clone();
                indices_to_remove.sort_by(|a, b| b.cmp(a));
                for idx in indices_to_remove {
                    assist_location.left.remove(idx);
                }

                Ok(AssistLocation {
                    left: left_src_assist,
                    right: right_src_assist,
                })
            } else {
                // 左侧多检：找出多余的检测点索引，过滤掉
                let extra_indices = find_extra_indices_cos(&right_y, &left_y);
                #[cfg(debug_assertions)]
                {
                    println!("左侧多检，多余索引: {:?}", extra_indices);
                }

                let final_left = filter_by_extra_indices(&left_src_assist, &extra_indices);

                Ok(AssistLocation {
                    left: final_left,
                    right: right_src_assist,
                })
            }
        }
    }

    // 求所有coor的x中位数，过滤掉x远离中位数的coor
    fn filter_assist_location_by_x(assist_location: &Vec<Coordinate>) -> Vec<Coordinate> {
        if assist_location.is_empty() {
            return Vec::new();
        }

        // 提取所有x坐标
        let mut x_values: Vec<i32> = assist_location.iter().map(|c| c.x).collect();

        // 计算中位数
        x_values.sort_unstable();
        let median = if x_values.len() % 2 == 0 {
            let mid = x_values.len() / 2;
            (x_values[mid - 1] + x_values[mid]) / 2
        } else {
            x_values[x_values.len() / 2]
        };

        let threshold = AssistLocationPageConfig::assist_point_x_median_diff();

        // 过滤掉x值远离中位数的坐标
        assist_location.iter()
            .filter(|c| (c.x - median).abs() <= threshold)
            .cloned()
            .collect()
    }

    pub fn infer_paper(&self, processed_image: &ProcessedImage, assist_location: &mut AssistLocation) -> Result<AssistLocation> {
        let mut assist_locations = Vec::new();
        let mut split_locations = assist_location.split();
        for single_location in split_locations.iter_mut() {
            let real_single_location = self.infer_single::<AssistLocationPageConfig>(processed_image, single_location)?;
            assist_locations.push(real_single_location);
        }
        let res = AssistLocation::merge(&assist_locations);

        // 把修改后的 split_locations 合并回 assist_location
        let merged_ref = AssistLocation::merge(&split_locations);
        assist_location.left = merged_ref.left;
        assist_location.right = merged_ref.right;

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

        let assist_points = Self::filter_assist_location_by_x(&assist_points);
        
        Ok(assist_points)
    }

}