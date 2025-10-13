use anyhow::{Context, Result};
use opencv::{
    core::{Mat, Vector, Point2i},
    imgproc,
    prelude::*,
};
use crate::models::{ContourInfo, Quad, ProcessedImage};
use crate::config::LocationConfig;

pub struct LocationModule;

impl LocationModule {

    pub fn new() -> Self {
        Self
    }

    pub fn infer(&self, processed_image: &ProcessedImage) -> Result<Quad> {
        let boundaries = self.detect_boundary(&processed_image.closed)?;
        let boundary = self.filter_boundary(&boundaries, &processed_image.closed)?;
        let valid = self.validate_boundary(&boundary);
        if !valid {
            anyhow::bail!("边界验证失败");
        }
        Ok(boundary)
    }

    pub fn detect_boundary(&self, morphology: &Mat) -> Result<Vec<ContourInfo>> {
        
        // 查找连通区域（外部轮廓）
        let mut contours = Vector::<Vector<Point2i>>::new();
        imgproc::find_contours(
            morphology,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point2i::new(0, 0),
        )?;

        let h = morphology.rows();
        let w = morphology.cols();
        let min_area = LocationConfig::MIN_AREA_RATIO * (w as f64) * (h as f64);

        let mut contour_infos = Vec::new();
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            
            // 第一步：只做最小面积占比的筛选
            if area < min_area {
                continue;
            }

            contour_infos.push(ContourInfo {
                points: contour,
                area
            });
        }

        Ok(contour_infos)
    }

    pub fn filter_boundary(&self, boundaries: &Vec<ContourInfo>, image: &Mat) -> Result<Quad> {
        // 过滤边界，选一个最合适的。
        if boundaries.is_empty() {
            anyhow::bail!("未找到合适的外部黑框");
        }

        let mut best_contour = None;
        let mut best_score = f64::NEG_INFINITY;

        // 获取图像尺寸
        let w = image.cols();
        let h = image.rows();

        for contour_info in boundaries {
            let area = contour_info.area;
            
            // 计算边界距离（基于轮廓点的边界矩形）
            let bounding_rect = imgproc::bounding_rect(&contour_info.points)?;
            let x = bounding_rect.x;
            let y = bounding_rect.y;
            let cw = bounding_rect.width;
            let ch = bounding_rect.height;
            
            // margin = min(x, y, w - x - cw, h - y - ch)
            let margin = x.min(y).min(w - x - cw).min(h - y - ch).max(0); // 取最小边距，且不小于0
            
            // score = area - margin * penalty
            let score = area - (margin as f64) * LocationConfig::MARGIN_PENALTY;
            
            if score > best_score {
                best_score = score;
                best_contour = Some(contour_info);
            }
        }

        let best = best_contour.ok_or_else(|| anyhow::anyhow!("未找到合适的外部黑框"))?;
        
        // 使用轮廓近似算法提取四边形
        let mut approx_curve = Vector::<Point2i>::new();
        let epsilon = LocationConfig::EPSILON_FACTOR * imgproc::arc_length(&best.points, true)?; 
        imgproc::approx_poly_dp(&best.points, &mut approx_curve, epsilon, true)?;
        
        // 确保我们有4个点
        if approx_curve.len() != 4 {
            anyhow::bail!("未能找到合适的四边形，检测到的顶点数: {}", approx_curve.len());
        }
        
        // 提取四个点
        let mut points_array: [Point2i; 4] = [
            Point2i::from(approx_curve.get(0)?),
            Point2i::from(approx_curve.get(1)?),
            Point2i::from(approx_curve.get(2)?),
            Point2i::from(approx_curve.get(3)?),
        ];
        
        // 确保四个点按顺时针方向排列，从左上角开始
        Self::order_points(&mut points_array);
        
        Ok(Quad {
            points: points_array,
        })
    }

    /// 对四边形的四个顶点进行排序，确保按顺时针方向排列，从左上角开始
    fn order_points(pts: &mut [Point2i; 4]) {
        // 计算质心
        let centroid_x = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) as f32 / 4.0;
        let centroid_y = (pts[0].y + pts[1].y + pts[2].y + pts[3].y) as f32 / 4.0;
        
        // 按角度排序
        pts.sort_by(|a, b| {
            let angle_a = (a.y as f32 - centroid_y).atan2(a.x as f32 - centroid_x);
            let angle_b = (b.y as f32 - centroid_y).atan2(b.x as f32 - centroid_x);
            
            angle_a.partial_cmp(&angle_b).unwrap()
        });
        
        // 确保第一个点是左上角（x和y值最小的点）
        let mut min_index = 0;
        let mut min_sum = pts[0].x + pts[0].y;
        
        for i in 1..4 {
            let sum = pts[i].x + pts[i].y;
            if sum < min_sum {
                min_sum = sum;
                min_index = i;
            }
        }
        
        // 旋转数组，使左上角点成为第一个点
        pts.rotate_left(min_index);
    }

    pub fn validate_boundary(&self, boundary: &Quad) -> bool {
        // 验证边界
        true
    }
}