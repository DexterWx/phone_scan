use anyhow::Result;
use opencv::{
    core::{Mat, Vector, Point2i},
    imgproc,
    prelude::*,
};
use crate::models::{ContourInfo, Quad, ProcessedImage};
use crate::config::ImageProcessingConfig;

pub struct LocationModule;

impl LocationModule {

    pub fn new() -> Self {
        Self
    }

    pub fn infer(&self, processed_image: &ProcessedImage) -> Result<Quad> {
        let boundaries = self.detect_boundary(&processed_image.closed)?;
        let contour = self.select_best_contour(&boundaries, &processed_image.closed)?;
        let boundary = self.contour_to_quad(contour)?;
        let valid = self.validate_boundary(&boundary, contour);
        if !valid {
            anyhow::bail!("需要铺平试卷!");
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
        let min_area = ImageProcessingConfig::MIN_AREA_RATIO * (w as f64) * (h as f64);

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

    /// 从多个候选轮廓中选择最合适的一个
    pub fn select_best_contour<'a>(&self, boundaries: &'a Vec<ContourInfo>, image: &Mat) -> Result<&'a ContourInfo> {
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
            let score = area - (margin as f64) * ImageProcessingConfig::MARGIN_PENALTY;

            if score > best_score {
                best_score = score;
                best_contour = Some(contour_info);
            }
        }

        best_contour.ok_or_else(|| anyhow::anyhow!("未找到合适的外部黑框"))
    }

    /// 将轮廓信息转换为四边形
    pub fn contour_to_quad(&self, contour_info: &ContourInfo) -> Result<Quad> {
        // 使用轮廓近似算法提取四边形
        let mut approx_curve = Vector::<Point2i>::new();
        let epsilon = ImageProcessingConfig::EPSILON_FACTOR * imgproc::arc_length(&contour_info.points, true)?;
        imgproc::approx_poly_dp(&contour_info.points, &mut approx_curve, epsilon, true)?;

        // 如果点数不是4，使用凸包作为备选方案
        if approx_curve.len() != 4 {
            let mut hull = Vector::<Point2i>::new();
            imgproc::convex_hull(&contour_info.points, &mut hull, true, true)?;

            // 对凸包进行多边形逼近
            imgproc::approx_poly_dp(&hull, &mut approx_curve, epsilon, true)?;

            // 如果凸包逼近后仍然不是4个点，则报错
            if approx_curve.len() != 4 {
                anyhow::bail!("未能找到合适的四边形，原始轮廓顶点数: {}，凸包逼近后顶点数: {}",
                    contour_info.points.len(), approx_curve.len());
            }
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

    pub fn filter_boundary(&self, boundaries: &Vec<ContourInfo>, image: &Mat) -> Result<Quad> {
        let best_contour = self.select_best_contour(boundaries, image)?;
        self.contour_to_quad(best_contour)
    }

    /// 对四边形的四个顶点进行排序，确保按顺时针方向排列，从左上角开始
    pub fn order_points(pts: &mut [Point2i; 4]) {
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

    // 以boundary构建四边形，计算contour上所有点到最近边的距离，对所有距离求平均值，判断平均距离是否在合理范围内
    pub fn validate_boundary(&self, boundary: &Quad, contour: &ContourInfo) -> bool {
        if contour.points.len() == 0 {
            return false;
        }

        // 四边形的四条边
        let edges = [
            (boundary.points[0], boundary.points[1]),
            (boundary.points[1], boundary.points[2]),
            (boundary.points[2], boundary.points[3]),
            (boundary.points[3], boundary.points[0]),
        ];

        let mut total_distance = 0.0;
        let mut count = 0;

        // 计算每个轮廓点到最近边的距离
        for i in 0..contour.points.len() {
            if let Ok(point) = contour.points.get(i) {
                let mut min_distance = f64::MAX;

                // 找到到四条边的最小距离
                for &(p1, p2) in &edges {
                    let distance = Self::point_to_line_distance(point, p1, p2);
                    if distance < min_distance {
                        min_distance = distance;
                    }
                }

                total_distance += min_distance;
                count += 1;
            }
        }

        if count == 0 {
            return false;
        }

        // 计算平均距离
        let avg_distance = total_distance / count as f64;

        // 判断平均距离是否在合理范围内（阈值可以根据实际情况调整）
        // 一般来说，如果轮廓与四边形匹配良好，平均距离应该很小（比如小于5像素）
        let threshold = ImageProcessingConfig::BOUNDARY_PENALTY; // 这个值可以根据实际情况调整

        #[cfg(debug_assertions)]
        {
            println!("边界验证: 平均距离 = {:.2}, 阈值 = {}", avg_distance, threshold);
        }

        avg_distance <= threshold
    }

    /// 计算点到线段的距离
    fn point_to_line_distance(point: Point2i, line_start: Point2i, line_end: Point2i) -> f64 {
        let px = point.x as f64;
        let py = point.y as f64;
        let x1 = line_start.x as f64;
        let y1 = line_start.y as f64;
        let x2 = line_end.x as f64;
        let y2 = line_end.y as f64;

        // 线段长度的平方
        let line_length_sq = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);

        if line_length_sq == 0.0 {
            // 线段退化为点
            return ((px - x1) * (px - x1) + (py - y1) * (py - y1)).sqrt();
        }

        // 计算点在线段上的投影参数 t
        // t = 0 表示投影在起点，t = 1 表示投影在终点
        let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq;

        // 将 t 限制在 [0, 1] 范围内，确保投影点在线段上
        let t = t.max(0.0).min(1.0);

        // 计算投影点坐标
        let proj_x = x1 + t * (x2 - x1);
        let proj_y = y1 + t * (y2 - y1);

        // 返回点到投影点的距离
        ((px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y)).sqrt()
    }
}