use anyhow::{Context, Result};
use opencv::{
    core::{Mat, Vector, Point2i, Size, AlgorithmHint},
    imgproc,
    prelude::*,
};
use crate::models::{Coordinate, ContourInfo};
use crate::config::{BoundaryDetectionConfig, ScoringConfig};
use crate::myutils::image::process_image;

pub struct LocationModule;

impl LocationModule {

    pub fn new() -> Self {
        Self
    }

    pub fn infer(&self, image: &Mat) -> Result<Coordinate> {
        let processed = process_image(image).context("图片预处理失败")?;
        let boundaries = self.detect_boundary(&processed.closed)?;
        let boundary = self.filter_boundary(&boundaries, &processed.closed)?;
        let valid = self.validate_boundary(&boundary);
        if !valid {
            return Err(anyhow::anyhow!("边界验证失败"));
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
        let min_area = BoundaryDetectionConfig::MIN_AREA_RATIO * (w as f64) * (h as f64);

        let mut contour_infos = Vec::new();
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            
            // 第一步：只做最小面积占比的筛选
            if area < min_area {
                continue;
            }

            let bounding_rect = imgproc::bounding_rect(&contour)?;
            
            contour_infos.push(ContourInfo {
                points: contour,
                area,
                bounding_rect,
                score: 0.0, // 评分在filter_boundary中计算
            });
        }

        Ok(contour_infos)
    }

    pub fn filter_boundary(&self, boundaries: &Vec<ContourInfo>, image: &Mat) -> Result<Coordinate> {
        // 过滤边界，选一个最合适的。
        if boundaries.is_empty() {
            return Err(anyhow::anyhow!("未找到合适的外部黑框"));
        }

        let mut best_contour = None;
        let mut best_score = f64::NEG_INFINITY;

        // 获取图像尺寸
        let w = image.cols();
        let h = image.rows();

        for contour_info in boundaries {
            let area = contour_info.area;
            let rect = &contour_info.bounding_rect;
            
            // 计算边界距离
            let x = rect.x;
            let y = rect.y;
            let cw = rect.width;
            let ch = rect.height;
            
            // margin = min(x, y, w - x - cw, h - y - ch)
            let margin = (x)
                .min(y)
                .min(w - x - cw)
                .min(h - y - ch)
                .min(0); // 确保不为负值
            
            // score = area - margin * penalty
            let score = area - (margin as f64) * ScoringConfig::MARGIN_PENALTY;
            
            if score > best_score {
                best_score = score;
                best_contour = Some(contour_info);
            }
        }

        let best = best_contour.ok_or_else(|| anyhow::anyhow!("未找到合适的外部黑框"))?;
        
        // 将边界矩形转换为Coordinate
        let rect = &best.bounding_rect;
        Ok(Coordinate {
            x: rect.x,
            y: rect.y,
            w: rect.width,
            h: rect.height,
        })
    }

    pub fn validate_boundary(&self, boundary: &Coordinate) -> bool {
        // 验证边界
        todo!()
    }
}