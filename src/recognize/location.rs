use anyhow::Result;
use opencv::{
    core::Mat,
    prelude::*,
};
use crate::models::Coordinate;

pub struct LocationModule;

impl LocationModule {

    pub fn new() -> Self {
        Self {}
    }

    pub fn infer(&self, image: &Mat) -> Result<Coordinate> {
        // 推理
        todo!()
    }

    pub fn detect_boundary(&self, image: &Mat) -> Result<Vec<Coordinate>> {
        // 检测外围边界
        todo!()
    }

    pub fn filter_boundary(&self, boundaries: &Vec<Coordinate>) -> Result<Coordinate> {
        // 过滤边界，选一个最合适的。
        todo!()
    }

    pub fn validate_boundary(&self, boundary: &Coordinate) -> bool {
        // 验证边界
        todo!()
    }
}