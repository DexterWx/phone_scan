use anyhow::{Context, Result};
use crate::models::{Mark, MobileOutput};
use crate::myutils::image::read_image;
use crate::myutils::myjson::from_json;
use crate::recognize::location::LocationModule;

/// 识别引擎
pub struct RecEngine {
    /// 定位模块
    location_module: LocationModule,
    /// 初始化mark信息
    mark: Mark,
}

impl RecEngine {
    pub fn new(mobile_input: &String) -> Result<Self> {
        Ok(Self {
            location_module: LocationModule::new(),
            mark: from_json(mobile_input)?,
        })
    }

    pub fn inference(&self, image_bs64: &String) -> Result<MobileOutput> {
        // 1. 读取图片
        let image = read_image(&image_bs64)
            .context("解析输入失败: 读图失败")?;
        
        // 2. 定位检测
        let location = self.location_module.infer(&image)?;
        
        todo!()
    }
}