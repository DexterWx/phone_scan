use anyhow::{Context, Result};
use crate::models::{Quad, Mark, MobileOutput};
use crate::myutils::image::{get_perspective_transform_matrix, integral_image, pers_trans_image, process_image, read_image};
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

    pub fn inference(&self, image_bs64: &String) -> Result<Quad> {
        // 1. 读取图片
        let image = read_image(image_bs64)
            .context("解析输入失败: 读图失败")?;
        // 2. 处理图片
        let processed_image = process_image(&image)?;
        
        // 3. 定位检测
        let location = self.location_module.infer(&processed_image)?;

        // 4. 获取变换矩阵
        let pers_trans_matrix = get_perspective_transform_matrix(&location, &self.mark.boundary)?;

        // 5. 透视变换图片
        let baizheng = pers_trans_image(
            &processed_image, &pers_trans_matrix, self.mark.boundary.x+self.mark.boundary.w, self.mark.boundary.y+self.mark.boundary.h
        )?;

        let integral_image = integral_image(&baizheng.thresh)?;


        // 渲染
        #[cfg(debug_assertions)]
        {
            use opencv::{core::Vector, imgcodecs::imwrite};

            use crate::myutils::rendering::{render_quad, RenderMode};

            let mut render_image = image.clone();
            let _ = render_quad(
                &mut render_image, &location, Some(RenderMode::Hollow), None, None
            )?;
            let debug_path = format!("dev/test_data/debug/{}.jpg", "debug_location");
            let params = Vector::<i32>::new();
            imwrite(&debug_path, &render_image, &params)
                .context("保存调试图片失败")?;

            let gray_path = format!("dev/test_data/debug/{}.jpg", "baizheng_gray");
            imwrite(&gray_path, &baizheng.gray, &params)
                .context("保存调试图片失败")?;
            
            let thresh_path = format!("dev/test_data/debug/{}.jpg", "baizheng_thresh");
            imwrite(&thresh_path, &baizheng.thresh, &params)
                .context("保存调试图片失败")?;
            
            let close_path = format!("dev/test_data/debug/{}.jpg", "baizheng_closed");
            imwrite(&close_path, &baizheng.closed, &params)
                .context("保存调试图片失败")?;
        }

        Ok(location)
    }
}