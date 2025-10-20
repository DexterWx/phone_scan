use anyhow::{Context, Result};
use opencv::core::Mat;
use crate::models::{Mark, MobileOutput};
use crate::myutils::image::{get_perspective_transform_matrix, pers_trans_image, process_image};
use crate::myutils::myjson::from_json;
use crate::recognize::fill::RecFillModule;
use crate::recognize::location::LocationModule;
use crate::recognize::assist_location::{self, AssistLocationModule};

/// 识别引擎
pub struct RecEngine {
    /// 定位模块
    location_module: LocationModule,
    /// 填涂识别模块
    rec_fill_module: RecFillModule,
    /// 辅助定位模块
    assist_location_module: AssistLocationModule,
    /// 初始化mark信息
    mark: Mark,
}

impl RecEngine {
    pub fn new(mobile_input: &String) -> Result<Self> {
        Ok(Self {
            location_module: LocationModule::new(),
            assist_location_module: AssistLocationModule::new(),
            rec_fill_module: RecFillModule::new(),
            mark: from_json(mobile_input)?,
        })
    }

    pub fn inference(&self, image: &Mat) -> Result<MobileOutput> {
        // 1. 初始化输出
        let mut mobile_output = MobileOutput::new(&self.mark);
        // 2. 处理图片
        let processed_image = process_image(&image)?;
        
        // 3. 定位检测
        let location = self.location_module.infer(&processed_image)?;

        // 4. 获取变换矩阵
        let pers_trans_matrix = get_perspective_transform_matrix(&location, &self.mark.boundary)?;

        // 5. 第一次变换
        let baizheng = pers_trans_image(
            &processed_image, &pers_trans_matrix, self.mark.boundary.x+self.mark.boundary.w, self.mark.boundary.y+self.mark.boundary.h
        )?;

        // 6. 找到辅助定位点
        let assist_location = self.assist_location_module.infer(&baizheng, &self.mark.assist_location)?;

        // 7. 填涂识别
        self.rec_fill_module.infer(&baizheng, &mut mobile_output)?;


        // 渲染
        #[cfg(debug_assertions)]
        {
            use opencv::{core::{AlgorithmHint, Vector}, imgcodecs::imwrite, imgproc};

            use crate::{config::LocationConfig, myutils::{image::resize_image, rendering::{render_assist_location, render_output, render_quad, Colors, RenderMode}}};

            let mut render_image = resize_image(image, LocationConfig::TARGET_WIDTH)?;
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

            let mut render_out = baizheng.gray.clone();
            // 将灰度图转换为RGB格式
            let mut rgb_image = Mat::default();
            imgproc::cvt_color(&render_out, &mut rgb_image, imgproc::COLOR_GRAY2BGR, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
            render_out = rgb_image;
            let _ = render_output(&mut render_out, &mobile_output, &assist_location,Some(RenderMode::Hollow), Some(Colors::orange()), Some(2), Some(2.0));

            let render_out_path = format!("dev/test_data/debug/{}.jpg", "render_out");
            imwrite(&render_out_path, &render_out, &params)
                .context("保存调试图片失败")?;
            
        }


        Ok(mobile_output)
    }
}