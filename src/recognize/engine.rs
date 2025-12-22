use anyhow::{Context, Ok, Result};
use opencv::core::Mat;
use crate::config::{AssistLocationSingleConfig, FillPageConfig, FillSingleConfig, ImageProcessingConfig};
use crate::models::{MarkPaper, MarkSingle, MobileOutput};
use crate::myutils::image::{get_perspective_transform_matrix_with_boundary, get_perspective_transform_matrix_with_assists, pers_trans_image, process_image};
use crate::myutils::myjson::from_json;
use crate::recognize::fill::RecFillModule;
use crate::recognize::location::LocationModule;
use crate::recognize::assist_location::AssistLocationModule;
use crate::recognize::page_number::PageNumberModule;
use crate::recognize::vx::RecVxModule;

/// 识别引擎
pub struct RecEngine {
    /// 定位模块
    location_module: LocationModule,
    /// 填涂识别模块
    rec_fill_module: RecFillModule,
    /// 辅助定位模块
    assist_location_module: AssistLocationModule,
    /// 划分识别模块
    rec_vx_module: RecVxModule,
    /// 页码识别模块
    page_number_module: PageNumberModule,
    /// 初始化mark信息
    mark_single: Option<MarkSingle>,
    mark_paper: Option<MarkPaper>
}

impl RecEngine {
    pub fn new_single(mobile_input: &String) -> Result<Self> {
        Ok(Self {
            location_module: LocationModule::new(),
            assist_location_module: AssistLocationModule::new(),
            rec_fill_module: RecFillModule::new(),
            rec_vx_module: RecVxModule::new(),
            page_number_module: PageNumberModule::new(),
            mark_single: Some(from_json(mobile_input)?),
            mark_paper: None
        })
    }

    pub fn new_paper(mobile_input: &String) -> Result<Self> {
        Ok(Self {
            location_module: LocationModule::new(),
            assist_location_module: AssistLocationModule::new(),
            rec_fill_module: RecFillModule::new(),
            rec_vx_module: RecVxModule::new(),
            page_number_module: PageNumberModule::new(),
            mark_paper: Some(from_json(mobile_input)?),
            mark_single: None
        })
    }

    pub fn inference_single(&self, image: &Mat) -> Result<MobileOutput> {
        let mark = self.mark_single.as_ref().context("引擎未初始化")?;
        // 1. 初始化输出
        let mut mobile_output = MobileOutput::new(&mark.rec_items);
        
        // 2. 处理图片
        let processed_image = process_image(&image, ImageProcessingConfig::TARGET_WIDTH_A4)?;
        
        // 3. 定位检测
        let location = self.location_module.infer(&processed_image)?;

        // 4. 获取变换矩阵
        let pers_trans_matrix = get_perspective_transform_matrix_with_boundary(&location, &mark.boundary)?;

        // 5. 第一次变换
        let baizheng = pers_trans_image(
            &processed_image, &pers_trans_matrix, mark.boundary.x+mark.boundary.w, mark.boundary.y+mark.boundary.h
        )?;

        // 6. 找到辅助定位点
        let assist_location = self.assist_location_module.infer_single::<AssistLocationSingleConfig>(&baizheng, &mark.assist_location)?;
        
        // 7. 获取变换矩阵
        let pers_trans_matrix = get_perspective_transform_matrix_with_assists(&assist_location, &mark.assist_location)?;
        
        // 8. 第二次变换
        let baizheng = pers_trans_image(
            &baizheng, &pers_trans_matrix, mark.boundary.x+mark.boundary.w, mark.boundary.y+mark.boundary.h
        )?;

        // 9. 填涂识别
        self.rec_fill_module.infer::<FillSingleConfig>(&baizheng, &mut mobile_output)?;

        // 渲染
        #[cfg(debug_assertions)]
        {
            use opencv::{core::{AlgorithmHint, Vector}, imgcodecs::imwrite, imgproc};

            use crate::myutils::rendering::{render_output, render_quad, Colors, RenderMode};

            let mut render_image = processed_image.rgb.clone();
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
            let _ = render_output(&mut render_out, &mobile_output, &mark.assist_location,Some(RenderMode::Hollow), Some(Colors::orange()), Some(2), Some(2.0));

            let render_out_path = format!("dev/test_data/debug/{}.jpg", "render_out");
            imwrite(&render_out_path, &render_out, &params)
                .context("保存调试图片失败")?;
            
        }


        Ok(mobile_output)
    }

    pub fn inference_paper(&self, image: &Mat) -> Result<MobileOutput> { 
        let mark = self.mark_paper.as_ref().context("引擎未初始化")?;
        let mark = &mark.resize(ImageProcessingConfig::PAPER_SCAN_TARGET_SCALE);
        let target_width = if mark.is_a4() {
            ImageProcessingConfig::TARGET_WIDTH_A4
        } else {
            ImageProcessingConfig::TARGET_WIDTH_A3
        };

        // 1. 处理图片
        let processed_image = process_image(&image, target_width)?;
        
        // 2. 定位检测
        let location = self.location_module.infer(&processed_image)?;

        // 3. 获取变换矩阵
        let tg_boundary = &mark.boundary;
        let pers_trans_matrix = get_perspective_transform_matrix_with_boundary(&location, tg_boundary)?;
        
        // 4. 第一次变换
        let baizheng = pers_trans_image(
            &processed_image, &pers_trans_matrix, tg_boundary.x+tg_boundary.w, tg_boundary.y+tg_boundary.h
        )?;

        let page_index = self.page_number_module.infer(&baizheng, &mark.page_number)?;
        let page_mark = mark.pages.get(page_index-1)
            .context("未找到对应的页码信息")?;

        // 5. 找到辅助定位点
        let assist_location = self.assist_location_module.infer_paper(&baizheng, &page_mark.assist_location)?;
        
        // 6. 获取变换矩阵
        let pers_trans_matrix = get_perspective_transform_matrix_with_assists(&assist_location, &page_mark.assist_location)?;
        
        // 7. 第二次变换
        let baizheng = pers_trans_image(
            &baizheng, &pers_trans_matrix, mark.boundary.x+mark.boundary.w, mark.boundary.y+mark.boundary.h
        )?;

        // 8. 初始化输出
        let mut mobile_output = MobileOutput::new(&page_mark.rec_items);
        mobile_output.page_number = page_index-1;
        
        // 9. 填涂识别
        self.rec_fill_module.infer::<FillPageConfig>(&baizheng, &mut mobile_output)?;

        // 10. vx识别
        self.rec_vx_module.infer(&baizheng, &mut mobile_output)?;
        

        // 渲染
        #[cfg(debug_assertions)] {
            use opencv::{core::Vector, imgcodecs::imwrite};
            use crate::myutils::rendering::{render_output, render_quad, Colors, RenderMode};

            let mut render_image = processed_image.rgb.clone();
            let _ = render_quad(
                &mut render_image, &location, Some(RenderMode::Hollow), None, None
            )?;
            let debug_path = format!("dev/test_data/debug/{}.jpg", "debug_location");
            let params = Vector::<i32>::new();
            imwrite(&debug_path, &render_image, &params)
                .context("保存调试图片失败")?;

            let rgb_path = format!("dev/test_data/debug/{}.jpg", "baizheng_rgb");
            imwrite(&rgb_path, &baizheng.rgb, &params)
                .context("保存调试图片失败")?;

            let mut render_out = baizheng.rgb.clone();
            let _ = render_output(&mut render_out, &mobile_output, &mark.pages[0].assist_location,Some(RenderMode::Hollow), Some(Colors::orange()), Some(2), Some(1.0));

            let render_out_path = format!("dev/test_data/debug/{}.jpg", "render_out");
            imwrite(&render_out_path, &render_out, &params)
                .context("保存调试图片失败")?;
        }
        
        
        
        Ok(mobile_output)
    }
}