use crate::{config::{VxConfig, VxPageConfig}, models::{Coordinate, MobileOutput, ProcessedImage, RecResult, RecType}, myutils::image::{crop_image, det_red_lab}};
use anyhow::{Ok, Result};
use opencv::{core::{Mat, MatTraitConstManual, Size}, imgproc, prelude::MatTraitConst};
use tract_onnx::prelude::*;
use ndarray::{Array4, ArrayViewD};
use std::{sync::Arc, thread::Thread};
use rayon::prelude::*;

pub struct RecVxModule {
    onnx_model: Option<Arc<TypedRunnableModel<Graph<TypedFact, Box<dyn TypedOp>>>>>,
    dictionary: Option<Vec<String>>,
    pool: Option<rayon::ThreadPool>,
}

impl RecVxModule {
    pub fn new_paper(model_path: &String, num_threads: usize) -> Result<Self> {
        let dictionary = vec!["blank".to_string(), "0".to_string(), "1".to_string()];
        let onnx_model = Self::load_model(model_path)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()?;

        Ok(Self {
            onnx_model: Some(Arc::new(onnx_model)),
            dictionary: Some(dictionary),
            pool: Some(pool),
        })
    }

    pub fn new_single() -> Result<Self> {
        Ok(Self {
            onnx_model: None,
            dictionary: None,
            pool: None,
        })
    }

    pub fn load_model(path: &str) -> Result<TypedRunnableModel<TypedModel>> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec![1, 3, 48, 96], // ✅ 固定宽度
                ),
            )?
            .into_optimized()?
            .into_runnable()?;   // ✅

        Ok(model)
    }


    pub fn infer(&self, process_image: &ProcessedImage, mobile_output: &mut MobileOutput) -> Result<()> {
        for rec_result in mobile_output.rec_results.iter_mut() {
            if rec_result.rec_type != RecType::Vx {
                continue;
            }
            self.rec_options(process_image, rec_result)?;
            // self.refine_options(rec_result)?;
        }
        self.set_vx(mobile_output)?;

        Ok(())
    }

    fn rec_options(&self, process_image: &ProcessedImage, options: &mut RecResult) -> Result<()> {
        for rec_option in options.rec_options.iter_mut() {
            let coor = &rec_option.coordinate;
            let sub_image = crop_image(&process_image.rgb, coor)?;
            let red_image = det_red_lab(&sub_image)?;
            let has_red = self.quick_filter::<VxPageConfig>(&red_image)?;
            let mut vx_res = false;
            if has_red {
                let res = self.infer_single_char(&sub_image)?;
                if res == "0" { vx_res = true; }
            }
            rec_option.vx = vx_res;
        }
        Ok(())
    }

    /// 并行推理：将所有 rec_results 中的 Vx 类型 options 扁平化后并行处理
    pub fn infer_parallel(&self, process_image: &ProcessedImage, mobile_output: &mut MobileOutput) -> Result<()> {
        // 1. 收集所有需要处理的 Vx 类型的 (rec_idx, opt_idx, coordinate)
        let tasks: Vec<(usize, usize, Coordinate)> = mobile_output.rec_results
            .iter()
            .enumerate()
            .filter(|(_, r)| r.rec_type == RecType::Vx)
            .flat_map(|(rec_idx, r)| {
                r.rec_options.iter().enumerate().map(move |(opt_idx, opt)| {
                    (rec_idx, opt_idx, opt.coordinate.clone())
                })
            })
            .collect();

        if tasks.is_empty() {
            return Ok(());
        }

        // 2. 串行裁剪所有子图像
        let sub_images: Vec<_> = tasks.iter()
            .map(|(_, _, coor)| crop_image(&process_image.rgb, coor))
            .collect::<Result<Vec<_>>>()?;

        // 3. 在函数内创建线程池，并行处理
        let pool = self.pool.as_ref().unwrap();

        let results: Vec<bool> = pool.install(|| {
            sub_images
                .par_iter()
            .map(|sub_image| {
                    let res = self.infer_single_char(sub_image).ok()?;
                    Some(res == "0")
                })
                .map(|opt| opt.unwrap_or(false))
                .collect()
        });

        // 4. 回写结果
        for ((rec_idx, opt_idx, _), vx_res) in tasks.iter().zip(results) {
            mobile_output.rec_results[*rec_idx].rec_options[*opt_idx].vx = vx_res;
        }

        // 5. 设置最终结果
        self.set_vx(mobile_output)?;

        Ok(())
    }

    pub fn infer_single_char(
        &self, bgr: &Mat
    ) -> Result<String> {
        let onnx_model = self.onnx_model.as_ref().unwrap();
        // 1. 预处理
        let input = self.preprocess_for_model(bgr)?;

        // 2. 前向推理
        let outputs = onnx_model.run(tvec!(input.into()))?;

        // 3. 拿第一个输出
        let output = outputs[0]
            .to_array_view::<f32>()?
            .to_owned();

        // 4. CTC 解码（单字符）
        let text = self.ctc_decode(output.view())?;

        Ok(text)
    }

    pub fn preprocess_for_model(&self,bgr: &Mat) -> Result<Tensor> {
        let h = bgr.rows();
        let w = bgr.cols();
        if h <= 0 || w <= 0 {
            anyhow::bail!("invalid image size {}x{}", w, h);
        }

        // Python: imgC=3, imgH=48, imgW=96
        let img_h = 48;
        let img_w = 96;

        // 等比缩放（Python 用 int(imgH * ratio)，不是 round）
        let ratio = w as f32 / h as f32;
        let mut resized_w = (img_h as f32 * ratio) as i32;
        if resized_w > img_w {
            resized_w = img_w;
        }

        // resize
        let mut resized = Mat::default();
        imgproc::resize(
            bgr,
            &mut resized,
            Size::new(resized_w, img_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        // 确保连续
        let resized = resized.try_clone()?;
        let data = resized.data_bytes()?; // BGRBGR...

        // padding_im: [1, 3, 48, 96]
        let mut input = Array4::<f32>::zeros((1, 3, img_h as usize, img_w as usize));

        for y in 0..img_h as usize {
            for x in 0..resized_w as usize {
                let idx = (y * resized_w as usize + x) * 3;
                let b = data[idx] as f32 / 255.0;
                let g = data[idx + 1] as f32 / 255.0;
                let r = data[idx + 2] as f32 / 255.0;

                // (x - 0.5) / 0.5
                input[[0, 0, y, x]] = (b - 0.5) / 0.5;
                input[[0, 1, y, x]] = (g - 0.5) / 0.5;
                input[[0, 2, y, x]] = (r - 0.5) / 0.5;
            }
        }
        Ok(input.into_tensor())
    }

    pub fn ctc_decode(
        &self,
        output: ArrayViewD<f32>
    ) -> Result<String> {
        let dictionary = self.dictionary.as_ref().unwrap();
        let shape = output.shape();
        if shape.len() != 3 || shape[0] != 1 {
            anyhow::bail!("invalid output shape {:?}", shape);
        }

        let time_steps = shape[1];
        let num_classes = shape[2];

        let mut result = Vec::new();
        let mut last_index = 0usize;

        for t in 0..time_steps {
            let mut max_idx = 0usize;
            let mut max_val = output[[0, t, 0]];

            for c in 1..num_classes {
                let v = output[[0, t, c]];
                if v > max_val {
                    max_val = v;
                    max_idx = c;
                }
            }

            if max_idx != 0 && max_idx != last_index {
                result.push(dictionary[max_idx].clone());
            }

            last_index = max_idx;
        }

        Ok(result.join(""))
    }

    /// 快速预筛选（基于填涂率）
    /// 输入图像：红色线条为白色（255），背景为黑色（0）
    fn quick_filter<T: VxConfig>(&self, binary_image: &Mat) -> Result<bool> {
        let total_pixels = (binary_image.rows() * binary_image.cols()) as f64;
        // count_non_zero 统计非零像素（白色=255），即红色线条
        let red_pixels = opencv::core::count_non_zero(binary_image)? as f64;
        let fill_ratio = red_pixels / total_pixels;

        // 填涂率过低 → 无效/空白
        if fill_ratio < T::fill_ratio_min() {
            return Ok(false);
        }

        // 填涂率过高 → 乱划/涂抹
        if fill_ratio > T::fill_ratio_max() {
            return Ok(false);
        }

        // 需要继续分析
        Ok(true)
    }

    fn set_vx(&self, mobile_output: &mut MobileOutput) -> Result<()> {
        for rec_result in mobile_output.rec_results.iter_mut() {
            if rec_result.rec_type != RecType::Vx {
                continue;
            }
            for (index, rec_option) in rec_result.rec_options.iter_mut().enumerate() {
                if rec_option.vx {
                    rec_result.rec_result[index] = true;
                } else {
                    rec_result.rec_result[index] = false;
                }
            }
        }
        Ok(())
    }
}

