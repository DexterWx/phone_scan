use crate::{config::{ImageProcessingConfig, VxConfig, VxPageConfig}, models::{ContourInfo, Coordinate, MarkPaper, MobileOutput, ProcessedImage, Quad, RecResult, RecType}, myutils::{image::{crop_image, det_red_lab, get_perspective_transform_matrix_with_points, merge_coordinates, pers_trans_image}, math::match_points, rendering::{RenderMode, render_quad}}, recognize::location::LocationModule};
use anyhow::{Ok, Result};
use opencv::{core::{Mat, MatTraitConstManual, Point2f, Point2i, Size, Vector}, imgproc, prelude::MatTraitConst};
use tract_onnx::prelude::*;
use ndarray::{Array4, ArrayViewD};
use std::sync::Arc;
use rayon::prelude::*;

static mut COUNT: i32 = 0;

pub struct RecVxModule {
    onnx_model: Option<Arc<TypedRunnableModel<Graph<TypedFact, Box<dyn TypedOp>>>>>,
    pool: Option<rayon::ThreadPool>,
}

impl RecVxModule {
    pub fn new_paper(model_path: &String, num_threads: usize) -> Result<Self> {
        let onnx_model = Self::load_model(model_path)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()?;

        Ok(Self {
            onnx_model: Some(Arc::new(onnx_model)),
            pool: Some(pool),
        })
    }

    pub fn new_single() -> Result<Self> {
        Ok(Self {
            onnx_model: None,
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
                    tvec![1, 3, 28, 42], // TinyCNN 输入尺寸
                ),
            )?
            .into_optimized()?
            .into_runnable()?;

        Ok(model)
    }

    pub fn refine_image(&self, process_image: &mut ProcessedImage, mobile_output: &mut MobileOutput, mark: &MarkPaper) -> Result<()> {
        let mut all_src_points: Vec<Point2f> = Vec::new();
        let mut all_target_points: Vec<Point2f> = Vec::new();
        for rec_result in mobile_output.rec_results.iter_mut() {
            if rec_result.rec_type != RecType::Vx {
                continue;
            }
            // 所有选项的coors list
            let coors: Vec<Coordinate> = rec_result.rec_options.iter().map(
                |option| option.coordinate.clone()
            ).collect();
            if coors.is_empty() {
                continue;
            }
            let _w = coors[0].w;
            let _h = coors[0].h;
            let extend_size = _w / 2;
            let big_coor = merge_coordinates(&coors, extend_size, extend_size);
            let crop_mor = crop_image(&process_image.closed, &big_coor)?;
    
            let boxes = self.detect_vx_box(&crop_mor, (_w*_h) as f64)?;
            let boxes = self.filter_vx_box(&boxes)?;
            if boxes.is_empty() {
                println!("Refine VX: No box found");
                continue;
            }
            if boxes.len() > rec_result.rec_options.len() {
                println!("Refine VX: Too many boxes");
                continue;
            }
            let src_points: Vec<Point2f> = boxes.iter().map(
                |_box| {
                    let mut center = _box.get_center();
                    center.x += big_coor.x as f32;
                    center.y += big_coor.y as f32;
                    center
                }
            ).collect();
            let target_points: Vec<Point2f> = rec_result.rec_options.iter().map(
                |option| option.coordinate.get_center()
            ).collect();
            let match_points = match_points(&src_points, &target_points)?;
            all_src_points.extend(&src_points);
            all_target_points.extend(&match_points);
        }
        println!("Refine VX: {}", all_src_points.len());
        if all_src_points.len() < 8{
            all_src_points.extend(mark.boundary.to_points());
            all_target_points.extend(mark.boundary.to_points());
        }
        let pers_trans_matrix = get_perspective_transform_matrix_with_points(
            &Vector::from_iter(all_src_points),
            &Vector::from_iter(all_target_points)
        )?;
        
        pers_trans_image(
            process_image, &pers_trans_matrix, mark.boundary.x+mark.boundary.w, mark.boundary.y+mark.boundary.h
        )?;
        Ok(())
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
            // let coor = &rec_option.coordinate;
            // let sub_image = crop_image(&process_image.rgb, coor)?;
            
            // let red_image = det_red_lab(&sub_image)?;
            // let has_red = self.quick_filter::<VxPageConfig>(&red_image)?;
            // let mut vx_res = false;
            // if has_red {
            //     let (class_id, _confidence) = self.infer_tiny_cnn(&sub_image)?;
            //     if class_id == 0 { vx_res = true; }
            // }
            // rec_option.vx = vx_res;
            rec_option.vx = true;
            // unsafe {
            //     COUNT += 1;
            //     let out_path = format!("dev/test_data/debug/vx_{:?}_{vx_res:?}.jpg", COUNT);
            //     opencv::imgcodecs::imwrite(&out_path, &sub_image, &Default::default())?;
            //     if has_red {
            //         let out_path = format!("dev/test_data/debug/vx_{:?}_red.jpg", COUNT);
            //         opencv::imgcodecs::imwrite(&out_path, &red_image, &Default::default())?;
            //     }
            // }
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
                    let (class_id, _confidence) = self.infer_tiny_cnn(sub_image).ok()?;
                    Some(class_id == 0)
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

    /// 新的分类模型推理：0=single（单线），1=cancel（非单线）
    /// 返回 (class_id, confidence)
    pub fn infer_tiny_cnn(&self, bgr: &Mat) -> Result<(usize, f64)> {
        let onnx_model = self.onnx_model.as_ref().unwrap();

        // 1. 预处理（保持比例resize + padding）
        let input = self.preprocess_for_tiny_cnn(bgr)?;

        // 2. 前向推理
        let outputs = onnx_model.run(tvec!(input.into()))?;

        // 3. 拿第一个输出 [1, 2]
        let output = outputs[0].to_array_view::<f32>()?;

        // 4. Softmax + Argmax，返回 (class_id, confidence)
        self.classify(output)
    }

    /// TinyCNN 预处理：保持宽高比 resize + 居中 padding
    /// Python: img_height=28, img_width=42, pad_value=1.0 (白色)
    pub fn preprocess_for_tiny_cnn(&self, bgr: &Mat) -> Result<Tensor> {
        let h = bgr.rows();
        let w = bgr.cols();
        if h <= 0 || w <= 0 {
            anyhow::bail!("invalid image size {}x{}", w, h);
        }

        // 目标尺寸
        let img_h = 28;
        let img_w = 42;

        // 计算缩放比例，保持宽高比
        let scale = f32::min(img_w as f32 / w as f32, img_h as f32 / h as f32);
        let new_w = (w as f32 * scale) as i32;
        let new_h = (h as f32 * scale) as i32;

        // Resize
        let mut resized = Mat::default();
        imgproc::resize(
            bgr,
            &mut resized,
            Size::new(new_w, new_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        // 确保连续
        let resized = resized.try_clone()?;
        let data = resized.data_bytes()?; // BGRBGR...

        // 创建 padding 后的 tensor [1, 3, 28, 42]，填充值为1.0（白色）
        let mut input = Array4::<f32>::zeros((1, 3, img_h as usize, img_w as usize));

        // 计算居中偏移
        let y_offset = ((img_h - new_h) / 2) as usize;
        let x_offset = ((img_w - new_w) / 2) as usize;

        // 填充 resized 图像到中心
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let idx = (y * new_w as usize + x) * 3;
                let b = data[idx] as f32 / 255.0;
                let g = data[idx + 1] as f32 / 255.0;
                let r = data[idx + 2] as f32 / 255.0;

                // 注意：OpenCV的BGR顺序 → 模型输入RGB顺序
                input[[0, 0, y_offset + y, x_offset + x]] = r;
                input[[0, 1, y_offset + y, x_offset + x]] = g;
                input[[0, 2, y_offset + y, x_offset + x]] = b;
            }
        }

        Ok(input.into_tensor())
    }

    /// 分类：Softmax + Argmax，返回 (class_id, confidence)
    pub fn classify(&self, output: ArrayViewD<f32>) -> Result<(usize, f64)> {
        let shape = output.shape();
        if shape.len() != 2 || shape[0] != 1 {
            anyhow::bail!("invalid output shape {:?}, expected [1, 2]", shape);
        }

        let num_classes = shape[1];
        if num_classes != 2 {
            anyhow::bail!("expected 2 classes, got {}", num_classes);
        }

        // Softmax
        let logit0 = output[[0, 0]];
        let logit1 = output[[0, 1]];

        let max_logit = f32::max(logit0, logit1);
        let exp0 = (logit0 - max_logit).exp();
        let exp1 = (logit1 - max_logit).exp();
        let sum_exp = exp0 + exp1;

        let prob0 = exp0 / sum_exp;
        let prob1 = exp1 / sum_exp;

        // Argmax
        if prob0 > prob1 {
            Ok((0, prob0 as f64))
        } else {
            Ok((1, prob1 as f64))
        }
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


    fn detect_vx_box(&self, morphology: &Mat, target_area: f64) -> Result<Vec<ContourInfo>> {
        
        // 查找连通区域（外部轮廓）
        let mut contours = Vector::<Vector<Point2i>>::new();
        imgproc::find_contours(
            morphology,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point2i::new(0, 0),
        )?;

        let mut contour_infos = Vec::new();
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            
            // 第一步：只做最小面积占比的筛选
            if (area - target_area).abs() > target_area.min(area) * 0.3 {
                continue;
            }

            contour_infos.push(ContourInfo {
                points: contour,
                area
            });
        }

        Ok(contour_infos)
    }

    pub fn filter_vx_box(&self, boxes: &Vec<ContourInfo>) -> Result<Vec<Quad>> {

        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        let mut res = Vec::new();

        for contour_info in boxes {
            // 使用轮廓近似算法提取四边形
            let mut approx_curve = Vector::<Point2i>::new();
            let epsilon = ImageProcessingConfig::EPSILON_FACTOR * imgproc::arc_length(&contour_info.points, true)?;
            imgproc::approx_poly_dp(&contour_info.points, &mut approx_curve, epsilon, true)?;

            if approx_curve.len() != 4 {
                continue;
            }

            // 提取四个点
            let mut points_array: [Point2i; 4] = [
                Point2i::from(approx_curve.get(0)?),
                Point2i::from(approx_curve.get(1)?),
                Point2i::from(approx_curve.get(2)?),
                Point2i::from(approx_curve.get(3)?),
            ];

            // 确保四个点按顺时针方向排列，从左上角开始
            LocationModule::order_points(&mut points_array);
            res.push(
                Quad {
                    points: points_array
                }
            );
        }
        Ok(res)
    }

    /// 临时方法：基于积分图在小范围内精调坐标
    /// 用于训练数据采集时，框被红笔贯穿无法通过矩形检测的情况
    ///
    /// # Arguments
    /// * `integral` - 预计算的积分图（基于 closed 二值图）
    /// * `coor` - 原始坐标
    /// * `search_range` - 搜索范围，±search_range 像素（推荐 2）
    /// * `border_width` - 边框宽度，1 或 2 像素
    ///
    /// # Returns
    /// (精调后的坐标, 偏移量dx, 偏移量dy, 最佳得分)
    pub fn refine_coordinate(
        integral: &Mat,
        coor: &Coordinate,
        search_range: i32,
        border_width: i32,
    ) -> Result<(Coordinate, i32, i32, i64)> {
        let img_h = integral.rows();
        let img_w = integral.cols();

        let mut best_offset_x = 0;
        let mut best_offset_y = 0;
        let mut best_score = i64::MIN;

        // 在 ±search_range 范围内搜索
        for dy in -search_range..=search_range {
            for dx in -search_range..=search_range {
                let new_x = coor.x + dx;
                let new_y = coor.y + dy;

                // 边界检查
                if new_x < border_width || new_y < border_width
                    || new_x + coor.w + border_width >= img_w
                    || new_y + coor.h + border_width >= img_h
                {
                    continue;
                }

                // 计算边框像素和 = 外框像素和 - 内框像素和
                let outer_sum = Self::rect_sum_from_integral(
                    integral,
                    new_x - border_width,
                    new_y - border_width,
                    coor.w + 2 * border_width,
                    coor.h + 2 * border_width,
                )?;

                let inner_sum = Self::rect_sum_from_integral(
                    integral,
                    new_x + border_width,
                    new_y + border_width,
                    coor.w - 2 * border_width,
                    coor.h - 2 * border_width,
                )?;

                let border_sum = outer_sum - inner_sum;

                if border_sum > best_score {
                    best_score = border_sum;
                    best_offset_x = dx;
                    best_offset_y = dy;
                }
            }
        }

        Ok((
            Coordinate {
                x: coor.x + best_offset_x,
                y: coor.y + best_offset_y,
                w: coor.w,
                h: coor.h,
            },
            best_offset_x,
            best_offset_y,
            best_score,
        ))
    }

    /// 使用积分图计算矩形区域像素和
    /// 积分图公式：sum(x,y,w,h) = I[y+h][x+w] - I[y][x+w] - I[y+h][x] + I[y][x]
    fn rect_sum_from_integral(
        integral: &Mat,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
    ) -> Result<i64> {
        // 积分图是 CV_32S 或 CV_64F 类型，这里假设用 i32
        let x1 = x;
        let y1 = y;
        let x2 = x + w;
        let y2 = y + h;

        // 积分图的索引：积分图比原图大1
        let a = *integral.at_2d::<i32>(y1, x1)?;
        let b = *integral.at_2d::<i32>(y1, x2)?;
        let c = *integral.at_2d::<i32>(y2, x1)?;
        let d = *integral.at_2d::<i32>(y2, x2)?;

        Ok((d - b - c + a) as i64)
    }

    /// 批量精调所有划分框的坐标（用于训练数据采集）
    ///
    /// # Arguments
    /// * `closed` - 形态学处理后的二值图
    /// * `mobile_output` - 包含所有识别结果的输出
    /// * `search_range` - 搜索范围，±search_range 像素（推荐 2）
    /// * `border_width` - 边框宽度，1 或 2 像素
    pub fn refine_all_coordinates(
        &self,
        closed: &Mat,
        mobile_output: &mut MobileOutput,
        search_range: i32,
        border_width: i32,
    ) -> Result<()> {
        // 预计算积分图
        let integral = crate::myutils::image::integral_image(closed)?;

        println!("=== Refine Coordinates (search_range={}, border_width={}) ===", search_range, border_width);

        for (q_idx, rec_result) in mobile_output.rec_results.iter_mut().enumerate() {
            if rec_result.rec_type != RecType::Vx {
                continue;
            }

            for (opt_idx, rec_option) in rec_result.rec_options.iter_mut().enumerate() {
                let (refined, dx, dy, score) = Self::refine_coordinate(
                    &integral,
                    &rec_option.coordinate,
                    search_range,
                    border_width,
                )?;
                println!(
                    "  Q{}-Opt{}: offset=({:+}, {:+}), score={}, pos=({},{}) -> ({},{})",
                    q_idx, opt_idx, dx, dy, score,
                    rec_option.coordinate.x, rec_option.coordinate.y,
                    refined.x, refined.y
                );
                rec_option.coordinate = refined;
            }
        }

        println!("=== Refine Done ===");
        Ok(())
    }
}

