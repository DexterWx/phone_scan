use opencv::{
    calib3d, core::{AlgorithmHint, BORDER_CONSTANT, CV_8UC1, CV_32S, CV_64F, Mat, Point2f, Point2i, Scalar, Size, Vector, bitwise_or, in_range, no_array, ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE, MatTraitConst}, imgcodecs::{IMREAD_COLOR, imdecode, imread}, imgproc, prelude::*
};
use base64::{Engine as _, engine::general_purpose};
use anyhow::{Result, Context, anyhow};
use crate::{config::{VxConfig, VxPageConfig}, models::{ConnectFeatures, Coordinate, ProcessedImage}};
use crate::config::ImageProcessingConfig;

/// 解码 NV12 数据为 BGR Mat
/// - nv12_data: NV12 原始数据
/// - width: 图像宽度
/// - height: 图像高度
/// - rotation: 旋转角度 (0, 90, 180, 270)
pub fn decode_nv12(nv12_data: &[u8], width: i32, height: i32, rotation: i32) -> Result<Mat> {
    // 验证数据长度
    let expected_size = (width * height * 3 / 2) as usize;
    if nv12_data.len() != expected_size {
        return Err(anyhow!(
            "NV12 数据长度不匹配: 期望 {} 字节, 实际 {} 字节",
            expected_size,
            nv12_data.len()
        ));
    }

    // 创建 NV12 Mat (Y + UV 平面)
    let nv12_height = height * 3 / 2;
    let nv12_mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            nv12_height,
            width,
            CV_8UC1,
            nv12_data.as_ptr() as *mut std::ffi::c_void,
            width as usize,
        ).context("创建 NV12 Mat 失败")?
    };

    // 转换 NV12 到 BGR
    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(
        &nv12_mat,
        &mut bgr_mat,
        imgproc::COLOR_YUV2BGR_NV12,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    ).context("NV12 转 BGR 失败")?;

    // 根据旋转角度旋转图像
    if rotation == 0 {
        Ok(bgr_mat)
    } else {
        let mut rotated = Mat::default();
        let rotate_code = match rotation {
            90 => ROTATE_90_CLOCKWISE,
            180 => ROTATE_180,
            270 => ROTATE_90_COUNTERCLOCKWISE,
            _ => return Err(anyhow!("不支持的旋转角度: {}", rotation)),
        };
        opencv::core::rotate(&bgr_mat, &mut rotated, rotate_code)
            .context("旋转图像失败")?;
        Ok(rotated)
    }
}

/// 计算图像的拉普拉斯方差（清晰度评分）
/// 返回值越大，图像越清晰
pub fn calc_laplacian_variance(image: &Mat) -> Result<f64> {
    // 转灰度图
    let mut gray = Mat::default();
    if image.channels() == 3 {
        imgproc::cvt_color(
            image,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        ).context("转灰度图失败")?;
    } else {
        gray = image.clone();
    }

    // 计算拉普拉斯算子
    let mut laplacian = Mat::default();
    imgproc::laplacian(
        &gray,
        &mut laplacian,
        CV_64F,
        3,  // ksize
        1.0,  // scale
        0.0,  // delta
        opencv::core::BORDER_DEFAULT,
    ).context("计算拉普拉斯失败")?;

    // 计算方差
    let mut mean = Mat::default();
    let mut stddev = Mat::default();
    opencv::core::mean_std_dev(&laplacian, &mut mean, &mut stddev, &no_array())
        .context("计算方差失败")?;

    let variance = *stddev.at::<f64>(0)? * *stddev.at::<f64>(0)?;
    Ok(variance)
}

/// 从 Mat 引用数组中选择最清晰的图片
/// 返回最清晰图片的索引
pub fn select_clearest_image(images: &[&Mat]) -> Result<usize> {
    if images.is_empty() {
        anyhow::bail!("图片数组为空");
    }

    let mut max_variance = f64::MIN;
    let mut clearest_idx = 0;

    for (idx, image) in images.iter().enumerate() {
        let variance = calc_laplacian_variance(image)?;
        if variance > max_variance {
            max_variance = variance;
            clearest_idx = idx;
        }
    }

    Ok(clearest_idx)
}

/// 从 Mat 数组中选择最清晰的图片（所有权版本）
/// 返回最清晰图片的索引
pub fn select_clearest_image_owned(images: &[Mat]) -> Result<usize> {
    if images.is_empty() {
        anyhow::bail!("图片数组为空");
    }

    let mut max_variance = f64::MIN;
    let mut clearest_idx = 0;

    for (idx, image) in images.iter().enumerate() {
        let variance = calc_laplacian_variance(image)?;
        if variance > max_variance {
            max_variance = variance;
            clearest_idx = idx;
        }
    }

    Ok(clearest_idx)
}

/// 从文件夹读取所有图片
/// 支持 jpg, jpeg, png 格式
/// 返回 (图片数组, 文件路径数组)
pub fn read_images_from_dir(dir_path: &str) -> Result<(Vec<Mat>, Vec<String>)> {
    use std::fs;
    use std::path::Path;

    let dir = Path::new(dir_path);
    if !dir.exists() {
        anyhow::bail!("文件夹不存在: {}", dir_path);
    }

    let mut images = Vec::new();
    let mut paths = Vec::new();

    let entries = fs::read_dir(dir).context("读取文件夹失败")?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            if ext_lower == "jpg" || ext_lower == "jpeg" || ext_lower == "png" {
                let image = imread(&path.to_string_lossy(), IMREAD_COLOR)?;
                if !image.empty() {
                    paths.push(path.to_string_lossy().to_string());
                    images.push(image);
                }
            }
        }
    }

    Ok((images, paths))
}

pub fn read_image(input: &String) -> Result<Mat> {
    // 判断输入是文件路径还是base64字符串
    // 根据需求：长度超过200认为是base64，否则是路径
    if input.len() > 200 {
        // 处理base64字符串
        let image_data = general_purpose::STANDARD
            .decode(input)
            .context("Base64 解码失败")?;
        
        // 将字节数组转换为 OpenCV 的 Vector<u8>
        let image_vector = opencv::core::Vector::<u8>::from_slice(&image_data);
        
        // 使用 imdecode 将字节数据解码为 Mat 对象
        let mat = imdecode(&image_vector, IMREAD_COLOR)
            .context("字节流 解码失败")?;
        
        // 检查图片是否为空
        if mat.empty() {
            anyhow::bail!("解码成功，但图片为空");
        }
        
        Ok(mat)
    } else {
        // 处理文件路径
        let mat = imread(input, IMREAD_COLOR)
            .context("读取图片文件失败")?;
        
        // 检查图片是否为空
        if mat.empty() {
            anyhow::bail!("读取成功，但图片为空");
        }
        
        Ok(mat)
    }
}

pub fn resize_image(image: &Mat, target_width: i32) -> Result<Mat> {
    let mut resized = Mat::default();
    let scale = target_width as f64 / image.cols() as f64;
    imgproc::resize(image, &mut resized, Size::new(target_width, -1), scale, scale, imgproc::INTER_LINEAR)?;
    Ok(resized)
}

/// 图片预处理：灰度化、高斯模糊、二值化、形态学操作
pub fn process_image(image: &Mat, target_width: i32) -> Result<ProcessedImage> {
    // 0. 图片统一到宽度
    let resized = resize_image(image, target_width)?;

    // 1. 灰度化
    let mut gray = Mat::default();
    imgproc::cvt_color(&resized, &mut gray, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // 2. 高斯模糊
    let mut blur = Mat::default();
    let kernel_size = Size::new(ImageProcessingConfig::GAUSSIAN_KERNEL_SIZE, ImageProcessingConfig::GAUSSIAN_KERNEL_SIZE);
    imgproc::gaussian_blur(&gray, &mut blur, kernel_size, ImageProcessingConfig::GAUSSIAN_SIGMA, ImageProcessingConfig::GAUSSIAN_SIGMA, opencv::core::BORDER_DEFAULT, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // 3. 自适应阈值二值化
    let mut thresh = Mat::default();
    imgproc::adaptive_threshold(
        &blur,
        &mut thresh,
        255.0,
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
        imgproc::THRESH_BINARY_INV,
        ImageProcessingConfig::BLOCK_SIZE,
        ImageProcessingConfig::C as f64,
    )?;

    // 4. 形态学闭操作
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(ImageProcessingConfig::MORPH_KERNEL, ImageProcessingConfig::MORPH_KERNEL),
        Point2i::new(-1, -1),
    )?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &thresh,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point2i::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    )?;

    Ok(ProcessedImage {
        rgb: resized,
        gray,
        thresh,
        closed,
    })
}

/// 计算透视变换矩阵
/// detected_quad: 检测到的四边形（实际图片中的四边形）
/// target_rect: 目标矩形区域（xywh格式）
pub fn get_perspective_transform_matrix_with_boundary(
    src_points: &Vector<Point2f>,
    target_points: &Vector<Point2f>,
) -> Result<Mat> {

    // 计算透视变换矩阵
    let transform_matrix = imgproc::get_perspective_transform(&src_points, &target_points, 0)
        .context("计算透视变换矩阵失败")?;

    Ok(transform_matrix)
}

pub fn get_perspective_transform_matrix_with_points(
    src_points: &Vector<Point2f>,
    target_points: &Vector<Point2f>,
) -> Result<Mat> {

    // 输出 mask，用于查看哪些点是 inlier
    let mut mask = Mat::default();

    // 计算透视矩阵
    let transform_matrix = calib3d::find_homography(
        src_points,
        target_points,
        &mut mask,
        0,  // 也可用 calib3d::LMEDS 或 RANSAC
        3.0,               // ransac_reproj_threshold (像素)
    )
    .context("使用 RANSAC 计算透视变换矩阵失败")?;

    Ok(transform_matrix)
}

pub fn get_points_from_coordinate(coordinate: &Coordinate) -> Vector<Point2f> {
    let points = Vector::<Point2f>::from_slice(&[
        Point2f::new(coordinate.x as f32, coordinate.y as f32),                                    // 左上角
        Point2f::new((coordinate.x + coordinate.w) as f32, coordinate.y as f32),                 // 右上角
        Point2f::new((coordinate.x + coordinate.w) as f32, (coordinate.y + coordinate.h) as f32), // 右下角
        Point2f::new(coordinate.x as f32, (coordinate.y + coordinate.h) as f32),                 // 左下角
    ]);
    points
}

pub fn get_points_from_coordinates(coors: &Vec<Coordinate>) -> Vector<Point2f> {
    let mut points = Vector::<Point2f>::new();
    for coor in coors {
        let center_x = (coor.x + coor.w / 2) as f32;
        let center_y = (coor.y + coor.h / 2) as f32;
        points.push(Point2f::new(center_x, center_y));
    }
    points
}


/// 透视变换
pub fn pers_trans_image(
    processed_image: &mut ProcessedImage,
    transform_matrix: &Mat,
    target_w: i32,
    target_h: i32
) -> Result<()> {
    // 对所有图像应用透视变换
    let mut rgb_warped = Mat::default();
    imgproc::warp_perspective(
        &processed_image.rgb,
        &mut rgb_warped,
        &transform_matrix,
        Size::new(target_w, target_h),
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    ).context("应用透视变换到RGB图失败")?;

    let mut gray_warped = Mat::default();
    imgproc::warp_perspective(
        &processed_image.gray,
        &mut gray_warped,
        &transform_matrix,
        Size::new(target_w, target_h),
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    ).context("应用透视变换到灰度图失败")?;

    let mut thresh_warped = Mat::default();
    imgproc::warp_perspective(
        &processed_image.thresh,
        &mut thresh_warped,
        &transform_matrix,
        Size::new(target_w, target_h),
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    ).context("应用透视变换到二值图失败")?;

    let mut closed_warped = Mat::default();
    imgproc::warp_perspective(
        &processed_image.closed,
        &mut closed_warped,
        &transform_matrix,
        Size::new(target_w, target_h),
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    ).context("应用透视变换到形态学处理图失败")?;

    processed_image.rgb = rgb_warped;
    processed_image.gray = gray_warped;
    processed_image.thresh = thresh_warped;
    processed_image.closed = closed_warped;

    Ok(())
}


/// 计算积分图
pub fn integral_image(image: &Mat) -> Result<Mat> {
    // 检查输入图像是否为空
    if image.empty() {
        anyhow::bail!("输入图像为空");
    }
    
    // 创建输出积分图
    let mut integral = Mat::default();
    
    // 使用OpenCV内置的积分图函数（简化版本，只需要3个参数）
    imgproc::integral(
        image,
        &mut integral,
        -1  // sdepth: 积分图的数据深度（-1表示与输入图像相同）
    ).context("计算积分图失败")?;
    
    Ok(integral)
}


pub fn merge_coordinates(coordinates: &Vec<Coordinate>, extend_size_w: i32, extend_size_h: i32) -> Coordinate {
    let mut x = coordinates.iter().map(|c| c.x).min().unwrap();
    let mut y = coordinates.iter().map(|c| c.y).min().unwrap();
    let mut w = coordinates.iter().map(|c| c.x + c.w).max().unwrap() - x;
    let mut h = coordinates.iter().map(|c| c.y + c.h).max().unwrap() - y;

    x-= extend_size_w;
    y-= extend_size_h;
    w+= extend_size_w*2;
    h+= extend_size_h*2;

    Coordinate { x, y, w, h }   
}

pub fn crop_image(image: &Mat, coordinate: &Coordinate) -> Result<Mat> {
    let cropimage = Mat::roi(
        image,
        opencv::core::Rect::new(
            coordinate.x.max(0),
            coordinate.y.max(0),
            coordinate.w.min(image.cols() - coordinate.x.max(0)),
            coordinate.h.min(image.rows() - coordinate.y.max(0))
        )
    )?;
    let cropimage = cropimage.try_clone()?;
    Ok(cropimage)
}

// 输出二值图
pub fn det_red_hsv(image: &Mat) -> Result<Mat> {
    // 1. BGR -> HSV
    let mut hsv = Mat::default();
    imgproc::cvt_color(image, &mut hsv, imgproc::COLOR_BGR2HSV, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // 2. 红色 mask（两段）
    let mut mask1 = Mat::default();
    let mut mask2 = Mat::default();

    let low1  = VxPageConfig::hsv_lower1_bound();
    let high1 = VxPageConfig::hsv_upper1_bound();

    let low2  = VxPageConfig::hsv_lower2_bound();
    let high2 = VxPageConfig::hsv_upper2_bound();
    // 低红区间
    in_range(&hsv, &low1, &high1, &mut mask1)?;
    // 高红区间
    in_range(&hsv, &low2, &high2, &mut mask2)?;

    // 合并 mask：红色区域是 255（白色），背景是 0（黑色）
    let mut red_mask = Mat::default();
    bitwise_or(&mask1, &mask2, &mut red_mask, &no_array())?;

    Ok(red_mask)
}

pub fn det_red_lab(image: &Mat) -> Result<Mat> {
    // 1. BGR -> Lab
    let mut lab = Mat::default();
    imgproc::cvt_color(
        image,
        &mut lab,
        imgproc::COLOR_BGR2Lab,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // 2. split Lab channels
    let mut channels = Vector::<Mat>::new();
    opencv::core::split(&lab, &mut channels)?;

    // a 通道
    let a_channel = channels.get(1)?;

    // 3. CLAHE
    let mut enhanced_a = Mat::default();
    let mut clahe = imgproc::create_clahe(3.0, Size::new(8, 8))?;
    clahe.apply(&a_channel, &mut enhanced_a)?;

    // 4. Threshold -> binary mask
    let mut mask = Mat::default();
    imgproc::threshold(
        &enhanced_a,
        &mut mask,
        VxPageConfig::lab_a_threshold(),
        255.0,
        imgproc::THRESH_BINARY,
    )?;

    Ok(mask)
}

pub fn sum_pixel(integral_image: &Mat, coordinate: &Coordinate) -> Result<f64> {
    let x1 = coordinate.x as i32; // 左上角x坐标
    let y1 = coordinate.y as i32; // 左上角y坐标
    let x2 = x1 + coordinate.w as i32; // 右下角x坐标
    let y2 = y1 + coordinate.h as i32; // 右下角y坐标

    // 从积分图获取四个角的值
    let a = *integral_image.at_2d::<i32>(y1, x1)? as f64; // 左上角上方
    let b = *integral_image.at_2d::<i32>(y1, x2)? as f64;     // 右上角上方
    let c = *integral_image.at_2d::<i32>(y2, x1)? as f64;     // 左下角左侧
    let d = *integral_image.at_2d::<i32>(y2, x2)? as f64;         // 右下角

    // 使用积分图计算区域和
    let sum = d - b - c + a;

    Ok(sum)
}

/// Zhang-Suen 细化算法（骨架化）
pub fn zhang_suen_thinning(image: &Mat) -> Result<Mat> {
    let mut skeleton = image.clone();
    let rows = skeleton.rows();
    let cols = skeleton.cols();

    let mut changed = true;
    let mut iter_count = 0;
    let max_iterations = 100;

    while changed && iter_count < max_iterations {
        changed = false;
        iter_count += 1;

        // 两个子迭代
        for step in 0..2 {
            let mut to_delete = Vec::new();

            for i in 1..(rows - 1) {
                for j in 1..(cols - 1) {
                    if *skeleton.at_2d::<u8>(i, j)? == 0 {
                        continue;
                    }

                    // 获取 8 邻域 (按顺时针从上方开始: P2, P3, P4, P5, P6, P7, P8, P9)
                    let p2 = *skeleton.at_2d::<u8>(i - 1, j)?;
                    let p3 = *skeleton.at_2d::<u8>(i - 1, j + 1)?;
                    let p4 = *skeleton.at_2d::<u8>(i, j + 1)?;
                    let p5 = *skeleton.at_2d::<u8>(i + 1, j + 1)?;
                    let p6 = *skeleton.at_2d::<u8>(i + 1, j)?;
                    let p7 = *skeleton.at_2d::<u8>(i + 1, j - 1)?;
                    let p8 = *skeleton.at_2d::<u8>(i, j - 1)?;
                    let p9 = *skeleton.at_2d::<u8>(i - 1, j - 1)?;

                    let neighbors = [p2, p3, p4, p5, p6, p7, p8, p9];

                    // 条件 1：2 <= B(P1) <= 6（非零邻居数）
                    let b = neighbors.iter().filter(|&&p| p != 0).count();
                    if b < 2 || b > 6 {
                        continue;
                    }

                    // 条件 2：A(P1) = 1（0→1 转换次数）
                    let mut a = 0;
                    for k in 0..8 {
                        let curr = if neighbors[k] != 0 { 1 } else { 0 };
                        let next = if neighbors[(k + 1) % 8] != 0 { 1 } else { 0 };
                        if curr == 0 && next == 1 {
                            a += 1;
                        }
                    }
                    if a != 1 {
                        continue;
                    }

                    // 条件 3 和 4：根据子迭代步骤不同
                    let condition_met = if step == 0 {
                        // 步骤 1
                        (p2 == 0 || p4 == 0 || p6 == 0) &&
                        (p4 == 0 || p6 == 0 || p8 == 0)
                    } else {
                        // 步骤 2
                        (p2 == 0 || p4 == 0 || p8 == 0) &&
                        (p2 == 0 || p6 == 0 || p8 == 0)
                    };

                    if condition_met {
                        to_delete.push((i, j));
                        changed = true;
                    }
                }
            }

            // 删除标记的像素
            for (i, j) in to_delete {
                *skeleton.at_2d_mut::<u8>(i, j)? = 0;
            }
        }
    }

    Ok(skeleton)
}

/// 判断两个位置是否在8邻域中相邻（切比雪夫距离<=1）
fn are_neighbors_adjacent(pos1: (i32, i32), pos2: (i32, i32)) -> bool {
    let dx = (pos1.0 - pos2.0).abs();
    let dy = (pos1.1 - pos2.1).abs();
    dx <= 1 && dy <= 1 && (dx + dy) > 0
}

/// 骨架精简算法：移除冗余的斜线中间点
///
/// 删除条件：
/// 1. 4邻域（上下左右）恰好有2个非零点
/// 2. 这2个点在8邻域中相邻（对角相邻）
/// 3. 立即删除，单轮扫描
pub fn refine_skeleton(skeleton: &Mat) -> Result<Mat> {
    let mut result = skeleton.clone();
    let rows = result.rows();
    let cols = result.cols();

    // 单轮扫描所有非边界点
    for i in 1..(rows - 1) {
        for j in 1..(cols - 1) {
            // 跳过黑色像素
            if *result.at_2d::<u8>(i, j)? == 0 {
                continue;
            }

            // 获取4邻域（上下左右）的非零点位置
            let mut four_neighbors = Vec::new();

            // 上 (i-1, j)
            if *result.at_2d::<u8>(i - 1, j)? != 0 {
                four_neighbors.push((i - 1, j));
            }
            // 左 (i, j-1)
            if *result.at_2d::<u8>(i, j - 1)? != 0 {
                four_neighbors.push((i, j - 1));
            }
            // 右 (i, j+1)
            if *result.at_2d::<u8>(i, j + 1)? != 0 {
                four_neighbors.push((i, j + 1));
            }
            // 下 (i+1, j)
            if *result.at_2d::<u8>(i + 1, j)? != 0 {
                four_neighbors.push((i + 1, j));
            }

            // 条件1：4邻域恰好有2个点
            if four_neighbors.len() != 2 {
                continue;
            }

            // 条件2：这2个点在8邻域中相邻
            let pos1 = four_neighbors[0];
            let pos2 = four_neighbors[1];

            if are_neighbors_adjacent(pos1, pos2) {
                // 立即删除当前点
                *result.at_2d_mut::<u8>(i, j)? = 0;
            }
        }
    }

    Ok(result)
}

/// 预处理二值图（输入已经是红色为白色的二值图）
pub fn preprocess_vx_line<T: VxConfig>(image: &Mat) -> Result<Mat> {
    // 1. 形态学闭操作：连接断裂的线条
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(T::preprocess_close_kernel_size(), T::preprocess_close_kernel_size()),
        Point2i::new(-1, -1),
    )?;

    let mut closed = Mat::default();
    imgproc::morphology_ex(
        image,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point2i::new(-1, -1),
        1,
        BORDER_CONSTANT,
        Scalar::default(),
    )?;

    // 2. 形态学开操作：去除小噪点
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(T::preprocess_open_kernel_size(), T::preprocess_open_kernel_size()),
        Point2i::new(-1, -1),
    )?;

    let mut opened = Mat::default();
    imgproc::morphology_ex(
        &closed,
        &mut opened,
        imgproc::MORPH_OPEN,
        &kernel,
        Point2i::new(-1, -1),
        1,
        BORDER_CONSTANT,
        Scalar::default(),
    )?;

    Ok(opened)
}


// /// 从骨架图中提取拓扑特征：
// /// - 每个连通域单独分析
// /// - 统计点数、端点数
// /// - 判断是否存在“真实分支”
// pub fn extract_topology_features(skeleton: &Mat) -> opencv::Result<TopologyFeatures> {
//     let mut labels = Mat::default();
//     let mut stats = Mat::default();
//     let mut centroids = Mat::default();

//     // 8 邻域连通域标记
//     let num = imgproc::connected_components_with_stats(
//         skeleton,
//         &mut labels,
//         &mut stats,
//         &mut centroids,
//         8,
//         CV_32S,
//     )?;

//     let mut features = TopologyFeatures::default();

//     for label in 1..num {
//         let points_count =
//             *stats.at_2d::<i32>(label, imgproc::CC_STAT_AREA)? as usize;

//         if points_count < VxPageConfig::component_min_points() {
//             continue;
//         }
//         // 直接从 stats 里拿 bounding box
//         let mut x = *stats.at_2d::<i32>(label, imgproc::CC_STAT_LEFT)?;
//         let mut y = *stats.at_2d::<i32>(label, imgproc::CC_STAT_TOP)?;
//         let mut w = *stats.at_2d::<i32>(label, imgproc::CC_STAT_WIDTH)?;
//         let mut h = *stats.at_2d::<i32>(label, imgproc::CC_STAT_HEIGHT)?;

//         if w <= 0 || h <= 0 {
//             continue;
//         }

//         // 向外扩 1 像素（注意边界）
//         let expand = 2;

//         let x0 = (x - expand).max(0);
//         let y0 = (y - expand).max(0);

//         let x1 = (x + w + expand).min(skeleton.cols());
//         let y1 = (y + h + expand).min(skeleton.rows());

//         x = x0;
//         y = y0;
//         w = x1 - x0;
//         h = y1 - y0;

//         // 在 bbox 内构建 component（而不是整张图）
//         let mut component = Mat::zeros(h, w, skeleton.typ())?.to_mat()?;

//         for yy in 0..h {
//             for xx in 0..w {
//                 let ly = y + yy;
//                 let lx = x + xx;
//                 if *labels.at_2d::<i32>(ly, lx)? == label {
//                     *component.at_2d_mut::<u8>(yy, xx)? = 255;
//                 }
//             }
//         }
        
//         let end_points = count_endpoints(&component)?;
//         let has_branch = has_true_branch(&component, 3)?;
//         let curvature_score = pca_line_error(&component)?;

//         features.connects.push(ConnectFeatures {
//             points_count,
//             has_branch,
//             end_points,
//             curvature_score,
//         });
//     }

//     Ok(features)
// }

/// 基于 PCA 的直线拟合误差
/// 返回：平均点到主轴的垂直距离
fn pca_line_error(component: &Mat) -> opencv::Result<f64> {
    let rows = component.rows();
    let cols = component.cols();

    let mut points: Vec<(f64, f64)> = Vec::new();

    // 收集前景点
    for y in 0..rows {
        for x in 0..cols {
            if *component.at_2d::<u8>(y, x)? > 0 {
                points.push((x as f64, y as f64));
            }
        }
    }

    let n = points.len();
    if n < VxPageConfig::pac_min_points_count() {
        return Ok(f64::MAX);
    }

    // 计算均值
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for (x, y) in &points {
        mean_x += x;
        mean_y += y;
    }
    mean_x /= n as f64;
    mean_y /= n as f64;

    // 协方差矩阵
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;

    for (x, y) in &points {
        let dx = x - mean_x;
        let dy = y - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }

    sxx /= n as f64;
    sxy /= n as f64;
    syy /= n as f64;

    // PCA 主方向（最大特征值对应向量）
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let dir_x = theta.cos();
    let dir_y = theta.sin();

    // 计算平均垂直距离
    let mut dist_sum = 0.0;
    for (x, y) in &points {
        let dx = x - mean_x;
        let dy = y - mean_y;
        // 点到直线的垂直距离
        let dist = (dx * dir_y - dy * dir_x).abs();
        dist_sum += dist;
    }

    Ok(dist_sum / n as f64)
}


/// 判断一个连通域中是否存在“真实分支”
///
/// 方法：
/// 1. 计算原始端点数
/// 2. 枚举每个前景像素作为 block 中心
/// 3. 删除 block_size × block_size 区域
/// 4. 若端点数增加 ≥ 3，则认为该位置是分支节点
fn has_true_branch(component: &Mat, block_size: i32) -> opencv::Result<bool> {
    let base_endpoints = count_endpoints(component)?;

    let rows = component.rows();
    let cols = component.cols();
    let half = block_size / 2;

    for y in 0..rows {
        for x in 0..cols {
            // 只在前景像素上尝试
            if *component.at_2d::<u8>(y, x)? == 0 {
                continue;
            }

            // block 左上角、右下角
            let y0 = y - half;
            let x0 = x - half;
            let y1 = y0 + block_size;
            let x1 = x0 + block_size;

            // 越界直接跳过
            if y0 < 0 || x0 < 0 || y1 > rows || x1 > cols {
                continue;
            }

            // 快速判断：该 block 内是否真的有前景
            let mut has_fg = false;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    if *component.at_2d::<u8>(yy, xx)? > 0 {
                        has_fg = true;
                        break;
                    }
                }
                if has_fg {
                    break;
                }
            }
            if !has_fg {
                continue;
            }

            // 复制一份，模拟“删除该 block”
            let mut tmp = component.clone();
            for yy in y0..y1 {
                for xx in x0..x1 {
                    *tmp.at_2d_mut::<u8>(yy, xx)? = 0;
                }
            }

            // 删除后重新计算端点
            let new_endpoints = count_endpoints(&tmp)?;

            // 若端点数显著增加，判定为分支
            if new_endpoints as i32 - base_endpoints as i32 >= 3 {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

/// 顺时针 8 邻域（Zhang–Suen 标准顺序）
///
/// p0 p1 p2
/// p7  x p3
/// p6 p5 p4
const CN_NEIGHBORS: [(i32, i32); 8] = [
    (-1,  0), // N
    (-1,  1), // NE
    ( 0,  1), // E
    ( 1,  1), // SE
    ( 1,  0), // S
    ( 1, -1), // SW
    ( 0, -1), // W
    (-1, -1), // NW
];

/// 计算某个像素点的 CN（Crossing Number）
///
/// CN 定义：
/// CN = 1/2 * Σ |p_i - p_{i+1}|
///
/// 语义：
/// - CN == 1 → 端点
/// - CN == 2 → 普通连线点
/// - CN >= 3 → 分支 / 交叉点
fn calc_cn(mat: &Mat, y: i32, x: i32) -> opencv::Result<i32> {
    let mut p = [0i32; 8];

    // 将 8 邻域映射成 0/1
    for (i, (dy, dx)) in CN_NEIGHBORS.iter().enumerate() {
        let ny = y + dy;
        let nx = x + dx;
        if *mat.at_2d::<u8>(ny, nx)? > 0 {
            p[i] = 1;
        }
    }

    // 计算 crossing number
    let mut sum = 0;
    for i in 0..8 {
        let next = (i + 1) % 8;
        sum += (p[i] - p[next]).abs();
    }

    Ok(sum / 2)
}

/// 统计端点数量
///
/// 判定规则：
/// - 前景像素
/// - CN == 1
///
/// 注意：
/// - 显式跳过边界，保证 8 邻域访问安全
fn count_endpoints(mat: &Mat) -> opencv::Result<usize> {
    let rows = mat.rows();
    let cols = mat.cols();
    let mut count = 0;

    for y in 1..rows - 1 {
        for x in 1..cols - 1 {
            if *mat.at_2d::<u8>(y, x)? == 0 {
                continue;
            }

            let cn = calc_cn(mat, y, x)?;
            if cn == 1 {
                count += 1;
            }
        }
    }

    Ok(count)
}
