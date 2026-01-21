use opencv::{
    calib3d, core::{AlgorithmHint, BORDER_CONSTANT, CV_8UC1, CV_64F, Mat, Point2f, Point2i, Scalar, Size, Vector, bitwise_or, in_range, no_array, ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE, MatTraitConst}, imgcodecs::{IMREAD_COLOR, imdecode, imread}, imgproc, prelude::*
};
use base64::{Engine as _, engine::general_purpose};
use anyhow::{Result, Context, anyhow};
use rayon::prelude::*;
use crate::{config::{VxConfig, VxPageConfig}, models::{Coordinate, ProcessedImage}};
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

    if bgr_mat.empty() {
        return Err(anyhow!("NV12 转 BGR 后图像为空"));
    }

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

        if rotated.empty() {
            return Err(anyhow!("旋转后图像为空, rotation={}", rotation));
        }
        Ok(rotated)
    }
}

/// 将 BGR Mat 编码为 NV12 数据（用于测试）
/// 返回 (nv12_data, width, height)
pub fn encode_nv12(bgr: &Mat) -> Result<(Vec<u8>, i32, i32)> {
    let width = bgr.cols();
    let height = bgr.rows();

    // BGR 转 YUV (I420/YV12 格式，再转 NV12)
    let mut yuv_i420 = Mat::default();
    imgproc::cvt_color(
        bgr,
        &mut yuv_i420,
        imgproc::COLOR_BGR2YUV_I420,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    ).context("BGR 转 YUV_I420 失败")?;

    // I420 布局: Y (w*h) + U (w/2 * h/2) + V (w/2 * h/2)
    // NV12 布局: Y (w*h) + UV 交错 (w/2 * h/2 * 2)
    let y_size = (width * height) as usize;
    let uv_size = (width * height / 4) as usize;
    let nv12_size = y_size + uv_size * 2;

    let mut nv12_data = vec![0u8; nv12_size];

    // 获取 I420 数据
    let i420_data = yuv_i420.data_bytes().context("获取 YUV 数据失败")?;

    // 复制 Y 平面
    nv12_data[..y_size].copy_from_slice(&i420_data[..y_size]);

    // 交错 U 和 V 平面
    let u_plane = &i420_data[y_size..y_size + uv_size];
    let v_plane = &i420_data[y_size + uv_size..y_size + uv_size * 2];

    for i in 0..uv_size {
        nv12_data[y_size + i * 2] = u_plane[i];
        nv12_data[y_size + i * 2 + 1] = v_plane[i];
    }

    Ok((nv12_data, width, height))
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

/// 从 NV12 图片批次数据中解码并选择最清晰的图片（多线程版本）
///
/// 参数:
/// - data: 所有图片拼接后的连续内存
/// - widths: 宽度数组
/// - heights: 高度数组
/// - rotations: 旋转角度数组 (0, 90, 180, 270)
/// - lens: 每张图片的字节长度数组
///
/// 返回: (最清晰图片的 Mat, 图片索引)
pub fn decode_nv12_batch_and_select_clearest(
    data: &[u8],
    widths: &[u32],
    heights: &[u32],
    rotations: &[u8],
    lens: &[u32],
) -> Result<(Mat, usize)> {
    let count = lens.len();
    if count == 0 {
        anyhow::bail!("图片数量为 0");
    }

    // 预计算每张图片的偏移量
    let mut offsets = Vec::with_capacity(count);
    let mut offset: usize = 0;
    for &len in lens {
        offsets.push(offset);
        offset += len as usize;
    }

    // 并行解码并计算清晰度
    let results: Vec<Result<(Mat, f64)>> = (0..count)
        .into_par_iter()
        .map(|i| {
            let width = widths[i] as i32;
            let height = heights[i] as i32;
            let rotation = rotations[i] as i32;
            let len = lens[i] as usize;
            let start = offsets[i];

            let image_data = &data[start..start + len];
            let mat = decode_nv12(image_data, width, height, rotation)
                .with_context(|| format!("解码第 {} 张图片失败", i + 1))?;
            let variance = calc_laplacian_variance(&mat)
                .with_context(|| format!("计算第 {} 张图片清晰度失败", i + 1))?;
            Ok((mat, variance))
        })
        .collect();

    // 检查错误并找出最清晰的图片
    let mut max_variance = f64::MIN;
    let mut clearest_idx = 0;
    let mut decoded_images = Vec::with_capacity(count);

    for (i, result) in results.into_iter().enumerate() {
        let (mat, variance) = result?;
        if variance > max_variance {
            max_variance = variance;
            clearest_idx = i;
        }
        decoded_images.push(mat);
    }

    // 返回最清晰图片和索引
    Ok((decoded_images.swap_remove(clearest_idx), clearest_idx))
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