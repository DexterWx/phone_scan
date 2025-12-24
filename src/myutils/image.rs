use opencv::{
    calib3d, core::{AlgorithmHint, BORDER_CONSTANT, Mat, Point2f, Point2i, Size, Vector, Scalar,bitwise_or, in_range, no_array}, imgcodecs::{IMREAD_COLOR, imdecode, imread}, imgproc, prelude::*
};
use base64::{Engine as _, engine::general_purpose};
use anyhow::{Result, Context};
use crate::{config::{VxConfig, VxPageConfig}, models::{AssistLocation, Coordinate, ProcessedImage, Quad, TopologyFeatures}};
use crate::config::ImageProcessingConfig;

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
    processed_image: &ProcessedImage,
    transform_matrix: &Mat,
    target_w: i32,
    target_h: i32
) -> Result<ProcessedImage> {
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

    Ok(ProcessedImage {
        rgb: rgb_warped,
        gray: gray_warped,
        thresh: thresh_warped,
        closed: closed_warped,
    })
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
pub fn det_red(image: &Mat) -> Result<Mat> {
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

/// 提取拓扑特征
pub fn extract_topology_features(skeleton: &Mat) -> Result<TopologyFeatures> {
    let rows = skeleton.rows();
    let cols = skeleton.cols();

    let mut branch_points = 0;
    let mut end_points = 0;
    let mut isolated_points = 0;
    let mut total_pixels = 0;

    for i in 1..(rows - 1) {
        for j in 1..(cols - 1) {
            let center = *skeleton.at_2d::<u8>(i, j)?;
            if center == 0 {
                continue;
            }

            total_pixels += 1;

            // 统计 8 邻域非零像素数量
            let mut neighbor_count = 0;
            for di in -1..=1 {
                for dj in -1..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let neighbor = *skeleton.at_2d::<u8>(i + di, j + dj)?;
                    if neighbor != 0 {
                        neighbor_count += 1;
                    }
                }
            }

            match neighbor_count {
                0 => isolated_points += 1,
                1 => end_points += 1,
                n if n >= 3 => branch_points += 1,
                _ => {} // 2 个邻居：普通骨架点
            }
        }
    }

    Ok(TopologyFeatures {
        branch_points,
        end_points,
        isolated_points,
        total_pixels,
        image_width: cols as usize,
        image_height: rows as usize,
    })
}
