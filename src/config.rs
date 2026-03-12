use opencv::core::Scalar;


/// 图像处理配置参数
pub struct ImageProcessingConfig;

impl ImageProcessingConfig {
    /// 高斯模糊核大小
    pub const GAUSSIAN_KERNEL_SIZE: i32 = 5;
    
    /// 高斯模糊sigma值
    pub const GAUSSIAN_SIGMA: f64 = 0.0;

    /// 统一输入图像的宽度
    pub const TARGET_WIDTH_A4: i32 = 2400;
    pub const TARGET_WIDTH_A3: i32 = 4000;

    /// 目标图片缩放比例
    pub const PAPER_SCAN_TARGET_SCALE: f64 = 2.0;

    /// 自适应阈值的块大小
    pub const BLOCK_SIZE: i32 = 51;
    
    /// 自适应阈值的常数
    pub const C: i32 = 5;
    
    /// 形态学操作的核大小
    pub const MORPH_KERNEL: i32 = 3;
    
    /// 多边形逼近的epsilon因子
    pub const EPSILON_FACTOR: f64 = 0.015;
    
    /// 最小面积占比
    pub const MIN_AREA_RATIO: f64 = 0.25;
    
    /// 边界惩罚系数
    pub const MARGIN_PENALTY: f64 = 50.0;

    /// 变换后，从边界向外拓展的距离
    pub const BOUNDARY_EXTEND_SIZE: i32 = 40;

    /// 边界贴合距离
    pub const BOUNDARY_PENALTY: f64 = 11.0;
}

/// 辅助定位点的寻找
pub trait AssistLocationConfig {
    fn assist_area_extend_size_h() -> i32;
    fn assist_area_extend_size_w() -> i32;
    fn assist_point_min_size() -> i32;
    fn assist_point_max_size() -> i32;
    fn assist_point_min_area() -> f64;
    fn assist_point_max_area() -> f64;
    fn assist_point_min_fill_ratio() -> f64;
    fn assist_point_whdiff_max() -> i32;
    fn assist_point_x_median_diff() -> i32;
}


pub struct AssistLocationSingleConfig;
impl AssistLocationConfig for AssistLocationSingleConfig {
    fn assist_area_extend_size_h() -> i32 { 8 }
    fn assist_area_extend_size_w() -> i32 { 8 }
    fn assist_point_min_size() -> i32 { 4 }
    fn assist_point_max_size() -> i32 { 9 }
    fn assist_point_min_area() -> f64 { 20.0 }
    fn assist_point_max_area() -> f64 { 70.0 }
    fn assist_point_min_fill_ratio() -> f64 { 0.9 }
    fn assist_point_whdiff_max() -> i32 { 2 }
    fn assist_point_x_median_diff() -> i32 { 20 }
}

pub struct AssistLocationPageConfig;
impl AssistLocationConfig for AssistLocationPageConfig {
    fn assist_area_extend_size_h() -> i32 { 35 }
    fn assist_area_extend_size_w() -> i32 { 20 }
    fn assist_point_min_size() -> i32 { 10 }
    fn assist_point_max_size() -> i32 { 20 }
    fn assist_point_min_area() -> f64 { 150.0 }
    fn assist_point_max_area() -> f64 { 310.0 }
    fn assist_point_min_fill_ratio() -> f64 { 0.88 }
    fn assist_point_whdiff_max() -> i32 { 4 }
    fn assist_point_x_median_diff() -> i32 { 20 }
}


pub trait FillConfig {
    fn fill_rate_min() -> f64;
    fn refine_coor_range() -> i32;
}

pub struct FillSingleConfig;
impl FillConfig for FillSingleConfig {
    fn fill_rate_min() -> f64 { 0.45 }
    fn refine_coor_range() -> i32 { 2 }
}

pub struct FillPageConfig;
impl FillConfig for FillPageConfig {
    fn fill_rate_min() -> f64 { 0.45 }
    fn refine_coor_range() -> i32 { 4 }
}

pub struct CommonConfig;
impl CommonConfig {
    pub const PAGE_NUMBER_FILL_RATE: f64 = 0.6;
    /// 页码点位置扩展大小
    pub const PAGE_NUMBER_EXTEND_SIZE: i32 = 20;
}

/// VX单线识别配置
pub trait VxConfig {
    fn fill_ratio_min() -> f64;
    fn fill_ratio_max() -> f64;
    fn preprocess_close_kernel_size() -> i32;
    fn preprocess_open_kernel_size() -> i32;
    fn max_end_points() -> usize;
    fn min_end_points() -> usize;
    fn max_branch_points() -> usize;
    fn hsv_lower1_bound() -> Scalar;
    fn hsv_lower2_bound() -> Scalar;
    fn hsv_upper1_bound() -> Scalar;
    fn hsv_upper2_bound() -> Scalar;
    fn lab_a_threshold() -> f64;
    fn component_min_points() -> usize;
    fn pac_min_points_count() -> usize;
    fn vx_box_expand_size() -> i32;

    // VX模型输入参数
    fn vx_model_channels() -> i32;  // 通道数：1=灰度，3=RGB
    fn vx_model_height() -> i32;    // 模型输入高度
    fn vx_model_width() -> i32;     // 模型输入宽度
    fn vx_model_padding_value() -> i32;  // padding填充值：0=黑色，1=白色
}

pub struct VxPageConfig;
impl VxConfig for VxPageConfig {
    fn fill_ratio_min() -> f64 { 0.03 }
    fn fill_ratio_max() -> f64 { 0.5 }
    fn preprocess_close_kernel_size() -> i32 { 3 }
    fn preprocess_open_kernel_size() -> i32 { 1 }
    fn max_end_points() -> usize { 2 }
    fn min_end_points() -> usize { 2 }
    fn max_branch_points() -> usize { 0 }
    fn hsv_lower1_bound() -> Scalar {
        Scalar::from([0.0, 20.0, 50.0, 0.0])
    }
    fn hsv_upper1_bound() -> Scalar {
        Scalar::from([15.0, 255.0, 255.0, 0.0])
    }
    fn hsv_lower2_bound() -> Scalar {
        Scalar::from([165.0, 20.0, 50.0, 0.0])
    }
    fn hsv_upper2_bound() -> Scalar {
        Scalar::from([180.0, 255.0, 255.0, 0.0])
    }
    fn lab_a_threshold() -> f64 { 160.0 }
    fn component_min_points() -> usize { 4 }
    fn pac_min_points_count() -> usize { 18 }
    fn vx_box_expand_size() -> i32 { 4 }

    // VX模型输入参数（默认3通道RGB）
    fn vx_model_channels() -> i32 { 1 }      // 3通道RGB
    fn vx_model_height() -> i32 { 36 }       // 高度36
    fn vx_model_width() -> i32 { 50 }        // 宽度50
    fn vx_model_padding_value() -> i32 { 0 } // padding用黑色
}

/// 初始化全局 rayon 线程池
/// 应在库初始化时调用一次，后续调用会被忽略
pub fn init_global_thread_pool(num_threads: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .ok(); // 只能设置一次，后续调用无效但不报错
}