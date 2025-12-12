
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
}

/// 辅助定位点的寻找
pub trait AssistLocationConfig {
    fn assist_area_extend_size() -> i32;
    fn assist_point_min_size() -> i32;
    fn assist_point_max_size() -> i32;
    fn assist_point_min_area() -> f64;
    fn assist_point_max_area() -> f64;
    fn assist_point_min_fill_ratio() -> f64;
    fn assist_point_whdiff_max() -> i32;
}


pub struct AssistLocationSingleConfig;
impl AssistLocationConfig for AssistLocationSingleConfig {
    fn assist_area_extend_size() -> i32 { 6 }
    fn assist_point_min_size() -> i32 { 4 }
    fn assist_point_max_size() -> i32 { 9 }
    fn assist_point_min_area() -> f64 { 20.0 }
    fn assist_point_max_area() -> f64 { 70.0 }
    fn assist_point_min_fill_ratio() -> f64 { 0.9 }
    fn assist_point_whdiff_max() -> i32 { 2 }
}

pub struct AssistLocationPageConfig;
impl AssistLocationConfig for AssistLocationPageConfig {
    fn assist_area_extend_size() -> i32 { 25 }
    fn assist_point_min_size() -> i32 { 10 }
    fn assist_point_max_size() -> i32 { 20 }
    fn assist_point_min_area() -> f64 { 180.0 }
    fn assist_point_max_area() -> f64 { 260.0 }
    fn assist_point_min_fill_ratio() -> f64 { 0.9 }
    fn assist_point_whdiff_max() -> i32 { 4 }
}

pub struct FillConfig;
impl FillConfig {
    pub const FILL_RATE_MIN: f64 = 0.45;
    pub const REFINE_COOR_RANGE: i32 = 2;
}
