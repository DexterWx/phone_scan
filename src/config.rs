/// 边界检测配置参数 - 静态配置
pub struct BoundaryDetectionConfig;

impl BoundaryDetectionConfig {
    /// 自适应阈值的块大小
    pub const BLOCK_SIZE: i32 = 51;
    
    /// 自适应阈值的常数
    pub const C: i32 = 5;
    
    /// 形态学操作的核大小
    pub const MORPH_KERNEL: i32 = 7;
    
    /// 多边形逼近的epsilon因子
    pub const EPSILON_FACTOR: f64 = 0.015;
    
    /// 最小面积占比
    pub const MIN_AREA_RATIO: f64 = 0.05;
}

/// 图像处理配置参数
pub struct ImageProcessingConfig;

impl ImageProcessingConfig {
    /// 高斯模糊核大小
    pub const GAUSSIAN_KERNEL_SIZE: i32 = 5;
    
    /// 高斯模糊sigma值
    pub const GAUSSIAN_SIGMA: f64 = 0.0;
}

/// 评分配置参数
pub struct ScoringConfig;

impl ScoringConfig {
    /// 边界惩罚系数
    pub const MARGIN_PENALTY: f64 = 50.0;
}
