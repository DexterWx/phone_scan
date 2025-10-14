use serde::{Deserialize, Serialize};
use opencv::core::Point2i as CvPoint2i;

/// 坐标信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coordinate {
    /// X坐标
    pub x: i32,
    /// Y坐标
    pub y: i32,
    /// 宽度
    pub w: i32,
    /// 高度
    pub h: i32,
}

/// 四边形坐标信息
#[derive(Debug, Clone)]
pub struct Quad {
    /// 四个顶点坐标
    pub points: [CvPoint2i; 4],
}

/// 轮廓信息，包含额外的检测数据
#[derive(Debug, Clone)]
pub struct ContourInfo {
    /// 轮廓点
    pub points: opencv::core::Vector<CvPoint2i>,
    /// 面积
    pub area: f64
}

/// 处理后的图片数据
#[derive(Debug)]
pub struct ProcessedImage {
    /// 灰度图
    pub gray: opencv::core::Mat,
    /// 二值图
    pub thresh: opencv::core::Mat,
    /// 形态学处理后的图
    pub closed: opencv::core::Mat,
}

/// 识别类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "i32", into = "i32")]
pub enum RecType {
    /// 单选题
    SingleChoice = 1,
    /// 多选题
    MultipleChoice = 2,
}

impl From<i32> for RecType {
    fn from(value: i32) -> Self {
        match value {
            1 => RecType::SingleChoice,
            2 => RecType::MultipleChoice,
            _ => RecType::SingleChoice, // 默认值
        }
    }
}

impl From<RecType> for i32 {
    fn from(rec_type: RecType) -> Self {
        rec_type as i32
    }
}

/// 识别项目信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecItem {
    /// 识别类型：1-单选，2-多选
    pub rec_type: RecType,
    /// 各个子选项的坐标
    pub sub_options: Vec<Coordinate>,
}

/// 标注信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mark {
    /// 外围矩形边框
    pub boundary: Coordinate,
    /// 需要识别的项目
    pub rec_items: Vec<RecItem>,
}

/// 识别结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecResult {
    /// 对应输入的sub_options，true表示选中，false表示未选中
    pub rec_result: Vec<bool>,
    pub fill_items: Vec<FillItem>,
    pub rec_tpye: RecType

}

/// 填涂率结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillItem {
    /// 对应输入的sub_options，true表示选中，false表示未选中
    pub fill_rate: f64,
    /// 原始坐标信息（仅在调试模式下包含）
    pub coordinate: Coordinate,
}

/// 输出数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileOutput {
    /// 识别状态：0-成功，1-失败
    pub code: i32,
    pub message: String,
    /// 对应输入的rec_items的识别结果
    pub rec_results: Vec<RecResult>
}

impl MobileOutput {
    /// 创建一个新的MobileOutput实例
    /// 根据输入的Mark结构初始化rec_results，所有选项默认为false（未选中）
    pub fn new(mark: &Mark) -> Self {
        let rec_results = mark.rec_items
            .iter()
            .map(|rec_item| {
                // 为每个rec_item创建对应的RecResult，初始化所有选项为false
                RecResult {
                    rec_result: vec![false; rec_item.sub_options.len()],
                    fill_items: rec_item.sub_options.iter().map(
                        |coordinate| FillItem {
                            fill_rate: 0.0,
                            coordinate: coordinate.clone(),
                        }
                    ).collect(),
                    rec_tpye: rec_item.rec_type
                }
            })
            .collect();

        MobileOutput {
            code: 0, // 默认状态为成功
            message: "success".to_string(),
            rec_results,
        }
    }
}

/// 初始化状态，c接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitInfo {
    pub code: u8,
    pub message: String
}
