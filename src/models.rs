use serde::{Deserialize, Serialize};

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

/// 识别项目信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecItem {
    /// 识别类型：1-单选，2-多选
    pub rec_type: i32,
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
}

/// 输出数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileOutput {
    /// 识别状态：0-成功，1-未检测到定位点，2-程序bug未知错误
    pub code: i32,
    /// 对应输入的rec_items的识别结果
    pub rec_results: Vec<RecResult>,
}
