use serde::{Deserialize, Serialize};
use opencv::core::Point2i as CvPoint2i;
use std::collections::HashMap;

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

impl Coordinate {
    /// 缩放
    pub fn resize(&self, scale: f64) -> Coordinate {
        Coordinate {
            x: (self.x as f64 * scale) as i32,
            y: (self.y as f64 * scale) as i32,
            w: (self.w as f64 * scale) as i32,
            h: (self.h as f64 * scale) as i32,
        }
    }
}

/// 非矩形四边形
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
    pub rgb: opencv::core::Mat,
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
    /// 识别类型：1-单选，2-多选，3-划分
    pub rec_type: RecType,
    /// 各个子选项的坐标
    pub sub_options: Vec<Coordinate>,
}

impl RecItem {
    pub fn resize(&self, scale: f64) -> RecItem {
        RecItem {
            rec_type: self.rec_type,
            sub_options: self.sub_options.iter().map(|coor| coor.resize(scale)).collect(),
        }
    }
}

/// 标注信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkSingle {
    /// 外围矩形边框
    pub boundary: Coordinate,
    /// 需要识别的项目
    pub rec_items: Vec<RecItem>,
    /// 辅助定位
    pub assist_location: AssistLocation,
}
/// 辅助定位点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistLocation {
    pub left: Vec<Coordinate>,
    pub right: Vec<Coordinate>,
}

impl AssistLocation {
    pub fn split(&self) -> Vec<AssistLocation> {
        let mut res = Vec::new();
        let left_groups = Self::split_with_x(&self.left);
        let right_groups = Self::split_with_x(&self.right);
        for i in 0..left_groups.len() {
            res.push(
                AssistLocation {
                    left: left_groups[i].clone(),
                    right: right_groups[i].clone()
                }
            );
        }
        res
    }

    pub fn resize(&self, scale: f64) -> AssistLocation {
        AssistLocation {
            left: self.left.iter().map(|coor| coor.resize(scale)).collect(),
            right: self.right.iter().map(|coor| coor.resize(scale)).collect(),
        }
    }

    pub fn merge(locations: &Vec<AssistLocation>) -> AssistLocation {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for assist_location in locations {
            left.extend(assist_location.left.clone());
            right.extend(assist_location.right.clone());
        }
        AssistLocation {
            left,
            right
        }
    }
    

    fn split_with_x(locations: &Vec<Coordinate>) -> Vec<Vec<Coordinate>> {
        
        // 使用 HashMap 按照 x 坐标对坐标进行分组
        let mut groups: HashMap<i32, Vec<Coordinate>> = HashMap::new();
        
        // 遍历所有坐标，按 x 值分组
        for coord in locations {
            groups.entry(coord.x).or_insert_with(Vec::new).push(coord.clone());
        }
        
        // 将 HashMap 转换为 Vec<Vec<Coordinate>>
        let mut result: Vec<(i32, Vec<Coordinate>)> = groups.into_iter().collect();
        
        // 按照 x 值排序
        result.sort_by_key(|(x, _)| *x);
        
        // 只返回坐标列表部分
        result.into_iter().map(|(_, coords)| coords).collect()
    }
    
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
    pub fill_rate: f64,
    pub coordinate: Coordinate,
}

/// 输出数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileOutput {
    /// 识别状态：0-成功，1-失败
    pub code: i32,
    pub message: String,
    pub page_number: u8,
    /// 对应输入的rec_items的识别结果
    pub rec_results: Vec<RecResult>
}

impl MobileOutput {
    /// 创建一个新的MobileOutput实例
    /// 根据输入的Mark结构初始化rec_results，所有选项默认为false（未选中）
    pub fn new(rec_items: &Vec<RecItem>) -> Self {
        let rec_results = rec_items
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
            page_number: 0,
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


/// 整卷标注信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkPaper {
    /// 外围矩形边框
    pub boundary: Coordinate,
    /// 页码点
    pub page_number: Vec<Coordinate>,
    /// 页面信息
    pub pages: Vec<MarkPage>
}

impl MarkPaper {
    pub fn is_a4(&self) -> bool {
        self.boundary.w < self.boundary.h
    }
    pub fn resize(&self, scale: f64) -> MarkPaper {
        MarkPaper {
            boundary: self.boundary.resize(scale),
            page_number: self.page_number.iter().map(|coor| coor.resize(scale)).collect(),
            pages: self.pages.iter().map(|page| page.resize(scale)).collect(),
        }
    }
}

/// 单页标注信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkPage {
    /// 需要识别的项目
    pub rec_items: Vec<RecItem>,
    /// 辅助定位
    pub assist_location: AssistLocation,
}

impl MarkPage {
    pub fn resize(&self, scale: f64) -> MarkPage {
        MarkPage {
            rec_items: self.rec_items.iter().map(|item| item.resize(scale)).collect(),
            assist_location: self.assist_location.resize(scale),
        }
    }
}

