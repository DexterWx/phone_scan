use serde::{Deserialize, Serialize};
use opencv::core::{Point2f, Point2i as CvPoint2i, Vector};
use std::collections::HashMap;

use crate::myutils::image::get_points_from_coordinates;

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

    pub fn to_points(&self) -> Vector<Point2f> {
        let mut points = Vector::new();
        points.push(Point2f::new(self.x as f32, self.y as f32));
        points.push(Point2f::new((self.x + self.w) as f32, self.y as f32));
        points.push(Point2f::new((self.x + self.w) as f32, (self.y + self.h) as f32));
        points.push(Point2f::new(self.x as f32, (self.y + self.h) as f32));
        points
    }

    pub fn get_center(&self) -> Point2f {
        Point2f {
            x: (self.x + self.w / 2) as f32,
            y: (self.y + self.h / 2) as f32,
        }
    }
}

/// 非矩形四边形
#[derive(Debug, Clone)]
pub struct Quad {
    /// 四个顶点坐标
    pub points: [CvPoint2i; 4],
}

impl Quad {
    pub fn to_points(&self) -> Vector<Point2f> {
        let mut points = Vector::new();
        for point in &self.points {
            points.push(Point2f::new(point.x as f32, point.y as f32));
        }
        points
    }

    pub fn to_coordinate(&self) -> Coordinate {
        Coordinate {
            x: self.points[0].x,
            y: self.points[0].y,
            w: self.points[1].x - self.points[0].x,
            h: self.points[2].y - self.points[0].y,
        }
    }

    pub fn get_center(&self) -> Point2f {
        let mut sum_x = 0f32;
        let mut sum_y = 0f32;
        for point in &self.points {
            sum_x += point.x as f32;
            sum_y += point.y as f32;
        }
        Point2f {
            x: sum_x / 4f32,
            y: sum_y / 4f32,
        }
    }
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
#[derive(Debug, Clone)]
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
    // 划分题
    Vx = 3,
    /// 手写识别
    HandWriting = 4,
    /// 定位
    Location = 5
    
}

impl From<i32> for RecType {
    fn from(value: i32) -> Self {
        match value {
            1 => RecType::SingleChoice,
            2 => RecType::MultipleChoice,
            3 => RecType::Vx,
            4 => RecType::HandWriting,
            5 => RecType::Location,
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
    // 先x后y排序
    fn init_sort(&mut self) {
        self.left.sort_by(|a, b| {                                                                                   
            match a.x.cmp(&b.x) {                                                                                    
                std::cmp::Ordering::Equal => a.y.cmp(&b.y),                                                          
                other => other,                                                                                      
            }                                                                                                        
        });
        self.right.sort_by(|a, b| {                                                                                   
            match a.x.cmp(&b.x) {                                                                                    
                std::cmp::Ordering::Equal => a.y.cmp(&b.y),                                                          
                other => other,                                                                                      
            }                                                                                                        
        });
    }

    pub fn to_points(&self) -> Vector<Point2f> {
        let coors = vec![self.left.clone(), self.right.clone()].concat();
        let points = get_points_from_coordinates(&coors);
        points
    }
    
}

/// 识别结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecResult {
    /// 对应输入的sub_options，true表示选中，false表示未选中
    pub rec_result: Vec<bool>,
    pub rec_options: Vec<RecOption>,
    pub rec_type: RecType
}

/// 识别结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecOption {
    /// 置信度
    pub fill_rate: f64,
    /// 坐标
    pub coordinate: Coordinate,
    /// VX 分类结果: 0=single(单线/有效), 1=cancel(非单线), 2=...(未来扩展)
    /// 只有 class_id=0 时才算"选中"
    pub class_id: u8,
}

/// 单一连通域信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectFeatures {
    pub points_count: usize,
    pub has_branch: bool,
    pub end_points: usize,
    pub curvature_score: f64
}


/// 输出数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileOutput {
    /// 识别状态：0-成功，1-失败
    pub code: i32,
    pub message: String,
    pub page_number: usize,
    pub image_index: usize,
    /// 对应输入的rec_items的识别结果
    pub rec_results: Vec<RecResult>,
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
                    rec_options: rec_item.sub_options.iter().map(
                        |coordinate| RecOption {
                            fill_rate: 0.0,
                            coordinate: coordinate.clone(),
                            class_id: 1, // 默认非选中
                        }
                    ).collect(),
                    rec_type: rec_item.rec_type
                }
            })
            .collect();

        MobileOutput {
            code: 0, // 默认状态为成功
            message: "success".to_string(),
            page_number: 0,
            image_index: 0,
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
    pub vx_model_path: String,
    /// 外围矩形边框
    pub boundary: Coordinate,
    /// 页码点
    pub page_number: Vec<Coordinate>,
    /// 页面信息
    pub pages: Vec<MarkPage>,
    /// vx识别线程数，默认1
    #[serde(default = "default_num_threads")]
    pub num_threads: usize
}

fn default_num_threads() -> usize { 1 }

impl MarkPaper {
    pub fn is_a4(&self) -> bool {
        self.boundary.w < self.boundary.h
    }
    pub fn resize(&self, scale: f64) -> MarkPaper {
        MarkPaper {
            vx_model_path: self.vx_model_path.clone(),
            boundary: self.boundary.resize(scale),
            page_number: self.page_number.iter().map(|coor| coor.resize(scale)).collect(),
            pages: self.pages.iter().map(|page| page.resize(scale)).collect(),
            num_threads: self.num_threads
        }
    }
    pub fn init_sort(&mut self) {
        for page in &mut self.pages {
            page.init_sort();
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

    fn init_sort(&mut self) {
        self.assist_location.init_sort();
    }
}

