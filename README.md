# Phone Scan

一个基于 Rust 和 OpenCV 的高性能光学标记识别 (OMR) 库,用于自动识别答题卡和试卷上的填涂信息。

## 项目简介

Phone Scan 是一个专门用于答题卡识别的跨平台库,支持 Android 和 iOS 平台。通过先进的图像处理算法和透视变换技术,能够准确识别答题卡上的单选题、多选题和划分题等填涂信息。

### 主要特性

- **高精度识别**: 使用 Otsu 自适应阈值算法,自动计算最佳填涂率阈值
- **自动定位**: 支持答题卡自动定位和透视矫正
- **辅助定位**: 通过辅助定位点进行二次精确校正,提高识别准确率
- **多种题型支持**:
  - 单选题 (SingleChoice)
  - 多选题 (MultipleChoice)
  - 划分题 (Vx)
- **页码识别**: 自动识别试卷页码
- **跨平台**: 编译为动态库/静态库,提供 C FFI 接口供移动端调用
- **性能优化**: Release 模式下开启 LTO 和最大优化等级

## 技术架构

### 核心模块

```
src/
├── lib.rs              # 库入口和 C FFI 接口
├── models.rs           # 数据模型定义
├── config.rs           # 配置参数
├── myutils/            # 工具模块
│   ├── image.rs        # 图像处理工具
│   ├── math.rs         # 数学计算工具
│   ├── myjson.rs       # JSON 序列化工具
│   └── rendering.rs    # 调试渲染工具
└── recognize/          # 识别模块
    ├── engine.rs       # 识别引擎
    ├── location.rs     # 答题卡定位
    ├── assist_location.rs  # 辅助定位点检测
    ├── fill.rs         # 填涂识别
    ├── vx.rs           # 划分题识别
    └── page_number.rs  # 页码识别
```

### 识别流程

```
输入图像
    ↓
图像预处理 (缩放、灰度化、二值化)
    ↓
答题卡定位检测
    ↓
第一次透视变换 (基于边界)
    ↓
辅助定位点检测
    ↓
第二次透视变换 (基于辅助点)
    ↓
填涂识别 (Otsu 自适应阈值)
    ↓
输出识别结果
```

## 依赖项

- **opencv**: 图像处理核心库 (v0.96.0)
- **serde/serde_json**: 数据序列化
- **base64**: Base64 编解码
- **anyhow**: 错误处理

## 使用方法

### Rust API

```rust
use phone_scan::recognize::engine::RecEngine;
use opencv::imgcodecs::imread;

// 单张答题卡识别
let scan_string = std::fs::read_to_string("scan_config.json")?;
let image = imread("answer_card.jpg", opencv::imgcodecs::IMREAD_COLOR)?;

let engine = RecEngine::new_single(&scan_string)?;
let result = engine.inference_single(&image)?;

// 整卷试卷识别
let engine = RecEngine::new_paper(&scan_string)?;
let result = engine.inference_paper(&image)?;
```

### C FFI 接口

```c
// 初始化引擎
char* result = initialize(mark_json_string);

// 执行识别
char* output = inference(image_data, image_length);

// 释放字符串内存
free_string(result);
free_string(output);
```

## 配置说明

### 输入配置 (JSON)

#### 单张答题卡配置

```json
{
  "boundary": {
    "x": 0,
    "y": 0,
    "w": 2400,
    "h": 3200
  },
  "rec_items": [
    {
      "rec_type": 1,
      "sub_options": [
        {"x": 100, "y": 100, "w": 50, "h": 50},
        {"x": 200, "y": 100, "w": 50, "h": 50}
      ]
    }
  ],
  "assist_location": {
    "left": [{"x": 50, "y": 50, "w": 10, "h": 10}],
    "right": [{"x": 2350, "y": 50, "w": 10, "h": 10}]
  }
}
```

#### 整卷试卷配置

```json
{
  "boundary": {
    "x": 0,
    "y": 0,
    "w": 4000,
    "h": 2800
  },
  "page_number": [
    {"x": 100, "y": 100, "w": 50, "h": 50}
  ],
  "pages": [
    {
      "rec_items": [...],
      "assist_location": {...}
    }
  ]
}
```

### 输出格式

```json
{
  "code": 0,
  "message": "success",
  "page_number": 1,
  "rec_results": [
    {
      "rec_type": 1,
      "rec_result": [false, true, false, false],
      "rec_options": [
        {
          "fill_rate": 0.12,
          "coordinate": {"x": 100, "y": 100, "w": 50, "h": 50},
          "vx": false
        }
      ]
    }
  ]
}
```

## 编译构建

### 本地开发

```bash
cargo build
cargo test
```

### Release 构建

```bash
cargo build --release
```

生成的库文件位于 `target/release/` 目录下:
- macOS: `libphone_scan.dylib` / `libphone_scan.a`
- Linux: `libphone_scan.so` / `libphone_scan.a`
- Windows: `phone_scan.dll` / `phone_scan.lib`

### iOS 构建

```bash
# 构建 iOS 架构
cargo build --release --target aarch64-apple-ios
```

### Android 构建

需要配置 Android NDK 并指定对应的 target。

## 开发工具

项目包含多个 Python 辅助脚本用于开发和测试:

- **det_red.py**: 红色标记检测工具
- **get_mark.py**: 标注信息生成工具
- **debug.py**: 调试和可视化工具
- **test_batch.py**: 批量测试工具
- **val.py**: 验证工具
- **any2jpg.py**: 图像格式转换工具

## 测试

```bash
# 运行所有测试
cargo test

# 运行单个测试
cargo test test_demo
cargo test test_paper
```

测试数据位于 `dev/test_data/cards/` 目录下。

## 配置参数

主要配置参数在 `src/config.rs` 中定义:

- **TARGET_WIDTH_A4**: A4 纸目标宽度 (2400px)
- **TARGET_WIDTH_A3**: A3 纸目标宽度 (4000px)
- **GAUSSIAN_KERNEL_SIZE**: 高斯模糊核大小 (5)
- **BLOCK_SIZE**: 自适应阈值块大小 (51)
- **MIN_AREA_RATIO**: 最小面积占比 (0.25)
- **fill_rate_min**: 填涂率最小阈值 (0.45)

## 算法说明

### 填涂识别

使用 **Otsu 自适应阈值算法**动态计算最佳填涂率阈值:

1. 计算积分图加速区域求和
2. 计算所有选项的填涂率
3. 使用 Otsu 算法找到最佳分割阈值
4. 应用阈值判断填涂状态

### 定位校正

采用**两次透视变换**确保高精度:

1. **粗定位**: 基于答题卡边界进行第一次透视变换
2. **精定位**: 基于辅助定位点进行第二次精确校正

### 页码识别

通过识别页码标记点的填涂状态确定当前页码。

## 性能优化

Release 模式下的编译优化:

```toml
[profile.release]
opt-level = 3           # 最大优化等级
lto = true              # 链接时优化
codegen-units = 1       # 单代码生成单元
panic = "abort"         # panic 直接 abort
strip = true            # 移除符号表
debug = false           # 移除调试信息
overflow-checks = false # 关闭溢出检查
```

## Debug 模式

Debug 模式下会自动保存中间处理图像到 `dev/test_data/debug/` 目录:

- `debug_location.jpg`: 定位检测结果
- `baizheng_gray.jpg`: 校正后的灰度图
- `baizheng_thresh.jpg`: 二值化图像
- `baizheng_closed.jpg`: 形态学处理后的图像
- `render_out.jpg`: 识别结果可视化

## 版本历史

- **v1.2**: 当前版本
- **v1.1**: 修复 Otsu 算法阈值计算错误
- 添加输入图像日志记录

## 许可证

请查阅项目许可证文件。

## 贡献

欢迎提交 Issue 和 Pull Request。
