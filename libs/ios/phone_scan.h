#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * 高斯模糊核大小
 */
#define ImageProcessingConfig_GAUSSIAN_KERNEL_SIZE 5

/**
 * 高斯模糊sigma值
 */
#define ImageProcessingConfig_GAUSSIAN_SIGMA 0.0

/**
 * 统一输入图像的宽度
 */
#define ImageProcessingConfig_TARGET_WIDTH_A4 2400

#define ImageProcessingConfig_TARGET_WIDTH_A3 4000

/**
 * 目标图片缩放比例
 */
#define ImageProcessingConfig_PAPER_SCAN_TARGET_SCALE 2.0

/**
 * 自适应阈值的块大小
 */
#define ImageProcessingConfig_BLOCK_SIZE 51

/**
 * 自适应阈值的常数
 */
#define ImageProcessingConfig_C 5

/**
 * 形态学操作的核大小
 */
#define ImageProcessingConfig_MORPH_KERNEL 3

/**
 * 多边形逼近的epsilon因子
 */
#define ImageProcessingConfig_EPSILON_FACTOR 0.015

/**
 * 最小面积占比
 */
#define ImageProcessingConfig_MIN_AREA_RATIO 0.25

/**
 * 边界惩罚系数
 */
#define ImageProcessingConfig_MARGIN_PENALTY 50.0

#define CommonConfig_PAGE_NUMBER_FILL_RATE 0.6

/**
 * 页码点位置扩展大小
 */
#define CommonConfig_PAGE_NUMBER_EXTEND_SIZE 20

/**
 * FFI 返回结构体，包含 JSON 和 RGB 图片数据
 */
typedef struct InferenceBatchResult {
  /**
   * JSON 字符串指针
   */
  char *json;
  /**
   * RGB 图片数据指针（3通道）
   */
  uint8_t *image_data;
  /**
   * 图片宽度
   */
  uint32_t width;
  /**
   * 图片高度
   */
  uint32_t height;
} InferenceBatchResult;

char *initialize(const char *mark_ptr);

char *initialize_paper(const char *mark_ptr);

char *inference(const uint8_t *data_ptr, uintptr_t data_len);

char *inference_paper(const uint8_t *data_ptr, uintptr_t data_len);

/**
 * 批量推理接口
 * 从多张 NV12 图片中选择最清晰的一张进行识别
 *
 * 参数:
 * - images: 所有图片拼接后的连续内存首地址 (NV12 格式)
 * - widths: 宽度数组指针
 * - heights: 高度数组指针
 * - rotations: 旋转角度数组指针 (0, 90, 180, 270)
 * - lens: 每张图片的字节长度数组指针
 * - count: 图片数量
 *
 * 返回: JSON 字符串 (MobileOutput)
 */
char *inference_batch(const uint8_t *images,
                      const uint32_t *widths,
                      const uint32_t *heights,
                      const uint8_t *rotations,
                      const uint32_t *lens,
                      uint32_t count);

/**
 * 返回: InferenceBatchResult 包含 JSON 和 RGB 图片数据
 */
struct InferenceBatchResult inference_batch_v2(const uint8_t *images,
                                               const uint32_t *widths,
                                               const uint32_t *heights,
                                               const uint8_t *rotations,
                                               const uint32_t *lens,
                                               uint32_t count);

/**
 * 释放 RGB 图片数据内存
 */
void free_image_data(uint8_t *image_data, uint32_t width, uint32_t height);

/**
 * 销毁引擎，释放资源
 */
void destroy_engine(void);

/**
 * 释放C字符串内存
 */
void free_string(char *s);

char *create_train_data(const uint8_t *data_ptr,
                        uintptr_t data_len,
                        char *out_dir,
                        char *file_name);
