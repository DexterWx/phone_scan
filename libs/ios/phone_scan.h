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

char *initialize(const char *mark_ptr);

char *initialize_paper(const char *mark_ptr);

char *inference(const uint8_t *data_ptr, uintptr_t data_len);

char *inference_paper(const uint8_t *data_ptr, uintptr_t data_len);

/**
 * 销毁引擎，释放资源
 */
void destroy_engine(void);

/**
 * 释放C字符串内存
 */
void free_string(char *s);
