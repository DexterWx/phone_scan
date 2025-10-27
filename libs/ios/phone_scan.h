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
#define ImageProcessingConfig_TARGET_WIDTH 2400

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
#define ImageProcessingConfig_MIN_AREA_RATIO 0.3

/**
 * 边界惩罚系数
 */
#define ImageProcessingConfig_MARGIN_PENALTY 50.0

#define AssistLocationConfig_ASSIST_AREA_EXTEND_SIZE 6

/**
 * 辅助定位点的标准大小
 */
#define AssistLocationConfig_ASSIST_POINT_MIN_SIZE 4

#define AssistLocationConfig_ASSIST_POINT_MAX_SIZE 9

#define AssistLocationConfig_ASSIST_POINT_MIN_AREA 20.0

#define AssistLocationConfig_ASSIST_POINT_MAX_AREA 70.0

#define AssistLocationConfig_ASSIST_POINT_MIN_FILL_RATIO 0.9

#define AssistLocationConfig_ASSIST_POINT_WHDIFF_MAX 2

#define FillConfig_FILL_RATE_MIN 0.5

char *initialize(const char *mark_ptr);

char *inference(const uint8_t *data_ptr, uintptr_t data_len);

/**
 * 释放C字符串内存
 */
void free_string(char *s);
