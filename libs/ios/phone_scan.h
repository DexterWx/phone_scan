#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * 生成一个黑色三角形在白色背景上的图片，返回base64编码
 */
char *generate_triangle_image(void);

/**
 * 释放C字符串内存
 */
void free_string(char *s);
