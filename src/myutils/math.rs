/// 实现Otsu阈值算法，用于计算一维直方图的最佳分割阈值
/// Otsu算法通过最大化类间方差来确定最佳阈值，适用于双峰直方图

/// 计算给定数据的最佳Otsu阈值
/// 输入: values - 数据值向量
/// 返回: 最佳阈值
pub fn otsu_threshold(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    // 确定直方图范围
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // 如果所有值相同，直接返回该值
    if (max_val - min_val).abs() < f64::EPSILON {
        return min_val;
    }
    
    // 创建256个bins的直方图
    const NUM_BINS: usize = 256;
    let mut histogram = [0usize; NUM_BINS];
    let bin_width = (max_val - min_val) / (NUM_BINS - 1) as f64;
    
    // 填充直方图
    for &value in values {
        let bin_index = ((value - min_val) / bin_width) as usize;
        // 防止索引越界
        let bin_index = bin_index.min(NUM_BINS - 1);
        histogram[bin_index] += 1;
    }
    
    // 计算累积直方图和累积矩
    let mut cumulative_histogram = [0usize; NUM_BINS];
    let mut cumulative_moments = [0.0f64; NUM_BINS];
    
    cumulative_histogram[0] = histogram[0];
    cumulative_moments[0] = 0.0 * histogram[0] as f64;
    
    for i in 1..NUM_BINS {
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i];
        cumulative_moments[i] = cumulative_moments[i - 1] + (i as f64) * histogram[i] as f64;
    }
    
    // 总像素数和总矩
    let total_pixels = cumulative_histogram[NUM_BINS - 1];
    let total_moments = cumulative_moments[NUM_BINS - 1];
    
    // 避免除零错误
    if total_pixels == 0 {
        return 0.0;
    }
    
    // 计算全局均值
    let global_mean = total_moments / total_pixels as f64;
    
    // 寻找最大类间方差
    let mut max_variance = 0.0;
    let mut best_threshold = 0.0;
    
    for i in 0..NUM_BINS - 1 {
        let pixels_background = cumulative_histogram[i];
        let pixels_foreground = total_pixels - pixels_background;
        
        // 避免除零错误
        if pixels_background == 0 || pixels_foreground == 0 {
            continue;
        }
        
        let moment_background = cumulative_moments[i];
        let moment_foreground = total_moments - moment_background;
        
        let mean_background = moment_background / pixels_background as f64;
        let mean_foreground = moment_foreground / pixels_foreground as f64;
        
        // 计算类间方差
        let diff = mean_background - mean_foreground;
        let variance = (pixels_background as f64) * (pixels_foreground as f64) * diff * diff;
        
        if variance > max_variance {
            max_variance = variance;
            // 将bin索引转换回值域
            best_threshold = min_val + (i as f64) * bin_width;
        }
    }
    
    best_threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otsu_threshold() {
        // 测试简单双峰分布
        let mut values = vec![];
        // 添加低值峰
        for _ in 0..100 {
            values.push(20.0);
        }
        // 添加高值峰
        for _ in 0..100 {
            values.push(200.0);
        }
        
        let threshold = otsu_threshold(&values);
        // 阈值应该在两个峰之间
        assert!(threshold > 20.0 && threshold < 200.0);
    }
    
    #[test]
    fn test_otsu_threshold_single_value() {
        let values = vec![100.0; 50];
        let threshold = otsu_threshold(&values);
        assert_eq!(threshold, 100.0);
    }
    
    #[test]
    fn test_otsu_threshold_empty() {
        let values: Vec<f64> = vec![];
        let threshold = otsu_threshold(&values);
        assert_eq!(threshold, 0.0);
    }
}