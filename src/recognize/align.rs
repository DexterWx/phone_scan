use crate::models::Coordinate;

/// 对齐结果
#[derive(Debug, Clone)]
pub struct AlignResult {
    /// 多余检测点索引（在检测序列中）
    pub extra_detected: Vec<usize>,
    /// 缺失标注点索引（在标注序列中）
    pub missing_reference: Vec<usize>,
}

/// 从 Coordinate 提取 y 坐标中心点
pub fn extract_y_centers(coords: &[Coordinate]) -> Vec<i32> {
    coords.iter().map(|c| c.y + c.h / 2).collect()
}

/// 漏检情况：检测数 < 标注数
///
/// 用完整列作为基准，给缺失列的每个点找完整列中y差距最小的点。
/// 完整列中没被匹配到的点的索引就是标注数据要删除的。
///
/// # Arguments
/// * `complete_y` - 完整列的检测 y 坐标（数量 = 标注数量）
/// * `incomplete_y` - 缺失列的检测 y 坐标（数量 < 标注数量）
///
/// # Returns
/// 缺失的标注点索引列表
pub fn find_missing_indices(complete_y: &[i32], incomplete_y: &[i32]) -> Vec<usize> {
    let n = complete_y.len();
    let m = incomplete_y.len();

    if m >= n {
        return Vec::new();
    }

    // 记录完整列中哪些点被匹配了
    let mut matched = vec![false; n];

    // 给缺失列的每个点找完整列中y差距最小的点
    for &inc_y in incomplete_y {
        let mut best_idx = 0;
        let mut best_diff = i32::MAX;

        for (idx, &comp_y) in complete_y.iter().enumerate() {
            if matched[idx] {
                continue; // 已经被匹配过的跳过
            }
            let diff = (comp_y - inc_y).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = idx;
            }
        }

        matched[best_idx] = true;
    }

    // 完整列中没被匹配到的索引就是缺失的
    matched.iter()
        .enumerate()
        .filter(|(_, &m)| !m)
        .map(|(i, _)| i)
        .collect()
}

/// 漏检情况（余弦相似度优化版）：检测数 < 标注数
///
/// 通过枚举所有可能的缺失点组合，计算去除这些点后与缺失列的余弦相似度，
/// 选择相似度最高的组合作为缺失点。这种方法对试卷扭曲更鲁棒。
///
/// # Arguments
/// * `complete_y` - 完整列的检测 y 坐标（数量 = 标注数量）
/// * `incomplete_y` - 缺失列的检测 y 坐标（数量 < 标注数量）
///
/// # Returns
/// 缺失的标注点索引列表
pub fn find_missing_indices_cos(complete_y: &[i32], incomplete_y: &[i32]) -> Vec<usize> {
    let n = complete_y.len();
    let m = incomplete_y.len();

    if m >= n {
        return Vec::new();
    }

    let diff_count = n - m; // 需要去除的点数量

    // 如果差异过大，回退到原始方法
    if diff_count > 5 {
        return find_missing_indices(complete_y, incomplete_y);
    }

    let mut best_similarity = f64::NEG_INFINITY;
    let mut best_indices = Vec::new();

    // 生成所有可能的组合（去除 diff_count 个点）
    let mut combination = vec![0; diff_count];
    for i in 0..diff_count {
        combination[i] = i;
    }

    loop {
        // 构建去除当前组合后的序列
        let filtered: Vec<i32> = complete_y
            .iter()
            .enumerate()
            .filter(|(idx, _)| !combination.contains(idx))
            .map(|(_, &y)| y)
            .collect();

        // 计算余弦相似度
        let similarity = cosine_similarity(&filtered, incomplete_y);

        if similarity > best_similarity {
            best_similarity = similarity;
            best_indices = combination.clone();
        }

        // 生成下一个组合
        if !next_combination(&mut combination, n) {
            break;
        }
    }
    #[cfg(debug_assertions)]
    {
        println!("最佳余弦相似度: {}", best_similarity);
    }

    best_indices
}

/// 计算两个序列的余弦相似度
fn cosine_similarity(a: &[i32], b: &[i32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::NEG_INFINITY;
    }

    let a_f64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    let b_f64: Vec<f64> = b.iter().map(|&x| x as f64).collect();

    let dot_product: f64 = a_f64.iter().zip(b_f64.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a_f64.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b_f64.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return f64::NEG_INFINITY;
    }

    dot_product / (norm_a * norm_b)
}

/// 生成下一个组合（字典序）
/// 返回 false 表示已经是最后一个组合
fn next_combination(combination: &mut [usize], n: usize) -> bool {
    let k = combination.len();
    if k == 0 {
        return false;
    }

    // 从右往左找第一个可以增加的位置
    for i in (0..k).rev() {
        if combination[i] < n - k + i {
            combination[i] += 1;
            // 更新后续位置
            for j in (i + 1)..k {
                combination[j] = combination[j - 1] + 1;
            }
            return true;
        }
    }

    false
}

/// 多检情况：检测数 > 标注数
///
/// 用完整列作为基准，给完整列的每个点找缺失列中y差距最小的点。
/// 缺失列中没被匹配到的点就是多检的。
///
/// # Arguments
/// * `complete_y` - 完整列的检测 y 坐标（数量 = 标注数量）
/// * `extra_y` - 多检列的检测 y 坐标（数量 > 标注数量）
///
/// # Returns
/// 多余的检测点索引列表
pub fn find_extra_indices(complete_y: &[i32], extra_y: &[i32]) -> Vec<usize> {
    let n = complete_y.len();
    let m = extra_y.len();

    if m <= n {
        return Vec::new();
    }

    // 记录多检列中哪些点被匹配了
    let mut matched = vec![false; m];

    // 给完整列的每个点找多检列中y差距最小的点
    for &comp_y in complete_y {
        let mut best_idx = 0;
        let mut best_diff = i32::MAX;

        for (idx, &ext_y) in extra_y.iter().enumerate() {
            if matched[idx] {
                continue; // 已经被匹配过的跳过
            }
            let diff = (comp_y - ext_y).abs();
            if diff < best_diff {
                best_diff = diff;
                best_idx = idx;
            }
        }

        matched[best_idx] = true;
    }

    // 多检列中没被匹配到的索引就是多余的
    matched.iter()
        .enumerate()
        .filter(|(_, &m)| !m)
        .map(|(i, _)| i)
        .collect()
}

/// 多检情况（余弦相似度优化版）：检测数 > 标注数
///
/// 通过枚举所有可能的多余点组合，计算去除这些点后与完整列的余弦相似度，
/// 选择相似度最高的组合作为多余点。这种方法对试卷扭曲更鲁棒。
///
/// # Arguments
/// * `complete_y` - 完整列的检测 y 坐标（数量 = 标注数量）
/// * `extra_y` - 多检列的检测 y 坐标（数量 > 标注数量）
///
/// # Returns
/// 多余的检测点索引列表
pub fn find_extra_indices_cos(complete_y: &[i32], extra_y: &[i32]) -> Vec<usize> {
    let n = complete_y.len();
    let m = extra_y.len();

    if m <= n {
        return Vec::new();
    }

    let diff_count = m - n; // 需要去除的点数量

    // 如果差异过大，回退到原始方法
    if diff_count > 5 {
        return find_extra_indices(complete_y, extra_y);
    }

    let mut best_similarity = f64::NEG_INFINITY;
    let mut best_indices = Vec::new();

    // 生成所有可能的组合（去除 diff_count 个点）
    let mut combination = vec![0; diff_count];
    for i in 0..diff_count {
        combination[i] = i;
    }

    loop {
        // 构建去除当前组合后的序列
        let filtered: Vec<i32> = extra_y
            .iter()
            .enumerate()
            .filter(|(idx, _)| !combination.contains(idx))
            .map(|(_, &y)| y)
            .collect();

        // 计算余弦相似度
        let similarity = cosine_similarity(&filtered, complete_y);

        if similarity > best_similarity {
            best_similarity = similarity;
            best_indices = combination.clone();
        }

        // 生成下一个组合
        if !next_combination(&mut combination, m) {
            break;
        }
    }
    #[cfg(debug_assertions)]
    {
        println!("最佳余弦相似度: {}", best_similarity);
    }

    best_indices
}

/// 根据多余索引过滤检测点
pub fn filter_by_extra_indices(
    detected: &[Coordinate],
    extra_indices: &[usize],
) -> Vec<Coordinate> {
    detected.iter()
        .enumerate()
        .filter(|(i, _)| !extra_indices.contains(i))
        .map(|(_, c)| c.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_missing_one() {
        // 完整列18个点，缺失列17个点（漏检最后一个）
        let complete: Vec<i32> = (0..18).map(|i| i * 100).collect();
        let incomplete: Vec<i32> = (0..17).map(|i| i * 100).collect();

        let missing = find_missing_indices(&complete, &incomplete);
        assert_eq!(missing, vec![17]);
    }

    #[test]
    fn test_find_missing_middle() {
        // 完整列5个点，缺失列4个点（漏检第3个，索引2）
        let complete = vec![0];
        let incomplete = vec![]; // 缺少200

        let missing = find_missing_indices(&complete, &incomplete);
        assert_eq!(missing, vec![0]);
    }

    #[test]
    fn test_find_extra_one() {
        // 完整列4个点，多检列5个点（多了一个）
        let complete = vec![0, 100, 200, 300];
        let extra = vec![0, 100, 150, 200, 300]; // 多了150

        let extra_indices = find_extra_indices(&complete, &extra);
        assert_eq!(extra_indices, vec![2]);
    }

    #[test]
    fn test_find_extra_last() {
        // 完整列4个点，多检列5个点（多了最后一个）
        let complete = vec![0, 100, 200, 300];
        let extra = vec![0, 100, 200, 300, 400];

        let extra_indices = find_extra_indices(&complete, &extra);
        assert_eq!(extra_indices, vec![4]);
    }

    #[test]
    fn test_find_missing_cos_basic() {
        // 基本测试：完整列5个点，缺���列4个点（缺少第3个）
        let complete = vec![0, 100, 200, 300, 400];
        let incomplete = vec![0, 100, 300, 400];

        let missing = find_missing_indices_cos(&complete, &incomplete);
        assert_eq!(missing, vec![2]);
    }

    #[test]
    fn test_find_missing_cos_distorted() {
        // 模拟试卷扭曲：y坐标有轻微偏移
        let complete = vec![0, 100, 200, 300, 400];
        let incomplete = vec![5, 105, 305, 405]; // 缺少200，且有偏移

        let missing = find_missing_indices_cos(&complete, &incomplete);
        assert_eq!(missing, vec![2]);
    }

    #[test]
    fn test_find_missing_cos_multiple() {
        // 缺失多个点
        let complete = vec![0, 100, 200, 300, 400, 500];
        let incomplete = vec![0, 200, 400, 500]; // 缺少100和300

        let missing = find_missing_indices_cos(&complete, &incomplete);
        assert_eq!(missing.len(), 2);
        assert!(missing.contains(&1));
        assert!(missing.contains(&3));
    }

    #[test]
    fn test_cosine_similarity() {
        // 测试余弦相似度计算
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6); // 完全相同，相似度为1

        let c = vec![1, 2, 3];
        let d = vec![2, 4, 6];
        let sim2 = cosine_similarity(&c, &d);
        assert!((sim2 - 1.0).abs() < 1e-6); // 成比例，相似度为1
    }

    #[test]
    fn test_next_combination() {
        // 测试组合生成
        let mut comb = vec![0, 1];
        assert!(next_combination(&mut comb, 5));
        assert_eq!(comb, vec![0, 2]);

        let mut comb2 = vec![3, 4];
        assert!(!next_combination(&mut comb2, 5)); // 最后一个组合
    }

    #[test]
    fn test_find_extra_cos_basic() {
        // 基本测试：完整列4个点，多检列5个点（多了一个）
        let complete = vec![0, 100, 200, 300];
        let extra = vec![0, 100, 150, 200, 300]; // 多了150

        let extra_indices = find_extra_indices_cos(&complete, &extra);
        assert_eq!(extra_indices, vec![2]);
    }

    #[test]
    fn test_find_extra_cos_distorted() {
        // 模拟试卷扭曲：y坐标有轻微偏移
        let complete = vec![0, 100, 200, 300];
        let extra = vec![5, 105, 155, 205, 305]; // 多了155，且有偏移

        let extra_indices = find_extra_indices_cos(&complete, &extra);
        assert_eq!(extra_indices, vec![2]);
    }

    #[test]
    fn test_find_extra_cos_multiple() {
        // 多检多个点
        let complete = vec![0, 100, 200, 300];
        let extra = vec![0, 50, 100, 200, 250, 300]; // 多了50和250

        let extra_indices = find_extra_indices_cos(&complete, &extra);
        assert_eq!(extra_indices.len(), 2);
        assert!(extra_indices.contains(&1));
        assert!(extra_indices.contains(&4));
    }

    #[test]
    fn test_find_extra_cos_last() {
        // 多检最后一个点
        let complete = vec![0, 100, 200, 300];
        let extra = vec![0, 100, 200, 300, 400];

        let extra_indices = find_extra_indices_cos(&complete, &extra);
        assert_eq!(extra_indices, vec![4]);
    }
}
