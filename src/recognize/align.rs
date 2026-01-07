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
        let complete = vec![0, 100, 200, 300, 400];
        let incomplete = vec![0, 100, 300, 400]; // 缺少200

        let missing = find_missing_indices(&complete, &incomplete);
        assert_eq!(missing, vec![2]);
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
}
