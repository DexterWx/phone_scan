import json
import numpy as np
import sys

def load_json(path):
    with open(path, 'r') as f:
        j = json.load(f)

    shape = j["shape"]
    data = np.array(j["data"], dtype=np.float32)

    expected_size = np.prod(shape)
    if data.size != expected_size:
        raise ValueError(
            f"{path}: data size {data.size} != shape product {expected_size}"
        )

    arr = data.reshape(shape)
    return arr


def compare_json(j1_path, j2_path, eps=1e-5):
    a = load_json(j1_path)
    b = load_json(j2_path)

    if a.shape != b.shape:
        print("❌ Shape mismatch")
        print("A:", a.shape)
        print("B:", b.shape)
        return

    diff = np.abs(a - b)

    max_diff = diff.max()
    mean_diff = diff.mean()
    num_bad = np.sum(diff > eps)

    print("==== Compare Result ====")
    print("Shape:", a.shape)
    print(f"Max diff : {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")
    print(f"> eps({eps}) count: {num_bad} / {diff.size}")

    # 打印前几个异常点
    if num_bad > 0:
        print("\nFirst 10 mismatches (index, A, B, diff):")
        idxs = np.argwhere(diff > eps)
        for idx in idxs[:10]:
            idx = tuple(idx)
            print(
                idx,
                a[idx],
                b[idx],
                diff[idx]
            )

    # 是否完全一致
    if num_bad == 0:
        print("✅ Arrays are numerically identical (within eps)")
    else:
        print("⚠️ Arrays differ")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_json.py a.json b.json [eps]")
        sys.exit(1)

    j1 = sys.argv[1]
    j2 = sys.argv[2]
    eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5

    compare_json(j1, j2, eps)
