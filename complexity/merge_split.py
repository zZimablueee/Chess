import pandas as pd
import os
import glob

# ================= 配置区域 =================
ORIGINAL_DATA_PATH = r"C:\Users\Administrator\Desktop\Chess\data\processed\hpc_results\March3_results\4\input(remain_from_3)\todo_remaining_full.csv"

HPC_RESULTS_DIR = r"C:\Users\Administrator\Desktop\Chess\data\processed\hpc_results\March3_results\4\output"

OUTPUT_DIR = r"C:\Users\Administrator\Desktop\Chess\data\processed\hpc_results\March3_results\5\input(remain_from_4)"

# 两边列名不同
HPC_ID_COLUMN = "UID"
ORIGINAL_ID_COLUMN = "uid"
# ===========================================


def to_int(series):
    """只在匹配阶段统一ID格式"""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("=== 第一步：合并HPC结果 ===")

    all_files = glob.glob(os.path.join(HPC_RESULTS_DIR, "*.csv"))

    if not all_files:
        print("❌ HPC结果文件夹为空")
        return

    print(f"找到 {len(all_files)} 个结果文件")

    df_list = []

    for f in all_files:
        try:
            df = pd.read_csv(f)

            if HPC_ID_COLUMN not in df.columns:
                raise ValueError(f"{f} 里没有列 {HPC_ID_COLUMN}")

            df_list.append(df)

        except Exception as e:
            print(f"⚠️ 跳过 {f}: {e}")

    if not df_list:
        print("❌ 没有有效结果")
        return

    merged_results = pd.concat(df_list, ignore_index=True)

    # 按 UID 去重
    merged_results = merged_results.drop_duplicates(subset=[HPC_ID_COLUMN])

    finished_ids = set(
        to_int(merged_results[HPC_ID_COLUMN]).dropna()
    )

    result_file = os.path.join(OUTPUT_DIR, "all_finished_results.csv")
    merged_results.to_csv(result_file, index=False)

    print(f"✅ 已生成 {result_file}")
    print(f"完成游戏数: {len(finished_ids)}")

    print("\n=== 第二步：计算剩余任务 ===")

    df_original = pd.read_csv(ORIGINAL_DATA_PATH)

    if ORIGINAL_ID_COLUMN not in df_original.columns:
        raise ValueError(f"原始文件没有列 {ORIGINAL_ID_COLUMN}")

    original_ids = to_int(df_original[ORIGINAL_ID_COLUMN])

    mask = ~original_ids.isin(finished_ids)
    df_remaining = df_original[mask]

    remaining_file = os.path.join(OUTPUT_DIR, "todo_remaining_full.csv")
    df_remaining.to_csv(remaining_file, index=False)

    print(f"✅ 已生成 {remaining_file}")

    print("\n=== 统计 ===")
    print("原始总数:", len(df_original))
    print("已完成:", len(df_original) - len(df_remaining))
    print("剩余:", len(df_remaining))

    print("\n=== Sanity Check ===")
    print("完成 + 剩余 =", len(finished_ids) + len(df_remaining))
    print("应等于原始 =", len(df_original))

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()