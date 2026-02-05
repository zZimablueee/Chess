import pandas as pd
import os
import glob

# ================= 配置区域 =================
# 1. 原始 100万局 大文件的路径
ORIGINAL_DATA_PATH = r"C:\Users\Administrator\Desktop\Chess\data\input\100wuid.csv"

# 2. 从 HPC 拷回来的、放着一堆小结果文件的文件夹路径
HPC_RESULTS_DIR = r"C:\Users\Administrator\Desktop\half"

# 3. 输出文件的保存文件夹
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\Chess\data\processed"

# 4. [关键] 请告诉脚本，你的 CSV 里哪一列是用来对比的唯一 ID？
#    - 如果是第一列，保持 default 即可
#    - 如果你的 ID 列叫 "GameID" 或 "id"，请修改下面：
ID_COLUMN_NAME = 0  # 填 0 表示第一列，填字符串例如 "GameID" 表示列名
# ===========================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("=== 第一步：合并已跑出的结果 ===")
    all_files = glob.glob(os.path.join(HPC_RESULTS_DIR, "*.csv"))
    
    if not all_files:
        print("❌ 错误：结果文件夹里是空的！请检查路径。")
        return

    print(f"找到 {len(all_files)} 个结果文件，开始合并...")
    
    df_list = []
    for filename in all_files:
        try:
            # 假设结果文件有表头，如果没有请改成 header=None
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ 跳过坏文件 {filename}: {e}")

    if not df_list:
        print("❌ 没有有效数据。")
        return

    # 1. 生成第一个文件：已完成任务总表
    merged_results = pd.concat(df_list, ignore_index=True)
    
    # 获取 ID 列的数据用于去重
    if isinstance(ID_COLUMN_NAME, int):
        finished_id_col = merged_results.iloc[:, ID_COLUMN_NAME]
    else:
        finished_id_col = merged_results[ID_COLUMN_NAME]

    # 转成集合，用来做减法
    finished_ids = set(finished_id_col.astype(str).unique())
    
    # 保存结果文件
    result_file_path = os.path.join(OUTPUT_DIR, "all_finished_results.csv")
    merged_results.to_csv(result_file_path, index=False)
    print(f"✅ [文件1] 已生成：{result_file_path}")
    print(f"   包含 {len(merged_results)} 行数据，覆盖 {len(finished_ids)} 个独立游戏。")

    print("\n=== 第二步：生成剩余任务大表 ===")
    print("正在读取原始 100万局 数据 (可能需要几秒)...")
    
    # 读取原始文件
    df_original = pd.read_csv(ORIGINAL_DATA_PATH)
    
    # 找到原始文件里的 ID 列
    if isinstance(ID_COLUMN_NAME, int):
        original_id_col = df_original.iloc[:, ID_COLUMN_NAME]
    else:
        original_id_col = df_original[ID_COLUMN_NAME]

    # 核心逻辑：原始ID 不在 完成ID 里的，就是没跑的
    # 使用 ~ (取反) 和 isin
    mask = ~original_id_col.astype(str).isin(finished_ids)
    df_remaining = df_original[mask]
    
    # 2. 生成第二个文件：剩余任务大表
    remaining_file_path = os.path.join(OUTPUT_DIR, "todo_remaining_full.csv")
    df_remaining.to_csv(remaining_file_path, index=False)
    
    print(f"✅ [文件2] 已生成：{remaining_file_path}")
    print(f"   原始总数: {len(df_original)}")
    print(f"   剔除已跑: {len(df_original) - len(df_remaining)}")
    print(f"   剩余待跑: {len(df_remaining)}")

    print("\n=== 完成！接下来你可以把 [文件2] 切分发给不同电脑跑了 ===")

if __name__ == "__main__":
    main()