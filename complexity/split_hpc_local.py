import pandas as pd
import os

# ================= 配置区域 (请修改这里) =================
# 1. 刚才生成的“剩余任务大表”的路径
INPUT_FILE = r"C:\Users\Administrator\Desktop\Chess\data\processed\remaining\todo_remaining_full.csv"

# 2. [修改] 指定 HPC 任务文件的保存文件夹
#    如果没有这个文件夹，脚本会自动帮你创建
HPC_SAVE_DIR = r"C:\Users\Administrator\Desktop\Chess\data\processed\remaining\remain_for_hpc"

# 3. [修改] 指定 本地 任务文件的保存文件夹
LOCAL_SAVE_DIR = r"C:\Users\Administrator\Desktop\Chess\data\processed\remaining\remain_for_local"

# 4. 切分给 HPC 的比例 (0.2 表示 20% 给 HPC)
# 如果想让大家同时跑完，建议 HPC 分少点 (比如 0.2 或 0.3)
HPC_RATIO = 0.7 
# ====================================================

def main():
    print(f"正在读取大文件: {INPUT_FILE} ...")
    
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ 找不到文件！请检查路径。")
        return

    total_rows = len(df)
    print(f"✅ 读取成功，剩余总任务数: {total_rows} 局")

    # 自动创建文件夹 (防止报错)
    if not os.path.exists(HPC_SAVE_DIR):
        os.makedirs(HPC_SAVE_DIR)
        print(f"📂 已自动新建文件夹: {HPC_SAVE_DIR}")
        
    if not os.path.exists(LOCAL_SAVE_DIR):
        os.makedirs(LOCAL_SAVE_DIR)
        print(f"📂 已自动新建文件夹: {LOCAL_SAVE_DIR}")

    # 计算切分点
    split_index = int(total_rows * HPC_RATIO)

    # 切分数据
    # Part 1: 给 HPC 的
    df_hpc = df.iloc[:split_index]
    
    # Part 2: 留给自己的
    df_local = df.iloc[split_index:]

    # [关键修改] 使用各自指定的目录来拼接路径
    hpc_filename = os.path.join(HPC_SAVE_DIR, "task_for_hpc.csv")
    local_filename = os.path.join(LOCAL_SAVE_DIR, "task_for_local.csv")

    print("\n正在保存文件...")
    df_hpc.to_csv(hpc_filename, index=False)
    df_local.to_csv(local_filename, index=False)

    print("-" * 30)
    print("🎉 切分完成！")
    print(f"1. 📦 [发给 HPC] : {hpc_filename}")
    print(f"   包含 {len(df_hpc)} 局 (约占比 {HPC_RATIO*100}%)")
    print("-" * 30)
    print(f"2. 🏎️ [留给自己] : {local_filename}")
    print(f"   包含 {len(df_local)} 局 (约占比 {(1-HPC_RATIO)*100}%)")

if __name__ == "__main__":
    main()