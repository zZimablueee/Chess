import pandas as pd
import os
import math

# ================= 配置区域 =================
# 1. 你的“法拉利”任务大文件路径
INPUT_FILE = r"C:\Users\Administrator\Desktop\Chess\data\processed\remaining\remain_for_local\task_for_local.csv"

# 2. 你打算分成几份？(包括你自己)
# 例如：你自己 + 3个朋友 = 4份
NUM_PARTS = 4 

# 3. 切好的小文件放哪里？
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\Chess\data\processed\remaining\remain_for_local\local_split"
# ===========================================

def main():
    print(f"正在读取文件: {INPUT_FILE} ...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ 找不到文件！请检查路径。")
        return

    total_rows = len(df)
    print(f"✅ 读取成功，共 {total_rows} 局游戏。")
    print(f"🎯 准备将其平均切分为 {NUM_PARTS} 份。")

    # 自动创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 已新建输出文件夹: {OUTPUT_DIR}")

    # 计算每份大概多少行
    chunk_size = math.ceil(total_rows / NUM_PARTS)

    print("\n开始切分...")
    
    for i in range(NUM_PARTS):
        # 计算切片的起始和结束索引
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        # 截取数据 (Pandas 会自动处理越界的情况，最后一份会自动截断到末尾)
        df_part = df.iloc[start_idx : end_idx]
        
        # 生成文件名：part_1.csv, part_2.csv ...
        # 建议重命名给具体的人，比如 part_1_mine.csv
        file_name = f"friend_task_part_{i+1}.csv"
        save_path = os.path.join(OUTPUT_DIR, file_name)
        
        df_part.to_csv(save_path, index=False)
        
        print(f"   📄 [第 {i+1} 份] 生成完毕: {file_name}")
        print(f"      行数: {len(df_part)} 局")
        print(f"      范围: {start_idx} -> {min(end_idx, total_rows)}")

    print("-" * 30)
    print(f"🎉 全部完成！文件都在这里: {OUTPUT_DIR}")
    print("🚀 快把这些文件发给兄弟们吧！")

if __name__ == "__main__":
    main()