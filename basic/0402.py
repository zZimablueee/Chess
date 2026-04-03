import pandas as pd

# 1. 读取数据 (请确保路径正确)
file_path = r"C:\Users\Administrator\Desktop\FINALL\RESULTS\games_to_rerun.csv" 
df = pd.read_csv(file_path)

# 2. 过滤掉 uid 为 81519 的行
# 注意：如果你的 uid 列名不是 'uid'，请替换为实际列名
original_count = len(df)
 

# 3. 检查是否成功删除
if len(df) < original_count:
    print(f"成功删除！剩余行数: {len(df)}")
else:
    print("未找到该 uid，请检查列名或数据类型。")

# 4. 保存回 CSV (建议先另存为一个新文件测试)
df.to_csv(r"C:\Users\Administrator\Desktop\FINALL\RESULTS\games_to_rerun2.csv" , index=False)