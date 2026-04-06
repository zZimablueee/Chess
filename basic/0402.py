import pandas as pd

# 1️⃣ 读取数据
file_path = r"C:\Users\Administrator\Desktop\FINALL\RESULTS\06.csv"
df = pd.read_csv(file_path)

# 2️⃣ 原始行数
original_rows = len(df)

# 3️⃣ 去重（保留第一次出现的）
df_dedup = df.drop_duplicates(subset='UID', keep='first')

# 4️⃣ 去重后行数
remaining_rows = len(df_dedup)

# 5️⃣ 删除的行数
deleted_rows = original_rows - remaining_rows

# 6️⃣ 检查是否还有重复 UID
duplicate_check = df_dedup['UID'].duplicated().any()

# 7️⃣ 输出结果
print(f"原始行数: {original_rows}")
print(f"删除行数: {deleted_rows}")
print(f"剩余行数: {remaining_rows}")
print(f"是否还有重复UID: {duplicate_check}")

# 8️⃣ 保存新文件（可选）
df_dedup.to_csv(r"C:\Users\Administrator\Desktop\FINALL\RESULTS\06.csv", index=False)