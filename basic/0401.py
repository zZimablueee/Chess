import sqlite3
import pandas as pd

# ================================
# 🔧 需要你修改的地方
# ================================
DB_PATH = r"C:\sqlite3\chess.db"        # 数据库路径
TABLE_NAME = "games"           # 表名
CSV_PATH = r"C:\Users\Administrator\Desktop\FINALL\RESULTS\06.csv"       # 已跑结果CSV
OUTPUT_PATH = r"C:\Users\Administrator\Desktop\FINALL\RESULTS\games_to_rerun.csv"  # 输出补跑文件

# ================================
# 📥 读取数据库
# ================================
conn = sqlite3.connect(DB_PATH)

df_db = pd.read_sql_query(f"""
SELECT *
FROM {TABLE_NAME}
""", conn)

conn.close()

# ================================
# 📥 读取CSV
# ================================
df_csv = pd.read_csv(CSV_PATH)

# ================================
# 🧼 数据清洗（防坑）
# ================================

# UID统一为字符串
df_db['uid'] = df_db['uid'].astype(str)
df_csv['UID'] = df_csv['UID'].astype(str)

# Moves列处理（防止 "" / "NULL"）
df_db['Moves'] = df_db['Moves'].replace("", pd.NA)
df_db['Moves'] = df_db['Moves'].replace("NULL", pd.NA)

# 去重（防止重复UID影响统计）
df_db = df_db.drop_duplicates(subset=['uid'])
df_csv = df_csv.drop_duplicates(subset=['UID'])

# ================================
# ✅ 任务1：统计不需要分析的游戏
# ================================
num_no_moves = df_db['Moves'].isna().sum()

# ================================
# ✅ 任务2：找需要补跑的游戏
# ================================

# 有Moves的（理论上应该被分析的）
df_valid = df_db[df_db['Moves'].notna()]

# 已跑完的UID集合
uid_done = set(df_csv['UID'])

# 需要补跑的
df_missing = df_valid[~df_valid['uid'].isin(uid_done)]

# ================================
# 📊 打印统计信息（强烈建议看）
# ================================
print("===================================")
print(f"数据库总游戏数: {len(df_db)}")
print(f"Moves为空（不需要分析）: {num_no_moves}")
print(f"有Moves（应分析）: {len(df_valid)}")
print(f"已完成（CSV）: {len(df_csv)}")
print(f"需要补跑: {len(df_missing)}")
print("===================================")

# sanity check
print("Sanity Check（应接近）:")
print(f"已完成 + 补跑 = {len(df_csv) + len(df_missing)}")
print(f"应分析总数 = {len(df_valid)}")

# ================================
# 📤 导出补跑数据
# ================================
df_missing.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ 已生成补跑文件: {OUTPUT_PATH}")