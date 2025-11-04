#运行完lifespan.py后运行
import pandas as pd
import sqlite3
df=pd.read_csv(r"C:\Users\Administrator\Desktop\playerstyles\lifespans_features_combined.csv")
df = df.rename(columns={"game_id": "uid"})
cols_to_front=["uid","player"]
new_order = cols_to_front + [col for col in df.columns if col not in cols_to_front]
df = df[new_order]
df = df.sort_values(by="uid", ascending=True)
df.to_csv("lifespan_reordered.csv", index=False)
gamesspan=pd.read_csv("lifespan_reordered.csv")
csv_path="lifespan_reordered.csv"
db_path=r"C:\sqlite3\chess.db"
conn=sqlite3.connect(db_path)
cursor=conn.cursor()
cursor.execute("PRAGMA journal_mode=WAL;")
new_table="lifespan"
cursor.execute(f'DROP TABLE IF EXISTS "{new_table}";')
conn.commit()
first_chunk = True
for chunk in pd.read_csv(csv_path, chunksize=2000, dtype=str):
    if first_chunk:
        cols = [f'"{c}" TEXT' for c in chunk.columns]
        col_def = ", ".join(cols)
        create_sql = f'CREATE TABLE "{new_table}" ({col_def});'
        cursor.execute(create_sql)
        conn.commit()
        first_chunk = False
        print(f"已创建新表 {new_table}，包含 {len(chunk.columns)} 列")

        placeholders = ", ".join(["?"] * len(chunk.columns))
        insert_sql = f'INSERT INTO "{new_table}" VALUES ({placeholders})'
    cursor.executemany(insert_sql, chunk.values.tolist())
    conn.commit()
    print(f"已插入 {len(chunk)} 行")

conn.close()
print(f"CSV 导入完成，已写入数据库表 {new_table}")