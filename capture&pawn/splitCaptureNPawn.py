#因为到尝试做分类器的阶段发现CaptureNPawn表格的几列合并着很不利于我扩充数据
import sqlite3
import pandas as pd
import numpy as np
db_path=r'C:\sqlite3\chess.db'
table_name='CaptureNPawn'
#CaptureRatio PawnCenter CheckRatio ForceRatio (a,b) 改成 Black_/White_xxx
conn = sqlite3.connect(db_path)
df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
cols_to_split = ["CaptureRatio", "PawnCenter", "CheckRatio", "ForceRatio"]

for col in cols_to_split:
    #拆成两列，转换为 float
    df[[f"Black_{col}", f"White_{col}"]] = df[col].str.split(",", expand=True).astype(float)

#如果想删除原始列（慎重选择）
#df = df.drop(columns=cols_to_split)

df.to_sql(table_name, conn, if_exists='replace', index=False)
print(f'finished')
conn.close()

df.head()