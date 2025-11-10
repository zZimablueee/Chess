#在运行stockfish分析前需要从数据库中把带有uid的文件拿出来
import pandas as pd
import sqlite3
data_base=r"C:\sqlite3\chess.db"
table_name='games'
conn=sqlite3.connect(data_base)
cursor=conn.cursor()
df=pd.read_sql(f"SELECT * FROM {table_name}",conn)
df.to_csv("100wuid.csv")
# 保存地址"C:\Users\Administrator\Desktop\playerstyles\100wuid.csv"
