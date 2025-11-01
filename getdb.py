#将csv文件放到数据库里
import sqlite3
import pandas as pd
import os

csv_path=r"C:\Users\Administrator\Desktop\titled-tuesday\basicgames.csv"       #CSV路径
db_path =r"C:\sqlite3\chess.db"        #要创建的数据库文件
table_name="games"

df = pd.read_csv(csv_path)

if not os.path.exists(db_path):
    open(db_path, 'w').close()

conn = sqlite3.connect(db_path)

df.to_sql(table_name, conn, if_exists='replace', index=False)

conn.close()

print(f"已将 {csv_path} 导入到 {db_path} 中的表 {table_name}。")