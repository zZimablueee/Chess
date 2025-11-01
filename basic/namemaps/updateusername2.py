import sqlite3
import pandas as pd
from tqdm import tqdm

# 读取映射关系
username_mapping_df = pd.read_csv(r"C:\Users\Administrator\Desktop\playerstyles\namemaps\username_to_name.csv")
username_to_name = dict(zip(username_mapping_df['Username'], username_mapping_df['Name']))
name_set = set(username_mapping_df['Name'])

conn = sqlite3.connect(r"C:\sqlite3\chess.db")
cursor = conn.cursor()

print("Creating indexes...")
# 创建索引以加速UPDATE（如果不存在）
cursor.execute("CREATE INDEX IF NOT EXISTS idx_white ON games(White);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_black ON games(Black);")
conn.commit()

print("Creating temporary mapping table...")
cursor.execute('DROP TABLE IF EXISTS temp_mapping;')
cursor.execute('CREATE TEMP TABLE temp_mapping(Username TEXT PRIMARY KEY, Name TEXT);')

# 批量插入映射数据
insert_data = [(username, name) for username, name in username_to_name.items()]
cursor.executemany('INSERT INTO temp_mapping (Username, Name) VALUES (?, ?);', insert_data)
conn.commit()

print("Updating White column...")
# 一次性更新White列
cursor.execute('''
    UPDATE games
    SET White = (SELECT Name FROM temp_mapping WHERE temp_mapping.Username = games.White)
    WHERE White IN (SELECT Username FROM temp_mapping);
''')
conn.commit()
print(f"Updated {cursor.rowcount} White records")

print("Updating Black column...")
# 一次性更新Black列
cursor.execute('''
    UPDATE games
    SET Black = (SELECT Name FROM temp_mapping WHERE temp_mapping.Username = games.Black)
    WHERE Black IN (SELECT Username FROM temp_mapping);
''')
conn.commit()
print(f"Updated {cursor.rowcount} Black records")

# 查找未更新的用户
print("Finding unupdated users...")
cursor.execute('''
    SELECT DISTINCT White FROM games 
    WHERE White NOT IN (SELECT Username FROM temp_mapping) 
    AND White NOT IN (SELECT Name FROM temp_mapping)
    UNION
    SELECT DISTINCT Black FROM games 
    WHERE Black NOT IN (SELECT Username FROM temp_mapping) 
    AND Black NOT IN (SELECT Name FROM temp_mapping)
''')

unupdated_users = {row[0] for row in cursor.fetchall()}

cursor.execute('DROP TABLE IF EXISTS temp_mapping;')
conn.close()

# 保存未更新的用户
with open("chess_unupdated_users.txt", "w", encoding='utf-8') as f:
    for user in sorted(unupdated_users):
        f.write(f"{user}\n")

print(f"Finished! Found {len(unupdated_users)} unupdated users")