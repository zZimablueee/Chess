#after running pawn_centers.py
#Capture Ratio and Pawn Center
#从数据库表格CaptureNPawn中读取游戏形式的数据，将其按照选手重新组合整理
#再把整理后的表格新增到players表格中
import pandas as pd
import sqlite3
from tqdm import tqdm

db_file=r"C:\sqlite3\chess.db"
table_name='CaptureNPawn'
conn=sqlite3.connect(db_file)
cursor=conn.cursor()
df=pd.read_sql_query(f"SELECT * FROM {table_name}",conn)
cap_split=df['CaptureRatio'].str.split(",",expand=True).apply(pd.to_numeric, errors='coerce')
pawn_split=df['PawnCenter'].str.split(",",expand=True).apply(pd.to_numeric, errors='coerce')
df_ea=pd.DataFrame({
    'uid':df['uid'],
    'BlackCaptureRatio':cap_split[0],
    'WhiteCaptureRatio':cap_split[1],
    'BlackPawnCenter':pawn_split[0],
    'WhitePawnCenter':pawn_split[1],
})
uid_name_table='games'
uid_name=pd.read_sql_query(f"SELECT uid,White,Black FROM {uid_name_table}",conn)
df_ea=pd.merge(df_ea,uid_name[['uid','Black','White']],on='uid',how='left') #保留df_ea里的所有行
df_ea=pd.concat([
    df_ea[['White','WhiteCaptureRatio','WhitePawnCenter']].rename(columns={
        'White':'user',
        'WhiteCaptureRatio':'CaptureRatio',
        'WhitePawnCenter':'PawnCenter'
    }),
    df_ea[['Black','BlackCaptureRatio','BlackPawnCenter']].rename(columns={
        'Black':'user',
        'BlackCaptureRatio':'CaptureRatio',
        'BlackPawnCenter':'PawnCenter'
    })
],axis=0)
cols=['CaptureRatio','PawnCenter']
df_ea[cols]=df_ea[cols].apply(pd.to_numeric,errors='coerce')
df_mean=df_ea.groupby('user',as_index=False).mean(numeric_only=True)

cursor.execute("PRAGMA table_info(players);")
existing_cols = [x[1] for x in cursor.fetchall()]
if 'CaptureRatio' not in existing_cols:
    cursor.execute("ALTER TABLE players ADD COLUMN CaptureRatio REAL;")
if 'PawnCenter' not in existing_cols:
    cursor.execute("ALTER TABLE players ADD COLUMN PawnCenter REAL;")
conn.commit()

chunksize = 2000
n = len(df_mean)
update_sql = """
UPDATE players
SET CaptureRatio = ?, PawnCenter = ?
WHERE user = ?;
"""

for start in range(0, n, chunksize):
    chunk = df_mean.iloc[start:start+chunksize].copy()
    rows = chunk[['CaptureRatio', 'PawnCenter', 'user']].where(pd.notnull(chunk), None).values.tolist()

    cursor.executemany(update_sql, rows)
    conn.commit()
    print(f"已更新 chunk rows {start}..{min(start+chunksize, n)-1} ({len(rows)} 行) 到 players")

print("所有玩家数据已成功更新到 players 表")


