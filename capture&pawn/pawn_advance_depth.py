#我怎么之前写了和pawn相关那么多函数...
import pandas as pd
import sqlite3
db_file=r"C:\sqlite3\chess.db"
batch_size=3000

#重复跑时同样的列避免加后缀导致报错
def ensure_column_exists(conn, table, column, dtype):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [col[1] for col in cursor.fetchall()]
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
    else:
        print(f"列 '{column}' 已存在，跳过添加")
#把games表格里的Moves和黑白方 按照uid合并至表格CaptureNPawn中
def merge_moves_to_CP(db_file):
    batch_count=0
    conn=sqlite3.connect(db_file)
    cursor=conn.cursor()
    games_df=pd.read_sql_query("SELECT uid,Moves,White,Black FROM games",conn)
    capturenpawn_df=pd.read_sql_query("SELECT * FROM CaptureNPawn",conn)

    # 检查并添加列 Moves, White, Black（如果这些列不存在的话）
    cursor.execute("PRAGMA table_info(CaptureNPawn)")  #获取 CaptureNPawn 表的列名
    columns = [column[1] for column in cursor.fetchall()]  
    #更新表格CaptureNPawn前确保有列，字段有处可写
    if 'Moves' not in columns:
        cursor.execute("ALTER TABLE CaptureNPawn ADD COLUMN Moves TEXT")
    if 'White' not in columns:
        cursor.execute("ALTER TABLE CaptureNPawn ADD COLUMN White TEXT")
    if 'Black' not in columns:
        cursor.execute("ALTER TABLE CaptureNPawn ADD COLUMN Black TEXT")
    merged_df = pd.merge(capturenpawn_df, games_df[['uid', 'Moves', 'White', 'Black']], on='uid', how='left', suffixes=('_cp', '_game'))
    print(merged_df.columns)
    print(merged_df.head())
    merged_df.columns = merged_df.columns.str.strip()
    merged_df['Moves'] = merged_df['Moves_game']
    merged_df['White'] = merged_df['White_game']
    merged_df['Black'] = merged_df['Black_game']
    merged_df.drop(columns=['Moves_game', 'White_game', 'Black_game'], inplace=True)

    for _,row in merged_df.iterrows():
        cursor.execute(
            '''UPDATE CaptureNPawn SET Moves = ?, White = ?, Black = ? WHERE uid=?''',
            (row['Moves'],row['White'],row['Black'],row['uid']) #要传递tuple
        )
        batch_count+=1
        if batch_count % batch_size == 0:  # 每处理批量大小的数据就提交并打印进度
            conn.commit()
            print(f"Updated {batch_count} rows...")
    conn.commit()
    print(f"Finishing updating Moves and players' names into CaptureNPawn's {batch_size} rows!")
    conn.close()
merge_moves_to_CP(db_file)

#计算每局游戏双方的PawnAdvanceDepth 
table_name='CaptureNPawn'
def pawn_advance_score(moves_str):
    if not moves_str or pd.isna(moves_str):
        return None, None

    white_score = 0
    black_score = 0
    moves = moves_str.split(',')

    for move in moves:
        if len(move) != 4:
            continue

        from_sq = move[:2]
        to_sq = move[2:]

        from_file, from_rank = from_sq[0], int(from_sq[1])
        to_file, to_rank = to_sq[0], int(to_sq[1])

        if from_file == to_file:
            delta = to_rank - from_rank
            if from_rank in [2, 3, 4, 5, 6] and delta > 0:
                white_score += delta
            elif from_rank in [7, 6, 5, 4, 3] and delta < 0:
                black_score += abs(delta)

    return white_score, black_score
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
try:
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN WhitePawnAdvanceDepth INTEGER")
except sqlite3.OperationalError:
    pass 

try:
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN BlackPawnAdvanceDepth INTEGER")
except sqlite3.OperationalError:
    pass 
#将PawnAdvanceDepth更新到游戏层面的表格 CaptureNPawn 中 
ensure_column_exists(conn, 'players', 'AveragePawnAdvanceDepth', 'REAL')
df = pd.read_sql_query(f"SELECT uid, Moves FROM {table_name}", conn)
df[['WhitePawnAdvanceDepth', 'BlackPawnAdvanceDepth']] = df['Moves'].apply(lambda x: pd.Series(pawn_advance_score(x)))
for _, row in df.iterrows():
    cursor.execute(
        f"""
        UPDATE {table_name}
        SET WhitePawnAdvanceDepth = ?, BlackPawnAdvanceDepth = ?
        WHERE uid = ?
        """,
        (row['WhitePawnAdvanceDepth'], row['BlackPawnAdvanceDepth'], row['uid'])
    )
conn.commit()
#从表格 CaptureNPawn 中的 xxPawnAdvanceDepth 列，总结到players层面，并更新player表格
df=pd.read_sql(f"SELECT * FROM {table_name}",conn)
white_depth_df=df[['White','WhitePawnAdvanceDepth']].dropna()
white_depth_df.columns=['user','PawnAdvanceDepth']
black_depth_df=df[['Black','BlackPawnAdvanceDepth']].dropna()
black_depth_df.columns=['user','PawnAdvanceDepth']
depth_df=pd.concat([white_depth_df,black_depth_df],ignore_index=True)
player_depth_df=(
    depth_df.groupby('user')
    .agg(AveragePawnAdvanceDepth=('PawnAdvanceDepth', 'mean'))
    .reset_index()
)
df_players = pd.read_sql_query("SELECT * FROM players", conn)
player_depth_df = player_depth_df.rename(columns={'PawnAdvanceDepth': 'AveragePawnAdvanceDepth'}) #确保player_depth_df列名一致
df_merged = pd.merge(df_players, player_depth_df, on='user', how='left', suffixes=('_left', '_right'))
df_merged.columns = df_merged.columns.str.strip()  # 去除空格
# 选择正确的列并删除冗余的带后缀的列
df_merged['PawnAdvanceDepth'] = df_merged['AveragePawnAdvanceDepth_right']  # 选择右侧的列
df_merged.drop(columns=['AveragePawnAdvanceDepth_left', 'AveragePawnAdvanceDepth_right'], inplace=True)
print(df_merged.columns)  #合并后的列名
print(df_merged.head())  #看数据

def ensure_column_exists(conn, table, column, dtype):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [col[1] for col in cursor.fetchall()]
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
    else:
        print(f"列 '{column}' 已存在，跳过添加")
# 确保列存在
ensure_column_exists(conn, 'players', 'AveragePawnAdvanceDepth', 'REAL')

# 用正确列名写入
batch_count = 0
for _, row in df_merged.iterrows():
    if pd.notna(row['PawnAdvanceDepth']):
        cursor.execute(
            "UPDATE players SET AveragePawnAdvanceDepth = ? WHERE user = ?",
            (row['PawnAdvanceDepth'], row['user'])  # 注意这里用 PawnAdvanceDepth，而不是不存在的 AveragePawnAdvanceDepth
        )
        batch_count += 1
        if batch_count % batch_size == 0:
            conn.commit()
            print(f"Updated {batch_count} rows...")
conn.commit()
print(f"Finished updating {batch_count} rows of AveragePawnAdvanceDepth in players table")

conn.close()