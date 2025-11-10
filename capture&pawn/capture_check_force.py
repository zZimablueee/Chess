#CaptureRatio CheckRatio ForceRatio
from typing import Tuple, List, Dict, Optional,Callable,Any
import chess
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
db_file=r'C:\sqlite3\chess.db'

def get_tactical_features(moves_str: Optional[str]) -> Optional[Tuple[float, float, float, float, float, float]]:
    if not moves_str:
        return None
    board = chess.Board()
    num_moves = [0, 0]       # 黑方总步数, 白方总步数
    num_captures = [0, 0]    # 黑方吃子数, 白方吃子数
    num_checks = [0, 0]      # 黑方将军数, 白方将军数
    num_forcing = [0, 0]     # 黑方唯一应对数, 白方唯一应对数
    moves_list=moves_str.split(',')

    for move_str in moves_list:
        move = chess.Move.from_uci(move_str.strip())
        if move not in board.legal_moves:
            continue
        player_idx = 1 if board.turn else 0
        #board.turn==True 该白方走子 索引为1 刚刚走的是黑子
        num_moves[player_idx] += 1 
        if board.is_capture(move): #是否吃子
            num_captures[player_idx] += 1
        board.push(move)
        if board.is_check():
            num_checks[player_idx]+=1
        if board.legal_moves.count()==1:
            num_forcing[player_idx]+=1

    capture_black=num_captures[0] / num_moves[0] if num_moves[0]>0 else 0
    capture_white=num_captures[1] / num_moves[1] if num_moves[1]>0 else 0
    check_black = num_checks[0] / num_moves[0] if num_moves[0] > 0 else 0
    check_white = num_checks[1] / num_moves[1] if num_moves[1] > 0 else 0
    force_black = num_forcing[0] / num_moves[0] if num_moves[0] > 0 else 0
    force_white = num_forcing[1] / num_moves[1] if num_moves[1] > 0 else 0
    return (capture_black,capture_white,check_black,check_white,force_black,force_white)

def column_exists(conn, table_name, col_name):
    cursor = conn.execute(f'PRAGMA table_info({table_name})')
    cols = [row[1] for row in cursor.fetchall()]
    return col_name in cols

conn = sqlite3.connect(db_file)

process_df = pd.read_sql("SELECT uid, Moves FROM CaptureNPawn", conn)

#新列
new_cols = ["CaptureRatio", "CheckRatio", "ForceRatio"]
for col in new_cols:
    if not column_exists(conn, "CaptureNPawn", col):
        conn.execute(f'ALTER TABLE CaptureNPawn ADD COLUMN "{col}" TEXT')

for idx, row in tqdm(process_df.iterrows(),total=len(process_df),desc="Updating table CaptureNPawn"):
    feats = get_tactical_features(row["Moves"])
    if feats:
        cb, cw, chb, chw, fb, fw = feats
        conn.execute("""
            UPDATE CaptureNPawn
            SET "CaptureRatio" = ?,
                "CheckRatio" = ?,
                "ForceRatio" = ?
            WHERE uid = ?
        """, (f"{cb:.3f},{cw:.3f}",
              f"{chb:.3f},{chw:.3f}",
              f"{fb:.3f},{fw:.3f}",
             row["uid"]))

conn.commit()
conn.close()

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

process = pd.read_sql_query("SELECT * FROM CaptureNPawn", conn)

process[['CaptureRatioBlack', 'CaptureRatioWhite']] = process['CaptureRatio'].str.split(',', expand=True).astype(float)
black_capture = process[['Black', 'CaptureRatioBlack']].rename(columns={'Black': 'Player', 'CaptureRatioBlack': 'Capture Ratio'})
white_capture = process[['White', 'CaptureRatioWhite']].rename(columns={'White': 'Player', 'CaptureRatioWhite': 'Capture Ratio'})
all_capture = pd.concat([black_capture, white_capture], axis=0)
avg_capture = all_capture.groupby('Player')['Capture Ratio'].mean().reset_index()
avg_capture.columns = ['Player', 'CaptureRatio']

try:
    cursor.execute("ALTER TABLE players ADD COLUMN CaptureRatio REAL")
except sqlite3.OperationalError:
    # 如果列已经存在，就跳过
    pass
cursor.execute("PRAGMA index_list(players)")
indexes=[row[1] for row in cursor.fetchall()]
if "idx_players_player" not in indexes:
    print("创建索引 idx_players_player ...")
    cursor.execute("CREATE INDEX idx_players_player ON players(user)")

avg_capture.to_sql("tmp_capture", conn, if_exists="replace", index=False)

cursor.execute("""
    UPDATE players
    SET CaptureRatio = (
        SELECT CaptureRatio
        FROM tmp_capture
        WHERE tmp_capture.Player = players.user
    )
    WHERE EXISTS (
        SELECT 1
        FROM tmp_capture
        WHERE tmp_capture.Player = players.user
    )
""")

conn.commit()
conn.close()
print("玩家CaptureRatio已更新完成")

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

process = pd.read_sql_query("SELECT * FROM CaptureNPawn", conn)

process[['CheckRatioBlack', 'CheckRatioWhite']] = process['CheckRatio'].str.split(',', expand=True).astype(float)
black_CheckRatio = process[['Black', 'CheckRatioBlack']].rename(columns={'Black': 'Player', 'CheckRatioBlack': 'Check Ratio'})
white_CheckRatio = process[['White', 'CheckRatioWhite']].rename(columns={'White': 'Player', 'CheckRatioWhite': 'Check Ratio'})
all_CheckRatio = pd.concat([black_CheckRatio, white_CheckRatio], axis=0)
avg_checkratio = all_CheckRatio.groupby('Player')['Check Ratio'].mean().reset_index()
avg_checkratio.columns = ['Player', 'CheckRatio']

try:
    cursor.execute("ALTER TABLE players ADD COLUMN CheckRatio REAL")
except sqlite3.OperationalError:
    # 如果列已经存在，就跳过
    pass
cursor.execute("PRAGMA index_list(players)")
indexes=[row[1] for row in cursor.fetchall()]
if "idx_players_player" not in indexes:
    print("创建索引 idx_players_player ...")
    cursor.execute("CREATE INDEX idx_players_player ON players(user)")

avg_checkratio.to_sql("tmp_checkratio", conn, if_exists="replace", index=False)

cursor.execute("""
    UPDATE players
    SET CheckRatio = (
        SELECT CheckRatio
        FROM tmp_checkratio
        WHERE tmp_checkratio.Player = players.user
    )
    WHERE EXISTS (
        SELECT 1
        FROM tmp_checkratio
        WHERE tmp_checkratio.Player = players.user
    )
""")

conn.commit()
conn.close()
print("玩家CheckRatio 已更新完成")

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

process = pd.read_sql_query("SELECT * FROM CaptureNPawn", conn)

process[['ForceRatioBlack', 'ForceRatioWhite']] = process['ForceRatio'].str.split(',', expand=True).astype(float)
black_forceratio = process[['Black', 'ForceRatioBlack']].rename(columns={'Black': 'Player', 'ForceRatioBlack': 'ForceRatio'})
white_forceratio = process[['White', 'ForceRatioWhite']].rename(columns={'White': 'Player', 'ForceRatioWhite': 'ForceRatio'})
all_forceratio = pd.concat([black_forceratio, white_forceratio], axis=0)
avg_forceratio = all_forceratio.groupby('Player')['ForceRatio'].mean().reset_index()
avg_forceratio.columns = ['Player', 'ForceRatio']

try:
    cursor.execute("ALTER TABLE players ADD COLUMN ForceRatio REAL")
except sqlite3.OperationalError:
    # 如果列已经存在，就跳过
    pass
cursor.execute("PRAGMA index_list(players)")
indexes=[row[1] for row in cursor.fetchall()]
if "idx_players_player" not in indexes:
    print("创建索引 idx_players_player ...")
    cursor.execute("CREATE INDEX idx_players_player ON players(user)")

avg_forceratio.to_sql("tmp_forceratio", conn, if_exists="replace", index=False)

cursor.execute("""
    UPDATE players
    SET ForceRatio = (
        SELECT ForceRatio
        FROM tmp_forceratio
        WHERE tmp_forceratio.Player = players.user
    )
    WHERE EXISTS (
        SELECT 1
        FROM tmp_forceratio
        WHERE tmp_forceratio.Player = players.user
    )
""")

conn.commit()
conn.close()
print("玩家 ForceRatio 已更新完成")
#drop temp tables in SQL
