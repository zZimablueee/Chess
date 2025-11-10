#PawnCenter (x,x) CaptureRatio(x,x)
#增加数据库中两部分的更新 PawnCenterAdvance好CaptureRatio，由capture_n_pawn.ipynb改来
#Capture part also be calculated in "capture_check_force.py", that part can be commented out
import chess
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import sys

#capture rate:
#(black's,white's)
def get_captures_fractions(moves_str:Optional[str]) -> Optional[Tuple[float, float]]:
    if not moves_str:
        return None
    
    board = chess.Board() 
    num_moves = [0, 0] 
    num_captures = [0, 0]

    #字符串变为列表
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

    capture_black=num_captures[0] / num_moves[0] if num_moves[0]>0 else 0
    capture_white=num_captures[1] / num_moves[1] if num_moves[1]>0 else 0
    return (capture_black, capture_white)

db_path=r"C:\sqlite3\chess.db"
conn=sqlite3.connect(db_path,timeout=15)
cur=conn.cursor()
#如果要更新整个表重算就运行
cur.execute("DROP TABLE IF EXISTS CaptureNPawn")
cur.execute("""CREATE TABLE CaptureNPawn(
            uid INTEGER PRIMARY KEY,
            CaptureRatio TEXT
            )
            """)
conn.commit()
cur.execute("SELECT uid,Moves FROM games")
inserts0=[]
rows=cur.fetchall()
for rowid,moves in tqdm(rows,desc="Calculating CaptureRatio:"):
    ratios = get_captures_fractions(moves)
    if ratios is not None:
        ratio_str = f"{ratios[0]:.3f},{ratios[1]:.3f}"
    else:
        ratio_str = None
    inserts0.append((rowid,ratio_str))
cur.executemany("INSERT INTO CaptureNPawn (uid,CaptureRatio) VALUES (?,?)", inserts0)
conn.commit()
conn.close()

def pawn_center_moves(moves_str:Optional[str]) -> Optional[Tuple[float, float]]:
    if not moves_str:
        return None
    CENTER_SQUARES = [chess.D4, chess.E4, chess.D6, chess.E6]
    #can change
    board=chess.Board()
    num_moves = [0, 0] 
    center_counts=[0,0]
    moves_list=moves_str.split(',')
    for move_str in moves_list:
        move_str=move_str.strip()
        if not move_str:
            continue
        try:
            move=chess.Move.from_uci(move_str)
        except Exception:
            continue
        if move not in board.legal_moves:
            continue

        player_idx = 1 if board.turn else 0  
        num_moves[player_idx] += 1

        board.push(move)

        for sq in CENTER_SQUARES:
            piece=board.piece_at(sq)
            if piece and piece.piece_type==chess.PAWN:
                if piece.color==chess.WHITE:
                    num_moves[1] += 1 
                    center_counts[1]+=1
                else:
                    center_counts[0]+=1
                    num_moves[0] += 1  

    center_black=center_counts[0]/num_moves[0] if num_moves[0] > 0 else 0                  
    center_white=center_counts[1]/num_moves[1] if num_moves[1] > 0 else 0              
    
    return (center_black,center_white)

conn = sqlite3.connect(db_path, timeout=30)
cur = conn.cursor()
# 如果"PawnCenter" 列还没加，就加它
cur.execute('PRAGMA table_info("CaptureNPawn")')
columns = [info[1] for info in cur.fetchall()] 
if "PawnCenter" not in columns:
    cur.execute('ALTER TABLE "CaptureNPawn" ADD COLUMN "PawnCenter" TEXT')
    conn.commit()
    print('列 "PawnCenter" 已添加')
else:
    print('列 "PawnCenter" 已存在，跳过添加')
cur.execute('SELECT uid, Moves FROM games')
rows = cur.fetchall()
batch_size = 500
for i, (uid, moves) in enumerate(rows, 1):
    value = pawn_center_moves(moves)
    value_str = None if value is None else f"{value[0]:.3f},{value[1]:.3f}"
    cur.execute('UPDATE CaptureNPawn SET "PawnCenter"=? WHERE uid=?', (value_str, uid))
    
    if i % batch_size == 0:
        conn.commit()  #分批提交

conn.commit()
conn.close()

print("批量更新完成")

