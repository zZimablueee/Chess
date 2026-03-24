import sqlite3
import chess
import pandas as pd
import numpy as np
from mpi4py import MPI
from tqdm import tqdm  # 引入进度条库

DB_PATH = r'C:\sqlite3\chess.db' 
SOURCE_TABLE = 'games'
TARGET_TABLE = 'checkmove'

def process_game(uid, moves_string, result_str):
    """提取 Check 和 Checkmate 的核心逻辑"""
    if pd.isna(moves_string) or not isinstance(moves_string, str):
        return None
        
    board = chess.Board()
    move_list = [m.strip() for m in moves_string.strip().split(',') if m.strip()]
    
    if not move_list:
        return None
        
    total_plies = len(move_list)
    check_plies = []
    checkmate_plies = []
    
    for ply_count, move_str in enumerate(move_list, start=1):
        try:
            move = chess.Move.from_uci(move_str)
            board.push(move)
            
            if board.is_checkmate():
                checkmate_plies.append(ply_count)
            elif board.is_check():
                check_plies.append(ply_count)
                
        except ValueError:
            total_plies = ply_count - 1
            break 
            
    if not check_plies and not checkmate_plies:
        return None
        
    return {
        'uid': uid,
        'Result': result_str,
        'total_plies': total_plies,
        'check_moves': str(check_plies),
        'checkmate_moves': str(checkmate_plies)
    }

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    chunks = None
    
    if rank == 0:
        print(f"[*] 启动 MPI 任务，总进程数: {size}")
        print(f"[*] 正在读取数据库: {DB_PATH}")
        
        try:
            # 原生 sqlite3 连接
            conn = sqlite3.connect(DB_PATH)
            query = f"SELECT uid, Result, Moves FROM {SOURCE_TABLE}"
            df = pd.read_sql_query(query, conn)
            conn.close() # 读完马上关掉，释放文件锁
            
            print(f"[*] 成功读取 {len(df)} 行数据，准备分发...")
            chunks = np.array_split(df, size)
            
        except Exception as e:
            print(f"[!] 数据库读取失败: {e}")
            comm.Abort()

    local_df = comm.scatter(chunks, root=0)
    local_results = []
    
    # position=rank 确保不同进程的进度条在终端里各占一行，不会互相覆盖
    for index, row in tqdm(local_df.iterrows(), total=len(local_df), desc=f"进程 {rank}", position=rank, leave=True):
        res = process_game(row['uid'], row['Moves'], row['Result'])
        if res is not None:
            local_results.append(res)

    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        print("\n" * size) 
        print("[*] 所有进程计算完毕，正在汇总并写入数据库...")
        
        final_list = [item for sublist in gathered_results for item in sublist]
        
        if final_list:
            final_df = pd.DataFrame(final_list)
            try:
                # 原生 sqlite3 写入
                conn = sqlite3.connect(DB_PATH)
                final_df.to_sql(name=TARGET_TABLE, con=conn, if_exists='replace', index=False, chunksize=10000)
                conn.close()
                print(f"[*] 大功告成！共 {len(final_df)} 条结果已成功写入表 `{TARGET_TABLE}`。")
            except Exception as e:
                print(f"[!] 写入失败: {e}")
                final_df.to_csv('emergency_backup.csv', index=False)
                print("[!] 结果已备份至 emergency_backup.csv")
        else:
            print("[-] 没有符合条件的数据。")

if __name__ == '__main__':
    main()

# mpiexec -n 4 python C:\Users\Administrator\Desktop\Chess\basic\mate\checkmate.py