#孤立兵 重叠兵 互相保护的兵相关 
import chess
import chess.pgn
import pandas as pd
import random
import sqlite3
from tqdm import tqdm
from mpi4py import MPI
import os
import sys
import traceback

def safe_barrier():
    try:
        comm.Barrier()
        return True
    except Exception as e:
        print(f"Rank {rank}: Barrier failed with error: {e}", flush=True)
        return False

FEATURE_COLUMNS = [
    'uid',
    'PawnIsolateScoresList',
    'PawnIsolateScoreMean',
    'PawnOverlapScoresList',
    'PawnOverlapScoreMean',
    'PawnProtectScoresList',
    'PawnProtectScoreMean'
]

def ensure_column_order(df, columns):
    """Ensure DataFrame columns match the specified order"""
    #Keep only existing columns
    existing_cols = [col for col in columns if col in df.columns]
    #Append extra columns not in the predefined order
    extra_cols = [col for col in df.columns if col not in columns]
    return df[existing_cols + extra_cols]

def to_side(color):
    if color == 'white' or color == chess.WHITE:
        return chess.WHITE
    if color == 'black' or color == chess.BLACK:
        return chess.BLACK
    raise ValueError("color must be 'white' or 'black' or chess.WHITE/chess.BLACK")

def extract_features(game, game_id, player_id_white, player_id_black):
    board = game.board()

    features = {
        'white': initialize_features(),
        'black': initialize_features(),
        'game_id': game_id,
        'player_id_white': player_id_white,
        'player_id_black': player_id_black
    }

    for move in game.mainline_moves():
        board.push(move)
        
        if board.turn != chess.WHITE:
            features['white'] = update_features(features['white'], board, "white",move)
        else:
            features['black'] = update_features(features['black'], board, 'black',move)

    features['white']['game_id'] = game_id
    features['white']['player_id'] = player_id_white
    features['black']['game_id'] = game_id
    features['black']['player_id'] = player_id_black

    features['white']['PawnIsolateScoreMean']=compute_mean_score(features['white'], 'PawnIsolateScoresList')
    features['black']['PawnIsolateScoreMean']=compute_mean_score(features['black'], 'PawnIsolateScoresList')
    features['white']['PawnOverlapScoreMean']=compute_mean_score(features['white'], 'PawnOverlapScoresList')
    features['black']['PawnOverlapScoreMean']=compute_mean_score(features['black'], 'PawnOverlapScoresList')
    features['white']['PawnProtectScoreMean']=compute_mean_score(features['white'], 'PawnProtectScoresList')
    features['black']['PawnProtectScoreMean']=compute_mean_score(features['black'], 'PawnProtectScoresList')
    
    return features['white'], features['black']

def initialize_features():
    return {
        'PawnIsolateScoresList': [],
        'PawnOverlapScoresList': [],
        'PawnProtectScoresList':[]
    }

def update_features(feature_dict, board,color,move):
    color_chess = to_side(color)
    feature_dict['PawnIsolateScoresList'].append(total_pawns_structure_score_for_color(board,color_chess))
    feature_dict['PawnOverlapScoresList'].append(total_overlapping_pawns_score_for_color(board,color_chess,penalty=-1))
    feature_dict['PawnProtectScoresList'].append(total_protected_pawn_score_for_color(board,color_chess))
    return feature_dict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def compute_mean_score(feature_dict, key):
    scores = feature_dict.get(key, [])
    return sum(scores) / len(scores) if scores else 0

#Determine whether a pawn is isolated
# an isolated pawn (also known as an isolani) is
# a pawn that cannot be supported or protected by another pawn 
def is_isolated_pawn(board,square,color):
    pawn_squares=board.pieces(chess.PAWN,color)
    direction = 1 if color == chess.WHITE else -1
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    support_squares=[]
    for df in [-1,1]:
        f=file+df
        r=rank-direction
        if 0<=f<=7 and 0<=r<=7:
            support_squares.append(chess.square(f,r))
    for sq in support_squares:
        if sq in pawn_squares:
            return False
    return True
#determine if a pawn is a passed pawn
def is_passed_pawn(square, board, color):
    file=chess.square_file(square)
    rank=chess.square_rank(square)
    if color==chess.WHITE:
        enemy_color=chess.BLACK
        forward_ranks=range(rank + 1, 8)
    else:
        enemy_color=chess.WHITE
        forward_ranks=range(rank - 1, -1, -1)
    #Check current file, left file, and right file
    files_to_check = [file]
    if file > 0:
        files_to_check.append(file - 1)
    if file < 7:
        files_to_check.append(file + 1)
    #check if any enemy pawn is blocking
    for f in files_to_check:
        for r in forward_ranks:
            sq=chess.square(f, r)
            piece=board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == enemy_color:
                return False  #Blocked → not a passed pawn
    return True  #No blockers →is a passed pawn
def protection_count(square, board, color):
    #Number of friendly pieces protecting this square
    return len(board.attackers(color, square))
def pawn_structure_score(board, square, color,reward_good=1.0, penalty_bad=-1.0 ):
    """Return pawn structural score for this pawn"""
    #Not isolated →score 0
    if not is_isolated_pawn(board, square, color):
        return 0

    #Isolated & passed →good pawn
    if is_passed_pawn(square, board, color):
        return reward_good

    #Isolated & not passed，evaluate the number of attackers & defenders
    defenders = protection_count(square, board, color)
    attackers = len(board.attackers(not color, square))
    #Enemy attackers significantly exceed defenders
    if attackers - defenders >= 2:
        return penalty_bad #bad pawn
    #Attacked but with zero defenders
    if attackers > 0 and defenders == 0:
        return penalty_bad #bad pawn

    #Otherwise → neutral isolated pawn
    return 0
def total_pawns_structure_score_for_color(board, color):
    """Sum pawn-structure scores for all pawns of a given color"""
    total_score = 0
    for square in board.pieces(chess.PAWN, color):
        total_score += pawn_structure_score(board, square, color)
    return total_score

#Doubled pawn detection
def is_overlapping_pawn(board,square,color):
    file=chess.square_file(square)
    pawns_in_file=[sq for sq in board.pieces(chess.PAWN,color) if chess.square_file(sq)==file]
    return len(pawns_in_file)>1
def total_overlapping_pawns_score_for_color(board,color,penalty=-1):
    total_score=0
    pawns=list(board.pieces(chess.PAWN,color))
    file_counts={}
    for sq in pawns:
        file=chess.square_file(sq)
        file_counts[file]=file_counts.get(file,0)+1
    for sq in pawns:
        file=chess.square_file(sq)
        if file_counts[file]>1:
            total_score+=penalty
    return total_score

#Protected pawn score
def protected_pawn_score(board, square, color, reward=1.0):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    direction=-1 if color==chess.WHITE else 1
    #Diagonal backward squares that could contain protecting pawns
    protection_squares=[]
    for df in [-1, 1]:
        f=file+df
        r=rank+direction
        if 0<=f<=7 and 0<=r<=7:
            protection_squares.append(chess.square(f, r))
    count=0
    for sq in protection_squares:
        piece=board.piece_at(sq)
        if piece and piece.piece_type==chess.PAWN and piece.color==color:
            count+=1
    return count*reward
def total_protected_pawn_score_for_color(board, color):
    total_score=0
    for square in board.pieces(chess.PAWN, color):
        total_score+=protected_pawn_score(board, square, color)
    return total_score


def extract_features_from_row(row, game_id):
    board = chess.Board()

    features = {
        'white': initialize_features(),
        'black': initialize_features(),
        'game_id': game_id,
        'player_id_white':getattr(row, 'White', None),
        'player_id_black':getattr(row, 'Black', None)
    }

    moves_str = getattr(row, 'Moves', '')
    if not moves_str or moves_str.strip() == "":
        return features['white'], features['black']

    moves_list=[m.strip() for m in str(moves_str).split(",") if m.strip()]

    for move_uci in moves_list:
        move=None
        try:
            move = chess.Move.from_uci(move_uci)
        except Exception:
            try:
                move = board.parse_san(move_uci)
            except Exception:
                continue

        if move not in board.legal_moves:
            continue

        board.push(move)

        if board.turn == chess.WHITE:
            features['black'] = update_features(features['black'], board, "black", move)
        else:
            features['white'] = update_features(features['white'], board, "white", move)

    features['white']['game_id'] = game_id
    features['white']['player_id'] =getattr(row, 'White', None)
    features['black']['game_id'] = game_id
    features['black']['player_id'] =getattr(row, 'Black', None)

    features['white']['PawnIsolateScoreMean']=compute_mean_score(features['white'], 'PawnIsolateScoresList')
    features['black']['PawnIsolateScoreMean']=compute_mean_score(features['black'], 'PawnIsolateScoresList')
    features['white']['PawnOverlapScoreMean']=compute_mean_score(features['white'], 'PawnOverlapScoresList')
    features['black']['PawnOverlapScoreMean']=compute_mean_score(features['black'], 'PawnOverlapScoresList')
    features['white']['PawnProtectScoreMean']=compute_mean_score(features['white'], 'PawnProtectScoresList')
    features['black']['PawnProtectScoreMean']=compute_mean_score(features['black'], 'PawnProtectScoresList')
    
    return features['white'], features['black']

try:
    #在 rank 0 读数据库，然后广播给所有 rank
    if rank == 0:
        print("Rank 0: Reading database...", flush=True)
        conn = sqlite3.connect(r"c:\sqlite3\chess.db")
        df = pd.read_sql_query("SELECT * FROM games", conn)
        conn.close()
        print(f"Rank 0: Loaded {len(df)} games", flush=True)
    else:
        df = None
    
    df = comm.bcast(df, root=0)
    print(f"Rank {rank}: Received dataframe with {len(df)} rows", flush=True)

    #只读取自己的临时文件，不读取其他 rank 的
    temp_csv = f"temp_features_rank_{rank}.csv"
    processed_uids_local = set()
    file_exists = os.path.exists(temp_csv)
    
    if file_exists:
        try:
            df_done = pd.read_csv(temp_csv, usecols=['uid'])
            processed_uids_local.update(df_done['uid'].astype(str).tolist())
            print(f"Rank {rank}: Found {len(processed_uids_local)} already processed UIDs", flush=True)
        except Exception as e:
            print(f"Rank {rank}: Error reading temp file: {e}", flush=True)
            processed_uids_local = set()

    #gather 收集所有 rank 已处理的 uid
    all_processed = comm.gather(processed_uids_local, root=0)
    
    if rank == 0:
        processed_uids_global = set()
        for proc_set in all_processed:
            processed_uids_global.update(proc_set)
        print(f"Rank 0: Total processed UIDs across all ranks: {len(processed_uids_global)}", flush=True)
    else:
        processed_uids_global = None
    
    #已处理的 uid
    processed_uids_global = comm.bcast(processed_uids_global, root=0)

    #基于全局已处理的 uid 来分配任务
    all_uids = df['uid'].astype(str).tolist()
    remaining_uids = [uid for uid in all_uids if uid not in processed_uids_global]
    
    print(f"Rank {rank}: {len(remaining_uids)} UIDs remaining to process", flush=True)

    chunk_size = len(remaining_uids) // size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank != size-1 else len(remaining_uids)
    my_uids = set(remaining_uids[start_idx:end_idx])

    print(f"Rank {rank}: Assigned {len(my_uids)} UIDs to process", flush=True)

    df_chunk = df[df['uid'].astype(str).isin(my_uids)].reset_index(drop=True)

    batch_size = 100
    batch = []
    error_count = 0 
    max_errors = 10  

    for i, row in enumerate(tqdm(df_chunk.itertuples(index=False), total=len(df_chunk),
                                 desc=f"Rank {rank} processing")):
        try:  
            uid = str(row.uid)
            if uid in processed_uids_local:
                continue

            white_features, black_features = extract_features_from_row(row, game_id=uid)
            white_features["uid"] = uid
            black_features["uid"] = uid

            batch.append(white_features)
            batch.append(black_features)
            processed_uids_local.add(uid)

            if len(batch) >= batch_size:
                batch_df = pd.DataFrame(batch)
                batch_df = ensure_column_order(batch_df, FEATURE_COLUMNS)
                batch_df.to_csv(temp_csv, mode='a', header=not file_exists, 
                               index=False, encoding='utf-8-sig')
                file_exists = True
                batch = []
                tqdm.write(f"Rank {rank}: saved progress at uid {uid}")
                
        except Exception as e: 
            error_count += 1
            print(f"Rank {rank}: Error processing uid {getattr(row, 'uid', 'unknown')}: {e}", flush=True)
            if error_count > max_errors:
                print(f"Rank {rank}: Too many errors ({error_count}), aborting", flush=True)
                raise
            continue

    if batch:
        batch_df = pd.DataFrame(batch)
        batch_df = ensure_column_order(batch_df, FEATURE_COLUMNS)
        batch_df.to_csv(temp_csv, mode='a', header=not file_exists, 
                       index=False, encoding='utf-8-sig')
        print(f"Rank {rank}: saved final batch ({len(batch)} records)", flush=True)

    print(f"Rank {rank}: Processing complete. Total errors: {error_count}", flush=True)

except Exception as e: 
    print(f"Rank {rank}: FATAL ERROR - {e}", flush=True)
    print(f"Rank {rank}: Traceback:\n{traceback.format_exc()}", flush=True)
    comm.Abort(1)  #出错时让所有进程退出

if not safe_barrier():
    print(f"Rank {rank}: Barrier failed, exiting", flush=True)
    sys.exit(1)

if rank == 0:
    try:  
        print("Rank 0: Merging results...", flush=True)
        all_dfs = []
        for r in range(size):
            file_r = f"temp_features_rank_{r}.csv"
            if os.path.exists(file_r):
                df_r = pd.read_csv(file_r)
                df_r = ensure_column_order(df_r, FEATURE_COLUMNS)
                all_dfs.append(df_r)
                print(f"Rank 0: Loaded {len(df_r)} rows from rank {r}, columns: {len(df_r.columns)}", flush=True)
        
        features_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Rank 0: Total rows: {len(features_df)}", flush=True)
        
        print(f"Rank 0: Final columns: {list(features_df.columns)}", flush=True)

        if len(features_df) % 2 != 0:
            print(f"WARNING: Odd number of rows ({len(features_df)}), will drop last row", flush=True)
            features_df = features_df.iloc[:-1]  #如果是奇数行就删掉最后一行

        white_df = features_df.iloc[0::2].reset_index(drop=True).add_prefix("White_")
        black_df = features_df.iloc[1::2].reset_index(drop=True).add_prefix("Black_")
        
        #行数匹配
        if len(white_df) != len(black_df):
            raise ValueError(f"White ({len(white_df)}) and Black ({len(black_df)}) row counts don't match!")
        
        merged_df = pd.concat([white_df, black_df], axis=1)
        merged_df.insert(0, 'uid', white_df['White_uid'])
        merged_df = merged_df.drop(columns=["White_uid", "Black_uid"])

        features_df.to_csv("2rows100w.csv", index=False, encoding='utf-8-sig')
        merged_df.to_csv("pawn100w.csv", index=False, encoding='utf-8-sig')
        print("All ranks finished. Final CSV saved.", flush=True)
    except Exception as e:  
        print(f"Rank 0: Error during merging: {e}", flush=True)
        print(f"Rank 0: Traceback:\n{traceback.format_exc()}", flush=True)

# mpiexec -n 4 python "C:\Users\Administrator\Desktop\playerstyles\pawn_structure\isolated_pawn.py"