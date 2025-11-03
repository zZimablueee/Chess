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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def safe_barrier():
    """安全的 barrier,带超时检测"""
    try:
        comm.Barrier()
        return True
    except Exception as e:
        print(f"Rank {rank}: Barrier failed with error: {e}", flush=True)
        return False

# 这个列表定义了所有特征列的顺序，确保每次运行都一致
FEATURE_COLUMNS = [
    'CenterControlScoresList',
    'PieceActivityScoresList',
    'KingSafetyScoresList',
    'CastlingScoresList',
    'KingTropismScoresList',
    'KingDefendersScoresList',
    'KingPawnShieldScoresList',
    'KingZoneControlScoresList',
    'KingDiagonalExposureScoresList',
    'KingEscapeSquaresScoresList',
    'AttackingMovesScoresList',
    'CapturesScoresList',
    'PawnStructureScoresList',
    'castled',
    'LostCastlingRights',
    'lost_castling_rights',
    'game_id',
    'player_id',
    'CenterControlScoreMean',
    'PieceActivityScoreMean',
    'KingSafetyScoreMean',
    'CastlingScoreMean',
    'KingTropismScoreMean',
    'KingDefendersScoreMean',
    'KingPawnShieldScoreMean',
    'KingZoneControlScoreMean',
    'KingDiagonalExposureScoreMean',
    'KingEscapeSquaresScoreMean',
    'uid'
]

def ensure_column_order(df, columns):
    """确保 DataFrame 的列顺序与指定顺序一致"""
    # 只保留存在的列
    existing_cols = [col for col in columns if col in df.columns]
    # 如果有新列不在预定义列表中，添加到末尾
    extra_cols = [col for col in df.columns if col not in columns]
    return df[existing_cols + extra_cols]

def to_side(color):
    """Normalize color input: accepts 'white'/'black' or chess.WHITE/chess.BLACK."""
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

    features['white']['CenterControlScoreMean']=compute_mean_score(features['white'], 'CenterControlScoresList')
    features['black']['CenterControlScoreMean']=compute_mean_score(features['black'], 'CenterControlScoresList')
    features['white']['PieceActivityScoreMean']=compute_mean_score(features['white'],'PieceActivityScoresList')
    features['black']['PieceActivityScoreMean']=compute_mean_score(features['black'],'PieceActivityScoresList')
    features['white']['KingSafetyScoreMean']=compute_mean_score(features['white'],'KingSafetyScoresList')
    features['black']['KingSafetyScoreMean']=compute_mean_score(features['black'],'KingSafetyScoresList')
    features['white']['CastlingScoreMean']=compute_mean_score(features['white'],'CastlingScoresList')
    features['black']['CastlingScoreMean']=compute_mean_score(features['black'],'CastlingScoresList')
    features['white']['KingTropismScoreMean']=compute_mean_score(features['white'],'KingTropismScoresList')
    features['black']['KingTropismScoreMean']=compute_mean_score(features['black'],'KingTropismScoresList')
    features['white']['KingDefendersScoreMean']=compute_mean_score(features['white'],'KingDefendersScoresList')
    features['black']['KingDefendersScoreMean']=compute_mean_score(features['black'],'KingDefendersScoresList')
    features['white']['KingPawnShieldScoreMean']=compute_mean_score(features['white'],'KingPawnShieldScoresList')
    features['black']['KingPawnShieldScoreMean']=compute_mean_score(features['black'],'KingPawnShieldScoresList')
    features['white']['KingZoneControlScoreMean']=compute_mean_score(features['white'],'KingZoneControlScoresList')
    features['black']['KingZoneControlScoreMean']=compute_mean_score(features['black'],'KingZoneControlScoresList')
    features['white']['KingDiagonalExposureScoreMean']=compute_mean_score(features['white'],'KingDiagonalExposureScoresList')
    features['black']['KingDiagonalExposureScoreMean']=compute_mean_score(features['black'],'KingDiagonalExposureScoresList')
    features['white']['KingEscapeSquaresScoreMean']=compute_mean_score(features['white'],'KingEscapeSquaresScoresList')
    features['black']['KingEscapeSquaresScoreMean']=compute_mean_score(features['black'],'KingEscapeSquaresScoresList')

    return features['white'], features['black']

def initialize_features():
    return {
        'CenterControlScoresList': [],
        'PieceActivityScoresList': [],
        'KingSafetyScoresList': [],
        'CastlingScoresList':[],
        'KingTropismScoresList':[],
        'KingDefendersScoresList':[],
        'KingPawnShieldScoresList':[],
        'KingZoneControlScoresList':[],
        'KingDiagonalExposureScoresList':[],
        'KingEscapeSquaresScoresList':[],
        'AttackingMovesScoresList': [],
        'CapturesScoresList': [],
        'PawnStructureScoresList': [],
        'castled': False,
        'LostCastlingRights': False
    }

def update_features(feature_dict, board,color,move):
    feature_dict['CenterControlScoresList'].append(calculate_center_control(board,color))
    feature_dict['PieceActivityScoresList'].append(calculate_piece_activity(board,color))
    feature_dict['KingSafetyScoresList'].append(calculate_king_safety(board,color,feature_dict,move))
    feature_dict['CastlingScoresList'].append(calculate_castling(board,color,feature_dict,move))
    feature_dict['KingTropismScoresList'].append(calculate_king_tropism(board, color))
    feature_dict['KingDefendersScoresList'].append(calculate_king_defenders(board,color))
    feature_dict['KingPawnShieldScoresList'].append(calculate_pawn_shield(board,color))
    feature_dict['KingZoneControlScoresList'].append(calculate_zone_control(board,color))
    feature_dict['KingDiagonalExposureScoresList'].append(calculate_diagonal_exposure(board,color))
    feature_dict['KingEscapeSquaresScoresList'].append(calculate_escape_squares(board,color))
    return feature_dict

def calculate_center_control(board,color):
    side=to_side(color)
    enemy=chess.BLACK if side==chess.WHITE else chess.WHITE
    center_control=0
    center_squares = [
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6
    ]

    for square in center_squares:
        piece=board.piece_at(square)
        rank = chess.square_rank(square)
        if color == 'white':
            weight=1 + 0.2 * (rank - 3)
        else:
            weight=1 + 0.2 * (4 - rank)
        
        control_score=0
        threat_score=0

        if piece and piece.color==color:
            control_score+=weight
        attackers=board.attackers(chess.WHITE if color=='white' else chess.BLACK, square)
        for attacker_sq in attackers:
            if piece and piece.color!=color:
                threat_score+=get_piece_value(piece.piece_type)
            else:
                control_score+=1
        center_control+=(control_score+threat_score)
    return center_control

def compute_mean_score(feature_dict, key):
    scores = feature_dict.get(key, [])
    return sum(scores) / len(scores) if scores else 0

def get_piece_value(piece_type:chess.PieceType) -> float:
    values={
        chess.PAWN:1.0,
        chess.KNIGHT:3.0,
        chess.BISHOP:3.0,
        chess.ROOK:5.0,
        chess.QUEEN:9.0,
        chess.KING:0.0
    }
    return values.get(piece_type,0)

def calculate_piece_activity(board, color):
    activity_score = 0
    side=to_side(color)
    for square, piece in board.piece_map().items():
        if piece.color == side:
            legal_moves=[m for m in board.legal_moves if m.from_square==square]
            activity_score += get_piece_value(piece.piece_type)*len(board.attacks(square))
    return activity_score

def calculate_king_safety(board,color,feature_dict,move):
    king_safety_score=0
    king_safety_score+=calculate_castling(board,color,feature_dict,move)
    king_safety_score+=calculate_king_tropism(board,color)
    king_safety_score+=calculate_king_defenders(board, color)
    king_safety_score+=calculate_pawn_shield(board, color)
    king_safety_score+=calculate_zone_control(board, color)
    king_safety_score+=calculate_diagonal_exposure(board,color)
    king_safety_score+=calculate_escape_squares(board,color)
    return king_safety_score

def calculate_castling(board, color, feature_dict, move):
    side=to_side(color)
    if feature_dict['castled']:
        return 30
    if board.is_castling(move):
        feature_dict['castled'] = True
        return 30
    if not feature_dict['castled'] and not board.has_castling_rights(side):
        feature_dict['lost_castling_rights'] = True
        return -20
    return 0

def calculate_king_tropism(board,color):
    king_color=to_side(color)
    king_square=board.king(king_color)
    king_tropism_score=0
    if king_square is None:
        return 0

    for square,piece in board.piece_map().items():
        if piece.color==king_color:
            continue
        dist=chess.square_distance(square,king_square)
        if dist==0:
            continue

        piece_value=get_piece_value(piece.piece_type)
        if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
            king_tropism_score-= 10*piece_value/(dist**2)
    
    king_tropism_score = max(-300, min(0, king_tropism_score))
    return king_tropism_score

def calculate_king_defenders(board,color):
    king_color=to_side(color)
    king_zone=get_king_zone(board,color,radius=1)
    king_defenders_score=0
    for sq in king_zone:
        attackers=board.attackers(king_color,sq)
        king_defenders_score+=len(attackers)
    king_defenders_score=min(king_defenders_score,4)
    return king_defenders_score

def get_king_zone(board,color,radius=1):
    king_color=to_side(color)
    king_square=board.king(king_color)
    if king_square is None:
        return []
    zone=[sq for sq in chess.SQUARES if chess.square_distance(sq,king_square)<=radius]
    return zone

def is_open_file(board,file_index,color):
    for row in range(8):
        sq=chess.square(file_index,row)
        piece=board.piece_at(sq)
        if piece and piece.color==color:
            return False
    return True
   
def calculate_pawn_shield(board,color):
    side=to_side(color)
    king_square=board.king(side)
    if king_square is None:
        return 0
    king_file=chess.square_file(king_square)
    king_rank=chess.square_rank(king_square)

    if side==chess.WHITE:
        if king_rank>1:
            return 0
        target_rank=king_rank+1
    if side==chess.BLACK:
        if king_rank<6:
            return 0
        target_rank=king_rank-1

    candidate_files=[king_file]
    if king_file-1>=0:
        candidate_files.append(king_file-1)
    if king_file+1<=7:
        candidate_files.append(king_file+1)
    
    pawn_shield_score=0
    for file in candidate_files:
        sq=chess.square(file,target_rank)
        piece=board.piece_at(sq)
        if piece and piece.piece_type==chess.PAWN and piece.color==side:
            pawn_shield_score+=5
        else:
            pawn_shield_score-=5
            if is_open_file(board,file,color):
                pawn_shield_score-=5
    
    pawn_shield_score=max(-30,min(15,pawn_shield_score))
    return pawn_shield_score

def calculate_zone_control(board, color,forward_extension=2):
    king_color=to_side(color)
    enemy_color=chess.BLACK if king_color == chess.WHITE else chess.WHITE
    king_square=board.king(king_color)
    if king_square is None:
        return 0
    
    king_zone=get_king_zone(board,color,radius=1)
    king_file=chess.square_file(king_square)
    king_rank=chess.square_rank(king_square)
    forward_ranks=[]
    if king_color==chess.WHITE:
        for r in range(king_rank+1,min(8, king_rank + 1 + forward_extension)):
            forward_ranks.append(r)
    else:
        for r in range(max(0, king_rank - forward_extension), king_rank):
            forward_ranks.append(r)
    for f in range(max(0,king_file-1),min(8,king_file+2)):
        for r in forward_ranks:
            sq=chess.square(f,r)
            if sq not in king_zone:
                king_zone.append(sq)

    zone_control_score=0
    for sq in king_zone:
        attackers=board.attackers(enemy_color,sq)
        for attacker in attackers:
            piece=board.piece_at(attacker)
            if piece:
                zone_control_score-=5*get_piece_value(piece.piece_type)

    zone_control_score = max(-100, min(0, zone_control_score))
    return zone_control_score

def calculate_diagonal_exposure(board,color):
    side = to_side(color)
    king_square=board.king(side)
    if king_square is None:
        return 0
    king_file=chess.square_file(king_square)
    king_rank=chess.square_rank(king_square)

    diagonal_exposure_score=0

    directions=[(1,0),(-1,0),
                (1,1),(-1,1),
                (1,-1),(-1,-1)
                ]
    for direction_file,direction_rank in directions:
        file,rank=king_file,king_rank
        our_flag=False

        for dist in range(1,8):
            file=king_file+direction_file*dist
            rank=king_rank+direction_rank*dist
            if not (0<=file<8 and 0<=rank<8):
                break
            sq=chess.square(file,rank)
            piece=board.piece_at(sq)
            if piece:
                if piece.color==side:
                    our_flag=True
                    break
                else:
                    if direction_rank == 0 and piece.piece_type in [chess.ROOK, chess.QUEEN]:
                        diagonal_exposure_score -= max(10, 40 // dist)
                    if abs(direction_file) == abs(direction_rank) and piece.piece_type in [chess.BISHOP, chess.QUEEN]:
                        diagonal_exposure_score -= max(10, 40 // dist)
                    break

        if not our_flag:
            diagonal_exposure_score-=3

    diagonal_exposure_score=max(-80,min(0,diagonal_exposure_score))
    return diagonal_exposure_score

def calculate_escape_squares(board,color):
    king_color=to_side(color)
    enemy_color=chess.BLACK if king_color == chess.WHITE else chess.WHITE
    king_square=board.king(king_color)
    if king_square is None:
        return 0

    escape_square_num=0
    for sq in get_king_zone(board,color,radius=1):
        piece=board.piece_at(sq)
        if piece and piece.color==king_color:
            continue
        if board.is_attacked_by(enemy_color,sq):
            continue
        escape_square_num+=1
    escape_square_score=5*escape_square_num
    return escape_square_score

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

    features['white']['CenterControlScoreMean']=compute_mean_score(features['white'], 'CenterControlScoresList')
    features['black']['CenterControlScoreMean']=compute_mean_score(features['black'], 'CenterControlScoresList')
    features['white']['PieceActivityScoreMean']=compute_mean_score(features['white'],'PieceActivityScoresList')
    features['black']['PieceActivityScoreMean']=compute_mean_score(features['black'],'PieceActivityScoresList')
    features['white']['KingSafetyScoreMean']=compute_mean_score(features['white'],'KingSafetyScoresList')
    features['black']['KingSafetyScoreMean']=compute_mean_score(features['black'],'KingSafetyScoresList')
    features['white']['CastlingScoreMean']=compute_mean_score(features['white'],'CastlingScoresList')
    features['black']['CastlingScoreMean']=compute_mean_score(features['black'],'CastlingScoresList')
    features['white']['KingTropismScoreMean']=compute_mean_score(features['white'],'KingTropismScoresList')
    features['black']['KingTropismScoreMean']=compute_mean_score(features['black'],'KingTropismScoresList')
    features['white']['KingDefendersScoreMean']=compute_mean_score(features['white'],'KingDefendersScoresList')
    features['black']['KingDefendersScoreMean']=compute_mean_score(features['black'],'KingDefendersScoresList')
    features['white']['KingPawnShieldScoreMean']=compute_mean_score(features['white'],'KingPawnShieldScoresList')
    features['black']['KingPawnShieldScoreMean']=compute_mean_score(features['black'],'KingPawnShieldScoresList')
    features['white']['KingZoneControlScoreMean']=compute_mean_score(features['white'],'KingZoneControlScoresList')
    features['black']['KingZoneControlScoreMean']=compute_mean_score(features['black'],'KingZoneControlScoresList')
    features['white']['KingDiagonalExposureScoreMean']=compute_mean_score(features['white'],'KingDiagonalExposureScoresList')
    features['black']['KingDiagonalExposureScoreMean']=compute_mean_score(features['black'],'KingDiagonalExposureScoresList')
    features['white']['KingEscapeSquaresScoreMean']=compute_mean_score(features['white'],'KingEscapeSquaresScoresList')
    features['black']['KingEscapeSquaresScoreMean']=compute_mean_score(features['black'],'KingEscapeSquaresScoresList')
    
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

        features_df.to_csv("2bei_mpi1103.csv", index=False, encoding='utf-8-sig')
        merged_df.to_csv("same_mpi1103.csv", index=False, encoding='utf-8-sig')
        print("All ranks finished. Final CSV saved.", flush=True)
    except Exception as e:  
        print(f"Rank 0: Error during merging: {e}", flush=True)
        print(f"Rank 0: Traceback:\n{traceback.format_exc()}", flush=True)

#mpiexec -n 4 python "C:\Users\Administrator\Desktop\playerstyles\kingsafety.py"