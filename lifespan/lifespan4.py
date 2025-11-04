import pandas as pd
import chess.pgn
import math
from collections import defaultdict
from mpi4py import MPI
from tqdm import tqdm
import sqlite3
import io
import glob
import os

db_file=r"C:\sqlite3\chess.db"
table_name='games'

def calculate_piece_lifespans(game, color):
    moves = list(game.mainline_moves())
    total_steps = len(moves)
    if total_steps == 0:
        return []

    replay_board=game.board()

    # 我们使用基于 board-square 的追踪，不用 id(piece)
    # square_to_pid: 当前棋盘上某个格子的"唯一棋子标识名" (e.g. "pawn_2")
    # pid_birth: pid -> birth_step
    # pid_type, pid_name 用于记录类型与输出名
    square_to_pid={}
    pid_birth={}
    pid_type={}
    pid_name_map={}
    lifespans=[]
    promo_count=defaultdict(int)
    piece_counter=defaultdict(int)

    # 注册初始棋子（出生 time = 0），用起始格子生成唯一 id
    for sq,piece in replay_board.piece_map().items():
        if piece.color==to_side(color):
            piece_counter[piece.piece_type] += 1
            name=f"{chess.piece_name(piece.piece_type)}_{piece_counter[piece.piece_type]}"
            pid=name  # 用字符串作为 pid，便于后续写入列名
            square_to_pid[sq]=pid
            pid_birth[pid]=0
            pid_type[pid]=piece.piece_type
            pid_name_map[pid]=name

    # 逐步回放
    for step, move in enumerate(moves, start=1):
        # 先计算被捕获的格子（考虑 en passant）
        if replay_board.is_en_passant(move):
            captured_square=move.to_square - 8 if replay_board.turn==chess.WHITE else move.to_square+8
        else:
            captured_square=move.to_square

        # 如果在 captured_square 上有我方的 pid（表示我方被吃）
        if captured_square in square_to_pid:
            pid=square_to_pid[captured_square]
            # 只关心本 color 的棋子：由于 square_to_pid 只注册了 color 的棋子，这里便为本方被吃事件
            birth=pid_birth.get(pid, 0)
            death=step
            lifespan=death-birth             # 用 death - birth，使 ratio 最大为 1
            name=pid_name_map.get(pid, pid)
            lifespans.append((name, (birth, death), lifespan/total_steps))
            # 从当前棋盘追踪表中移除被吃掉的 pid
            # 注意：被吃掉的格子在 push 之后也会变为空，这里先移除映射
            del pid_birth[pid]
            del pid_type[pid]
            del pid_name_map[pid]
            del square_to_pid[captured_square]

        # 现在执行推子，棋盘状态更新
        from_sq=move.from_square
        to_sq=move.to_square
        # 在 move 之前，某个 pid 可能位于 from_sq（如果是我方棋子）
        pid_moving=None
        if from_sq in square_to_pid:
            pid_moving=square_to_pid[from_sq]
            # 移动 pid 映射：从 from_sq -> to_sq
            del square_to_pid[from_sq]
            square_to_pid[to_sq]=pid_moving
        else:
            # 如果没有 pid 在 from_sq，表示这个 move 不是我们追踪颜色那方的棋子（对方走棋）
            # 但仍可能是对方的升变/吃子，不影响我们对本 color 的映射
            pass

        replay_board.push(move)

        # 如果此次 move 是升变并且生成的是本 color 的新棋子
        if move.promotion:
            promoted_piece=replay_board.piece_at(move.to_square)
            if promoted_piece and promoted_piece.color==to_side(color):
                ptype_name = chess.piece_name(promoted_piece.piece_type)
                promo_count[ptype_name]+=1
                promo_name=f"{ptype_name}_promo_{promo_count[ptype_name]}"
                pid=promo_name
                #新 pid，出生时间是当前 step（promotion 生效在 push 之后）
                pid_birth[pid]=step
                pid_type[pid]=promoted_piece.piece_type
                pid_name_map[pid]=promo_name
                # 该新 pid 位于 to_sq（覆盖前面的映射）
                square_to_pid[to_sq]=pid

    # 最后：所有仍在 pid_birth 的 pid 都是"活到终局"的
    for pid, birth in pid_birth.items():
        death=total_steps
        lifespan=death - birth
        name=pid_name_map.get(pid, pid)
        lifespans.append((name, (birth, death),lifespan/total_steps))

    return lifespans

BASIC_PIECE_COLUMNS = [
    "king_1", "queen_1",
    "rook_1", "rook_2",
    "bishop_1", "bishop_2",
    "knight_1", "knight_2",
    "pawn_1", "pawn_2", "pawn_3", "pawn_4",
    "pawn_5", "pawn_6", "pawn_7", "pawn_8"
]
PROMOTION_PIECE_COLUMNS = [
    "queen_promo_1", "queen_promo_2", "queen_promo_3", "queen_promo_4",
    "queen_promo_5", "queen_promo_6", "queen_promo_7", "queen_promo_8",
    "rook_promo_1", "rook_promo_2", "rook_promo_3", "rook_promo_4",
    "bishop_promo_1", "bishop_promo_2", "bishop_promo_3", "bishop_promo_4",
    "knight_promo_1", "knight_promo_2", "knight_promo_3", "knight_promo_4"
]
ALL_PIECE_COLUMNS = BASIC_PIECE_COLUMNS + PROMOTION_PIECE_COLUMNS

def to_side(color):
    """Normalize color input: accepts 'white'/'black' or chess.WHITE/chess.BLACK."""
    if color=='white' or color==chess.WHITE:
        return chess.WHITE
    if color=='black' or color==chess.BLACK:
        return chess.BLACK
    raise ValueError("color must be 'white' or 'black' or chess.WHITE/chess.BLACK")

def initialize_features():
    """初始化每局棋子特征为空"""
    features = {}
    for p in ALL_PIECE_COLUMNS:
        features[f"{p}Span"] = None
        features[f"{p}LifeRatio"] = math.nan
    return features

def flatten_lifespans_into_features(lifespans_list, features_dict):
    """把计算得到的寿命信息写入 features_dict"""
    for name, (birth, death), ratio in lifespans_list:
        features_dict[f"{name}Span"] = (birth, death)
        features_dict[f"{name}LifeRatio"] = ratio

def extract_features(game, game_id, player_white, player_black):
    """提取单局游戏的特征"""
    white_features = initialize_features()
    black_features = initialize_features()

    white_lifespans = calculate_piece_lifespans(game, "white")
    black_lifespans = calculate_piece_lifespans(game, "black")

    flatten_lifespans_into_features(white_lifespans, white_features)
    flatten_lifespans_into_features(black_lifespans, black_features)

    # 添加游戏信息
    white_features["game_id"] = game_id
    white_features["player"] = player_white
    black_features["game_id"] = game_id
    black_features["player"] = player_black

    return white_features, black_features

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 读取数据库
if rank == 0:
    conn = sqlite3.connect(db_file)
    games_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    # 将任务按 rank 数量切分
    game_chunks = [games_df.iloc[i::size] for i in range(size)]
else:
    game_chunks = None

chunk = comm.scatter(game_chunks, root=0)
results = []

for idx, row in enumerate(tqdm(chunk.itertuples(), desc=f"Rank {rank}")):
    pgn_text = row.Moves
    if not isinstance(pgn_text, str) or len(pgn_text.strip()) == 0:
        continue  # 跳过空棋谱
    
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        # 有的数据库只有走子文本，不是完整 PGN，跳过
        continue
    
    try:
        white_feat, black_feat = extract_features(game, row.uid, row.White, row.Black)
        results.append(white_feat)
        results.append(black_feat)
    except Exception as e:
        print(f"Rank {rank} 出错于行 {idx}: {e}")

# 每个 rank 直接写自己的 CSV
if len(results) > 0:
    df = pd.DataFrame(results)
    df.to_csv(f"lifespans_features_rank{rank}.csv", index=False)
    print(f"Rank {rank} saved {len(results)} records.")
else:
    print(f"Rank {rank} has no results to save.")

# 等待所有 rank 完成写入
comm.Barrier()

# Rank 0 负责合并所有 CSV 文件
if rank == 0:
    print("Combining all CSV files...")
    all_csvs = glob.glob("lifespans_features_rank*.csv")
    
    if len(all_csvs) > 0:
        dfs = [pd.read_csv(f) for f in all_csvs]
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv("lifespans_features_combined.csv", index=False)
        print(f"Combined {len(combined_df)} total records into lifespans_features_combined.csv")
        
        # 清理单个 rank 的文件（可选，如果不想保留可以取消注释）
        # for f in all_csvs:
        #     os.remove(f)
        # print("Cleaned up individual rank files.")
    else:
        print("No CSV files found to combine.")

print(f"Rank {rank} finished.")

# mpiexec -n 8 python "C:\Users\Administrator\Desktop\playerstyles\lifespan4.py"
