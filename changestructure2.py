import chess
import chess.pgn
import pandas as pd
import random
import sqlite3
from tqdm import tqdm
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

    #增加的：calculate_center_control normalized:
    features['white']['CenterControlScoreMean']=compute_mean_score(features['white'], 'CenterControlScoresList')
    features['black']['CenterControlScoreMean']=compute_mean_score(features['black'], 'CenterControlScoresList')
    #calculate_piece_activity normalized:
    features['white']['PieceActivityScoreMean']=compute_mean_score(features['white'],'PieceActivityScoresList')
    features['black']['PieceActivityScoreMean']=compute_mean_score(features['black'],'PieceActivityScoresList')
    #calculate king_safety normalized:
    features['white']['KingSafetyScoreMean']=compute_mean_score(features['white'],'KingSafetyScoresList')
    features['black']['KingSafetyScoreMean']=compute_mean_score(features['black'],'KingSafetyScoresList')
    #子方面：
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

# Additional function implementations...



def initialize_features():
    # Initialize all feature scores to zero
    return {
        'CenterControlScoresList': [],
        'PieceActivityScoresList': [],
        'KingSafetyScoresList': [],
        #子方面：
        'CastlingScoresList':[],
        'KingTropismScoresList':[],
        'KingDefendersScoresList':[],
        'KingPawnShieldScoresList':[],
        'KingZoneControlScoresList':[],
        'KingDiagonalExposureScoresList':[],
        'KingEscapeSquaresScoresList':[],



        #havent write functions yet
        'AttackingMovesScoresList': [],
        'CapturesScoresList': [],
        'PawnStructureScoresList': [],

        #new
        'castled': False,
        'LostCastlingRights': False
    }

def update_features(feature_dict, board,color,move):
    # Update each feature score based on the current board state
    feature_dict['CenterControlScoresList'].append(calculate_center_control(board,color))
    feature_dict['PieceActivityScoresList'].append(calculate_piece_activity(board,color))
    feature_dict['KingSafetyScoresList'].append(calculate_king_safety(board,color,feature_dict,move))
    #子方面：
    feature_dict['CastlingScoresList'].append(calculate_castling(board,color,feature_dict,move))
    feature_dict['KingTropismScoresList'].append(calculate_king_tropism(board, color))
    feature_dict['KingDefendersScoresList'].append(calculate_king_defenders(board,color))
    feature_dict['KingPawnShieldScoresList'].append(calculate_pawn_shield(board,color))
    feature_dict['KingZoneControlScoresList'].append(calculate_zone_control(board,color))
    feature_dict['KingDiagonalExposureScoresList'].append(calculate_diagonal_exposure(board,color))
    feature_dict['KingEscapeSquaresScoresList'].append(calculate_escape_squares(board,color))




    #need to be added after
    #feature_dict['attacking_moves_score'] += calculate_attacking_moves(board,color)
    #feature_dict['captures_score'] += calculate_captures(board,color)
    #feature_dict['pawn_structure_score'] += calculate_pawn_structure(board,color)
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
            weight=1 + 0.2 * (rank - 3)  # rank 4 是中线
        else:
            weight=1 + 0.2 * (4 - rank)  # 黑方从 rank 4 向下推进
        
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

#增加的：
def compute_mean_score(feature_dict, key):
    scores = feature_dict.get(key, [])
    return sum(scores) / len(scores) if scores else 0
#增加的：
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

#github fixed:
def calculate_piece_activity(board, color):
    activity_score = 0
    side=to_side(color)
    # Using items() to retrieve both the square and the piece
    for square, piece in board.piece_map().items():
        if piece.color == side:
            # Calculate the number of attacked squares from this square,but also need to make sure it's legal
            legal_moves=[m for m in board.legal_moves if m.from_square==square]
            activity_score += get_piece_value(piece.piece_type)*len(board.attacks(square))
    return activity_score


#fixed:   7   dimensions of KingSafety
#calculate_castling(board, color)
#calculate_king_tropism(board, color)
#calculate_defenders(board, color)
#calculate_pawn_shield(board, color)
#calculate_pawn_storm(board, color)
#calculate_zone_control(board, color)
#calculate_diagonal_exposure(board,color)
#calculate_escape_squares(board,color) 
def calculate_king_safety(board,color,feature_dict,move):
    king_safety_score=0

    king_safety_score+=calculate_castling(board,color,feature_dict,move)
    king_safety_score+=calculate_king_tropism(board,color)
    king_safety_score+=calculate_king_defenders(board, color)
    king_safety_score+=calculate_pawn_shield(board, color)
    #delete
    #king_safety_score+=calculate_pawn_storm(board, color)
    king_safety_score+=calculate_zone_control(board, color)
    king_safety_score+=calculate_diagonal_exposure(board,color)  #斜着的线
    king_safety_score+=calculate_escape_squares(board,color)   #国王是否有足够的逃生格子

    return king_safety_score


#castling part
def calculate_castling(board, color, feature_dict, move):
    side=to_side(color)

    #already castled, plus score since then
    if feature_dict['castled']:
        return 30

    #current move is castling, pluse score
    if board.is_castling(move):
        feature_dict['castled'] = True
        return 30

    #havent catstled and cant castle
    if not feature_dict['castled'] and not board.has_castling_rights(side):
        feature_dict['lost_castling_rights'] = True
        return -20

    #havent castled but can castle
    return 0

#king tropsim part
def calculate_king_tropism(board,color):
    king_color=to_side(color)
    king_square=board.king(king_color)
    king_tropism_score=0
    if king_square is None:
        return 0

    #iterate all the pieces on the board
    for square,piece in board.piece_map().items():
        #skip the same color pieces
        if piece.color==king_color:
            continue
        #for enemy's pieces,calculate the distance to our king
        dist=chess.square_distance(square,king_square)
        if dist==0:  #king itself, skip
            continue

        piece_value=get_piece_value(piece.piece_type)
        #enemy's QUEEN ROOK BISHOP KNIGHT PAWN
        if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
            #use "(dist**2)" to better describe: more close, more dangerous
            king_tropism_score-= 10*piece_value/(dist**2)
    
    #limit range:
    king_tropism_score = max(-300, min(0, king_tropism_score))
    return king_tropism_score

#king's defender pieces part:   1--4 score
def calculate_king_defenders(board,color):
    king_color=to_side(color)
    king_zone=get_king_zone(board,color,radius=1)
    king_defenders_score=0
    for sq in king_zone:
        attackers=board.attackers(king_color,sq)
        king_defenders_score+=len(attackers)  #no matter what kind of defender is, only plus 1 score
    king_defenders_score=min(king_defenders_score,4)
    return king_defenders_score

def get_king_zone(board,color,radius=1):
    king_color=to_side(color)
    king_square=board.king(king_color)
    if king_square is None:
        return []
    zone=[sq for sq in chess.SQUARES if chess.square_distance(sq,king_square)<=radius]
    return zone

#pawn sheild part
def is_open_file(board,file_index,color):
    for row in range(8):
        sq=chess.square(file_index,row)
        piece=board.piece_at(sq)
        if piece and piece.color==color: #have self-pieces in this line then is not open-file
            return False
    return True
   
def calculate_pawn_shield(board,color):
    #find where is the king
    side=to_side(color)
    king_square=board.king(side)
    if king_square is None:
        return 0   #actually not possible
    #the king's column & row
    king_file=chess.square_file(king_square)
    king_rank=chess.square_rank(king_square)

    #only calculate when king are in initial 2 rows
    #chess.square_rank() 返回 0..7 indicates to row 1..8 on board：
    if side==chess.WHITE:
        if king_rank>1:
            return 0
        target_rank=king_rank+1
    if side==chess.BLACK:
        if king_rank<6:
            return 0
        target_rank=king_rank-1

    #check the pawns which their column are next to king's:
    candidate_files=[king_file]   #0--7，indicates to a--h
    if king_file-1>=0:#king not in a
        candidate_files.append(king_file-1) #king's left column
    if king_file+1<=7:   #king not in h
        candidate_files.append(king_file+1)  #king's right column
    
    pawn_shield_score=0
    for file in candidate_files:
        sq=chess.square(file,target_rank)
        piece=board.piece_at(sq)
        if piece and piece.piece_type==chess.PAWN and piece.color==side:
            pawn_shield_score+=5
        else:
            pawn_shield_score-=5
            if is_open_file(board,file,color):
                pawn_shield_score-=5   #open file punishment
    
    pawn_shield_score=max(-30,min(15,pawn_shield_score))    #this score's range:-30 to +15
    return pawn_shield_score

#king zone control part
#forward_extension can choose 
def calculate_zone_control(board, color,forward_extension=2):
    king_color=to_side(color)
    enemy_color=chess.BLACK if king_color == chess.WHITE else chess.WHITE
    king_square=board.king(king_color)
    if king_square is None:
        return 0
    
    king_zone=get_king_zone(board,color,radius=1)
    #additional squares
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

    #iterate every square in king_zone:
    zone_control_score=0
    for sq in king_zone:
        attackers=board.attackers(enemy_color,sq)
        for attacker in attackers:
            piece=board.piece_at(attacker)
            if piece:
                zone_control_score-=5*get_piece_value(piece.piece_type)

    #limit range:
    zone_control_score = max(-100, min(0, zone_control_score))
    return zone_control_score

#diagonal exposure part: the extension of open-file punishment
def calculate_diagonal_exposure(board,color):
    side = to_side(color)
    king_square=board.king(side)
    if king_square is None:
        return 0
    king_file=chess.square_file(king_square)
    king_rank=chess.square_rank(king_square)

    diagonal_exposure_score=0

    #directions
    directions=[(1,0),(-1,0),  #king's row
                (1,1),(-1,1), #king's to enemy diagonals
                (1,-1),(-1,-1) #king's to back diagonals
                ]
    for direction_file,direction_rank in directions:
        file,rank=king_file,king_rank
        our_flag=False #if have pieces from us in that direction

        for dist in range(1,8):
            file=king_file+direction_file*dist
            rank=king_rank+direction_rank*dist
            #break when out of board
            if not (0<=file<8 and 0<=rank<8):
                break
            sq=chess.square(file,rank)
            piece=board.piece_at(sq)
            if piece:
                if piece.color==side:
                    our_flag=True  #detect out our piece
                    break
                else:
                    #have enemy in king's row, see if it can attack king(only ROOK QUEEN)
                    if direction_rank == 0 and piece.piece_type in [chess.ROOK, chess.QUEEN]:
                        diagonal_exposure_score -= max(10, 40 // dist)
                    #have enemy in king's diagonal, see if it can attack king(only BISHOP QUEEN)
                    if abs(direction_file) == abs(direction_rank) and piece.piece_type in [chess.BISHOP, chess.QUEEN]:
                        diagonal_exposure_score -= max(10, 40 // dist)
                    break

        if not our_flag:
            diagonal_exposure_score-=3

    #limit range
    diagonal_exposure_score=max(-80,min(0,diagonal_exposure_score))
    return diagonal_exposure_score

#king's escape square part:
#when king will be check
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
            continue  #cant escape to here because have our own piece
        #when sq is taken by enemy or is free
        #check if that sq is safe:
        if board.is_attacked_by(enemy_color,sq):
            continue
        escape_square_num+=1
    escape_square_score=5*escape_square_num
    return escape_square_score

def extract_features_from_row(row, game_id):
    #game_id: 唯一编号
    board = chess.Board()

    features = {
        'white': initialize_features(),
        'black': initialize_features(),
        'game_id': game_id,
        'player_id_white':getattr(row, 'White', None),
        'player_id_black':getattr(row, 'Black', None)
    }

    #把Moves 列变成pgn可以读的格式
    moves_str = getattr(row, 'Moves', '')
    if not moves_str or moves_str.strip() == "":
        return features['white'], features['black']  # 空对局

    moves_list=[m.strip() for m in str(moves_str).split(",") if m.strip()]

    for move_uci in moves_list:
        move=None
        try:
            move = chess.Move.from_uci(move_uci)
        except Exception:
            # try SAN fallback (some datasets use SAN)
            try:
                move = board.parse_san(move_uci)
            except Exception:
                # invalid move string: skip
                continue

        if move not in board.legal_moves:
            # 非法走子（有些数据集可能不完整），跳过
            continue

        board.push(move)

        # 注意：board.turn 表示 **下一步走棋方**
        if board.turn == chess.WHITE:  # 刚刚是黑方走的
            features['black'] = update_features(features['black'], board, "black", move)
        else:  # 刚刚是白方走的
            features['white'] = update_features(features['white'], board, "white", move)

    #添加基本信息
    features['white']['game_id'] = game_id
    features['white']['player_id'] =getattr(row, 'White', None)
    features['black']['game_id'] = game_id
    features['black']['player_id'] =getattr(row, 'Black', None)

    features['white']['CenterControlScoreMean']=compute_mean_score(features['white'], 'CenterControlScoresList')
    features['black']['CenterControlScoreMean']=compute_mean_score(features['black'], 'CenterControlScoresList')
    #calculate_piece_activity normalized:
    features['white']['PieceActivityScoreMean']=compute_mean_score(features['white'],'PieceActivityScoresList')
    features['black']['PieceActivityScoreMean']=compute_mean_score(features['black'],'PieceActivityScoresList')
    #calculate king_safety normalized:
    features['white']['KingSafetyScoreMean']=compute_mean_score(features['white'],'KingSafetyScoresList')
    features['black']['KingSafetyScoreMean']=compute_mean_score(features['black'],'KingSafetyScoresList')
    #子方面：
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

conn = sqlite3.connect(r"c:\sqlite3\chess.db")
df = pd.read_sql_query("SELECT * FROM games", conn)
conn.close()

# ======= MPI 分配剩余 uid =======
processed_uids = set()
for r in range(size):
    temp_csv_r = f"temp_features_rank_{r}.csv"
    if os.path.exists(temp_csv_r):
        try:
            df_done = pd.read_csv(temp_csv_r, usecols=['uid'])
            processed_uids.update(df_done['uid'].astype(str).tolist())
        except:
            pass

all_uids = df['uid'].astype(str).tolist()
remaining_uids = [uid for uid in all_uids if uid not in processed_uids]

chunk_size = len(remaining_uids) // size
start_idx = rank * chunk_size
end_idx = (rank + 1) * chunk_size if rank != size-1 else len(remaining_uids)
my_uids = set(remaining_uids[start_idx:end_idx])

df_chunk = df[df['uid'].astype(str).isin(my_uids)].reset_index(drop=True)

# ======= 分批写 CSV =======
batch_size = 100
temp_csv = f"temp_features_rank_{rank}.csv"
file_exists = os.path.exists(temp_csv)

batch = []
processed_uids_local = set()
if file_exists:
    try:
        df_done = pd.read_csv(temp_csv, usecols=['uid'])
        processed_uids_local.update(df_done['uid'].astype(str).tolist())
    except:
        pass

for i, row in enumerate(tqdm(df_chunk.itertuples(index=False), total=len(df_chunk),
                             desc=f"Rank {rank} processing")):
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
        pd.DataFrame(batch).to_csv(temp_csv, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
        file_exists = True
        batch = []
        tqdm.write(f"Rank {rank}: saved progress at uid {uid}")

if batch:
    pd.DataFrame(batch).to_csv(temp_csv, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
    tqdm.write(f"Rank {rank}: saved final batch")

comm.Barrier()

if rank == 0:
    all_dfs = []
    for r in range(size):
        file_r = f"temp_features_rank_{r}.csv"
        if os.path.exists(file_r):
            df_r = pd.read_csv(file_r)
            all_dfs.append(df_r)
    features_df = pd.concat(all_dfs, ignore_index=True)

    if len(features_df) % 2 != 0:
        raise ValueError("行数不是偶数，说明有不完整的对局")

    white_df = features_df.iloc[0::2].reset_index(drop=True).add_prefix("White")
    black_df = features_df.iloc[1::2].reset_index(drop=True).add_prefix("Black")
    merged_df = pd.concat([white_df, black_df], axis=1)
    merged_df.insert(0,'uid',white_df['White_uid'])
    merged_df = merged_df.drop(columns=["White_uid", "Black_uid"])

    features_df.to_csv("2bei_mpi1103.csv", index=False, encoding='utf-8-sig')
    merged_df.to_csv("same_mpi1103.csv", index=False, encoding='utf-8-sig')
    print("All ranks finished. Final CSV saved.")
#mpiexec -n 4 python "C:\Users\Administrator\Desktop\playerstyles\changestructure2.py"