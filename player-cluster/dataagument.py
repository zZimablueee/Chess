#前面的ipynb文件和狗屎一样恶心,改了一下
#cd C:\Users\Administrator\Desktop\Chess\player-cluster
import pandas as pd
import sqlite3
import numpy as np
import os

# 1. 配置区域 (每次只需要改这里)
# 基础文件路径
BASE_CSV_PATH = 'player_features.csv'
DB_PATH = r"C:\sqlite3\chess.db"
#主题词
TOPIC_NAME = "Solid"
#正样本名单
POSITIVE_NAMES = ['Varuzhan Akobian', 'Dariusz Swiercz', 'Jakub Pulpan', 'Vaclav Finek', 'Artiom Stribuk', 'Roven Vogel', 'Atanas Dimitrov', 'Aleksandr Lenderman', 'Alexey Dreev', 'Gerson Príncipe', 'Mikhail Demidov', 'Anish Giri', 'Arnaldo Jesus Fernandez De La Vara Mulet', 'Tamas Banusz', 'Kirill Klukin', 'Denis Lazavik', 'Benjamin Bok', 'Christos Krallis', 'Igor Lysyj', 'Hrant Melkumyan', 'Rasmus Svane', 'Alfonso Llorente Zaro', 'Vignesh N.R', 'Vincent Keymer', 'Vugar Rasulov', 'Zviad Izoria', 'Boris Grachev', 'Mahammad Muradli', 'VladimirKramnik', 'Jan Gustafsson', 'Dmitry MIschuk', 'Artem Timofeev', 'Alexander Donchenko', 'Aradhya Garg', 'Seo Jungmin', 'Ramil Faizrakhmanov', 'Suleyman Suleymanlı', 'Alexander Zubov', 'Markus Ragger', 'Sam Shankland', 'Haowen Xue', 'Alexander Grischuk', 'Shant Sargsyan', 'Vladyslav Sydoryka', 'Bahadir Ozen', 'Jaime Santos Latasa', 'Kirk Ghazarian', 'Arash Tahbaz', 'Bogdan Daniel Deac', 'Grigor Grigorov', 'Grigoriy Oparin', 'Matthias Bluebaum', 'Gabriel Gähwiler', 'Nikoloz Petriashvili', 'Krikor Sevag Mekhitarian']
#负样本名单
NEGATIVE_NAMES = ['Sebastian Mihajlov', 'Aaron Jacobson', 'Deniel Safarov', 'Egor Koshulyan', 'Almas Rakhmatullaev', 'Dachey Lin', 'Sahaj Grover', 'Zachary Tanenbaum', 'Johan-Sebastian Christiansen', 'Nitin Senthilvel', 'Klementy Sychev', 'David Gavrilescu', 'Silvius Tiberius', 'Satria Duta Cahaya', 'Denes Boros', 'Daniel Chan', 'Leo Valle Luis', 'Levy Rozman', 'Krzysztof Jakubowski', 'Lucas Do Valle Cardoso', 'Dominik Horvath', 'Aydin Suleymanli', 'Shahruh Turayev', 'Ilamparthi A R', 'Alexei Shirov', 'Francisco Javier Muñoz', 'Gevorg Harutjunyan', 'Vadym Petrovskyi', 'Maxim Lugovskoy', 'Owen McCoy', 'Kirill Alekseenko', 'Alexander Velikanov', 'Maksim Ivannikov', 'Adar TARHAN', 'Pranesh M', 'Matvey Galchenko', 'Seyed Abolfazl Moosavifar', 'Floryan Eugene', 'Mikhail Antipov', 'Nikolozi Kacharava', 'Maciej Klekowski', 'Abdulla Gadimbayli', 'Emin Ohanyan', 'Bobur Sattarov', 'Andrew Tang', 'Sathvik Adiga', 'Apoorv Kamble', 'Шубин Кирилл', 'Luka Paichadze', 'Alex Fier', 'Sravan Renjith', 'Miłosz Szpar', 'Andy Woodward', 'Farid Orujov', 'Egor Baskakov']

#最终需要保留的标准列
FINAL_COLUMNS = [
    'CenterControlScore', 'PieceActivityScore', 'KingSafetyScore', 'CastlingScore', 
    'KingTropismScore', 'KingDefendersScore', 'KingPawnShieldScore', 'KingZoneControlScore', 
    'KingDiagonalExposureScore', 'KingEscapeSquaresScore', 'CaptureRatio', 'PawnCenter', 
    'AveragePawnAdvanceDepth', 'CheckRatio', 'ForceRatio', 'PawnIsolateScore', 
    'PawnOverlapScore', 'PawnProtectScore', 'queen_1LifeRatio', 'rook_1LifeRatio', 
    'rook_2LifeRatio', 'bishop_1LifeRatio', 'bishop_2LifeRatio', 'knight_1LifeRatio', 
    'knight_2LifeRatio', 'pawn_1LifeRatio', 'pawn_2LifeRatio', 'pawn_3LifeRatio', 
    'pawn_4LifeRatio', 'pawn_5LifeRatio', 'pawn_6LifeRatio', 'pawn_7LifeRatio', 
    'pawn_8LifeRatio', 'queen_promo_1LifeRatio', 'queen_promo_2LifeRatio', 
    'queen_promo_3LifeRatio', 'queen_promo_4LifeRatio', 'queen_promo_5LifeRatio', 
    'queen_promo_6LifeRatio', 'queen_promo_7LifeRatio', 'queen_promo_8LifeRatio', 
    'Name', 'ELO'
]

# 2.核心函数
def get_db_connection(db_file):
    return sqlite3.connect(db_file)

def clean_names(df, col_name='Name'):
    """强力去除名字中的空格"""
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str).str.strip()
    return df

def fetch_white_black_data(conn, table_name, player_list, prefix_map=None, special_rename=None):
    """
    通用函数：分别抓取白方和黑方数据，改名并合并。
    适用于 kingsafety 和 CaptureNPawn 这种结构的表。
    """
    placeholders = ",".join(["?"] * len(player_list))
    
    # 获取所有列
    cols_info = pd.read_sql(f"PRAGMA table_info({table_name});", conn)['name'].tolist()
    
    # 定义处理逻辑：查询 -> 改名 -> 标记颜色
    results = []
    
    for color in ['White', 'Black']:
        # 1. 筛选该颜色的列
        # 你的逻辑中 kingsafety 是 White_player_id, CaptureNPawn 是 White
        if table_name == 'kingsafety':
            id_col = f"{color}_player_id"
            target_cols = ['uid'] + [c for c in cols_info if c.startswith(f"{color}_")]
        elif table_name == 'CaptureNPawn':
            id_col = color
            target_cols = ['uid', color] # 基础列
            # 添加带前缀的列
            target_cols += [c for c in cols_info if c.startswith(f"{color}_") or c == f"{color}PawnAdvanceDepth"]
            # 去重
            target_cols = list(set(target_cols))
        
        # 2. SQL 查询
        query = f"SELECT {', '.join(target_cols)} FROM {table_name} WHERE {id_col} IN ({placeholders})"
        df_part = pd.read_sql(query, conn, params=player_list)
        df_part['color'] = color
        
        # 3. 列名清洗 (去除前缀)
        rename_dict = {}
        for col in df_part.columns:
            if col.startswith(f"{color}_"):
                rename_dict[col] = col.replace(f"{color}_", "")
            elif col.startswith(color) and col != 'color': # 处理 WhitePawnAdvanceDepth 这种情况
                rename_dict[col] = col.replace(color, "")
                
        df_part = df_part.rename(columns=rename_dict)
        
        # 4. 特殊改名 (针对 CaptureNPawn)
        if special_rename and color in special_rename:
            df_part = df_part.rename(columns=special_rename[color])

        results.append(df_part)
        
    # 合并黑白
    df_all = pd.concat(results, axis=0, ignore_index=True)
    return df_all

def data_augmentation(df, target_players, n_samples=10, sample_size=10):
    """数据增强核心逻辑：随机采样并求平均"""
    augmented_rows = []
    # 强制把所有数值列转为数字，非数值列变为 NaN
    cols_to_exclude = ['uid', 'color', 'player', 'Name']
    cols_to_convert = [c for c in df.columns if c not in cols_to_exclude]
    
    # 这里的copy是为了不影响原始df，虽然在这个流程里无所谓
    df_process = df.copy()
    for col in cols_to_convert:
        df_process[col] = pd.to_numeric(df_process[col], errors='coerce')

    print(f"   - 开始增强数据: 共 {len(target_players)} 名选手...")
    
    for player_name in target_players:
        # 清洗名字对比，防止空格问题
        player_clean = str(player_name).strip()
        player_data = df_process[df_process['player'] == player_clean]
        
        if len(player_data) < sample_size:
            # print(f"     [跳过] {player_name}: 数据行数不足 {sample_size}")
            continue

        for _ in range(n_samples):
            # 随机采样
            sample = player_data.sample(n=sample_size, replace=False)
            # 丢弃非特征列
            numeric_sample = sample.drop(columns=cols_to_exclude, errors='ignore')
            # 求平均
            avg_row = numeric_sample.mean(numeric_only=True)
            # 加回名字
            avg_row['Name'] = player_clean 
            augmented_rows.append(avg_row)
            
    if not augmented_rows:
        return pd.DataFrame()
        
    df_aug = pd.DataFrame(augmented_rows)
    
    # 列名处理：去掉 Mean 后缀 (如果你的增强前列名带Mean的话，如kingsafety列原本就带Mean)
    # 如果列名以 Mean 结尾，去掉它
    df_aug.columns = [col[:-4] if col.endswith("Mean") else col for col in df_aug.columns]
    
    return df_aug

# ==========================================
# 3. 主流程管道

def process_chess_group(group_name, player_list, conn, base_features_df):
    """
    处理单个组（正样本或负样本）的完整流程
    """
    print(f"\n>>> 开始处理组: {group_name} (人数: {len(player_list)})")
    
    # 清洗输入列表里的名字
    player_list = [str(x).strip() for x in player_list]
    placeholders = ",".join(["?"] * len(player_list))

    #1. Lifespan 表 ---
    query_life = f"SELECT * FROM lifespan WHERE player IN ({placeholders})"
    df_life = pd.read_sql(query_life, conn, params=player_list)
    
    #2. KingSafety 表 ---
    df_ks = fetch_white_black_data(conn, 'kingsafety', player_list)
    df_ks = df_ks.drop(columns=[c for c in df_ks.columns if c.endswith('List')])

    #3. 合并 KS 和 Lifespan ---
    df_ks = df_ks.sort_values("uid").reset_index(drop=True)
    df_life = df_life.sort_values("uid").reset_index(drop=True)
    
    if not df_ks['uid'].equals(df_life['uid']):
        print("   [警告] KingSafety 和 Lifespan 的 UID 不匹配！尝试使用 merge 而不是 concat...")
        df_base = pd.merge(df_ks, df_life, on='uid', suffixes=('', '_drop'))
        df_base = df_base.drop(columns=[c for c in df_base.columns if c.endswith('_drop')])
    else:
        df_life_clean = df_life.drop(columns=['uid'])
        df_base = pd.concat([df_ks, df_life_clean], axis=1)

    #4. CaptureNPawn 表 ---
    special_rename = {
        'White': {'': 'Name', 'PawnAdvanceDepth': 'AveragePawnAdvanceDepth'},
        'Black': {'': 'Name', 'PawnAdvanceDepth': 'AveragePawnAdvanceDepth'}
    }
    df_cp = fetch_white_black_data(conn, 'CaptureNPawn', player_list, special_rename=special_rename)
    df_cp = df_cp.drop(columns=[c for c in df_cp.columns if c.endswith('List')])
    
    #5. 大合并 (Join CaptureNPawn) ---
    df_base['uid'] = df_base['uid'].astype(int)
    df_cp['uid'] = df_cp['uid'].astype(int)
    
    df_merged = df_base.merge(
        df_cp,
        left_on=['uid', 'player_id'],
        right_on=['uid', 'Name'],
        how='left'
    )
    
    # 6. 洗数据
    cols_to_remove = ['Name', 'castled', 'LostCastlingRights', 'lost_castling_rights', 'game_id', 'player_id']
    df_merged.drop(columns=cols_to_remove, errors='ignore', inplace=True)
    df_merged.drop(columns=[c for c in df_merged.columns if c.endswith("Span")], inplace=True, errors='ignore')
    
    cols_to_drop = []
    rename_map = {}
    for col in df_merged.columns:
        if col.endswith("_x"):
            base = col[:-2]
            rename_map[col] = base
            if base + "_y" in df_merged.columns:
                cols_to_drop.append(base + "_y")
        elif col.endswith("_y"):
            base = col[:-2]
            if base + "_x" not in df_merged.columns:
                rename_map[col] = base
            else:
                cols_to_drop.append(col)
    
    df_merged.drop(columns=cols_to_drop, inplace=True)
    df_merged.rename(columns=rename_map, inplace=True)
    df_merged = df_merged.fillna(0)

    #7. 数据增强
    print("   7. 执行数据增强...")
    if 'player' in df_merged.columns:
        df_aug_result = data_augmentation(df_merged, player_list)
    else:
        print("   [错误] 找不到 player 列用于分组增强！")
        return pd.DataFrame()

    #8. 最终合并 (Original + Augmented)
    print("   8. 最终整合 (Original + Augmented)...")
    
    # 准备原始数据
    df_original_players = base_features_df[base_features_df['user'].isin(player_list)].copy()
    
    # [修复逻辑]：防止 user 改名导致 Name 重复
    # 1. 强制去除完全重复的列
    df_original_players = df_original_players.loc[:, ~df_original_players.columns.duplicated()]

    # 2. 如果想把 'user' 改成 'Name'，但我发现 'Name' 已经存在了，那就先把旧的 'Name' 删掉
    if 'user' in df_original_players.columns and 'Name' in df_original_players.columns:
        df_original_players = df_original_players.drop(columns=['Name'])
    
    # 3. 可以安全改名了
    if 'user' in df_original_players.columns:
        df_original_players = df_original_players.rename(columns={'user': 'Name'})
    # [修复逻辑结束]
    
    # ELO 映射表
    elo_mapping = base_features_df[['user', 'ELO']].drop_duplicates()
    elo_mapping['user'] = elo_mapping['user'].astype(str).str.strip()
    elo_dict = elo_mapping.set_index('user')['ELO'].to_dict()
    
    # 给增强数据补 ELO
    if not df_aug_result.empty:
        df_aug_result['ELO'] = df_aug_result['Name'].map(elo_dict)
    
        # 统一列筛选
        if 'PawnAdvanceDepth' in df_aug_result.columns:
            df_aug_result.rename(columns={'PawnAdvanceDepth': 'AveragePawnAdvanceDepth'}, inplace=True)
        
        # 使用 reindex 安全筛选
        df_orig_clean = df_original_players.reindex(columns=FINAL_COLUMNS)
        df_aug_clean = df_aug_result.reindex(columns=FINAL_COLUMNS)
        
        # 上下合并
        df_final = pd.concat([df_orig_clean, df_aug_clean], ignore_index=True)
        df_final = df_final.sort_values('Name')
    else:
        # 如果增强没产生数据（比如人数太少），至少返回原始数据
        print("   [提示] 增强数据为空，仅返回原始数据。")
        df_final = df_original_players.reindex(columns=FINAL_COLUMNS)
    
    print(f"   >>> 完成! 组 {group_name} 生成了 {len(df_final)} 行数据。")
    if 'ELO' in df_final.columns:
        print(f"       ELO 缺失数: {df_final['ELO'].isnull().sum()}")
    
    return df_final

def run_chess_pipeline():
    print("=== 初始化管道 ===")
    
    # 1. 建立 DB 连接
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"找不到数据库: {DB_PATH}")
    conn = get_db_connection(DB_PATH)
    
    # 2. 读取基础 CSV
    if not os.path.exists(BASE_CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV: {BASE_CSV_PATH}")
    base_df = pd.read_csv(BASE_CSV_PATH)
    # 清洗基础表的名字
    base_df['user'] = base_df['user'].astype(str).str.strip()
    if 'Name' in base_df.columns:
         base_df['Name'] = base_df['Name'].astype(str).str.strip()
    
    # 3. 运行 Positive 组
    final_pos = process_chess_group(
        group_name="Positive (1)", 
        player_list=POSITIVE_NAMES, 
        conn=conn, 
        base_features_df=base_df
    )
    
    # 4. 运行 Negative 组
    final_neg = process_chess_group(
        group_name="Negative (0)", 
        player_list=NEGATIVE_NAMES, 
        conn=conn, 
        base_features_df=base_df
    )
    
    conn.close()
    
    # 5. 保存结果
    output_pos = f"{TOPIC_NAME}_1_final.csv"
    output_neg = f"{TOPIC_NAME}_0_final.csv"
    
    final_pos.to_csv(output_pos, index=False)
    final_neg.to_csv(output_neg, index=False)
    
    print("\n" + "="*30)
    print("=== 所有任务完成 ===")
    print(f"正样本已保存: {output_pos}")
    print(f"负样本已保存: {output_neg}")
    print("="*30)

# ==========================================
# 4. 执行

if __name__ == "__main__":
    run_chess_pipeline()