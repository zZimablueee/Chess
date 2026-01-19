import pandas as pd
import sqlite3
import re
import os
import glob
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm

DB_PATH = r"C:\sqlite3\chess.db"
NAMEMAP_CSV_PATH = r"C:\Users\Administrator\Desktop\Chess\basic\namemaps\username_to_name.csv"
PGN_FOLDER_PATH = r"C:\Users\Administrator\Desktop\titled-tuesday\checked"
OUTPUT_CSV_PATH = r"C:\Users\Administrator\Desktop\Chess\data\output\matched_games_clocks.csv"


def load_lookup_data():

    print("正在加载名字映射表...")
    try:
        map_df = pd.read_csv(NAMEMAP_CSV_PATH)
        map_df.columns = [c.strip() for c in map_df.columns]
        username_to_realname = pd.Series(map_df['Name'].values, index=map_df['Username']).to_dict()
    except Exception as e:
        print(f"警告：名字映射表加载失败: {e}")
        username_to_realname = {}

    print(f"正在连接数据库: {DB_PATH} ...")
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT uid, Event, Date, Round, White, Black, EndTime, Termination FROM games"
        games_df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"致命错误：数据库读取失败: {e}")
        return {}, {}
    
    print("正在构建内存索引...")
    cols_to_str = ['Event', 'Date', 'Round', 'White', 'Black', 'EndTime', 'Termination']
    for col in cols_to_str:
        if col in games_df.columns:
            games_df[col] = games_df[col].astype(str).str.strip()
    
    game_lookup = {}
    keys_iterator = zip(
        games_df['Event'], games_df['Date'], games_df['Round'], 
        games_df['EndTime'], games_df['Termination'], 
        games_df['White'], games_df['Black']
    )
    
    for (event, date, rnd, endtime, term, white, black), uid in zip(keys_iterator, games_df['uid']):
        # Key是真实姓名
        key = (event, date, rnd, endtime, term, white, black)
        game_lookup[key] = uid
        
    print(f"索引构建完成，准备处理。")
    return username_to_realname, game_lookup

def process_pgn_file_worker(file_path):
    parsed_games = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return []

    games_raw = content.split('[Event "')
    
    header_pattern = re.compile(r'^\[(\w+) "([^"]+)"\]', re.MULTILINE)
    clock_pattern = re.compile(r'\{\[%clk\s+([\d:\.]+)\s*\]\}')

    for raw_game in games_raw:
        if not raw_game.strip():
            continue
            
        full_game_text = '[Event "' + raw_game
        
        headers = {}
        header_section = full_game_text[:1500] 
        matches = header_pattern.findall(header_section)
        for k, v in matches:
            headers[k] = v.strip() 
            
        # 检查必要字段是否存在
        required_keys = ['Event', 'Date', 'Round', 'EndTime', 'Termination', 'White', 'Black']
        if not all(k in headers for k in required_keys):
            continue
            
        clock_times = clock_pattern.findall(full_game_text)
        parsed_games.append((headers, clock_times))
            
    return parsed_games

def main():
    freeze_support()
    
    username_to_realname, game_lookup = load_lookup_data()
    
    if not game_lookup:
        return

    print("正在搜索 PGN 文件...")
    pgn_files = glob.glob(os.path.join(PGN_FOLDER_PATH, "*.pgn"))
    print(f"找到 {len(pgn_files)} 个 PGN 文件。")
    
    # 多进程
    num_processes = max(1, cpu_count() - 1)
    print(f"启动 {num_processes} 个解析进程...")
    
    final_results = []
    
    with Pool(processes=num_processes) as pool:
        iterator = pool.imap_unordered(process_pgn_file_worker, pgn_files, chunksize=1)
        
        for file_parsed_data in tqdm(iterator, total=len(pgn_files), unit="file"):
            if not file_parsed_data:
                continue
                
            # 匹配逻辑
            for headers, clock_times in file_parsed_data:
                # A. 名字转换 (Username -> RealName)
                white_u = headers.get('White')
                black_u = headers.get('Black')
                
                white_real = username_to_realname.get(white_u)
                black_real = username_to_realname.get(black_u)
                
                if not white_real or not black_real:
                    continue
                
                # B. 组装 Key (必须和构建索引时的顺序一致)
                key = (
                    headers['Event'],
                    headers['Date'],
                    headers['Round'],
                    headers['EndTime'],
                    headers['Termination'],
                    white_real.strip(),
                    black_real.strip()
                )
                
                # C. 查字典
                if key in game_lookup:
                    final_results.append({
                        'uid': game_lookup[key],
                        'clock_times': clock_times
                    })

    print(f"处理完成，共匹配到 {len(final_results)} 局游戏。")
    if final_results:
        df_result = pd.DataFrame(final_results)
        print(f"正在保存结果到 {OUTPUT_CSV_PATH} ...")
        df_result.to_csv(OUTPUT_CSV_PATH, index=False)
        print("保存成功！")
    else:
        print("未匹配到数据。")

if __name__ == '__main__':
    main()