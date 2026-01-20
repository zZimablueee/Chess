import pandas as pd
import sqlite3
from tqdm import tqdm  

DB_PATH =r"C:\sqlite3\chess.db" 
CSV_PATH =r"C:\Users\Administrator\Desktop\Chess\data\output\matched_games_clocks.csv"  
TABLE_NAME = 'games'        
DB_ID_COLUMN = 'uid'   
CSV_ID_COLUMN = 'uid'  

def add_clock_times_to_db():
    print("正在读取 CSV 文件...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"CSV 加载完成，共 {len(df)} 行。预览：")
    print(df.head(2))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    NEW_COLUMN_NAME = 'ClockTimes' 
    
    try:
        cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {NEW_COLUMN_NAME} TEXT")
        conn.commit()
        print(f"成功添加新列: {NEW_COLUMN_NAME}")
    except sqlite3.OperationalError:
        print(f"列 {NEW_COLUMN_NAME} 已存在，将直接更新数据。")

    print("正在准备更新数据...")

    update_data = []
    for index, row in df.iterrows():
        time_data = str(row['clock_times']) 
        uid = row[CSV_ID_COLUMN]
        update_data.append((time_data, uid))

    print(f"开始更新数据库中的 {len(update_data)} 条记录...")
    
    update_sql = f"UPDATE {TABLE_NAME} SET {NEW_COLUMN_NAME} = ? WHERE {DB_ID_COLUMN} = ?"
    
    try:

        batch_size = 10000
        for i in tqdm(range(0, len(update_data), batch_size)):
            batch = update_data[i : i + batch_size]
            cursor.executemany(update_sql, batch)
            conn.commit() #每1万条提交一次
            
        print("更新完成！")
        
    except Exception as e:
        print(f"更新过程中出错: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    add_clock_times_to_db()