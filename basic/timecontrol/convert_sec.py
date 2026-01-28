import sqlite3
import ast
from tqdm import tqdm

db_path = r"C:\sqlite3\chess.db"

#convert
#['0:03:00.1', '0:02:58.3',  '0:02:54.1', '0:02:52.2', '0:02:53.6', '0:02:50.6']
def time_str_to_seconds(t):
    h, m, s = t.split(':')
    return int(h)*60*60 + int(m)*60 + float(s)
times = ['0:03:00.1', '0:02:58.3',  '0:02:54.1', '0:02:52.2', '0:02:53.6', '0:02:50.6']

for t in times:
    seconds = time_str_to_seconds(t)
    print(seconds)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS clock(
               uid INTEGER PRIMARY KEY,
               seconds TEXT)""")
cursor.execute("SELECT uid, ClockTimes from games")

rows = cursor.fetchall()
insert_data = []

for uid, clock_str in tqdm(rows, desc="Converting ClockTimes"):
    if clock_str is None:
        continue
    # clock_str 不是真的列表，要先转换
    clock_list = ast.literal_eval(clock_str)
    seconds_list = []
    for t in clock_list:
        seconds = time_str_to_seconds(t)
        seconds_list.append(seconds)
    insert_data.append((uid,str(seconds_list)))

for chunk_start in tqdm(range(0,len(insert_data), 500), desc = "Inserting into database"):
    chunk = insert_data[chunk_start: chunk_start+500]
    cursor.executemany("""INSERT OR REPLACE INTO clock (uid, seconds) VALUES (?,?)""", chunk)
    conn.commit()

conn.close()