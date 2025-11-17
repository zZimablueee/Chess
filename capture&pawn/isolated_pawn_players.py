##孤立兵 重叠兵 互相保护的兵相关 以player形式存储
import pandas as pd
import sqlite3
from tqdm import tqdm
import os
def import_csv_to_db(csv_path, db_path, table_name="CaptureNPawn", chunksize=2000):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    first_chunk = True
    new_cols = []
    
    for chunk in pd.read_csv(csv_path, dtype=str, chunksize=chunksize):
        if first_chunk:
            #获取旧表列
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            existing_cols = [row[1] for row in cursor.fetchall()]
            # 找出 CSV 中的新列
            new_cols = [c for c in chunk.columns if c not in existing_cols]
            print("旧表列：", existing_cols)
            print("CSV 新列：", new_cols)
            #给旧表添加这些新列
            for col in new_cols:
                cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" TEXT;')
            conn.commit()
            print(f"已为旧表添加 {len(new_cols)} 个新列")
            first_chunk = False
        #处理当前 chunk
        if "uid" not in chunk.columns:
            raise ValueError("CSV 中必须包含 uid 列！")
        #仅保留uid和新列
        sub = chunk[["uid"] + new_cols].copy()
        #批量更新
        for _, row in sub.iterrows():
            uid = row["uid"]
            for col in new_cols:
                cursor.execute(
                    f'UPDATE "{table_name}" SET "{col}"=? WHERE uid=?',
                    (row[col], uid)
                )
        conn.commit()
        print(f"本次 chunk（{len(chunk)} 行）更新完成")
    conn.close()
    print(f"全部 CSV 数据已按 uid 成功更新到旧表{table_name}中！")

def calculate_player_isolated_pawn(df):
    df_ea=pd.concat([
        df[['White_player_id','White_PawnIsolateScoreMean','White_PawnOverlapScoreMean',
            'White_PawnProtectScoreMean']]
            .rename(columns={
                'White_player_id':'user',
                'White_PawnIsolateScoreMean':'PawnIsolateScore',
                'White_PawnOverlapScoreMean':'PawnOverlapScore',
                'White_PawnProtectScoreMean':'PawnProtectScore',
            }),
    df[['Black_player_id','Black_PawnIsolateScoreMean','Black_PawnOverlapScoreMean',
    'Black_PawnProtectScoreMean']].rename(columns={
    'Black_player_id':'user',
    'Black_PawnIsolateScoreMean':'PawnIsolateScore',
    'Black_PawnOverlapScoreMean':'PawnOverlapScore',
    'Black_PawnProtectScoreMean':'PawnProtectScore',
    })
],axis=0)
    cols=['PawnIsolateScore','PawnOverlapScore','PawnProtectScore']
    df_ea[cols]=df_ea[cols].apply(pd.to_numeric,errors='coerce')
    grouped=df_ea.groupby('user')
    df_mean=df_ea.groupby('user',as_index=False).mean(numeric_only=True)
    return df_mean
def comprehensive_player_analysis(df,output_csv):
    df_pawn_isolate=calculate_player_isolated_pawn(df)
    df_pawn_isolate.to_csv(output_csv,index=False)
    print(f"已生成 {output_csv}")
    return df_pawn_isolate

#把CaptureNPawn的整理成player表格
#把CaptureNPawn的player表格和原先的玩家表格合并
def import_csv_as_new_table(csv_path, db_path, table='players', chunksize=2000):
    """
    读取 CSV 文件，并将其列添加到已存在的 SQLite 表中（保留原有数据）。
    根据 user 列匹配更新原表数据。
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    conn.commit()

    # 获取原表列名
    cursor.execute(f'PRAGMA table_info("{table}")')
    existing_cols = [row[1] for row in cursor.fetchall()]

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str):
        # 必须存在 user 列
        if "user" not in chunk.columns:
            raise ValueError("CSV 中必须包含 user 列，用于匹配玩家。")
        # 找出 CSV 中的新增列（排除 user）
        new_cols = [c for c in chunk.columns if c not in existing_cols and c != "user"]
        # 给表添加新列
        for col in new_cols:
            cursor.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" TEXT;')
            existing_cols.append(col)
            print(f"新增列: {col}")

        # 没有新增列 跳过所有 update
        if not new_cols:
            print("本批数据无新增列，跳过更新。")
            continue
        # 一行一行更新
        for row in chunk.itertuples(index=False):
            row_dict = row._asdict()
            user = row_dict.get('user')
            if user is None:
                continue

            set_clause = ", ".join([f'"{c}"=?' for c in new_cols])
            values = [row_dict[c] for c in new_cols]
            values.append(user)

            sql = f'UPDATE "{table}" SET {set_clause} WHERE user=?'
            cursor.execute(sql, values)
        conn.commit()
        print(f"已处理 {len(chunk)} 行")
    conn.close()
    print(f"CSV 导入完成，表 {table} 已更新")


def main():
    df = pd.read_csv(r"C:\Users\Administrator\Desktop\playerstyles\pawn100w.csv")
    import_csv_to_db(r"C:\Users\Administrator\Desktop\playerstyles\pawn100w.csv", r"C:\sqlite3\chess.db", table_name='CaptureNPawn', chunksize=2000)
    comprehensive_player_analysis(df, "player_pawn.csv")
    import_csv_as_new_table("player_pawn.csv", r"C:\sqlite3\chess.db", table="players")

if __name__=='__main__':
    main()
