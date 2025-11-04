import pandas as pd
import sqlite3
from tqdm import tqdm
import os

def import_csv_to_db(csv_path, db_path, table_name, chunksize=2000):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    first_chunk = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str):
        if first_chunk:
            cols = [f'"{c}" TEXT' for c in chunk.columns]
            col_def = ", ".join(cols)
            create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_def});'
            cursor.execute(create_sql)
            first_chunk = False
            print(f"已创建新表 {table_name}，共 {len(chunk.columns)} 列")

        placeholders = ", ".join(["?"] * len(chunk.columns))
        insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
        cursor.executemany(insert_sql, chunk.values.tolist())
        conn.commit()
        print(f"已插入 {len(chunk)} 行")

    conn.close()
    print("全部完成csv导入到数据库表 kingsafety 中")

def calculate_player_kingsafety(df):
    df_ea=pd.concat([
        df[['White_player_id','White_CenterControlScoreMean','White_PieceActivityScoreMean',
            'White_KingSafetyScoreMean','White_CastlingScoreMean',
            'White_KingTropismScoreMean','White_KingDefendersScoreMean','White_KingPawnShieldScoreMean',
            'White_KingZoneControlScoreMean',
            'White_KingDiagonalExposureScoreMean','White_KingEscapeSquaresScoreMean',]]
            .rename(columns={
                'White_player_id':'user',
                'White_CenterControlScoreMean':'CenterControlScore',
                'White_PieceActivityScoreMean':'PieceActivityScore',
                'White_KingSafetyScoreMean':'KingSafetyScore',
                'White_CastlingScoreMean':'CastlingScore',
                'White_KingTropismScoreMean':'KingTropismScore',
                'White_KingDefendersScoreMean':'KingDefendersScore',
                'White_KingPawnShieldScoreMean':'KingPawnShieldScore',
                'White_KingZoneControlScoreMean':'KingZoneControlScore',
                'White_KingDiagonalExposureScoreMean':'KingDiagonalExposureScore',
                'White_KingEscapeSquaresScoreMean':'KingEscapeSquaresScore'
            }),
    df[['Black_game_id','Black_player_id','Black_CenterControlScoreMean','Black_PieceActivityScoreMean',
    'Black_KingSafetyScoreMean','Black_CastlingScoreMean','Black_KingTropismScoreMean',
    'Black_KingDefendersScoreMean','Black_KingPawnShieldScoreMean','Black_KingZoneControlScoreMean',
    'Black_KingDiagonalExposureScoreMean','Black_KingEscapeSquaresScoreMean',]].rename(columns={
    'Black_player_id':'user',
    'Black_CenterControlScoreMean':'CenterControlScore',
    'Black_PieceActivityScoreMean':'PieceActivityScore',
    'Black_KingSafetyScoreMean':'KingSafetyScore',
    'Black_CastlingScoreMean': 'CastlingScore',
    'Black_KingTropismScoreMean':'KingTropismScore',
    'Black_KingDefendersScoreMean':'KingDefendersScore',
    'Black_KingPawnShieldScoreMean':'KingPawnShieldScore',
    'Black_KingZoneControlScoreMean':'KingZoneControlScore',
    'Black_KingDiagonalExposureScoreMean':'KingDiagonalExposureScore',
    'Black_KingEscapeSquaresScoreMean':'KingEscapeSquaresScore'
    })
],axis=0)
    cols=['CenterControlScore','PieceActivityScore','KingSafetyScore','CastlingScore',
        'KingTropismScore','KingDefendersScore','KingPawnShieldScore',
        'KingZoneControlScore','KingDiagonalExposureScore','KingEscapeSquaresScore']
    df_ea[cols]=df_ea[cols].apply(pd.to_numeric,errors='coerce')
    grouped=df_ea.groupby('user')
    df_mean=df_ea.groupby('user',as_index=False).mean(numeric_only=True)
    return df_mean
def comprehensive_player_analysis(df,output_csv):
    df_KSplayer=calculate_player_kingsafety(df)
    df_KSplayer.to_csv(output_csv,index=False)
    print(f"已生成 {output_csv}")
    return df_KSplayer

#把kingsafety的整理成player表格
#把kingsafety的player表格和原先的玩家表格合并
def import_csv_as_new_table(csv_path, db_path, new_table='players', chunksize=2000):
    """
    把一个CSV文件写入SQLite数据库中，创建一个新表。
    如果表已存在，会覆盖旧表。
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    #表已存在就先删再建
    cursor.execute(f'DROP TABLE IF EXISTS "{new_table}";')
    conn.commit()

    first_chunk = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str):
        if first_chunk:
            cols = [f'"{c}" TEXT' for c in chunk.columns]
            col_def = ", ".join(cols)
            create_sql = f'CREATE TABLE "{new_table}" ({col_def});'
            cursor.execute(create_sql)
            conn.commit()
            first_chunk = False
            print(f"已创建新表 {new_table}，包含 {len(chunk.columns)} 列")

        placeholders = ", ".join(["?"] * len(chunk.columns))
        insert_sql = f'INSERT INTO "{new_table}" VALUES ({placeholders})'
        cursor.executemany(insert_sql, chunk.values.tolist())
        conn.commit()
        print(f"已插入 {len(chunk)} 行")

    conn.close()
    print(f"CSV 导入完成，已写入数据库表 {new_table}")

def main():
    #从数据库的kingsafety里读取数据，然后应用函数重新组合一下，
    #生成新表KSplayers.csv,再把这个生成的csv写入到数据库里一个新表players中
    db_path=r"C:\sqlite3\chess.db"
    source_table='kingsafety'
    output_csv='KSplayers.csv'
    new_table='players'
    print(f"从{source_table}读取原始数据")
    conn=sqlite3.connect(db_path)
    df=pd.read_sql_query(f"SELECT * FROM {source_table}",conn)
    conn.close()
    print(f"计算玩家KingSafety并输出文件")
    df_KSplayer = comprehensive_player_analysis(df, output_csv)
    print(f"已生成 CSV: {output_csv}")
    print(f"把 {output_csv} 导入新表 {new_table} ")
    import_csv_as_new_table(output_csv,db_path,new_table)
    print("完成")

if __name__=='__main__':
    main()
