#这版是从mpiidk.py里拿过来的，删除了获取评估值的部分，仅仅可以从pgn得到的信息
from mpi4py import MPI
import chess.pgn
import os
import csv
import io

CSV_FIELDS = [
    'Event', 'Site', 'Date', 'Round', 'White', 'Black',
    'Result', 'WhiteElo', 'BlackElo', 'TimeControl',
    'EndTime', 'Termination', 'Moves'
]

# 配置路径
input_dir = r'C:\Users\Administrator\Desktop\titled-tuesday\checked'
output_dir = r'C:\Users\Administrator\Desktop\titled-tuesday\basicinfo'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
    # 主进程获取文件列表
    if rank == 0:
        pgn_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pgn')])
        os.makedirs(output_dir, exist_ok=True)
    else:
        pgn_files = None

    pgn_files = comm.bcast(pgn_files, root=0)

    for pgn_file in pgn_files:
        if rank == 0:
            file_path = os.path.join(input_dir, pgn_file)
            print(f"Processing {pgn_file}...")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_content = f.read()
        else:
            raw_content = None

        # 广播文件内容
        raw_content = comm.bcast(raw_content, root=0)

        # 解析所有游戏
        pgn_stream = io.StringIO(raw_content)
        games = []
        while True:
            game = chess.pgn.read_game(pgn_stream)
            if game is None:
                break
            games.append(game)

        # 动态任务分配
        total_games = len(games)
        chunk_size = (total_games + size - 1) // size
        start = rank * chunk_size
        end = min(start + chunk_size, total_games)

        local_results = []
        for game in games[start:end]:
            try:
                board = chess.Board()
                moves = []

                for move in game.mainline_moves():
                    board.push(move)
                    moves.append(move.uci())

                result = {field: game.headers.get(field, '') for field in CSV_FIELDS[:12]}
                result['Moves'] = ','.join(moves)
                local_results.append(result)
            except Exception as e:
                print(f"Error processing game: {str(e)}")

        # 主进程统一写入
        all_results = comm.gather(local_results, root=0)

        if rank == 0:
            output_path = os.path.join(output_dir, f"{pgn_file}_processed.csv")
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for sublist in all_results:
                    if sublist:  # 过滤空列表
                        writer.writerows(sublist)


#mpiexec -n 4 python "C:\Users\Administrator\Desktop\simple eda\simple eda\BASICSTEPS\basicinfo.py"
