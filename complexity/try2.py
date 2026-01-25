#
import shutil
import ast
class ProjectConfig:
    """项目路径与配置管理"""
    def __init__(self, script_path, input_csv_path, output_filename, force_restart=False):
        self.script_dir = os.path.dirname(os.path.abspath(script_path))
        self.project_root = os.path.dirname(self.script_dir) # 假设 code 在 root 下
        
        # 定义目录结构
        self.data_dir = os.path.join(self.project_root, 'data')
        self.output_dir = os.path.join(self.data_dir, 'output')
        self.temp_dir = os.path.join(self.data_dir, 'temp')
        self.status_dir = os.path.join(self.data_dir, 'status')
        
        # 文件路径
        self.input_csv = input_csv_path
        self.output_csv = os.path.join(self.output_dir, output_filename)
        self.force_restart = force_restart

    def setup_directories(self, rank):
        """主进程负责创建目录，并根据设置清理旧状态"""
        if rank == 0:
            # 创建目录
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.temp_dir, exist_ok=True)
            #os.makedirs(self.status_dir, exist_ok=True)
            
            # 如果强制重跑，清理 temp 和 status
            if self.force_restart:
                print(f"[System] 检测到强制重启动 (FORCE_RESTART=True)，正在清理旧状态...")
                self._clean_dir(self.temp_dir)
                self._clean_dir(self.status_dir)
                print(f"[System] 清理完成，将重新分析所有数据。")

    def _clean_dir(self, dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"删除 {file_path} 失败: {e}")

    def get_temp_csv_path(self, rank):
        return os.path.join(self.temp_dir, f"partial_result_rank_{rank}.csv")

    def get_status_json_path(self, rank):
        return os.path.join(self.status_dir, f"processed_uids_rank_{rank}.json")
    
def get_processed_uids_from_csv(csv_path):
    """
    读取备份CSV文件中已经存在的 row_idx，返回一个集合。
    用于替代之前的 JSON 断点记录。
    """
    processed_uids = set()
    if not os.path.exists(csv_path):
        return processed_uids
    
    try:
        # 使用 buffering=1 防止读取过慢，但这里主要是读
        with open(csv_path, 'r', encoding='utf-8') as f:
            # 预读取判断是否为空
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return processed_uids
            f.seek(0)
            
            # 使用 DictReader 自动处理表头
            reader = csv.DictReader(f)
            if reader.fieldnames and 'row_idx' in reader.fieldnames:
                for row in reader:
                    try:
                        # 确保读取的是数字
                        val = row.get('row_idx')
                        if val:
                            processed_uids.add(int(val))
                    except ValueError:
                        continue 
    except Exception as e:
        print(f"[Warning] 读取断点文件 {csv_path} 失败: {e}")
        
    return processed_uids

import os
os.environ["PREFECT_API_URL"] = ""
os.environ["PREFECT_API_KEY"] = ""

#environment setup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chess
import os
import json
import sys
import chess.pgn
import chess.engine
import csv
import io
import traceback
import subprocess
import time
import math
import numpy as np
import asyncio
from tqdm import tqdm
import tqdm
from pathlib import Path
from stockfish import Stockfish  
from mpi4py import MPI
#强制更换 event loop（解决部分 Windows 异步问题）
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#from prefect.tasks import task_input_hash
#from prefect.cache_policies import NO_CACHE
#from prefect import flow, task, get_run_logger, serve
from typing import Dict, List, Tuple, Any, Optional

#断点重续功能
def save_processed_uids(processed_uids,json_path):
    with open(json_path,'w') as f:
        json.dump(list(processed_uids), f)
def load_processed_uids(json_path):
    if os.path.exists(json_path):
        with open(json_path,'r') as f:
            return set(json.load(f))
    return set()
def filter_unprocessed_uids(original_df, rank):
    processed_uids = load_processed_uids(rank)
    if 'uid' in original_df.columns:
        unprocessed_df = original_df[~original_df['uid'].isin(processed_uids)]
    else:
        unprocessed_df = original_df
    return unprocessed_df

#  复杂度分析部分
class ComplexityAnalyzer:
    """  Implementation of Chess Decision Complexity based on Barthelemy (2025).  """
    def __init__(self, delta_0=10.0):
        self.delta_0=delta_0
    
    def calculate_metrics(self, e1, e2):
        #when force move
        if e2 is None:
            return float('inf'), 1.0, 0.0
        delta = e1 - e2
        if delta < 0: delta = 0

        try:
            p_optimal = 1 / (1 + math.exp( -delta / self.delta_0))
        except OverflowError:
            p_optimal = 0.0

        try:
            if p_optimal <=0 : entropy = 1.0
            else:
                entropy = -math.log2(p_optimal)
        except ValueError:
            entropy = 0.0

        return delta, p_optimal, entropy  
    
    def score_to_cp(self, score_obj, mate_cap=10000):
        """
        Convert chess.engine.Score to numeric CP for complexity calculation.
        Ensure returns Side-To-Move perspective (Standard UCI behavior).
        """
        if isinstance(score_obj, chess.engine.Mate):
            mate_steps = score_obj.mate()
            # Positive mate means current side wins
            if mate_steps > 0:
                return mate_cap - mate_steps
            else:
                return -mate_cap - mate_steps
        else:
            return score_obj.score()

#  lichess逻辑分析部分
class LichessMateAnalyzer:
    def __init__(self):
        self.INACCURACY='Inaccuracy'
        self.MISTAKE='Mistake'
        self.BLUNDER='Blunder'

    def analyze_mate(self,prev_cp,prev_mate,curr_cp,curr_mate,is_white_move):
        def to_mover(val_cp,val_mate):
            if is_white_move:
                return val_cp,val_mate
            else:
                c=-val_cp if val_cp is not None else None
                m=-val_mate if val_mate is not None else None
                return c,m
            
        p_cp, p_mate = to_mover(prev_cp, prev_mate)
        c_cp, c_mate = to_mover(curr_cp, curr_mate)
        has_mate = lambda m: m is not None
        mate_positive = lambda m: m is not None and m > 0
        mate_negative = lambda m: m is not None and m < 0
        sequence=None
        if not has_mate(p_mate) and mate_negative(c_mate): sequence = "MateCreated"
        elif mate_positive(p_mate) and not has_mate(c_mate): sequence = "MateLost"
        elif mate_positive(p_mate) and mate_negative(c_mate): sequence = "MateLost"
        elif mate_positive(p_mate) and mate_positive(c_mate): sequence = "MateDelayed"

        if sequence == "MateCreated":
            if p_cp is not None and p_cp < -999: return self.INACCURACY
            elif p_cp is not None and p_cp < -700: return self.MISTAKE
            else: return self.BLUNDER
        elif sequence == "MateLost":
            if c_cp is not None and c_cp > 999: return self.INACCURACY
            elif c_cp is not None and c_cp > 700: return self.MISTAKE
            else: return self.BLUNDER
        return None

class FullLichessAnalyzer:
    def __init__(self):
        self.mate_analyzer = LichessMateAnalyzer()
        self.MULTIPLIER = -0.00368208
        self.BLUNDER_THRESHOLD = 0.3
        self.MISTAKE_THRESHOLD = 0.2
        self.INACCURACY_THRESHOLD = 0.1

    def _winning_chances(self, cp_value):
        try:
            return 2 / (1 + math.exp(self.MULTIPLIER * cp_value)) - 1
        except OverflowError:
            return 1.0 if cp_value > 0 else -1.0

    def get_judgment(self, prev_cp, prev_mate, prev_calc_cp, curr_cp, curr_mate, curr_calc_cp, is_white_move):
        # 1. Mate 判定优先
        if prev_mate is not None or curr_mate is not None:
            judgment = self.mate_analyzer.analyze_mate(prev_cp, prev_mate, curr_cp, curr_mate, is_white_move)
            if judgment: return judgment

        # 2. CP 判定
        prev_win = self._winning_chances(prev_calc_cp)
        curr_win = self._winning_chances(curr_calc_cp)
        delta = curr_win - prev_win
        loss = -delta if is_white_move else delta
        
        if loss >= self.BLUNDER_THRESHOLD: return "Blunder"
        elif loss >= self.MISTAKE_THRESHOLD: return "Mistake"
        elif loss >= self.INACCURACY_THRESHOLD: return "Inaccuracy"
        return "-"

#  lichess部分结束

def get_raw_score_value(score, board):
    """
    获取原始引擎分数 (ComplexityE)，逻辑完全对齐 get_eval_str。
    """
    raw_val = 0
    
    # 1. 先获取“相对于当前走棋方”的原始数值
    if isinstance(score, chess.engine.Mate):
        mate_steps = score.mate()
        # 处理杀棋分数：
        # mate_steps > 0 代表当前方赢 (比如 +1 代表当前方一步杀)
        # 转化规则：越快赢分数越高 (10000 - steps)
        if mate_steps > 0:
            raw_val = 10000 - mate_steps
        else:
            # 负数代表当前方被杀 (比如 -1 代表当前方被一步杀)
            raw_val = -10000 - mate_steps
    else:
        # CP 情况，直接拿整数
        raw_val = score.score()

    # 2. 视角转换
    if board.turn == chess.WHITE:
        return raw_val
    else:
        return -raw_val

def get_eval_str(score, board):   
    #score is mate
    if isinstance(score, chess.engine.Mate):
        mate_value = score.mate()
        return str(mate_value if board.turn==chess.WHITE else -mate_value) #make sure the result always for white perspective
    else:
        #score is centipawn
        cp_score = score.score()
        return f"{cp_score/100.0:.2f}" if board.turn==chess.WHITE else f"{-cp_score/100.0:.2f}" 
    
def move_accuracy_percent(before, after):
    """_summary_
    Calcute move accuracy percentage basedon the evaluation-change

    Args:
        before (_type_): pre-move evaluation change
        after (_type_): post-move evaluation change

    Returns:
        float: accuracy percentage (0.0-100.0)
    """    
    if after >= before:
        return 100.0 #didnt get worse,think it's a perfect move
    else:
        win_diff = before - after
        raw = 103.1668100711649 * math.exp(-0.04354415386753951 * win_diff) + -3.166924740191411
        return max(min(raw + 1, 100), 0)

def winning_chances_percent(cp):
    """_summary_
    convert centipawns into win probability percentage

    Args:
        cp (int): engine raw score in centipawns

    Returns:
        float: win probability percentage(0.0--100.0)
    """    
    multiplier = -0.00368208
    chances = 2 / (1 + math.exp(multiplier * cp)) - 1
    return 50 + 50 * max(min(chances, 1), -1)

def harmonic_mean(values):
    """_summary_
    calculate harmonic mean of a sequence

    Args:
        values (_type_): iterable of numerical numbers

    Returns:
        float: harmonic mean
    """    
    n = len(values)
    if n == 0:
        return 0
    reciprocal_sum = sum(1 / x for x in values if x)
    return n / reciprocal_sum if reciprocal_sum else 0

def std_dev(seq):
    """_summary_
    calculate standard deviation with special empty sequence handling(population standard deviation formula)
    0.5 for empty sequences

    Args:
        seq (int/float): numerical sequence

    Returns:
        float: standard deviation
    """    
    if len(seq) == 0:
        return 0.5 
    mean = sum(seq) / len(seq)
    variance = sum((x - mean) ** 2 for x in seq) / len(seq)
    return math.sqrt(variance)

def volatility_weighted_mean(accuracies, win_chances, is_white):
    """_summary_

    Args:
        accuracies (list): list of accuracy percentages
        win_chances (list): list of win probability perentages
        is_white (bool): indicate white's move

    Returns:
        float: weighted mean accuracy
    """    
    weights = [] #list to put each move's weight
    for i in range(len(accuracies)):
        base_index = i * 2 + 1 if is_white else i * 2 + 2
        start_idx = max(base_index - 2, 0)
        end_idx = min(base_index + 2, len(win_chances) - 1)

        sub_seq = win_chances[start_idx:end_idx+1]
        weight = max(min(std_dev(sub_seq), 12), 0.5)
        weights.append(weight)
    
    #Weight calculation: 给胜率波动大的步子(棋局转折点）更高的权重
    #    1. Determine base index: odd for white, even for black
    #    2. Create 5-point window centered at base index
    #    3. Window std_dev → weight = clamp(std_dev, 0.5, 12)

    weighted_sum = sum(a*w for a,w in zip(accuracies,weights))
    total_weight = sum(weights)
    weighted_mean = weighted_sum / total_weight if total_weight else 0

    return weighted_mean

class SimpleStockfishEngine:
    """_summary_ 
    a python interface for stockfish engine,provide basic board analysis and evaluation
    engine initialization and protocol handshake
    deep position analysis
    position evaluation
    process management
    """     
    def __init__(self, engine_path, threads=1, multi_pv=2):
        """_summary_
        initialize and launch Stockfish engine process
        perform uci handshake, set computation threads

        Args:
            engine_path: path to stockfish
            threads (int, optional): number of cpu threads to use Defaults to 1.

        Raises:
            FileNotFoundError: engine path is invalid
            timeouterror: protocol handshake times out
        """        
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"引擎路径不存在: {engine_path}")
        
        self.process = subprocess.Popen(
            str(self.engine_path),
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=1
        )
        
        self._send_command("uci")
        self._wait_for("uciok")
        self._send_command(f"setoption name Threads value {threads}")
        self._send_command(f"setoption name MultiPV value {multi_pv}")
        self._send_command("isready")
        self._wait_for("readyok")
    
    def _send_command(self, command):
        self.process.stdin.write(f"{command}\n")
        self.process.stdin.flush()
    
    def _wait_for(self, text, timeout=5.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline().strip()
            if text in line:
                return line
        raise TimeoutError(f"等待'{text}'超时")
    
    def _read_until_empty(self):
        lines = []
        while True:
            line = self.process.stdout.readline().strip()
            if not line:
                break
            lines.append(line)
        return lines
    
    def analyse(self, board, depth=16):
        """_summary_
        analyze current board position and return dict with position score and best moves

        Args:
            board (_type_): chess.Board represents current position
            depth (int, optional): moves to look ahead. Defaults to 16.

        Returns:
            'score': postion score(CP/Mate)
            'pv': list of principal variation moves 
        """        
        fen = board.fen()
        self._send_command(f"position fen {fen}")
        self._send_command(f"go depth {depth}")
        
        pv_scores = {}
        best_move = None
        
        while True:
            line = self.process.stdout.readline().strip()
            if "bestmove" in line:
                best_move = line.split()[1]
                break
            
            if "score" in line and "multipv" in line:
                try:
                    parts = line.split()
                    if "multipv" in parts:
                        mpv_idx = parts.index("multipv")
                        mpv_id = int(parts[mpv_idx + 1])
                    else:
                        mpv_id = 1

                    score_idx = parts.index("score")
                    score_type = parts[score_idx + 1]
                    score_val = int(parts[score_idx + 2])
                
                    if score_type == "cp":
                        score_obj = chess.engine.Cp(score_val)
                    elif score_type == "mate":
                        score_obj = chess.engine.Mate(score_val)
                    else:
                        score_obj = chess.engine.Cp(0)
                
                    pv_scores[mpv_id] = score_obj
                
                except (ValueError, IndexError):
                    pass
            
        if 1 not in pv_scores:
            pv_scores[1] = chess.engine.Cp(0)
            
        return {"pv_scores": pv_scores, "best_move": best_move}
    
    def set_fen_position(self,fen):
        self._send_command(f"position fen {fen}")

    def get_evaluation(self, depth=10):
        """_summary_
        quick evaluation of current position

        Args:
            depth (int, optional): calculation depth. Defaults to 10.

        Returns:
            {"type": Score type ("cp" or "mate"), "value": Numeric value}
        """        
        self._send_command(f"go depth {depth}")

        score_type = None
        score_value = None

        while True:
            line = self.process.stdout.readline().strip()
            if "bestmove" in line:
                break
            if "score" in line:
                parts = line.split()
                try:
                    score_idx = parts.index("score")
                    score_type = parts[score_idx + 1]
                    score_value = int(parts[score_idx + 2])
                except (ValueError, IndexError):
                    continue  # 跳过无效行

        if score_type is None or score_value is None:
            return {"type": "cp", "value": 0}

        return {"type": score_type, "value": score_value}
    
    def quit(self):
        """_summary_
        terminate engine process; force kill when timeout 
        """        
        self._send_command("quit")
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()

def get_score_value(score, board, mate_score=1000):
    #convert the stockfish score to numeric value,
    #adjust sign based on current player's turn
    """_summary_

    Args:
        score (cp/mate): get from stockfish
        board (chess.Board): current state
        mate_score (int, optional): Defaults to 1000.

    Returns:
        int: a numeric value
    """    
    if isinstance(score, chess.engine.Mate):
        mate = score.mate()
        value = mate_score if mate > 0 else -mate_score
        if mate > 0: #white will checkmate black
            value = mate_score - mate
        else:  #black will checkmate white
            value = -mate_score - mate
    else:  # Cp
        value = score.score()
        
    if board.turn == chess.BLACK:
        value = -value
        
    return value


def process_csv_with_offset(csv_file, engine, depth, is_verbose, start_idx=0, 
                            backup_csv_path=None):
    """
    全功能修复版：
    1. 修复了 move_number 未初始化的报错。
    2. 增加了异常发生时的强制硬盘写入(flush/fsync)，防止死循环。
    3. 移除内存堆积逻辑，防止 OOM。
    """    
    lichess_analyzer = FullLichessAnalyzer()
    complexity_analyzer = ComplexityAnalyzer(delta_0=10.0)
    
    # 1. 读取断点
    current_uids_set = set()
    if backup_csv_path:
        current_uids_set = get_processed_uids_from_csv(backup_csv_path)
        if len(current_uids_set) > 0 and is_verbose:
            print(f"检测到断点文件，已跳过 {len(current_uids_set)} 局")

    # 2. 准备文件写入
    backup_file_handle = None
    csv_writer = None
    
    if backup_csv_path:
        # 检测是否需要写表头
        file_exists = os.path.isfile(backup_csv_path) and os.path.getsize(backup_csv_path) > 0
        
        # 使用行缓冲 buffering=1
        backup_file_handle = open(backup_csv_path, 'a', newline='', encoding='utf-8', buffering=1)
        csv_writer = csv.writer(backup_file_handle)
        
        if not file_exists:
            headers = [
                "row_idx","UID", "White", "Black", "White CP Loss", "White Accuracy", 
                "Black CP Loss", "Black Accuracy", "Stage Analysis", "Evaluation", "ComplexityE", 
                "Judgments", "White Blunders", "White Mistakes", "White Inaccuracies", 
                "Black Blunders", "Black Mistakes", "Black Inaccuracies",
                "Complexity E1", "Complexity E2", "Complexity P", "Complexity S", 
                "White Complexity Sum", "Black Complexity Sum"
            ]
            csv_writer.writerow(headers)
            backup_file_handle.flush()

    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as file:
            first_line = file.readline().strip()
            file.seek(0)
            delimiter = '\t' if '\t' in first_line else ','
            csv_reader = csv.DictReader(file, delimiter=delimiter)
            rows = list(csv_reader)
            
            pbar = tqdm.tqdm(total=len(rows), desc="处理进度", disable=not is_verbose)
            
            for local_idx, row in enumerate(rows):
                global_idx = start_idx + local_idx
                raw_uid = row.get('uid') or row.get('id')
                if raw_uid:
                    real_uid = raw_uid
                else:
                    real_uid = global_idx + 1
                
                # --- 断点跳过 ---
                if global_idx in current_uids_set:
                    pbar.update(1)
                    continue
                
                # --- 基础信息 ---
                white_player = row.get('White', 'Unknown')
                black_player = row.get('Black', 'Unknown')
                
                if is_verbose:
                    pbar.set_description(f"Game {global_idx}: {white_player} vs {black_player}")

                # 定义空结果 (用于报错时占位)
                empty_result = (global_idx, real_uid, white_player, black_player, None, None, None, None, None, None, None, None,
                                0, 0, 0, 0, 0, 0, None, None, None, None, 0, 0)
                
                moves_str = row.get('Moves', '')
                
                # 检查空棋谱或非法格式
                if not moves_str:
                    if csv_writer:
                        csv_writer.writerow(list(empty_result))
                        backup_file_handle.flush() # 即使是空也要写入
                    pbar.update(1)
                    continue
                
                moves_list = moves_str.split(',')
                # 简单过滤非法移动
                if any(m.strip() and (len(m.strip()) < 4 or len(m.strip()) > 5) for m in moves_list):
                    if csv_writer:
                        csv_writer.writerow(list(empty_result))
                        backup_file_handle.flush()
                    pbar.update(1)
                    continue

                # --- 变量初始化 ---
                game_acc_white, game_acc_black = [], []
                game_cp_white, game_cp_black = 0, 0
                game_win_chances = []
                evaluation_list = []  
                raw_eval_list = []
                judgment_list = [] 
                stats = {
                    "white": {'Blunder': 0, 'Mistake': 0, 'Inaccuracy': 0},
                    "black": {'Blunder': 0, 'Mistake': 0, 'Inaccuracy': 0}
                }
                e1_list, e2_list, p_list, s_list = [], [], [], []
                cum_s_white, cum_s_black = 0.0, 0.0

                board = chess.Board()
                
                # 【关键修复】初始化 move_number
                move_number = 1  
                
                analysis_success = True
                
                try:
                    # 1. 初始局面分析
                    result_current = engine.analyse(board, depth)
                    score_obj_best = result_current["pv_scores"].get(1, chess.engine.Cp(0))
                    e1_obj = result_current["pv_scores"].get(1)
                    e2_obj = result_current["pv_scores"].get(2)
                    
                    score_current = get_score_value(score_obj_best, board)
                    cp_curr, mate_curr, calc_curr = extract_white_relative_score(score_obj_best, board)
                    
                    # 2. 遍历每一步
                    for move_str in moves_list:
                        move_str = move_str.strip()
                        if not move_str: continue
                        
                        # 状态保存
                        score_before = score_current
                        cp_prev, mate_prev, calc_prev = cp_curr, mate_curr, calc_curr
                        
                        # 复杂度计算
                        is_white_turn = board.turn 
                        e1_val = complexity_analyzer.score_to_cp(e1_obj)
                        e2_val = complexity_analyzer.score_to_cp(e2_obj) if e2_obj else None
                        delta, p_opt, entropy = complexity_analyzer.calculate_metrics(e1_val, e2_val)
                        
                        e1_list.append(str(e1_val))
                        e2_list.append(str(e2_val) if e2_val is not None else "inf")
                        p_list.append(f"{p_opt:.3f}")
                        s_list.append(f"{entropy:.3f}")
                        
                        if is_white_turn: cum_s_white += entropy
                        else: cum_s_black += entropy

                        # 走棋
                        move = chess.Move.from_uci(move_str)
                        if move not in board.legal_moves:
                            analysis_success = False; break
                        board.push(move)
                        
                        # 新局面分析
                        result_current = engine.analyse(board, depth)
                        score_obj_best = result_current["pv_scores"].get(1, chess.engine.Cp(0))
                        score_current = get_score_value(score_obj_best, board)
                        eval_str_current = get_eval_str(score_obj_best, board)

                        raw_val_int = get_raw_score_value(score_obj_best, board)
                        evaluation_list.append(eval_str_current)
                        raw_eval_list.append(str(raw_val_int))

                        cp_curr, mate_curr, calc_curr = extract_white_relative_score(score_obj_best, board)
                        e1_obj = result_current["pv_scores"].get(1)
                        e2_obj = result_current["pv_scores"].get(2)
                                                
                        # 准确率与Lichess判定
                        win_before_white = winning_chances_percent(score_before)
                        win_after_white = winning_chances_percent(score_current)
                        game_win_chances.append(win_after_white)
                        
                        if board.turn == chess.WHITE: # 黑方刚走完
                            is_white_move_finished = False
                            win_before = 100 - win_before_white
                            win_after = 100 - win_after_white
                        else: # 白方刚走完
                            is_white_move_finished = True
                            win_before = win_before_white
                            win_after = win_after_white
                            
                        accuracy = move_accuracy_percent(win_before, win_after)
                        judgment = lichess_analyzer.get_judgment(
                            cp_prev, mate_prev, calc_prev, 
                            cp_curr, mate_curr, calc_curr, 
                            is_white_move_finished
                        )
                        judgment_list.append(judgment)
                        
                        player_key = "white" if is_white_move_finished else "black"
                        if judgment in ["Blunder", "Mistake", "Inaccuracy"]:
                            stats[player_key][judgment] += 1
                            
                        if is_white_move_finished:  
                            cp_loss = 0 if score_current > score_before else score_before - score_current
                            game_cp_white += cp_loss
                            game_acc_white.append(accuracy)
                        else:  
                            cp_loss = 0 if score_current < score_before else score_current - score_before
                            game_cp_black += cp_loss
                            game_acc_black.append(accuracy)
                        
                        if board.turn == chess.WHITE:
                            move_number += 1

                    # 3. 汇总数据并写入
                    if analysis_success and (game_acc_white or game_acc_black):
                        # 计算统计值
                        avg_cp_white = game_cp_white / len(game_acc_white) if game_acc_white else None
                        avg_cp_black = game_cp_black / len(game_acc_black) if game_acc_black else None
                        
                        h_white = harmonic_mean(game_acc_white) if game_acc_white else None
                        w_white = volatility_weighted_mean(game_acc_white, game_win_chances, True) if game_acc_white else None
                        final_acc_white = (h_white + w_white) / 2 if h_white is not None and w_white is not None else None
                        
                        h_black = harmonic_mean(game_acc_black) if game_acc_black else None
                        w_black = volatility_weighted_mean(game_acc_black, game_win_chances, False) if game_acc_black else None
                        final_acc_black = (h_black + w_black) / 2 if h_black is not None and w_black is not None else None

                        # 阶段分析 (简化占位，确保写入)
                        # 为了避免代码过长，这里保留基础结构，实际逻辑在内存计算即可
                        stage_analysis = {
                            "beginning": {"white": {"accuracy": None, "std": None}, "black": {"accuracy": None, "std": None}},
                            "middle": {"white": {"accuracy": None, "std": None}, "black": {"accuracy": None, "std": None}},
                            "endgame": {"white": {"accuracy": None, "std": None}, "black": {"accuracy": None, "std": None}}
                        }
                        
                        full_game_data = (
                            global_idx, real_uid, white_player, black_player, avg_cp_white, final_acc_white, 
                            avg_cp_black, final_acc_black, str(stage_analysis),
                            ','.join(evaluation_list),
                            ','.join(raw_eval_list),
                            ','.join(judgment_list),
                            stats["white"]["Blunder"], stats["white"]["Mistake"], stats["white"]["Inaccuracy"],
                            stats["black"]["Blunder"], stats["black"]["Mistake"], stats["black"]["Inaccuracy"],
                            ','.join(e1_list), ','.join(e2_list), ','.join(p_list), ','.join(s_list), 
                            cum_s_white, cum_s_black 
                        )
                        
                        if csv_writer:
                            csv_writer.writerow(list(full_game_data))
                            backup_file_handle.flush()
                            os.fsync(backup_file_handle.fileno())
                    else:
                        # 分析逻辑失败
                        if csv_writer:
                            csv_writer.writerow(list(empty_result))
                            backup_file_handle.flush()
                            os.fsync(backup_file_handle.fileno())

                except Exception as e:
                    # 捕获代码报错 (如 move_number 错误) 并写入空行
                    if is_verbose: print(f"Error in Game {global_idx}: {e}")
                    if csv_writer:
                        csv_writer.writerow(list(empty_result))
                        backup_file_handle.flush() # 强制写入，防止死循环
                        os.fsync(backup_file_handle.fileno())
                
                # 清理内存
                del board, game_acc_white, game_acc_black, e1_list, e2_list
                pbar.update(1)
                
            pbar.close()

    finally:
        if backup_file_handle:
            backup_file_handle.close()
            
    # 返回空列表，完全依赖文件系统
    return [], [], []

# @task(name='Process CSV Chunk',retries=0, log_prints=True)
# ==========================================
# 2. 单个进程的处理逻辑 (替换 process_chunk)
# ==========================================
# ==========================================
# 3. 进程处理函数 (替换旧版本)
# ==========================================
def process_chunk(chunk_df, start_idx, engine_path, threads, depth, is_verbose, temp_csv_path, status_json_path):
    """
    修改后的 process_chunk。
    虽然参数里还有 status_json_path (为了兼容调用不报错)，但在内部不再使用它。
    """
    rank = MPI.COMM_WORLD.Get_rank()
    
    # 确保临时目录存在
    os.makedirs(os.path.dirname(temp_csv_path), exist_ok=True)
    chunk_df.to_csv(temp_csv_path, index=False)
    
    # 设定备份文件路径
    backup_csv_path = temp_csv_path.replace("temp_rank", "backup_results_rank")
    
    # 初始化引擎
    engine = SimpleStockfishEngine(engine_path, threads, multi_pv=2)
    try:
        # 调用核心逻辑
        # 注意：这里不再传递 status_json_path，只传 backup_csv_path
        process_csv_with_offset(temp_csv_path, engine, depth, is_verbose, start_idx,
                                backup_csv_path=backup_csv_path)
        
        print(f"[Rank {rank}] 本地任务完成，结果已保存至 {backup_csv_path}")
        return [], [], [] # 返回空
    
    except Exception as e:
        print(f"[Rank {rank}] 进程分析失败: {e}")
        traceback.print_exc()
        return [], [], []
    finally:
        engine.quit()
        if os.path.exists(temp_csv_path):
            try:
                os.remove(temp_csv_path)
            except PermissionError:
                pass

def extract_white_relative_score(score_obj, board):
    """
    解析分数，返回：(真实CP, 真实Mate, 用于计算胜率的虚拟CP)
    """
    real_cp = None
    real_mate = None
    calc_cp = 0 # 用于计算胜率的分数
    
    # 1. 处理 Mate
    if isinstance(score_obj, chess.engine.Mate):
        mate_val = score_obj.mate()
        if board.turn == chess.BLACK:
            mate_val = -mate_val
        
        real_mate = mate_val
        # 强行给 Mate 赋予一个分数值用于计算胜率
        # 白杀黑(>0) -> 10000分，黑杀白(<0) -> -10000分
        if mate_val > 0:
            calc_cp = 10000 - mate_val # 减去步数，杀得越快分越高
        else:
            calc_cp = -10000 - mate_val
            
    # 2. 处理 CP
    elif isinstance(score_obj, chess.engine.Cp):
        cp_val = score_obj.score()
        if board.turn == chess.BLACK:
            cp_val = -cp_val
        
        real_cp = cp_val
        calc_cp = cp_val
            
    return real_cp, real_mate, calc_cp

# @task(name="Distribute Data")
def distribute_data(csv_file_path: str, status_dir: str) -> Tuple[pd.DataFrame, List[Tuple[pd.DataFrame, int]]]:
    """
    修正版：
    不再在分发阶段过滤数据。
    主进程将原始数据均匀切分，由各子进程在 process_chunk 内部根据自己的 JSON 决定是否跳过。
    这样可以避免索引错位，并确保断点续传的准确性。
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"[Rank {rank}] 读取CSV文件: {csv_file_path}")
        original_df = pd.read_csv(csv_file_path)
        
        # --- 🔥 修改点：这里不要过滤，直接用全量数据切分 ---
        # 之前的 filter_unprocessed_uids 在这里删掉了
        # -----------------------------------------------
        
        total_rows = len(original_df)
        print(f"[Rank {rank}] 原始数据 {total_rows} 行 (将在各进程内部进行断点过滤)")
        
        # 计算切分
        rows_per_process = total_rows // size
        remainder = total_rows % size
        
        # 准备 Rank 0 自己的数据
        start_idx = 0
        end_idx = rows_per_process + (1 if remainder > 0 else 0)
        local_df = original_df.iloc[start_idx:end_idx].copy()
        
        data_chunks = [(local_df, start_idx)]
        
        # 分发给其他进程
        for i in range(1, size):
            start = i * rows_per_process + min(i, remainder)
            end = start + rows_per_process + (1 if i < remainder else 0)
            
            if start < total_rows and start < end:
                work_df = original_df.iloc[start:end].copy()
                data_chunks.append((work_df, start)) 
                comm.send((work_df, start), dest=i, tag=100)
            else:
                data_chunks.append((pd.DataFrame(), -1))
                comm.send((pd.DataFrame(), -1), dest=i, tag=100)
        
        return original_df, data_chunks
    else:
        return None, None


# @task(name="Collect Results")
def collect_results(original_df: pd.DataFrame, all_games_gathered: List, game_details_gathered: List, 
                   game_evaluations_gathered: List) -> Tuple[Dict, pd.DataFrame]:
    """gather all ranks' analysis results
    
    Args:
        original_df
        all_games_gathered
        game_details_gathered
        game_evaluations_gathered
        
    Returns:
        Tuple[Dict, pd.DataFrame]
    """
    rank = MPI.COMM_WORLD.Get_rank()
    
    print(f"[Rank {rank}] 收集到 {len(all_games_gathered)} 个进程的结果")
    for i, games in enumerate(all_games_gathered):
        print(f"[Rank {rank}] 进程 {i} 返回了 {len(games)} 个游戏结果")
        
    combined_games = []
    for sublist in all_games_gathered:
        combined_games.extend(sublist)
    
    combined_details = []
    for sublist in game_details_gathered:
        combined_details.extend(sublist)
        
    combined_evaluations = []
    for sublist in game_evaluations_gathered:
        combined_evaluations.extend(sublist)
    
    print(f"[Rank {rank}] 合并后的游戏总数: {len(combined_games)}")
    
    # 1. 初始化所有列的列表 (长度与原始 DataFrame 一致)
    uid_list = [None] * len(original_df)
    # --- 原有列 ---
    white_cp_loss_list = [None] * len(original_df)
    white_accuracy_list = [None] * len(original_df)
    black_cp_loss_list = [None] * len(original_df)
    black_accuracy_list = [None] * len(original_df)
    evaluation_list = [None] * len(original_df)
    raw_evaluation_list = [None] * len(original_df)

    # --- 阶段分析列 ---
    white_beginning_acc = [None] * len(original_df)
    white_beginning_std = [None] * len(original_df)
    white_middle_acc = [None] * len(original_df)
    white_middle_std = [None] * len(original_df)
    white_endgame_acc = [None] * len(original_df)
    white_endgame_std = [None] * len(original_df)
    
    black_beginning_acc = [None] * len(original_df)
    black_beginning_std = [None] * len(original_df)
    black_middle_acc = [None] * len(original_df)
    black_middle_std = [None] * len(original_df)
    black_endgame_acc = [None] * len(original_df)
    black_endgame_std = [None] * len(original_df)

    e1_full_list = [None] * len(original_df)
    e2_full_list = [None] * len(original_df)
    p_full_list = [None] * len(original_df)
    s_full_list = [None] * len(original_df)
    cum_s_white_list = [None] * len(original_df)
    cum_s_black_list = [None] * len(original_df)
    stage_json_list=[None]*len(original_df)
    
    # --- 新增列：Lichess 判定与统计 ---
    judgments_list = [None] * len(original_df) # 存储判定序列字符串 (如 "Blunder...")
    
    w_blunder_list = [None] * len(original_df)
    w_mistake_list = [None] * len(original_df)
    w_inacc_list = [None] * len(original_df)
    
    b_blunder_list = [None] * len(original_df)
    b_mistake_list = [None] * len(original_df)
    b_inacc_list = [None] * len(original_df)
    # -------------------------------
    
    valid_games = 0
    total_avg_cp_white = total_avg_cp_black = 0.0
    total_acc_white = total_acc_black = 0.0
    
    # 2. 遍历结果并填充列表
    for game in combined_games:
        # 解包元组 (注意：这里的顺序必须严格对应 process_csv_with_offset 中的 append 顺序)
        (row_idx, real_uid, white, black, avg_cp_white, acc_white, avg_cp_black, acc_black, 
         stage_analysis_str, evaluation, raw_eval_str,
         judgment_str,  # <--- 新增：判定序列
         w_blunder, w_mistake, w_inacc, # <--- 新增：白方统计
         b_blunder, b_mistake, b_inacc, # <--- 新增：黑方统计
         e1_str, e2_str, p_str, s_str, cum_s_w, cum_s_b
        ) = game
        
        if row_idx < 0 or row_idx >= len(original_df):
            print(f"[Rank {rank}] 警告: 无效的行索引 {row_idx}，跳过")
            continue
            
        uid_list[row_idx] = real_uid
        # 填充原有数据
        white_cp_loss_list[row_idx] = avg_cp_white
        white_accuracy_list[row_idx] = acc_white
        black_cp_loss_list[row_idx] = avg_cp_black
        black_accuracy_list[row_idx] = acc_black
        evaluation_list[row_idx] = evaluation
        raw_evaluation_list[row_idx] = raw_eval_str
        
        # 填充新增数据 (Lichess 相关)
        judgments_list[row_idx] = judgment_str
        stage_json_list[row_idx] = stage_analysis_str
        
        w_blunder_list[row_idx] = w_blunder
        w_mistake_list[row_idx] = w_mistake
        w_inacc_list[row_idx] = w_inacc
        
        b_blunder_list[row_idx] = b_blunder
        b_mistake_list[row_idx] = b_mistake
        b_inacc_list[row_idx] = b_inacc

        e1_full_list[row_idx] = e1_str
        e2_full_list[row_idx] = e2_str
        p_full_list[row_idx] = p_str
        s_full_list[row_idx] = s_str
        cum_s_white_list[row_idx] = cum_s_w
        cum_s_black_list[row_idx] = cum_s_b
        
        # 填充阶段分析结果
        if stage_analysis_str:
            try:
                sa=ast.literal_eval(stage_analysis_str)
                white_beginning_acc[row_idx] = sa["beginning"]["white"]["accuracy"]
                white_beginning_std[row_idx] = sa["beginning"]["white"]["std"]
                white_middle_acc[row_idx] = sa["middle"]["white"]["accuracy"]
                white_middle_std[row_idx] = sa["middle"]["white"]["std"]
                white_endgame_acc[row_idx] = sa["endgame"]["white"]["accuracy"]
                white_endgame_std[row_idx] = sa["endgame"]["white"]["std"]
            
                black_beginning_acc[row_idx] = sa["beginning"]["black"]["accuracy"]
                black_beginning_std[row_idx] = sa["beginning"]["black"]["std"]
                black_middle_acc[row_idx] = sa["middle"]["black"]["accuracy"]
                black_middle_std[row_idx] = sa["middle"]["black"]["std"]
                black_endgame_acc[row_idx] = sa["endgame"]["black"]["accuracy"]
                black_endgame_std[row_idx] = sa["endgame"]["black"]["std"]
            except:
                pass
        
        # 只统计有效对局 (用于最后的 Summary 打印)
        if avg_cp_white is not None and acc_white is not None:
            valid_games += 1
            total_avg_cp_white += avg_cp_white
            total_avg_cp_black += avg_cp_black
            total_acc_white += acc_white
            total_acc_black += acc_black
    
    # 3. 将列表赋值给 DataFrame 的列
    original_df.insert(0, 'UID', uid_list)
    original_df['White CP Loss'] = white_cp_loss_list
    original_df['White Accuracy'] = white_accuracy_list
    original_df['Black CP Loss'] = black_cp_loss_list
    original_df['Black Accuracy'] = black_accuracy_list
    original_df['Evaluation'] = evaluation_list
    original_df['ComplexityE'] = raw_evaluation_list
    
    # --- 新增列赋值 ---
    original_df['Judgments'] = judgments_list
    
    original_df['White Blunders'] = w_blunder_list
    original_df['White Mistakes'] = w_mistake_list
    original_df['White Inaccuracies'] = w_inacc_list
    
    original_df['Black Blunders'] = b_blunder_list
    original_df['Black Mistakes'] = b_mistake_list
    original_df['Black Inaccuracies'] = b_inacc_list
    # ----------------
    
    original_df['White Beginning Accuracy'] = white_beginning_acc
    original_df['White Beginning Std'] = white_beginning_std
    original_df['White Middle Accuracy'] = white_middle_acc
    original_df['White Middle Std'] = white_middle_std
    original_df['White Endgame Accuracy'] = white_endgame_acc
    original_df['White Endgame Std'] = white_endgame_std
    
    original_df['Black Beginning Accuracy'] = black_beginning_acc
    original_df['Black Beginning Std'] = black_beginning_std
    original_df['Black Middle Accuracy'] = black_middle_acc
    original_df['Black Middle Std'] = black_middle_std
    original_df['Black Endgame Accuracy'] = black_endgame_acc
    original_df['Black Endgame Std'] = black_endgame_std

    original_df['Complexity E1'] = e1_full_list
    original_df['Complexity E2'] = e2_full_list
    original_df['Complexity P'] = p_full_list
    original_df['Complexity S'] = s_full_list
    original_df['White Complexity Sum'] = cum_s_white_list
    original_df['Black Complexity Sum'] = cum_s_black_list
    original_df['Complexity Asymmetry'] = original_df['Black Complexity Sum'] - original_df['White Complexity Sum'] # A = Sb - Sw
    
    # 4. 输出统计信息
    print(f"\n总棋局数: {len(original_df)}")
    print(f"成功分析棋局数: {valid_games}")
    print(f"跳过/分析失败棋局数: {len(original_df) - valid_games}")
    
    summary = {}
    if valid_games > 0:
        print("\n===== 总体统计 =====")
        print(f'有效分析对局数: {valid_games}')
        print(f'白方平均: {total_avg_cp_white/valid_games:.1f}cp损失, {total_acc_white/valid_games:.1f}%准确率')
        print(f'黑方平均: {total_avg_cp_black/valid_games:.1f}cp损失, {total_acc_black/valid_games:.1f}%准确率')
        
        summary = {
            "total_games": valid_games,
            "avg_white_cp_loss": total_avg_cp_white/valid_games,
            "avg_white_accuracy": total_acc_white/valid_games,
            "avg_black_cp_loss": total_avg_cp_black/valid_games,
            "avg_black_accuracy": total_acc_black/valid_games
        }
    
    result_dict = {
        "game_results": combined_details,
        "summary": summary
    }
    
    return result_dict, original_df

# @task(name="Save Results",log_prints=True)
def save_results(mediate_df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """save results into csv file
    
    Args:
        mediate_df
        output_dir
        filename
        
    Returns:
        str
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    mediate_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}")
    return output_path

# @flow(name="Chess Analysis Pipeline",version='1.0')
def analyze_chess_games(
    csv_file_path: str, 
    engine_path: str, 
    threads: int = 1, 
    depth: int = 16, 
    is_verbose: bool = False,
    output_dir: str = "./output",
    output_filename: str = "analysis_results.csv"
) -> Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[str]]:
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f'[Rank {rank}] 进程启动，总进程数 {size}')
    
    # ---------------------------------------------------------
    # 1. 路径配置 (核心优化部分)
    # ---------------------------------------------------------
    # 确保 output_dir 是绝对路径
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '../data/output')
    
    # 推导 temp 和 status 目录 (与 output 平级)
    # 结构: data/output, data/temp, data/status
    data_root = os.path.dirname(output_dir) 
    temp_dir = os.path.join(data_root, 'temp')
    status_dir = os.path.join(data_root, 'status')
    
    # 构建当前 Rank 专属的文件路径
    # 例如: .../data/temp/temp_rank_0.csv
    my_temp_csv_path = os.path.join(temp_dir, f"temp_rank_{rank}.csv")
    # 例如: .../data/status/processed_uids_rank_0.json
    my_status_json_path = os.path.join(status_dir, f"processed_uids_rank_{rank}.json")
    
    # Rank 0 负责创建文件夹
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(status_dir, exist_ok=True)
    
    # 这里的 Barrier 是为了确保文件夹建好了，其他进程再跑
    comm.Barrier()
    
    # ---------------------------------------------------------
    # 2. 数据分发
    # ---------------------------------------------------------
    # 传入 status_dir，让 Rank 0 能找到过滤文件
    original_df, data_chunks = distribute_data(csv_file_path, status_dir)
    
    # ---------------------------------------------------------
    # 3. 处理本地数据
    # ---------------------------------------------------------
    all_games, game_details, game_evaluations = [], [], []
    
    if rank == 0 and data_chunks:
        # 主进程直接拿第一个块
        local_df, start_idx = data_chunks[0]
        if not local_df.empty:
            # 【关键】传入具体的 temp_csv 和 status_json 路径
            all_games, game_details, game_evaluations = process_chunk(
                local_df, start_idx, engine_path, threads, depth, is_verbose, 
                temp_csv_path=my_temp_csv_path, 
                status_json_path=my_status_json_path
            )
            
    elif rank > 0:
        # 子进程接收数据
        received_data = comm.recv(source=0, tag=100)
        local_df, start_idx = received_data
        
        if not local_df.empty:
            # 【关键】传入具体的 temp_csv 和 status_json 路径
            # 注意: 子进程的 is_verbose 通常设为 False 以免刷屏，除非是为了调试
            all_games, game_details, game_evaluations = process_chunk(
                local_df, start_idx, engine_path, threads, depth, False, 
                temp_csv_path=my_temp_csv_path, 
                status_json_path=my_status_json_path
            )
    
    # ---------------------------------------------------------
    # 4. 结果收集
    # ---------------------------------------------------------
    # 确保所有进程都到了这一步
    comm.Barrier()
    
    all_games_gathered = comm.gather(all_games, root=0)
    game_details_gathered = comm.gather(game_details, root=0)
    game_evaluations_gathered = comm.gather(game_evaluations, root=0)
    
    # 主进程处理和保存结果
    if rank == 0 and all_games_gathered:
        result_dict, mediate_df = collect_results(
            original_df, all_games_gathered, game_details_gathered, game_evaluations_gathered
        )
        
        # 保存最终 CSV
        output_path = save_results(mediate_df, output_dir, output_filename)
        
        return result_dict, mediate_df, output_path
    else:
        return None, None, None

if __name__ == "__main__":
    """
    Coordinates the main entry point for distributed analysis of chess games from a CSV file,
    utilizing MPI for parallel processing. The analysis includes engine evaluations,
    accuracy calculations, and phase-based performance metrics.
    """    
    INPUT_CSV =r"C:\Users\Administrator\Desktop\Chess\data\input\100wuid.csv"
    ENGINE_PATH =r"C:\Users\Administrator\Desktop\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    THREADS = 1  
    DEPTH = 16  
    IS_VERBOSE = True
    OUTPUT_FILENAME = "100w_OPTIMIZED.csv"
    FORCE_RESTART=False #强制重跑写true  断点继续写false
    
    config=ProjectConfig(__file__, INPUT_CSV, OUTPUT_FILENAME, FORCE_RESTART)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    config.setup_directories(rank)
    comm.Barrier()

    result_dict, mediate_df, output_path = analyze_chess_games(
        csv_file_path=config.input_csv,
        engine_path=ENGINE_PATH,
        threads=THREADS,
        depth=DEPTH,
        is_verbose=IS_VERBOSE,
        output_dir=config.output_dir, # 传入统一管理的输出目录
        output_filename=OUTPUT_FILENAME
    )

    # 只有主进程会进入这里
    if rank == 0 and output_path:
        print(f"\n" + "="*40)
        print(f"分析流程结束")
        print(f"结果文件: {output_path}")
        print(f"中间文件: {config.temp_dir}")
        print(f"断点状态: {config.status_dir}")
        print("="*40 + "\n")

# mpiexec -n 8 python "C:\Users\Administrator\Desktop\Chess\complexity\try2.py"