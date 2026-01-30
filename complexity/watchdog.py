import subprocess
import time
import os

# 你的命令
cmd = [
    "mpiexec", 
    "-n", "8", 
    "python", 
    r"C:\Users\Administrator\Desktop\Chess\complexity\trymore.py"
]

print("开始自动监控运行...")

while True:
    try:
        # 启动子进程
        print(f"\n[System] 正在启动分析程序...")
        start_time = time.time()
        
        # 运行命令并等待它结束
        process = subprocess.run(cmd)
        
        # 计算运行了多久
        duration = time.time() - start_time
        print(f"[System] 程序结束/崩溃，本次运行了 {duration:.1f} 秒")
        
        # 如果你想：当程序正常跑完(比如 exit code 0)就停止循环，可以加下面这两行
        if process.returncode == 0:
            print("[System] 程序返回 0 (成功)，任务完成，停止重启。")
            break

    except KeyboardInterrupt:
        print("\n[System] 用户手动停止。")
        break
    except Exception as e:
        print(f"[System] 发生错误: {e}")

    # 休息一会儿再重启
    print("[System] 等待 90 秒后重启...")
    time.sleep(90)

# 电脑终端：  python C:\Users\Administrator\Desktop\Chess\complexity\watchdog.py