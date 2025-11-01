#把经过stockfish得到evaluation值后的一堆小csv文件合并起来
import os
import pandas as pd

folder_path = r'C:\Users\Administrator\Desktop\titled-tuesday\basicinfo'  # 替换为你想要合并的有一堆CSV文件夹路径
output_file = r'C:\Users\Administrator\Desktop\titled-tuesday\basicgames.csv'  # 输出合并后的文件，合并后的文件在放该py文件的文件夹里

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

df_list = []
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    if os.path.getsize(file_path) > 0:  # 确保文件不是空的
        df_list.append(pd.read_csv(file_path))
    else:
        print(f"警告：{file} 是空文件，已跳过。")

if df_list:  # 确保至少有一个有效的 CSV 文件
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"合并完成，共合并 {len(df_list)} 个 CSV 文件，已保存为 {output_file}")
else:
    print("错误：所有 CSV 文件都是空的，未生成合并文件。")
