#下载对应link的很多文件 链接后加/download-results即可
import os
import pandas as pd
import numpy as np
import time
import requests
from urllib.parse import urlparse
csv_path=r"C:\Users\Administrator\Downloads\tournament_link.csv"
save_folder=r'C:\Users\Administrator\Desktop\playerstyles\namemaps\downloads'
os.makedirs(save_folder,exist_ok=True)

df=pd.read_csv(csv_path)
for i,url in enumerate(df['tournament_url']):
    if not isinstance(url,str) or not url.strip():
        print(f"第{i+1}行是无效的")
        continue
    download_url=url+'/download-results'

    parsed=urlparse(url)
    file_name=os.path.basename(parsed.path)+'.csv'
    save_path=os.path.join(save_folder,file_name)

    try:
        response=requests.get(download_url,timeout=30)
        response.raise_for_status()
        with open(save_path,'wb') as f:
            f.write(response.content)
        print(f"第{i+1}个文件已保存到{save_path}")
    except Exception as e:
        print(f"下载失败{download_url}({e})")
