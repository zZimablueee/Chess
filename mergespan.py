#运行完lifespan.py后运行
import pandas as pd
df=pd.read_csv(r"C:\Users\Administrator\Desktop\playerstyles\lifespans_features_combined.csv")
df = df.rename(columns={"game_id": "uid"})
cols_to_front=["uid","player"]
new_order = cols_to_front + [col for col in df.columns if col not in cols_to_front]
df = df[new_order]
df = df.sort_values(by="uid", ascending=True)
df.to_csv("lifespan_reordered.csv", index=False)
