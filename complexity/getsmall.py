import pandas as pd
df = pd.read_csv(r"C:\Users\Administrator\Desktop\playerstyles\100wuid.csv")
df_head20 = df.head(20)
df_head20.to_csv("output_head20.csv", index=False)