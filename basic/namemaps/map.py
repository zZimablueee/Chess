#三个映射关系
import pandas as pd
import itertools
file_path=r"C:\Users\Administrator\Desktop\playerstyles\namemaps\downloads\all.csv"
df=pd.read_csv(file_path)

#Name有时候会是空的，此时让Username=Name
df['Name']=df['Name'].fillna(df['Username'])

#映射1：给出Name，有对应的所有Username
name_to_usernames={}
for _,row in df.iterrows():
    name=row['Name']
    username=row['Username']
    if name not in name_to_usernames:
        name_to_usernames[name]={
            'Usernames':set(),
            'Fed':row['Fed'],
            'Title':row['Title']
        }
    name_to_usernames[name]['Usernames'].add(username)
#映射2：给出Username，有对应的Name
username_to_name={}
for _,row in df.iterrows():
    username=row['Username']
    name=row['Name']
    username_to_name[username]=name

for idx, (name, data) in enumerate(itertools.islice(name_to_usernames.items(), 5)):
    print(f"Name:{name}")
    print(f"Usernames:{data['Usernames']}")
    print(f"Fed:{data['Fed']}, Title:{data['Title']}")
    print("-" * 5)
for idx,(username,data) in enumerate(itertools.islice(username_to_name.items(),5)):
    print(f"Username:{username}  Name:{name}")
    print("-"*5)

name_mapping_df=pd.DataFrame.from_dict(name_to_usernames, orient='index')
name_mapping_df.to_csv("name_to_usernames.csv",index_label="Name")
username_mapping_df=pd.DataFrame(list(username_to_name.items()), columns=["Username", "Name"])
username_mapping_df.to_csv("username_to_name.csv", index=False)