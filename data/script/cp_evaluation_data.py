import os
import re
import pandas as pd

df=pd.read_csv("../evaluation_data.csv")
df["painting_cp"]=df["painting"].apply(lambda x: re.sub('[(,)]','*',x))
img_path="/ibex/scratch/kimds/Research/P2/data/wikiart/"
target_dir="../evaluation_data/"

for i,row in df.iterrows():
    os.makedirs(target_dir+row['style'],exist_ok=True)
    cmd="cp "+img_path+row['style']+"/"+row['painting_cp']+" "+target_dir+row['style']

    os.system(cmd)

    
    
    



