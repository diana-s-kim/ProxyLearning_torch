#remove symbolism and evalution set faizan csv file 
import pandas as pd


df=pd.read_csv("train.csv",header=0)


#rm eval
df_eval=pd.read_csv("../evaluation_data.csv")
eval_painting=[]
for i, row in df_eval.iterrows():
    eval_painting.append(row["style"]+"/"+row["painting"])
df=df[~df.iloc[:,1].isin(eval_painting)]
df.reset_index(inplace=True, drop=True)

#rm symbolism 
df=df[~(df.iloc[:,2]=="Symbolism")]
df.reset_index(inplace=True, drop=True)

#merge
df_diana=pd.DataFrame({"painting":df.iloc[:,2].values+"/"+df.iloc[:,1].values})
df_diana["style"]=df.iloc[:,2].values

path="/ibex/ai/home/kimds/Research/P2/data/wikiart_resize/"
def change_name(x):
    import glob
    if "#" in x:
        idx=x.index("#")        
        x=x.replace(x[idx:idx+13],"*")
        candidate=glob.glob(path+x)
        if len(candidate)>1:
            print("duplicate")
        x=candidate[0]
    return x

df_diana["painting"]=df_diana["painting"].apply(change_name)
df_diana.to_csv("../train.csv",index=False)

print("...complete train")

##val
df=pd.read_csv("val.csv",header=0)

#rm eval
df_eval=pd.read_csv("../evaluation_data.csv")
eval_painting=[]
for i, row in df_eval.iterrows():
    eval_painting.append(row["style"]+"/"+row["painting"])
df=df[~df.iloc[:,1].isin(eval_painting)]
df.reset_index(inplace=True, drop=True)

#rm symbolism
df=df[~(df.iloc[:,2]=="Symbolism")]
df.reset_index(inplace=True, drop=True)

df_diana=pd.DataFrame({"painting":df.iloc[:,2].values+"/"+df.iloc[:,1].values})
df_diana["style"]=df.iloc[:,2].values

df_diana["painting"]=df_diana["painting"].apply(change_name)
df_diana.to_csv("../val.csv",index=False)
print("...complete val")
