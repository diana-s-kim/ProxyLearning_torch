#1. collect all images (except for pointillism)
#2. remove evaluation imagegs
#3. merge styles
#3. divide by train/val/test
import pandas as pd
import os
import numpy as np
import unicodedata

STYLE_MERGE={"Early_Renaissance": "early-renaissance", "High_Renaissance": "high-renaissance", "Mannerism_Late_Renaissance": "mannerism-late-renaissance", "Northern_Renaissance": "northern-renaissance",
"Baroque": "baroque", "Rococo": "rococo", "Romanticism": "romanticism", "Impressionism": "impressionism","Pointillism": "post-impressionism", "Post_Impressionism": "post-impressionism","Contemporary_Realism":"realism",
"Realism": "realism", "New_Realism": "realism", "Art_Nouveau_Modern": "art-nouveau-modern","Analytical_Cubism": "cubism","Cubism":"cubism","Synthetic_Cubism":"cubism","Expressionism":"expressionism","Fauvism":"fauvism","Action_painting": "abstract-expressionism","Abstract_Expressionism":"abstract-expressionism","Color_Field_Painting":"color-field-painting","Minimalism":"minimalism","Naive_Art_Primitivism":"na-ve-art-primitivism",
"Ukiyo_e":"ukiyo-e","Pop_Art":"pop-art"}

SPLIT=(0.85,0.15)

if __name__=='__main__':
    #collect imge
    img_path='/ibex/scratch/kimds/Research/P2/data/wikiart_kaust/' #***#
    filelst=[]
    for style in os.listdir(img_path):
        filelst.extend([style+"/"+img for img in os.listdir(img_path+style)])

    #dataframe
    df=pd.DataFrame({"painting":filelst})
    df["style"]=df.painting.apply(lambda x: x.split("/")[0].strip())

    #remove 120 evaluation data (test)
    df_eval=pd.read_csv("evaluation_data.csv")
    eval_painting=[]
    for i, row in df_eval.iterrows():
        eval_painting.append(row["style"]+"/"+row["painting"])
    df=df[~df["painting"].isin(eval_painting)]
    df.reset_index(inplace=True, drop=True)
    print(df)
    #remove symbolism  
    df=df[~(df["style"]=="Symbolism")]
    df.reset_index(inplace=True, drop=True)
    print(df)
    
    #merged style
    df["style"]=df["style"].apply(lambda x: STYLE_MERGE[x])
    
    #divide train vs test
    np.random.seed(0)
    df["split"]=np.random.rand(len(df))
    def data_split(x):
        if x<=SPLIT[0]:
            return "train"
        else:
            return "test"
    df["split"]=df["split"].apply(data_split)
#    df["painting"]=df["painting"].apply(lambda x: unicodedata.normalize('NFKD', x))
    #save
    df.to_csv("wiki.csv",index=False)
    
