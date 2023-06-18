generative data: different generative models aa styles                                                                                                             
import pandas as pd
import os
import numpy as np
import unicodedata

if __name__=='__main__':
    #collect imge                                                                                                                                                   
    img_path="/ibex/ai/home/kimds/Research/P1/data/faizan_embedding/features/generated_artworks/"
    filelst=[]
    for style in os.listdir(img_path):
        filelst.extend([style+"/"+img for img in os.listdir(img_path+style)])
    df=pd.DataFrame({"painting":filelst})
    df["style"]=df.painting.apply(lambda x: x.split("/")[0].strip())
    df.to_csv("generative.csv",index=False)
