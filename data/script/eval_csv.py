import pandas as pd
import unicodedata
with open("../eval_painting_lst.txt") as f:
    eval_imgs=f.readlines()
f.close()
print(eval_imgs)

style=[img.split(" ")[1].split("/")[1].strip() for img in eval_imgs]
style=map(lambda x: unicodedata.normalize('NFC', x),style)
painting=[img.split(" ")[1].split("/")[2].strip() for img in eval_imgs]
painting=map(lambda x: unicodedata.normalize('NFC', x),painting)
df=pd.DataFrame({"style":style,"painting":painting})
df.to_csv("../evaluation_data.csv",index=False)
