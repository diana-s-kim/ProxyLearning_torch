#create MAT
import numpy as np
import pandas as pd
import sys
sys.path.append("/ibex/scratch/kimds/Research/P1/ProxyLearning/ProxyLearning")
import style_attribute
org_styles=['abstract-expressionism','art-nouveau-modern','color-field-painting','early-renaissance','na-ve-art-primitivism','northern-renaissance','realism','romanticism','baroque','cubism','expressionism','fauvism','high-renaissance','impressionism','mannerism-late-renaissance','minimalism','pop-art','post-impressionism','rococo','ukiyo-e']
#attributes=['non-representational', 'representational', 'repetition', 'rhythm', 'pattern', 'contrast', 'symmetry', 'proportion', 'active', 'straight', 'curved', 'meandering', 'horizontal', 'vertical', 'diagonal', 'energetic', 'gestural', 'geometric', 'abstract', 'amorphous', 'biomorphic', 'decorative', 'planar', 'chromatic', 'closed', 'open', 'smooth', 'rough', 'atmospheric', 'monochromatic', 'warm', 'cool', 'transparent', 'overlapping', 'perspective', 'parallel', 'blurred', 'broken', 'controlled', 'thick', 'thin', 'bumpy', 'flat', 'bright', 'calm', 'muted', 'distorted', 'heavy', 'light', 'linear', 'organic', 'kinetic', 'dark', 'ambiguous', 'balance', 'harmony', 'unity', 'variety']
styles=style_attribute.STYLES
attributes=style_attribute.ATTRIBUTES
with open("./GT/groundtruth.txt","r") as f:
    GTs=f.readlines()
f.close()

gt_=np.empty((0,58))
for line in GTs:
   gt_=np.append(gt_,[np.array([float(val.strip()) for val in line.split(":")[1].strip().split(",")])],axis=0)

gt_matrix_dict=dict()#np.empty((0,58))

for i in range(20):
    #gt_matrix=np.append(gt_matrix,[gt_[i*3:i*3+3,:].mean(axis=0)],axis=0)
    gt_matrix_dict[org_styles[i]]=gt_[i*3:i*3+3,:].mean(axis=0)

print(gt_matrix_dict)
gt_matrix=np.empty((0,58))
#change the order according to the new label
for style in styles:
    gt_matrix=np.append(gt_matrix,[gt_matrix_dict[style]],axis=0)
for i in range (20):
    print(gt_matrix[i,:])
    print(gt_matrix_dict[styles[i]])

#save as csv file
df=pd.concat([pd.DataFrame(pd.Series(styles),columns=['style']),pd.DataFrame(gt_matrix,columns=attributes)],axis=1)
df.to_csv("G.csv")





