import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np

model = VGG16(weights='imagenet')
model.save_weights("vgg.ckpt")


##inspect ##
reader=tf.train.load_checkpoint("./model/vgg.ckpt")
#shape_from_key = reader.get_variable_to_shape_map()
#dtype_from_key = reader.get_variable_to_dtype_map()

weights=[]
for i in range(16):
    name_kernel="layer_with_weights-"+str(i)+"/kernel/.ATTRIBUTES/VARIABLE_VALUE"
    name_bias="layer_with_weights-"+str(i)+"/bias/.ATTRIBUTES/VARIABLE_VALUE"
    
    kernel=reader.get_tensor(name_kernel) #need to reshape from (kernel h, kernel w, input, output) => (output, input, kernel h, kernel w)
    print(kernel.shape)
    bias=reader.get_tensor(name_bias)
    if i<=12:
        kernel=np.moveaxis(kernel,[0,1,2,3],[2,3,1,0])
        print(name_kernel,kernel)
    else:
        kernel=np.moveaxis(kernel,[0,1],[1,0])
    
    weights.append(kernel)
    weights.append(bias)


np.savez("tensorflow_weights.npz",weights=weights)
    
