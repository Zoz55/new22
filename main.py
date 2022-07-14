


import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model



def getPrediction(filename):
    dic = {0 : 'Covid', 1 : 'Healthy', 2 : 'Lung Tumor', 3 : 'Common Pneumonia'}
    
    
    #Load model
    my_model=load_model("chest_model_deploy.h5")
    
    SIZE = 64 #Resize to same size as training images
    img_path = 'static/'+filename
    img= Image.open(img_path).resize((SIZE,SIZE))
    img= img.convert('RGB')
    img= np.asarray(img)

    img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    pred = my_model.predict(img) #Predict                    
    pred_class = dic[[np.argmax(pred)]]
    
    print("Diagnosis is:", pred_class)
    return pred_class

