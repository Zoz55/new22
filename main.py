


import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model



def getPrediction(filename):
    
    classes = ['Covid', 'Healthy', 
               'Lung Tumor', 'Common Pneumonia']
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
    
    
    #Load model
    my_model=load_model("chest_model_deploy.h5")
    
    SIZE = 64 #Resize to same size as training images
    img_path = 'static/'+filename
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    img=img.convert('RGB')

    img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    pred = my_model.predict(img) #Predict                    
    
    #Convert prediction to class name
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Diagnosis is:", pred_class)
    return pred_class


#test_prediction =getPrediction('example.jpg')
