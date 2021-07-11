import numpy as np
import cv2
from PIL import Image
import os
import pickle

from flask_cors import CORS
from flask import Flask, render_template, request, redirect

BASE_DIR= os.path.dirname(os.path.abspath(__file__))
image_dir= os.path.join(BASE_DIR,"images")

face_cascade = cv2.CascadeClassifier('D:/Haar Cascade/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_authetication/trainner.yml")

labels={}
with open("face_authetication/labels.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}


def predict_label(path):
    #print("inside")
    #print(path)
    pil_image=Image.open(path).convert("L")
    #print(pil_image)
    image_array=np.array(pil_image,"uint8")
    #print(image_array)
    faces=face_cascade.detectMultiScale(image_array, scaleFactor=1.5 , minNeighbors=5)
    #print(faces)
    for(x,y,w,h) in faces:
        roi_gray=image_array[y:y+h,x:x+w]
        id_,conf=recognizer.predict(roi_gray)
        #print(id_)
        #print(conf)
        person=labels[id_]
        p=person.split("-")
        person=""
        for i in p:
            person+=i+" "
        
        print(person)
        

app = Flask(__name__)

CORS(app)

@app.route("/face", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        result=""
        print("FORM DATA RECEIVED")
        
        if "file" not in request.files:
            print("Case 1")
            return redirect(request.url)

        file = request.files["file"]
        
        if file:
            file_name="input"
            file.save(file_name)
            p = predict_label(file)

            dictionary={"person":p}
                
      
         
         
if __name__ == "__main__":
    app.run()

