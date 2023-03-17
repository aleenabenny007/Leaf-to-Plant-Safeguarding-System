from flask import Flask ,render_template, request
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import cv2
import numpy as np
import os

from flask_cors import CORS, cross_origin



names=["Pepper__bell___Bacterial_spot","Pepper__bell___healthy","Potato___Early_blight","Potato___healthy","Potato___Late_blight","Tomato__Target_Spot","Tomato__Tomato_mosaic_virus","Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy","Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites_Two_spotted_spider_mite"]


def processImg(IMG_PATH):
    
    model = load_model("cnn_model.pkl")
    
    
    image = cv2.imread(IMG_PATH)
    print(image)
    image = cv2.resize(image, (30, 30))
    
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    res = model.predict(image)
    label = np.argmax(res)
    print("Label", label)
    labelName = names[label-1]
    print("Label name:", labelName)
    return labelName



app = Flask(__name__,template_folder='template')
cors = CORS(app)

@app.route("/")
def main():
    
    return render_template("interface.html")


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      fname=secure_filename(f.filename);
      f.save("static/"+secure_filename(f.filename))
      
      new_path = os.path.abspath(fname)
      resp = processImg(new_path)

         
   return  render_template("interface.html",value=resp,finame="static/"+fname)
     




if __name__ == "__main__":
    app.run(debug=True)