import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import io
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as tt
from torchvision.utils import save_image
from torchvision.transforms import Compose
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
import PIL
from PIL import Image
import pickle 
from flask import Flask, flash, render_template, request, url_for
from flask_uploads import IMAGES, UploadSet, configure_uploads
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import Flask, make_response
import gdown
from flask_ngrok import run_with_ngrok
from pytorch_pretrained_vit import ViT
from utils import accuracy, TripletLoss, SiameseBase, criterion, pre_trained_model

# Download Model
if not os.path.isfile("./vit_32.sav"):
    url = 'https://drive.google.com/uc?id=1Q6nDOS-RthQdjgjDOQ7XiGS0ppzt-dMA'
    output = 'vit_32.sav'
    gdown.download(url, output, quiet=False)
# Load Model 
filename = './vit_32.sav'
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
model = CPU_Unpickler(open(filename, 'rb')).load()
normalization_values=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

app = Flask(__name__, static_folder='generated')
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "images"
app.config["SECRET_KEY"] = os.urandom(24)
configure_uploads(app, photos)
UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
test_dir ="./"
CORS(app)
# Start ngrok when the app is running
run_with_ngrok(app)


def denorm(img_tensor):
    return img_tensor*0.5 + 0.5


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           

@app.route("/", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' :
        print("this is request.files:")
        print(request.files)
        if 'file' in request.files:

            print("request has photo!!")
            os.system('rm -rf images')
            os.system('rm -rf generated')
            os.system('mkdir images')
            os.system('mkdir generated')

            photo = request.files['file']
            filename = secure_filename(photo.filename)
            photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #photos.save(request.files['photo'])
            
            flash("Photo saved successfully.", "p")
            # img = cv2.imread('images/'+str(request.files['photo'].filename))
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            image = Image.open('./images/'+filename)

            image = image.convert("RGB")
            transform=Compose([ 
                                tt.Resize((384,384)),
                                tt.ToTensor(),
                                tt.Normalize(*normalization_values)])


            image = transform(image)
            image = torch.unsqueeze(image, 0)

            prediction = model(image)[0][0]
            # print(prediction)
            # fname = "/test-embeding"
            # save_image(prediction, test_dir + fname)
            prediction = tuple([pred.item() for pred in prediction])
            # plt.imshow(image) [1,786]
            
            # gen_path_to_save = "generated/"+str(request.files['photo'].filename)
            # orig_path_to_save = "generated/orig"+str(request.files['photo'].filename)
            # plt.imsave(gen_path_to_save, prediction)
            # plt.imsave(orig_path_to_save, denorm(image.squeeze(0)).permute(1,2,0).numpy())
            # flash("Processed Successfully", "p")
            # path_to_save = [orig_path_to_save, gen_path_to_save]

            # return path_to_save[1]
            # response = make_response(prediction, 200)
            # response.mimetype = "text/plain"
            # print(response)
            # return response
            return render_template('upload.html', embedding=prediction)

        else:
            return "'photo' not found in form-data!!"

    # return prediction
    print("Its a GET request!!")
    return render_template('upload.html')

if __name__ == "__main__":
    # app.run(debug=True, use_reloader=True, threaded=True)
    app.run()