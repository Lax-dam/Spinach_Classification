import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from keras.utils import load_img,img_to_array
from keras.utils import load_img,img_to_array
from matplotlib import pyplot as plt
import numpy as np
# image_path = "2.jpg"
# new_img = load_img(image_path, target_size=(224, 224))
# img = img_to_array(new_img)
# img = np.expand_dims(img, axis=0)
# img = img/255

from keras.models import load_model
classifier = load_model('MymodelSpinach.h5')
import numpy as np
li=['Amaranthus Green', 'Amaranthus Red', 'Arai Keerai', 'August tree', 'Balloon vine', 'Betel Leaves', 'Black Night Shade', 'Chinese Spinach', 'Coriander Leaves', 'Curry Leaf', 'Dwarf Copperleaf Green', 'Dwarf copperleaf Red', 'False Amarnath', 'Fenugreek Leaves', 'Gongura', 'Indian pennywort', 'Lagos Spinach', 'Lambs Quarters', 'Lettuce Tree', 'Malabar Spinach (Green)', 'Mint Leaves', 'Moringa', 'Mustard', 'Palak', 'Siru Keerai']



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
        
	if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('upload_image filename: ' + filename)
            file = "static/uploads/"+filename
            img = load_img(file, target_size=(224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img/255
            h = classifier.predict(img)
            print(h)
            val1=np.argmax(h)
            print(val1)
            file1=li[val1]
            text='sug/'+file1+'.txt'
            # print('text .. ',text)
            file2 = open(text,"r")
            data=file2.read()
            data1 = data.split('\n')
            # print(data1)
            # print(data1[0])
            # msgval=a[val1],' ',pest
            flash(data1[0])
            flash(data1[1])
            flash(data1[2])
            return render_template('display.html', filename=filename)
    # else:
	# 	flash('Allowed image types are -> png, jpg, jpeg, gif')
	# 	return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()