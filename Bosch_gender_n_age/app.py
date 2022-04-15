
# Python Flask- File upload

# import packages
from flask import Flask, flash
import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import sys
from PIL import Image
import shutil

UPLOAD_FOLDER = 'input/'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG', 'mp4', 'mkv', 'gif'}

application = Flask(__name__)

application.secret_key = "secret key"

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# application = Flask(__name__)

@application.route('/')
def home():
    return render_template('upload.html')

@application.route('/', methods=['POST'])
def upload():
    if request.method == 'POST':
        if not os.path.exists('./input'):
            os.makedirs('./input')
        empty_directory('./input')
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if request.form['submit_button'] == 'Perform Inference':
            if file.filename == '':
                flash('No video/image selected for uploading')
                return redirect(request.url)
            else:
                filename = secure_filename(file.filename)
                file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
                import agengender
                agengender.main()
                return render_template('upload1.html')

    else:
        return render_template('upload.html')

def empty_directory(dir_path):
    shutil.rmtree(dir_path)
    os.makedirs(dir_path)


if __name__ == "__main__":
   
    application.debug = True
    application.run()
    
