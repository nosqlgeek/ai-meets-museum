import math
import json
import urllib
import redis
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, session
import time
import database
import shutil
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import data_required

load_dotenv()  # take environment variables from .env.

IMG_FOLDER = './static/img'
UPLOAD_FOLDER = 'ImgUpload/'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

REDIS_CLIENT = redis.Redis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"),
                           password=os.getenv("REDIS_PASSWORD"))


# form to enter the object data
class ObjectForm(FlaskForm):
    InventarNr = StringField(validators=[data_required()])
    Bezeichnung = StringField(validators=[data_required()])
    Material = StringField(validators=[data_required()])
    TrachslerNr = StringField(validators=[data_required()])
    Beschreibung = TextAreaField()
    submit = SubmitField()


@app.route("/", methods=["GET", "POST"])
def home():
    form = ObjectForm()

    flash_message = None
    flash_time = session.get('flash_time')
    if flash_time and time.time() - flash_time < 3:
        flash_message = "Objekt erfolgreich gespeichert"

    # if request.args.get("filename"):
    #     os.remove(UPLOAD_FOLDER + request.args.get("filename"))
    
    #delete_files_from_folder()
    session.clear()
    return render_template("index.html", form=form, flash_message=flash_message)


@app.route('/upload-image', methods=["GET", "POST"])
def upload_image():
    # check image
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # save uploaded image to UPLOAD_FOLDER
        file_path = UPLOAD_FOLDER + filename
        tensor = database.create_tensor(file_path)
        knn_result = database.search_knn(REDIS_CLIENT, tensor, 'searchIndex')
        neighbours = database.get_neighbours(knn_result, 10)
        # get images for neighbours by image-number and add them to the json
        neighbours_param = []
        for neighbour in neighbours:
            image_nr = neighbour["BildNr"]
            image_filename = f"{image_nr}.jpg"
            neighbour["image_filename"] = image_filename
            neighbour_json = json.dumps(neighbour)
            neighbour = urllib.parse.quote(neighbour_json)
            neighbours_param.append(neighbour)

        nearest_neighbour = neighbours[0]
        nearest_neighbour_json = json.dumps(nearest_neighbour)
        nearest_neighbour = urllib.parse.quote(nearest_neighbour_json)

        return redirect(url_for('fill_form', filename=filename, form=nearest_neighbour,
                                neighbours_param=neighbours_param))
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


# get uploaded image file from directory
@app.route('/uploads/<path:filename>')
def show_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


# search for objects in the redis database
@app.route('/search-objects', methods=["GET", "POST"])
def search_object():
    neighbours_param = request.args.getlist("neighbours_param")
    form_param = request.args.get("form_param")
    filename = request.args.get("filename")
    if request.args.get('search'):
        search_keywords = request.args.get('search')
    else:
        search_keywords = request.form.get('search')

    # Get page-number from the URL (default: 1)
    page = int(request.args.get('page', 1))

    search_count = database.get_full_text_search_count(REDIS_CLIENT, search_keywords=search_keywords,
                                                       index_name='searchIndex')
    page_count = math.ceil(search_count / 10)

    search_data = database.full_text_search(REDIS_CLIENT, search_keywords=search_keywords, index_name='searchIndex',
                                            page=page)

    # get images for search objects by image-number
    for search in search_data:
        image_nr = search["BildNr"]
        image_filename = f"{image_nr}.jpg"
        search["image_filename"] = image_filename

    return render_template("search.html", search_data=search_data, searched_keyword=search_keywords,
                           page_count=page_count, current_page=page, filename=filename, form_param=form_param,
                           neighbours_param=neighbours_param)


# transfer data from a neighbour to the form
@app.route("/transfer-form", methods=["POST", "GET"])
def transfer():
    form = ObjectForm(request.form)
    filename = request.args.get("filename")
    form_data = form.data
    form_json = json.dumps(form_data)
    form_param = urllib.parse.quote(form_json)

    neighbours_param = request.args.getlist("neighbours_param")
    neighbours = []
    for neighbour in neighbours_param:
        neighbour = urllib.parse.unquote(neighbour)
        neighbour = json.loads(neighbour)
        neighbours.append(neighbour)

    return render_template('index.html', form=form, filename=filename, neighbours=neighbours,
                           neighbours_param=neighbours_param, form_param=form_param)


# save the form to the database
@app.route("/save-form", methods=["POST", "GET"])
def save_to_database():
    data = request.form.to_dict()
    del data['submit']
    str_data = json.dumps(data)
    json_data = json.loads(str_data)
    object_nr = database.upload_object_to_redis(REDIS_CLIENT, json_data, object_class='art:')

    # Move image from ImgUpload to ImgStore
    src = f"ImgUpload/{request.args.get('filename')}"
    destination = f'static/ImgStore/{object_nr}.jpg'
    shutil.move(src, destination)

    session['flash_time'] = time.time()
    #delete_files_from_folder()
    return redirect(url_for('home'))


# fill form with the data stored in the session
@app.route("/filled-form", methods=["GET", "POST"])
def fill_form():
    filename = request.args.get("filename")
    neighbours_param = request.args.getlist("neighbours_param")
    neighbours = []
    for neighbour in neighbours_param:
        neighbour = urllib.parse.unquote(neighbour)
        neighbour = json.loads(neighbour)
        neighbours.append(neighbour)

    form_param = request.args.get("form")
    form_json = urllib.parse.unquote(form_param)
    form_data = json.loads(form_json)

    form = ObjectForm(data=form_data)
    return render_template('index.html', form=form, filename=filename, neighbours=neighbours,
                           neighbours_param=neighbours_param, form_param=form_param)


# Methods:
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def delete_files_from_folder():
    if os.listdir('ImgUpload/'):
        for f in os.listdir('ImgUpload/'):
            os.remove(os.path.join('ImgUpload/', f))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)