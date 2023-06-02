import math
import json
import redis
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, session
import time
import Util
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import data_required

load_dotenv()  # take environment variables from .env.

# Test123

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
    session.clear()
    form = ObjectForm()
    return render_template("index.html", form=form)


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
        session['uploaded_filename'] = filename
        file_path = UPLOAD_FOLDER + filename
        tensor = Util.createTensor(file_path)
        knn_result = Util.searchKNN(tensor, 'vectorIndex', REDIS_CLIENT)
        neighbours = Util.getNeighbours(knn_result, 10)
        # get images for neighbours by inventory-number and add them to the json
        for neighbour in neighbours:
            inv_nr = neighbour["InventarNr"]
            image_filename = f"{inv_nr}.jpg"
            neighbour["image_filename"] = image_filename
        session["neighbours"] = neighbours

        nearest_neighbour = neighbours[0]
        session["nearest_neighbour"] = nearest_neighbour

        session["inv_nr"] = nearest_neighbour["InventarNr"]
        session["bezeichnung"] = nearest_neighbour["Bezeichnung"]
        session["material"] = nearest_neighbour["Material"]
        session["trachsler"] = nearest_neighbour["TrachslerNr"]
        session["beschreibung"] = nearest_neighbour["Beschreibung"]
        return redirect(url_for('fill_form'))  # fill form with session data
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
    if request.args.get('search'):
        searchKeywords = request.args.get('search')
    # check if search-form is empty
    elif request.form.get('search') is None and session.get('neighbours'):
        return redirect(url_for('fill_form'))
    elif request.form.get('search') is None:
        return redirect(url_for('home'))
    else:
        searchKeywords = request.form.get('search')

    # Seitennummer aus der URL abrufen (Standardwert: 1)
    page = int(request.args.get('page', 1))

    session["searchKeywords"] = searchKeywords
    search_count = Util.getFullTextSearchCount(searchKeywords=searchKeywords, index_name='searchIndex')
    page_count = math.ceil(search_count / 10)

    search_data = Util.fullTextSearch(searchKeywords=searchKeywords, index_name='searchIndex', page=page)

    # get images for search objects by inventory-number
    for search in search_data:
        inv_nr = search.get("InventarNr")
        image_filename = f"{inv_nr}.jpg"
        search["image_filename"] = image_filename

    # remember the last opened url
    referer = request.headers.get('Referer')
    return render_template("search.html", search_data=search_data, referer=referer, searched_keyword=searchKeywords,
                           page_count=page_count, current_page=page)


# transfer data from a neighbour to the form
@app.route("/transfer-form", methods=["POST", "GET"])
def transfer():
    form = ObjectForm(
        InventarNr=request.form.get("InventarNr"),
        Bezeichnung=request.form.get("Bezeichnung"),
        Material=request.form.get("Material"),
        TrachslerNr=request.form.get("TrachslerNr"),
        Beschreibung=request.form.get("Beschreibung")
    )
    filename = session.get('uploaded_filename')
    neighbours = session.get('neighbours')
    return render_template('index.html', form=form, filename=filename, neighbours=neighbours)


# save the form to the database
@app.route("/save-form", methods=["POST", "GET"])
def save_to_database():
    data = request.form.to_dict()
    del data['submit']
    json_data = json.dumps(data)
    print(json_data)
    # REDIS_CLIENT.save(json_data)

    # Move image from ImgUpload to ImgStore
    src = f"ImgUpload/{session.get('uploaded_filename')}"
    dest = f'static/ImgStore/{data["InventarNr"]}.jpg'
    os.replace(src, dest)

    flash("Objekt erfolgreich gespeichert", "success")
    session['flash_time'] = time.time()
    return redirect(url_for('home'))


# fill form with the data stored in the session
@app.route("/filled-form", methods=["GET", "POST"])
def fill_form():
    filename = session.get('uploaded_filename')
    neighbours = session.get('neighbours')
    neighbours_images = session.get('neighbours_images')
    form = ObjectForm(
        InventarNr=session.get("inv_nr"),
        Bezeichnung=session.get("bezeichnung"),
        Material=session.get("material"),
        TrachslerNr=session.get("trachsler"),
        Beschreibung=session.get("beschreibung")
    )
    return render_template('index.html', form=form, filename=filename, neighbours=neighbours, neighbours_images=neighbours_images)


# Methods:
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)
