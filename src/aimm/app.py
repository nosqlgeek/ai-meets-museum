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

# Take environment variables from .env.
load_dotenv()  

IMG_FOLDER = './static/img'
UPLOAD_FOLDER = 'ImgUpload/'
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

REDIS_CLIENT = redis.Redis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"),
                           password=os.getenv("REDIS_PASSWORD"))

"""
 Class form to enter the object data

 Returns:
    The Flask form 
""" 
class ObjectForm(FlaskForm):
    InventarNr = StringField(validators=[data_required()])
    Bezeichnung = StringField(validators=[data_required()])
    Material = StringField(validators=[data_required()])
    TrachslerNr = StringField(validators=[data_required()])
    Beschreibung = TextAreaField()
    submit = SubmitField()

"""
    Shows the homepage with the form and a message after saving
    
    Returns:
        rendering the index.html
"""
@app.route("/", methods=["GET", "POST"])
def home():
    form = ObjectForm()
    # Show a message after saving the form and the image
    flash_message = None
    flash_time = session.get('flash_time')
    if flash_time and time.time() - flash_time < 3:
        flash_message = "Objekt erfolgreich gespeichert"

    delete_files_from_folder()
    session.clear()
    return render_template("index.html", form=form, flash_message=flash_message)

"""
    Upload and check if a new image was uploaded, save the image to the upload folder

    Returns:
        redirect the url to the fill-form route, if the image was successfully uploaded
        redirect the url to the homepage if the upload failed
"""
@app.route('/upload-image', methods=["GET", "POST"])
def upload_image():
    # Check if a image was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        # If no image was selected for uploading, redirect to the homepage
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Save uploaded image to UPLOAD_FOLDER
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = UPLOAD_FOLDER + filename
        # Create Tensor for nearest neighbours search
        tensor = database.create_tensor(file_path)
        # Search for nearest neighbours
        knn_result = database.search_knn(REDIS_CLIENT, tensor, 'searchIndex')
        # Get the 10 nearest neighbours
        neighbours = database.get_neighbours(knn_result, 10)
        # Get images for neighbours by image-number and add them to the json
        neighbours_param = []
        for neighbour in neighbours:
            image_nr = neighbour["BildNr"]
            image_filename = f"{image_nr}.jpg"
            neighbour["image_filename"] = image_filename
            neighbour_json = json.dumps(neighbour)
            neighbour = urllib.parse.quote(neighbour_json)
            neighbours_param.append(neighbour)
        # Split the url to load the neighbours
        nearest_neighbour = neighbours[0]
        nearest_neighbour_json = json.dumps(nearest_neighbour)
        nearest_neighbour = urllib.parse.quote(nearest_neighbour_json)

        return redirect(url_for('fill_form', filename=filename, form=nearest_neighbour,
                                neighbours_param=neighbours_param))
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


"""
    Get the uploaded image file from the directory 
    
    Args:
        filename (str): A string to save the filename of the image
        
    Returns:
        Function: sending a file from the directory
"""
@app.route('/uploads/<path:filename>')
def show_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


"""
    Search for the objects in the redis database with paging by using the full_text_search and get_full_text_search_count methods
    
    Returns:
        The render template search.html to show the results on the website
"""
@app.route('/search-objects', methods=["GET", "POST"])
def search_object():
    # Load the searched objects into the forms
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

    # Fet images for search objects by image-number
    for search in search_data:
        image_nr = search["BildNr"]
        image_filename = f"{image_nr}.jpg"
        search["image_filename"] = image_filename

    return render_template("search.html", search_data=search_data, searched_keyword=search_keywords,
                           page_count=page_count, current_page=page, filename=filename, form_param=form_param,
                           neighbours_param=neighbours_param)


"""
    Transfer one of the nearest neighbours into the form by using a list
    
    Returns:
        The render template index.html to fill the form 
"""
@app.route("/transfer-form", methods=["POST", "GET"])
def transfer():
    # Get the form from one of 10 nearest neighbours
    form = ObjectForm(request.form)
    filename = request.args.get("filename")
    form_data = form.data
    form_json = json.dumps(form_data)
    form_param = urllib.parse.quote(form_json)

    neighbours_param = request.args.getlist("neighbours_param")
    # Split the url to load the neighbours
    neighbours = []
    for neighbour in neighbours_param:
        neighbour = urllib.parse.unquote(neighbour)
        neighbour = json.loads(neighbour)
        neighbours.append(neighbour)

    return render_template('index.html', form=form, filename=filename, neighbours=neighbours,
                           neighbours_param=neighbours_param, form_param=form_param)


"""
    Save the uploaded image with the json to the redis database

    Returns:
        After saving, redirect to the home.html to go to the homepage
"""
@app.route("/save-form", methods=["POST", "GET"])
def save_to_database():
    # Get the image 
    filename = request.args.get('filename')
    data = request.form.to_dict()
    del data['submit']
    data['Zeitstempel'] = time.time()
    # Create a tensor and save it to the database
    data['Tensor'] = database.create_tensor(UPLOAD_FOLDER + filename)[0].tolist()
    str_data = json.dumps(data)
    json_data = json.loads(str_data)
    object_nr = database.upload_object_to_redis(REDIS_CLIENT, json_data, object_class='art:')

    # Move image from ImgUpload to ImgStore
    src = f"ImgUpload/{filename}"
    destination = f'static/ImgStore/{object_nr}.jpg'
    shutil.move(src, destination)

    session['flash_time'] = time.time()
    return redirect(url_for('home'))


"""
    To fill form with the data stored in the last session, after using the search method

    Returns:
        The render template index.html with the filled form datas
"""
@app.route("/filled-form", methods=["GET", "POST"])
def fill_form():
    # Get the filename
    filename = request.args.get("filename")
    # Get the list with the neighbours
    neighbours_param = request.args.getlist("neighbours_param")
    # Split the url to load the neighbours
    neighbours = []
    for neighbour in neighbours_param:
        neighbour = urllib.parse.unquote(neighbour)
        neighbour = json.loads(neighbour)
        neighbours.append(neighbour)

    form_param = request.args.get("form")
    form_json = urllib.parse.unquote(form_param)
    form_data = json.loads(form_json)

    # Load the form with new datas
    form = ObjectForm(data=form_data)
    return render_template('index.html', form=form, filename=filename, neighbours=neighbours,
                           neighbours_param=neighbours_param, form_param=form_param)


"""
    Check if filename has one of the allowed file extensions 
    
    Returns:
        the filename of the uploaded image
        
"""
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


"""
    Method to clear the files in the upload folder
"""
def delete_files_from_folder():
    if os.listdir('ImgUpload/'):
        for f in os.listdir('ImgUpload/'):
            os.remove(os.path.join('ImgUpload/', f))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
