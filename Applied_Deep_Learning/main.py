# Import necessary ML things-- I may change this to a separate file later on
import uuid
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Import Web app things
from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES

app = Flask(__name__)

model = ResNet50(weights='imagenet')
photos = UploadSet(name="photos", extensions=IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, upload_sets=photos)

@app.route("/", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'], name=uuid.uuid4().hex[:8] + '.')
        return redirect(url_for('show', filename=filename))
    return render_template('upload.html')


@app.route('/photo/<filename>')
def show(filename):
    img_path = app.config['UPLOADED_PHOTOS_DEST'] = 'static/img' + '/' + filename
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img[np.newaxis, ...]
    img = preprocess_input(img)

    y_pred = model.predict(img)
    predictions = decode_predictions(y_pred, top=5)[0]
    url = photos.url(filename)
    return render_template('view_results.html', filename=filename, url=url, predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True, port=8889)