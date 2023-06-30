import time

from flask import Flask, render_template, flash, redirect, request, url_for, session
from werkzeug.utils import secure_filename
import os

from facial_emotion_recognition import EmotionRecognition
import cv2
import ffmpeg
from pathlib import Path

app = Flask(__name__)

app.secret_key = "LhdaAUIL78w57daskjhdkas"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

er = EmotionRecognition(device='cpu', model_file='model/ResNet18_lr_0.01_ep_30_model.pt')

def mark_emotions(video_file, out_dir):
    video_capture = cv2.VideoCapture(video_file)

    if not os.path.exists(f"{out_dir}/images"):
        os.makedirs(f"{out_dir}/images")

    for f in Path(f"{out_dir}/images").glob('*.jpg'):
        try:
            f.unlink()
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    i = 1
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        text = ""
        detect, frame = er.recognise_emotion(frame, return_type='BGR')

        if detect:
            cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            cv2.imwrite(f'{out_dir}/images/frame-{i:03}.jpg', frame)
            i += 1
#            print('Face detected')

            # cv2.imshow('frame', frame)
        # else:
        #     print('no face detected')

        if cv2.waitKey(1) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


    # add timestamp to filename
    res_file = f'{out_dir}/emo_video_{int(time.time())}.mp4'
    ffmpeg.input(f'{out_dir}/images/*.jpg', pattern_type='glob', framerate=25).output(res_file).run()

    return res_file

@app.route("/")
def index():
    return render_template('index.html', stage='initial')


@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)

        source_file_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(source_file_name)

        res_file_name = mark_emotions(source_file_name, app.config['UPLOAD_FOLDER'])

        return render_template('index.html', resulting_file_name=res_file_name)


@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename=filename), code=301)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)