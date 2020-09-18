from flask import Flask, render_template, request, redirect, url_for, jsonify
from gtts import gTTS
import os
import vlc
import time
import sda
import base64


app = Flask(__name__)
check_video = False

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/show_video")
def show_video():
    return render_template('show_video.html')

@app.route('/background_process_test')
def background_process_test():
    textVal = request.args.get('textVal')
    image = request.args.get('image')
    print(textVal)
    # print(image)
    base64_image_str = image[image.find(",")+1:]
    # print(base64_image_str)
    imgdata = base64.b64decode(base64_image_str)
    filename = 'input/image.bmp'  
    with open(filename, 'wb') as f:
        f.write(imgdata)
    text2speech(textVal)
    # code1 = 'function nextPage(){ window.location.href = "/show_video"; }'
    # nextPage = js2py.eval_js(code1)
    # nextPage()
    # return "nothing"
    # redirect(url_for('show_video'), code=302)
    print("done")
    return jsonify({'data': 'success'})
    # return (''), 204

def text2speech(myText):
    output = gTTS(text=myText, lang='en-us', slow=False)
    output.save("input/audio.mp3")
    p = vlc.MediaPlayer("input/audio.mp3")
    p.play()
    time.sleep(1.5)
    duration = p.get_length() / 1000
    time.sleep(duration)
    initiate_video()

def initiate_video():
    va = sda.VideoAnimator(model_path="grid")
    vid, aud = va("input/image.bmp", "input/audio.mp3")
    va.save_video(vid, aud, "static/generated.mp4")
    # ffmpeg -i generated.mp4 -vcodec h264 generated1.mp4
    print("Video done")


if __name__ == '__main__':
    app.run()