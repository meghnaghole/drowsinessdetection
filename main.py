
from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
import json

app = Flask(__name__)
vc = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html', ear=vc.get_EAR())

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(vc),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/display_EAR')
def display_EAR():
    return Response(json.dumps({'result':vc.get_EAR()}), mimetype='application/json')

@app.route('/display_MAR')
def display_MAR():
    return Response(json.dumps({'result':vc.get_MAR()}), mimetype='application/json')

if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True)
    app.run(debug=True)
