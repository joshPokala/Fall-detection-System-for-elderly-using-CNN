from flask import Flask, Response
from scripts.real_time_detection import generate_frames, fall_detection, start_capture
app = Flask(__name__)

@app.route('/style.css')
def style():
    return Response(open('style.css').read(), mimetype='text/css')
    
@app.route('/')
def index():
    return Response(open('index.html').read(), mimetype='text/html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fall_status')
def fall_status():
    fall_status, prediction = fall_detection()
    return Response("Fall Detected!!, Prediction: " + str(prediction) if fall_status else "No Fall Detected, Prediction: " + str(prediction), mimetype="text/plain")

if __name__ == '__main__':
    start_capture()
    app.run(host='0.0.0.0', port=5000)