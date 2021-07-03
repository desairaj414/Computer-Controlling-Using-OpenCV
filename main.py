#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from camera0 import VideoCamera0
from camera1 import VideoCamera1
from camera2 import VideoCamera2
from camera3 import VideoCamera3
from camera4 import VideoCamera4
import pickle

# Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('Model_wo_DPF.pkl', 'rb'))

# default page of our web-app
@app.route('/')
def home():
    return render_template('counter.html')

def gen0(camera0):
    while True:
        #get camera frame
        pTime = 0
        frame = camera0.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen1(camera1):
    while True:
        #get camera frame
        pTime = 0
        frame = camera1.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen2(camera2):
    while True:
        #get camera frame
        pTime = 0
        frame = camera2.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen3(camera3):
    while True:
        #get camera frame
        pTime = 0
        frame = camera3.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen4(camera4):
    while True:
        #get camera frame
        pTime = 0
        frame = camera4.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed0')
def video_feed0():
    return Response(gen0(VideoCamera0()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(gen1(VideoCamera1()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(VideoCamera2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(gen3(VideoCamera3()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed4')
def video_feed4():
    return Response(gen4(VideoCamera4()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/HAND')
def HAND():
    '''
    For GOING TO HAND PAGE
    '''
    return render_template('hand.html')

@app.route('/COUNTER')
def COUNTER():
    '''
    For GOING TO COUNTER PAGE
    '''
    return render_template('counter.html')

@app.route('/PAINTER')
def PAINTER():
    '''
    For GOING TO PAINTER PAGE
    '''
    return render_template('painter.html')

@app.route('/MOUSE')
def MOUSE():
    '''
    For GOING TO MOUSE PAGE
    '''
    return render_template('mouse.html')

@app.route('/VOLUME')
def VOLUME():
    '''
    For GOING TO VOLUME PAGE
    '''
    return render_template('volume.html')

@app.route('/HOME')
def HOME():
    '''
    For GOING TO HOME PAGE
    '''
    return render_template('home.html')

@app.route('/BMI')
def BMI():
    '''
    For GONG TO BMI PAGE
    '''
    return render_template('BMI.html')

@app.route('/bmichart', methods=['POST'])
def bmichart():
    '''
    Displays BMI Chart
    '''
    # bmiParam[0]= weight
    bmiParam = [float(x) for x in request.form.values()]
    bmiResult = float(bmiParam[0])/(float(bmiParam[1])**2)

    output = str(round(bmiResult, 2))
    return render_template('bmiChart.html', BMI_cal='Your BMI is : {}'.format(output))

@app.route('/DIABETES')
def DIABETES():
    '''
    For GOING TO TIPS PAGE
    '''
    return render_template('diabetes.html')


@app.route('/FOOD')
def FOOD():
    '''
    For GOING TO FOOD PAGE
    '''
    return render_template('food.html')


@app.route('/MEDICINE')
def MEDICINE():
    '''
    For GOING TO MEDICINE PAGE
    '''
    return render_template('medicine.html')


@app.route('/PREDICTION')
def PREDICTION():
    '''
    For GOING TO PREDICTION PAGE
    '''
    return render_template('prediction.html')

# To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict_proba(final_features)[:, 1]

    output = float(prediction)

    prediction_text = f'Your chances of having diabetes are {output*100}%.\n'

    if (output >= 0.85):
        prediction_text = prediction_text + 'You should consider seeing a doctor.'
    elif(output >= 0.45):
        prediction_text = prediction_text + 'You are likely to develop Diabetes.'
    else:
        prediction_text = prediction_text + 'Congratulations!!!! You are healthy.'

    return render_template('prediction.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
