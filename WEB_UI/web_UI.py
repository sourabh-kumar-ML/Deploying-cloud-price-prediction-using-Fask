
import numpy as np
from flask import Flask,render_template,request
import pickle
from Model import Predict
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Predict()

app = Flask(__name__)

#index page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    features= [x for x in request.form.values()]
    features.insert(1,"ap-northeast-1a")
    model.intialize(features)
    with model.get_graph().as_default():
        pred1 = model.pred()[0][0]
        
    features= [x for x in request.form.values()]
    features.insert(1,"ap-northeast-1c")
    model.intialize(features)
    with model.get_graph().as_default():
        pred2 = model.pred()[0][0]
    pred = ((pred1,"ap-northeast-1a") if pred1<pred2 else (pred2,"ap-northeast-1c"))
    return render_template('index.html', pred_value='Price of instance is :{:.2f}\n It is lowest in {}'.format(pred[0],pred[1]))



if __name__ == "__main__":
    app.run(debug=True)

'''
features= ['SUSE Linux', 'ap-northeast-1a', '3', '1', '1', '1', '26', 'm4.xlarge']
model.intialize(features)
pred = model.pred()[0][0]
print(pred)
'''
