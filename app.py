import json
import pickle
import tensorflow as tf
from flask import Flask,request,app,jsonify,url_for,render_template
from keras.models import load_model
import numpy as np
import os
import pandas as pd
from test import prediction,download_image
import time
import asyncio
app=Flask(__name__)


@app.route('/' , methods = ['POST'])
async def home():
    print("*********************************")
    data=request.json
    print(data["url"])
    url = data["url"]
    # print(data)
    # url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRNt1OfqHEJWL6sAv6wwNnYS5_-KVE1FRHX0Oju6-IX&s'
    img = await download_image(url)

    print(prediction())
     
    
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json()
#     utl = json.load(data)
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

# # @app.route('/predict',methods=['POST'])
# # def predict():
# #     data=[float(x) for x in request.form.values()]
# #     final_input=scalar.transform(np.array(data).reshape(1,-1))
# #     print(final_input)
# #     output=regmodel.predict(final_input)[0]
# #     return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)