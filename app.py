# from ast import Import
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('dragon.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = np.round(prediction[0], 12)
    return render_template('index.html', prediction_text='Price should be: $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
        