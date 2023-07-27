# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image


app = Flask(__name__)


def kpredict(values, dic):
    if len(values) == 18:
        kidney_model = pickle.load(open('kidney.pkl', 'rb'))
        values = np.asarray(values)
        return kidney_model.predict(values.reshape(1, -1))[0]


"Main Front page"


@app.route('/')
def index():
    return render_template("index.html")


"Heart"


@app.route('/heart')
def heart():
    return render_template('heart.html')


@app.route('/hpredict', methods=['GET', 'POST'])
def predict():
    filename = 'heart_KNN-model.pkl'
    heart_model = pickle.load(open(filename, 'rb'))
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        bp = int(request.form['bp'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        ekg = int(request.form['ekg'])
        hr = int(request.form['hr'])
        exang = request.form.get('exang')
        st = float(request.form['st'])
        slope = request.form.get('slope')
        count = int(request.form['count'])
        thal = request.form.get('thal')

        data = np.array(
            [[age, sex, cp, bp, chol, fbs, ekg, hr, exang, st, slope, count, thal]], dtype=np.float64)
        my_prediction = heart_model.predict(data)

        return render_template('heart_result.html', prediction=my_prediction)


"Kidney"


@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    return render_template('kidney.html')


@app.route('/kpredict', methods=['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = kpredict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("kidney.html", message=message)

    return render_template('predict.html', pred=pred)


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/dpredict', methods=['POST'])
def dpredict():
    # Load the Random Forest CLassifier model
    filename = 'ddiabetes_rfc.pkl'
    classifier = pickle.load(open(filename, 'rb'))
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form.get('sex')
        bmi = float(request.form['bmi'])
        scale = int(request.form['scale'])
        bp = request.form.get('bp')
        chol = request.form.get('chol')
        check_chol = request.form.get('check_chol')
        smoker = request.form.get('smoker')
        stroke = request.form.get('stroke')
        chd = request.form.get('chd')
        phy = request.form.get('phy')
        fruit = request.form.get('fruit')
        veg = request.form.get('veg')
        alc = request.form.get('alc')
        plan = request.form.get('plan')
        doc = request.form.get('doc')
        stress = int(request.form['stress'])
        phyh = int(request.form['phyh'])
        walk = request.form.get('walk')

        data = np.array([[age, sex, bmi, scale, bp, chol, check_chol,
                          smoker, stroke, chd, phy, fruit, veg,
                          alc, plan, doc, stress, phyh, walk]], dtype=np.float64)
        my_prediction = classifier.predict(data)

        return render_template('diabetes_result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
