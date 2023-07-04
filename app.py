from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load('decisionTree.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        month = float(request.form['month'])
        day = int(request.form['day'])
        quarter = int(request.form['quarter'])
        origin = request.form['origin']
        if (origin=="ATL"):
            origin = 1
        if (origin=="DTW"):
            origin = 2
        if (origin=="SEA"):
            origin = 3
        if (origin=="MSP"):
            origin = 4
        if (origin=="JFK"):
            origin = 5
        departure = request.form['departure']
        if (departure=="ATL"):
            departure = 1
        if (departure=="DTW"):
            departure = 2
        if (departure=="SEA"):
            departure = 3
        if (departure=="MSP"):
            departure = 4
        if (departure=="JFK"):
            departure = 5


        user_data = pd.DataFrame({
            'month': [month],
            'day': [day],
            'quarter': [quarter],
            'origin': [origin],
            'departure': [departure]
        })

        user_data_encoded = pd.get_dummies(user_data, columns=['quarter', 'day'])

        X_columns = model.classes_[0]
        X = user_data_encoded.reindex(columns=X_columns, fill_value=0)

        prediction = model.predict(X)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
