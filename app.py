









from flask import Flask, render_template, request

import pickle

import numpy as np

from sklearn.ensemble import RandomForestClassifier



app = Flask(__name__)



# Load the trained model (dummy model in this case)

model = pickle.load(open('model/model.pkl', 'rb'))



@app.route('/')

def home():

    return render_template('index.html')



@app.route('/predict', methods=['POST'])

def predict():

    # Collecting inputs from the form

    age = int(request.form['age'])

    bmi = float(request.form['bmi'])

    surgery_type = request.form['surgery_type']

    diabetes = int(request.form['diabetes'])

    prev_surgeries = int(request.form['prev_surgeries'])



    # Map surgery type to numerical value

    surgery_map = {'cardiac': 0, 'neuro': 1, 'ortho': 2, 'general': 3}

    surgery_code = surgery_map[surgery_type]



    # Creating feature array to pass into model

    features = np.array([[age, bmi, surgery_code, diabetes, prev_surgeries]])

    

    # Predict the success probability using the model

    prediction = model.predict_proba(features)[0][1] * 100  # Success probability percentage

    

    # Example of remedies based on the prediction

    if prediction < 85:

        remedies = [

            "Improve nutrition and exercise habits.",

            "Manage chronic conditions before surgery.",

            "Consult specialists."

        ]

    else:

        remedies = [

            "Continue with pre-operative care as planned.",

            "Maintain your current health routine."

        ]

    

    # Render result page with prediction and remedies

    return render_template('result.html', 

                           prediction=round(prediction, 2),

                           remedies=remedies)



if __name__ == '__main__':

    app.run(debug=True)




