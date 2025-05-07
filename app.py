from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pickle
import pandas as pd
import warnings
import lightgbm as lgb


warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)

# Elias Alkharma
blood_glucose_model = joblib.load('BloodGlucose_model.pkl')
blood_glucose_scaler = joblib.load('scalerBloodGlucose.pkl')
blood_glucose_selector = joblib.load('selectorBloodGlucose.pkl')


# Nayaz
heart_disease_model = pickle.load(open('heart_disease_pred.sav', 'rb'))

# Nayaz2
Breast_cancer_model = pickle.load(open('Breast_Cancer_pred.sav', 'rb'))


# gharam
Lung_cancer_model = pickle.load(open('LUNG_CANCERfinal (1).sav', 'rb'))


# Load the LightGBM model
Deepression_model = joblib.load('Deepression_model.pkl')
Deepression_scaler = joblib.load('Deepression_scaler.pkl')


@app.route('/')
def index():
    return render_template('index.html')

def validate_input(data):
    required_fields = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    for field in required_fields:
        if field not in data:
            return False, f'Missing field: {field}'
        if not isinstance(data[field], (int, float)):
            return False, f'Invalid type for field: {field}'
    return True, ''


@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        data = request.json
        try:
            input_data = [
                int(data.get('Age', 0)),
                int(data.get('Sex', 0)),
                int(data.get('ChestPainType', 0)),
                int(data.get('RestingBP', 0)),
                int(data.get('Cholesterol', 0)),
                int(data.get('FastingBS', 0)),
                int(data.get('RestingECG', 0)),
                int(data.get('MaxHR', 0)),
                int(data.get('ExerciseAngina', 0)),
                float(data.get('Oldpeak', 0.0)),
                int(data.get('ST_Slope', 0))
            ]

            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            prediction = heart_disease_model.predict(input_data_reshaped)

            if prediction[0] == 0:
                diagnosis = 'Low Risk Detected'
            else:
                diagnosis = 'High Risk Detected'

            return jsonify({'diagnosis': diagnosis})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('page1.html')


@app.route('/page2', methods=['GET', 'POST'])
def page2():
    if request.method == 'POST':
        data = request.json
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        try:
            input_features = np.array([[
                data['Pregnancies'],
                data['Glucose'],
                data['BloodPressure'],
                data['SkinThickness'],
                data['Insulin'],
                data['BMI'],
                data['DiabetesPedigreeFunction'],
                data['Age']
            ]])
            scaled_features = blood_glucose_scaler.transform(input_features)
            selected_features = blood_glucose_selector.transform(scaled_features)
            blood_glucose_prediction = blood_glucose_model.predict(selected_features)[0]

            return jsonify({
                'blood_glucose_prediction': int(blood_glucose_prediction)
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('page2.html')



#nayaz2
@app.route('/page3', methods=['GET', 'POST'])
def page3():
    if request.method == 'POST':
        data = request.json
        try:
            # استخراج الفيتشرات الجديدة والتحقق من صحتها
            input_data = [
                float(data.get('mean_radius', 0.0)),
                float(data.get('mean_texture', 0.0)),
                float(data.get('mean_perimeter', 0.0)),
                float(data.get('mean_area', 0.0)),
                float(data.get('mean_smoothness', 0.0))
            ]

            # تحويل البيانات إلى مصفوفة numpy وتغيير شكلها لتناسب المدخلات النموذجية
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            # التنبؤ باستخدام النموذج
            prediction = Breast_cancer_model.predict(input_data_reshaped)

            # تحديد التشخيص بناءً على التنبؤ
            if prediction[0] == 0:
                diagnosis = 'Low Risk Detected'
            else:
                diagnosis = 'High Risk Detected'

            # إرجاع استجابة JSON بالتشخيص
            return jsonify({'diagnosis': diagnosis})

        except ValueError as ve:
            return jsonify({'error': f'Invalid input data: {str(ve)}'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # عرض قالب page1.html عند الطلب GET
    return render_template('page3.html')

@app.route('/page4', methods=['GET', 'POST'])
def page4():
    global resul
    if request.method == 'POST':
        try:
            data = request.json
            input_features = np.array([[
                data.get('GENDER', 0),
                data.get('AGE', 0),
                data.get('SMOKING', 0),
                data.get('YELLOW_FINGERS', 0),
                data.get('ANXIETY', 0),
                data.get('PEER_PRESSURE', 0),
                data.get('CHRONIC_DISEASE', 0),
                data.get('FATIGUE', 0),
                data.get('ALLERGY', 0),
                data.get('WHEEZING', 0),
                data.get('ALCOHOL_CONSUMING', 0),
                data.get('COUGHING', 0),
                data.get('SHORTNESS_OF_BREATH', 0),
                data.get('SWALLOWING_DIFFICULTY', 0),
                data.get('CHEST_PAIN', 0)
            ]])

            input_data_reshaped = input_features.reshape(1, -1)
            lung_cancer_prediction = Lung_cancer_model.predict(input_data_reshaped)[0]
            if lung_cancer_prediction == 1:
                resul = 'High Risk Detected'
            elif lung_cancer_prediction == 0:
                resul = 'Low Risk Detected'
            print(input_features)
            return jsonify({'lung_cancer_prediction': resul})

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('page4.html')
















if __name__ == '__main__':
    app.run(debug=True)
