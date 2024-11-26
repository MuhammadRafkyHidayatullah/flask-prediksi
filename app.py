import pandas as pd
import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Mengaktifkan CORS untuk mengizinkan semua domain
CORS(app)

# membuat model yag sudah disimpan
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        # ambil data dari request
        data = request.get_json()

        # input unuk memprediksi 
        input_data = pd.DataFrame([{
            "Pregnancies": data['Pregnancies'],
            "Glucose": data['Glucose'],
            "BloodPressure": data['BloodPressure'],
            "SkinThickness": data['SkinThickness'],
            "Insulin": data['Insulin'],
            "BMI": data['BMI'],
            "DiabetesPedigreeFunction": data['DiabetesPedigreeFunction'],
            "Age": data['Age']
        }])

        # melakukan prediksi
        prediction = model.predic(input_data)

          # Mendapatkan probabilitas prediksi
        probabilities = model.predict_proba(input_data)


         # Probabilitas positif dan negatif dalam bentuk persentase
        probability_negative = probabilities[0][0] * 100  # Probabilitas untuk kelas 0 (negatif)
        probability_positive = probabilities[0][1] * 100  # Probabilitas untuk kelas 1 (positif)

        #Prediksi output (0 atau 1, di mana 1 berarti positif diabetes)
        if prediction[0] == 1:
            result = f'Anda memiliki peluang menderita diabetes berdasarkan model KNN kami. Kemungkinan menderita diabetes adalah {probability_positive:.2f}%.'
        else:
            result = 'Hasil prediksi menunjukkan Anda kemungkinan rendah terkena diabetes.'

        # Kembalikan hasil prediksi dan probabilitas dalam bentuk JSON
        return jsonify({
            'prediction': result,
            'probabilities': {
                'negative': f"{probability_negative:.2f}%",  # Format 2 desimal
                'positive': f"{probability_positive:.2f}%"
            }
        })





    except Exception as e:
        return jsonify({'error': str(e)}),400


if __name__ =='main':
    app.run(debug=True)