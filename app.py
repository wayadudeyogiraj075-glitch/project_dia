from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model (1).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features in the specific order the model expects
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        
        result_text = "Positive (High Risk)" if prediction[0] == 1 else "Negative (Low Risk)"
        alert_class = "danger" if prediction[0] == 1 else "success"

        return render_template('index.html', 
                               prediction_text=f'Result: {result_text}',
                               alert_class=alert_class)

if __name__ == "__main__":
    app.run(debug=True)
