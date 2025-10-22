from flask import Flask, render_template, request
import pickle
import numpy as np
import json

app = Flask(__name__)

# --- Load Models ---
with open("knn.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("Navie.pkl", "rb") as f:
    nb_model = pickle.load(f)

# --- Load JSON Files ---
with open("knn_Test.json", "r") as f:
    knn_info = json.load(f)

with open("Navie_Test.json", "r") as f:
    nb_info = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        inputs = [
            float(request.form['input1']),
            float(request.form['input2']),
            float(request.form['input3']),
            float(request.form['input4'])
        ]

        model_choice = request.form['model']

        # Convert inputs to array
        final_input = np.array([inputs])

        # Initialize metric variables
        prediction = None
        model_name = ""
        accuracy = None
        conf_matrix = None
        class_report = None

        # Predict based on model selection
        if model_choice == "KNN":
            prediction = int(knn_model.predict(final_input)[0]) # Convert to int for cleaner display
            if prediction == 0:
                prediction = 'setosa'
            elif prediction == 1:
                prediction = 'versicolor'
            else:
                prediction = 'virginica'
            model_name = "K-Nearest Neighbors"
            accuracy = knn_info["Accuracy_Score"]
            conf_matrix = knn_info["Confusion_Matrix"]
            class_report = knn_info["Classification_Report"]

        elif model_choice == "NaiveBayes":
            prediction = int(nb_model.predict(final_input)[0]) # Convert to int for cleaner display
            if prediction == 0:
                predicition = 'setosa'
            elif prediction == 1:
                prediction = 'versicolor'
            else:
                prediction = 'virginica'
            model_name = "Naive Bayes"
            accuracy = nb_info["Accuracy_Score"]
            conf_matrix = nb_info["Confusion_Matrix"]
            class_report = nb_info["Classification_Report"]

        else:
            return render_template('index.html', prediction_text="❌ Invalid model selection.")

        # Pass all results and metrics to the template
        return render_template(
            'index.html',
            prediction_text=f"✅ Model: {model_name} | Predicted Output: {prediction}",
            accuracy_text=f"Test Accuracy: {accuracy:.4f}",
            conf_matrix=conf_matrix,
            class_report=class_report
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)