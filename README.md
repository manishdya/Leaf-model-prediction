# 🌸 Leaf Classification Prediction Portal (KNN vs. Naive Bayes)

This project is a **Machine Learning Prediction Portal** designed to classify leaves based on four key measurements such as sepal length, sepal width, petal length, and petal width. The application integrates two different machine learning algorithms — **K-Nearest Neighbors (KNN)** and **Gaussian Naive Bayes** — to demonstrate comparative model performance and prediction accuracy. It serves as an interactive system where users can input new sample data, select a model of their choice, and instantly view the predicted leaf category along with evaluation metrics like the confusion matrix, classification report, and accuracy score.  

The main objective of this project is to showcase the process of building, evaluating, and deploying machine learning models for classification tasks. It emphasizes how different algorithms perform under similar datasets and helps users understand which model generalizes better based on accuracy and precision. The portal is structured to be both simple and informative, providing transparency into the model evaluation process. The inclusion of pre-trained models ensures quick prediction generation without retraining each time, making it efficient for demonstration and testing purposes.

In this project, both models — KNN and Naive Bayes — are trained on a standard dataset (for example, the Iris dataset) and serialized using the `pickle` module. The model evaluation metrics are stored in JSON files so that they can be dynamically displayed in the interface whenever a user performs a prediction. This allows for a consistent comparison of model performance in real time. The application interface is neatly divided into three sections: the left side provides a brief introduction or image banner related to the project, the middle section collects user inputs and displays prediction results, and the right section presents the pre-computed test metrics, including the confusion matrix, classification report, and overall accuracy of the selected model.

The entire system operates with minimal dependencies and is implemented using Python, leveraging core data science libraries such as **NumPy**,**Pandas**, **scikit-learn**. These libraries handle data preprocessing, model training, evaluation, and serialization. The web interface allows interactive access to the models, creating a user-friendly environment for exploring how predictions differ across algorithms. Although a lightweight web framework is used for interactivity, the main focus remains on the machine learning aspect — training models, analyzing performance, and making predictions based on user-provided input features.

The portal is also designed with scalability in mind. Additional models or datasets can easily be integrated in the future for broader experimentation. The system architecture follows a modular approach — separating model training, evaluation, and front-end interaction — making it adaptable for further extensions like incorporating Decision Trees, SVMs, or ensemble methods. By keeping the trained models and their metrics as independent files, the portal ensures flexibility in comparing various algorithms without altering the code significantly.

The typical workflow for using the portal involves loading the trained models, entering feature inputs, choosing between the KNN or Naive Bayes algorithm, and viewing the predicted class along with the associated performance statistics. The confusion matrix provides insights into the distribution of correct and incorrect predictions, while the classification report includes precision, recall, and F1-score for each class. The overall accuracy score offers a simple quantitative comparison between models. Together, these elements form a comprehensive overview of each model’s effectiveness.

This project highlights the end-to-end pipeline of a machine learning application — starting from model training and evaluation to deployment and user interaction. It demonstrates how results can be visualized and interpreted in a clear, accessible format. The design follows clean, structured coding practices and ensures reproducibility by including all essential model and metrics files. Through this project, the intention is to provide both educational insight and a practical demonstration of machine learning model comparison, allowing users to see theory in action through real-time predictive analytics.

---

## 🧩 Technologies and Resources Used

The project is implemented using **Python 3.x** and leverages essential machine learning and data manipulation libraries such as **NumPy** ,**Pandas**, **scikit-learn**. The models are trained offline and saved as serialized files using the **pickle** module for quick loading during predictions. Data visualization for performance metrics, such as confusion matrices and reports, is presented in structured tabular formats to make interpretation straightforward. The user interface is created with basic web technologies, ensuring accessibility and simplicity without diverting focus from the machine learning functionality.

---


## 📁 Project Structure

| File/Folder | Description |
|--------------|-------------|
| **app.py** | Main application logic for running the Leaf Prediction Portal |
| **requirements.txt** | List of dependencies required to run the project |
| **templates/** | Contains HTML templates for the web interface |
| └── **index.html** | Main webpage interface |
| **static/** | Folder for static assets like CSS and images |
| ├── **style.css** | Stylesheet for layout and design |
| └── **collegepic.jpg** | Optional banner or background image |
| **knn.pkl** | Pre-trained KNN model for leaf prediction |
| **Navie.pkl** | Pre-trained Naive Bayes model for leaf prediction |
| **knn_Test.json** | Evaluation metrics for KNN model |
| **Navie_Test.json** | Evaluation metrics for Naive Bayes model |


---

## 🧠 Example of Stored Metrics (JSON)

The following is an example of how model metrics are stored in JSON format. These metrics are displayed dynamically in the portal when a prediction is made:

```json
{
    "Confusion_Matrix": [
        [11, 0, 0],
        [0, 7, 0],
        [0, 1, 11]
    ],
    "Classification_Report": "              precision    recall  f1-score   support\n\n           0       1.00      1.00      1.00        11\n           1       0.88      1.00      0.93         7\n           2       1.00      0.92      0.96        12\n\n    accuracy                           0.97        30\n   macro avg       0.96      0.97      0.96        30\nweighted avg       0.97      0.97      0.97        30\n",
    "Accuracy_Score": 0.9666666666666667
}
⚙️ Installation and Usage
To set up the project locally, ensure Python and pip are installed. Clone the repository, create a virtual environment, and install the dependencies from requirements.txt. Once setup is complete, you can run the main file to start the local server and access the interactive interface through your browser. The portal will allow you to input measurements, select the desired model, and view classification results immediately. The results include not only the predicted class but also complete evaluation details for the selected model, allowing for transparent comparison and analysis.

📈 Future Scope
This project can be further enhanced by incorporating more advanced classification algorithms such as Support Vector Machines, Decision Trees, Random Forests, and Neural Networks. Visual dashboards could be added to display performance charts or allow users to upload CSV data for batch predictions. In the future, the system could also include automated model retraining using new datasets or cloud-based deployment for broader accessibility. These improvements would transform this simple prediction portal into a robust, full-scale machine learning experimentation platform.

