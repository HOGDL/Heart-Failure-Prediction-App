Heart Failure Prediction

This project uses machine learning to predict heart failure based on clinical records. It includes data preprocessing, model training (Random Forest, Decision Tree, Logistic Regression, XGBoost, SVM), hyperparameter tuning, and a Gradio web app for interactive predictions.

Dataset
-------
The dataset is from Kaggle: Heart Failure Clinical Data (https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data).
- Download heart_failure_clinical_records_dataset.csv and place it in the project root or /content/ if using Google Colab.
- Features: age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time.
- Target: DEATH_EVENT (0 or 1, indicating heart failure outcome).

Features
--------
- Data loading and exploration using Pandas.
- Model training and evaluation with Scikit-learn and XGBoost.
- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
- Model persistence with Joblib.
- Interactive prediction interface built with Gradio.

Prerequisites
-------------
- Python 3.8+
- Jupyter Notebook or Google Colab
- Git (for cloning the repository)

Installation
------------
1. Clone the repository:
   git clone https://github.com/your-username/heart-failure-prediction.git
   cd heart-failure-prediction
2. Install dependencies:
   pip install -r requirements.txt
3. Download the dataset (heart_failure_clinical_records_dataset.csv) from Kaggle and place it in the project root.
4. (Optional) Run the notebook to train models and generate the saved model (grid_search_rf.best_estimator_.pkl).

Usage
-----
Training the Model:
1. Open Heart_Failure_Prediction.ipynb in Jupyter Notebook or Google Colab.
2. Ensure the dataset (heart_failure_clinical_records_dataset.csv) is in the correct directory.
3. Run all cells to:
   - Load and preprocess the data.
   - Train multiple models (Random Forest, etc.).
   - Perform hyperparameter tuning.
   - Save the best Random Forest model as grid_search_rf.best_estimator_.pkl.

Running the Gradio App:
1. Run the Gradio section in the notebook to launch the web interface.
2. Alternatively, if you extract the Gradio code into app.py, run:
   python app.py
3. Access the app at http://127.0.0.1:7860 (or the public URL provided by Colab).
4. Enter patient details (e.g., age, anaemia, etc.) to get a prediction:
   - "High risk of heart disease" (with probability, if applicable).
   - "Your results are normal" (with probability, if applicable).

Models Evaluated
---------------
- Random Forest (best performer after hyperparameter tuning).
- Decision Tree.
- Logistic Regression.
- XGBoost.
- SVM.
- Metrics: Accuracy, F1-score, Recall, Classification Report.

Files in Repository
-------------------
- Heart_Failure_Prediction.ipynb: Main Jupyter notebook with all code.
- requirements.txt: List of Python dependencies.
- .gitignore: Excludes unnecessary files (e.g., .pkl, .csv if large).
- LICENSE: MIT License for open-source usage.
- (Optional) grid_search_rf.best_estimator_.pkl: Saved model (generate by running the notebook).
- (Optional) heart_failure_clinical_records_dataset.csv: Dataset (download from Kaggle).

Contributing
------------
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch: git checkout -b feature/your-feature
3. Commit your changes: git commit -m "Add your feature"
4. Push to the branch: git push origin feature/your-feature
5. Open a Pull Request.
Please report issues or suggest improvements via GitHub Issues.

License
-------
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
---------------
- Dataset: Kaggle Heart Failure Clinical Data.
- Libraries: Scikit-learn, XGBoost, Gradio, Pandas, NumPy, Matplotlib, Seaborn.
- Built with Python and Jupyter Notebook.