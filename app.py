import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', "Model Information"])

with tab1:
    age = st.number_input("Age (years)", min_value=1, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting blood pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting blood Sugar", ["<=120 mg/dl", ">120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", 'ST-T wave abnormality', 'Left Ventricular Hypertrophy'])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

# Mapping categorical to numerical
sex_map = {"Male": 1, "Female": 0}
chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
fasting_bs_map = {"<=120 mg/dl": 0, ">120 mg/dl": 1}
resting_ecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left Ventricular Hypertrophy": 2}
exercise_angina_map = {"No": 0, "Yes": 1}
st_slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

# Convert input
input_data = {
    "Age": age,
    "Sex": sex_map[sex],
    "ChestPainType": chest_pain_map[chest_pain],
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs_map[fasting_bs],
    "RestingECG": resting_ecg_map[resting_ecg],
    "MaxHR": max_hr,
    "ExerciseAngina": exercise_angina_map[exercise_angina],
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope_map[st_slope]
}

input_df = pd.DataFrame([input_data])

# Model info
algorithms = ['Decision Trees', 'Logistic Regression', 'Random Forest', "Support Vector Machine"]
modelnames = ['DecisionTree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl']

# Prediction function
def predict_heart_disease(data_df):
    results = []
    for modelname in modelnames:
        model = pickle.load(open(modelname, 'rb'))
        prediction = model.predict(data_df)
        results.append(prediction)
    return results

# Submit button
if st.button("Submit"):
    st.subheader('Results...')
    st.markdown('--------------------')

    results = predict_heart_disease(input_df)

    for i in range(len(results)):
        st.subheader(algorithms[i])
        if results[i][0] == 0:
            st.write("No heart disease detected.")
        else:
            st.write("Heart disease detected.")
        st.markdown('--------------------')


with tab2:
    st.header("Bulk Prediction from CSV File")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Check required columns
        expected_columns = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                            "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
                            "Oldpeak", "ST_Slope"]

        if not all(col in df.columns for col in expected_columns):
            st.error(f"CSV must contain the following columns: {expected_columns}")
        else:
            # Apply mapping
            df["Sex"] = df["Sex"].map(sex_map)
            df["ChestPainType"] = df["ChestPainType"].map(chest_pain_map)
            df["FastingBS"] = df["FastingBS"].map(fasting_bs_map)
            df["RestingECG"] = df["RestingECG"].map(resting_ecg_map)
            df["ExerciseAngina"] = df["ExerciseAngina"].map(exercise_angina_map)
            df["ST_Slope"] = df["ST_Slope"].map(st_slope_map)

            # Predictions
            bulk_results = df.copy()
            for i, modelname in enumerate(modelnames):
                model = pickle.load(open(modelname, 'rb'))
                bulk_results[algorithms[i]] = model.predict(df)

            st.subheader("Prediction Results")
            st.dataframe(bulk_results)

            # Download link
            csv = bulk_results.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # bytes to string
            href = f'<a href="data:file/csv;base64,{b64}" download="bulk_predictions.csv">Download Results CSV</a>'
            st.markdown(href, unsafe_allow_html=True)


with tab3:
    st.header("Model Information")

    # Model descriptions
    model_descriptions = {
        "Decision Trees": "A non-parametric supervised learning method used for classification and regression. It splits the data into subsets based on the most important features.",
        "Logistic Regression": "A statistical model that uses a logistic function to model a binary dependent variable. Despite the name, it is used for classification tasks.",
        "Random Forest": "An ensemble method that creates multiple decision trees and merges their results for more accurate and stable predictions.",
        "Support Vector Machine": "A supervised learning model that finds the best boundary (hyperplane) that separates classes with the maximum margin."
    }

    for i, modelname in enumerate(modelnames):
        try:
            model = pickle.load(open(modelname, 'rb'))
            st.subheader(algorithms[i])
            st.write(model_descriptions[algorithms[i]])

            # Display model parameters
            st.markdown("**Hyperparameters:**")
            st.json(model.get_params())

        except FileNotFoundError:
            st.error(f"Model file `{modelname}` not found.")
        except Exception as e:
            st.error(f"Could not load `{modelname}`: {e}")



