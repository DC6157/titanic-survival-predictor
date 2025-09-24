
import streamlit as st
import pandas as pd
import joblib


# Load the trained model & scaler

model = joblib.load("titanic_model.pkl")   # your trained model
scaler = joblib.load("scaler.pkl")         # scaler used for Fare


# Streamlit UI

st.title("Titanic Survival Prediction 🛳️")
st.markdown("Enter passenger details below:")

# User inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Ticket Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert categorical inputs
gender_no = 1 if gender == "Male" else 0


# Feature engineering

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Gender_no": gender_no,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

# Family size & alone
input_df['Family_size'] = input_df['SibSp'] + input_df['Parch'] + 1
input_df['Is_alone'] = (input_df['Family_size'] == 1).astype(int)

# Age group (use the same bins you used during training)
# If you trained with qcut, you may need the same bin edges
# Safe way to create Age groups
input_df['Age_group_no'] = pd.qcut(input_df['Age'], 5, labels=False, duplicates='drop')


# Scale Fare
input_df['Fare_scaled'] = scaler.transform(input_df[['Fare']])

# Encode Embarked
input_df['Embarked_no'] = input_df['Embarked'].map({'C':0, 'Q':1, 'S':2})

# Select final features in same order as training
final_features = ['Pclass', 'Fare_scaled', 'Gender_no', 'Age_group_no',
                  'Family_size', 'Is_alone', 'Embarked_no']

input_df_ready = input_df[final_features]


# Prediction

if st.button("Predict Survival"):
   input_df_ready = input_df_ready.fillna(0)

   prediction = model.predict(input_df_ready)[0]
   prediction_probability = model.predict_proba(input_df_ready)[0][1]

   if prediction == 1:
        st.success(f"The passenger is likely to survive! 🟢 (Probability: {prediction_probability:.2f})")
   else:
        st.error(f"The passenger is unlikely to survive. 🔴 (Probability: {prediction_probability:.2f})")

