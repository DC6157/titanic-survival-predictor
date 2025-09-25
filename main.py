# main.py
import streamlit as st
import pandas as pd
import joblib
import altair as alt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------
# ✅ Page config (must be first Streamlit command)
# -------------------------------------------------
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# -------------------------------------------------
# Train and Save Model (cached so it runs once only)
# -------------------------------------------------
@st.cache_resource
def train_and_save_model():
    df = pd.read_csv("Titanic-Dataset.csv")

    # Drop rows with missing target
    df = df.dropna(subset=["Survived"])

    # Select features
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    y = df["Survived"]

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Build pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)

    # Save model
    joblib.dump(pipeline, "titanic_model.pkl")
    joblib.dump(list(X.columns), "feature_names.pkl")

    return pipeline, list(X.columns), df

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("titanic_model.pkl")
        feature_names = joblib.load("feature_names.pkl")
        df = pd.read_csv("Titanic-Dataset.csv")
    except:
        model, feature_names, df = train_and_save_model()
    return model, feature_names, df

model, feature_names, df = load_model()

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("⚙️ Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.radio("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Create input DataFrame
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

# One-hot encode to match training
input_encoded = pd.get_dummies(input_data, drop_first=True)
input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

# -------------------------------------------------
# Main Content Layout
# -------------------------------------------------
st.title("🚢 Titanic Survival Predictor")
st.markdown("A machine learning app to predict survival chances on the Titanic.")

tabs = st.tabs(["🔮 Prediction", "📊 Data Insights", "ℹ️ About"])

# -------------------------------------------------
# Prediction Tab
# -------------------------------------------------
with tabs[0]:
    st.subheader("Passenger Prediction")

    if st.button("Predict Survival", use_container_width=True):
        pred = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1]

        if pred == 1:
            st.success(f"✅ The passenger **survives** with probability **{prob:.2%}** 🎉")
        else:
            st.error(f"❌ The passenger **does not survive** with probability **{(1-prob):.2%}** 💔")

    with st.expander("📄 Show Input Data"):
        st.write(input_data)

# -------------------------------------------------
# Data Insights Tab
# -------------------------------------------------
with tabs[1]:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Survival Rate by Class**")
        survival_by_class = df.groupby("Pclass")["Survived"].mean().reset_index()
        chart_class = alt.Chart(survival_by_class).mark_bar(color="steelblue").encode(
            x="Pclass:N", y="Survived:Q"
        )
        st.altair_chart(chart_class, use_container_width=True)

    with col2:
        st.markdown("**Survival Rate by Gender**")
        survival_by_sex = df.groupby("Sex")["Survived"].mean().reset_index()
        chart_sex = alt.Chart(survival_by_sex).mark_bar(color="orange").encode(
            x="Sex:N", y="Survived:Q"
        )
        st.altair_chart(chart_sex, use_container_width=True)

    st.markdown("**Sample of Dataset**")
    st.dataframe(df.head(10))

# -------------------------------------------------
# About Tab
# -------------------------------------------------
with tabs[2]:
    st.subheader("ℹ️ About This App")
    st.markdown("""
    This app uses a **Logistic Regression** model trained on the Titanic dataset.  
    - Inputs are collected in the sidebar.  
    - Data preprocessing includes missing value imputation, scaling, and encoding.  
    - Predictions are shown with probabilities.  
    - Data insights are provided for context.  

    👨‍💻 Built with **Streamlit + scikit-learn**.
    """)
