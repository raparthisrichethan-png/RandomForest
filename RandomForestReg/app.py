import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Medical Insurance Cost Prediction")
st.markdown("---")

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "data" / "insurance.csv"
model_path = BASE_DIR / "models" / "rf_model.pkl"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(data_path)

# -----------------------------
# LOAD MODEL
# -----------------------------
with open(model_path, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Section",
    ["Dataset", "EDA", "Prediction"]
)

# =====================================================
# DATASET SECTION
# =====================================================
if menu == "Dataset":

    st.header("📊 Dataset Overview")

    st.subheader("First 10 Rows")
    st.dataframe(df.head(10))

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Datatype": df.dtypes.astype(str)
    })

    st.dataframe(info_df)

    st.subheader("Missing Values")

    missing = pd.DataFrame(
        df.isnull().sum(),
        columns=["Missing Values"]
    )

    st.dataframe(missing)

# =====================================================
# EDA SECTION
# =====================================================
elif menu == "EDA":

    st.header("📈 Exploratory Data Analysis")

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    # Charges Distribution
    st.subheader("Distribution of Insurance Charges")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["charges"], bins=30)
    ax.set_title("Insurance Charges Distribution")
    ax.set_xlabel("Charges")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # BMI Distribution
    st.subheader("BMI Distribution")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(df["bmi"], bins=20)
    ax2.set_title("BMI Distribution")

    st.pyplot(fig2)

    # Average Charges by Smoker
    st.subheader("Average Charges by Smoker")

    smoker_avg = df.groupby("smoker")["charges"].mean()

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    smoker_avg.plot(kind="bar", ax=ax3)

    ax3.set_ylabel("Average Charges")
    ax3.set_title("Smoker vs Charges")

    st.pyplot(fig3)

    # Correlation Heatmap
    temp_df = df.copy()

    temp_df["sex"] = temp_df["sex"].map({
        "female": 0,
        "male": 1
    })

    temp_df["smoker"] = temp_df["smoker"].map({
        "no": 0,
        "yes": 1
    })

    temp_df["region"] = temp_df["region"].map({
        "northeast": 0,
        "northwest": 1,
        "southeast": 2,
        "southwest": 3
    })

    st.subheader("Correlation Heatmap")

    corr = temp_df.corr()

    fig4, ax4 = plt.subplots(figsize=(8, 6))

    im = ax4.imshow(corr)

    ax4.set_xticks(range(len(corr.columns)))
    ax4.set_yticks(range(len(corr.columns)))

    ax4.set_xticklabels(
        corr.columns,
        rotation=45
    )

    ax4.set_yticklabels(corr.columns)

    plt.colorbar(im)

    st.pyplot(fig4)

    # Feature Importance
    st.subheader("Feature Importance")

    feature_names = [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region"
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(importance_df)

    fig5, ax5 = plt.subplots(figsize=(8, 5))

    ax5.bar(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax5.set_title("Feature Importance")

    st.pyplot(fig5)

# =====================================================
# PREDICTION SECTION
# =====================================================
else:

    st.header("🔮 Predict Insurance Charges")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    children = st.number_input(
        "Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoker = st.selectbox(
        "Smoker",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        [
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ]
    )

    if st.button("Predict Insurance Cost"):

        # Manual Encoding
        sex_encoded = 1 if sex == "male" else 0

        smoker_encoded = 1 if smoker == "yes" else 0

        region_map = {
            "northeast": 0,
            "northwest": 1,
            "southeast": 2,
            "southwest": 3
        }

        region_encoded = region_map[region]

        input_df = pd.DataFrame(
            [[
                age,
                sex_encoded,
                bmi,
                children,
                smoker_encoded,
                region_encoded
            ]],
            columns=[
                "age",
                "sex",
                "bmi",
                "children",
                "smoker",
                "region"
            ]
        )

        prediction = model.predict(input_df)[0]

        st.success(
            f"Estimated Insurance Charges: ${prediction:,.2f}"
        )

        st.balloons()

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(
    "Random Forest Regressor | Medical Insurance Cost Prediction"
)