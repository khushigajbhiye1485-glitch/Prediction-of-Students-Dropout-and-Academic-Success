
import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Outcome Classification",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = os.path.join("models", "student_outcome_pipeline.joblib")
DATA_PATH = "dataset.csv"
CONFIG_PATH = os.path.join("models", "feature_config.json")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_feature_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Human-readable labels for the encoded dataset
#
# The uploaded dataset uses compact integer labels for several original
# UCI categorical codes. The model was trained on these compact labels,
# so we KEEP the numeric values internally and only show descriptions
# in the Streamlit interface.
# ---------------------------------------------------------------------

def rank_map(original_codes, labels):
    """Map encoded 1..N values to labels in ascending original-code order."""
    pairs = sorted(zip(original_codes, labels), key=lambda x: x[0])
    return {i + 1: label for i, (_, label) in enumerate(pairs)}


LABELS = {
    "Marital status": {
        1: "Single",
        2: "Married",
        3: "Widower",
        4: "Divorced",
        5: "Facto union",
        6: "Legally separated",
    },

    "Application mode": rank_map(
        [1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57],
        [
            "1st phase - general contingent",
            "Ordinance No. 612/93",
            "1st phase - special contingent (Azores Island)",
            "Holders of other higher courses",
            "Ordinance No. 854-B/99",
            "International student (bachelor)",
            "1st phase - special contingent (Madeira Island)",
            "2nd phase - general contingent",
            "3rd phase - general contingent",
            "Ordinance No. 533-A/99 - item b2 (Different Plan)",
            "Ordinance No. 533-A/99 - item b3 (Other Institution)",
            "Over 23 years old",
            "Transfer",
            "Change of course",
            "Technological specialization diploma holders",
            "Change of institution/course",
            "Short cycle diploma holders",
            "Change of institution/course (International)",
        ],
    ),

    "Course": rank_map(
        [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254,
         9500, 9556, 9670, 9773, 9853, 9991],
        [
            "Biofuel Production Technologies",
            "Animation and Multimedia Design",
            "Social Service (evening attendance)",
            "Agronomy",
            "Communication Design",
            "Veterinary Nursing",
            "Informatics Engineering",
            "Equinculture",
            "Management",
            "Social Service",
            "Tourism",
            "Nursing",
            "Oral Hygiene",
            "Advertising and Marketing Management",
            "Journalism and Communication",
            "Basic Education",
            "Management (evening attendance)",
        ],
    ),

    "Daytime/evening attendance": {
        0: "Evening",
        1: "Daytime",
    },

    "Previous qualification": rank_map(
        [1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43],
        [
            "Secondary education",
            "Higher education - bachelor's degree",
            "Higher education - degree",
            "Higher education - master's",
            "Higher education - doctorate",
            "Frequency of higher education",
            "12th year of schooling - not completed",
            "11th year of schooling - not completed",
            "Other - 11th year of schooling",
            "10th year of schooling",
            "10th year of schooling - not completed",
            "Basic education 3rd cycle (9th/10th/11th year) or equivalent",
            "Basic education 2nd cycle (6th/7th/8th year) or equivalent",
            "Technological specialization course",
            "Higher education - degree (1st cycle)",
            "Professional higher technical course",
            "Higher education - master (2nd cycle)",
        ],
    ),

    "Nacionality": rank_map(
        [1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62,
         100, 101, 103, 105, 108, 109],
        [
            "Portuguese",
            "German",
            "Spanish",
            "Italian",
            "Dutch",
            "English",
            "Lithuanian",
            "Angolan",
            "Cape Verdean",
            "Guinean",
            "Mozambican",
            "Santomean",
            "Turkish",
            "Brazilian",
            "Romanian",
            "Moldova (Republic of)",
            "Mexican",
            "Ukrainian",
            "Russian",
            "Cuban",
            "Colombian",
        ],
    ),

    "Mother's qualification": rank_map(
        [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27,
         29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
        [
            "Secondary Education - 12th Year or equivalent",
            "Higher Education - Bachelor's Degree",
            "Higher Education - Degree",
            "Higher Education - Master's",
            "Higher Education - Doctorate",
            "Frequency of Higher Education",
            "12th Year - Not Completed",
            "11th Year - Not Completed",
            "7th Year (Old)",
            "Other - 11th Year",
            "10th Year",
            "General commerce course",
            "Basic Education 3rd Cycle (9th/10th/11th Year) or equivalent",
            "Technical-professional course",
            "7th year of schooling",
            "2nd cycle of the general high school course",
            "9th Year - Not Completed",
            "8th year of schooling",
            "Unknown",
            "Can't read or write",
            "Can read without having a 4th year of schooling",
            "Basic education 1st cycle (4th/5th year) or equivalent",
            "Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent",
            "Technological specialization course",
            "Higher education - degree (1st cycle)",
            "Specialized higher studies course",
            "Professional higher technical course",
            "Higher Education - Master (2nd cycle)",
            "Higher Education - Doctorate (3rd cycle)",
        ],
    ),

    "Father's qualification": rank_map(
        [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22,
         25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41,
         42, 43, 44],
        [
            "Secondary Education - 12th Year or equivalent",
            "Higher Education - Bachelor's Degree",
            "Higher Education - Degree",
            "Higher Education - Master's",
            "Higher Education - Doctorate",
            "Frequency of Higher Education",
            "12th Year - Not Completed",
            "11th Year - Not Completed",
            "7th Year (Old)",
            "Other - 11th Year",
            "2nd year complementary high school course",
            "10th Year",
            "General commerce course",
            "Basic Education 3rd Cycle (9th/10th/11th Year) or equivalent",
            "Complementary High School Course",
            "Technical-professional course",
            "Complementary High School Course - not concluded",
            "7th year of schooling",
            "2nd cycle of the general high school course",
            "9th Year - Not Completed",
            "8th year of schooling",
            "General Course of Administration and Commerce",
            "Supplementary Accounting and Administration",
            "Unknown",
            "Can't read or write",
            "Can read without having a 4th year of schooling",
            "Basic education 1st cycle (4th/5th year) or equivalent",
            "Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent",
            "Technological specialization course",
            "Higher education - degree (1st cycle)",
            "Specialized higher studies course",
            "Professional higher technical course",
            "Higher Education - Master (2nd cycle)",
            "Higher Education - Doctorate (3rd cycle)",
        ],
    ),

    "Mother's occupation": rank_map(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125,
         131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175,
         191, 192, 193, 194],
        [
            "Student",
            "Legislative/executive representatives, directors and managers",
            "Specialists in intellectual and scientific activities",
            "Intermediate level technicians and professions",
            "Administrative staff",
            "Personal services, security/safety workers and sellers",
            "Farmers and skilled workers in agriculture, fisheries and forestry",
            "Skilled workers in industry, construction and crafts",
            "Installation and machine operators and assembly workers",
            "Unskilled workers",
            "Armed forces professions",
            "Other situation",
            "Blank / not specified",
            "Health professionals",
            "Teachers",
            "ICT specialists",
            "Intermediate science and engineering technicians",
            "Intermediate-level health technicians",
            "Intermediate legal/social/sports/cultural technicians",
            "Office workers/secretaries/data processing operators",
            "Data/accounting/statistical/financial/registry operators",
            "Other administrative support staff",
            "Personal service workers",
            "Sellers",
            "Personal care workers",
            "Skilled construction workers except electricians",
            "Skilled printing/precision/jewellery/artisan workers",
            "Food processing/woodworking/clothing/other industry craft workers",
            "Cleaning workers",
            "Unskilled agriculture/animal/fisheries/forestry workers",
            "Unskilled extractive/construction/manufacturing/transport workers",
            "Meal preparation assistants",
        ],
    ),

    "Father's occupation": rank_map(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103,
         112, 114, 121, 122, 123, 124, 131, 132, 134, 135, 141, 143,
         144, 151, 152, 153, 154, 161, 163, 171, 172, 174, 175, 181,
         182, 183, 192, 193, 194, 195],
        [
            "Student",
            "Legislative/executive representatives, directors and managers",
            "Specialists in intellectual and scientific activities",
            "Intermediate level technicians and professions",
            "Administrative staff",
            "Personal services, security/safety workers and sellers",
            "Farmers and skilled workers in agriculture, fisheries and forestry",
            "Skilled workers in industry, construction and crafts",
            "Installation and machine operators and assembly workers",
            "Unskilled workers",
            "Armed forces professions",
            "Other situation",
            "Blank / not specified",
            "Armed forces officers",
            "Armed forces sergeants",
            "Other armed forces personnel",
            "Directors of administrative/commercial services",
            "Hotel/catering/trade/services directors",
            "Physical sciences/mathematics/engineering specialists",
            "Health professionals",
            "Teachers",
            "Finance/accounting/admin organization/public-commercial relations specialists",
            "Intermediate science and engineering technicians",
            "Intermediate-level health technicians",
            "Intermediate legal/social/sports/cultural technicians",
            "ICT technicians",
            "Office workers/secretaries/data processing",
            "Data/accounting/statistical/financial/registry operators",
            "Other administrative support staff",
            "Personal service workers",
            "Sellers",
            "Personal care workers",
            "Protection/security services",
            "Market-oriented farmers/skilled agricultural workers",
            "Subsistence farmers/livestock/fishermen/hunters/gatherers",
            "Skilled construction workers except electricians",
            "Skilled metallurgy/metalworking workers",
            "Skilled electricity/electronics workers",
            "Food processing/woodworking/clothing/other craft workers",
            "Fixed plant/machine operators",
            "Assembly workers",
            "Vehicle drivers/mobile equipment operators",
            "Unskilled agriculture/animal/fisheries/forestry",
            "Unskilled extractive/construction/manufacturing/transport",
            "Meal preparation assistants",
            "Street vendors/street service providers",
        ],
    ),

    "Displaced": {0: "No", 1: "Yes"},
    "Educational special needs": {0: "No", 1: "Yes"},
    "Debtor": {0: "No", 1: "Yes"},
    "Tuition fees up to date": {0: "No", 1: "Yes"},
    "Gender": {0: "Female", 1: "Male"},
    "Scholarship holder": {0: "No", 1: "Yes"},
    "International": {0: "No", 1: "Yes"},
}


def display_label(feature, value):
    value = int(value) if isinstance(value, (int, float)) and float(value).is_integer() else value
    return LABELS.get(feature, {}).get(value, f"Code {value}")


if not os.path.exists(MODEL_PATH):
    st.error(
        "The trained model pipeline was not found. Make sure "
        "`models/student_outcome_pipeline.joblib` exists."
    )
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("`dataset.csv` was not found. Keep the dataset in the project root.")
    st.stop()

if not os.path.exists(CONFIG_PATH):
    st.error(
        "The saved feature configuration was not found. Make sure "
        "`models/feature_config.json` exists."
    )
    st.stop()

model = load_model()
dataset = load_dataset()
feature_config = load_feature_config()

TARGET = feature_config["target"]
MODEL_FEATURES = feature_config["features"]
NUMERIC_FEATURES = feature_config["numeric_features"]

missing_dataset_features = [c for c in MODEL_FEATURES if c not in dataset.columns]
if missing_dataset_features:
    st.error(f"dataset.csv is missing model features: {missing_dataset_features}")
    st.stop()


def select_encoded(feature):
    options = sorted(dataset[feature].dropna().unique().tolist())
    return st.selectbox(
        feature,
        options,
        format_func=lambda x: display_label(feature, x)
    )


st.title("🎓 Student Outcome Classification")
st.caption("Machine Learning Mini Project — enrollment-time early-warning model")

st.write(
    "Enter information available at or around enrollment. The model predicts "
    "whether the student outcome is **Dropout**, **Enrolled**, or **Graduate**."
)

with st.form("prediction_form"):
    st.subheader("Student Information")
    inputs = {}

    st.markdown("#### Personal / Background Information")
    personal_cols = [
        "Age at enrollment",
        "Gender",
        "Marital status",
        "Nacionality",
        "Displaced",
        "International",
        "Educational special needs",
    ]

    for start in range(0, len(personal_cols), 3):
        row = st.columns(3)
        for j, feature in enumerate(personal_cols[start:start + 3]):
            with row[j]:
                if feature in NUMERIC_FEATURES:
                    inputs[feature] = st.number_input(
                        feature,
                        min_value=float(dataset[feature].min()),
                        max_value=float(dataset[feature].max()),
                        value=float(dataset[feature].median()),
                        step=1.0,
                    )
                else:
                    inputs[feature] = select_encoded(feature)

    st.markdown("#### Admission / Academic Background")
    admission_cols = [
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Previous qualification",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ]

    for start in range(0, len(admission_cols), 3):
        row = st.columns(3)
        for j, feature in enumerate(admission_cols[start:start + 3]):
            with row[j]:
                if feature == "Application order":
                    options = sorted(dataset[feature].dropna().unique().tolist())
                    inputs[feature] = st.selectbox(
                        feature,
                        options,
                        format_func=lambda x: (
                            "First choice" if x == 0
                            else "Last choice" if x == 9
                            else f"{int(x)}th choice"
                        )
                    )
                else:
                    inputs[feature] = select_encoded(feature)

    st.markdown("#### Financial / Enrollment Information")
    finance_cols = [
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
    ]

    row = st.columns(3)
    for j, feature in enumerate(finance_cols):
        with row[j]:
            inputs[feature] = select_encoded(feature)

    st.markdown("#### Economic Context")
    row = st.columns(3)
    for j, feature in enumerate(["Unemployment rate", "Inflation rate", "GDP"]):
        with row[j]:
            inputs[feature] = st.number_input(
                feature,
                min_value=float(dataset[feature].min()),
                max_value=float(dataset[feature].max()),
                value=float(dataset[feature].median()),
                format="%.4f",
            )

    submitted = st.form_submit_button("Predict Student Outcome", type="primary")


if submitted:
    missing = [feature for feature in MODEL_FEATURES if feature not in inputs]

    if missing:
        st.error(f"Missing input fields: {missing}")
        st.stop()

    input_df = pd.DataFrame(
        [{feature: inputs[feature] for feature in MODEL_FEATURES}]
    )

    try:
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Outcome: **{prediction}**")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            probability_df = pd.DataFrame({
                "Outcome": classes,
                "Probability": probabilities,
            }).sort_values("Probability", ascending=False)

            st.subheader("Prediction Probabilities")

            st.dataframe(
                probability_df.style.format({"Probability": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )

        st.info(
            "This is a machine-learning prediction based on the supplied inputs. "
            "It is not a guaranteed outcome and should not be treated as a "
            "definitive judgment about a student."
        )

    except Exception as exc:
        st.error(
            "The prediction could not be generated. Check that the model, "
            "dataset, and feature configuration match."
        )
        st.exception(exc)


with st.expander("Why are semester-performance fields not shown?"):
    st.write(
        "The primary model is defined as an enrollment-time early-warning model. "
        "First- and second-semester curricular variables are observed after "
        "enrollment, so using them here would create temporal leakage. They can "
        "be used in a separate later-semester prediction model if that prediction "
        "point is explicitly defined."
    )

with st.expander("About the dataset codes"):
    st.write(
        "The dataset used by this project contains compact numeric encodings for "
        "several categorical variables. The dropdowns above show human-readable "
        "descriptions, while the original encoded values are sent internally to "
        "the trained model so the model receives the exact format it was trained on."
    )
