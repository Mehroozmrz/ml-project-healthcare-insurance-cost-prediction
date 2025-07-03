import pandas as pd
from joblib import load

model_rest = load("artifacts/model_rest.joblib")
model_young = load("artifacts/model_young.joblib")

scaler_rest = load("artifacts/scaler_rest.joblib")
scaler_young = load("artifacts/scaler_young.joblib")


def calculate_normalised_risk(medical_history):
    # Internal risk scores
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    # Preprocess input
    diseases = medical_history.lower().split(" & ")
    diseases = [d.strip() for d in diseases]

    # If only one disease, add "none" as the second
    if len(diseases) == 1:
        diseases.append("none")

    # Calculate total risk score
    total_score = sum([risk_scores.get(d, 0) for d in diseases])

    # Normalize: max possible score = highest 2 disease scores combined
    max_score = max(risk_scores.values()) * 2  # 8 * 2 = 16
    min_score = 0

    normalized_score = (total_score - min_score) / (max_score - min_score)
    return round(normalized_score, 4)




def preprocess_input(input_data):
    expected_columns = [
        'age','number_of_dependants','income_lakhs','insurance_plan','genetical_risk','normalized_risk_score',
        'gender_Male','region_Northwest','region_Southeast','region_Southwest','marital_status_Unmarried',
        'bmi_category_Obesity','bmi_category_Overweight','bmi_category_Underweight','smoking_status_Occasional',
        'smoking_status_Regular','employment_status_Salaried','employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze':1, 'Silver':2, 'Gold':3}
    df = pd.DataFrame(0,columns=expected_columns, index=[0])

    for key,value in input_data.items():
        if key == 'age':
            df['age'] = value
        elif key == 'number_of_dependants':
            df['number_of_dependants'] = value
        elif key == 'income_lakhs':
            df['income_lakhs'] = value
        elif key == 'insurance_plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 0)
        elif key == 'genetical_risk':
            df['genetical_risk'] = value
        elif key == 'normalized_risk_score':
            df['normalized_risk_score'] = value

        elif key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1

        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1

        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1

        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1

        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1

        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1


    df['normalized_risk_score'] = calculate_normalised_risk(input_data['medical_history'])

    df = handle_scaling(input_data['age'], df)
    return df

def handle_scaling(age, df):
    if age<=25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis='columns', inplace=True)
    return df

def predict(input_data):
    input_df = preprocess_input(input_data)

    if input_data['age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction)