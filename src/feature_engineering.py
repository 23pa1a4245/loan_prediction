import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df, is_train=True):
    # Feature creation
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    # Log transforms (avoid log(0))
    df['LoanAmount_log'] = np.log(df['LoanAmount'] + 1)
    df['Total_Income_log'] = np.log(df['Total_Income'] + 1)

    # Encoding categorical columns
    le = LabelEncoder()

    categorical_cols = [
        'Gender', 'Married', 'Dependents',
        'Education', 'Self_Employed', 'Property_Area'
    ]

    if is_train and 'Loan_Status' in df.columns:
        categorical_cols.append('Loan_Status')

    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    return df
