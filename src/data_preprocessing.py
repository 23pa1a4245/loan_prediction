def preprocess_data(df):
    # Drop Loan_ID if present
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])

    # Categorical missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Numerical missing values
    if 'LoanAmount' in df.columns:
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

    if 'Loan_Amount_Term' in df.columns:
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

    if 'Credit_History' in df.columns:
        df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    return df
