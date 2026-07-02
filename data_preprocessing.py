"""
data_preprocessing.py
Handles data loading, cleaning, feature engineering, feature selection, and scaling.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def load_data(filepath):
    """Load CSV data into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  # remove leading/trailing whitespace in column names
    return df


def clean_data(df):
    """Handle missing values, fix bad numeric entries, and drop irrelevant columns."""
    df = df.copy()

    # Drop identifier and junk columns
    drop_cols = ['Sl. No', 'Patient File No.', 'Unnamed: 44']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Some numeric columns in this dataset contain stray text/typos (e.g. "1.99.")
    numeric_like_cols = [
        'AMH(ng/mL)', 'II    beta-HCG(mIU/mL)', '  I   beta-HCG(mIU/mL)'
    ]
    for col in numeric_like_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill remaining missing numeric values with column mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Drop any column that's still fully non-numeric junk (safety net)
    df.dropna(axis=1, how='all', inplace=True)

    return df


def engineer_features(df):
    """Calculate BMI manually and encode categorical variables."""
    df = df.copy()

    # Feature engineering: calculate BMI ourselves from Weight and Height
    if 'Weight (Kg)' in df.columns and 'Height(Cm)' in df.columns:
        df['BMI'] = df['Weight (Kg)'] / ((df['Height(Cm)'] / 100) ** 2)

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df


def select_features(df, target_col='PCOS (Y/N)'):
    """Keep only clinically relevant features known to correlate with PCOS."""
    relevant_cols = [
        target_col,
        'BMI', 'Follicle No. (L)', 'Follicle No. (R)',
        'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)',
        'Cycle(R/I)', 'Cycle length(days)',
        'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)',
        'Weight gain(Y/N)', 'Pimples(Y/N)',
        'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'AMH(ng/mL)',
        'Fast food (Y/N)', 'Reg.Exercise(Y/N)'
    ]
    available_cols = [c for c in relevant_cols if c in df.columns]
    return df[available_cols]


def scale_features(X):
    """Apply MinMax scaling to feature matrix."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def prepare_dataset(filepath, target_col='PCOS (Y/N)'):
    """Full preprocessing pipeline. Returns X, y, and cleaned df."""
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)

    df_selected = select_features(df, target_col)

    y = df_selected[target_col]
    X = df_selected.drop(columns=[target_col])

    X_scaled, scaler = scale_features(X)
    return X_scaled, y, df  # return full df so visualize.py can still access BMI etc.