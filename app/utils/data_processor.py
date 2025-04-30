import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """
    Preprocess the customer data for machine learning
    
    Parameters:
    df (pandas.DataFrame): DataFrame with customer data
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Drop customer_id as it's not a feature
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)
    
    # Define target variable
    y = df['churned']
    
    # Drop target from features
    X = df.drop('churned', axis=1)
    
    # Split categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    onehot_features = []
    if categorical_features:
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        onehot_categories = onehot_encoder.categories_
        for i, category in enumerate(onehot_categories):
            for cat_value in category:
                onehot_features.append(f"{categorical_features[i]}_{cat_value}")
    
    feature_names = numerical_features + onehot_features
    
    return X_train_processed, X_test_processed, y_train, y_test, feature_names

if __name__ == "__main__":
    # Test the preprocessor
    from data_generator import generate_data
    df = generate_data(100)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names: {feature_names[:5]}...")
