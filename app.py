from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from app.utils.data_generator import generate_data
from app.models.train_model import train_churn_model

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Define global variables for model, preprocessor, and feature information
model = None
preprocessor = None
feature_names = []
feature_importance = []
categorical_features = []
numerical_features = []

# Check if model exists, if not train it
model_path = 'app/models/churn_model.pkl'
preprocessor_path = 'app/models/preprocessor.pkl'

def load_or_train_model():
    global model, preprocessor, feature_names, feature_importance, categorical_features, numerical_features

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        # Generate data if it doesn't exist
        data_path = 'app/data/customer_data.csv'
        if not os.path.exists(data_path):
            print("Generating synthetic data...")
            df = generate_data(5000)  # Generate 5000 customer records
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
        else:
            df = pd.read_csv(data_path)

        # Drop customer_id if it exists
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

        # Train model
        print("Training churn prediction model...")
        model, accuracy, feature_importance = train_churn_model(X_train_processed, X_test_processed, y_train, y_test, feature_names)

        # Save model, preprocessor, and feature names
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump((model, feature_names, feature_importance, categorical_features, numerical_features), model_path)
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Model trained with accuracy: {accuracy:.2f}")
    else:
        print("Loading existing model and preprocessor...")
        model, feature_names, feature_importance, categorical_features, numerical_features = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

# Load the model at startup
load_or_train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        try:
            # Create a dictionary to hold input features
            input_data = {}

            # Extract numerical features
            for feature in numerical_features:
                if feature in request.form:
                    input_data[feature] = float(request.form[feature])

            # Extract categorical features
            for feature in categorical_features:
                # For categorical features, we need to handle them differently
                # We'll use the value directly from the form
                if feature in request.form:
                    input_data[feature] = request.form[feature]

            # Create a DataFrame with the input features
            input_df = pd.DataFrame([input_data])

            # Apply the same preprocessing as during training
            input_processed = preprocessor.transform(input_df)

            # Make prediction
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0][1]

            result = {
                'churn_prediction': int(prediction),
                'churn_probability': float(probability),
                'features': input_data
            }

            return render_template('result.html', result=result)
        except Exception as e:
            return render_template('predict.html', error=str(e),
                                  numerical_features=numerical_features,
                                  categorical_features=categorical_features)

    # GET request - show the prediction form
    return render_template('predict.html',
                          numerical_features=numerical_features,
                          categorical_features=categorical_features)

@app.route('/dashboard')
def dashboard():
    # Load data for dashboard
    data_path = 'app/data/customer_data.csv'
    df = pd.read_csv(data_path)

    # Calculate some statistics for the dashboard
    total_customers = len(df)
    churn_rate = df['churned'].mean() * 100
    avg_tenure = df['tenure_months'].mean()

    # Feature importance is already loaded globally

    # Convert feature importance to list for the template
    feature_imp_list = []
    for feature, importance in feature_importance:
        feature_imp_list.append({'feature': feature, 'importance': importance})

    return render_template('dashboard.html',
                          total_customers=total_customers,
                          churn_rate=churn_rate,
                          avg_tenure=avg_tenure,
                          feature_importance=feature_imp_list)

@app.route('/export-data')
def export_data():
    # Load data
    data_path = 'app/data/customer_data.csv'

    # Return the CSV file as a download
    return send_file(data_path,
                     mimetype='text/csv',
                     download_name='customer_data.csv',
                     as_attachment=True)

@app.route('/segmentation')
def segmentation():
    # Define customer segments
    segments = [
        {
            'name': 'High-Value Loyalists',
            'size': 1250,
            'churn_rate': 5.2,
            'avg_value': 850,
            'color': 'success',
            'bg_color': 'rgba(40, 167, 69, 0.7)',
            'border_color': 'rgba(40, 167, 69, 1)',
            'traits': [
                'High order frequency (8+ orders per month)',
                'Long tenure (12+ months)',
                'High average order value (₹700+)',
                'Low delivery issues rate',
                'Subscription members'
            ],
            'strategy': 'Focus on exclusivity and premium experiences. Offer early access to new features, premium delivery options, and personalized recommendations.'
        },
        {
            'name': 'At-Risk High Spenders',
            'size': 750,
            'churn_rate': 35.8,
            'avg_value': 920,
            'color': 'danger',
            'bg_color': 'rgba(220, 53, 69, 0.7)',
            'border_color': 'rgba(220, 53, 69, 1)',
            'traits': [
                'High average order value (₹700+)',
                'Decreasing order frequency',
                'Recent delivery issues',
                'Multiple support interactions',
                'Long periods between orders'
            ],
            'strategy': 'Immediate intervention with personalized outreach. Offer premium support, order credits for past issues, and exclusive discounts to regain trust.'
        },
        {
            'name': 'Consistent Mid-Tier',
            'size': 1850,
            'churn_rate': 12.5,
            'avg_value': 450,
            'color': 'primary',
            'bg_color': 'rgba(0, 123, 255, 0.7)',
            'border_color': 'rgba(0, 123, 255, 1)',
            'traits': [
                'Medium order frequency (3-7 orders per month)',
                'Medium tenure (6-12 months)',
                'Consistent ordering patterns',
                'Occasional use of discounts',
                'Few delivery issues'
            ],
            'strategy': 'Encourage increased usage through targeted promotions. Implement a tiered loyalty program with achievable benefits to increase engagement.'
        },
        {
            'name': 'New Customers',
            'size': 1150,
            'churn_rate': 28.3,
            'avg_value': 380,
            'color': 'warning',
            'bg_color': 'rgba(255, 193, 7, 0.7)',
            'border_color': 'rgba(255, 193, 7, 1)',
            'traits': [
                'Short tenure (< 3 months)',
                'Variable order frequency',
                'High discount usage',
                'Exploring different offerings',
                'Sensitive to delivery experience'
            ],
            'strategy': 'Focus on onboarding and first impressions. Provide guided experiences, educational content, and early-win discounts to establish ordering habits.'
        }
    ]

    return render_template('segmentation.html', segments=segments)

if __name__ == '__main__':
    app.run(debug=True)
