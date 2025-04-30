# Churn Prediction App for Food Delivery Platforms

This is a Flask-based web application that predicts customer churn for food delivery and quick commerce platforms like Swiggy, Zomato, Zepto, and Blinkit. The application uses machine learning to identify customers who are at risk of churning, allowing businesses to take proactive measures to retain them.

## Features

- **Synthetic Data Generation**: Creates realistic customer data for food delivery platforms
- **Machine Learning Model**: Predicts customer churn based on various features
- **Interactive Web Interface**: User-friendly interface for making predictions
- **Dashboard**: Visualizes churn analytics and insights
- **Retention Recommendations**: Suggests strategies to reduce churn

## Supported Platforms

- Swiggy
- Zomato
- Zepto
- Blinkit

## Installation

1. Clone the repository:
```
git clone https://github.com/usshaa/Churn_Prediction.git
cd Churn_Prediction
```

2. Create a virtual environment and activate it:
```
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Run the application:
```
python app.py
```

5. Open your browser and navigate to `http://127.0.0.1:5000/`

## Project Structure

```
churn-prediction/
├── app/
│   ├── data/
│   │   └── customer_data.csv (generated on first run)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── churn_model.pkl (generated on first run)
│   │   └── preprocessor.pkl
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── main.js
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── index.html
│   │   ├── predict.html
│   │   └── result.html
│   ├── utils/
│   │   ├── data_generator.py
│   │   └── data_processor.py
│   └── __init__.py
├── app.py
├── README.md
└── requirements.txt
```

## Usage

1. **Home Page**: Overview of the application
2. **Predict Page**: Enter customer details to predict churn
3. **Dashboard**: View churn analytics and insights
4. **Result Page**: See prediction results and retention recommendations

## Model Features

The churn prediction model uses the following features:

- Platform (Swiggy, Zomato, Zepto, Blinkit)
- Customer demographics (age, gender, city)
- Tenure (months)
- Order frequency (orders per month)
- Average order value (INR)
- Discount usage rate
- Delivery issues rate
- Support interactions
- App usage frequency
- Payment method
- Subscription status
- Days since last order
- Review score
- Referrals made

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project uses synthetic data to simulate customer behavior
- Built with Flask, scikit-learn, pandas, and other open-source libraries

---

# Django-demo
