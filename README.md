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
│   │   └── 

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
│   │   └── rpy
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


##

1. Application
Welcome, everyone! In this video, we’ll explore a project called Churn Prediction for Food Delivery Platforms like Zepto, Blinkit, etc.

2. What Does It Do?
This project predicts customer churn based on input features.
We’ve trained a Random Forest Classification model to perform this prediction effectively.

3. To Build This Project
We need two essential software tools: Python and Visual Studio Code (VS Code).

Let me show you how to install them:

Go to your browser and search for the official Python website. Click the Download button to get the installer.

Follow the same steps to download VS Code from its official website.

Once downloaded, install them step-by-step as prompted by the installer windows.

4. Understanding the Structure of the Project Source Code
Now, let’s understand the structure of the project:

First, we have the requirements.txt file, which contains all the necessary Python modules and packages required to run this project.
You can install them using the pip install -r requirements.txt command in your terminal.

The core backend logic of this project is inside the app.py file.
This file connects the machine learning model, the preprocessed data, and the frontend interface.

The data used for this project is simulated using Python, and preprocessing is also handled in this folder.
The final preprocessed data is saved in the data folder.

Let me show you the data being used to train the model.
It contains several fields that help us train the machine learning model.

The model folder contains two important files:

prediction.pkl

preprocessing.pkl
These are generated using the Random Forest Classification model in the train_model.py file.

On the frontend side, the templates folder includes all the HTML pages that make up the user interface.

To run this project, use the terminal and enter the command:

bash
Copy code
python app.py
This will start a local server, and you can access the application through the browser using the provided link.

5. Application Workflow
Let me walk you through the application workflow:

On the homepage, you’ll find a navigation bar with three menu options.

Prediction:
Clicking this will take you to a page where you can enter user input values.
Based on the inputs, the model will predict the customer churn status and provide actionable insights and recommendations to help the platform retain the customer.

Dashboard:
This page displays detailed churn analysis across different services on the platform.
It helps the business understand churn patterns and make data-driven decisions to improve their services.

Customer Segmentation:
This feature helps us understand the risk level of each customer — who is likely to churn and who is still engaged.
It is built using a segmentation model trained on historical data.

Conclusion
That’s how this project works, combining data analytics and AI/ML models to deliver valuable insights.

I hope you’ve gained some useful insights from this project.
Thank you so much for watching this video!

