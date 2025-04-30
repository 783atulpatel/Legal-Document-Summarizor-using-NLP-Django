import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_data(n_samples=5000):
    """
    Generate synthetic customer data for food delivery platforms
    
    Parameters:
    n_samples (int): Number of customer records to generate
    
    Returns:
    pandas.DataFrame: DataFrame with synthetic customer data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f'CUST{i:06d}' for i in range(1, n_samples + 1)]
    
    # Generate platform data (Swiggy, Zomato, Zepto, Blinkit)
    platforms = np.random.choice(['Swiggy', 'Zomato', 'Zepto', 'Blinkit'], size=n_samples, 
                                p=[0.35, 0.35, 0.15, 0.15])
    
    # Generate demographic data
    age = np.random.normal(30, 8, n_samples).astype(int)
    age = np.clip(age, 18, 65)  # Clip age between 18 and 65
    
    gender = np.random.choice(['Male', 'Female', 'Other'], size=n_samples, p=[0.48, 0.48, 0.04])
    
    # Generate location data (major Indian cities)
    cities = np.random.choice(
        ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad'],
        size=n_samples,
        p=[0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.08, 0.07]
    )
    
    # Generate tenure (in months)
    tenure_months = np.random.gamma(shape=10, scale=2, size=n_samples).astype(int)
    tenure_months = np.clip(tenure_months, 1, 60)  # Clip tenure between 1 and 60 months
    
    # Generate order frequency (orders per month)
    order_frequency = np.random.gamma(shape=3, scale=2, size=n_samples)
    order_frequency = np.clip(order_frequency, 0.1, 30)  # Clip between 0.1 and 30 orders per month
    
    # Generate average order value (in INR)
    avg_order_value = np.random.gamma(shape=10, scale=30, size=n_samples)
    avg_order_value = np.clip(avg_order_value, 100, 2000)  # Clip between 100 and 2000 INR
    
    # Generate total spend
    total_spend = order_frequency * avg_order_value * tenure_months
    
    # Generate discount usage rate (percentage of orders with discounts)
    discount_usage_rate = np.random.beta(2, 5, n_samples)
    
    # Generate delivery issues (percentage of orders with issues)
    delivery_issues_rate = np.random.beta(1.5, 8, n_samples)
    
    # Generate customer support interactions per month
    support_interactions = np.random.gamma(shape=0.5, scale=0.5, size=n_samples)
    support_interactions = np.clip(support_interactions, 0, 5)
    
    # Generate app usage frequency (days per week)
    app_usage_frequency = np.random.gamma(shape=2, scale=1, size=n_samples)
    app_usage_frequency = np.clip(app_usage_frequency, 0, 7)
    
    # Generate payment method
    payment_methods = np.random.choice(
        ['UPI', 'Credit Card', 'Debit Card', 'Wallet', 'Cash on Delivery'],
        size=n_samples,
        p=[0.4, 0.2, 0.15, 0.15, 0.1]
    )
    
    # Generate subscription status (premium/pro membership)
    subscription_status = np.random.choice([1, 0], size=n_samples, p=[0.3, 0.7])
    
    # Generate days since last order
    days_since_last_order = np.random.gamma(shape=1.5, scale=10, size=n_samples).astype(int)
    days_since_last_order = np.clip(days_since_last_order, 0, 180)
    
    # Generate review score (average of customer reviews)
    review_score = np.random.normal(4, 0.8, n_samples)
    review_score = np.clip(review_score, 1, 5)
    
    # Generate referrals made
    referrals = np.random.poisson(lam=0.8, size=n_samples)
    
    # Generate churn status based on features
    # This creates a realistic relationship between features and churn
    churn_prob = (
        0.1 +  # Base churn rate
        0.2 * (days_since_last_order > 60).astype(int) +  # Inactive customers
        0.15 * (delivery_issues_rate > 0.2).astype(int) +  # Many delivery issues
        0.1 * (support_interactions > 2).astype(int) +  # Many support interactions
        0.1 * (order_frequency < 1).astype(int) +  # Low order frequency
        0.1 * (tenure_months < 3).astype(int) -  # New customers
        0.1 * (subscription_status) -  # Subscribers less likely to churn
        0.05 * (referrals > 0).astype(int) -  # Customers who refer others less likely to churn
        0.1 * (review_score > 4).astype(int)  # Satisfied customers less likely to churn
    )
    
    # Clip probabilities between 0 and 1
    churn_prob = np.clip(churn_prob, 0.01, 0.99)
    
    # Generate churn based on probability
    churned = np.random.binomial(n=1, p=churn_prob, size=n_samples)
    
    # Create DataFrame
    data = {
        'customer_id': customer_ids,
        'platform': platforms,
        'age': age,
        'gender': gender,
        'city': cities,
        'tenure_months': tenure_months,
        'order_frequency': order_frequency,
        'avg_order_value': avg_order_value,
        'total_spend': total_spend,
        'discount_usage_rate': discount_usage_rate,
        'delivery_issues_rate': delivery_issues_rate,
        'support_interactions': support_interactions,
        'app_usage_frequency': app_usage_frequency,
        'payment_method': payment_methods,
        'subscription_status': subscription_status,
        'days_since_last_order': days_since_last_order,
        'review_score': review_score,
        'referrals': referrals,
        'churned': churned
    }
    
    df = pd.DataFrame(data)
    
    return df

if __name__ == "__main__":
    # Test the data generator
    df = generate_data(100)
    print(df.head())
    print(df.describe())
    print(f"Churn rate: {df['churned'].mean():.2f}")
