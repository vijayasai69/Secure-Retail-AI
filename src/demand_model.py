import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def train_demand_model(sales_df):
    """
    Trains a simple linear regression model for each product to predict 
    sales based on daily trends and seasonality.
    """
    models = {}
    
    # Process the 'date' column
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    sales_df['day_of_week'] = sales_df['date'].dt.dayofweek
    sales_df['month'] = sales_df['date'].dt.month
    sales_df['day_of_year'] = sales_df['date'].dt.dayofyear
    
    products = sales_df['product_id'].unique()
    
    for product_id in products:
        product_data = sales_df[sales_df['product_id'] == product_id].copy()
        
        # Features: day_of_week, month, day_of_year
        X = product_data[['day_of_week', 'month', 'day_of_year']]
        y = product_data['sales_quantity']
        
        model = LinearRegression()
        model.fit(X, y)
        models[product_id] = model
        
    return models

def predict_demand(models, product_id, forecast_days=30):
    """
    Predicts the demand for a specific product for the next `forecast_days` days.
    """
    if product_id not in models:
        return None
        
    model = models[product_id]
    
    # Generate future dates
    today = pd.Timestamp.today()
    future_dates = pd.date_range(start=today, periods=forecast_days, freq='D')
    
    # Create feature dataframe
    future_X = pd.DataFrame({
        'day_of_week': future_dates.dayofweek,
        'month': future_dates.month,
        'day_of_year': future_dates.dayofyear
    })
    
    predictions = model.predict(future_X)
    predictions = np.maximum(0, predictions).astype(int) # ensure no negative predictions
    
    result_df = pd.DataFrame({
        'date': future_dates,
        'predicted_sales': predictions
    })
    
    return result_df

def get_total_predicted_demand(result_df):
    """Returns scalar sum of predicted demand over the forecast period"""
    if result_df is None or result_df.empty:
        return 0
    return result_df['predicted_sales'].sum()
