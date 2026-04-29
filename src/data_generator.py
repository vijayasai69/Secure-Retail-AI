import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_sales_and_inventory():
    np.random.seed(42)
    
    products = [
        {"id": "P001", "name": "Aashirvaad Atta 5kg", "category": "Groceries"},
        {"id": "P002", "name": "Tata Salt 1kg", "category": "Groceries"},
        {"id": "P003", "name": "Maggi Noodles 140g", "category": "Snacks"},
        {"id": "P004", "name": "Amul Butter 100g", "category": "Dairy"},
        {"id": "P005", "name": "Britannia Marie Gold 250g", "category": "Snacks"},
        {"id": "P006", "name": "Surf Excel Matic 1kg", "category": "Household"},
        {"id": "P007", "name": "Colgate Toothpaste 100g", "category": "Personal Care"},
        {"id": "P008", "name": "Paracetamol 500mg (Strip)", "category": "Pharmacy"},
        {"id": "P009", "name": "Dettol Antiseptic Liquid 250ml", "category": "Pharmacy"},
        {"id": "P010", "name": "Cough Syrup 100ml", "category": "Pharmacy"},
        {"id": "P011", "name": "Fortune Sunflower Oil 1L", "category": "Groceries"},
        {"id": "P012", "name": "Lays Classic Salted 50g", "category": "Snacks"},
        {"id": "P013", "name": "Coca Cola 1.25L", "category": "Beverages"},
        {"id": "P014", "name": "Thumbs Up 1.25L", "category": "Beverages"},
        {"id": "P015", "name": "Red Label Tea 250g", "category": "Beverages"},
        {"id": "P016", "name": "Nescafe Coffee 50g", "category": "Beverages"},
        {"id": "P017", "name": "Vim Dishwash Bar 200g", "category": "Household"},
        {"id": "P018", "name": "Harpic Toilet Cleaner 500ml", "category": "Household"},
        {"id": "P019", "name": "Dove Soap 100g", "category": "Personal Care"},
        {"id": "P020", "name": "Band-Aid (Pack of 100)", "category": "Pharmacy"},
    ]
    
    # Generate 2 years of daily data (from 2 years ago to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    sales_data = []
    
    for product in products:
        # Base daily demand and seasonality
        base_demand = np.random.randint(5, 30)
        trend = np.linspace(0, base_demand * 0.2, len(date_range)) # slight upward trend
        
        # Add weekly seasonality (weekends have higher sales for groceries)
        weekly_seasonality = np.array([1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 1.3])
        season_multiplier = np.tile(weekly_seasonality, len(date_range)//7 + 1)[:len(date_range)]
        
        # Generate daily sales volume with noise
        noise = np.random.normal(0, base_demand * 0.2, len(date_range))
        daily_sales = (base_demand + trend) * season_multiplier + noise
        daily_sales = np.maximum(0, daily_sales).astype(int) # non-negative integer
        
        # Add occasional spikes (e.g., festivals, panic buying)
        spike_indices = np.random.choice(len(date_range), size=10, replace=False)
        daily_sales[spike_indices] += np.random.randint(20, 50, size=10)
        
        for date, qty in zip(date_range, daily_sales):
            sales_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "product_id": product["id"],
                "product_name": product["name"],
                "category": product["category"],
                "sales_quantity": qty
            })
            
    sales_df = pd.DataFrame(sales_data)
    
    # Generate current inventory
    inventory_data = []
    for product in products:
        # Generate varied stock levels to simulate low, healthy, and overstock
        avg_monthly_sales = sales_df[sales_df['product_id'] == product['id']]['sales_quantity'][-30:].sum()
        
        rand_val = np.random.rand()
        if rand_val < 0.2: # 20% chance of low stock (danger)
            stock = int(avg_monthly_sales * np.random.uniform(0.1, 0.4))
        elif rand_val < 0.4: # 20% chance of over stock
            stock = int(avg_monthly_sales * np.random.uniform(2.0, 3.5))
        else: # 60% healthy stock
            stock = int(avg_monthly_sales * np.random.uniform(0.8, 1.5))
            
        inventory_data.append({
            "product_id": product["id"],
            "product_name": product["name"],
            "category": product["category"],
            "current_stock": max(0, stock)
        })
        
    inventory_df = pd.DataFrame(inventory_data)
    
    os.makedirs('data', exist_ok=True)
    sales_df.to_csv('data/sales_history.csv', index=False)
    inventory_df.to_csv('data/current_inventory.csv', index=False)
    print("Sales and inventory data generated.")

def generate_phishing_data():
    messages = [
        {"text": "Hey distributor, can you deliver 5 cartons of Maggie tomorrow?", "label": "safe"},
        {"text": "URGENT: Your account will be locked. Click here to verify: http://bit.ly/secure-account", "label": "phishing"},
        {"text": "Your supplier payment of Rs 5000 is received successfully. Ref: HDF12994", "label": "safe"},
        {"text": "Dear Retailer, you have won Rs. 1,00,000 cash prize! Reply with your bank details to claim.", "label": "phishing"},
        {"text": "Reminder: Complete your GST return filing before the 20th to avoid penalties.", "label": "safe"},
        {"text": "Delivery expected by 4PM today. Driver name: Raju (Ph: 9876543210).", "label": "safe"},
        {"text": "Update your banking details for direct deposit immediately. Link: http://paytm-kyc-verify.com", "label": "phishing"},
        {"text": "Congratulations! Your store is selected for a free iPhone 15. Click here www.free-apple-gift.com", "label": "phishing"},
        {"text": "Hi, we are out of Tata Salt. Can you send 20 packets?", "label": "safe"},
        {"text": "Stockout alert: Aashirvaad Atta is heavily demanded.", "label": "safe"},
        {"text": "Your EMI is due on 5th. Kindly maintain sufficient balance.", "label": "safe"},
        {"text": "Your parcel #9871 has been detained by customs. Pay Rs 500 fee here: http://customs.secure-pay.in", "label": "phishing"},
        {"text": "Govt Alert: Download this app to receive subsidy for small businesses: http://subsidyscheme.apk", "label": "phishing"},
        {"text": "Can you check my last invoice? I think you billed me twice for Vim Bar.", "label": "safe"},
        {"text": "ALERT: Unauthorized login attempt on your bank account from Dubai. If this wasn't you, secure account immediately at http://bank-secure-alert.info", "label": "phishing"},
        {"text": "Invoice #402 from distributor is attached for your reference.", "label": "safe"},
        {"text": "Claim your FREE trial of our new retail POS software. Install now: http://malware-link.xyz", "label": "phishing"},
        {"text": "Greetings, your shop registration is expiring. Renew here to avoid closure: http://msme-renewal1.com", "label": "phishing"},
        {"text": "I am sending you the payment via UPI in 5 mins.", "label": "safe"},
        {"text": "Happy Diwali! As a valued seller, we have a special gift. Tap the link to view: http://festival-bonus-claim.com", "label": "phishing"}
    ]
    
    # Duplicate and add slight variations to generate a larger dataset for TF-IDF training
    expanded_messages = []
    for _ in range(20): # Make dataset ~400 rows
        for msg in messages:
            # We just copy them; Naive Bayes will learn the keywords.
            expanded_messages.append(msg)
            
    df = pd.DataFrame(expanded_messages)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    df.to_csv('data/messages.csv', index=False)
    print("Phishing data generated.")

if __name__ == "__main__":
    generate_sales_and_inventory()
    generate_phishing_data()
