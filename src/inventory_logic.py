import pandas as pd

def compute_inventory_insights(inventory_df, demand_models, forecast_days=30):
    """
    Computes inventory insights by comparing current stock against predicted demand.
    Returns a dataframe containing risk levels, reorder quantities, and alerts.
    """
    from src.demand_model import predict_demand, get_total_predicted_demand
    
    insights = []
    
    for index, row in inventory_df.iterrows():
        product_id = row['product_id']
        current_stock = row['current_stock']
        product_name = row['product_name']
        category = row['category']
        
        # Predict demand
        pred_df = predict_demand(demand_models, product_id, forecast_days=forecast_days)
        total_pred_demand = get_total_predicted_demand(pred_df)
        
        # Calculate Reorder Quantity
        # We assume the goal is to have enough stock for the forecast period + 20% safety buffer
        target_stock_level = int(total_pred_demand * 1.2)
        reorder_qty = max(0, target_stock_level - current_stock)
        
        # Calculate Surplus Quantity (for partner exchange)
        surplus_qty = 0
        
        # Determine Risk Level
        risk_level = "Healthy"
        alert_msg = ""
        
        if current_stock < total_pred_demand * 0.5:
            # Having less than half the monthly demand is extremely risky
            risk_level = "High Risk (Stockout)"
            alert_msg = f"Critical low stock for {product_name}. Recommend immediate reorder of {reorder_qty} units."
        elif current_stock < total_pred_demand:
            risk_level = "Medium Risk (Low Stock)"
            alert_msg = f"Stock running low for {product_name}. Recommend reorder of {reorder_qty} units."
        elif current_stock > total_pred_demand * 2.5:
            risk_level = "High Risk (Overstock)"
            surplus_qty = current_stock - int(total_pred_demand * 1.5) # keep 1.5 months worth, liquidate rest
            alert_msg = f"Overstock detected for {product_name}. Transfer {surplus_qty} units to partner stores."
            
        insights.append({
            "Product ID": product_id,
            "Product Name": product_name,
            "Category": category,
            "Current Stock": current_stock,
            "Predicted Demand (30 Days)": int(total_pred_demand),
            "Reorder Quantity": reorder_qty,
            "Surplus Quantity": surplus_qty,
            "Risk Level": risk_level,
            "Alert": alert_msg
        })
        
    insights_df = pd.DataFrame(insights)
    return insights_df
