import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page config
st.set_page_config(
    page_title="SecureRetail AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "scanner_history" not in st.session_state:
    st.session_state["scanner_history"] = []
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

# Helper function to load data
@st.cache_data
def load_data(sales_file=None, inv_file=None):
    if sales_file is not None:
        sales = pd.read_csv(sales_file)
    elif os.path.exists("data/sales_history.csv"):
        sales = pd.read_csv("data/sales_history.csv")
    else:
        sales = pd.DataFrame()

    if inv_file is not None:
        inventory = pd.read_csv(inv_file)
    elif os.path.exists("data/current_inventory.csv"):
        inventory = pd.read_csv("data/current_inventory.csv")
    else:
        inventory = pd.DataFrame()

    return sales, inventory

# Helper function to initialize ML models
@st.cache_resource
def initialize_models(_sales_df):
    from src.demand_model import train_demand_model
    from src.security_model import train_security_model, load_security_model
    
    models = {}
    security_pipeline = None
    
    # Check if we need to generate initial mock data
    if not os.path.exists("data/sales_history.csv") or not os.path.exists("data/messages.csv"):
        from src.data_generator import generate_sales_and_inventory, generate_phishing_data
        with st.spinner("Generating mock dataset and training models..."):
            generate_sales_and_inventory()
            generate_phishing_data()
            _sales_df, _ = load_data()
            
    if not _sales_df.empty:
        models = train_demand_model(_sales_df)
        
    if not os.path.exists("models/phishing_pipeline.pkl"):
        security_pipeline = train_security_model()
    else:
        security_pipeline = load_security_model()
        
    return models, security_pipeline


def apply_theme():
    """Generates the CSS string based on the theme state."""
    base_css = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        
        .stButton>button {
            border-radius: 8px; font-weight: 600; padding: 0.5rem 1rem;
            transition: all 0.2s ease-in-out; border: none;
        }
        .stButton>button:hover { transform: translateY(-1px); }
    """

    if st.session_state["theme"] == "Dark":
        theme_css = """
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #172554 100%);
            color: #f8fafc;
        }
        
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        /* Fix the navigation text color in dark mode */
        .st-emotion-cache-16txtl3, .st-emotion-cache-1v0mbdj {
            color: #f8fafc !important; 
        }
        
        div[data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.85);
            border-radius: 16px; padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            color: white;
        }
        
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        
        .main-header {
            background: -webkit-linear-gradient(45deg, #f8fafc, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem; font-weight: 800;
        }
        
        .alert-card-danger { background: rgba(220, 38, 38, 0.15); padding: 1.2rem; border-radius: 12px; border-left: 5px solid #ef4444; color: #fca5a5; margin-bottom: 0.8rem; }
        .alert-card-warning { background: rgba(217, 119, 6, 0.15); padding: 1.2rem; border-radius: 12px; border-left: 5px solid #f59e0b; color: #fcd34d; margin-bottom: 0.8rem; }
        .stButton>button { background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); color: white; }
        """
    else:
        theme_css = """
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            color: #1e293b;
        }
        
        [data-testid="stSidebar"] {
            background-color: rgba(248, 250, 252, 0.95);
            border-right: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        /* Fix the navigation text color in light mode */
        .st-emotion-cache-16txtl3, .st-emotion-cache-1v0mbdj {
            color: #1e293b !important; 
        }

        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px; padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.6);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            color: #1e293b;
        }
        
        div[data-testid="stMetricValue"] {
            color: #1e293b;
        }
        
        .main-header {
            background: -webkit-linear-gradient(45deg, #1e293b, #2563eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem; font-weight: 800;
        }
        
        .alert-card-danger { background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); padding: 1.2rem; border-radius: 12px; border-left: 5px solid #ef4444; color: #991b1b; margin-bottom: 0.8rem; }
        .alert-card-warning { background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); padding: 1.2rem; border-radius: 12px; border-left: 5px solid #f59e0b; color: #92400e; margin-bottom: 0.8rem; }
        .stButton>button { background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); color: white; }
        """

    st.markdown(f"<style>{base_css}{theme_css}</style>", unsafe_allow_html=True)


def login_screen():
    st.markdown("<h1 style='text-align: center; margin-top: 10vh;'>SecureRetail AI Access</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.markdown("<div style='background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 12px; border: 1px solid rgba(125,125,125,0.2);'>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="password")
        if st.button("Secure Login", use_container_width=True):
            if username == "admin" and password == "password":
                st.session_state["authenticated"] = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials. Try admin / password.")
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    apply_theme()
    
    if not st.session_state["authenticated"]:
        login_screen()
        return

    st.sidebar.image("https://img.icons8.com/nolan/96/shop.png", width=84)
    st.sidebar.markdown("## SecureRetail AI\n<span style='color: #64748b;'>Admin Terminal</span>", unsafe_allow_html=True)
    
    page = st.sidebar.radio("Navigation", 
                           ["Dashboard Overview", 
                            "Inventory & Surplus", 
                            "Demand Forecast", 
                            "Security Scanner",
                            "Data Management"])
    
    st.sidebar.markdown("---")
    
    # Theme Toggle
    is_dark = st.sidebar.toggle("🌙 Dark Mode", value=(st.session_state["theme"] == "Dark"))
    if is_dark and st.session_state["theme"] == "Light":
        st.session_state["theme"] = "Dark"
        st.rerun()
    elif not is_dark and st.session_state["theme"] == "Dark":
        st.session_state["theme"] = "Light"
        st.rerun()
        
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

    from src.inventory_logic import compute_inventory_insights
    
    sales_df, inventory_df = load_data()
    demand_models, security_pipeline = initialize_models(sales_df)
    
    insights_df = pd.DataFrame()
    if sales_df.empty or inventory_df.empty:
        st.warning("No data found. Please go to Data Management and upload your files or generate mock data.")
    else:
        insights_df = compute_inventory_insights(inventory_df, demand_models, forecast_days=30)
    
    # --- PAGE 1: OVERVIEW ---
    if page == "Dashboard Overview":
        st.markdown("<h1 class='main-header'>Business Health Overview</h1>", unsafe_allow_html=True)
        
        if not insights_df.empty:
            total_products = len(inventory_df)
            stockout_risks = len(insights_df[insights_df['Risk Level'].str.contains("Stockout")])
            overstock_risks = len(insights_df[insights_df['Risk Level'].str.contains("Overstock")])
            total_value = int(inventory_df['current_stock'].sum())
            total_surplus = int(insights_df['Surplus Quantity'].sum())
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SKUs Tracked", total_products, "Active Catalog")
            c2.metric("Critical Low Stock", stockout_risks, f"-{stockout_risks} Requires Action", delta_color="inverse")
            c3.metric("Overstock Risk", overstock_risks, f"{total_surplus} Surplus Items", delta_color="off")
            c4.metric("Total Items In Stock", total_value)
            
            st.markdown("### 🔔 Actionable Smart Alerts")
            
            alerts = insights_df[insights_df['Alert'] != ""]
            if alerts.empty:
                st.success("All stock levels look healthy! No immediate action required.")
            else:
                for idx, row in alerts.iterrows():
                    if "High Risk" in row['Risk Level'] and "Stockout" in row['Risk Level']:
                        st.markdown(f"""<div class="alert-card-danger">
                            <h4>🚨 Critical Stockout: {row['Product Name']} ({row['Product ID']})</h4>
                            <p>{row['Alert']}</p>
                        </div>""", unsafe_allow_html=True)
                    elif "Overstock" in row['Risk Level']:
                        st.markdown(f"""<div class="alert-card-warning">
                            <h4>📦 Excess Inventory: {row['Product Name']} ({row['Product ID']})</h4>
                            <p>{row['Alert']} <strong>Potential Liquidation: {row['Surplus Quantity']} units.</strong></p>
                        </div>""", unsafe_allow_html=True)

    # --- PAGE 2: INVENTORY & SURPLUS ---
    elif page == "Inventory & Surplus":
        st.markdown("<h1 class='main-header'>Inventory Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("Automated comparison between your current stock and 30-day predicted forecasting.")
        
        if not insights_df.empty:
            def highlight_risk(val):
                if pd.isna(val): return ''
                if "Stockout" in str(val): return 'background-color: #fee2e2; color: #b91c1c; font-weight: bold'
                elif "Overstock" in str(val): return 'background-color: #fef3c7; color: #b45309; font-weight: bold'
                return 'background-color: #dcfce7; color: #15803d'
                
            styled_df = insights_df.style.applymap(highlight_risk, subset=['Risk Level'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            st.markdown("### 📊 Distribution Breakdown")
            # Decide plotly template based on theme
            template = "plotly_dark" if st.session_state["theme"] == "Dark" else "plotly_white"
            fig = px.bar(insights_df, x="Product Name", y=["Current Stock", "Predicted Demand (30 Days)", "Surplus Quantity"], 
                         barmode="group",
                         color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"],
                         template=template)
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- PAGE 3: DEMAND FORECAST ---
    elif page == "Demand Forecast":
        st.markdown("<h1 class='main-header'>Demand Trajectory</h1>", unsafe_allow_html=True)
        
        if not inventory_df.empty:
            product_names = inventory_df['product_name'].tolist()
            selected_product = st.selectbox("Select a Product Schedule", product_names)
            
            product_id = inventory_df[inventory_df['product_name'] == selected_product]['product_id'].values[0]
            
            hist_df = sales_df[sales_df['product_id'] == product_id].copy()
            hist_df['date'] = pd.to_datetime(hist_df['date'])
            recent_hist = hist_df[hist_df['date'] >= (pd.Timestamp.today() - pd.Timedelta(days=90))]
            
            from src.demand_model import predict_demand
            pred_df = predict_demand(demand_models, product_id, forecast_days=30)
            
            template = "plotly_dark" if st.session_state["theme"] == "Dark" else "plotly_white"
            fig = px.line(title=f"90-Day History & 30-Day Forecast: {selected_product}", template=template)
            fig.add_scatter(x=recent_hist['date'], y=recent_hist['sales_quantity'], mode='lines+markers', 
                            name='Historical Sales', line=dict(color="#3b82f6", width=2))
            
            if pred_df is not None:
                fig.add_scatter(x=pred_df['date'], y=pred_df['predicted_sales'], mode='lines', 
                                name='AI Forecast', line=dict(color="#f43f5e", width=4, dash='dot'))
            
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)", 
                hovermode="x unified",
                xaxis_title="Timeline",
                yaxis_title="Units Requested"
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- PAGE 4: SECURITY SCANNER ---
    elif page == "Security Scanner":
        st.markdown("<h1 class='main-header'>AI Phishing Defender</h1>", unsafe_allow_html=True)
        st.markdown("Protect your cash flow. Scan distributor emails, SMS payment links, and WhatsApp warnings.")
        
        message_input = st.text_area("Paste suspicious message here:", height=150, 
                                     placeholder="e.g. URGENT GST WARNING: Your account is locked...")
        
        if st.button("Initial Security Scan", type="primary"):
            if not message_input.strip():
                st.warning("Please enter a message to scan.")
            elif security_pipeline is None:
                st.error("Security model is offline.")
            else:
                from src.security_model import scan_message
                prediction, risk_score = scan_message(security_pipeline, message_input)
                
                scan_record = {
                    "Message Snippet": message_input[:50] + "...",
                    "Classification": prediction.upper(),
                    "Risk Score": f"{risk_score}%"
                }
                st.session_state["scanner_history"].insert(0, scan_record)
                
                if prediction == "phishing":
                    st.error("🚨 CRITICAL ALERT: Phishing Detected!")
                    st.metric("Threat Probability", f"{risk_score}%")
                    st.markdown("Do **NOT** click any links. Delete this message immediately.")
                else:
                    st.success("✅ SAFE: No obvious phishing vectors detected.")
                    st.metric("Threat Probability", f"{risk_score}%")
                    st.markdown("Proceed with standard caution.")
                    
        if st.session_state["scanner_history"]:
            st.markdown("---")
            st.markdown("### Scanner Session Log")
            st.dataframe(pd.DataFrame(st.session_state["scanner_history"]), use_container_width=True)

    # --- PAGE 5: DATA MANAGEMENT ---
    elif page == "Data Management":
        st.markdown("<h1 class='main-header'>System Settings & Data Management</h1>", unsafe_allow_html=True)
        
        st.markdown("### 1. Upload Custom Data")
        col1, col2 = st.columns(2)
        sales_file = col1.file_uploader("Upload Sales History (CSV)", type="csv")
        inv_file = col2.file_uploader("Upload Current Inventory (CSV)", type="csv")
        
        if st.button("Process New Files"):
            if sales_file or inv_file:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Data loaded. Navigate to Overview to see updated insights.")
            else:
                st.warning("Please upload at least one file first.")
                
        st.markdown("---")
        st.markdown("### 2. Sandbox Testing")
        if st.button("Factory Reset (Regenerate Mock Data)"):
            from src.data_generator import generate_sales_and_inventory, generate_phishing_data
            with st.spinner("Rebuilding database..."):
                generate_sales_and_inventory()
                generate_phishing_data()
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Sandbox rebuilt! Refreshing...")
                st.rerun()

if __name__ == "__main__":
    main()
