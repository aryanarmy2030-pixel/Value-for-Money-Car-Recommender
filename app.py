import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Value-for-Money Car Recommender",
    page_icon="🚗",
    layout="wide"
)

# Custom Styling & CSS for Professional UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Description
st.title("🚗 Advanced Value-for-Money Car Recommender")
st.markdown("""
This advanced dashboard evaluates second-hand car listings using a supervised **Linear Regression** model. 
It dynamically computes a **Value Score** ($\text{Predicted Market Price} - \text{Listed Price}$) to identify lucrative bargains and overpriced listings in real time.
""")

# Load dataset and model with caching
@st.cache_resource
def load_data_and_model():
    df_path = "cleaned_car_price.csv"
    df = pd.read_csv(df_path) if os.path.exists(df_path) else None
        
    model_path = "linear_regression_model.pkl"
    model = None
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
    return df, model

df, model = load_data_and_model()

if df is None or model is None:
    st.error("⚠️ Model files or dataset (`cleaned_car_price.csv`, `linear_regression_model.pkl`) not found! Please run `train_model.py` first.")
else:
    # Sidebar Filters & Custom Prediction Engine
    st.sidebar.header("🎛️ Interactive Controls & Filters")
    
    # Global Dataset Filter
    st.sidebar.subheader("Dataset Filter")
    max_price_filter = st.sidebar.slider("Max Listed Price (Lakhs)", float(df['Selling_Price'].min()), float(df['Selling_Price'].max()), float(df['Selling_Price'].max()))
    filtered_df = df[df['Selling_Price'] <= max_price_filter]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Live Prediction Engine")
    
    present_price = st.sidebar.number_input("Present Showroom Price (Lakhs)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=100, max_value=500000, value=25000, step=1000)
    age = st.sidebar.slider("Car Age (Years)", min_value=1, max_value=30, value=5, step=1)
    owner = st.sidebar.selectbox("Previous Owners", [0, 1, 2, 3])
    
    fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
    
    listed_price = st.sidebar.number_input("Seller's Listed Price (Lakhs)", min_value=0.1, max_value=50.0, value=4.5, step=0.1)

    # Encoding features
    fuel_diesel = 1.0 if fuel_type == "Diesel" else 0.0
    fuel_petrol = 1.0 if fuel_type == "Petrol" else 0.0
    seller_individual = 1.0 if seller_type == "Individual" else 0.0
    transmission_manual = 1.0 if transmission == "Manual" else 0.0

    if st.sidebar.button("Predict Fair Market Value", type="primary"):
        # Construct feature array mapping directly to model's expected feature names and order
        input_data = pd.DataFrame([[
            present_price, kms_driven, age, owner,
            fuel_diesel, fuel_petrol, seller_individual, transmission_manual
        ]], columns=model.feature_names_in_)
        
        predicted_price = model.predict(input_data)[0]
        value_score = predicted_price - listed_price

        st.sidebar.markdown("### 📊 Prediction Analytics")
        st.sidebar.info(f"""
        * **Predicted Price:** ₹ {predicted_price:.2f} Lakhs
        * **Listed Price:** ₹ {listed_price:.2f} Lakhs
        """)
        
        if value_score > 0:
            st.sidebar.success(f"🔥 Great Deal! Underpriced by ₹ {value_score:.2f} Lakhs")
        else:
            st.sidebar.error(f"⚠️ Overpriced by ₹ {abs(value_score):.2f} Lakhs")

    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Best Value Deals", "📈 Interactive Visualizations", "📋 Filtered Dataset", "⚠️ Overpriced Inventory"])

    with tab1:
        st.subheader("Top 10 High-Value Car Bargains")
        st.markdown("Ranked by highest **Value Score** ($\text{Predicted Price} - \text{Listed Price}$), representing top underpriced market listings.")
        if 'Value_Score' in filtered_df.columns:
            best_deals = filtered_df.sort_values(by='Value_Score', ascending=False).head(10)
            st.dataframe(best_deals, use_container_width=True)
            
            # Quick visual for top deals
            fig_top = px.bar(
                best_deals.head(5), 
                x='Value_Score', 
                y=best_deals.head(5).index.astype(str),
                orientation='h',
                title="Top 5 Value Scores Breakdown",
                labels={'y': 'Car Record Index', 'Value_Score': 'Value Score (Lakhs)'},
                color='Value_Score',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("Value Score metric not computed in dataset.")

    with tab2:
        st.subheader("Advanced Exploratory Data Analytics")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if 'Present_Price' in filtered_df.columns and 'Selling_Price' in filtered_df.columns:
                fig_scatter = px.scatter(
                    filtered_df, x='Present_Price', y='Selling_Price', 
                    color='Age', size='Kms_Driven',
                    title="Showroom Price vs Selling Price (Colored by Age)",
                    hover_data=['Owner']
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
        with col_b:
            if 'Selling_Price' in filtered_df.columns:
                fig_hist = px.histogram(
                    filtered_df, x='Selling_Price', nbins=20,
                    title="Distribution of Selling Prices Across Dataset",
                    color_discrete_sequence=['#3b82f6']
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Complete Filtered Dataset Explorer")
        st.markdown(f"Showing {len(filtered_df)} records matching the price filter criteria.")
        st.dataframe(filtered_df, use_container_width=True)

    with tab4:
        st.subheader("Top 10 Overpriced Inventory Warnings")
        st.markdown("Listings with negative value scores where asking prices substantially exceed linear model predictions.")
        if 'Value_Score' in filtered_df.columns:
            worst_deals = filtered_df.sort_values(by='Value_Score', ascending=True).head(10)
            st.dataframe(worst_deals, use_container_width=True)
        else:
            st.info("Value Score metric not found.")
        