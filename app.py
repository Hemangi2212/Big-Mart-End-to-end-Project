import streamlit as st
import pandas as pd


st.set_page_config(page_title="BigMart Sales Predictor", layout="wide", page_icon="🛒")

# Load model
import pickle
import streamlit as st

@st.cache_resource
def load_model():
    with open("bigmart_best_model.pkl", "rb") as f:
        obj = pickle.load(f)

    # If it's a tuple, find the actual model safely
    if isinstance(obj, tuple):
        for item in obj:
            if hasattr(item, "predict"):
                return item
        raise ValueError("No model with predict() found in PKL file")

    # If it's a dict
    if isinstance(obj, dict):
        for v in obj.values():
            if hasattr(v, "predict"):
                return v
        raise ValueError("No model with predict() found in PKL dictionary")

    # Otherwise assume it's already a model
    return obj

model = load_model()




# ---------------- HEADER ----------------
st.title("🛍️ BigMart Sales Prediction Dashboard")
st.markdown("### Predict the sales of a product at different BigMart outlets")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📦 Input Product Details")
st.sidebar.write("Fill in the details below:")

# Input fields
Item_Weight = st.sidebar.number_input("Item Weight (kg)", min_value=1.0, max_value=50.0, value=10.0)
Item_Fat_Content = st.sidebar.selectbox("Item Fat Content", ["Low Fat", "Regular", "LF", "low fat"])
Item_Visibility = st.sidebar.number_input("Item Visibility", min_value=0.0, max_value=1.0, value=0.05)
Item_Type = st.sidebar.selectbox("Item Type", [
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
    "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Canned",
    "Health and Hygiene", "Breads", "Hard Drinks", "Others"
])
Item_MRP = st.sidebar.number_input("Item MRP (₹)", min_value=20.0, max_value=500.0, value=150.0)
Outlet_Size = st.sidebar.selectbox("Outlet Size", ["Small", "Medium", "High"])
Outlet_Location_Type = st.sidebar.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
Outlet_Type = st.sidebar.selectbox("Outlet Type", [
    "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"
])
Outlet_Establishment_Year = st.sidebar.number_input("Outlet Establishment Year", min_value=1980, max_value=2025, value=2000)

# ---------------- DATAFRAME CREATION ----------------
input_data = pd.DataFrame({
    'Item_Weight': [Item_Weight],
    'Item_Fat_Content': [Item_Fat_Content],
    'Item_Visibility': [Item_Visibility],
    'Item_Type': [Item_Type],
    'Item_MRP': [Item_MRP],
    'Outlet_Size': [Outlet_Size],
    'Outlet_Location_Type': [Outlet_Location_Type],
    'Outlet_Type': [Outlet_Type],
    'Outlet_Establishment_Year': [Outlet_Establishment_Year]
})

# ---------------- PREDICTION ----------------
st.markdown("### 📊 Prediction Output")
if st.button("🚀 Predict Sales"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Item Outlet Sales: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ---------------- INFO ----------------
st.markdown("---")
st.markdown("#### ℹ️ About")
st.info(
    "This app predicts the sales of retail items at BigMart outlets using a trained machine learning model. "
    "Adjust the inputs in the sidebar to see how different factors affect sales."
)
st.caption("Developed by **Hema Ransing** | Powered by Streamlit & Machine Learning 🤖")
