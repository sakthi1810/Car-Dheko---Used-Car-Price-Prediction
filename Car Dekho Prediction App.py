import pickle
import pandas as pd
import streamlit as st

# Page settings
st.set_page_config(page_title="Car Dekho - Price Prediction", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .header-container {
            background-color:rgb(248, 250, 250);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .header-title {
            font-size: 36px;
            font-weight: bold;
            color: #007bff;
            text-align: center;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 14px;
            color: #6c757d;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header-container">
        <div class="header-title">Car Dekho - Price Prediction 🚗</div>
    </div>
""", unsafe_allow_html=True)

# Load data
try:
    df = pd.read_csv(r"C:\Users\DELL\Desktop\Sakthi_files\P_Car\env\Scripts\Car_Dheko.csv")
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Encoding mappings
encoding_maps = {
    'Fuel_Type': {ft: idx for idx, ft in enumerate(df['Fuel_Type'].unique())},
    'Transmission_Type': {tt: idx for idx, tt in enumerate(df['Transmission_Type'].unique())},
    'Location': {loc: idx for idx, loc in enumerate(df['Location'].unique())},
    'Car_Model': {cm: idx for idx, cm in enumerate(df['Car_Model'].unique())},
}

# Feature selection
st.write("### Select Car Features")
col1, col2 = st.columns([2, 1])

with col1:
    Ft = st.selectbox("Fuel Type", df['Fuel_Type'].unique())
    Tr = st.selectbox("Transmission Type", df['Transmission_Type'].unique())
    Owner = st.selectbox("Number of Owners", df['No_of_Owners'].unique())
    Car_Model = st.selectbox("Car Model", df['Car_Model'].unique())
    Car_Produced_Year = st.selectbox("Car Produced Year", sorted(df['Car_Produced_Year'].unique()))
    Km = st.slider("Kilometers Driven", min_value=int(df['Kilometers_Driven'].min()), 
                   max_value=int(df['Kilometers_Driven'].max()), step=500)
    Engine = st.slider("Engine CC", min_value=int(df['Engine_CC'].min()), 
                       max_value=int(df['Engine_CC'].max()), step=100)
    ML = st.number_input("Mileage (kmpl)", min_value=int(df['Mileage(kmpl)'].min()), 
                         max_value=int(df['Mileage(kmpl)'].max()), step=1)
    seats = st.selectbox("Number of Seats", sorted(df['No_of_Seats'].unique()))
    city = st.selectbox("Location", df['Location'].unique())

with col2:
    st.write("#### Prediction")
    if st.button("Predict"):
        try:
            with open(r"C:\Users\DELL\Desktop\Sakthi_files\P_Car\env\Scripts\RandomForest_model.pkl", 'rb') as file:
                pipeline = pickle.load(file)
            
            # Get expected feature names from the model
            expected_features = pipeline.feature_names_in_

            # Prepare the input data with consistent encoding
            input_data = pd.DataFrame({
                'Fuel_Type': [encoding_maps['Fuel_Type'][Ft]],
                'Transmission_Type': [encoding_maps['Transmission_Type'][Tr]],
                'No_of_Owners': [Owner],
                'Car_Model': [encoding_maps['Car_Model'][Car_Model]],
                'Car_Produced_Year': [Car_Produced_Year],
                'Kilometers_Driven': [Km],
                'Mileage(kmpl)': [ML],
                'Engine_CC': [Engine],
                'No_of_Seats': [seats],
                'Location': [encoding_maps['Location'][city]]
            })

            # Align input data with expected features
            input_data = input_data.reindex(columns=expected_features, fill_value=0)

            # Prediction
            prediction = pipeline.predict(input_data)
            st.success(f"The predicted price of the car is ₹{round(prediction[0], 2):,}.")
        except KeyError as ke:
            st.error(f"Encoding failed: {ke}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Footer Section
st.markdown("""
    <div class="footer">
        © 2025 Car Dekho | All rights reserved |
        Project presented by Ms. Sakthi (Aspiring Data Scientist)
    </div>
""", unsafe_allow_html=True)
