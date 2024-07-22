import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Paths to your models
model_paths = {
    'CatBoost': 'pickle_file/best_catboost_model.pkl',
    'LightGBM': 'pickle_file/best_lgbm_model.pkl',
    # 'Random Forest': 'pickle_file/best_rf_model.pkl'
}

# Load all models
models = {name: pickle.load(open(path, 'rb')) for name, path in model_paths.items()}

# List of old_node_type values
old_node_types = [
    'c5a.4xlarge', 'c5.4xlarge', 'c5d.2xlarge', 'c6g.8xlarge', 'c6id.2xlarge',
    'c5d.12xlarge', 'c6g.2xlarge', 'c5d.xlarge', 'c5.xlarge', 'c6g.4xlarge',
    'c5d.9xlarge', 'c6i.xlarge', 'c6gd.2xlarge', 'c6id.4xlarge', 'c6id.xlarge',
    'c5a.2xlarge', 'c6g.xlarge', 'c7gd.xlarge', 'c6gd.xlarge', 'c5a.8xlarge',
    'c7gd.4xlarge', 'c5d.24xlarge', 'c6gd.8xlarge', 'c5.9xlarge', 'c6gd.16xlarge',
    'c5.2xlarge', 'c6i.8xlarge', 'c5.12xlarge', 'c6id.8xlarge', 'c6in.8xlarge',
    'c5d.4xlarge', 'c6gd.12xlarge', 'c4.2xlarge', 'c4.8xlarge', 'c4.4xlarge',
    'c6gd.4xlarge', 'i3.16xlarge', 'i4i.large', 'i4i.4xlarge', 'i4i.xlarge',
    'i3.8xlarge', 'i3.xlarge', 'i3.2xlarge', 'i4i.16xlarge', 'i3.4xlarge',
    'm5d.xlarge', 'm5d.16xlarge', 'm5dn.4xlarge', 'm6g.4xlarge', 'm6gd.4xlarge',
    'm5.2xlarge', 'm7gd.4xlarge', 'm6id.4xlarge', 'm7g.xlarge', 'm6gd.8xlarge',
    'm5n.2xlarge', 'm5d.12xlarge', 'm6id.8xlarge', 'm5.4xlarge', 'm7gd.8xlarge',
    'm6i.2xlarge', 'm5a.4xlarge', 'm6g.8xlarge', 'm7gd.large', 'm6gd.xlarge',
    'm5d.large', 'm5a.24xlarge', 'm5d.24xlarge', 'm6gd.2xlarge', 'm5dn.16xlarge',
    'm7gd.xlarge', 'm6gd.16xlarge', 'm6in.4xlarge', 'm6gd.12xlarge', 'm4.large',
    'm4.4xlarge', 'm7g.large', 'm4.xlarge', 'm6g.large', 'm7g.2xlarge',
    'm6g.xlarge', 'm4.2xlarge', 'm4.16xlarge', 'm5.xlarge', 'm6i.large',
    'm7gd.2xlarge', 'm5.large', 'm5d.2xlarge', 'm5d.8xlarge', 'm6id.xlarge',
    'm5a.large', 'm5.8xlarge', 'm7g.4xlarge', 'm6id.2xlarge', 'm5dn.large',
    'm6gd.large', 'm6g.2xlarge', 'm6id.large', 'm6i.4xlarge', 'm5d.4xlarge',
    'r6gd.xlarge', 'r6id.32xlarge', 'r7gd.4xlarge', 'r6id.8xlarge', 'r6id.4xlarge',
    'r6id.xlarge', 'r6gd.8xlarge', 'r7g.large', 'r7g.xlarge', 'r5d.xlarge',
    'r5dn.xlarge', 'r5dn.4xlarge', 'r5dn.large', 'r7g.2xlarge', 'r4.4xlarge',
    'r6id.2xlarge', 'r5.xlarge', 'r5.large', 'r6gd.large', 'r6gd.4xlarge',
    'r6g.2xlarge', 'r7g.4xlarge', 'r4.2xlarge', 'r5d.2xlarge', 'r6i.4xlarge',
    'r5d.large', 'r6g.xlarge', 'r4.xlarge', 'r6i.2xlarge', 'r5.4xlarge',
    'r5d.8xlarge', 'r7gd.2xlarge', 'r7gd.xlarge', 'r6g.large', 'r6i.xlarge',
    'r6g.4xlarge', 'r6gd.2xlarge', 'r6i.large', 'r4.8xlarge', 'r5.2xlarge',
    'r7gd.8xlarge', 'r7gd.16xlarge', 'r5d.4xlarge'
]

# Mapping from numerical cluster IDs to cluster names
cluster_mapping = {
    28: 'r6g.xlarge',
    25: 'r6g.4xlarge',
    22: 'r5d.large',
    27: 'r6g.large',
    18: 'm6g.large',
    0: 'c5d.2xlarge',
    24: 'r6g.2xlarge',
    23: 'r5d.xlarge',
    15: 'm6g.2xlarge',
    12: 'm5d.xlarge',
    5: 'c6g.4xlarge',
    3: 'c5d.xlarge',
    4: 'c6g.2xlarge',
    19: 'm6g.xlarge',
    2: 'c5d.9xlarge',
    11: 'm5d.large',
    7: 'c6g.xlarge',
    8: 'm5d.2xlarge',
    1: 'c5d.4xlarge',
    16: 'm6g.4xlarge',
    20: 'r5d.2xlarge',
    13: 'm5zn.3xlarge',
    6: 'c6g.8xlarge',
    21: 'r5d.4xlarge',
    9: 'm5d.4xlarge',
    10: 'm5d.8xlarge',
    14: 'm5zn.6xlarge',
    17: 'm6g.8xlarge',
    26: 'r6g.8xlarge'
}

# Function to preprocess input data
def preprocess_input(cpu_user_percent, mem_used_percent, old_node_type, old_core_count, old_memory_mb, old_cost):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'cpu_user_percent': [cpu_user_percent],
        'mem_used_percent': [mem_used_percent],
        'old_node_type': [old_node_type],
        'old_core_count': [old_core_count],
        'old_memory_mb': [old_memory_mb],
        'old_cost': [old_cost]
    })
    
    # Split 'old_node_type' into new columns
    def split_node_type(old_node_type):
        series_node_type = old_node_type[0]
        family_variant = old_node_type[1:].split('.')[0]
        variant_node_type = old_node_type.split('.')[1]
        return pd.Series([series_node_type, family_variant, variant_node_type])
    
    input_data[['series_node_type', 'family_variant', 'variant_node_type']] = input_data['old_node_type'].apply(split_node_type)
    
    # Encode categorical columns
    label_encoder = LabelEncoder()
    input_data['series_node_type'] = label_encoder.fit_transform(input_data['series_node_type'])
    input_data['family_variant'] = label_encoder.fit_transform(input_data['family_variant'])
    
    # Define the desired order for the 'variant_node_type' column
    variant_order = ['large', 'xlarge', '2xlarge', '4xlarge', '8xlarge', '9xlarge', '12xlarge', '16xlarge', '24xlarge', '32xlarge']
    input_data['variant_node_type'] = pd.Categorical(input_data['variant_node_type'], categories=variant_order, ordered=True)
    input_data['variant_node_type'] = input_data['variant_node_type'].cat.codes
    
    input_data['old_node_type'] = input_data['series_node_type'].astype(str) + input_data['family_variant'].astype(str) + '.' + input_data['variant_node_type'].astype(str)
    input_data["old_node_type"] = pd.to_numeric(input_data["old_node_type"], errors='coerce')

    # Drop original 'old_node_type' column
    input_data.drop(['series_node_type', 'family_variant', 'variant_node_type'], axis=1, inplace=True)
    
    return input_data

# Streamlit UI
st.title("Cluster Suitability Prediction")

# Model selection dropdown
selected_model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

# Input fields
cpu_user_percent = st.number_input("CPU User Percent", min_value=0.0, max_value=100.0, value=0.0)
mem_used_percent = st.number_input("Memory Used Percent", min_value=0.0, max_value=100.0, value=0.0)
old_node_type = st.selectbox("Old Node Type", old_node_types)
old_core_count = st.number_input("Old Core Count", min_value=0, value=0)
old_memory_mb = st.number_input("Old Memory MB", min_value=0, value=0)
old_cost = st.number_input("Old Cost", min_value=0.0, value=0.0)

# Predict button
if st.button("Predict"):
    # Preprocess input
    preprocessed_data = preprocess_input(cpu_user_percent, mem_used_percent, old_node_type, old_core_count, old_memory_mb, old_cost)
    # Make prediction
    prediction = selected_model.predict(preprocessed_data)
    
    # Handle prediction output
    if prediction.ndim > 1:  # Check if prediction is a 2D array
        prediction = prediction.flatten()
    
    cluster_id = prediction[0] if len(prediction) > 0 else None
    cluster_name = cluster_mapping.get(cluster_id, 'Unknown Cluster')
    
    st.write(f"Predicted Suitable Cluster: {cluster_name}")
