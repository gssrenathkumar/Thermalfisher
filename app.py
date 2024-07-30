import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Paths to your models and CSV file
model_paths = {
    'CatBoost': 'pickle_file/best_catboost_model.pkl',
    'LightGBM': 'pickle_file/best_lgbm_model.pkl',
    # 'Random Forest': 'pickle_file/best_rf_model.pkl'
}
csv_file_path = 'cluster_family_reshare.csv'

# Load all models
models = {name: pickle.load(open(path, 'rb')) for name, path in model_paths.items()}

# Load the CSV file
df_clusters = pd.read_csv(csv_file_path)

# List of old_node_type values
old_node_types = df_clusters['Instance Name'].tolist()

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

# Extract values from the CSV file based on selected old_node_type
selected_row = df_clusters[df_clusters['Instance Name'] == old_node_type]
if not selected_row.empty:
    old_core_count = selected_row['core_count'].values[0]
    old_memory_mb = selected_row['memory_mb'].values[0]
    old_cost = selected_row['On Demand Hourly Cost'].values[0]
else:
    old_core_count = 0
    old_memory_mb = 0
    old_cost = 0

# Input fields for core count, memory, and cost (disabled for editing)
old_core_count = st.number_input("Old Core Count", value=old_core_count, disabled=True)
old_memory_mb = st.number_input("Old Memory MB", value=old_memory_mb, disabled=True)
old_cost = st.number_input("Old Cost", value=old_cost, disabled=True)

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
    
    # Extract details for the predicted cluster
    predicted_row = df_clusters[df_clusters['Instance Name'] == cluster_name]
    if not predicted_row.empty:
        new_core_count = predicted_row['core_count'].values[0]
        new_memory_mb = predicted_row['memory_mb'].values[0]
        new_cost = predicted_row['On Demand Hourly Cost'].values[0] + predicted_row['DBU / Hour'].values[0]
        
        st.write(f"New Core Count: {new_core_count}")
        st.write(f"New Memory MB: {new_memory_mb}")
        st.write(f"New Cost: {new_cost}")
    # Display cost details in a table
        cost_data = {
            'Cost Type': ['AWS Cost', 'Databricks Cost'],
            'Cost': [predicted_row['On Demand Hourly Cost'].values[0], predicted_row['DBU / Hour'].values[0]]
        }
        cost_df = pd.DataFrame(cost_data)
        st.table(cost_df)
        # Display the comparison table without index
        st.table(comparison_df.style.hide(axis='index'))

        # Calculate the difference and percentage saved
        difference = old_cost - new_cost
        percentage_saved = (difference / old_cost) * 100 if old_cost != 0 else 0
        
        # Create a DataFrame for the comparison table
        comparison_data = {
            'Old Cost ($)': [old_cost],
            'New Cost ($)': [new_cost],
            'Difference ($)': [difference],
            'Cost Saved (%)': [percentage_saved]
        }
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display the comparison table
        st.table(comparison_df)
    else:
        st.write("Details for the predicted cluster are not available.")
