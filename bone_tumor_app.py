import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

# Function to load or initialize the model
@st.cache_resource
def load_model():
    # Load dataset
    df = pd.read_csv("bone_tumor_dataset.csv")
    
    # Preprocessing
    # Convert target variable
    df['Status (NED, AWD, D)'] = df['Status (NED, AWD, D)'].map({'NED': 0, 'AWD': 1, 'D': 2})
    
    # Separate features and target
    X = df.drop(columns=['Status (NED, AWD, D)'])
    y = df['Status (NED, AWD, D)']
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Create full pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(max_iter=1000))])
    
    # Train model
    model.fit(X, y)
    
    return model, numeric_features, categorical_features, X, y

# Function to save prediction history
def save_prediction(features, prediction):
    history = load_history()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert prediction code to label
    prediction_label = {0: 'NED', 1: 'AWD', 2: 'D'}.get(prediction, 'Unknown')
    
    new_entry = {
        'timestamp': timestamp,
        'prediction': prediction_label,
        **features
    }
    
    history.append(new_entry)
    with open('prediction_history.pkl', 'wb') as f:
        pickle.dump(history, f)

# Function to load prediction history
def load_history():
    try:
        with open('prediction_history.pkl', 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return []

# Function to clear history
def clear_history():
    with open('prediction_history.pkl', 'wb') as f:
        pickle.dump([], f)

# Main app function
def main():
    st.title("Bone Tumor Status Prediction")
    
    try:
        # Load model and feature information
        model, numeric_features, categorical_features, X, y = load_model()
        
        # Create form for input
        with st.form("prediction_form"):
            st.header("Patient Information")
            
            input_features = {}
            
            # Numeric inputs
            if numeric_features:
                st.subheader("Numeric Features")
                cols = st.columns(2)
                for i, feature in enumerate(numeric_features):
                    with cols[i % 2]:
                        input_features[feature] = st.number_input(
                            f"{feature}",
                            value=0.0,
                            step=0.1,
                            format="%.2f"
                        )
            
            # Categorical inputs
            if categorical_features:
                st.subheader("Categorical Features")
                df = pd.read_csv("bone_tumor_dataset.csv")
                for feature in categorical_features:
                    unique_values = df[feature].dropna().unique().tolist()
                    if not unique_values:  # Handle empty categories
                        unique_values = ['missing']
                    
                    input_features[feature] = st.selectbox(
                        f"{feature}",
                        options=unique_values
                    )
            
            submitted = st.form_submit_button("Predict Status")
        
        # Prediction and results section
        if submitted:
            # Prepare input data
            input_df = pd.DataFrame([input_features])
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                # Map prediction code to label
                status_map = {0: 'NED (No Evidence of Disease)', 
                            1: 'AWD (Alive With Disease)', 
                            2: 'D (Deceased)'}
                prediction_label = status_map.get(prediction, 'Unknown')
                
                # Save to history
                save_prediction(input_features, prediction)
                
                # Display results
                st.success("Prediction Complete!")
                st.subheader("Results")
                
                # Prediction
                st.markdown(f"**Predicted Status:** {prediction_label}")
                
                # Probabilities
                st.markdown("**Prediction Probabilities:**")
                proba_df = pd.DataFrame({
                    'Status': ['NED', 'AWD', 'D'],
                    'Probability': prediction_proba
                })
                st.bar_chart(proba_df.set_index('Status'))
                
                # Show model evaluation metrics
                st.subheader("Model Performance Metrics")
                
                # Get cross-validated predictions for metrics
                y_pred = cross_val_predict(model, X, y, cv=5)
                
                # Confusion matrix
                st.markdown("**Confusion Matrix:**")
                cm = confusion_matrix(y, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['NED', 'AWD', 'D'], 
                            yticklabels=['NED', 'AWD', 'D'],
                            ax=ax)
                st.pyplot(fig)
                
                # Classification report
                st.markdown("**Classification Report:**")
                report = classification_report(y, y_pred, target_names=['NED', 'AWD', 'D'], output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.error("Please check your input values and try again.")
    
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.error("Please make sure 'bone_tumor_dataset.csv' exists in the correct format.")
    
    # History section
    st.sidebar.header("Prediction History")
    
    if st.sidebar.button("View History"):
        history = load_history()
        if history:
            st.subheader("Prediction History")
            history_df = pd.DataFrame(history)
            st.dataframe(history_df)
        else:
            st.info("No prediction history found.")
    
    if st.sidebar.button("Clear History"):
        clear_history()
        st.sidebar.success("Prediction history cleared.")

if __name__ == "__main__":
    main()