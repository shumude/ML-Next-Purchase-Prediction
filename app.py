# Streamlit app (save as app.py)
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Next Purchase Predictor",
    page_icon="🛒",
    layout="wide"
)
'Amount'
# Load the model and components
@st.cache_resource
def load_model():
    with open('/Users/mohammadsalem/Desktop/S2K/StreamLit-Next-Purchase-ML/next_purchase_model.pkl', 'rb') as f:
        model_components = pickle.load(f)
    return model_components

try:
    model_components = load_model()
    model = model_components['model']
    label_encoder_category = model_components['label_encoder_category']
    label_encoder_next = model_components['label_encoder_next']
    categories = model_components['categories']
    next_categories = model_components['next_categories']
    transition_probs = model_components['transition_probs']
    feature_map = model_components['feature_map']
    feature_columns = model_components['feature_columns']
    scaler = model_components['scaler']
    model_loaded = True
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.error("Please make sure 'next_purchase_model.pkl' is in the same directory as this app.")
    model_loaded = False

# App title and description
st.title("Next Purchase Predictor")
st.markdown("""
This app predicts what a customer is likely to purchase next based on their purchase history and patterns.
Upload your transaction data to train a new model or use the prediction tool with the existing model.
""")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Predict Next Purchase", "Train New Model", "Model Insights"])

if page == "Predict Next Purchase" and model_loaded:
    st.header("Predict Next Purchase")
    
    # Input form
    with st.form("prediction_form"):
        first_category = st.selectbox("Previous Purchase Category", options=categories)
        
        # Only show fields that were used in training
        feature_inputs = {}
        
        if feature_map.get('has_qty', False):
            feature_inputs['Qty'] = st.number_input("Quantity", min_value=1, value=1, step=1)
        
        if feature_map.get('has_sales_price', False):
            feature_inputs['Sales Price'] = st.number_input("Sales Price ($)", min_value=0.0, value=100.0, step=10.0)
        submit_button = st.form_submit_button("Predict Next Purchase")
    
    if submit_button:
        try:
            # Encode the category
            category_encoded = label_encoder_category.transform([first_category])[0]
            
            # Prepare input features
            features = [category_encoded]
            
            # Add and scale numeric features
            numeric_feature_values = []
            for feature in feature_columns:
                if feature == 'Category_Encoded':
                    continue
                    
                if feature in feature_inputs:
                    numeric_feature_values.append(feature_inputs[feature])
            
            # Scale numeric features if we have any
            if numeric_feature_values and hasattr(scaler, 'transform'):
                scaled_values = scaler.transform([numeric_feature_values])[0]
                features.extend(scaled_values)
            
            # Make prediction
            features_array = np.array(features).reshape(1, -1)
            next_category_encoded = model.predict(features_array)[0]
            next_category = label_encoder_next.inverse_transform([next_category_encoded])[0]
            
            # Get prediction probabilities
            probabilities = model.predict_proba(features_array)[0]
            
            # Map probabilities to categories
            prob_dict = {next_categories[i]: prob for i, prob in enumerate(probabilities)}
            
            # Sort by probability (descending)
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Display results
            st.success(f"Predicted next purchase: **{next_category}**")
            
            # Show top 5 recommendations
            st.subheader("Top Recommendations")
            col1, col2 = st.columns([3, 2])
            
            with col1:
                for i, (cat, prob) in enumerate(sorted_probs[:5], 1):
                    confidence_color = "green" if prob > 0.5 else "orange" if prob > 0.2 else "red"
                    st.markdown(f"{i}. {cat} **(Confidence: <span style='color:{confidence_color}'>{prob:.2f}</span>)**", unsafe_allow_html=True)
            
            with col2:
                # Visualize probabilities
                fig, ax = plt.subplots(figsize=(10, 6))
                top_5_cats = [cat for cat, _ in sorted_probs[:5]]
                top_5_probs = [prob for _, prob in sorted_probs[:5]]
                
                # Create bars with gradient colors based on probability
                cmap = plt.cm.get_cmap('Blues')
                colors = [cmap(p) for p in top_5_probs]
                
                bars = sns.barplot(x=top_5_probs, y=top_5_cats, palette=colors, ax=ax)
                
                ax.set_xlabel("Probability")
                ax.set_ylabel("Category")
                ax.set_title("Top 5 Predictions")
                plt.tight_layout()
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("If the category wasn't found in training data, try another category or train a new model.")

elif page == "Train New Model":
    st.header("Train New Model")
    st.write("Upload your transaction data to train a new prediction model with enhanced features.")
    
    # Explain required format
    with st.expander("Data Requirements"):
        st.markdown("""
        Your CSV file should include the following columns:
        - **Category**: The category of the purchase
        - **Next_Order_Category**: The category of the next purchase
        
        Optional but recommended columns:
        - **Qty**: Quantity purchased
        - **Sales Price**: Price per unit
        - **Amount**: Total purchase amount
        
        The more data you have, the better the model will perform.
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display sample of the data
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Check required columns
            required_cols = ['Category', 'Next_Order_Category']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("Your data must include 'Category' and 'Next_Order_Category' columns.")
            else:
                # Train button
                if st.button("Train New Model"):
                    with st.spinner("Training model... This may take a few minutes."):
                        try:
                            # Save uploaded file temporarily
                            df.to_csv("temp_data.csv", index=False)
                            
                            # Import the training function
                            from train_model import train_and_save_model
                            
                            # Train and save the model
                            model_components = train_and_save_model("temp_data.csv")
                            
                            st.success("Model trained and saved successfully!")
                            st.info("To use the new model, please restart the app.")
                            
                            # Show feature importance
                            if hasattr(model_components['model'], 'feature_importances_'):
                                importances = model_components['model'].feature_importances_
                                feature_names = model_components['feature_columns']
                                
                                # Create importance dataframe
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                st.subheader("Feature Importance")
                                st.dataframe(importance_df)
                                
                                # Plot
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                                ax.set_xlabel("Importance")
                                ax.set_ylabel("Feature")
                                st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Error training model: {e}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "Model Insights" and model_loaded:
    st.header("Model Insights")
    
    # Feature importance
    st.subheader("Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(importance_df)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            st.pyplot(fig)
    
    # Transition probabilities
    st.subheader("Purchase Transition Patterns")
    st.write("This heatmap shows the probability of a customer purchasing each category after a specific previous purchase.")
    
    # Plot transition matrix as heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Check the size of the transition matrix
    if len(transition_probs) > 20:
        st.warning("Large number of categories detected. Showing a sample of the transition matrix.")
        # Sample the top categories by frequency
        top_cats = transition_probs.sum(axis=1).sort_values(ascending=False).head(15).index
        transition_sample = transition_probs.loc[top_cats, top_cats]
        sns.heatmap(transition_sample, annot=True, cmap="Blues", fmt=".2f", ax=ax)
    else:
        sns.heatmap(transition_probs, annot=True, cmap="Blues", fmt=".2f", ax=ax)
    
    plt.title("Probability of Next Purchase Category Based on Previous Purchase")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Class distribution
    st.subheader("Category Distribution")
    
    # Category counts from transition matrix
    category_counts = transition_probs.sum(axis=1).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax)
    ax.set_xlabel("Relative Frequency")
    ax.set_ylabel("Category")
    ax.set_title("Purchase Category Distribution")
    st.pyplot(fig)

else:
    st.info("Please load a model to use this feature.")
