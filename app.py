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
# Load the model and components
def load_model():
    with open('/Users/mohammadsalem/Desktop/S2K/StreamLit-Next-Purchase-ML/next_purchase_model.pkl', 'rb') as f:
        model_components = pickle.load(f)
    return model_components

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

print(feature_map)

# App title and description
st.title("Next Purchase Predictor")
st.markdown("""
This app predicts what a customer is likely to purchase next based on their purchase history.
""")


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
