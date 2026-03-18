# import packages
# import packages
import streamlit as st
import pandas as pd
import re
import os
import string  # <--- Add this right here!
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
from dotenv import load_dotenv


def clean_text(text):
    """
    Clean text by removing punctuation, lowercasing, and stripping whitespace.
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: The cleaned text
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase
    text = text.lower()
    # Strip whitespace
    text = text.strip()
    return text

# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, "..", "..", "data", "customer_reviews.csv")
    return csv_path


st.title("Hello, GenAI!")
st.write("This is your GenAI-powered data processing app.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")



with col2:
    if st.button("🧹 Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned!")
        else:
            st.warning("Please ingest the dataset first.")


if "df" in st.session_state:
    # Product filter dropdown
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Dataset Preview")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)
    
    st.subheader("Sentiment score by product")
    grouped = st.session_state["df"].groupby(["PRODUCT"])["SENTIMENT_SCORE"].mean()
    st.bar_chart(grouped)

     # Create Altair histogram using add_params instead of add_selection
    interval = alt.selection_interval()
    chart = alt.Chart(filtered_df).mark_bar().add_params(
        interval 
    ).encode(
        alt.X("SENTIMENT_SCORE:Q", bin=alt.Bin(maxbins=10), title="Sentiment Score"),
        alt.Y("count():Q", title="Frequency"),
        tooltip=["count():Q"]
    ).properties(
        width=600,
        height=400,
        title="Distribution of Sentiment Scores"
    )
    st.altair_chart(chart, use_container_width=True)

    # Create matplotlib histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(filtered_df["SENTIMENT_SCORE"], bins=10, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Sentiment Scores')
    st.pyplot(fig)

    # Create Plotly histogram
    fig = px.histogram(
        filtered_df, 
        x="SENTIMENT_SCORE", 
        nbins=10,
        title="Distribution of Sentiment Scores",
        labels={"SENTIMENT_SCORE": "Sentiment Score", "count": "Frequency"}
    )
    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig,width='stretch')

