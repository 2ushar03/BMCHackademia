import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
import time

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")

# Load dataset
df = pd.read_csv("merged_data.csv")  # Replace with actual dataset file

# Create LangChain agent for CSV Analysis
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# Define Static Information
STATIC_INFO = """
We evaluated multiple classification models on a dataset with 14,876 samples to compare their accuracy, precision, and recall.

Top-Performing Models (100% Accuracy):
- Random Forest (Accuracy: 1.000)
- Gradient Boosting (Accuracy: 1.000)
- AdaBoost (Accuracy: 1.000)
- XGBoost (Accuracy: 1.000)
- LightGBM (Accuracy: 1.000)
- Decision Tree (Accuracy: 1.000)

High-Performing Models (Above 99% Accuracy):
- K-Nearest Neighbors (Accuracy: 0.9996)
  - Precision (Class 1): 0.93
  - Recall (Class 1): 1.00
- Naïve Bayes (Accuracy: 0.9972)
  - Precision (Class 1): 0.67
  - Recall (Class 1): 1.00
"""

# Create a Prompt Template for Static Info
prompt = PromptTemplate.from_template(
    "You are a helpful assistant that strictly answers based on the given information:\n{info}\n\nUser: {question}\nAssistant:"
)

# Define a Runnable Chain for Static Chatbot
chat_chain = RunnableMap({"question": lambda x: x, "info": lambda _: STATIC_INFO}) | prompt | llm

# Streamlit Layout
st.set_page_config(page_title="AI Chatbot & Data Analysis", layout="wide")
st.title("📊 Data Insights & AI Chatbot 🤖")

# Power BI Dashboard in Main Layout
st.subheader("📈 Power BI Dashboard")
power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiODU3ZTQ3ZWItNWFmMi00MDA2LThkZWUtZTE0ZTBjNjY5NGE5IiwidCI6IjhjNzhjMTIyLWY3ODEtNDUwMC05YzJhLWY2NDVhNzYyODFmNSJ9"
st.markdown(
    f'<iframe title="Power BI Report" width="100%" height="600px" src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>',
    unsafe_allow_html=True
)

# Dropdown for Chatbot Selection
chatbot_choice = st.selectbox("Choose Chatbot:", ["📌 Static Info Bot", "📊 CSV Analysis Bot"])

# Predefined Questions (Scrollable)
static_questions = [
    "Which model has the highest accuracy?",
    "What are the precision and recall values for Naïve Bayes?",
    "How does Gradient Boosting perform?",
    "What is the accuracy of the Decision Tree?",
    "Show a comparison between Random Forest and XGBoost.",
]

csv_questions = [
    "What is the average value of column X?"
]


if chatbot_choice == "📌 Static Info Bot":
    st.subheader("💡 Suggested Questions")
    selected_question = st.selectbox("Select a predefined question:", ["Type your own"] + static_questions)

elif chatbot_choice == "📊 CSV Analysis Bot":
    selected_question = 1

user_query = ""
if selected_question == 1 or selected_question!="Type your own":
    user_query = selected_question

# Chatbot Response Handling
if user_query:
    if chatbot_choice == "📌 Static Info Bot":
        with st.spinner("🤖 Thinking... Please wait"):
            try:
                response = chat_chain.invoke({"question": user_query})
                clean_response = response.content
                st.success("### 🤖 AI Response:")
                st.write(clean_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif chatbot_choice == "📊 CSV Analysis Bot":
        try:
            processed_data = pd.read_csv('org_data_up.csv')  # Processed data (features and target)

            # Ensure no leading/trailing spaces in column names
            processed_data.columns = processed_data.columns.str.strip()

        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            st.stop()

        # Define the total number of rows
        total_rows = 74378

        # User input for conditions
        st.title("📈 Predicting Printer Retention: Probability Calculation Tool")

        # New list of columns
        columns = [
            'index_x', 'Order_ID', 'Customer_ID', 'Product_ID', 'Sales', 'Quantity',
            'Discount', 'Profit', 'index_y', 'Product_Name', 'Customer Name',
            'Segment', 'Country', 'City', 'State', 'Postal Code', 'Order_Returned',
            'Order ID', 'Is_Printer_Product', 'Order_Year', 'Order_Month',
            'Order_Day', 'Ship_Year', 'Ship_Month', 'Ship_Day', 'Profit_Margin',
            'Ship Mode_First Class', 'Ship Mode_Same Day', 'Ship Mode_Second Class',
            'Ship Mode_Standard Class', 'Product_Category_Furniture',
            'Product_Category_Office Supplies', 'Product_Category_Technology',
            'Product_SubCategory_Accessories', 'Product_SubCategory_Appliances',
            'Product_SubCategory_Art', 'Product_SubCategory_Binders',
            'Product_SubCategory_Bookcases', 'Product_SubCategory_Chairs',
            'Product_SubCategory_Copiers', 'Product_SubCategory_Envelopes',
            'Product_SubCategory_Fasteners', 'Product_SubCategory_Furnishings',
            'Product_SubCategory_Labels', 'Product_SubCategory_Machines',
            'Product_SubCategory_Paper', 'Product_SubCategory_Phones',
            'Product_SubCategory_Storage', 'Product_SubCategory_Supplies',
            'Product_SubCategory_Tables', 'Region_Central', 'Region_East',
            'Region_South', 'Region_West', 'Order_Weekday', 'Ship_Weekday',
            'Printer Kept'
        ]

        # Initialize the query filter (empty list to store conditions)
        query_filter = []

        # Ask the user for conditions on columns
        num_conditions = st.number_input("Enter the number of conditions to apply:", min_value=1, max_value=5, value=1)

        for i in range(num_conditions):
            column = st.selectbox(f"Select column for condition {i + 1}:", columns, key=f"column_{i}")
            operator = st.selectbox(f"Select condition operator for {column}:", ['=', '>', '<', '>=', '<='], key=f"operator_{i}")
            value = st.text_input(f"Enter value for {column} ({operator}):", key=f"value_{i}")
            query_filter.append((column, operator, value))

        # Function to apply conditions and calculate probability
        def calculate_probability():
            filtered_data = processed_data.copy()

            # Loop through all the conditions and apply them to filter the data
            for column, operator, value in query_filter:
                try:
                    value = float(value)
                    is_numeric = True
                except ValueError:
                    is_numeric = False

                # Apply condition based on whether the column is numeric
                if is_numeric:
                    if operator == '=':
                        filtered_data = filtered_data[filtered_data[column] == value]
                    elif operator == '>':
                        filtered_data = filtered_data[filtered_data[column] > value]
                    elif operator == '<':
                        filtered_data = filtered_data[filtered_data[column] < value]
                    elif operator == '>=':
                        filtered_data = filtered_data[filtered_data[column] >= value]
                    elif operator == '<=':
                        filtered_data = filtered_data[filtered_data[column] <= value]
                else:
                    if operator == '=':
                        filtered_data = filtered_data[filtered_data[column].str.strip().str.lower() == value.strip().lower()]
                    elif operator == '>':
                        filtered_data = filtered_data[filtered_data[column].str.strip().str.lower() > value.strip().lower()]
                    elif operator == '<':
                        filtered_data = filtered_data[filtered_data[column].str.strip().str.lower() < value.strip().lower()]
                    elif operator == '>=':
                        filtered_data = filtered_data[filtered_data[column].str.strip().str.lower() >= value.strip().lower()]
                    elif operator == '<=':
                        filtered_data = filtered_data[filtered_data[column].str.strip().str.lower() <= value.strip().lower()]

            # Count how many times 'Printer Kept' is 1
            printer_kept_count = filtered_data['Printer Kept'].sum()

            # More readable output
            st.write("### Results after applying the filter:")
            st.write(f"**Number of rows where 'Printer Kept' = 1:** {printer_kept_count}")

            if printer_kept_count > 0:
                probability = printer_kept_count / total_rows
                st.write(f"**The Probability of 'Printer Kept' being 1:** {probability:.4f}")
            else:
                st.write("**The Probability of 'Printer Kept' being 1:** 0.0000")

            return printer_kept_count, probability if printer_kept_count > 0 else 0.0

        if st.button("Calculate Probability"):
            count, probability = calculate_probability()

# Footer
st.markdown("---")
st.markdown("📌 Built with Streamlit, Power BI & LangChain")



