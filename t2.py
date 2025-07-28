# --- Installation ---
# Before running this script, ensure you have all required packages installed.
# Open your terminal or command prompt and run the following command:
# pip install streamlit pandas chardet langchain-experimental langchain-community langchain-google-genai openai google-generativeai

import streamlit as st
import pandas as pd
import chardet
import os
from io import BytesIO
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import traceback
import hashlib

# --- App Configuration ---
st.set_page_config(
    page_title="Compliance AI Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    /* Main App Styling */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Card Styling */
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    div[data-testid="stHorizontalBlock"] .stButton>button {
        background-color: #e7eaf0;
        color: #333;
        border: 1px solid #d1d5db;
        width: 100%;
    }
    div[data-testid="stVerticalBlock"] .stButton>button[kind="primary"] {
        background-color: #0068c9;
        color: white;
    }
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    .css-1d391kg .stRadio > label {
        font-weight: 600;
        color: #1a5276;
    }
    /* Headings & Text */
    h1, h2, h3 {
        color: #1a5276;
    }
    h1 {
        display: flex;
        align-items: center;
    }
    h1 .icon {
        font-size: 2.5rem;
        margin-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- AI Model Initialization ---
@st.cache_resource(ttl=3600)
def get_openai_model(api_key: str):
    """Initializes and returns the ChatOpenAI model."""
    try:
        return ChatOpenAI(openai_api_key=api_key, temperature=0, model="gpt-4o")
    except Exception as e:
        st.error(f"Failed to initialize OpenAI model: {e}")
        return None

@st.cache_resource(ttl=3600)
def get_google_model(api_key: str):
    """Initializes and returns the Google Gemini model."""
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    except Exception as e:
        st.error(f"Failed to initialize Google model: {e}")
        return None

# --- Direct Data Analysis with Pandas ---
def get_iam_summary(df: pd.DataFrame):
    """Performs a direct analysis of the IAM DataFrame."""
    total_users = len(df)
    # Normalize column names for reliable access
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    inactive_users = df[df['Unused_for_45_Days'] == 'Yes'].shape[0]
    unrotated_keys = df[df['Access_Keys_Rotated_(90_Days)'] == 'No'].shape[0]
    no_mfa = df[df['MFA_Enabled'] == 'No'].shape[0]
    high_risk = df[(df['Access_Keys_Rotated_(90_Days)'] == 'No') & (df['MFA_Enabled'] == 'No')]
    
    summary_data = {
        "Total Users": total_users,
        "Inactive Users": f"{inactive_users} ({inactive_users/total_users:.2%})",
        "Users with Unrotated Keys": f"{unrotated_keys} ({unrotated_keys/total_users:.2%})",
        "Users without MFA": f"{no_mfa} ({no_mfa/total_users:.2%})",
        "High-Risk Accounts": high_risk['IAM_User'].tolist()
    }
    return summary_data

def get_s3_summary(df: pd.DataFrame):
    """Performs a direct analysis of the S3 DataFrame."""
    total_buckets = len(df)
    # Normalize column names
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    public_buckets = df[df['Public_Access_Status'] == 'Public'].shape[0]
    no_lifecycle = df[df['Lifecycle_Policy_Status'] == 'Disabled'].shape[0]
    no_encryption = df[df['Encryption_Status'] == 'Disabled'].shape[0]
    no_logging = df[df['Server_Access_Logging_Status'] == 'Disabled'].shape[0]

    summary_data = {
        "Total Buckets": total_buckets,
        "Public Buckets": f"{public_buckets} ({public_buckets/total_buckets:.2%})",
        "Buckets without Lifecycle Policy": f"{no_lifecycle} ({no_lifecycle/total_buckets:.2%})",
        "Buckets without Encryption": f"{no_encryption} ({no_encryption/total_buckets:.2%})",
        "Buckets without Server Logging": f"{no_logging} ({no_logging/total_buckets:.2%})",
    }
    return summary_data

def get_generic_summary(df: pd.DataFrame):
    """Performs a generic analysis of any CSV DataFrame."""
    num_rows, num_cols = df.shape
    column_names = df.columns.tolist()
    data_types = df.dtypes.to_dict()
    missing_values = df.isnull().sum().to_dict()

    summary_data = {
        "Number of Rows": num_rows,
        "Number of Columns": num_cols,
        "Column Names": column_names,
        "Data Types": {k: str(v) for k, v in data_types.items()},
        "Missing Values per Column": {k: v for k, v in missing_values.items() if v > 0}
    }
    return summary_data

# --- Main Logic ---
def process_csv_question_logic(user_question: str, report_type: str, csv_file_path: str, provider: str, api_key: str):
    """Core logic to select a model and process a CSV question."""
    model = None
    if provider == "OpenAI":
        model = get_openai_model(api_key)
    elif provider == "Google Gemini":
        model = get_google_model(api_key)

    if model is None:
        return f"AI model for {provider} could not be initialized. Please check your API key."

    if not os.path.exists(csv_file_path):
        return f"Error: CSV file not found. Please re-upload."

    # --- New logic branch for summary reports ---
    if "Summary" in report_type:
        try:
            with st.spinner("Analyzing full dataset..."):
                df = pd.read_csv(csv_file_path)
                summary_data = {}
                if "IAM" in report_type:
                    summary_data = get_iam_summary(df)
                elif "S3" in report_type:
                    summary_data = get_s3_summary(df)
                elif "CSV" in report_type:
                    summary_data = get_generic_summary(df)
            
            with st.spinner(f"{provider} is formatting the report..."):
                # The AI now only formats the pre-calculated data
                formatting_prompt = f"""
                You are a reporting assistant. Your task is to format the following data into a professional and easy-to-read report.
                Use markdown for formatting, including bold text, bullet points, and headers.
                Provide brief, actionable recommendations based on the data if it is a compliance report, otherwise just summarize the data's structure.
                
                Data to format:
                {summary_data}
                """
                response = model.invoke(formatting_prompt)
                return response.content

        except Exception as e:
            st.error(f"An error occurred during direct analysis: {e}")
            return f"An error occurred: {e}\n\n{traceback.format_exc()}"

    # --- Original logic for query-based questions ---
    else:
        try:
            # SECURITY WARNING: allow_dangerous_code=True is a security risk.
            # It should be disabled in production environments.
            agent = create_csv_agent(model, csv_file_path, verbose=True, allow_dangerous_code=True)
            with st.spinner(f"{provider} is thinking..."):
                response = agent.invoke({"input": user_question})
                return response.get('output', "No response generated.")
        except Exception as e:
            st.error(f"An error occurred during AI analysis: {e}")
            return f"An error occurred: {e}\n\n{traceback.format_exc()}"


# --- Sidebar for Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    st.subheader("1. Select AI Provider")
    ai_provider = st.radio(
        "Choose the AI model provider:",
        ("OpenAI", "Google Gemini"),
        key="ai_provider",
        horizontal=True,
    )
    st.markdown("---")

    st.subheader("2. Enter API Key")
    api_key = ""
    if ai_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key", help="Starts with 'sk-...'")
    elif ai_provider == "Google Gemini":
        api_key = st.text_input("Google API Key", type="password", key="google_api_key", help="Also known as a Gemini API Key.")
    
    if not api_key:
        st.warning(f"Please enter your {ai_provider} API key to proceed.")
    st.markdown("---")
    
    st.subheader("3. Select Report Type")
    report_type = st.radio(
        "Choose the type of report for analysis:",
        options=["IAM Query", "IAM Summary", "S3 Query", "S3 Summary", "CSV Query", "CSV Summary"],
        key="report_type"
    )
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.9rem; color: #808080;'>
            <p>👨‍💻 Developed by <strong>Sameer</strong><br>July 29, 2025</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Main Application UI ---
st.markdown("<h1><span class='icon'>🛡️</span>Compliance AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("Visually analyze your AWS compliance posture in seconds. This tool leverages AI to transform your raw IAM and S3 reports into attractive, actionable insights. Upload any CSV to get started!")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Step 1: Upload Your Report")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload your compliance report or any other data in CSV format."
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file and api_key:
    file_content = uploaded_file.getvalue()
    
    @st.cache_data(show_spinner="Processing file...")
    def load_and_save_csv(content_bytes, original_filename):
        os.makedirs("Server", exist_ok=True)
        file_hash = hashlib.md5(content_bytes).hexdigest()
        temp_csv_path = os.path.join("Server", f"{os.getpid()}_{file_hash}_{original_filename}")
        with open(temp_csv_path, "wb") as f:
            f.write(content_bytes)
        return temp_csv_path

    temp_csv_path = load_and_save_csv(file_content, uploaded_file.name)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"Step 2: Ask a Question about the {report_type} Report")

    def set_query_text(query):
        st.session_state.user_question_text_input = query

    if 'user_question_text_input' not in st.session_state:
        st.session_state.user_question_text_input = ""

    # For Summary reports, the question is pre-filled.
    is_summary = "Summary" in report_type
    default_question = "Generate a full compliance summary." if is_summary else ""
    
    user_question = st.text_area(
        "Enter your question:",
        value=default_question,
        placeholder="e.g., 'How many rows are there?' or 'List all IAM users without MFA'",
        key="user_question_text_input",
        height=120,
        disabled=is_summary # Disable text area for summary reports
    )

    # Quick Queries are only shown for Query reports
    if not is_summary:
        quick_queries = {
            "IAM Query": ["List all IAM users.", "List users inactive for 45 days.", "List users with unrotated keys."],
            "S3 Query": ["List all S3 buckets.", "Identify buckets without encryption.", "Identify buckets with public access."],
            "CSV Query": ["How many rows are there?", "Show the first 5 rows.", "List all column names."]
        }
        current_queries = quick_queries.get(report_type, [])

        if current_queries:
            st.write("**Or, select a quick query:**")
            num_cols = len(current_queries)
            cols = st.columns(num_cols)
            for i, query in enumerate(current_queries):
                with cols[i]:
                    st.button(query, key=f"quick_query_{i}", on_click=set_query_text, args=(query,))
    
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Get AI Analysis", type="primary", use_container_width=True):
        question_to_ask = user_question.strip() if not is_summary else default_question
        if question_to_ask:
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("💡 AI Assistant's Response")
            response = process_csv_question_logic(question_to_ask, report_type, temp_csv_path, ai_provider, api_key)
            st.markdown(response)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a question or select a quick query.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('</div>', unsafe_allow_html=True)

elif uploaded_file and not api_key:
    st.warning(f"Please enter your {ai_provider} API key in the sidebar to begin analysis.")
