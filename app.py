import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine, inspect
import sqlite3
import pandas as pd
from langchain_groq import ChatGroq

# Set up the page configuration
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")

# App title
st.title("🦜 LangChain: Chat with SQL DB")

# Define constants for database options
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

# Sidebar for database selection and connection details
with st.sidebar:
    st.header("Database Configuration")
    radio_opt = ["Use SQLLite 3 Database - Student.db", "Connect to your MySQL Database"]
    selected_opt = st.radio("Choose the DB which you want to chat", options=radio_opt)

    if radio_opt.index(selected_opt) == 1:
        db_uri = MYSQL
        mysql_host = st.text_input("MySQL Host", placeholder="localhost")
        mysql_user = st.text_input("MySQL User", placeholder="root")
        mysql_password = st.text_input("MySQL Password", type="password")
        mysql_db = st.text_input("MySQL Database", placeholder="test_db")
    else:
        db_uri = LOCALDB

    api_key = st.text_input("GRoq API Key", type="password", placeholder="Enter your GRoq API Key")

    # Button to clear message history
    if st.button("Clear Message History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Ensure API key is provided
if not api_key:
    st.error("Please add the GRoq API key.")
    st.stop()

# Initialize LLM model
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

# Configure the database based on user input
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

# Set up the SQL agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Initialize message history if not present
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display database schema if checkbox is selected
if st.sidebar.checkbox("Show Database Schema"):
    st.sidebar.subheader("Database Schema")
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    for table in tables:
        st.sidebar.write(f"**Table:** {table}")
        columns = [col['name'] for col in inspector.get_columns(table)]
        st.sidebar.write(f"**Columns:** {', '.join(columns)}")

# Display query history if checkbox is selected
if st.sidebar.checkbox("Show Query History"):
    st.sidebar.subheader("Query History")
    for msg in st.session_state.messages:
        st.sidebar.write(f"**{msg['role'].capitalize()}:** {msg['content']}")

# Main content area for chat messages
st.markdown("---")
st.subheader("Chat with Database")

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**User:** {msg['content']}", unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        st.markdown(f"**Assistant:** {msg['content']}", unsafe_allow_html=True)

# Input field for user queries
user_query = st.text_input("Ask anything from the database", placeholder="Type your query here...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.markdown(f"**User:** {user_query}", unsafe_allow_html=True)

    with st.spinner("Getting response..."):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f"**Assistant:** {response}", unsafe_allow_html=True)

    # Export query results as CSV
    if st.sidebar.checkbox("Export Results"):
        df = pd.DataFrame([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(label="Download Results as CSV", data=csv, file_name="query_results.csv", mime="text/csv")
