import psycopg2
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "your_database")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        # Clean up the host URL if it's a Supabase connection
        host = DB_HOST
        if host.startswith("http://"):
            host = host.replace("http://", "")
        if host.startswith("https://"):
            host = host.replace("https://", "")
        if host.endswith("/"):
            host = host[:-1]
        
        st.info(f"Connecting to database at {host}")
        
        conn = psycopg2.connect(
            host=host,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        
        # Provide more detailed error information
        st.error("Please check your database configuration in the .env file:")
        st.code(f"""
        DB_HOST={DB_HOST} (should be hostname only, without http:// or trailing slash)
        DB_NAME={DB_NAME}
        DB_USER={DB_USER}
        DB_PORT={DB_PORT}
        """)
        
        return None

def load_taxonomy_paths(conn):
    """Load all taxonomy paths from database"""
    try:
        cursor = conn.cursor()
        
        # Adjust the query to match your database schema
        cursor.execute("""
            SELECT id, full_name FROM category_paths
        """)
        
        paths = cursor.fetchall()
        cursor.close()
        
        # Convert to DataFrame
        df_paths = pd.DataFrame(paths, columns=['id', 'full_path'])
        
        return df_paths
    except Exception as e:
        st.error(f"Error loading taxonomy paths: {e}")
        st.error("Please check that your database has a 'taxonomy_paths' table with 'id' and 'full_name' columns.")
        return pd.DataFrame(columns=['id', 'full_path'])