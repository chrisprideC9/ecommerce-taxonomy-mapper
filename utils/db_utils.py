import psycopg2
import pandas as pd
import numpy as np
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "your_database")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_PORT = os.getenv("DB_PORT", "5432")

# Connection pool to reuse connections
connection_pool = None

def get_db_connection():
    """Get a connection from the connection pool or create a new one"""
    global connection_pool
    
    try:
        # Clean up the host URL if needed
        host = DB_HOST
        if host.startswith(("http://", "https://")):
            host = host.replace("http://", "").replace("https://", "")
        if host.endswith("/"):
            host = host[:-1]
        
        # Create new connection if none exists
        if connection_pool is None:
            connection_pool = psycopg2.connect(
                host=host,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                port=DB_PORT
            )
            
        # Test if connection is still alive
        cursor = connection_pool.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        
        return connection_pool
    except Exception as e:
        st.error(f"Database connection error: {e}")
        
        # Reset pool on error
        connection_pool = None
        
        # Provide debug info
        st.error("Please check your database configuration")
        st.code(f"""
        DB_HOST={DB_HOST} (should be hostname only)
        DB_NAME={DB_NAME}
        DB_USER={DB_USER}
        DB_PORT={DB_PORT}
        """)
        
        return None

def load_taxonomy_paths_with_vectors(conn):
    """Load all taxonomy paths with vectors from database without recreating vectors"""
    try:
        cursor = conn.cursor()
        
        # First check if the tables exist
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'category_paths'
            )
        """)
        
        category_paths_exists = cursor.fetchone()[0]
        if not category_paths_exists:
            st.error("The 'category_paths' table does not exist in the database.")
            cursor.close()
            return pd.DataFrame(columns=['id', 'full_path', 'vector'])
            
        # Check if the category_vectors table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'category_vectors'
            )
        """)
        
        category_vectors_exists = cursor.fetchone()[0]
        
        # Load data based on available tables
        if category_vectors_exists:
            with st.spinner("Loading taxonomy paths with existing vectors..."):
                # Use a JOIN query to get both paths and vectors
                cursor.execute("""
                    SELECT cp.id, cp.full_name as full_path, cv.vector 
                    FROM category_paths cp
                    LEFT JOIN category_vectors cv ON cp.id = cv.id
                """)
                
                combined_data = cursor.fetchall()
                
                # Convert to DataFrame
                df_paths = pd.DataFrame(combined_data, columns=['id', 'full_path', 'vector'])
                
                # Check if we actually got vectors
                vector_counts = df_paths['vector'].notna().sum()
                st.success(f"Loaded {len(df_paths)} taxonomy paths with {vector_counts} vectors from database")
                
                # Important: Do NOT re-create vectors here
                return df_paths
        else:
            # Just load paths without vectors
            with st.spinner("Loading taxonomy paths (no vectors found in database)..."):
                cursor.execute("""
                    SELECT id, full_name as full_path FROM category_paths
                """)
                
                paths = cursor.fetchall()
                
                # Convert to DataFrame
                df_paths = pd.DataFrame(paths, columns=['id', 'full_path'])
                df_paths['vector'] = None
                
                st.success(f"Loaded {len(df_paths)} taxonomy paths from database (without vectors)")
                return df_paths
                
    except Exception as e:
        st.error(f"Error loading taxonomy paths: {e}")
        return pd.DataFrame(columns=['id', 'full_path', 'vector'])