import psycopg2
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
import numpy as np

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
        st.error("Please check that your database has a 'category_paths' table with 'id' and 'full_name' columns.")
        return pd.DataFrame(columns=['id', 'full_path'])

def load_taxonomy_paths_with_vectors(conn):
    """Load all taxonomy paths with their vector representations from database"""
    try:
        cursor = conn.cursor()
        
        # First check if the category_paths table exists and has the expected columns
        try:
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
            
            if not category_vectors_exists:
                st.warning("The 'category_vectors' table does not exist. Loading paths without vectors.")
                df_paths = load_taxonomy_paths(conn)
                df_paths['vector'] = None
                return df_paths
                
            # Now attempt to join the tables
            st.info("Joining category_paths and category_vectors tables...")
            cursor.execute("""
                SELECT cp.id, cp.full_name as full_path, cv.vector 
                FROM category_paths cp
                LEFT JOIN category_vectors cv ON cp.id = cv.id
            """)
            
            combined_data = cursor.fetchall()
            cursor.close()
            
            # Convert to DataFrame
            df_paths = pd.DataFrame(combined_data, columns=['id', 'full_path', 'vector'])
            
            # Check if we got any vectors
            if df_paths['vector'].isnull().all():
                st.warning("No vectors found in the joined tables. Generating vectors...")
                # Generate vectors for each path using the vectorize_text function
                from utils.vector_utils import vectorize_text
                
                for idx, row in df_paths.iterrows():
                    text = row['full_path']
                    vector = vectorize_text(text)
                    df_paths.at[idx, 'vector'] = vector
            
            return df_paths
            
        except Exception as e:
            st.warning(f"Error joining tables: {e}")
            st.warning("Attempting to load paths directly...")
            
            # Reset cursor for a new query
            cursor.close()
            cursor = conn.cursor()
            
            # Load paths without vectors
            cursor.execute("""
                SELECT id, full_name FROM category_paths
            """)
            
            paths = cursor.fetchall()
            cursor.close()
            
            # Convert to DataFrame
            df_paths = pd.DataFrame(paths, columns=['id', 'full_path'])
            
            # Add empty vector column
            df_paths['vector'] = None
            
            # Generate vectors for each path
            st.info("Generating vectors for taxonomy paths...")
            from utils.vector_utils import vectorize_text
            
            for idx, row in df_paths.iterrows():
                text = row['full_path']
                vector = vectorize_text(text)
                df_paths.at[idx, 'vector'] = vector
            
            return df_paths
            
    except Exception as e:
        st.error(f"Error loading taxonomy paths with vectors: {e}")
        return pd.DataFrame(columns=['id', 'full_path', 'vector'])