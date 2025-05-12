import pandas as pd
import numpy as np
import ast
import streamlit as st

def prepare_vectors(df, vector_column):
    """Convert vector strings to numpy arrays"""
    df['content_vector'] = None
    
    for idx, row in df.iterrows():
        vector_str = row.get(vector_column, None)
        if vector_str and isinstance(vector_str, str):
            try:
                # Handle various vector string formats
                if vector_str.startswith('[') and vector_str.endswith(']'):
                    # Format: [0.1, 0.2, 0.3]
                    vector = np.array(ast.literal_eval(vector_str))
                elif ',' in vector_str:
                    # Format: 0.1,0.2,0.3
                    vector = np.array([float(x) for x in vector_str.split(',')])
                else:
                    # Format: 0.1 0.2 0.3
                    vector = np.array([float(x) for x in vector_str.split()])
                
                df.at[idx, 'content_vector'] = vector
            except (ValueError, SyntaxError) as e:
                st.warning(f"Could not parse vector at row {idx}: {e}")
    
    return df

def vectorize_text(text, dimension=1536):
    """Create a normalized vector representation of text"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    # Create TF-IDF vector
    vectorizer = TfidfVectorizer(max_features=min(1000, dimension))
    
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Convert to dense array and get first row
        vector = tfidf_matrix.toarray()[0]
        
        # If the vector is shorter than dimension, pad it
        if len(vector) < dimension:
            padded_vector = np.pad(vector, (0, dimension - len(vector)), 'constant')
            vector = padded_vector
        elif len(vector) > dimension:
            # If longer, truncate
            vector = vector[:dimension]
            
        # Normalize the vector (L2 norm)
        norm = np.linalg.norm(vector)
        if norm > 0:  # Avoid division by zero
            vector = vector / norm
            
        return vector
    except Exception as e:
        # Return a zero vector of proper dimension if processing fails
        st.warning(f"Error creating vector: {e}")
        return np.zeros(dimension)