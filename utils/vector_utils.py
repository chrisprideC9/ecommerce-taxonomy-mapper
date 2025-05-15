import pandas as pd
import numpy as np
import ast
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

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
            except Exception as e:
                # Just log the error, don't display to user to avoid cluttering UI
                pass
    
    return df

def vectorize_text(text, dimension=1536):
    """Create a vector representation of text (only used for products, not taxonomy)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    # Return zero vector for empty text
    if not isinstance(text, str) or not text.strip():
        return np.zeros(dimension)
    
    try:
        # Create TF-IDF vector with fewer features for speed
        vectorizer = TfidfVectorizer(max_features=min(768, dimension))
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Convert to dense array and get first row
        vector = tfidf_matrix.toarray()[0]
        
        # Pad or truncate to match dimension
        if len(vector) < dimension:
            vector = np.pad(vector, (0, dimension - len(vector)), 'constant')
        elif len(vector) > dimension:
            vector = vector[:dimension]
            
        # Normalize the vector (L2 norm) for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    except Exception as e:
        # Return zero vector on error
        return np.zeros(dimension)

def batch_vectorize(texts, dimension=1536):
    """Vectorize multiple texts at once for efficiency"""
    if not texts:
        return []
    
    # Filter empty texts and non-strings
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not valid_texts:
        return [np.zeros(dimension) for _ in range(len(texts))]
    
    try:
        # Create TF-IDF vectors for all texts at once
        vectorizer = TfidfVectorizer(max_features=min(768, dimension))
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        
        # Convert to dense arrays
        vectors = tfidf_matrix.toarray()
        
        # Process each vector for proper dimensionality
        result = []
        for i, vec in enumerate(vectors):
            # Ensure proper dimension
            if len(vec) < dimension:
                vec = np.pad(vec, (0, dimension - len(vec)), 'constant')
            elif len(vec) > dimension:
                vec = vec[:dimension]
                
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
                
            result.append(vec)
            
        # Add zero vectors for any invalid texts
        final_result = []
        valid_idx = 0
        for text in texts:
            if isinstance(text, str) and text.strip():
                final_result.append(result[valid_idx])
                valid_idx += 1
            else:
                final_result.append(np.zeros(dimension))
                
        return final_result
    except Exception as e:
        # Return zero vectors on error
        return [np.zeros(dimension) for _ in range(len(texts))]