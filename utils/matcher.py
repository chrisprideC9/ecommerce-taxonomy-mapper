import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_processor import preprocess_text, extract_url_segments
import streamlit as st

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

def find_best_matches(url, title, description, df_paths, content_vector=None, 
                      text_weight=0.7, threshold=0.3, top_n=5):
    """Find the best matching category using hybrid approach"""
    # Extract URL segments for better matching
    url_segments = extract_url_segments(url)
    url_text = ' '.join(url_segments)
    
    # Combine text features
    combined_text = f"{url_text} {title} {title} {description}"
    processed_text = preprocess_text(combined_text)
    
    if not processed_text.strip() or df_paths.empty:
        st.warning("Either input text is empty or taxonomy paths dataframe is empty")
        return pd.DataFrame(columns=['id', 'full_path', 'similarity'])
    
    # Prepare corpus for TF-IDF
    df_processed = df_paths.copy()
    if 'processed_path' not in df_processed.columns:
        df_processed['processed_path'] = df_processed['full_path'].apply(preprocess_text)
    
    # Make sure we have valid processed paths
    df_processed = df_processed[df_processed['processed_path'].notna() & 
                               (df_processed['processed_path'].str.strip() != '')]
    
    if df_processed.empty:
        st.warning("No valid taxonomy paths after preprocessing")
        return pd.DataFrame(columns=['id', 'full_path', 'similarity'])
    
    corpus = list(df_processed['processed_path'])
    corpus.append(processed_text)
    
    # Create TF-IDF vectors
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate text-based similarities
        input_vector = tfidf_matrix[-1]
        
        # Check if we have path vectors
        if tfidf_matrix.shape[0] <= 1:
            st.warning("No valid path vectors to compare against")
            return pd.DataFrame(columns=['id', 'full_path', 'similarity'])
            
        path_vectors = tfidf_matrix[:-1]
        text_similarities = cosine_similarity(input_vector, path_vectors).flatten()
        
        # Create results with text similarities
        df_results = df_processed.copy()
        df_results['text_similarity'] = text_similarities
        
        # If content vector is available, enhance matching
        if content_vector is not None and isinstance(content_vector, (list, np.ndarray)):
            # Convert to numpy array if it's a list
            if isinstance(content_vector, list):
                content_vector = np.array(content_vector)
            
            # Handle NaN and Inf values
            if np.isnan(content_vector).any() or np.isinf(content_vector).any():
                content_vector = np.nan_to_num(content_vector)
            
            # Normalize content vector
            content_norm = np.linalg.norm(content_vector)
            if content_norm > 0:
                content_vector = content_vector / content_norm
            
            # Get top text matches for refinement
            top_indices = df_results['text_similarity'].nlargest(min(10, len(df_results))).index
            
            # For each top match, create a vector from the category path
            for idx in top_indices:
                category_path = df_results.loc[idx, 'full_path']
                
                # Convert category path to a vector
                category_vector = vectorize_text(category_path)
                
                # Calculate vector similarity with error handling
                try:
                    # Handle both vectors properly
                    if isinstance(category_vector, (list, np.ndarray)):
                        category_vector = np.array(category_vector)
                        
                        # Handle NaN and Inf values
                        if np.isnan(category_vector).any() or np.isinf(category_vector).any():
                            category_vector = np.nan_to_num(category_vector)
                        
                        # Normalize category vector
                        category_norm = np.linalg.norm(category_vector)
                        if category_norm > 0:
                            category_vector = category_vector / category_norm
                        
                        # Use dot product instead of cosine_similarity for numerical stability
                        vector_similarity = np.dot(content_vector, category_vector)
                        
                        # Ensure the similarity is within valid range [-1, 1]
                        vector_similarity = max(min(vector_similarity, 1.0), -1.0)
                    else:
                        vector_similarity = 0.0
                except Exception as e:
                    st.warning(f"Error in vector similarity calculation: {e}")
                    vector_similarity = 0.0
                
                # Store vector similarity
                df_results.at[idx, 'vector_similarity'] = vector_similarity
            
            # Calculate combined similarity for top matches
            for idx in top_indices:
                if 'vector_similarity' in df_results.columns and not pd.isna(df_results.at[idx, 'vector_similarity']):
                    text_sim = df_results.at[idx, 'text_similarity']
                    vector_sim = df_results.at[idx, 'vector_similarity']
                    
                    # Combined weighted similarity
                    combined_sim = (text_weight * text_sim) + ((1 - text_weight) * vector_sim)
                    df_results.at[idx, 'similarity'] = combined_sim
                else:
                    df_results.at[idx, 'similarity'] = df_results.at[idx, 'text_similarity']
        else:
            # Use text similarity only
            df_results['similarity'] = df_results['text_similarity']
        
        # Sort by similarity
        df_results = df_results.sort_values('similarity', ascending=False)
        
        # Filter by threshold
        if threshold > 0:
            df_results = df_results[df_results['similarity'] >= threshold]
        
        return df_results.head(top_n)
    
    except Exception as e:
        st.error(f"Error in vector matching: {e}")
        return pd.DataFrame(columns=['id', 'full_path', 'similarity'])