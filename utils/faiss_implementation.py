import numpy as np
import streamlit as st

def build_faiss_index(vectors, dimension=1536):
    """Build a FAISS index for fast vector search"""
    try:
        import faiss
        
        # Check if we have vectors
        if not vectors or len(vectors) == 0:
            st.warning("No vectors provided for FAISS index")
            return None
            
        # Filter out None values and convert to numpy array
        valid_vectors = []
        valid_indices = []
        
        for i, vec in enumerate(vectors):
            if vec is not None:
                # Convert lists to numpy arrays
                if isinstance(vec, list):
                    vec = np.array(vec)
                # Make sure it's a numpy array
                if isinstance(vec, np.ndarray):
                    # Handle NaN values
                    if np.isnan(vec).any():
                        vec = np.nan_to_num(vec)
                    valid_vectors.append(vec.astype('float32'))
                    valid_indices.append(i)
        
        if not valid_vectors:
            st.warning("No valid vectors found for FAISS index")
            return None
            
        # Stack into a single array
        vectors_np = np.vstack(valid_vectors).astype('float32')
        
        # Create FAISS index - using inner product for cosine similarity on normalized vectors
        index = faiss.IndexFlatIP(vectors_np.shape[1])
        index.add(vectors_np)
        
        # Return the index and mapping of indices
        return {
            'index': index,
            'valid_indices': valid_indices,
            'dimension': vectors_np.shape[1]
        }
        
    except ImportError:
        st.error("FAISS is not installed. Run: pip install faiss-cpu")
        return None
    except Exception as e:
        st.error(f"Error building FAISS index: {e}")
        return None

def search_faiss_index(faiss_index, query_vectors, top_k=5):
    """Search the FAISS index for similar vectors"""
    try:
        import faiss
        
        # Check if we have a valid index
        if faiss_index is None or 'index' not in faiss_index:
            st.warning("No valid FAISS index provided")
            return None, None
            
        # Get the FAISS index and dimension
        index = faiss_index['index']
        dimension = faiss_index['dimension']
        
        # Process query vectors
        if isinstance(query_vectors, list):
            # Filter out None values
            valid_queries = []
            for vec in query_vectors:
                if vec is not None:
                    # Convert lists to numpy arrays
                    if isinstance(vec, list):
                        vec = np.array(vec)
                    # Make sure it's a numpy array with correct shape
                    if isinstance(vec, np.ndarray) and vec.shape[0] == dimension:
                        # Handle NaN values
                        if np.isnan(vec).any():
                            vec = np.nan_to_num(vec)
                        valid_queries.append(vec.astype('float32'))
                    else:
                        valid_queries.append(np.zeros(dimension, dtype='float32'))
                else:
                    valid_queries.append(np.zeros(dimension, dtype='float32'))
                    
            if not valid_queries:
                return None, None
                
            # Stack into a single array
            query_np = np.vstack(valid_queries).astype('float32')
        else:
            # Single vector case
            if query_vectors is None:
                return None, None
                
            # Convert to numpy array
            if isinstance(query_vectors, list):
                query_vectors = np.array(query_vectors)
                
            # Reshape if needed
            if len(query_vectors.shape) == 1:
                query_vectors = query_vectors.reshape(1, -1)
                
            # Handle NaN values
            if np.isnan(query_vectors).any():
                query_vectors = np.nan_to_num(query_vectors)
                
            query_np = query_vectors.astype('float32')
        
        # Search the index
        distances, indices = index.search(query_np, top_k)
        
        return distances, indices
        
    except ImportError:
        st.error("FAISS is not installed. Run: pip install faiss-cpu")
        return None, None
    except Exception as e:
        st.error(f"Error searching FAISS index: {e}")
        return None, None