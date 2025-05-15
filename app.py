import streamlit as st

# IMPORTANT: set_page_config MUST be at the top level of the script, 
# NOT inside any function, and before any other Streamlit commands
st.set_page_config(page_title="E-commerce Taxonomy Mapper", page_icon="🏷️", layout="wide")

# Now import everything else
import pandas as pd
import numpy as np
import time
import re
import ssl
import nltk
import os

# Fix NLTK SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    st.info("SSL certificate verification disabled for NLTK")

# Import other utilities
from utils.db_utils import get_db_connection, load_taxonomy_paths_with_vectors
from utils.text_processor import preprocess_text, extract_url_segments
from utils.performance_monitor import measure_time, time_this, show_performance_metrics, clear_metrics

# Initialize session state
if 'taxonomy_df' not in st.session_state:
    st.session_state.taxonomy_df = None
    
if 'taxonomy_index' not in st.session_state:
    st.session_state.taxonomy_index = None
    
if 'data_source_config' not in st.session_state:
    st.session_state.data_source_config = None
    
if 'filter_key' not in st.session_state:
    st.session_state.filter_key = None

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def cached_load_taxonomy_paths(use_db, use_sample_taxonomy, db_host, db_name, db_user, db_password, db_port):
    """Cached function to load taxonomy paths"""
    
    st.info("Loading taxonomy paths (cached)...")
    
    # Create connection if using database
    conn = None
    if use_db:
        try:
            import psycopg2
            
            # Clean up the host URL if needed
            host = db_host
            if host.startswith(("http://", "https://")):
                host = host.replace("http://", "").replace("https://", "")
            if host.endswith("/"):
                host = host[:-1]
            
            conn = psycopg2.connect(
                host=host,
                database=db_name,
                user=db_user,
                password=db_password,
                port=db_port
            )
            
            # Try to load from database
            from utils.db_utils import load_taxonomy_paths_with_vectors
            df_paths = load_taxonomy_paths_with_vectors(conn)
            
            if df_paths is not None and not df_paths.empty:
                st.success(f"Successfully loaded {len(df_paths)} taxonomy paths from database")
                return df_paths
        except Exception as e:
            st.error(f"Database error: {e}")
    
    # If database load failed or is not being used, try sample data
    if use_sample_taxonomy or not use_db or conn is None:
        st.warning("Using sample taxonomy data")
        sample_file = "taxonomy_paths.csv"
        try:
            df_paths = pd.read_csv(sample_file)
            st.success(f"Loaded {len(df_paths)} taxonomy paths from sample file")
            return df_paths
        except Exception as e:
            st.warning(f"Error loading sample taxonomy file: {e}")
            # Create a minimal sample taxonomy
            sample_data = {
                'id': range(1, 9),
                'full_path': [
                    "Apparel & Accessories > Clothing",
                    "Apparel & Accessories > Clothing > Activewear",
                    "Apparel & Accessories > Clothing > Outerwear",
                    "Health & Beauty > Personal Care",
                    "Health & Beauty > Personal Care > Shaving & Grooming",
                    "Health & Beauty > Personal Care > Shaving & Grooming > Electric Razors",
                    "Health & Beauty > Personal Care > Shaving & Grooming > Shaving Brushes",
                    "Health & Beauty > Personal Care > Shaving & Grooming > Razors & Blades"
                ]
            }
            df_paths = pd.DataFrame(sample_data)
            df_paths['vector'] = None
            
            # Vectorize this small sample dataset (it's OK for small samples)
            from utils.vector_utils import vectorize_text
            df_paths['vector'] = df_paths['full_path'].apply(vectorize_text)
            
            st.success(f"Created {len(df_paths)} sample taxonomy paths")
            return df_paths
    
    # Fallback empty dataframe if all else fails
    return pd.DataFrame(columns=['id', 'full_path', 'vector'])

@st.cache_data
def build_faiss_index(taxonomy_df_json, filter_categories=None):
    """Build and cache the FAISS index for taxonomy vectors"""
    # Convert the JSON serialized DataFrame back to a DataFrame
    taxonomy_df = pd.read_json(taxonomy_df_json)
    
    # Apply category filter if provided
    filtered_df = taxonomy_df
    if filter_categories and len(filter_categories) > 0:
        filtered_df = taxonomy_df[taxonomy_df['full_path'].apply(
            lambda x: any(x.startswith(cat) for cat in filter_categories)
        )]
        
        if filtered_df.empty:
            st.warning("No taxonomy paths match your category filter. Using all paths.")
            filtered_df = taxonomy_df
    
    # Create FAISS index
    st.info(f"Building search index for {len(filtered_df)} taxonomy paths...")
    
    try:
        import faiss
        from utils.vector_utils import prepare_vectors
        
        # Check if vectors need parsing from strings
        vector_column = 'vector'
        if not filtered_df.empty:
            sample_vector = filtered_df[vector_column].iloc[0]
            
            # Parse vectors if needed
            if sample_vector is not None and isinstance(sample_vector, str):
                filtered_df = prepare_vectors(filtered_df, vector_column)
                vector_column = 'content_vector'
            
            # Create vector list, filtering out None values
            vectors = []
            valid_indices = []
            
            for i, vec in enumerate(filtered_df[vector_column]):
                if vec is not None:
                    if isinstance(vec, list):
                        vec = np.array(vec)
                    if isinstance(vec, np.ndarray):
                        vectors.append(vec.astype('float32'))
                        valid_indices.append(i)
            
            if not vectors:
                st.warning("No valid vectors found for creating search index")
                return None
            
            # Convert to numpy array
            vector_array = np.vstack(vectors).astype('float32')
            
            # Build the index
            dimension = vector_array.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            index.add(vector_array)
            
            # Create index object
            taxonomy_index = {
                'index': index,
                'valid_indices': valid_indices,
                'dimension': dimension,
                'filtered_df': filtered_df
            }
            
            st.success(f"Successfully built search index with {len(vectors)} vectors")
            return taxonomy_index
    
    except Exception as e:
        st.error(f"Error building search index: {e}")
    
    return None

def main():
    st.title("E-commerce Taxonomy Mapper")
    st.write("Upload your Screaming Frog data to map products to taxonomies")
    
    # Sidebar for settings
    st.sidebar.header("Matching Settings")
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.05,  # Lower default threshold
        step=0.01,
        help="Minimum similarity score required for a match"
    )
    
    # Store context setting
    store_context = st.sidebar.text_input(
        "Store Context (optional)",
        placeholder="e.g., barber supplies, electronics, furniture",
        help="Provide context about the store type to improve matching"
    )
    
    store_context_weight = st.sidebar.slider(
        "Store Context Weight", 
        min_value=0.0, 
        max_value=5.0, 
        value=2.0,
        step=0.1,
        help="Weight given to store context in matching (higher = more influential)"
    )
    
    show_alternatives = st.sidebar.checkbox(
        "Show Alternative Matches", 
        value=True,
        help="Show alternative category matches for review"
    )
    
    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=10,
        max_value=5000,
        value=500,  # Increased default batch size
        help="Number of products to process in each batch"
    )
    
    # Use FAISS for faster matching
    use_faiss = st.sidebar.checkbox(
        "Use FAISS for faster matching", 
        value=True,
        help="Dramatically speeds up matching for large datasets"
    )
    
    # Vector settings
    st.sidebar.header("Advanced Vector Settings")
    use_content_vectors = st.sidebar.checkbox(
        "Use Content Vectors", 
        value=True,
        help="Use pre-computed content vectors if available in 'embeds 1' column"
    )

    text_vector_weight = st.sidebar.slider(
        "Text Vector Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3,
        step=0.05,
        help="Weight for text-based vectors when blending with content vectors (1.0 = only text vectors, 0.0 = only content vectors)"
    )
    
    # Database settings
    st.sidebar.header("Data Source")
    use_db = st.sidebar.checkbox("Use Database", value=True)
    use_sample_taxonomy = st.sidebar.checkbox("Use Sample Taxonomy", value=False)
    
    # Generate current configuration
    current_config = {
        'use_db': use_db,
        'use_sample': use_sample_taxonomy,
        'db_host': os.environ.get("DB_HOST", "localhost"),
        'db_name': os.environ.get("DB_NAME", "your_database"),
        'db_user': os.environ.get("DB_USER", "postgres"),
        'db_password': os.environ.get("DB_PASSWORD", "your_password"),
        'db_port': os.environ.get("DB_PORT", "5432")
    }
    
    # Only reload taxonomy data if necessary
    if (st.session_state.taxonomy_df is None or 
        st.session_state.data_source_config != current_config):
        
        # Load taxonomy data with caching
        taxonomy_df = cached_load_taxonomy_paths(
            use_db, 
            use_sample_taxonomy,
            current_config['db_host'],
            current_config['db_name'],
            current_config['db_user'],
            current_config['db_password'],
            current_config['db_port']
        )
        
        # Update session state
        st.session_state.taxonomy_df = taxonomy_df
        st.session_state.data_source_config = current_config
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Screaming Frog CSV", type=['csv'])
    
    if uploaded_file:
        # Load the CSV data
        df = pd.read_csv(uploaded_file)
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Check if required columns are present
        required_columns = ['Address', 'Title 1', 'Meta Description 1']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return
        
        # Use taxonomy paths from session state
        taxonomy_df = st.session_state.taxonomy_df
        
        if taxonomy_df is None or taxonomy_df.empty:
            st.error("Could not load taxonomy paths from any source")
            return
        
        # Category filtering
        top_level_categories = sorted(taxonomy_df['full_path'].apply(lambda x: x.split('>')[0].strip()).unique())
        
        category_filter = st.sidebar.multiselect(
            "Filter to These Top-Level Categories",
            options=top_level_categories,
            default=[],
            help="Limit matching to these top-level categories"
        )
        
        # Check if we need to rebuild the FAISS index
        filter_key = ','.join(sorted(category_filter))
        if filter_key != st.session_state.filter_key or st.session_state.taxonomy_index is None:
            # We need to serialize the DataFrame to JSON for caching
            taxonomy_df_json = taxonomy_df.to_json()
            # Build the FAISS index with the filtered data
            st.session_state.taxonomy_index = build_faiss_index(taxonomy_df_json, category_filter)
            st.session_state.filter_key = filter_key
        
        # Get the filtered paths
        if category_filter:
            filtered_paths = taxonomy_df[taxonomy_df['full_path'].apply(
                lambda x: any(x.startswith(cat) for cat in category_filter)
            )]
            if filtered_paths.empty:
                st.warning("No taxonomy paths match your category filter. Using all paths.")
                filtered_paths = taxonomy_df
            else:
                st.success(f"Filtered to {len(filtered_paths)} taxonomy paths in selected categories")
        else:
            filtered_paths = taxonomy_df
        
        # Display taxonomy stats
        vector_count = filtered_paths['vector'].notna().sum()
        st.success(f"Ready to match against {len(filtered_paths)} taxonomy paths ({vector_count} with vectors)")
        
        # Process button
        if st.button("Start Mapping"):
            # Add debug mode for zero matches
            show_debug = st.checkbox("Show matching debug info", value=True)
            
            # Process in batches
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            start_time = time.time()
            
            # Check if we have content vectors
            has_content_vectors = 'embeds 1' in df.columns
            if has_content_vectors and use_content_vectors:
                st.info("Using pre-computed content vectors from 'embeds 1' column")
            
            for i in range(total_batches):
                batch_start_time = time.time()
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(df))
                batch = df.iloc[start_idx:end_idx].copy()
                
                status_text.text(f"Processing batch {i+1}/{total_batches} ({start_idx} to {end_idx})...")
                
                # Process this batch
                if use_faiss and st.session_state.taxonomy_index is not None:
                    if has_content_vectors and use_content_vectors:
                        # Use the content vector enhanced matcher
                        batch_results = process_batch_with_content_vectors(
                            batch, 
                            filtered_paths,
                            st.session_state.taxonomy_index,
                            store_context=store_context,
                            context_weight=store_context_weight,
                            threshold=similarity_threshold,
                            top_n=5 if show_alternatives else 1,
                            text_vector_weight=text_vector_weight,
                            debug_mode=show_debug
                        )
                    else:
                        # Use the regular FAISS matcher
                        batch_results = process_batch_with_faiss(
                            batch, 
                            filtered_paths,
                            st.session_state.taxonomy_index,
                            store_context=store_context,
                            context_weight=store_context_weight,
                            threshold=similarity_threshold,
                            top_n=5 if show_alternatives else 1,
                            debug_mode=show_debug
                        )
                else:
                    # Use the regular matcher
                    batch_results = process_batch(
                        batch, 
                        filtered_paths,
                        store_context=store_context,
                        context_weight=store_context_weight,
                        threshold=similarity_threshold,
                        top_n=5 if show_alternatives else 1,
                        debug_mode=show_debug
                    )
                
                results.append(batch_results)
                batch_time = time.time() - batch_start_time
                status_text.text(f"Batch {i+1}/{total_batches} completed in {batch_time:.2f}s ({batch_size/(batch_time+0.001):.1f} products/sec)")
                progress_bar.progress((i + 1) / total_batches)
            
            # Combine results
            result_df = pd.concat(results, ignore_index=True)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            products_per_second = len(df) / processing_time
            st.success(f"Processing completed in {processing_time:.2f} seconds ({products_per_second:.1f} products/second)")
            
            # Show statistics
            matched_count = result_df['predicted_category'].notna().sum()
            total_count = len(result_df)
            match_rate = (matched_count / total_count) * 100 if total_count > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Products", total_count)
            col2.metric("Matched Products", matched_count)
            col3.metric("Match Rate", f"{match_rate:.1f}%")
            col4.metric("Products/second", f"{products_per_second:.1f}")
            
            # If we have zero matches, show debug information
            if matched_count == 0 and show_debug:
                st.warning("No matches found. The similarity threshold may be too high.")
                st.info("Here are the highest similarity scores for each product:")
                
                # Display the confidence scores
                st.dataframe(result_df[['Address', 'confidence', 'best_match', 'debug_info']])
                
                st.info("Try lowering the similarity threshold to get matches. Current threshold: " + str(similarity_threshold))
            
            # Display the results
            st.subheader("Results")
            st.dataframe(result_df)
            
            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Mapped Data",
                data=csv,
                file_name="taxonomy_mapped_products.csv",
                mime="text/csv"
            )

def process_batch_with_faiss(batch, df_paths, taxonomy_index, store_context=None, 
                            context_weight=2.0, threshold=0.1, top_n=5, debug_mode=False):
    """Process a batch using FAISS for faster vector searching"""
    try:
        import faiss
        import numpy as np
        from utils.vector_utils import vectorize_text
        
        # Create result dataframe
        result_batch = batch.copy()
        
        # Initialize debug columns
        if debug_mode:
            result_batch['debug_info'] = None
            result_batch['best_match'] = None
        
        # Prepare product vectors
        product_vectors = []
        
        # Process each product
        for idx, row in batch.iterrows():
            try:
                url = row['Address']
                title = row['Title 1']
                description = row.get('Meta Description 1', '')
                
                # Extract URL segments
                url_segments = extract_url_segments(url)
                url_text = ' '.join(url_segments)
                
                # Combine text features with title given more weight
                combined_text = f"{url_text} {title} {title} {description}"
                
                # Add store context if provided
                if store_context and store_context.strip():
                    context_repeat = max(1, int(context_weight))
                    context_text = ' '.join([store_context] * context_repeat)
                    combined_text = f"{combined_text} {context_text}"
                
                # Clean text and create vector
                cleaned_text = preprocess_text(combined_text)
                product_vector = vectorize_text(cleaned_text, dimension=taxonomy_index['dimension'])
                
                # Normalize vector for cosine similarity
                norm = np.linalg.norm(product_vector)
                if norm > 0:
                    product_vector = product_vector / norm
                
                product_vectors.append(product_vector)
                
            except Exception as e:
                # Handle errors for individual products
                product_vectors.append(np.zeros(taxonomy_index['dimension']))
                result_batch.at[idx, 'predicted_category'] = None
                result_batch.at[idx, 'category_id'] = None
                result_batch.at[idx, 'confidence'] = 0.0
                result_batch.at[idx, 'error'] = str(e)
                if debug_mode:
                    result_batch.at[idx, 'debug_info'] = f"Error: {str(e)}"
        
        # Convert to numpy array
        product_array = np.vstack(product_vectors).astype('float32')
        
        # Search for nearest neighbors
        D, I = taxonomy_index['index'].search(product_array, top_n)
        
        # Process results
        for i, (indices, distances) in enumerate(zip(I, D)):
            valid_matches = []
            
            # For debug mode, always get the best match regardless of threshold
            best_match_idx = indices[0] if len(indices) > 0 else None
            best_match_dist = distances[0] if len(distances) > 0 else 0.0
            
            if best_match_idx is not None and debug_mode:
                original_best_idx = taxonomy_index['valid_indices'][best_match_idx]
                best_match_path = df_paths.iloc[original_best_idx]['full_path']
                result_batch.iloc[i, result_batch.columns.get_loc('best_match')] = best_match_path
                result_batch.iloc[i, result_batch.columns.get_loc('debug_info')] = f"Best score: {best_match_dist:.4f}, Threshold: {threshold}"
            
            for j, (idx, dist) in enumerate(zip(indices, distances)):
                if dist >= threshold:
                    # Map back to original index
                    original_idx = taxonomy_index['valid_indices'][idx]
                    path_row = df_paths.iloc[original_idx]
                    
                    valid_matches.append((original_idx, dist, path_row))
            
            if valid_matches:
                # Get best match
                best_idx, best_sim, best_match = valid_matches[0]
                
                # Add to results
                row_idx = batch.index[i]
                result_batch.at[row_idx, 'predicted_category'] = best_match['full_path']
                result_batch.at[row_idx, 'category_id'] = best_match['id']
                result_batch.at[row_idx, 'confidence'] = float(best_sim)
                
                # Add alternative matches
                if len(valid_matches) > 1:
                    match_details = []
                    for _, sim, match in valid_matches:
                        match_details.append(f"{match['full_path']} ({sim:.3f})")
                    result_batch.at[row_idx, 'alternative_matches'] = "; ".join(match_details)
            else:
                # No match above threshold
                row_idx = batch.index[i]
                result_batch.at[row_idx, 'predicted_category'] = None
                result_batch.at[row_idx, 'category_id'] = None
                
                # Store best similarity even if below threshold
                best_sim = distances[0] if len(distances) > 0 else 0.0
                result_batch.at[row_idx, 'confidence'] = float(best_sim)
                result_batch.at[row_idx, 'alternative_matches'] = None
        
        return result_batch
        
    except Exception as e:
        st.error(f"Error in FAISS batch processing: {e}")
        return batch
    
def process_batch_with_content_vectors(batch, df_paths, taxonomy_index, 
                                store_context=None, context_weight=2.0, 
                                threshold=0.1, top_n=5, text_vector_weight=0.3,
                                debug_mode=False):
    """Process a batch using pre-computed content vectors and text-based vectors"""
    try:
        import faiss
        import numpy as np
        import ast
        from utils.vector_utils import vectorize_text
        
        # Create result dataframe
        result_batch = batch.copy()
        
        # Initialize debug columns
        if debug_mode:
            result_batch['debug_info'] = None
            result_batch['best_match'] = None
        
        # Prepare vectors and weights for blending
        has_content_vectors = 'embeds 1' in batch.columns
        text_weight = text_vector_weight  # Weight for text-based vectors
        embed_weight = 1.0 - text_weight  # Weight for pre-computed content vectors
        
        # Process each product
        for idx, row in batch.iterrows():
            try:
                # Get content vector if available
                content_vector = None
                if has_content_vectors and not pd.isna(row['embeds 1']) and row['embeds 1']:
                    try:
                        # Parse vector string from 'embeds 1' column
                        embed_str = row['embeds 1']
                        if isinstance(embed_str, str):
                            if embed_str.startswith('[') and embed_str.endswith(']'):
                                content_vector = np.array(ast.literal_eval(embed_str))
                            elif ',' in embed_str:
                                content_vector = np.array([float(x) for x in embed_str.split(',')])
                            else:
                                content_vector = np.array([float(x) for x in embed_str.split()])
                            
                            # Normalize the vector
                            norm = np.linalg.norm(content_vector)
                            if norm > 0:
                                content_vector = content_vector / norm
                                
                            if debug_mode:
                                result_batch.at[idx, 'debug_info'] = f"Content vector loaded, dims: {content_vector.shape}"
                    except Exception as e:
                        if debug_mode:
                            result_batch.at[idx, 'debug_info'] = f"Error parsing vector: {str(e)}"
                        content_vector = None
                
                # Always create a text-based vector
                url = row['Address']
                title = row['Title 1']
                description = row.get('Meta Description 1', '')
                
                # Extract URL segments
                url_segments = extract_url_segments(url)
                url_text = ' '.join(url_segments)
                
                # Combine text features
                combined_text = f"{url_text} {title} {title} {description}"
                
                # Add store context if provided
                if store_context and store_context.strip():
                    context_repeat = max(1, int(context_weight))
                    context_text = ' '.join([store_context] * context_repeat)
                    combined_text = f"{combined_text} {context_text}"
                
                # Clean text and create vector
                cleaned_text = preprocess_text(combined_text)
                text_vector = vectorize_text(cleaned_text, dimension=taxonomy_index['dimension'])
                
                # Normalize vector
                norm = np.linalg.norm(text_vector)
                if norm > 0:
                    text_vector = text_vector / norm
                
                # Blend vectors if we have both
                final_vector = None
                vector_source = "text_only"
                
                if content_vector is not None:
                    # Check dimensions match
                    if content_vector.shape[0] == text_vector.shape[0]:
                        # Weighted blend of vectors
                        final_vector = (text_weight * text_vector) + (embed_weight * content_vector)
                        # Normalize the blended vector
                        norm = np.linalg.norm(final_vector)
                        if norm > 0:
                            final_vector = final_vector / norm
                        vector_source = "blended"
                        
                        if debug_mode:
                            result_batch.at[idx, 'debug_info'] = f"{result_batch.at[idx, 'debug_info']} | Blended vectors with weights {text_weight}:{embed_weight}"
                    else:
                        if debug_mode:
                            result_batch.at[idx, 'debug_info'] = f"Dimension mismatch: text={text_vector.shape[0]}, content={content_vector.shape[0]}"
                        final_vector = text_vector
                else:
                    final_vector = text_vector
                
                # Search for similar vectors using FAISS
                query_vector = final_vector.reshape(1, -1).astype('float32')
                D, I = taxonomy_index['index'].search(query_vector, top_n)
                
                # For debug mode, always get the best match regardless of threshold
                best_match_idx = I[0][0] if I.shape[1] > 0 else None
                best_match_dist = D[0][0] if D.shape[1] > 0 else 0.0
                
                if best_match_idx is not None and debug_mode:
                    original_best_idx = taxonomy_index['valid_indices'][best_match_idx]
                    best_match_path = df_paths.iloc[original_best_idx]['full_path']
                    result_batch.at[idx, 'best_match'] = best_match_path
                    result_batch.at[idx, 'debug_info'] = f"{result_batch.at[idx, 'debug_info']} | Best score: {best_match_dist:.4f}, Threshold: {threshold}"
                
                # Process results
                valid_matches = []
                for j, (idx_j, dist) in enumerate(zip(I[0], D[0])):
                    if dist >= threshold:
                        # Map back to original index
                        original_idx = taxonomy_index['valid_indices'][idx_j]
                        path_row = df_paths.iloc[original_idx]
                        valid_matches.append((original_idx, dist, path_row))
                
                if valid_matches:
                    # Get best match
                    best_idx, best_sim, best_match = valid_matches[0]
                    
                    # Add to results
                    result_batch.at[idx, 'predicted_category'] = best_match['full_path']
                    result_batch.at[idx, 'predicted_category'] = best_match['full_path']
                    result_batch.at[idx, 'category_id'] = best_match['id']
                    result_batch.at[idx, 'confidence'] = float(best_sim)
                    result_batch.at[idx, 'vector_source'] = vector_source
                    
                    # Add alternative matches
                    if len(valid_matches) > 1:
                        match_details = []
                        for _, sim, match in valid_matches:
                            match_details.append(f"{match['full_path']} ({sim:.3f})")
                        result_batch.at[idx, 'alternative_matches'] = "; ".join(match_details)
                else:
                    # No match above threshold
                    result_batch.at[idx, 'predicted_category'] = None
                    result_batch.at[idx, 'category_id'] = None
                    
                    # Store best similarity even if below threshold
                    best_sim = D[0][0] if D.shape[1] > 0 else 0.0
                    result_batch.at[idx, 'confidence'] = float(best_sim)
                    result_batch.at[idx, 'vector_source'] = vector_source
            
            except Exception as e:
                # Handle errors for individual products
                result_batch.at[idx, 'predicted_category'] = None
                result_batch.at[idx, 'category_id'] = None
                result_batch.at[idx, 'confidence'] = 0.0
                result_batch.at[idx, 'error'] = str(e)
                result_batch.at[idx, 'vector_source'] = 'error'
                if debug_mode:
                    result_batch.at[idx, 'debug_info'] = f"Error: {str(e)}"
        
        return result_batch
        
    except Exception as e:
        st.error(f"Error in batch processing with content vectors: {e}")
        return batch

def process_batch(batch, df_paths, store_context=None, context_weight=2.0, 
                 threshold=0.1, top_n=5, debug_mode=False):
    """Process a batch of products using standard vector similarity"""
    # This is the original function, kept for fallback
    from utils.vector_utils import vectorize_text
    
    result_batch = batch.copy()
    
    # Initialize debug columns
    if debug_mode:
        result_batch['debug_info'] = None
        result_batch['best_match'] = None
    
    for idx, row in batch.iterrows():
        try:
            url = row['Address']
            title = row['Title 1']
            description = row.get('Meta Description 1', '')
            
            # Extract URL segments
            url_segments = extract_url_segments(url)
            url_text = ' '.join(url_segments)
            
            # Combine text features
            combined_text = f"{url_text} {title} {title} {description}"
            
            # Add store context if provided
            if store_context and store_context.strip():
                context_repeat = max(1, int(context_weight))
                context_text = ' '.join([store_context] * context_repeat)
                combined_text = f"{combined_text} {context_text}"
            
            # Clean text
            cleaned_text = preprocess_text(combined_text)
            
            # Create vector for this product
            product_vector = vectorize_text(cleaned_text)
            
            # Calculate similarities with all taxonomy paths
            similarities = []
            
            for path_idx, path_row in df_paths.iterrows():
                path_vector = path_row.get('vector') or path_row.get('content_vector')
                
                if path_vector is not None:
                    # Handle string vectors
                    if isinstance(path_vector, str):
                        # Parse vector string
                        import ast
                        path_vector = np.array(ast.literal_eval(path_vector))
                    
                    # Calculate cosine similarity
                    similarity = np.dot(product_vector, path_vector)
                    
                    # Ensure the similarity is within valid range
                    similarity = max(min(similarity, 1.0), -1.0)
                    
                    similarities.append((path_idx, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # For debug mode, always record the best match
            if similarities and debug_mode:
                best_idx, best_sim = similarities[0]
                best_match = df_paths.iloc[best_idx]['full_path']
                result_batch.at[idx, 'best_match'] = best_match
                result_batch.at[idx, 'debug_info'] = f"Best score: {best_sim:.4f}, Threshold: {threshold}"
            
            # Filter by threshold
            valid_matches = [(idx, sim) for idx, sim in similarities if sim >= threshold]
            
            if valid_matches:
                # Get best match
                best_idx, best_sim = valid_matches[0]
                best_match = df_paths.iloc[best_idx]
                
                # Add to results
                result_batch.at[idx, 'predicted_category'] = best_match['full_path']
                result_batch.at[idx, 'category_id'] = best_match['id']
                result_batch.at[idx, 'confidence'] = float(best_sim)
                
                # Add alternative matches
                if len(valid_matches) > 1:
                    match_details = []
                    for match_idx, sim in valid_matches[:top_n]:
                        match = df_paths.iloc[match_idx]
                        match_details.append(f"{match['full_path']} ({sim:.3f})")
                    result_batch.at[idx, 'alternative_matches'] = "; ".join(match_details)
            else:
                # No match above threshold
                result_batch.at[idx, 'predicted_category'] = None
                result_batch.at[idx, 'category_id'] = None
                
                # Store best similarity even if below threshold
                best_sim = similarities[0][1] if similarities else 0.0
                result_batch.at[idx, 'confidence'] = float(best_sim)
        
        except Exception as e:
            st.error(f"Error processing row {idx}: {e}")
            result_batch.at[idx, 'predicted_category'] = None
            result_batch.at[idx, 'category_id'] = None
            result_batch.at[idx, 'confidence'] = 0.0
            result_batch.at[idx, 'error'] = str(e)
            if debug_mode:
                result_batch.at[idx, 'debug_info'] = f"Error: {str(e)}"
    
    return result_batch

if __name__ == "__main__":
    main()