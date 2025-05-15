import time
import streamlit as st
import pandas as pd
import numpy as np
import functools
from contextlib import contextmanager

# Global storage for performance metrics
_PERFORMANCE_METRICS = []

@contextmanager
def measure_time(operation_name):
    """Context manager to measure execution time of a code block"""
    start_time = time.time()
    yield
    end_time = time.time()
    duration = end_time - start_time
    
    # Store metrics
    _PERFORMANCE_METRICS.append({
        'operation': operation_name,
        'duration': duration,
        'timestamp': end_time
    })

def time_this(func):
    """Decorator to measure execution time of a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with measure_time(func.__name__):
            return func(*args, **kwargs)
    return wrapper

def clear_metrics():
    """Clear all stored performance metrics"""
    global _PERFORMANCE_METRICS
    _PERFORMANCE_METRICS = []

def show_performance_metrics():
    """Display performance metrics in Streamlit"""
    if not _PERFORMANCE_METRICS:
        st.info("No performance metrics collected yet")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(_PERFORMANCE_METRICS)
    
    # Add formatted duration
    df['duration_str'] = df['duration'].apply(lambda x: f"{x:.4f}s")
    
    # Group by operation
    grouped = df.groupby('operation').agg({
        'duration': ['count', 'mean', 'min', 'max', 'sum'],
        'timestamp': ['min', 'max']
    })
    
    # Flatten column names
    grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]
    
    # Reset index and sort by total time
    grouped = grouped.reset_index().sort_values('duration_sum', ascending=False)
    
    # Format for display
    display_df = pd.DataFrame({
        'Operation': grouped['operation'],
        'Count': grouped['duration_count'],
        'Total Time': grouped['duration_sum'].apply(lambda x: f"{x:.2f}s"),
        'Average': grouped['duration_mean'].apply(lambda x: f"{x:.4f}s"),
        'Min': grouped['duration_min'].apply(lambda x: f"{x:.4f}s"),
        'Max': grouped['duration_max'].apply(lambda x: f"{x:.4f}s"),
    })
    
    # Display metrics
    st.subheader("Performance Metrics")
    st.dataframe(display_df)
    
    # Create a bar chart of total time by operation
    chart_data = grouped[['operation', 'duration_sum']].sort_values('duration_sum', ascending=True).tail(10)
    
    # Display chart
    st.subheader("Top Time-Consuming Operations")
    
    chart = {
        'data': [
            {
                'x': chart_data['duration_sum'].tolist(),
                'y': chart_data['operation'].tolist(),
                'type': 'bar',
                'orientation': 'h',
                'marker': {'color': 'rgba(50, 171, 96, 0.7)'}
            }
        ],
        'layout': {
            'title': 'Total Execution Time by Operation (seconds)',
            'xaxis': {'title': 'Time (seconds)'},
            'yaxis': {'title': 'Operation'},
            'height': 400,
            'margin': {'l': 150, 'r': 50, 't': 50, 'b': 50}
        }
    }
    
    st.plotly_chart(chart)
    
    # Add button to clear metrics
    if st.button("Clear Metrics"):
        clear_metrics()
        st.success("Performance metrics cleared")
        st.experimental_rerun()

def get_current_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except:
        return 0

def log_memory_usage(operation_name):
    """Log current memory usage"""
    memory_mb = get_current_memory_usage()
    st.info(f"Memory usage after {operation_name}: {memory_mb:.1f} MB")
    return memory_mb