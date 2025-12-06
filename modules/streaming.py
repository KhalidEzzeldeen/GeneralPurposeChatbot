"""
Streaming response utilities for better UX.
"""

import streamlit as st
from typing import Iterator, Any

def stream_response(text: str, container=None):
    """
    Stream text response token by token for better perceived performance.
    
    Args:
        text: The text to stream
        container: Streamlit container to write to (defaults to current context)
    """
    if container is None:
        container = st
    
    # Split text into words for streaming effect
    words = text.split()
    full_text = ""
    
    for word in words:
        full_text += word + " "
        container.markdown(full_text + "▌")  # Cursor indicator
    
    # Final render without cursor
    container.markdown(full_text)
    return full_text

def stream_llm_response(response_generator: Iterator[str], container=None):
    """
    Stream LLM response from a generator.
    
    Args:
        response_generator: Iterator that yields text chunks
        container: Streamlit container to write to
    """
    if container is None:
        container = st
    
    full_text = ""
    placeholder = container.empty()
    
    for chunk in response_generator:
        full_text += chunk
        placeholder.markdown(full_text + "▌")
    
    # Final render without cursor
    placeholder.markdown(full_text)
    return full_text

