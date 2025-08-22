#-*- coding: cp949 -*-
# utils.py
import os
import openai
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

def set_openai_api_key():
    """Set the OpenAI API key from environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key  # Set the API key for OpenAI


def set_gpt_model():
    """Set the OpenAI API key from environment variable."""
    gpt_model= os.getenv("MODEL")
    if not gpt_model:
        raise ValueError("MODEL key not found. Please set the MODEL environment variable.")
    return gpt_model
