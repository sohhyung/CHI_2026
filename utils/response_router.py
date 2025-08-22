# response_router.py
from .models.gpt_response import get_gpt_response
from .models.custom_model import get_custom_response


def get_response(mode: str, history: list[str]) -> str:
    """
    Routes the chat history to the appropriate response generator
    depending on the chat mode.
    
    Parameters:
        mode (str): 'A' for GPT, 'B' for custom model
        history (list[str]): list of recent chat messages (strings only)

    Returns:
        str: generated response
    """
    if mode == 'A':
        return get_gpt_response(history)
    elif mode == 'B':
        return get_custom_response(history)
    else:
        return '(No automated response available for this mode.)'
