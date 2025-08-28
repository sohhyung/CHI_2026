# response_router.py
from .models.gpt_response import get_gpt_response
from .models.custom_model import get_custom_response

def get_response(mode: str, user_id: str, history: list[str]) -> str:
    """
    Routes the chat history to the appropriate response generator
    depending on the chat mode.

    Parameters:
        mode (str): 'A' for GPT, 'B' for custom model
        user_id (str): unique identifier of the user (used to load survey context)
        history (list[str]): list of recent chat messages (strings only)

    Returns:
        str: generated response
    """
    if mode == 'A':
        return get_gpt_response(user_id, history)
    elif mode == 'B':
        return get_custom_response(user_id, history)
    else:
        return '(No automated response available for this mode.)'
