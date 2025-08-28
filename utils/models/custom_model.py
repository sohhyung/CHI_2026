# utils/models/custom_model.py

def get_custom_response(messages: list[str]) -> str:
    """
    Generate a response from a custom model based on the conversation history.

    Args:
        messages (list[str]): Recent user and agent messages (combined) as plain text strings.

    Returns:
        str: Generated response from the custom model.
    """
    # TODO: Replace this with actual custom model logic
    return "안녕하세요! 만나서 반가워요! 오늘 기분은 어떠신가요?"
