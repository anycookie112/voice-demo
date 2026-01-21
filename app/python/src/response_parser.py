import re

def parse_agent_response(response_text: str) -> dict:
    """
    Parse the agent response to extract MARKDOWN and TTS content.
    
    Supports three tag formats:
    1. <NORMAL>...</NORMAL> - Same content for both display and speech (saves tokens)
    2. <MARKDOWN>...</MARKDOWN> + <TTS>...</TTS> - Different content for visual and audio
    3. No tags - Fallback to raw text for both
    
    Args:
        response_text: The full response string from the LLM.
        
    Returns:
        dict: A dictionary with keys 'markdown' and 'tts'.
    """
    # First check for NORMAL tag (same content for both)
    normal_match = re.search(r'<NORMAL>(.*?)</NORMAL>', response_text, re.DOTALL)
    if normal_match:
        content = normal_match.group(1).strip()
        return {
            "markdown": content,
            "tts": content
        }
    
    # Check for separate MARKDOWN and TTS tags
    markdown_match = re.search(r'<MARKDOWN>(.*?)</MARKDOWN>', response_text, re.DOTALL)
    tts_match = re.search(r'<TTS>(.*?)</TTS>', response_text, re.DOTALL)
    
    markdown_content = markdown_match.group(1).strip() if markdown_match else None
    tts_content = tts_match.group(1).strip() if tts_match else None
    
    # Fallback if no tags found
    if markdown_content is None and tts_content is None:
        return {
            "markdown": response_text.strip(),
            "tts": response_text.strip()
        }
        
    return {
        "markdown": markdown_content or "",
        "tts": tts_content or ""
    }
