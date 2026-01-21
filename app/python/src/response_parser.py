import re

def parse_agent_response(response_text: str) -> dict:
    """
    Parse the agent response to extract MARKDOWN and TTS content.
    
    Args:
        response_text: The full response string from the LLM containing <MARKDOWN> and <TTS> tags.
        
    Returns:
        dict: A dictionary with keys 'markdown' and 'tts'.
              If tags are missing, it attempts to provide reasonable defaults
              or returns the raw text if no tags are found.
    """
    markdown_match = re.search(r'<MARKDOWN>(.*?)</MARKDOWN>', response_text, re.DOTALL)
    tts_match = re.search(r'<TTS>(.*?)</TTS>', response_text, re.DOTALL)
    
    markdown_content = markdown_match.group(1).strip() if markdown_match else None
    tts_content = tts_match.group(1).strip() if tts_match else None
    
    # Fallback logic if tags are not present (for backward compatibility or partial responses)
    if markdown_content is None and tts_content is None:
        # If no tags, assume the whole text is for both (or handle as plain text)
        # But since we enforced tags, this case might mean a raw response.
        # We'll treat it as markdown for display and text for TTS.
        return {
            "markdown": response_text.strip(),
            "tts": response_text.strip()
        }
        
    return {
        "markdown": markdown_content or "",
        "tts": tts_content or ""
    }
