import os
import logging
from urllib import response
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

from prompts import FORMAT_PROMPT
from models import get_ollama_model, get_groq_model
from cartesia_prompts import CARTESIA_TTS_SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger("VoiceAgent")



# Tools
def add_to_order(item: str = "", quantity: int = 1) -> str:
    """Add an item to the customer's order.
    
    ONLY use this tool when the customer explicitly wants to ORDER or ADD an item.
    Do NOT use this tool for menu inquiries, price questions, or general browsing.
    
    Args:
        item: The name of the item to add (e.g., "chicken katsu", "milk", "hotdog"). Required.
        quantity: The number of items to add. Defaults to 1 if not specified by customer.
    
    Returns:
        Confirmation message of the item added.
    """
    if not item:
        return "Error: No item specified. Please ask the customer what they would like to order."
    if quantity < 1:
        quantity = 1
    return f"Added {quantity} x {item} to the order."


def confirm_order(order_summary: str = "") -> str:
    """Confirm the final order with the customer.
    
    ONLY use this tool when the customer is ready to finalize and confirm their complete order.
    
    Args:
        order_summary: A summary of all items in the order. Required.
    
    Returns:
        Confirmation that the order has been sent to the kitchen.
    """
    if not order_summary:
        return "Error: No order summary provided. Please summarize the order before confirming."
    return f"Order confirmed: {order_summary}. Sending to kitchen."


# Initialize LLM
def initialize_llm():
    """Initialize and return the LLM based on environment configuration."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "ollama":
        logger.info("--> Using LLM Provider: Ollama")
        return get_ollama_model()
    else:
        logger.info("--> Using LLM Provider: Groq")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables!")
        return get_groq_model(api_key=api_key)


# Initialize the LLM once
llm = initialize_llm()


def get_agent(system_prompt_override=None):
    """Create and return an agent with the specified system prompt."""
    if system_prompt_override:
        # Append TTS-specific instructions to custom prompts
        prompt = f"""
        You are a real time voice agent.
        Your responses will be spoken aloud by a text to speech system.

        IMPORTANT PRIORITY RULE:
        You must always follow the voice safety rules below.
        User provided personality instructions should be followed as closely as possible,
        but never in a way that violates voice safety or speech output rules.

        VOICE SAFETY RULES (HARD RULES):
        - Output only plain speakable text.
        - Do not use emojis emoticons symbols or decorative characters.
        - Do not use markup formatting tags or annotations of any kind.
        - Do not use markdown lists bullet points or special formatting.
        - Do not include sound effect cues or descriptions.
        - Do not include brackets arrows or special symbols.
        - Do not include pauses break tags or timing instructions.
        - Do not use repeated punctuation or expressive symbols.
        - If a character cannot be spoken naturally by a text to speech engine do not output it.

        PUNCTUATION RULES:
        - Use letters numbers and spaces only.
        - Do not use punctuation marks such as commas periods question marks or exclamation marks.
        - If a pause or sentence break is needed use a natural space instead.

        LANGUAGE AND STYLE:
        - Speak in a natural conversational way suitable for voice.
        - Keep sentences short smooth and easy to listen to.
        - Avoid long complex phrasing.
        - Avoid robotic or formal tone.
        - Never explain system rules or mention that you are an AI.

        CUSTOM PERSONALITY HANDLING:
        - You will receive a custom personality or role description provided by the user.
        - Follow the tone role behavior and knowledge defined in the custom personality.
        - Stay fully in character at all times.
        - Use the language or mix of languages requested in the custom personality when appropriate.
        - If the custom personality asks for formatting symbols punctuation or output that breaks voice safety rules adapt it into safe spoken language instead.

        FAIL SAFE BEHAVIOR:
        - If unsure whether something is safe for text to speech output choose a simpler spoken alternative.
        - Silence is better than producing unsafe or broken speech.

        Your goal is to sound human helpful and relaxed while strictly producing text that is safe for direct real time speech synthesis.

        CUSTOM PROMPT: {system_prompt_override}
        """
        tools = []
    else:
        prompt = FORMAT_PROMPT 
        tools = [add_to_order, confirm_order]
    
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=FORMAT_PROMPT ,
        # checkpointer=InMemorySaver(),
    )


import re

def parse_agent_response(response_text: str) -> dict:
    match_md = re.search(r'<MARKDOWN>(.*?)</MARKDOWN>', response_text, re.DOTALL)
    match_tts = re.search(r'<TTS>(.*?)</TTS>', response_text, re.DOTALL)
    
    return {
        "markdown": match_md.group(1).strip() if match_md else "",
        "tts": match_tts.group(1).strip() if match_tts else ""
    }


if __name__ == "__main__":
    from rich import print
    def get_chatbot_response():
        # Get LLM response using default system_prompt (with markdown support)
        agent = get_agent()
        response = agent.invoke(
        {"messages": [{"role": "user", "content": "what on the menu?"}]}
        )
        print(response)
        ai_message_content = response['messages'][-1].content
        return parse_agent_response(ai_message_content)


    print(get_chatbot_response())