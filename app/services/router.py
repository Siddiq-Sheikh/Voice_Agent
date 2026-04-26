import os
import time
from groq import AsyncGroq

# Initialize the blazing-fast Groq client
groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

async def requires_vision(user_input: str) -> str:
    """
    Ultra-fast intent classifier using Few-Shot Prompting.
    Returns: 'VISION_SINGLE', 'VISION_FULL', or 'NO'.
    """
    router_prompt = (
        "You are a strict routing AI for a voice assistant. Your ONLY job is to classify the user's intent into exactly ONE of three categories: 'VISION_SINGLE', 'VISION_FULL', or 'NO'.\n\n"
        
        "CATEGORIES:\n"
        "1. VISION_SINGLE: The user wants you to look at their current screen, a specific error, a graph, or a piece of code currently visible.\n"
        "2. VISION_FULL: The user explicitly asks you to read a WHOLE page, scroll through a document, or scan an entire file/codebase.\n"
        "3. NO: The user is just chatting, asking a general knowledge question, or asking about a database.\n\n"
        
        "EXAMPLES:\n"
        "User: 'What is on my screen right now?'\n"
        "Decision: VISION_SINGLE\n\n"
        
        "User: 'Why is this Python code throwing a syntax error?'\n"
        "Decision: VISION_SINGLE\n\n"
        
        "User: 'Can you scroll down and read this whole document to me?'\n"
        "Decision: VISION_FULL\n\n"
        
        "User: 'Check the whole code file for bugs.'\n"
        "Decision: VISION_FULL\n\n"
        
        "User: 'What time is it in London?'\n"
        "Decision: NO\n\n"
        
        "User: 'How much does Marcus make?'\n"
        "Decision: NO\n\n"
        
        "User: 'Can you write a python script for a calculator?'\n"
        "Decision: NO\n\n"
        
        f"User: '{user_input}'\n"
        "Decision:"
    )

    try:
        start_time = time.perf_counter()
        
        response = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0.0, # Zero creativity. Just math.
            max_completion_tokens=5 
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        print(f">> [Router Latency] Intent check took: {latency:.2f}ms")
        
        decision = response.choices[0].message.content.strip().upper()
        
        # Strict parsing to ensure it never crashes the downstream logic
        if "VISION_FULL" in decision:
            return "VISION_FULL"
        elif "VISION_SINGLE" in decision:
            return "VISION_SINGLE"
        else:
            return "NO"

    except Exception as e:
        print(f">> [Router Error] {e}")
        return "NO"