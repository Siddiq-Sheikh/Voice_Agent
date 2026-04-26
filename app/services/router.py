import os
import time
from groq import AsyncGroq

# Initialize the Groq client
groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

async def requires_vision(user_input: str) -> str:
    """
    3-Way LLM Router: VLM (Images/UI) vs OCR (Text/Code) vs Normal Chat.
    """
    router_prompt = (
        "You are the central nervous system of an AI voice assistant. Your ONLY job is routing. "
        "Classify the user's intent into exactly ONE of three strict categories: 'VISION_SINGLE', 'READ_TEXT', or 'NO'.\n\n"
        
        "=== CATEGORY DEFINITIONS ===\n"
        "1. READ_TEXT (OCR Extraction): \n"
        "   - Trigger when the user needs you to READ, ANALYZE, or EXPLAIN large blocks of TEXT or CODE.\n"
        "   - Keywords: 'this code', 'read the page', 'what does this script do', 'check this document for errors', 'find the bug'.\n"
        
        "2. VISION_SINGLE (Visual AI Analysis): \n"
        "   - Trigger when the user wants you to look at a UI ELEMENT, GRAPH, PICTURE, or general SCREEN STATE.\n"
        "   - Keywords: 'what is this image', 'explain this graph', 'what UI button is this', 'what am i looking at'.\n"
        "   - CRITICAL: Do NOT use this for reading code or long text.\n"

        "3. NO (Normal Chat & Generation): \n"
        "   - Trigger when the request requires GENERATION or general knowledge, not observation.\n"
        "   - Keywords: 'write a script', 'what time is it', 'query the db', 'hello'.\n\n"
        
        "=== EXAMPLES ===\n"
        "User: 'Why is this Python code throwing a syntax error?'\n"
        "Decision: READ_TEXT\n\n"
        
        "User: 'Can you read this document to me?'\n"
        "Decision: READ_TEXT\n\n"
        
        "User: 'What is this graph showing?'\n"
        "Decision: VISION_SINGLE\n\n"
        
        "User: 'What is on my screen right now?'\n"
        "Decision: VISION_SINGLE\n\n"
        
        "User: 'Can you write a python script for a calculator?'\n"
        "Decision: NO\n\n"
        
        "User: 'Hello there!'\n"
        "Decision: NO\n\n"
        
        "=== CURRENT REQUEST ===\n"
        f"User: '{user_input}'\n"
        "Decision:"
    )

    try:
        start_time = time.perf_counter()
        
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0.0, 
            max_completion_tokens=5 
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        decision = response.choices[0].message.content.strip().upper()
        
        if "READ_TEXT" in decision:
            return "READ_TEXT"
        elif "VISION_SINGLE" in decision:
            return "VISION_SINGLE"
        else:
            return "NO"

    except Exception as e:
        print(f">> [Router Error] {e}")
        return "NO"