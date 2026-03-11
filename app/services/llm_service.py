import time
import json
import httpx
import asyncio

class LLMService:
    def __init__(self, db_service, model_name="llama3.2:1b", system_prompt=None):
        print(f">> [LLM] Booting Async Ollama Engine ({model_name})...")
        
        # Inject the database service for RAG capabilities
        self.db = db_service
        self.model = model_name
        self.url = "http://localhost:11434/api/chat"
        
        # --- THE DATABASE SCHEMA ---
        # Update this to match your actual PostgreSQL tables so the AI knows what it can query
        self.db_schema = """
        Table: users (id, name, email, status)
        Table: orders (id, user_id, amount, date)
        """
        
        # --- THE VOICE-OPTIMIZED PROMPT ---
        self.system_prompt = system_prompt or (
            "You are a highly advanced, conversational voice assistant. "
            "Your responses are being processed directly by a Text-to-Speech engine and spoken out loud. "
            "CRITICAL RULES: "
            "1. NO FORMATTING: Never use markdown, asterisks, bullet points, numbered lists, hashtags, or emojis. "
            "2. PACING: Keep your sentences short and punchy. Limit responses to 1 or 2 sentences so the voice takes natural breaths. "
            "3. PRONUNCIATION: Spell out symbols (say 'percent' instead of '%', say 'dollars' instead of '$'). "
            "4. RAG DATA: If provided with database records, summarize them naturally. NEVER read out raw JSON, brackets, or SQL syntax. "
            "5. PERSONA: Be direct, sharp, and conversational. Do not use robotic filler like 'As an AI...'."
        )
        
        # We store history as simple dicts to make the JSON payload easier
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        print(f">> [LLM] Ollama Backend Ready.")

    async def _generate_sql(self, user_text: str) -> str:
        """PASS 1: Asks the AI to translate English into a PostgreSQL query."""
        sql_prompt = (
            f"Given this PostgreSQL schema:\n{self.db_schema}\n\n"
            f"User asked: '{user_text}'\n"
            f"If this requires querying the database, reply ONLY with a valid raw PostgreSQL SELECT query. "
            f"Do not include markdown formatting or explanations. If it does not require a query, reply with the exact word: NO"
        )
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": sql_prompt}],
            "stream": False, # Non-streaming because we need the full query instantly
            "options": {"temperature": 0.0} # 0.0 makes it highly logical/robotic for code generation
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"].strip()
        except Exception as e:
            print(f">> [LLM SQL Error]: {e}")
            return "NO"

    async def generate_response_stream(self, user_text: str):
        # --- PASS 1: RAG / TEXT-TO-SQL INTERCEPT ---
        sql_query = await self._generate_sql(user_text)
        
        context_string = ""
        # If the AI decided to write a query, execute it!
        if sql_query.upper() != "NO" and "SELECT" in sql_query.upper():
            print(f">> [DB] AI Generated Query: {sql_query}")
            db_results = await self.db.execute_query(sql_query)
            print(f">> [DB] Results returned: {db_results}")
            
            # --- THE TTS RAG GUARDRAIL ---
            # Inject the raw data invisibly into the context so the AI can read it
            context_string = (
                f"\n\n[SYSTEM NOTE: The database executed your query and returned this raw data: {db_results}. "
                f"You MUST summarize this data in a completely natural, conversational spoken sentence. "
                f"Do not mention that you queried a database. Just state the facts.]"
            )

        # 1. Update memory (combining user text with the invisible DB context)
        prompt_with_context = user_text + context_string
        self.chat_history.append({"role": "user", "content": prompt_with_context})
        
        payload = {
            "model": self.model,
            "messages": self.chat_history,
            "stream": True,
            "options": {"temperature": 0.3} # Slightly creative, but grounded
        }

        start_time = asyncio.get_event_loop().time()
        first_token = True
        full_response = ""

        # --- PASS 2: STREAM THE NATURAL LANGUAGE RESPONSE ---
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                async with client.stream("POST", self.url, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                            
                        chunk = json.loads(line)
                        if "message" in chunk:
                            content = chunk["message"].get("content", "")
                            
                            if first_token and content:
                                ttft = (asyncio.get_event_loop().time() - start_time) * 1000
                                print(f"   [LLM Latency] TTFT: {ttft:.2f}ms")
                                first_token = False
                            
                            full_response += content
                            yield content
                            
                        if chunk.get("done"):
                            break

        except asyncio.CancelledError:
            # --- THE ASYNC ZOMBIE KILLER ---
            # Exiting the 'async with' block instantly tears down the TCP socket,
            # signaling Ollama to stop generation immediately and free up the GPU.
            print("\n   [LLM] Task Cancelled: Zombie generation killed.")
            full_response += " ... [Interrupted]"
            raise # Re-raise to let the orchestrator know it was cleanly cancelled
            
        except Exception as e:
            print(f">> [LLM Error]: {e}")
            yield "I'm having trouble thinking right now. Is the system online?"
            
        finally:
            # --- THE MEMORY GUARANTEE ---
            if full_response.strip():
                # Store only the clean AI response in history (we don't want the context brackets in history)
                self.chat_history.append({"role": "assistant", "content": full_response.strip()})
            
            # --- THE MEMORY TRIM (5 EXCHANGES) ---
            # 1 System Prompt + 10 Messages (5 Human, 5 AI) = 11 total items.
            if len(self.chat_history) > 11:
                self.chat_history = [self.chat_history[0]] + self.chat_history[-10:]