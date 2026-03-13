import asyncpg

class DatabaseService:
    def __init__(self, db_user, db_pass, db_name, db_host="localhost", db_port=5432):
        # We build the DSN using the variables passed from settings.py
        self.dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        self.pool = None

    async def connect(self):
        """Creates an async connection pool to PostgreSQL."""
        try:
            # On some laptops, 'localhost' resolves to IPv6 (::1), 
            # while Postgres listens on IPv4 (127.0.0.1). 
            # If it fails, you can try replacing localhost with 127.0.0.1 in .env
            self.pool = await asyncpg.create_pool(self.dsn)
            print(">> [DB] 🟢 PostgreSQL Async Pool Ready.")
        except Exception as e:
            print(f">> [DB] 🔴 Connection Failed: {e}")

    async def execute_query(self, query: str):
        """Executes the AI's SQL query and returns the results."""
        if not self.pool:
            return "Database not connected."
            
        # SECURITY: Read-only guardrail
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE"]
        clean_query = query.strip().upper()
        
        if not clean_query.startswith("SELECT") or any(k in clean_query for k in forbidden_keywords):
            return "Error: I am only allowed to read data (SELECT queries)."

        try:
            async with self.pool.acquire() as conn:
                records = await conn.fetch(query)
                # Convert asyncpg records to a standard list of dictionaries
                return [dict(record) for record in records]
        except Exception as e:
            return f"Database Error: {str(e)}"