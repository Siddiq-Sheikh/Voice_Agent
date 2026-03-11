import asyncpg

class DatabaseService:
    def __init__(self, db_user="postgres", db_pass="YOUR_PASSWORD", db_name="postgres"):
        self.dsn = f"postgresql://{db_user}:{db_pass}@localhost:5432/{db_name}"
        self.pool = None

    async def connect(self):
        """Creates an async connection pool to PostgreSQL."""
        try:
            self.pool = await asyncpg.create_pool(self.dsn)
            print(">> [DB] 🟢 PostgreSQL Async Pool Ready.")
        except Exception as e:
            print(f">> [DB] 🔴 Connection Failed: {e}")

    async def execute_query(self, query: str):
        """Executes the AI's SQL query and returns the results."""
        if not self.pool:
            return "Database not connected."
            
        # SECURITY: Prevent the AI from accidentally deleting your database
        if not query.strip().upper().startswith("SELECT"):
            return "Error: I am only allowed to read data (SELECT queries)."

        try:
            async with self.pool.acquire() as conn:
                records = await conn.fetch(query)
                # Convert asyncpg records to a standard list of dictionaries
                return [dict(record) for record in records]
        except Exception as e:
            return f"Database Error: {str(e)}"