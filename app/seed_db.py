import asyncio
import asyncpg
from datetime import date
from app.core.config import settings  # <-- Import your secure settings

async def seed_database():
    print(">> [DB Seeder] Connecting to PostgreSQL...")
    
    # Dynamically build the DSN using your hidden .env variables
    dsn = f"postgresql://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    
    try:
        conn = await asyncpg.connect(dsn)
        print(">> [DB Seeder] 🟢 Connected Successfully.")

        # 1. Create the HR Employees Table
        print(">> [DB Seeder] Building 'employees' table...")
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                department VARCHAR(50) NOT NULL,
                role VARCHAR(50) NOT NULL,
                salary INTEGER NOT NULL,
                hire_date DATE NOT NULL
            )
        ''')

        # 2. Clear out any old data
        await conn.execute('TRUNCATE TABLE employees RESTART IDENTITY;')

        # 3. Inject the Dummy HR Data
        print(">> [DB Seeder] Injecting employee records...")
        
        await conn.executemany('''
            INSERT INTO employees (name, department, role, salary, hire_date) 
            VALUES ($1, $2, $3, $4, $5)
        ''', [
            ("Sarah Connor", "Engineering", "Lead AI Engineer", 150000, date(2023, 1, 15)),
            ("Marcus Wright", "Engineering", "Backend Developer", 120000, date(2023, 6, 22)),
            ("Grace Harper", "Product", "Product Manager", 135000, date(2022, 11, 1)),
            ("Miles Dyson", "Research", "Director of AI", 200000, date(2021, 8, 10)),
            ("Dani Ramos", "Operations", "HR Manager", 95000, date(2024, 2, 15)),
            ("T-Bob", "Support", "IT Specialist", 80000, date(2024, 5, 20))
        ])

        # Verify the injection
        count = await conn.fetchval('SELECT COUNT(*) FROM employees')
        print(f">> [DB Seeder] 🎉 Success! Database is now loaded with {count} employee records.")

    except Exception as e:
        print(f">> [DB Seeder] 🔴 Error: {e}")
    finally:
        if 'conn' in locals():
            await conn.close()

if __name__ == "__main__":
    asyncio.run(seed_database())