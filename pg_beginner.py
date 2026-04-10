import psycopg2

# Connect to your database
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="localhost"
)

def create_table():
    # Create a table with a vector column
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id serial PRIMARY KEY,
                embedding vector(3)
            );
        """)
        conn.commit()
        print("postgres.items table created successfully.\n")

def inser_record(id: int, embedding):
    with conn.cursor() as cur:
        try:        
            cur.execute(f"INSERT INTO items(id, embedding) values ({id}, '{embedding}')")
            conn.commit()
            print("Record inserted successfully in postgres.items table")
        except Exception as e:
            print(f"Failed to insert record due to:>>  {e}")
        conn.commit()

def fetchall_print():
    print("\nRecords in postgres.items:: \n")
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM items")
        for cur in cur.fetchall():
            print(cur);


create_table()
inser_record(6, [7,65,7])
fetchall_print()