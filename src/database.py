# The command to show all about the database
schemacmd = """
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_type = 'BASE TABLE'
AND table_schema NOT IN ('information_schema', 'pg_catalog');
"""

import mysql.connector
from mysql.connector import Error

def run_sql_comparison_pipeline(query1, query2):
    # Configuration
    config = {
        "host": "localhost",
        "user": "root",
        "password": "password",
        "database": "atis"
    }
    
    results1 = None
    results2 = None
    
    # Connect to the database
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        # Execute the first query
        try:
            print(f"Executing query 1: {query1}")
            cursor.execute(query1)
            results1 = cursor.fetchall()
            print(f"Query 1 returned {len(results1)} rows")
        except Error as e:
            print(f"Error executing query 1: {e}")
        
        # Execute the second query
        try:
            print(f"Executing query 2: {query2}")
            cursor.execute(query2)
            results2 = cursor.fetchall()
            print(f"Query 2 returned {len(results2)} rows")
        except Error as e:
            print(f"Error executing query 2: {e}")
        
        # Close connections
        cursor.close()
        conn.close()
        
        # Compare results
        if results1 is not None and results2 is not None:
            # Sort results to make them order-independent
            sorted_results1 = sorted([tuple(row) for row in results1])
            sorted_results2 = sorted([tuple(row) for row in results2])
            
            if len(sorted_results1) != len(sorted_results2):
                print("RESULT: DIFFERENT - Queries returned different number of rows")
                print(f"Query 1: {len(results1)} rows")
                print(f"Query 2: {len(results2)} rows")
            elif sorted_results1 == sorted_results2:
                if results1 == results2:
                    print("RESULT: SAME - Both queries returned identical results in the same order")
                else:
                    print("RESULT: SAME DATA, DIFFERENT ORDER - Both queries returned the same data but in different order")
            else:
                print("RESULT: DIFFERENT - Queries returned different results")
            
            # Display results
            print("\nResults from Query 1:")
            for row in results1[:10]:  # Limit to first 10 rows
                print(row)
            
            if len(results1) > 10:
                print(f"...and {len(results1) - 10} more rows")
                
            print("\nResults from Query 2:")
            for row in results2[:10]:  # Limit to first 10 rows
                print(row)
            
            if len(results2) > 10:
                print(f"...and {len(results2) - 10} more rows")
        else:
            print("RESULT: CANNOT COMPARE - One or both queries failed to execute")
            
    except Error as e:
        print(f"Database connection error: {e}")