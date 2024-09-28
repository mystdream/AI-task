import json
import sqlite3
import pandas as pd

def map_data_to_objects(conn):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT object_id, master_id, filepath, object_class, extracted_text, summary
        FROM objects
    ''')
    columns = [column[0] for column in cursor.description]
    results = []
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row)))
    return results

def create_json_mapping(mapped_data):
    return json.dumps(mapped_data, indent=2)

def create_dataframe(mapped_data):
    return pd.DataFrame(mapped_data)
