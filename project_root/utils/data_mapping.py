import json

def map_data(master_id, objects_data):
    # Map all extracted data and attributes to each object and the master input image
    mapped_data = {
        "master_id": master_id,
        "objects": objects_data
    }
    return json.dumps(mapped_data)