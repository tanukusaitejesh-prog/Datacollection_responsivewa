import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from typing import Any, Dict, Optional

load_dotenv()

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    db = None
    collection = None

    def __init__(self):
        uri = os.getenv("MONGODB_URL", "mongodb://127.0.0.1:27017")
        db_name = os.getenv("MONGODB_DB_NAME", "pose_data")
        
        try:
            self.client = AsyncIOMotorClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db["captures"]
            print(f"Connected to MongoDB: {uri} (DB: {db_name})")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")

mongo_db = MongoDB()

async def push_metadata_to_mongo(data: Dict[str, Any]):
    """Insert capture metadata into MongoDB 'captures' collection."""
    if mongo_db.collection is None:
        print("MongoDB collection not initialized.")
        return None
    
    try:
        # Use capture_id as the unique identifier if possible
        capture_id = data.get("capture_id")
        if capture_id:
            result = await mongo_db.collection.update_one(
                {"capture_id": capture_id},
                {"$set": data},
                upsert=True
            )
        else:
            result = await mongo_db.collection.insert_one(data)
        return result
    except Exception as e:
        print(f"Failed to push metadata to MongoDB: {e}")
        return None
