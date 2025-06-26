from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Database:
    _instance = None
    client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None

    def __init__(self):
        if Database._instance is not None:
            raise RuntimeError("Call get_instance() instead")
        Database._instance = self

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_db(cls) -> Optional[AsyncIOMotorDatabase]:
        """Get the MongoDB database instance"""
        return cls._instance._db if cls._instance else None

    async def init_app(self, app):
        """Initialize database connection with proper error handling"""
        try:
            mongo_uri = app.config.get('MONGO_URI')
            if not mongo_uri:
                error_msg = "MONGO_URI configuration is missing"
                logger.error(error_msg)
                raise ValueError(error_msg)

            self.client = AsyncIOMotorClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000  # 5 second timeout
            )

            # Get the database instance
            self._db = self.client.get_database('truthguard')

            # Test the connection
            try:
                await self._db.command('ping')
                logger.info(f"Successfully connected to MongoDB at: {mongo_uri}")
                return True
            except ConnectionFailure as e:
                error_msg = f"Could not connect to MongoDB. Please ensure MongoDB is running and accessible at: {mongo_uri}"
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e

        except ConnectionFailure as e:
            error_msg = f"Failed to connect to MongoDB: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to MongoDB: {e}"
            logger.error(error_msg)
            raise

    def close(self):
        """Close the database connection"""
        if self.client:
            self.client.close()
            self.client = None
            self._db = None
            logger.info("Database connection closed")

# Create a global instance
db = Database.get_instance()
