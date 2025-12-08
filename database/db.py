"""
Database connection and session management
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import declarative_base
from pathlib import Path

# Database file location
DB_PATH = Path(__file__).parent.parent / "data" / "dna_explorer.db"
DB_PATH.parent.mkdir(exist_ok=True)

# Create async engine
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"
engine = create_async_engine(DATABASE_URL, echo=False)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()


async def init_database():
    """Initialize database tables"""
    from database.models import Model, Experiment, Pattern
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print(f"[OK] Database created at: {DB_PATH}")


async def get_db():
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
