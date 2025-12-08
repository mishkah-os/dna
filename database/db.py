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
    from database.models import Model, Experiment, Pattern, User, Session
    from src.dna.auth import get_auth_service
    from sqlalchemy import select
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print(f"[OK] Database created at: {DB_PATH}")
    
    # Create default users if they don't exist
    async with AsyncSessionLocal() as session:
        auth_service = get_auth_service()
        
        # Check if admin exists
        result = await session.execute(
            select(User).where(User.username == "admin")
        )
        if not result.scalar_one_or_none():
            # Create admin user
            admin_user = User(
                username="admin",
                email="admin@dna.local",
                hashed_password=auth_service.hash_password("admin123"),
                encrypted_password=auth_service.encrypt_password("admin123"),
                role="admin"
            )
            session.add(admin_user)
            print("[OK] Created default admin user (admin/admin123)")
        
        # Check if test user exists
        result = await session.execute(
            select(User).where(User.username == "user")
        )
        if not result.scalar_one_or_none():
            # Create test user
            test_user = User(
                username="user",
                email="user@dna.local",
                hashed_password=auth_service.hash_password("user123"),
                encrypted_password=auth_service.encrypt_password("user123"),
                role="user"
            )
            session.add(test_user)
            print("[OK] Created default user (user/user123)")
        
        await session.commit()


async def get_db():
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
