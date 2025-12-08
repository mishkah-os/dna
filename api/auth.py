"""
🔐 Authentication API
Login, logout, and user info endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from database.models import User
from src.dna.auth import get_auth_service
from src.dna.dependencies import get_db, get_current_user

router = APIRouter()


# ============================================================================
# Schemas
# ============================================================================

class LoginRequest(BaseModel):
    """Login request"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response"""
    token: str
    token_type: str = "bearer"
    role: str
    username: str
    email: str


class UserInfo(BaseModel):
    """User information"""
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str
    last_login: str = None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/auth/login", response_model=LoginResponse)
async def login(
    credentials: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Login endpoint.
    
    Returns JWT token on successful authentication.
    
    - **username**: Username
    - **password**: Password
    """
    auth_service = get_auth_service()
    
    # Find user by username
    result = await db.execute(
        select(User).where(User.username == credentials.username)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Verify password
    if not auth_service.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Check if active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()
    
    # Create token
    token = auth_service.create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role
    )
    
    return LoginResponse(
        token=token,
        role=user.role,
        username=user.username,
        email=user.email
    )


@router.post("/auth/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout endpoint.
    
    Currently just a placeholder (JWT tokens can't be invalidated).
    In production, implement token blacklist or use refresh tokens.
    """
    return {
        "message": "Logged out successfully",
        "username": current_user.username
    }


@router.get("/auth/me", response_model=UserInfo)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Get current user information.
    
    Returns authenticated user's profile.
    """
    return UserInfo(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None,
        last_login=current_user.last_login.isoformat() if current_user.last_login else None
    )


@router.post("/auth/verify-token")
async def verify_token(current_user: User = Depends(get_current_user)):
    """
    Verify if token is valid.
    
    Returns user info if token is valid, 401 otherwise.
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "role": current_user.role
    }
