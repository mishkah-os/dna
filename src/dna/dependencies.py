"""
🔒 FastAPI Dependencies for Authentication
"""

from typing import Optional
from fastapi import Depends, HTTPException, Header, status
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import AsyncSessionLocal
from database.models import User
from src.dna.auth import get_auth_service


# ============================================================================
# Database Dependency
# ============================================================================

async def get_db():
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ============================================================================
# Token Extraction
# ============================================================================

async def get_token_from_header(authorization: Optional[str] = Header(None)) -> str:
    """
    Extract JWT token from Authorization header.
    
    Expected format: "Bearer <token>"
    
    Raises:
        HTTPException(401): If no authorization header or invalid format
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Use Bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================================
# Current User Dependencies
# ============================================================================

async def get_current_user(
    token: str = Depends(get_token_from_header),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Returns:
        User object
        
    Raises:
        HTTPException(401): If token invalid or user not found
    """
    auth_service = get_auth_service()
    
    try:
        # Verify and decode token
        payload = auth_service.verify_token(token)
        user_id = int(payload.get("sub"))
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
    except (JWTError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    from sqlalchemy import select
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    
    Alias for get_current_user (already checks is_active)
    """
    return current_user


# ============================================================================
# Role-Based Dependencies
# ============================================================================

async def require_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Require admin role.
    
    Use this dependency to protect admin-only endpoints.
    
    Raises:
        HTTPException(403): If user is not admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def require_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Require any authenticated user (admin or regular user).
    
    Alias for get_current_user.
    """
    return current_user


# ============================================================================
# Optional Authentication
# ============================================================================

async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    Use for endpoints that work with or without authentication.
    """
    if not authorization:
        return None
    
    try:
        token = await get_token_from_header(authorization)
        return await get_current_user(token, db)
    except HTTPException:
        return None
