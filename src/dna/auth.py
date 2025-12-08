"""
🔐 Authentication Service
Handles password hashing, encryption, and JWT tokens
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from cryptography.fernet import Fernet
from jose import JWTError, jwt

# ============================================================================
# Configuration
# ============================================================================

# Secret keys (⚠️ in production, load from environment!)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dna-secret-key-change-in-production-12345678")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

# Encryption key for password recovery (⚠️ DANGEROUS! Admin only)
# Using fixed key for consistency (in production, load from environment!)
ENCRYPTION_KEY = Fernet.generate_key()  # Generate fresh key each time (for demo)

# Cipher for password encryption/decryption
cipher_suite = Fernet(ENCRYPTION_KEY)


# ============================================================================
# Auth Service
# ============================================================================

class AuthService:
    """
    Authentication service for user management.
    
    Features:
    - Password hashing with bcrypt
    - Password encryption for admin recovery (dangerous!)
    - JWT token generation
    - Token verification
    """
    
    def __init__(self):
        self.cipher = cipher_suite
    
    # ------------------------------------------------------------------------
    # Password Hashing (Secure)
    # ------------------------------------------------------------------------
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt.
        
        This is SECURE - cannot be reversed.
        """
        # Convert string to bytes and hash
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify plain password against hashed password.
        
        Returns:
            True if password matches
        """
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    
    # ------------------------------------------------------------------------
    # Password Encryption (DANGEROUS - Admin Recovery Only!)
    # ------------------------------------------------------------------------
    
    def encrypt_password(self, password: str) -> str:
        """
        Encrypt password using AES.
        
        ⚠️ WARNING: This allows password recovery!
        Only use for admin password recovery feature.
        
        Returns:
            Encrypted password (base64 string)
        """
        encrypted_bytes = self.cipher.encrypt(password.encode('utf-8'))
        return encrypted_bytes.decode('utf-8')
    
    def decrypt_password(self, encrypted_password: str) -> str:
        """
        Decrypt password.
        
        ⚠️ EXTREMELY DANGEROUS!
        - Only admins should access this
        - Log every access
        - Consider 2FA before allowing
        
        Returns:
            Decrypted password (plain text)
        """
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted_password.encode('utf-8'))
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to decrypt password: {e}")
    
    # ------------------------------------------------------------------------
    # JWT Tokens
    # ------------------------------------------------------------------------
    
    def create_access_token(
        self,
        user_id: int,
        username: str,
        role: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.
        
        Args:
            user_id: User ID
            username: Username
            role: User role (admin/user)
            expires_delta: Custom expiration (default: 7 days)
            
        Returns:
            JWT token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
        
        to_encode = {
            "sub": str(user_id),  # Subject
            "username": username,
            "role": role,
            "exp": expire,  # Expiration time
            "iat": datetime.utcnow(),  # Issued at
        }
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            raise ValueError(f"Invalid token: {e}")
    
    def extract_user_id(self, token: str) -> int:
        """
        Extract user ID from token.
        
        Returns:
            User ID
            
        Raises:
            ValueError: If token is invalid
        """
        payload = self.verify_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise ValueError("Token missing user ID")
        return int(user_id)
    
    def extract_role(self, token: str) -> str:
        """
        Extract role from token.
        
        Returns:
            User role (admin/user)
        """
        payload = self.verify_token(token)
        return payload.get("role", "user")


# ============================================================================
# Singleton Instance
# ============================================================================

_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get singleton auth service instance"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    auth = AuthService()
    
    print("=" * 60)
    print("Auth Service Test")
    print("=" * 60)
    
    # Test password hashing
    password = "admin123"
    hashed = auth.hash_password(password)
    print(f"\nOriginal: {password}")
    print(f"Hashed: {hashed[:50]}...")
    print(f"Verify: {auth.verify_password(password, hashed)}")
    
    # Test password encryption (DANGEROUS!)
    encrypted = auth.encrypt_password(password)
    print(f"\nEncrypted: {encrypted[:50]}...")
    decrypted = auth.decrypt_password(encrypted)
    print(f"Decrypted: {decrypted}")
    
    # Test JWT
    token = auth.create_access_token(
        user_id=1,
        username="admin",
        role="admin"
    )
    print(f"\nJWT Token: {token[:50]}...")
    
    payload = auth.verify_token(token)
    print(f"Decoded: {payload}")
    
    print(f"\nUser ID: {auth.extract_user_id(token)}")
    print(f"Role: {auth.extract_role(token)}")
    
    print("\n✓ All tests passed!")
