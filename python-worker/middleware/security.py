"""
Security middleware for FastAPI application.
Includes API key verification, IP whitelist, and request logging.
"""

import logging
import time
from ipaddress import ip_address, ip_network, IPv4Address, IPv6Address
from typing import List, Union, Callable
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to verify API key in request headers."""

    def __init__(self, app, api_key: str, exempt_paths: List[str] = None):
        """
        Initialize API Key middleware.

        Args:
            app: FastAPI application
            api_key: Valid API key
            exempt_paths: List of paths exempt from API key check (e.g., ["/health"])
        """
        super().__init__(app)
        self.api_key = api_key
        self.exempt_paths = exempt_paths or ["/health", "/docs", "/redoc", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Verify API key for each request.

        Args:
            request: FastAPI Request object
            call_next: Next middleware/handler

        Returns:
            Response or HTTPException
        """
        # Skip API key check for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key")

        # Verify API key
        if not api_key or api_key != self.api_key:
            client_host = request.client.host if request.client else "unknown"
            logger.warning(f"Invalid API key attempt from {client_host}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"}
            )

        # API key valid, continue
        return await call_next(request)


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """Middleware to verify client IP against whitelist."""

    def __init__(self, app, allowed_ips: List[str], exempt_paths: List[str] = None):
        """
        Initialize IP Whitelist middleware.

        Args:
            app: FastAPI application
            allowed_ips: List of allowed IPs or CIDR ranges (e.g., ["127.0.0.1", "192.168.1.0/24"])
            exempt_paths: List of paths exempt from IP check (e.g., ["/health"])
        """
        super().__init__(app)
        self.exempt_paths = set(exempt_paths or ["/health"])

        # Pre-parse IPs and networks at startup for fast lookup
        self._allowed_addresses = set()
        self._allowed_networks = []

        for entry in allowed_ips:
            entry = entry.strip()
            try:
                if "/" in entry:
                    self._allowed_networks.append(ip_network(entry, strict=False))
                else:
                    self._allowed_addresses.add(ip_address(entry))
            except ValueError as e:
                logger.warning(f"Skipping invalid IP/CIDR entry '{entry}': {e}")

        logger.info(
            f"IP Whitelist enabled: {len(self._allowed_addresses)} IPs, "
            f"{len(self._allowed_networks)} CIDR ranges"
        )

    def is_ip_allowed(self, client_ip: str) -> bool:
        """
        Check if client IP is in whitelist.

        Supports both individual IPs and CIDR notation.
        IPs and networks are pre-parsed at startup for performance.

        Args:
            client_ip: Client IP address

        Returns:
            True if IP is allowed, False otherwise
        """
        try:
            client_addr = ip_address(client_ip)

            # Fast exact-match check first
            if client_addr in self._allowed_addresses:
                return True

            # Then check CIDR ranges
            for network in self._allowed_networks:
                if client_addr in network:
                    return True

            return False

        except ValueError as e:
            logger.error(f"Invalid IP address format: {client_ip} - {e}")
            return False

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Verify client IP for each request.

        Args:
            request: FastAPI Request object
            call_next: Next middleware/handler

        Returns:
            Response or HTTPException
        """
        # Skip IP check for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Check if IP is allowed
        if not self.is_ip_allowed(client_ip):
            logger.warning(f"Access denied for IP: {client_ip} to {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": f"Access forbidden for IP: {client_ip}"}
            )

        # IP is allowed, continue
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests."""

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Log request details and response time.

        Args:
            request: FastAPI Request object
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Record start time
        start_time = time.time()

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log request
        logger.info(
            f"[{client_ip}] {request.method} {request.url.path} "
            f"→ {response.status_code} ({duration:.2f}s)"
        )

        return response


# Helper functions for use in route handlers

def verify_api_key(request: Request, api_key: str):
    """
    Verify API key from request header.

    Args:
        request: FastAPI Request object
        api_key: Expected API key

    Raises:
        HTTPException: If API key is invalid or missing

    Example:
        >>> from fastapi import Depends
        >>> @app.get("/protected")
        >>> async def protected_route(request: Request):
        ...     verify_api_key(request, settings.API_KEY)
        ...     return {"status": "ok"}
    """
    request_api_key = request.headers.get("X-API-Key")

    if not request_api_key or request_api_key != api_key:
        client_host = request.client.host if request.client else "unknown"
        logger.warning(f"Invalid API key attempt from {client_host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )


def verify_ip_whitelist(request: Request, allowed_ips: List[str]):
    """
    Verify client IP against whitelist.

    Args:
        request: FastAPI Request object
        allowed_ips: List of allowed IPs or CIDR ranges

    Raises:
        HTTPException: If IP is not allowed

    Example:
        >>> @app.get("/protected")
        >>> async def protected_route(request: Request):
        ...     verify_ip_whitelist(request, ["127.0.0.1", "192.168.1.0/24"])
        ...     return {"status": "ok"}
    """
    client_ip = request.client.host if request.client else "unknown"

    try:
        client_addr = ip_address(client_ip)
        is_allowed = False

        for allowed in allowed_ips:
            if "/" in allowed:
                network = ip_network(allowed, strict=False)
                if client_addr in network:
                    is_allowed = True
                    break
            else:
                if client_addr == ip_address(allowed):
                    is_allowed = True
                    break

        if not is_allowed:
            logger.warning(f"Access denied for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden for IP: {client_ip}"
            )

    except ValueError as e:
        logger.error(f"Invalid IP address format: {client_ip} - {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid IP address format"
        )
