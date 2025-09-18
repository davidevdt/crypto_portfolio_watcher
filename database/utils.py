"""
Database utilities for handling SQLite concurrency and retries
"""

import time
import logging
from contextlib import contextmanager
from typing import Callable, Any, Optional
from sqlalchemy.exc import OperationalError
from sqlalchemy import text
from .models import get_session

logger = logging.getLogger(__name__)


def retry_db_operation(func: Callable, max_retries: int = 3, delay: float = 0.1) -> Any:
    """
    Retry a database operation with exponential backoff.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (exponentially increased)

    Returns:
        Result of the function call

    Raises:
        OperationalError: If all retries are exhausted
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries:
                wait_time = delay * (2**attempt)
                logger.warning(
                    f"Database locked, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                time.sleep(wait_time)
                continue
            else:
                logger.error(
                    f"Database operation failed after {attempt + 1} attempts: {e}"
                )
                raise
        except Exception as e:
            logger.error(f"Non-recoverable database error: {e}")
            raise


@contextmanager
def get_db_session_with_retry(max_retries: int = 3):
    """
    Context manager for database sessions with automatic retry and proper cleanup.

    Args:
        max_retries: Maximum number of retry attempts for locked database

    Usage:
        with get_db_session_with_retry() as session:
            result = session.query(Model).all()
    """
    session = None
    for attempt in range(max_retries + 1):
        try:
            session = get_session()
            yield session
            session.commit()
            break
        except OperationalError as e:
            if session:
                session.rollback()
                session.close()
                session = None

            if "database is locked" in str(e) and attempt < max_retries:
                wait_time = 0.1 * (2**attempt)
                logger.warning(
                    f"Database locked, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                time.sleep(wait_time)
                continue
            else:
                logger.error(
                    f"Database session failed after {attempt + 1} attempts: {e}"
                )
                raise
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Non-recoverable database error: {e}")
            raise
        finally:
            if session:
                session.close()


def safe_db_operation(operation: Callable, *args, **kwargs) -> Optional[Any]:
    """
    Safely execute a database operation with retry logic and error handling.

    Args:
        operation: Database operation function to execute
        *args, **kwargs: Arguments to pass to the operation

    Returns:
        Result of the operation, or None if it failed
    """
    try:
        return retry_db_operation(lambda: operation(*args, **kwargs))
    except Exception as e:
        logger.error(f"Database operation failed permanently: {e}")
        return None


@contextmanager
def optimized_db_session():
    """
    Context manager that provides an optimized database session for better concurrency.
    Uses shorter transactions and proper connection management.
    """
    session = None
    try:
        session = get_session()

        # Set session-level optimizations
        session.execute(text("PRAGMA busy_timeout=10000"))  # 10 second timeout

        yield session
        session.commit()

    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        if session:
            session.close()


def check_database_health() -> bool:
    """
    Check if the database is accessible and responsive.

    Returns:
        True if database is healthy, False otherwise
    """
    try:
        with get_db_session_with_retry(max_retries=1) as session:
            # Simple query to test database connectivity
            session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def get_database_info() -> dict:
    """
    Get information about the current database state.

    Returns:
        Dictionary with database information
    """
    info = {
        "healthy": False,
        "journal_mode": "unknown",
        "busy_timeout": "unknown",
        "cache_size": "unknown",
    }

    try:
        with get_db_session_with_retry(max_retries=1) as session:
            info["healthy"] = True

            # Get journal mode
            result = session.execute(text("PRAGMA journal_mode")).fetchone()
            if result:
                info["journal_mode"] = result[0]

            # Get busy timeout
            result = session.execute(text("PRAGMA busy_timeout")).fetchone()
            if result:
                info["busy_timeout"] = f"{result[0]}ms"

            # Get cache size
            result = session.execute(text("PRAGMA cache_size")).fetchone()
            if result:
                info["cache_size"] = result[0]

    except Exception as e:
        logger.error(f"Failed to get database info: {e}")

    return info
