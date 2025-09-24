from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    event,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class Portfolio(Base):
    """Portfolio model representing a collection of cryptocurrency assets."""

    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    assets = relationship(
        "Asset", back_populates="portfolio", cascade="all, delete-orphan"
    )


class Asset(Base):
    """Asset model representing a cryptocurrency holding within a portfolio."""

    __tablename__ = "assets"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)  # e.g., 'BTCUSDT', 'BTC'
    quantity = Column(Float, nullable=False)
    average_buy_price = Column(Float, nullable=False)
    total_spent = Column(Float, nullable=False)  # quantity * average_buy_price
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="assets")
    transactions = relationship(
        "Transaction", back_populates="asset", cascade="all, delete-orphan"
    )
    take_profit_levels = relationship(
        "TakeProfitLevel", back_populates="asset", cascade="all, delete-orphan"
    )


class Transaction(Base):
    """Transaction model recording buy/sell operations for assets."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    transaction_type = Column(String(10), nullable=False)  # 'buy', 'sell'
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)  # quantity * price
    timestamp = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)

    asset = relationship("Asset", back_populates="transactions")


class TrackedAsset(Base):
    """Central table to track all assets that need price/data updates."""

    __tablename__ = "tracked_assets"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True)
    source = Column(String(20), nullable=False)  # 'portfolio', 'watchlist', 'both'
    is_active = Column(Boolean, default=True)  # Whether to actively fetch data
    last_price_update = Column(DateTime, nullable=True)  # Last successful price fetch
    last_historical_update = Column(
        DateTime, nullable=True
    )  # Last historical data fetch

    # Data provider tracking
    preferred_data_provider = Column(
        String(20), nullable=True
    )  # 'BinanceProvider', 'BybitProvider', etc.
    provider_success_count = Column(
        Integer, default=0
    )  # How many times preferred provider succeeded
    provider_fail_count = Column(
        Integer, default=0
    )  # How many times preferred provider failed
    last_provider_success = Column(
        DateTime, nullable=True
    )  # Last time preferred provider worked

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<TrackedAsset(symbol='{self.symbol}', source='{self.source}', active={self.is_active}, provider='{self.preferred_data_provider}')>"


class Watchlist(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)


class ProfitHistory(Base):
    __tablename__ = "profit_history"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    quantity_sold = Column(Float, nullable=False)
    sell_price = Column(Float, nullable=False)
    average_buy_price = Column(Float, nullable=False)
    realized_profit = Column(
        Float, nullable=False
    )  # (sell_price - avg_buy_price) * quantity
    profit_percentage = Column(Float, nullable=False)
    sold_at = Column(DateTime, default=datetime.utcnow)
    portfolio_name = Column(String(100), nullable=False)


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True)
    setting_key = Column(String(100), nullable=False, unique=True)
    setting_value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TakeProfitLevel(Base):
    __tablename__ = "take_profit_levels"

    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    target_price = Column(Float, nullable=False)
    percentage_to_sell = Column(Float, nullable=False)  # 0-100%
    strategy_type = Column(
        String(20), nullable=False
    )  # 'fixed', 'dca_out', 'fibonacci', 'custom'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    triggered_at = Column(DateTime, nullable=True)  # When the level was hit
    notes = Column(Text)

    # Relationships
    asset = relationship("Asset", back_populates="take_profit_levels")


class CachedPrice(Base):
    __tablename__ = "cached_prices"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True)
    price = Column(Float, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<CachedPrice(symbol='{self.symbol}', price={self.price}, last_updated='{self.last_updated}')>"


class HistoricalPrice(Base):
    """Store historical price data for portfolio value calculations."""

    __tablename__ = "historical_prices"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False)  # Date of this price point
    interval = Column(String(10), nullable=False)  # '1h', '4h', '1d', '1w', '1M'

    # OHLC data - with backward compatibility
    open_price = Column(Float, nullable=True)
    high_price = Column(Float, nullable=True)
    low_price = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    price = Column(
        Float, nullable=False
    )  # Kept for backward compatibility (usually = close_price)
    volume = Column(Float, default=0)

    def __repr__(self):
        return f"<HistoricalPrice(symbol='{self.symbol}', price={self.price}, date='{self.date}')>"


class PortfolioValueHistory(Base):
    """Store calculated portfolio values over time."""

    __tablename__ = "portfolio_value_history"

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    total_value = Column(Float, nullable=False)
    total_invested = Column(Float, nullable=False)
    total_pnl = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    portfolio = relationship("Portfolio")

    def __repr__(self):
        return f"<PortfolioValueHistory(portfolio_id={self.portfolio_id}, date='{self.date}', value={self.total_value})>"


class AlertHistory(Base):
    """Store alert history and notifications."""

    __tablename__ = "alert_history"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    alert_type = Column(
        String(50), nullable=False
    )  # 'PRICE_CHANGE', 'SIGNAL_CHANGE', 'TP_TRIGGERED', etc.
    message = Column(Text, nullable=False)
    severity = Column(String(10), default="MEDIUM")  # 'LOW', 'MEDIUM', 'HIGH'
    triggered_at = Column(DateTime, default=datetime.utcnow)
    is_read = Column(Boolean, default=False)
    dismissed_at = Column(DateTime, nullable=True)

    # Additional data as JSON string
    alert_data = Column(Text)  # Store JSON data like old_price, new_price, etc.

    def __repr__(self):
        return f"<AlertHistory(symbol='{self.symbol}', type='{self.alert_type}', triggered_at='{self.triggered_at}')>"


class NotificationSettings(Base):
    """Enhanced notification settings storage."""

    __tablename__ = "notification_settings"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), default="default")  # Future multi-user support

    # Email settings
    email_enabled = Column(Boolean, default=False)
    email_address = Column(String(255))
    smtp_server = Column(String(255))
    smtp_port = Column(Integer, default=587)
    smtp_username = Column(String(255))
    smtp_password = Column(String(255))  # Should be encrypted in production

    # Desktop settings
    desktop_enabled = Column(Boolean, default=False)

    # WhatsApp settings
    whatsapp_enabled = Column(Boolean, default=False)
    whatsapp_number = Column(String(20))
    whatsapp_api_key = Column(String(255))

    # Alert thresholds
    price_change_threshold = Column(Float, default=5.0)
    signal_change_alerts = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<NotificationSettings(email_enabled={self.email_enabled}, desktop_enabled={self.desktop_enabled})>"


class TechnicalIndicatorCache(Base):
    """Cache calculated technical indicators."""

    __tablename__ = "technical_indicator_cache"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    indicator_type = Column(
        String(20), nullable=False
    )  # 'RSI', 'MACD', 'SMA', 'EMA', etc.
    timeframe = Column(String(10), nullable=False)  # '1h', '4h', '1d'
    period = Column(Integer, nullable=False)  # Period used for calculation
    value = Column(Float, nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<TechnicalIndicatorCache(symbol='{self.symbol}', type='{self.indicator_type}', value={self.value})>"


# Database setup
def get_database_url():
    """
    Get database URL, supporting configurable database folder.

    Returns:
        str: SQLite database URL with proper path configuration
    """
    import streamlit as st

    # Check for environment variable override (for testing)
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(__file__))

    # Get database folder from settings (default: 'db_data')
    try:
        if hasattr(st, "session_state") and hasattr(st.session_state, "settings"):
            database_folder = st.session_state.settings.get(
                "database_folder", "db_data"
            )
        else:
            database_folder = "db_data"
    except:
        database_folder = "db_data"

    # Create database folder if it doesn't exist
    db_folder_path = os.path.join(project_root, database_folder)
    os.makedirs(db_folder_path, exist_ok=True)

    # Construct database file path
    db_path = os.path.join(db_folder_path, "portfolio.db")

    # Check for legacy database in root and offer migration
    legacy_db_path = os.path.join(project_root, "portfolio.db")
    if os.path.exists(legacy_db_path) and not os.path.exists(db_path):
        try:
            import shutil

            shutil.copy2(legacy_db_path, db_path)
            logger.info(f"Migrated database from {legacy_db_path} to {db_path}")
        except Exception as e:
            logger.warning(f"Could not migrate database: {e}")
            # Fall back to legacy location
            db_path = legacy_db_path

    return f"sqlite:///{db_path}"


def get_database_info():
    """Get information about the current database location and size."""
    import streamlit as st

    project_root = os.path.dirname(os.path.dirname(__file__))

    # Get current database folder setting
    try:
        if hasattr(st, "session_state") and hasattr(st.session_state, "settings"):
            database_folder = st.session_state.settings.get(
                "database_folder", "db_data"
            )
        else:
            database_folder = "db_data"
    except:
        database_folder = "db_data"

    # Get current database path
    db_folder_path = os.path.join(project_root, database_folder)
    db_path = os.path.join(db_folder_path, "portfolio.db")

    # Check if legacy database exists
    legacy_db_path = os.path.join(project_root, "portfolio.db")

    info = {
        "database_folder": database_folder,
        "db_folder_path": db_folder_path,
        "db_path": db_path,
        "db_exists": os.path.exists(db_path),
        "legacy_db_exists": os.path.exists(legacy_db_path),
        "db_size": 0,
        "db_size_human": "0 B",
    }

    # Get database size
    if info["db_exists"]:
        try:
            size_bytes = os.path.getsize(db_path)
            info["db_size"] = size_bytes

            # Convert to human-readable format
            for unit in ["B", "KB", "MB", "GB"]:
                if size_bytes < 1024.0:
                    info["db_size_human"] = f"{size_bytes:.1f} {unit}"
                    break
                size_bytes /= 1024.0
        except:
            pass

    return info


def create_database():
    """
    Create database and tables with optimized SQLite settings.

    Returns:
        Engine: SQLAlchemy engine instance
    """
    db_url = get_database_url()

    # Create engine with WAL mode and connection pooling for better concurrency
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={
            "check_same_thread": False,
            "timeout": 30,
            "isolation_level": None,  # Autocommit mode
        },
        echo=False,  # Set to True for SQL debugging
    )

    # Enable WAL mode for better concurrent access
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


# Global engine instance for connection reuse
_engine = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        _engine = create_database()
    return _engine


def get_session():
    """Get database session with proper error handling."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def migrate_tracked_assets():
    """Initialize new provider tracking columns for existing TrackedAsset records."""
    try:
        # Use a fresh engine to avoid recursion issues
        engine = get_engine()
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=engine)
        session = Session()

        # Check if the table and columns exist
        from sqlalchemy import inspect

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if "tracked_assets" not in tables:
            logger.info("TrackedAsset table doesn't exist yet, skipping migration")
            session.close()
            return

        columns = [col["name"] for col in inspector.get_columns("tracked_assets")]
        if "preferred_data_provider" not in columns:
            logger.info("Provider tracking columns don't exist yet, skipping migration")
            session.close()
            return

        # Find TrackedAsset records with NULL provider tracking fields that need initialization
        assets_to_update = (
            session.query(TrackedAsset)
            .filter(TrackedAsset.provider_success_count == None)
            .all()
        )

        if assets_to_update:
            logger.info(
                f"Initializing provider tracking for {len(assets_to_update)} tracked assets"
            )

            for asset in assets_to_update:
                # Initialize new columns with default values
                if asset.provider_success_count is None:
                    asset.provider_success_count = 0
                if asset.provider_fail_count is None:
                    asset.provider_fail_count = 0
                # preferred_data_provider and last_provider_success remain None until first success

            session.commit()
            logger.info("Successfully initialized provider tracking columns")
        else:
            logger.info("All tracked assets already have provider tracking initialized")

        session.close()

    except Exception as e:
        logger.warning(f"Could not migrate tracked assets: {e}")
        try:
            session.close()
        except:
            pass
