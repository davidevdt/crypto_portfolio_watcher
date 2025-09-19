from sqlalchemy.orm import Session
from database.models import (
    Portfolio,
    Asset,
    Transaction,
    ProfitHistory,
    TakeProfitLevel,
    Watchlist,
    UserSettings,
    CachedPrice,
    TrackedAsset,
    get_session,
)
from database.utils import (
    get_db_session_with_retry,
    safe_db_operation,
    retry_db_operation,
)
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Manages portfolio operations including CRUD for assets and portfolios.

    Provides comprehensive portfolio management functionality including
    creating/updating portfolios, managing assets, tracking transactions,
    and calculating portfolio performance metrics.
    """

    def __init__(self):
        pass

    def create_portfolio(self, name: str) -> Portfolio:
        """Create a new portfolio.

        Args:
            name: Portfolio name

        Returns:
            Created Portfolio object

        Raises:
            ValueError: If portfolio name is empty
        """
        if not name or not name.strip():
            raise ValueError("Portfolio name cannot be empty")

        name = name.strip()

        session = get_session()
        try:
            portfolio = Portfolio(name=name)
            session.add(portfolio)
            session.commit()
            session.refresh(portfolio)
            session.expunge(portfolio)

            return portfolio
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating portfolio {name}: {e}")
            raise
        finally:
            session.close()

    def get_all_portfolios(self) -> List[Portfolio]:
        """Get all portfolios.

        Returns:
            List of all Portfolio objects
        """
        session = get_session()
        try:
            return session.query(Portfolio).all()
        finally:
            session.close()

    def get_portfolio_by_id(self, portfolio_id: int) -> Optional[Portfolio]:
        """
        Get portfolio by ID.

        Args:
            portfolio_id: Portfolio ID to retrieve

        Returns:
            Portfolio object if found, None otherwise
        """
        session = get_session()
        try:
            return session.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        finally:
            session.close()

    def get_portfolio_by_name(self, name: str) -> Optional[Portfolio]:
        """
        Get portfolio by name.

        Args:
            name: Portfolio name to search for

        Returns:
            Portfolio object if found, None otherwise
        """
        session = get_session()
        try:
            return session.query(Portfolio).filter(Portfolio.name == name).first()
        finally:
            session.close()

    def add_asset(
        self,
        portfolio_id: int,
        symbol: str,
        quantity: float,
        buy_price: float,
        force_immediate_data_fetch: bool = True,
    ) -> Asset:
        """Add a new asset or update existing one."""
        session = get_session()
        try:
            # Check if asset already exists in portfolio
            existing_asset = (
                session.query(Asset)
                .filter(Asset.portfolio_id == portfolio_id, Asset.symbol == symbol)
                .first()
            )

            is_new_asset = existing_asset is None

            if existing_asset:
                # Update existing asset - calculate new average buy price
                total_spent_old = existing_asset.total_spent
                total_quantity_old = existing_asset.quantity

                new_total_spent = total_spent_old + (quantity * buy_price)
                new_total_quantity = total_quantity_old + quantity
                new_average_price = new_total_spent / new_total_quantity

                existing_asset.quantity = new_total_quantity
                existing_asset.average_buy_price = new_average_price
                existing_asset.total_spent = new_total_spent
                existing_asset.updated_at = datetime.utcnow()

                # Record transaction
                transaction = Transaction(
                    asset_id=existing_asset.id,
                    transaction_type="buy",
                    quantity=quantity,
                    price=buy_price,
                    total_value=quantity * buy_price,
                )
                session.add(transaction)

                session.commit()
                session.refresh(existing_asset)
                result_asset = existing_asset
            else:
                # Create new asset
                asset = Asset(
                    symbol=symbol,
                    quantity=quantity,
                    average_buy_price=buy_price,
                    total_spent=quantity * buy_price,
                    portfolio_id=portfolio_id,
                )
                session.add(asset)
                session.commit()
                session.refresh(asset)

                # Record initial transaction
                transaction = Transaction(
                    asset_id=asset.id,
                    transaction_type="buy",
                    quantity=quantity,
                    price=buy_price,
                    total_value=quantity * buy_price,
                )
                session.add(transaction)
                session.commit()

                result_asset = asset

            # Force immediate data fetch for new assets (excluding stablecoins)
            if (
                force_immediate_data_fetch
                and is_new_asset
                and symbol not in ["USDT", "USDC", "BUSD", "DAI", "USDD", "TUSD"]
            ):
                try:
                    self._force_immediate_data_fetch(symbol)
                except Exception as e:
                    logger.warning(
                        f"Could not immediately fetch data for new asset {symbol}: {e}"
                    )

            return result_asset
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding asset {symbol}: {e}")
            raise
        finally:
            session.close()

    def update_asset(
        self,
        asset_id: int,
        quantity: Optional[float] = None,
        buy_price: Optional[float] = None,
    ) -> Asset:
        """Update asset quantity or buy price."""
        session = get_session()
        try:
            asset = session.query(Asset).filter(Asset.id == asset_id).first()
            if not asset:
                raise ValueError(f"Asset with ID {asset_id} not found")

            if quantity is not None:
                if quantity < 0:
                    raise ValueError("Quantity cannot be negative")
                asset.quantity = quantity
                asset.total_spent = quantity * asset.average_buy_price

            if buy_price is not None:
                if buy_price <= 0:
                    raise ValueError("Buy price must be positive")
                asset.average_buy_price = buy_price
                asset.total_spent = asset.quantity * buy_price

            asset.updated_at = datetime.utcnow()

            # Flush changes to check for any constraint violations before committing
            session.flush()

            # If we get here, commit the changes
            session.commit()
            session.refresh(asset)

            logger.info(
                f"Successfully updated asset {asset_id}: quantity={asset.quantity}, price={asset.average_buy_price}"
            )
            return asset

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating asset {asset_id}: {e}")
            # Re-raise with more context for constraint errors
            if "NOT NULL constraint failed" in str(e):
                raise ValueError(
                    f"Database constraint error when updating asset. This may be due to related take-profit levels. Error: {e}"
                )
            raise
        finally:
            session.close()

    def sell_asset(
        self, asset_id: int, quantity_to_sell: float, sell_price: float
    ) -> Tuple[Asset, ProfitHistory]:
        """Sell part or all of an asset and record profit."""
        session = get_session()
        try:
            asset = session.query(Asset).filter(Asset.id == asset_id).first()
            if not asset:
                raise ValueError(f"Asset with ID {asset_id} not found")

            if quantity_to_sell > asset.quantity:
                raise ValueError("Cannot sell more than owned quantity")

            # Calculate profit
            profit_per_unit = sell_price - asset.average_buy_price
            realized_profit = profit_per_unit * quantity_to_sell
            profit_percentage = (profit_per_unit / asset.average_buy_price) * 100

            # Record profit history
            profit_record = ProfitHistory(
                symbol=asset.symbol,
                quantity_sold=quantity_to_sell,
                sell_price=sell_price,
                average_buy_price=asset.average_buy_price,
                realized_profit=realized_profit,
                profit_percentage=profit_percentage,
                portfolio_name=asset.portfolio.name,
            )
            session.add(profit_record)

            # Record sell transaction
            sell_transaction = Transaction(
                asset_id=asset.id,
                transaction_type="sell",
                quantity=quantity_to_sell,
                price=sell_price,
                total_value=quantity_to_sell * sell_price,
            )
            session.add(sell_transaction)

            # Update asset
            asset.quantity -= quantity_to_sell
            asset.total_spent = asset.quantity * asset.average_buy_price
            asset.updated_at = datetime.utcnow()

            # If all sold, remove asset
            if asset.quantity <= 0:
                session.delete(asset)
                asset = None

            session.commit()
            if asset:
                session.refresh(asset)
            session.refresh(profit_record)

            return asset, profit_record
        except Exception as e:
            session.rollback()
            logger.error(f"Error selling asset {asset_id}: {e}")
            raise
        finally:
            session.close()

    def delete_asset(self, asset_id: int, force_immediate_cleanup: bool = True) -> bool:
        """Delete an asset completely."""
        session = get_session()
        try:
            asset = session.query(Asset).filter(Asset.id == asset_id).first()
            if not asset:
                return False

            symbol = asset.symbol
            session.delete(asset)
            session.commit()

            # Force immediate cleanup of data if no other assets use this symbol
            if force_immediate_cleanup:
                try:
                    self._force_immediate_cleanup(symbol)
                except Exception as e:
                    logger.warning(
                        f"Could not immediately cleanup data for deleted asset {symbol}: {e}"
                    )

            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting asset {asset_id}: {e}")
            raise
        finally:
            session.close()

    def get_portfolio_assets(self, portfolio_id: int) -> List[Asset]:
        """Get all assets in a portfolio."""
        session = get_session()
        try:
            from sqlalchemy.orm import joinedload

            return (
                session.query(Asset)
                .options(joinedload(Asset.portfolio))
                .filter(Asset.portfolio_id == portfolio_id)
                .all()
            )
        finally:
            session.close()

    def get_all_assets(self) -> List[Asset]:
        """Get all assets across all portfolios."""
        session = get_session()
        try:
            from sqlalchemy.orm import joinedload

            return session.query(Asset).options(joinedload(Asset.portfolio)).all()
        finally:
            session.close()

    def get_portfolio_summary(
        self, portfolio_id: int, current_prices: Dict[str, float]
    ) -> Dict:
        """Calculate portfolio summary statistics.

        Args:
            portfolio_id: Portfolio ID to calculate summary for
            current_prices: Dictionary of current asset prices

        Returns:
            Dictionary containing portfolio summary statistics
        """
        assets = self.get_portfolio_assets(portfolio_id)

        total_value = 0
        total_spent = 0
        total_return = 0

        for asset in assets:
            current_price = current_prices.get(asset.symbol, asset.average_buy_price)
            current_value = asset.quantity * current_price
            asset_return = current_value - asset.total_spent

            total_value += current_value
            total_spent += asset.total_spent
            total_return += asset_return

        return {
            "total_value": total_value,
            "total_spent": total_spent,
            "total_return": total_return,
            "total_return_percentage": (
                (total_return / total_spent * 100) if total_spent > 0 else 0
            ),
            "asset_count": len(assets),
        }

    def get_all_portfolios_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get summary for all portfolios combined.

        Args:
            current_prices: Dictionary of current asset prices

        Returns:
            Dictionary containing combined portfolio summary statistics
        """
        all_assets = self.get_all_assets()

        total_value = 0
        total_spent = 0
        total_return = 0

        for asset in all_assets:
            current_price = current_prices.get(asset.symbol, asset.average_buy_price)
            current_value = asset.quantity * current_price
            asset_return = current_value - asset.total_spent

            total_value += current_value
            total_spent += asset.total_spent
            total_return += asset_return

        return {
            "total_value": total_value,
            "total_spent": total_spent,
            "total_return": total_return,
            "total_return_percentage": (
                (total_return / total_spent * 100) if total_spent > 0 else 0
            ),
            "asset_count": len(all_assets),
        }

    def update_portfolio_name(self, portfolio_id: int, new_name: str):
        """Update portfolio name."""
        session = get_session()
        try:
            # Check if new name already exists
            existing_portfolio = (
                session.query(Portfolio)
                .filter(Portfolio.name == new_name, Portfolio.id != portfolio_id)
                .first()
            )

            if existing_portfolio:
                raise ValueError(f"Portfolio named '{new_name}' already exists")

            # Update the portfolio name
            portfolio = session.query(Portfolio).get(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio with ID {portfolio_id} not found")

            old_name = portfolio.name
            portfolio.name = new_name
            session.commit()

            logger.info(f"Portfolio renamed from '{old_name}' to '{new_name}'")
            
            # Return a dict with the updated information instead of detached object
            return {
                "id": portfolio.id,
                "name": new_name,
                "old_name": old_name
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating portfolio name: {e}")
            raise
        finally:
            session.close()

    def delete_portfolio(self, portfolio_id: int, move_assets_to_portfolio_id: Optional[int] = None):
        """
        Delete a portfolio and optionally move its assets to another portfolio.
        
        Args:
            portfolio_id: ID of the portfolio to delete
            move_assets_to_portfolio_id: Optional portfolio ID to move assets to.
                                       If None, assets will be deleted with the portfolio.
        
        Returns:
            dict: Result information including deleted portfolio name and asset count
        """
        session = get_session()
        try:
            # Get the portfolio to delete
            portfolio = session.query(Portfolio).get(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio with ID {portfolio_id} not found")
            
            portfolio_name = portfolio.name
            
            # Get assets in this portfolio
            assets = session.query(Asset).filter(Asset.portfolio_id == portfolio_id).all()
            asset_count = len(assets)
            
            # Handle asset migration if requested
            if move_assets_to_portfolio_id:
                # Verify target portfolio exists
                target_portfolio = session.query(Portfolio).get(move_assets_to_portfolio_id)
                if not target_portfolio:
                    raise ValueError(f"Target portfolio with ID {move_assets_to_portfolio_id} not found")
                
                # Move assets to target portfolio
                for asset in assets:
                    asset.portfolio_id = move_assets_to_portfolio_id
                
                session.flush()  # Flush asset updates before deleting portfolio
                logger.info(f"Moved {asset_count} assets from '{portfolio_name}' to '{target_portfolio.name}'")
            else:
                # Delete assets first (they will be deleted with portfolio anyway due to CASCADE)
                for asset in assets:
                    session.delete(asset)
                logger.info(f"Deleted {asset_count} assets with portfolio '{portfolio_name}'")
            
            # Delete related portfolio value history records
            from database.models import PortfolioValueHistory
            history_records = session.query(PortfolioValueHistory).filter(
                PortfolioValueHistory.portfolio_id == portfolio_id
            ).all()
            for record in history_records:
                session.delete(record)
            
            if history_records:
                logger.info(f"Deleted {len(history_records)} portfolio value history records for '{portfolio_name}'")
            
            # Delete the portfolio
            session.delete(portfolio)
            session.commit()
            
            # Sync tracked assets after portfolio deletion
            try:
                self.sync_tracked_assets()
            except Exception as e:
                logger.warning(f"Could not sync tracked assets after portfolio deletion: {e}")
            
            result = {
                "portfolio_name": portfolio_name,
                "asset_count": asset_count,
                "assets_moved": move_assets_to_portfolio_id is not None,
                "target_portfolio_id": move_assets_to_portfolio_id
            }
            
            logger.info(f"Successfully deleted portfolio '{portfolio_name}' with {asset_count} assets")
            return result
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting portfolio: {e}")
            raise
        finally:
            session.close()

    def get_portfolio_by_id(self, portfolio_id: int):
        """Get portfolio by ID."""
        session = get_session()
        try:
            portfolio = session.query(Portfolio).get(portfolio_id)
            if portfolio:
                # Detach from session to avoid issues
                session.expunge(portfolio)
            return portfolio
        finally:
            session.close()

    def get_profit_history(
        self, portfolio_name: Optional[str] = None
    ) -> List[ProfitHistory]:
        """Get profit history, optionally filtered by portfolio."""
        session = get_session()
        try:
            query = session.query(ProfitHistory)
            if portfolio_name:
                query = query.filter(ProfitHistory.portfolio_name == portfolio_name)
            return query.order_by(ProfitHistory.sold_at.desc()).all()
        finally:
            session.close()

    def get_total_realized_profit(
        self, portfolio_name: Optional[str] = None
    ) -> Dict[str, float]:
        """Get total realized profit statistics."""
        session = get_session()
        try:
            query = session.query(ProfitHistory)
            if portfolio_name:
                query = query.filter(ProfitHistory.portfolio_name == portfolio_name)

            profits = query.all()

            total_profit = sum(p.realized_profit for p in profits)
            total_sold_value = sum(p.quantity_sold * p.sell_price for p in profits)
            total_invested = sum(p.quantity_sold * p.average_buy_price for p in profits)

            return {
                "total_profit": total_profit,
                "total_sold_value": total_sold_value,
                "total_invested": total_invested,
                "profit_percentage": (
                    (total_profit / total_invested * 100) if total_invested > 0 else 0
                ),
                "trade_count": len(profits),
            }
        finally:
            session.close()

    def delete_profit_record(self, profit_id: int) -> bool:
        """Delete a single profit record."""
        session = get_session()
        try:
            record = (
                session.query(ProfitHistory)
                .filter(ProfitHistory.id == profit_id)
                .first()
            )
            if not record:
                return False

            session.delete(record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting profit record {profit_id}: {e}")
            raise
        finally:
            session.close()

    def clear_profit_history(self, portfolio_name: Optional[str] = None) -> int:
        """Clear profit history, optionally filtered by portfolio. Returns count of deleted records."""
        session = get_session()
        try:
            query = session.query(ProfitHistory)
            if portfolio_name:
                query = query.filter(ProfitHistory.portfolio_name == portfolio_name)

            deleted_count = query.delete(synchronize_session=False)
            session.commit()
            return deleted_count
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing profit history: {e}")
            raise
        finally:
            session.close()

    def add_take_profit_level(
        self,
        asset_id: int,
        target_price: float,
        percentage_to_sell: float,
        strategy_type: str = "fixed",
        notes: Optional[str] = None,
    ) -> TakeProfitLevel:
        """Add a take profit level for an asset."""
        session = get_session()
        try:
            level = TakeProfitLevel(
                asset_id=asset_id,
                target_price=target_price,
                percentage_to_sell=percentage_to_sell,
                strategy_type=strategy_type,
                notes=notes,
            )
            session.add(level)
            session.commit()
            session.refresh(level)
            return level
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding take profit level: {e}")
            raise
        finally:
            session.close()

    def get_take_profit_levels(
        self, asset_id: Optional[int] = None, active_only: bool = True
    ) -> List[TakeProfitLevel]:
        """Get take profit levels, optionally filtered by asset."""
        session = get_session()
        try:
            from sqlalchemy.orm import joinedload

            query = session.query(TakeProfitLevel).options(
                joinedload(TakeProfitLevel.asset)
            )

            if asset_id:
                query = query.filter(TakeProfitLevel.asset_id == asset_id)

            if active_only:
                query = query.filter(TakeProfitLevel.is_active == True)

            return query.order_by(TakeProfitLevel.target_price).all()
        finally:
            session.close()

    def update_take_profit_level(
        self,
        level_id: int,
        target_price: Optional[float] = None,
        percentage_to_sell: Optional[float] = None,
        is_active: Optional[bool] = None,
        notes: Optional[str] = None,
    ) -> TakeProfitLevel:
        """Update a take profit level."""
        session = get_session()
        try:
            level = (
                session.query(TakeProfitLevel)
                .filter(TakeProfitLevel.id == level_id)
                .first()
            )
            if not level:
                raise ValueError(f"Take profit level with ID {level_id} not found")

            if target_price is not None:
                level.target_price = target_price
            if percentage_to_sell is not None:
                level.percentage_to_sell = percentage_to_sell
            if is_active is not None:
                level.is_active = is_active
            if notes is not None:
                level.notes = notes

            session.commit()
            session.refresh(level)
            return level
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating take profit level {level_id}: {e}")
            raise
        finally:
            session.close()

    def delete_take_profit_level(self, level_id: int) -> bool:
        """Delete a take profit level."""
        session = get_session()
        try:
            level = (
                session.query(TakeProfitLevel)
                .filter(TakeProfitLevel.id == level_id)
                .first()
            )
            if not level:
                return False

            session.delete(level)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting take profit level {level_id}: {e}")
            raise
        finally:
            session.close()

    def trigger_take_profit_level(
        self, level_id: int, current_price: float
    ) -> Tuple[Asset, ProfitHistory]:
        """Trigger a take profit level and execute the sell order."""
        session = get_session()
        try:
            level = (
                session.query(TakeProfitLevel)
                .filter(TakeProfitLevel.id == level_id)
                .first()
            )
            if not level:
                raise ValueError(f"Take profit level with ID {level_id} not found")

            if not level.is_active:
                raise ValueError("Take profit level is not active")

            # Calculate quantity to sell
            asset = level.asset
            quantity_to_sell = asset.quantity * (level.percentage_to_sell / 100)

            # Execute the sell
            updated_asset, profit_record = self.sell_asset(
                asset.id, quantity_to_sell, current_price
            )

            # Mark the level as triggered
            level.is_active = False
            level.triggered_at = datetime.utcnow()

            session.commit()
            return updated_asset, profit_record

        except Exception as e:
            session.rollback()
            logger.error(f"Error triggering take profit level {level_id}: {e}")
            raise
        finally:
            session.close()

    def generate_dca_out_levels(
        self,
        asset_id: int,
        num_levels: int = 5,
        profit_start: float = 10.0,
        profit_increment: float = 10.0,
        sell_percentage: float = 20.0,
    ) -> List[TakeProfitLevel]:
        """Generate DCA-out take profit levels for an asset."""
        session = get_session()
        try:
            asset = session.query(Asset).filter(Asset.id == asset_id).first()
            if not asset:
                raise ValueError(f"Asset with ID {asset_id} not found")

            levels = []

            for i in range(num_levels):
                profit_percentage = profit_start + (i * profit_increment)
                target_price = asset.average_buy_price * (1 + profit_percentage / 100)

                level = TakeProfitLevel(
                    asset_id=asset_id,
                    target_price=target_price,
                    percentage_to_sell=sell_percentage,
                    strategy_type="dca_out",
                    notes=f"DCA-out level {i+1}: {profit_percentage}% profit target",
                )
                session.add(level)
                levels.append(level)

            session.commit()
            for level in levels:
                session.refresh(level)

            return levels
        except Exception as e:
            session.rollback()
            logger.error(f"Error generating DCA-out levels: {e}")
            raise
        finally:
            session.close()

    def generate_fibonacci_levels(
        self, asset_id: int, max_price: float
    ) -> List[TakeProfitLevel]:
        """Generate Fibonacci retracement take profit levels."""
        session = get_session()
        try:
            asset = session.query(Asset).filter(Asset.id == asset_id).first()
            if not asset:
                raise ValueError(f"Asset with ID {asset_id} not found")

            # Fibonacci levels (common retracement levels)
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            price_range = max_price - asset.average_buy_price

            levels = []

            for i, fib_ratio in enumerate(fib_levels):
                target_price = asset.average_buy_price + (price_range * fib_ratio)
                sell_percentage = 15.0 + (i * 5.0)  # 15%, 20%, 25%, 30%, 35%

                level = TakeProfitLevel(
                    asset_id=asset_id,
                    target_price=target_price,
                    percentage_to_sell=sell_percentage,
                    strategy_type="fibonacci",
                    notes=f"Fibonacci {fib_ratio:.3f} level",
                )
                session.add(level)
                levels.append(level)

            session.commit()
            for level in levels:
                session.refresh(level)

            return levels
        except Exception as e:
            session.rollback()
            logger.error(f"Error generating Fibonacci levels: {e}")
            raise
        finally:
            session.close()

    # Watchlist Methods
    def add_to_watchlist(
        self,
        symbol: str,
        notes: Optional[str] = None,
        force_immediate_data_fetch: bool = True,
    ) -> Watchlist:
        """Add a symbol to the watchlist."""
        session = get_session()
        try:
            # Check if already exists
            existing = (
                session.query(Watchlist)
                .filter(Watchlist.symbol == symbol.upper())
                .first()
            )
            if existing:
                raise ValueError(f"Symbol {symbol} is already in watchlist")

            watchlist_item = Watchlist(symbol=symbol.upper(), notes=notes)
            session.add(watchlist_item)
            session.commit()
            session.refresh(watchlist_item)

            # Force immediate data fetch for new watchlist assets (excluding stablecoins)
            if force_immediate_data_fetch and symbol.upper() not in [
                "USDT",
                "USDC",
                "BUSD",
                "DAI",
                "USDD",
                "TUSD",
            ]:
                try:
                    self._force_immediate_data_fetch(symbol.upper())
                except Exception as e:
                    logger.warning(
                        f"Could not immediately fetch data for new watchlist asset {symbol.upper()}: {e}"
                    )

            return watchlist_item
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding {symbol} to watchlist: {e}")
            raise
        finally:
            session.close()

    def remove_from_watchlist(
        self, symbol: str, force_immediate_cleanup: bool = True
    ) -> bool:
        """Remove a symbol from the watchlist."""
        session = get_session()
        try:
            item = (
                session.query(Watchlist)
                .filter(Watchlist.symbol == symbol.upper())
                .first()
            )
            if not item:
                return False

            symbol_upper = symbol.upper()
            session.delete(item)
            session.commit()

            # Force immediate cleanup if no other assets use this symbol
            if force_immediate_cleanup:
                try:
                    self._force_immediate_cleanup(symbol_upper)
                except Exception as e:
                    logger.warning(
                        f"Could not immediately cleanup data for removed watchlist asset {symbol_upper}: {e}"
                    )

            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error removing {symbol} from watchlist: {e}")
            raise
        finally:
            session.close()

    def get_watchlist(self) -> List[Watchlist]:
        """Get all watchlist items."""
        session = get_session()
        try:
            return session.query(Watchlist).order_by(Watchlist.added_at.desc()).all()
        finally:
            session.close()

    def update_watchlist_notes(self, symbol: str, notes: str) -> Watchlist:
        """Update notes for a watchlist item."""
        session = get_session()
        try:
            item = (
                session.query(Watchlist)
                .filter(Watchlist.symbol == symbol.upper())
                .first()
            )
            if not item:
                raise ValueError(f"Symbol {symbol} not found in watchlist")

            item.notes = notes
            session.commit()
            session.refresh(item)
            return item
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating watchlist notes for {symbol}: {e}")
            raise
        finally:
            session.close()

    # Settings Methods
    def get_setting(
        self, key: str, default_value: Optional[str] = None
    ) -> Optional[str]:
        """Get a user setting value."""
        session = get_session()
        try:
            setting = (
                session.query(UserSettings)
                .filter(UserSettings.setting_key == key)
                .first()
            )
            return setting.setting_value if setting else default_value
        finally:
            session.close()

    def set_setting(self, key: str, value: str) -> UserSettings:
        """Set a user setting value."""
        session = get_session()
        try:
            setting = (
                session.query(UserSettings)
                .filter(UserSettings.setting_key == key)
                .first()
            )

            if setting:
                setting.setting_value = value
                setting.updated_at = datetime.utcnow()
            else:
                setting = UserSettings(setting_key=key, setting_value=value)
                session.add(setting)

            session.commit()
            session.refresh(setting)
            return setting
        except Exception as e:
            session.rollback()
            logger.error(f"Error setting {key} = {value}: {e}")
            raise
        finally:
            session.close()

    def get_all_settings(self) -> Dict[str, str]:
        """Get all user settings as a dictionary."""
        session = get_session()
        try:
            settings = session.query(UserSettings).all()
            return {setting.setting_key: setting.setting_value for setting in settings}
        finally:
            session.close()

    def delete_setting(self, key: str) -> bool:
        """Delete a user setting."""
        session = get_session()
        try:
            setting = (
                session.query(UserSettings)
                .filter(UserSettings.setting_key == key)
                .first()
            )
            if not setting:
                return False

            session.delete(setting)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting setting {key}: {e}")
            raise
        finally:
            session.close()

    # Price Caching Methods
    def get_cached_prices(self) -> Dict[str, float]:
        """Get all cached prices from database."""
        session = get_session()
        try:
            cached_prices = session.query(CachedPrice).all()
            return {cp.symbol: cp.price for cp in cached_prices}
        except Exception as e:
            logger.error(f"Error getting cached prices: {e}")
            return {}
        finally:
            session.close()

    def update_cached_prices(self, prices: Dict[str, float]) -> bool:
        """Update cached prices in database."""
        session = get_session()
        try:
            for symbol, price in prices.items():
                # Use merge to update existing or create new
                cached_price = (
                    session.query(CachedPrice)
                    .filter(CachedPrice.symbol == symbol)
                    .first()
                )
                if cached_price:
                    cached_price.price = price
                    cached_price.last_updated = datetime.now()
                else:
                    cached_price = CachedPrice(
                        symbol=symbol, price=price, last_updated=datetime.now()
                    )
                    session.add(cached_price)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating cached prices: {e}")
            return False
        finally:
            session.close()

    def get_cached_price(self, symbol: str) -> Optional[float]:
        """Get cached price for a specific symbol."""
        session = get_session()
        try:
            cached_price = (
                session.query(CachedPrice).filter(CachedPrice.symbol == symbol).first()
            )
            return cached_price.price if cached_price else None
        except Exception as e:
            logger.error(f"Error getting cached price for {symbol}: {e}")
            return None
        finally:
            session.close()

    def get_cached_prices_age(self) -> Dict[str, datetime]:
        """Get the age of cached prices (when they were last updated)."""
        session = get_session()
        try:
            cached_prices = session.query(CachedPrice).all()
            return {cp.symbol: cp.last_updated for cp in cached_prices}
        except Exception as e:
            logger.error(f"Error getting cached prices age: {e}")
            return {}
        finally:
            session.close()

    def clean_empty_portfolios(self) -> int:
        """Remove portfolios that have no assets. Returns count of removed portfolios."""
        session = get_session()
        try:
            # Find portfolios with no assets (except Main Portfolio which should never be deleted)
            empty_portfolios = (
                session.query(Portfolio)
                .filter(Portfolio.name != "Main Portfolio", ~Portfolio.assets.any())
                .all()
            )

            count = len(empty_portfolios)
            for portfolio in empty_portfolios:
                session.delete(portfolio)

            session.commit()
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning empty portfolios: {e}")
            return 0
        finally:
            session.close()

    # Tracked Asset Management Functions
    def sync_tracked_assets(self) -> None:
        """Synchronize tracked_assets table with current portfolio and watchlist assets."""
        try:
            with get_db_session_with_retry() as session:
                # Get all current portfolio assets
                portfolio_symbols = set()
                assets = session.query(Asset).all()
                for asset in assets:
                    # Exclude stablecoins from tracking (they don't need price updates)
                    if asset.symbol not in [
                        "USDT",
                        "USDC",
                        "BUSD",
                        "DAI",
                        "USDD",
                        "TUSD",
                    ]:
                        portfolio_symbols.add(asset.symbol)

                # Get all watchlist symbols
                watchlist_symbols = set()
                watchlist_items = session.query(Watchlist).all()
                for item in watchlist_items:
                    if item.symbol not in [
                        "USDT",
                        "USDC",
                        "BUSD",
                        "DAI",
                        "USDD",
                        "TUSD",
                    ]:
                        watchlist_symbols.add(item.symbol)

            # Get all currently tracked symbols
            tracked_symbols = {}
            tracked_assets = session.query(TrackedAsset).all()
            for tracked in tracked_assets:
                tracked_symbols[tracked.symbol] = tracked

            # Update existing and add new tracked assets
            all_symbols = portfolio_symbols | watchlist_symbols

            for symbol in all_symbols:
                # Determine source
                if symbol in portfolio_symbols and symbol in watchlist_symbols:
                    source = "both"
                elif symbol in portfolio_symbols:
                    source = "portfolio"
                else:
                    source = "watchlist"

                if symbol in tracked_symbols:
                    # Update existing
                    tracked = tracked_symbols[symbol]
                    tracked.source = source
                    tracked.is_active = True
                    tracked.updated_at = datetime.utcnow()
                else:
                    # Add new
                    new_tracked = TrackedAsset(
                        symbol=symbol, source=source, is_active=True
                    )
                    session.add(new_tracked)

            # Deactivate assets no longer in portfolios or watchlist
            for symbol, tracked in tracked_symbols.items():
                if symbol not in all_symbols:
                    tracked.is_active = False
                    tracked.updated_at = datetime.utcnow()

            session.commit()
            logger.info(
                f"Synchronized tracked assets: {len(all_symbols)} active symbols"
            )

        except Exception as e:
            session.rollback()
            logger.error(f"Error synchronizing tracked assets: {e}")
            raise
        finally:
            session.close()

    def get_tracked_assets(self, active_only: bool = True) -> List[TrackedAsset]:
        """Get all tracked assets."""
        session = get_session()
        try:
            query = session.query(TrackedAsset)
            if active_only:
                query = query.filter(TrackedAsset.is_active == True)
            return query.order_by(TrackedAsset.symbol).all()
        finally:
            session.close()

    def update_tracked_asset_timestamp(self, symbol: str, update_type: str) -> bool:
        """Update the last update timestamp for a tracked asset."""
        session = get_session()
        try:
            tracked = (
                session.query(TrackedAsset)
                .filter(TrackedAsset.symbol == symbol)
                .first()
            )
            if not tracked:
                return False

            if update_type == "price":
                tracked.last_price_update = datetime.utcnow()
            elif update_type == "historical":
                tracked.last_historical_update = datetime.utcnow()

            tracked.updated_at = datetime.utcnow()
            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating tracked asset timestamp for {symbol}: {e}")
            return False
        finally:
            session.close()

    def get_stale_tracked_assets(self, max_age_minutes: int = 60) -> List[TrackedAsset]:
        """Get tracked assets that need price updates based on age."""
        session = get_session()
        try:
            from datetime import timedelta

            cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)

            stale_assets = (
                session.query(TrackedAsset)
                .filter(
                    TrackedAsset.is_active == True,
                    (TrackedAsset.last_price_update.is_(None))
                    | (TrackedAsset.last_price_update < cutoff_time),
                )
                .order_by(TrackedAsset.symbol)
                .all()
            )

            return stale_assets

        except Exception as e:
            logger.error(f"Error getting stale tracked assets: {e}")
            return []
        finally:
            session.close()

    def _force_immediate_data_fetch(self, symbol: str):
        """Force immediate data fetch for a new asset."""
        try:
            import asyncio
            from data_providers.data_fetcher import CryptoPriceFetcher
            from datetime import datetime, timedelta

            # Create price fetcher instance
            price_fetcher = CryptoPriceFetcher()

            async def fetch_data():
                try:
                    # 1. Fetch current price immediately
                    current_price = await price_fetcher.get_real_time_price(symbol)
                    if current_price:
                        self.update_cached_prices({symbol: current_price})
                        logger.info(
                            f"Immediately fetched current price for {symbol}: ${current_price}"
                        )

                    # 2. Fetch essential historical data for charts
                    # Due to API 1000-point limit, fetch recent data only on immediate fetch (background service will get full history)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(
                        days=1000
                    )  # Get ~3 years immediately, background will complete

                    # Get daily data (recent portion)
                    daily_data = await price_fetcher.get_historical_data(
                        symbol, start_date, end_date, "1d"
                    )
                    if daily_data:
                        session = get_session()
                        try:
                            self._store_historical_data_direct(
                                session, symbol, daily_data, "1d"
                            )
                            logger.info(
                                f"Immediately fetched daily historical data for {symbol}: {len(daily_data)} points (background service will complete full 5-year history)"
                            )
                        finally:
                            session.close()

                    # 3. Fetch recent hourly data for 1h and 4h charts
                    # Due to API 1000-record limits, we can only get ~41 days of hourly data in one request
                    # Prioritize the most recent data for immediate chart functionality
                    hourly_start = end_date - timedelta(
                        days=41
                    )  # ~1000 hours = 41.7 days
                    hourly_data = await price_fetcher.get_historical_data(
                        symbol, hourly_start, end_date, "1h"
                    )
                    if hourly_data:
                        session = get_session()
                        try:
                            self._store_historical_data_direct(
                                session, symbol, hourly_data, "1h"
                            )
                            logger.info(
                                f"Immediately fetched hourly historical data for {symbol}: {len(hourly_data)} points"
                            )
                        finally:
                            session.close()

                    # 4. Update tracked assets to ensure background service picks it up
                    self.sync_tracked_assets()

                except Exception as e:
                    logger.error(f"Error in immediate data fetch for {symbol}: {e}")

            # Run the async function
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is None:
                # No event loop running, create a new one
                asyncio.run(fetch_data())
            else:
                # Event loop is running, schedule the task
                task = loop.create_task(fetch_data())
                # Don't wait for completion to avoid blocking the UI

        except Exception as e:
            logger.error(f"Failed to force immediate data fetch for {symbol}: {e}")

    def _store_historical_data_direct(
        self, session, symbol: str, data: List[Dict], interval: str
    ):
        """Store historical data directly in the database."""
        try:
            from database.models import HistoricalPrice

            for item in data:
                try:
                    # Convert timestamp to datetime
                    if "timestamp" in item:
                        dt = datetime.fromtimestamp(item["timestamp"] / 1000)
                    else:
                        continue

                    # Check if data already exists
                    existing = (
                        session.query(HistoricalPrice)
                        .filter(
                            HistoricalPrice.symbol == symbol,
                            HistoricalPrice.interval == interval,
                            HistoricalPrice.date == dt,
                        )
                        .first()
                    )

                    if not existing:
                        # Create new historical price record
                        hist_price = HistoricalPrice(
                            symbol=symbol,
                            interval=interval,
                            date=dt,
                            price=float(item.get("close", item.get("price", 0))),
                            open_price=float(
                                item.get(
                                    "open", item.get("close", item.get("price", 0))
                                )
                            ),
                            high_price=float(
                                item.get(
                                    "high", item.get("close", item.get("price", 0))
                                )
                            ),
                            low_price=float(
                                item.get("low", item.get("close", item.get("price", 0)))
                            ),
                            volume=float(item.get("volume", 0)),
                        )
                        session.add(hist_price)
                except Exception as e:
                    logger.warning(
                        f"Error storing historical data point for {symbol}: {e}"
                    )
                    continue

            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing historical data for {symbol}: {e}")

    def _force_immediate_cleanup(self, symbol: str):
        """Force immediate cleanup of data when an asset is deleted."""
        try:
            # Check if this symbol is still used in any portfolio or watchlist
            session = get_session()
            try:
                # Check if symbol is still in any portfolio
                assets_using_symbol = (
                    session.query(Asset).filter(Asset.symbol == symbol).count()
                )

                # Check if symbol is still in watchlist
                from database.models import Watchlist

                watchlist_using_symbol = (
                    session.query(Watchlist).filter(Watchlist.symbol == symbol).count()
                )

                # If no longer used anywhere, cleanup data
                if assets_using_symbol == 0 and watchlist_using_symbol == 0:
                    # 1. Remove from cached prices
                    cached_prices = self.get_cached_prices()
                    if symbol in cached_prices:
                        # Update cached prices without this symbol
                        updated_prices = {
                            k: v for k, v in cached_prices.items() if k != symbol
                        }
                        from database.models import CachedPrice

                        cached_price_record = (
                            session.query(CachedPrice)
                            .filter(CachedPrice.symbol == symbol)
                            .first()
                        )
                        if cached_price_record:
                            session.delete(cached_price_record)

                    # 2. Remove from tracked assets
                    tracked_asset = (
                        session.query(TrackedAsset)
                        .filter(TrackedAsset.symbol == symbol)
                        .first()
                    )
                    if tracked_asset:
                        session.delete(tracked_asset)

                    # 3. Optionally remove historical data (commented out to preserve data)
                    # from database.models import HistoricalPrice
                    # session.query(HistoricalPrice).filter(HistoricalPrice.symbol == symbol).delete()

                    session.commit()
                    logger.info(
                        f"Immediately cleaned up data for deleted symbol {symbol}"
                    )
                else:
                    # Symbol is still in use, just sync tracked assets
                    self.sync_tracked_assets()
                    logger.info(
                        f"Symbol {symbol} still in use elsewhere, updated tracked assets"
                    )

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Failed to cleanup data for deleted symbol {symbol}: {e}")

    def get_aggregated_assets(self) -> List[Dict]:
        """Get all assets aggregated by symbol across all portfolios for 'All Portfolios' view."""
        all_assets = self.get_all_assets()

        # Aggregate assets by symbol
        aggregated = {}

        for asset in all_assets:
            symbol = asset.symbol

            if symbol not in aggregated:
                # Initialize aggregated asset
                aggregated[symbol] = {
                    "symbol": symbol,
                    "quantity": 0.0,
                    "total_spent": 0.0,
                    "portfolios": [],  # Track which portfolios contain this asset
                    "assets": [],  # Keep reference to original assets for portfolio names
                }

            # Add to aggregated totals
            aggregated[symbol]["quantity"] += asset.quantity
            aggregated[symbol]["total_spent"] += asset.total_spent
            aggregated[symbol]["portfolios"].append(
                asset.portfolio.name
                if hasattr(asset, "portfolio") and asset.portfolio
                else "Unknown"
            )
            aggregated[symbol]["assets"].append(asset)

        # Calculate weighted average buy price for each aggregated asset
        for symbol, agg_asset in aggregated.items():
            if agg_asset["quantity"] > 0:
                agg_asset["average_buy_price"] = (
                    agg_asset["total_spent"] / agg_asset["quantity"]
                )
            else:
                agg_asset["average_buy_price"] = 0.0

            # Create portfolio name summary (unique portfolios only)
            unique_portfolios = list(set(agg_asset["portfolios"]))
            if len(unique_portfolios) == 1:
                agg_asset["portfolio_summary"] = unique_portfolios[0]
            else:
                agg_asset["portfolio_summary"] = f"{len(unique_portfolios)} portfolios"

        # Return as list sorted by symbol
        return sorted(aggregated.values(), key=lambda x: x["symbol"])

    def get_all_portfolios_summary_aggregated(
        self, current_prices: Dict[str, float]
    ) -> Dict:
        """Get summary for all portfolios with properly aggregated assets.

        Args:
            current_prices: Dictionary of current asset prices

        Returns:
            Dictionary containing aggregated portfolio summary statistics
        """
        aggregated_assets = self.get_aggregated_assets()

        total_value = 0
        total_spent = 0
        total_return = 0

        for agg_asset in aggregated_assets:
            symbol = agg_asset["symbol"]
            quantity = agg_asset["quantity"]
            total_spent_asset = agg_asset["total_spent"]
            avg_buy_price = agg_asset["average_buy_price"]

            current_price = current_prices.get(symbol, avg_buy_price)
            current_value = quantity * current_price
            asset_return = current_value - total_spent_asset

            total_value += current_value
            total_spent += total_spent_asset
            total_return += asset_return

        return {
            "total_value": total_value,
            "total_spent": total_spent,
            "total_return": total_return,
            "total_return_percentage": (
                (total_return / total_spent * 100) if total_spent > 0 else 0
            ),
            "asset_count": len(aggregated_assets),
        }
