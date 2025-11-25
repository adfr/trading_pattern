"""Database management for the trading system."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .logger import get_logger

Base = declarative_base()
logger = get_logger(__name__)


class Pattern(Base):
    """Pattern model for storing AI-generated patterns."""
    __tablename__ = "patterns"

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    pattern_type = Column(String(50))  # bullish, bearish, neutral
    timeframe = Column(String(20))
    symbol = Column(String(20))

    # Pattern definition as JSON
    definition = Column(Text, nullable=False)

    # Entry/exit rules as JSON
    entry_rules = Column(Text)
    exit_rules = Column(Text)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(50))  # 'ai' or 'manual'

    # Status
    status = Column(String(20), default="pending")  # pending, backtested, deployed, retired

    # Backtest results
    backtest_results = Column(Text)
    passed_backtest = Column(Boolean, default=False)


class Trade(Base):
    """Trade model for storing executed trades."""
    __tablename__ = "trades"

    id = Column(String(36), primary_key=True)
    pattern_id = Column(String(36))
    symbol = Column(String(20), nullable=False)

    # Trade details
    direction = Column(String(10))  # long, short
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Integer)

    # Timestamps
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)

    # Performance
    pnl = Column(Float)
    pnl_percent = Column(Float)

    # Risk info
    stop_loss = Column(Float)
    take_profit = Column(Float)

    # Status
    status = Column(String(20))  # open, closed, cancelled
    exit_reason = Column(String(50))  # target, stop, signal, manual

    # Mode
    mode = Column(String(10))  # paper, live, backtest


class BacktestResult(Base):
    """Backtest result model."""
    __tablename__ = "backtest_results"

    id = Column(String(36), primary_key=True)
    pattern_id = Column(String(36), nullable=False)

    # Date range
    start_date = Column(DateTime)
    end_date = Column(DateTime)

    # Performance metrics
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)

    profit_factor = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)

    total_return = Column(Float)
    annualized_return = Column(Float)

    # Statistical tests
    t_statistic = Column(Float)
    p_value = Column(Float)

    # Detailed results as JSON
    detailed_results = Column(Text)

    # Walk-forward results
    is_walk_forward = Column(Boolean, default=False)
    walk_forward_window = Column(Integer)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Pass/Fail
    passed = Column(Boolean)
    failure_reasons = Column(Text)


class MarketData(Base):
    """Market data cache model."""
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)

    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    # Technical indicators as JSON
    indicators = Column(Text)


class Database:
    """Database manager for the trading system."""

    def __init__(self, db_path: str = "data/trading.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at {db_path}")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # Pattern operations
    def save_pattern(self, pattern_data: dict[str, Any]) -> str:
        """Save a pattern to the database."""
        import uuid

        with self.get_session() as session:
            pattern = Pattern(
                id=pattern_data.get("id", str(uuid.uuid4())),
                name=pattern_data["name"],
                description=pattern_data.get("description", ""),
                pattern_type=pattern_data.get("pattern_type", "neutral"),
                timeframe=pattern_data.get("timeframe", "1min"),
                symbol=pattern_data.get("symbol", ""),
                definition=json.dumps(pattern_data["definition"]),
                entry_rules=json.dumps(pattern_data.get("entry_rules", {})),
                exit_rules=json.dumps(pattern_data.get("exit_rules", {})),
                created_by=pattern_data.get("created_by", "ai"),
                status=pattern_data.get("status", "pending"),
            )
            session.add(pattern)
            session.commit()
            logger.info(f"Saved pattern: {pattern.id}")
            return pattern.id

    def get_pattern(self, pattern_id: str) -> Optional[dict[str, Any]]:
        """Get a pattern by ID."""
        with self.get_session() as session:
            pattern = session.query(Pattern).filter(Pattern.id == pattern_id).first()
            if pattern:
                return self._pattern_to_dict(pattern)
            return None

    def get_patterns(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get patterns with optional filtering."""
        with self.get_session() as session:
            query = session.query(Pattern)
            if status:
                query = query.filter(Pattern.status == status)
            if symbol:
                query = query.filter(Pattern.symbol == symbol)
            patterns = query.order_by(Pattern.created_at.desc()).limit(limit).all()
            return [self._pattern_to_dict(p) for p in patterns]

    def update_pattern_status(
        self,
        pattern_id: str,
        status: str,
        backtest_results: Optional[dict] = None
    ) -> None:
        """Update pattern status and backtest results."""
        with self.get_session() as session:
            pattern = session.query(Pattern).filter(Pattern.id == pattern_id).first()
            if pattern:
                pattern.status = status
                if backtest_results:
                    pattern.backtest_results = json.dumps(backtest_results)
                    pattern.passed_backtest = backtest_results.get("passed", False)
                session.commit()

    def _pattern_to_dict(self, pattern: Pattern) -> dict[str, Any]:
        """Convert Pattern model to dictionary."""
        return {
            "id": pattern.id,
            "name": pattern.name,
            "description": pattern.description,
            "pattern_type": pattern.pattern_type,
            "timeframe": pattern.timeframe,
            "symbol": pattern.symbol,
            "definition": json.loads(pattern.definition) if pattern.definition else {},
            "entry_rules": json.loads(pattern.entry_rules) if pattern.entry_rules else {},
            "exit_rules": json.loads(pattern.exit_rules) if pattern.exit_rules else {},
            "created_at": pattern.created_at,
            "created_by": pattern.created_by,
            "status": pattern.status,
            "backtest_results": json.loads(pattern.backtest_results) if pattern.backtest_results else None,
            "passed_backtest": pattern.passed_backtest,
        }

    # Trade operations
    def save_trade(self, trade_data: dict[str, Any]) -> str:
        """Save a trade to the database."""
        import uuid

        with self.get_session() as session:
            trade = Trade(
                id=trade_data.get("id", str(uuid.uuid4())),
                pattern_id=trade_data.get("pattern_id"),
                symbol=trade_data["symbol"],
                direction=trade_data["direction"],
                entry_price=trade_data.get("entry_price"),
                exit_price=trade_data.get("exit_price"),
                quantity=trade_data.get("quantity", 0),
                entry_time=trade_data.get("entry_time"),
                exit_time=trade_data.get("exit_time"),
                pnl=trade_data.get("pnl"),
                pnl_percent=trade_data.get("pnl_percent"),
                stop_loss=trade_data.get("stop_loss"),
                take_profit=trade_data.get("take_profit"),
                status=trade_data.get("status", "open"),
                exit_reason=trade_data.get("exit_reason"),
                mode=trade_data.get("mode", "paper"),
            )
            session.add(trade)
            session.commit()
            return trade.id

    def get_trades(
        self,
        symbol: Optional[str] = None,
        pattern_id: Optional[str] = None,
        status: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get trades with optional filtering."""
        with self.get_session() as session:
            query = session.query(Trade)
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if pattern_id:
                query = query.filter(Trade.pattern_id == pattern_id)
            if status:
                query = query.filter(Trade.status == status)
            if mode:
                query = query.filter(Trade.mode == mode)
            trades = query.order_by(Trade.entry_time.desc()).limit(limit).all()
            return [self._trade_to_dict(t) for t in trades]

    def update_trade(self, trade_id: str, updates: dict[str, Any]) -> None:
        """Update a trade."""
        with self.get_session() as session:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                for key, value in updates.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                session.commit()

    def _trade_to_dict(self, trade: Trade) -> dict[str, Any]:
        """Convert Trade model to dictionary."""
        return {
            "id": trade.id,
            "pattern_id": trade.pattern_id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "pnl": trade.pnl,
            "pnl_percent": trade.pnl_percent,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "status": trade.status,
            "exit_reason": trade.exit_reason,
            "mode": trade.mode,
        }

    # Backtest result operations
    def save_backtest_result(self, result_data: dict[str, Any]) -> str:
        """Save backtest result to database."""
        import uuid

        with self.get_session() as session:
            result = BacktestResult(
                id=result_data.get("id", str(uuid.uuid4())),
                pattern_id=result_data["pattern_id"],
                start_date=result_data.get("start_date"),
                end_date=result_data.get("end_date"),
                total_trades=result_data.get("total_trades", 0),
                winning_trades=result_data.get("winning_trades", 0),
                losing_trades=result_data.get("losing_trades", 0),
                win_rate=result_data.get("win_rate"),
                profit_factor=result_data.get("profit_factor"),
                sharpe_ratio=result_data.get("sharpe_ratio"),
                sortino_ratio=result_data.get("sortino_ratio"),
                max_drawdown=result_data.get("max_drawdown"),
                total_return=result_data.get("total_return"),
                annualized_return=result_data.get("annualized_return"),
                t_statistic=result_data.get("t_statistic"),
                p_value=result_data.get("p_value"),
                detailed_results=json.dumps(result_data.get("detailed_results", {})),
                is_walk_forward=result_data.get("is_walk_forward", False),
                walk_forward_window=result_data.get("walk_forward_window"),
                passed=result_data.get("passed", False),
                failure_reasons=json.dumps(result_data.get("failure_reasons", [])),
            )
            session.add(result)
            session.commit()
            return result.id

    def get_backtest_results(
        self,
        pattern_id: Optional[str] = None,
        passed_only: bool = False
    ) -> list[dict[str, Any]]:
        """Get backtest results."""
        with self.get_session() as session:
            query = session.query(BacktestResult)
            if pattern_id:
                query = query.filter(BacktestResult.pattern_id == pattern_id)
            if passed_only:
                query = query.filter(BacktestResult.passed == True)
            results = query.order_by(BacktestResult.created_at.desc()).all()
            return [self._backtest_to_dict(r) for r in results]

    def _backtest_to_dict(self, result: BacktestResult) -> dict[str, Any]:
        """Convert BacktestResult model to dictionary."""
        return {
            "id": result.id,
            "pattern_id": result.pattern_id,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "t_statistic": result.t_statistic,
            "p_value": result.p_value,
            "detailed_results": json.loads(result.detailed_results) if result.detailed_results else {},
            "is_walk_forward": result.is_walk_forward,
            "walk_forward_window": result.walk_forward_window,
            "passed": result.passed,
            "failure_reasons": json.loads(result.failure_reasons) if result.failure_reasons else [],
        }
