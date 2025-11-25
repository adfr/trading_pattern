"""Pattern manager for persistent pattern storage and lifecycle management."""

from typing import Any, Optional

from ..core.database import Database
from ..core.logger import get_logger
from .pattern import PatternDefinition, PatternMatcher

logger = get_logger(__name__)


class PatternManager:
    """
    Manages pattern lifecycle and persistence.

    Provides a singleton pattern matcher that persists patterns
    across operations (generation, backtesting, deployment).
    """

    _instance: Optional["PatternManager"] = None

    def __new__(cls, database: Optional[Database] = None):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, database: Optional[Database] = None):
        """
        Initialize pattern manager.

        Args:
            database: Database instance for pattern persistence
        """
        # Only initialize once
        if self._initialized:
            return

        self.database = database
        self.pattern_matcher = PatternMatcher()
        self._initialized = True

        logger.info("PatternManager initialized (singleton)")

    def set_database(self, database: Database) -> None:
        """Set or update the database instance."""
        self.database = database

    def add_pattern(self, pattern_data: dict[str, Any]) -> PatternDefinition:
        """
        Add a pattern to the manager.

        Args:
            pattern_data: Pattern dictionary

        Returns:
            PatternDefinition instance
        """
        pattern = self.pattern_matcher.load_pattern_from_dict(pattern_data)
        logger.info(f"Added pattern to manager: {pattern.name} ({pattern.id})")
        return pattern

    def remove_pattern(self, pattern_id: str) -> None:
        """
        Remove a pattern from the manager.

        Args:
            pattern_id: Pattern ID to remove
        """
        self.pattern_matcher.remove_pattern(pattern_id)
        logger.info(f"Removed pattern from manager: {pattern_id}")

    def get_pattern(self, pattern_id: str) -> Optional[PatternDefinition]:
        """
        Get a pattern by ID.

        Args:
            pattern_id: Pattern ID

        Returns:
            PatternDefinition if found, None otherwise
        """
        return self.pattern_matcher._patterns.get(pattern_id)

    def load_from_database(
        self,
        status: Optional[str] = None,
        reload: bool = False
    ) -> int:
        """
        Load patterns from database.

        Args:
            status: Filter by status (pending, backtested, deployed, failed)
            reload: If True, clear existing patterns before loading

        Returns:
            Number of patterns loaded
        """
        if not self.database:
            logger.warning("No database configured, cannot load patterns")
            return 0

        if reload:
            self.pattern_matcher._patterns.clear()
            logger.info("Cleared existing patterns")

        patterns = self.database.get_patterns(status=status)
        count = 0

        for pattern_data in patterns:
            try:
                self.add_pattern(pattern_data)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load pattern {pattern_data.get('id')}: {e}")

        logger.info(f"Loaded {count} pattern(s) from database" +
                   (f" with status='{status}'" if status else ""))
        return count

    def load_deployed_patterns(self) -> int:
        """
        Load patterns with 'deployed' status.

        Returns:
            Number of patterns loaded
        """
        return self.load_from_database(status="deployed")

    def load_backtested_patterns(self) -> int:
        """
        Load patterns with 'backtested' status.

        Returns:
            Number of patterns loaded
        """
        return self.load_from_database(status="backtested")

    def load_all_patterns(self) -> int:
        """
        Load all patterns from database.

        Returns:
            Number of patterns loaded
        """
        return self.load_from_database(status=None)

    def get_loaded_patterns(self) -> list[str]:
        """
        Get list of loaded pattern IDs.

        Returns:
            List of pattern IDs currently in memory
        """
        return list(self.pattern_matcher._patterns.keys())

    def get_pattern_count(self) -> int:
        """
        Get count of patterns currently in memory.

        Returns:
            Number of patterns loaded
        """
        return len(self.pattern_matcher._patterns)

    def clear_patterns(self) -> None:
        """Clear all patterns from memory."""
        self.pattern_matcher._patterns.clear()
        logger.info("Cleared all patterns from memory")


def get_pattern_manager(database: Optional[Database] = None) -> PatternManager:
    """
    Get the singleton PatternManager instance.

    Args:
        database: Database instance (only used on first call)

    Returns:
        PatternManager singleton instance
    """
    return PatternManager(database)
