from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRAGStorage(ABC):
    """
    Base class for RAG-based Storage implementations.
    """

    app: Any | None = None

    def __init__(
        self,
        type: str,
        allow_reset: bool = True,
        embedder_config: Optional[Dict[str, Any]] = None,
        crew: Any = None,
    ):
        self.type = type
        self.allow_reset = allow_reset
        self.embedder_config = embedder_config
        self.crew = crew
        self.agents = self._initialize_agents()

    def _initialize_agents(self) -> str:
        if self.crew:
            return "_".join(
                [self._sanitize_role(agent.role) for agent in self.crew.agents]
            )
        return ""

    @abstractmethod
    def _sanitize_role(self, role: str) -> str:
        """Sanitizes agent roles to ensure valid directory names."""

    @abstractmethod
    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """Save a value with metadata to the storage."""

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        """Search for entries in the storage."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the storage."""

    @abstractmethod
    def _generate_embedding(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Generate an embedding for the given text and metadata."""

    @abstractmethod
    def _initialize_app(self):
        """Initialize the vector db."""

    def setup_config(self, config: Dict[str, Any]):
        """Setup the config of the storage."""

    def initialize_client(self):
        """Initialize the client of the storage. This should setup the app and the db collection"""
