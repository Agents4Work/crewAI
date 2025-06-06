from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from pydantic import BaseModel, ConfigDict, Field


class BaseKnowledgeSource(BaseModel, ABC):
    """Abstract base class for knowledge sources."""

    chunk_size: int = 4000
    chunk_overlap: int = 200
    chunks: List[str] = Field(default_factory=list)
    chunk_embeddings: List[np.ndarray] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: Optional[KnowledgeStorage] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Currently unused
    collection_name: Optional[str] = Field(default=None)

    @abstractmethod
    def validate_content(self) -> Any:
        """Load and preprocess content from the source."""

    @abstractmethod
    def add(self) -> None:
        """Process content, chunk it, compute embeddings, and save them."""

    def get_embeddings(self) -> List[np.ndarray]:
        """Return the list of embeddings for the chunks."""
        return self.chunk_embeddings

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    def _save_documents(self):
        """
        Save the documents to the storage.
        This method should be called after the chunks and embeddings are generated.
        """
        if self.storage:
            self.storage.save(self.chunks)
        else:
            raise ValueError("No storage found to save documents.")
