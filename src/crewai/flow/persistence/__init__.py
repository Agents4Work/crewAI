"""
CrewAI Flow Persistence.

This module provides interfaces and implementations for persisting flow states.
"""

from typing import Any, Dict, TypeVar, Union

from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.persistence.decorators import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from pydantic import BaseModel

__all__ = ["FlowPersistence", "persist", "SQLiteFlowPersistence"]

StateType = TypeVar("StateType", bound=Union[Dict[str, Any], BaseModel])
DictStateType = Dict[str, Any]
