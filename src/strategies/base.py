import abc
from typing import Dict, Any
import pandas as pd


class Strategy(abc.ABC):
    """
    Abstract base class for all trading strategies.
    All strategies must implement these methods.
    """
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the strategy"""
        pass

    @abc.abstractmethod
    def suggest_parameters(self, trial) -> Dict[str, Any]:
        """Suggest parameters for optimization trials"""
        pass

    @abc.abstractmethod
    def prepare(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Prepare the dataframe with strategy-specific indicators"""
        pass

    @abc.abstractmethod
    def warmup_period(self, params: Dict[str, Any]) -> int:
        """Return the warmup period needed for the strategy"""
        pass

    @abc.abstractmethod
    def decide_position(self, df: pd.DataFrame, i: int, prev_position: int, params: Dict[str, Any]) -> int:
        """Decide the position for the current data point"""
        pass
