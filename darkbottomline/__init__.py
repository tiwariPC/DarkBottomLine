"""
Core analysis modules for DarkBottomLine framework.
"""

from ._version import __version__
from .processor import DarkBottomLineProcessor, DarkBottomLineCoffeaProcessor
from .analyzer import DarkBottomLineAnalyzer
from .regions import Region, RegionManager
from .plotting import PlotManager

__all__ = [
    '__version__',
    'DarkBottomLineProcessor',
    'DarkBottomLineCoffeaProcessor',
    'DarkBottomLineAnalyzer',
    'Region',
    'RegionManager',
    'PlotManager',
]
