"""
Core analysis modules for DarkBottomLine framework.
"""

from ._version import __version__
from .processor import DarkBottomLineProcessor
from .analyzer import DarkBottomLineAnalyzer
from .regions import Region, RegionManager
from .plotting import PlotManager

__all__ = [
    '__version__',
    'DarkBottomLineProcessor',
    'DarkBottomLineAnalyzer',
    'Region',
    'RegionManager',
    'PlotManager',
]

# DarkBottomLineCoffeaProcessor only exists when coffea imported successfully
# inside processor.py (see COFFEA_AVAILABLE there). Import it opportunistically
# so a broken/missing coffea degrades to "feature unavailable" instead of
# taking down `import darkbottomline` entirely.
try:
    from .processor import DarkBottomLineCoffeaProcessor
    __all__.append('DarkBottomLineCoffeaProcessor')
except ImportError:
    pass
