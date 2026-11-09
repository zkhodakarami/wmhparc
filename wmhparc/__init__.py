"""wmhparc: regional parcellation of white matter hyperintensities in MNI-space atlases.

Main entry points
-----------------
- :mod:`wmhparc.atlases`      registry of released / buildable parcellations and their label tables
- :mod:`wmhparc.bullseye`     bullseye (lobe x depth layer) construction from tissue labels
- :mod:`wmhparc.registration` ANTs SyN subject-to-template registration (T1 +/- FLAIR, FLAIR-only)
- :mod:`wmhparc.parcellate`   apply a parcellation to a WMH mask and compute regional volumes
- :mod:`wmhparc.cli`          command line interface (``wmhparc --help``)
"""

__version__ = "0.1.0"

from .atlases import ATLASES, get_atlas, load_atlas, load_lut
from .parcellate import parcellate_wmh, regional_volumes

__all__ = ["ATLASES", "get_atlas", "load_atlas", "load_lut", "parcellate_wmh", "regional_volumes", "__version__"]
