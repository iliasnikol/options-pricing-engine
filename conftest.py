"""Make the src/ layout importable when running pytest from the project root.

Not needed once the package is installed (`pip install -e .`), but this lets
`pytest` work out of the box.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
