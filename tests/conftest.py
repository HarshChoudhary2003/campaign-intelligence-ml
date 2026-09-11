import sys
from pathlib import Path

# Add the project root to the path so that app and src can be imported
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
