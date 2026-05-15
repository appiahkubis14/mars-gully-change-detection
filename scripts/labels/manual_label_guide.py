"""QGIS manual labelling guide — prints instructions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.downloaders.manual_data_guide import LABELS_GUIDE

def main():
    print(LABELS_GUIDE)

if __name__ == "__main__":
    main()
