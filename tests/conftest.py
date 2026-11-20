import pathlib
import sys

# `code/update.py` is a script rather than a package, so its directory is put on the path for the tests to import it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "code"))
