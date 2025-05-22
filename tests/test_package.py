from importlib import resources
import json


def test_package():
    """
    Test the package imports
    """
    from bottom_line import tools
    from bottom_line import utils
    from bottom_line import workflows


def test_metaconditions():
    """
    Test the metaconditions
    1. Check that by opening the metacondition files with this procedure we get dictionaries
    """
    from bottom_line.metaconditions import metaconditions

    for json_file in metaconditions.values():
        with resources.open_text("bottom_line.metaconditions", json_file) as f:
            dct = json.load(f)
            assert isinstance(dct, dict)
