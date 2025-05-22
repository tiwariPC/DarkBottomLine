import os

"""
metaconditions will be a dictionary in the form:
{
    "Era2017_legacy_v1": "Era2017_legacy_v1.json",
    "Era2017_legacy_v2": "Era2017_legacy_v2.json",
    ...
}
"""
metaconditions = {
    json_file.replace(".json", ""): json_file
    for json_file in [
        fl for fl in os.listdir(os.path.dirname(__file__)) if fl.endswith(".json")
    ]
}
