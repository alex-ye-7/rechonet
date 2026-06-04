"""Sets paths based on configuration files."""

import configparser
import os
import types

_FILENAME = None
_PARAM = {}
for filename in ["rechonet.cfg",
                 ".rechonet.cfg",
                 os.path.expanduser("~/rechonet.cfg"),
                 os.path.expanduser("~/.rechonet.cfg"),
                 ]:
    if os.path.isfile(filename):
        _FILENAME = filename
        config = configparser.ConfigParser()
        with open(filename, "r") as f:
            config.read_string("[config]\n" + f.read())
            _PARAM = config["config"]
        break

CONFIG = types.SimpleNamespace(
    FILENAME=_FILENAME,
    DATA_DIR=_PARAM.get("data_dir", "/content/data/extracted/EchoNet-Dynamic/"))