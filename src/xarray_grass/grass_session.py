import os
import sys
import shutil
import subprocess


def get_grass_bin(version=None):
    """Taken from the grass_session package from Pietro Zambelli.
    Return the path to the GRASS GIS binary file command.
    If available takes the value from os.environ GRASSBIN variable,
    else the GRASS binary found by which (only Python 3) or the latest GRASS
    executable on the path."""
    version = "" if not version else version
    grassbin = os.environ.get("GRASSBIN")
    if grassbin:
        return grassbin

    grassbin_path = shutil.which("grass{}".format(version))
    if grassbin_path:
        grassbin = os.path.split(grassbin_path)[1]
        return grassbin

    if not grassbin:
        raise RuntimeError(
            (
                "Cannot find GRASS GIS start script: 'grass{}', "
                "set the right one using the GRASSBIN environment. "
                "variable"
            ).format(version)
        )


# Set GRASS python path
grassbin = get_grass_bin()
grass_cmd = [grassbin, "--config", "python_path"]
grass_python_path = subprocess.check_output(grass_cmd, text=True).strip()
sys.path.append(grass_python_path)

import grass.script as gs


def set_grass_session(gisdb: str, project: str, mapset: str) -> gs.setup.SessionHandle:
    """ "wrapper for the grass factory session function"""
    return gs.setup.init(path=gisdb, location=project, mapset=mapset)
