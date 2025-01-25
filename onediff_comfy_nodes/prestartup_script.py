import os
import sys
import importlib


ONEDIFF_COMFY_NODES_DIR = os.path.dirname(os.path.abspath(__file__))
ONEDIFF_COMFY_PRESTARTUP_SCRIPTS_DIR = os.path.join(
    ONEDIFF_COMFY_NODES_DIR, "prestartup_scripts"
)

sys.path.append(ONEDIFF_COMFY_NODES_DIR)

for filename in sorted(os.listdir(ONEDIFF_COMFY_PRESTARTUP_SCRIPTS_DIR)):
    if filename.endswith(".py") and filename[0] != "_":
        importlib.import_module(f"prestartup_scripts.{filename[:-3]}")
    elif filename.endswith(".so"):
        importlib.import_module(f"prestartup_scripts.{filename.split('.')[0]}")
