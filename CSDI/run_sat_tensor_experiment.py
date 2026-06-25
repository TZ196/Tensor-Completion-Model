import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_baseline_lib.runner import run


if __name__ == "__main__":
    run("csdi")
