import pycuda.autoinit
import argparse
from neural_nets.config import load_path

def run_from_config(config_file):
    config = load_path(config_file)
    optimizer = config['optimizer']
    run_conf = config['run_conf']
    print run_conf
    optimizer.run(**run_conf)

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1]
    run_from_config(config_file)





