import pycuda.autoinit
import argparse
from neural_nets.config import load

def run_from_config(config_file):
    yaml_src = ''.join(open(config_file).readlines())
    config = load(yaml_src)
    optimizer = config['optimizer']
    run_conf = config['run_conf']
    run_conf['yaml_config'] = yaml_src
    optimizer.run(**run_conf)

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1]
    run_from_config(config_file)
