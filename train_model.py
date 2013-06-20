from neural_nets.config import run_from_config
from neural_nets.remote import run_experiment
from neural_nets.random import hyperparameter_search

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('-r', '--remote', dest='remote', action='store_true')
    parser.add_argument('-t', '--hyper-template')
    parser.add_argument('-n', '--n-trials', type=int, default=10)
    args = parser.parse_args()

    yaml_src = ''.join(open(args.config_file).readlines())

    if args.hyper_template is not None:
        hyperparameter_config = open(args.hyper_template).read()
        hyperparameter_search(yaml_src, hyperparameter_config, args.n_trials)
    elif args.remote:
        task = run_experiment.delay(yaml_src)
        print task.task_id
    else:
        import pycuda.autoinit
        from scikits.cuda import linalg
        linalg.init()
        run_from_config(yaml_src)
