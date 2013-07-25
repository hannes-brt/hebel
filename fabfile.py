from fabric.api import *
from fabric.contrib.project import rsync_project

env.roledefs = {
    'local': ['localhost'],
    'gpu_servers': ['gpu1.psi.utoronto.ca', 'gpu2.psi.utoronto.ca']
}

def sync_project():
    """Sync current state of repository to GPU servers"""
    run('mkdir -p ~/git')
    rsync_project('~/git', '~/git/neural-nets-pycuda', exclude=['.git', 'examples'])

@roles('local', 'gpu_servers')
def start_celery():
    with cd('~/git/neural-nets-pycuda'):
        run('screen -dmS celery && '
            r'screen -S celery -p 0 -X stuff "celery -A neural_nets.remote worker -c 1 $(printf \\\\r)"')

@hosts()
def start_flower():
    local('screen -S celery -X screen')
    local(r'screen -S celery -p 1 -X stuff "celery -A neural_nets.remote flower $(printf \\r)"')

def start_all():
    execute(start_celery)
    execute(start_flower)
