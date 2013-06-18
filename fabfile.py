from fabric.api import *
from fabric.contrib.project import rsync_project

env.hosts = ['gpu1.psi.utoronto.ca',
             'gpu2.psi.utoronto.ca']

def sync_project():
    """Sync current state of repository to GPU servers"""
    run('mkdir -p ~/git')
    rsync_project('~/git', '~/git/neural-nets-pycuda', exclude=['.git', 'examples'])

