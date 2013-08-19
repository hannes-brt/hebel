# Copyright (C) 2013  Hannes Bretschneider

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

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
