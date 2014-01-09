# Copyright (c) 2008--2011, Theano Development Team, Hannes Bretschneider
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Theano nor the names of its contributors may be
#       used to endorse or promote products derived from this software without
#       specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS

"""Parse YAML configuration files that describe experiment setups.
Heavily copied from pylearn2
"""

import re, yaml, os, uuid
from .utils.call_check import checked_call
from .utils import serial
from .utils.string_utils import match
import warnings
from itertools import izip

is_initialized = False
root = os.path.curdir

def run_from_config(yaml_src):
    config = load(yaml_src)
    optimizer = config['optimizer']
    run_conf = config['run_conf']
    run_conf['yaml_config'] = yaml_src
    run_conf['task_id'] = str(uuid.uuid4())
    optimizer.run(**run_conf)

    if config.has_key('test_dataset'):
        test_data = config['test_dataset']['test_data']
        model = optimizer.model
        progress_monitor = optimizer.progress_monitor

        test_error = model.test_error(test_data, average=True)
        progress_monitor.test_error = test_error

def load(stream, overrides=None, **kwargs):
    """
    Loads a YAML configuration from a string or file-like object.

    Parameters
    ----------
    stream : str or object
        Either a string containing valid YAML or a file-like object
        supporting the .read() interface.
    overrides : dict, optional
        A dictionary containing overrides to apply. The location of
        the override is specified in the key as a dot-delimited path
        to the desired parameter, e.g. "model.corruptor.corruption_level".

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified an
        Python object to instantiate).

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """

    global is_initialized
    if not is_initialized:
        initialize()

    if isinstance(stream, basestring):
        string = stream
    else:
        string = '\n'.join(stream.readlines())

    # processed_string = preprocess(string)

    proxy_graph = yaml.load(string, **kwargs)

    from . import init
    init_dict = proxy_graph.get('init', {})
    init(**init_dict)
    
    if overrides is not None:
        handle_overrides(proxy_graph, overrides)
    return instantiate_all(proxy_graph)


def load_path(path, overrides=None, **kwargs):
    """
    Convenience function for loading a YAML configuration from a file.

    Parameters
    ----------
    path : str
        The path to the file to load on disk.
    overrides : dict, optional
        A dictionary containing overrides to apply. The location of
        the override is specified in the key as a dot-delimited path
        to the desired parameter, e.g. "model.corruptor.corruption_level".

    Returns
    -------
    graph : dict or object
        The dictionary or object (if the top-level element specified an
        Python object to instantiate).

    Notes
    -----
    Other keyword arguments are passed on to `yaml.load`.
    """
    f = open(path, 'r')
    content = ''.join(f.readlines())
    f.close()

    if not isinstance(content, str):
        raise AssertionError("Expected content to be of type str but it is "+str(type(content)))

    return load(content, **kwargs)


def handle_overrides(graph, overrides):
    """
    Handle any overrides for this model configuration.

    Parameters
    ----------
    graph : dict or object
        A dictionary (or an ObjectProxy) containing the object graph
        loaded from a YAML file.
    overrides : dict
        A dictionary containing overrides to apply. The location of
        the override is specified in the key as a dot-delimited path
        to the desired parameter, e.g. "model.corruptor.corruption_level".
    """
    for key in overrides:
        levels = key.split('.')
        part = graph
        for lvl in levels[:-1]:
            try:
                part = part[lvl]
            except KeyError:
                raise KeyError("'%s' override failed at '%s'", (key, lvl))
        try:
            part[levels[-1]] = overrides[key]
        except KeyError:
            raise KeyError("'%s' override failed at '%s'", (key, levels[-1]))


def instantiate_all(graph):
    """
    Instantiate all ObjectProxy objects in a nested hierarchy.

    Parameters
    ----------
    graph : dict or object
        A dictionary (or an ObjectProxy) containing the object graph
        loaded from a YAML file.

    Returns
    -------
    graph : dict or object
        The dictionary or object resulting after the recursive instantiation.
    """

    def should_instantiate(obj):
        classes = [ObjectProxy, dict, list]
        return True in [isinstance(obj, cls) for cls in classes]

    if not isinstance(graph, list):
        for key in graph:
            if should_instantiate(graph[key]):
                graph[key] = instantiate_all(graph[key])
        if hasattr(graph, 'keys'):
            for key in graph.keys():
                if should_instantiate(key):
                    new_key = instantiate_all(key)
                    graph[new_key] = graph[key]
                    del graph[key]

    if isinstance(graph, ObjectProxy):
        graph = graph.instantiate()

    if isinstance(graph, list):
        for i, elem in enumerate(graph):
            if should_instantiate(elem):
                graph[i] = instantiate_all(elem)

    return graph


class ObjectProxy(object):
    """
    Class used to delay instantiation of objects so that overrides can be
    applied.
    """
    def __init__(self, cls, kwds, yaml_src):
        """

        """
        self.cls = cls
        self.kwds = kwds
        self.yaml_src = yaml_src
        self.instance = None

    def __setitem__(self, key, value):
        self.kwds[key] = value

    def __getitem__(self, key):
        return self.kwds[key]

    def __iter__(self):
        return self.kwds.__iter__()

    def keys(self):
        return list(self.kwds)

    def instantiate(self):
        """
        Instantiate this object with the supplied parameters in `self.kwds`,
        or if already instantiated, return the cached instance.
        """
        if self.instance is None:
            self.instance = checked_call(self.cls, self.kwds)
        #endif
        try:
            self.instance.yaml_src = self.yaml_src
        except AttributeError:
            pass
        return self.instance


def try_to_import(tag_suffix):
    components = tag_suffix.split('.')
    modulename = '.'.join(components[:-1])
    try:
        exec('import %s' % modulename)
    except ImportError, e:
        # We know it's an ImportError, but is it an ImportError related to
        # this path,
        # or did the module we're importing have an unrelated ImportError?
        # and yes, this test can still have false positives, feel free to
        # improve it
        pieces = modulename.split('.')
        str_e = str(e)
        found = True in [piece.find(str(e)) != -1 for piece in pieces]

        if found:
            # The yaml file is probably to blame.
            # Report the problem with the full module path from the YAML
            # file
            raise ImportError("Could not import %s; ImportError was %s" %
                              (modulename, str_e))
        else:

            pcomponents = components[:-1]
            assert len(pcomponents) >= 1
            j = 1
            while j <= len(pcomponents):
                modulename = '.'.join(pcomponents[:j])
                try:
                    exec('import %s' % modulename)
                except:
                    base_msg = 'Could not import %s' % modulename
                    if j > 1:
                        modulename = '.'.join(pcomponents[:j-1])
                        base_msg += ' but could import %s' % modulename
                    raise ImportError(base_msg + '. Original exception: '+str(e))
                j += 1



    try:
        obj = eval(tag_suffix)
    except AttributeError, e:
        try:
            # Try to figure out what the wrong field name was
            # If we fail to do it, just fall back to giving the usual
            # attribute error
            pieces = tag_suffix.split('.')
            module = '.'.join(pieces[:-1])
            field = pieces[-1]
            candidates = dir(eval(module))

            msg = ('Could not evaluate %s. ' % tag_suffix) + \
            'Did you mean ' + match(field, candidates) +'? '+ \
            'Original error was '+str(e)

        except:
            warnings.warn("Attempt to decipher AttributeError failed")
            raise AttributeError( ('Could not evaluate %s. ' % tag_suffix) +
                'Original error was '+str(e))
        raise AttributeError( msg )
    return obj


def multi_constructor(loader, tag_suffix, node):
    """
    Constructor function passed to PyYAML telling it how to construct
    objects from argument descriptions. See PyYAML documentation for
    details on the call signature.
    """
    yaml_src = yaml.serialize(node)
    mapping = loader.construct_mapping(node)
    if '.' not in tag_suffix:
        classname = tag_suffix
        rval = ObjectProxy(classname, mapping, yaml_src)
    else:
        classname = try_to_import(tag_suffix)
        rval = ObjectProxy(classname, mapping, yaml_src)

    return rval


def multi_constructor_pkl(loader, tag_suffix, node):
    """
    Constructor function passed to PyYAML telling it how to load
    objects from paths to .pkl files. See PyYAML documentation for
    details on the call signature.
    """

    mapping = loader.construct_yaml_str(node)
    if tag_suffix != "" and tag_suffix != u"":
        raise AssertionError('Expected tag_suffix to be "" but it is "'+tag_suffix+'"')

    rval = ObjectProxy(None, {}, yaml.serialize(node))
    rval.instance = serial.load(mapping)

    return rval


def multi_constructor_import(loader, tag_suffix, node):
    yaml_src = yaml.serialize(node)
    mapping = loader.construct_mapping(node)
    if '.' not in tag_suffix:
        raise yaml.YAMLError("import tag suffix contains no '.'")
    else:
        rval = try_to_import(tag_suffix)
    return rval

def multi_constructor_include(load, tag_suffix, node):
    global root

    old_root = root

    filename = os.path.join(root, loader.construct_scalar(node))
    root = os.path.split(filename)[0]
    data = yaml.load(open(filename), 'r')

    root = old_root
    return data

def initialize():
    """
    Initialize the configuration system by installing YAML handlers.
    Automatically done on first call to load() specified in this file.
    """
    global is_initialized
    # Add the custom multi-constructor
    yaml.add_multi_constructor('!obj:', multi_constructor)
    yaml.add_multi_constructor('!pkl:', multi_constructor_pkl)
    yaml.add_multi_constructor('!import:', multi_constructor_import)
    yaml.add_multi_constructor('!include:', multi_constructor_include)

    def import_constructor(loader, node):
        value = loader.construct_scalar(node)
        return try_to_import(value)

    yaml.add_constructor('!import', import_constructor)
    yaml.add_implicit_resolver(
        '!import',
        re.compile(r'(?:[a-zA-Z_][\w_]+\.)+[a-zA-Z_][\w_]+')
    )
    is_initialized = True
