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

import numpy as np
from pycuda import gpuarray
from pycuda.curandom import rand as curand
from itertools import izip
from .. import pycuda_ops
from . import MaxPoolingLayer
from neural_nets.models import HiddenLayer
from neural_nets.pycuda_ops.elementwise import sign, sample_dropout_mask, \
     apply_dropout_mask
from neural_nets.pycuda_ops.matrix import extract_columns, insert_columns
from neural_nets.pycuda_ops.reductions import matrix_sum_out_axis


class MultiSequenceConvolutionLayer(HiddenLayer):
    def __init__(self, subregion_layers,
                 fully_connected_layers=None,
                 n_filters=None,
                 filter_width=None,
                 pool_size=None,
                 operation=None,
                 activation_function=None,
                 dropout=None,
                 lr_multiplier=None,
                 l1_penalty_weight=None,
                 l2_penalty_weight=None,
                 dtype=np.float32,
                 weight_scale=.01):

        self.subregion_layers = subregion_layers
        self.dtype = dtype
        self.dropout = dropout

        self.W = []
        self.b = []

        output_offset = 0
        param_idx = 0
        for layer in subregion_layers:
            n_in = layer['n_in']

            # Replace defaults
            if n_filters is not None:
                layer['n_filters'] = n_filters
            if operation is not None:
                layer['operation'] = operation
            if filter_width is not None:
                layer['filter_width'] = filter_width
            if pool_size is not None:
                layer['pool_size'] = pool_size
            if activation_function is not None:
                layer['activation_function'] = activation_function
            if l1_penalty_weight is not None:
                layer['l1_penalty_weight'] = l1_penalty_weight
            if l2_penalty_weight is not None:
                layer['l2_penalty_weight'] = l2_penalty_weight
            if lr_multiplier is not None:
                layer['lr_multiplier'] = lr_multiplier

            assert layer['operation'] in ('convolution', 'fully_connected')
            if layer['operation'] == 'fully_connected':
                layer['filter_width'] = layer['n_in']

            if not layer.has_key('weight_share'):
                layer['layer_type'] = 'master'
                _weight_scale = layer.get('weight_scale', weight_scale)
                if not layer.has_key('W'):
                    W = _weight_scale * \
                      curand((layer['n_filters'], 4 * layer['filter_width']),
                             dtype) - .5 * _weight_scale
                else:
                    W = layer['W']

                assert W.shape == (layer['n_filters'],
                                   4 * layer['filter_width'])
                self.W.append(W)

                if not layer.has_key('b'):
                    b = gpuarray.zeros((layer['n_filters'],), dtype)
                else:
                    b = layer['b']

                assert b.shape == (layer['n_filters'],)
                self.b.append(b)

                layer['param_idx'] = param_idx
                param_idx += 1

                layer['f'], layer['df'] = \
                  self._resolve_activation_fct(layer['activation_function'])

                if not layer.has_key('l1_penalty_weight'):
                    layer['l1_penalty_weight'] = 0.
                if not layer.has_key('l2_penalty_weight'):
                    layer['l2_penalty_weight'] = 0.
                if not layer.has_key('lr_multiplier'):
                    if layer['operation'] == 'convolution':
                        layer['lr_multiplier'] = 1.
                    elif layer['operation'] == 'fully_connected':
                        layer['lr_multiplier'] = \
                            float(1. / np.sqrt([layer['n_in']]))

            else:
                layer['layer_type'] = 'slave'
                master_layer = subregion_layers[layer['weight_share']]
                layer['n_filters'] = master_layer['n_filters']
                layer['filter_width'] = master_layer['filter_width']
                layer['param_idx'] = master_layer['param_idx']
                layer['activation_function'] = master_layer['activation_function']
                layer['f'] = master_layer['f']
                layer['df'] = master_layer['df']
                layer['operation'] = master_layer['operation']

            if layer['operation'] == 'convolution':
                layer['n_units'] = MaxPoolingLayer._compute_n_units(
                    layer['n_in'], layer['pool_size'], layer['n_filters'])
            elif layer['operation'] == 'fully_connected':
                layer['n_units'] = layer['n_filters']
            else:
                raise ValueError

            layer['output_offset'] = output_offset
            output_offset += layer['n_units']

        if fully_connected_layers is None:
            self.fully_connected_layers = None
        else:
            self.fully_connected_layers = []
            for fcl in fully_connected_layers:
                if isinstance(fcl, dict):
                    self.fully_connected_layers.append(HiddenLayer(*fcl))
                elif isinstance(fcl, HiddenLayer):
                    self.fully_connected_layers.append(fcl)
                else:
                    raise TypeError("fully connected layer must be a dictionary or "
                      "an instance of HiddenLayer")

        self.n_units = sum((layer['n_units'] for layer in subregion_layers))
        if self.fully_connected_layers is not None:
            self.fc_layer_offset = [self.n_units]

            for fcl in self.fully_connected_layers[:-1]:
                self.fc_layer_offset.append(self.fc_layer_offset[-1] + fcl.n_units)

            self.n_units += sum((fcl.n_units for fcl in self.fully_connected_layers))
        else:
            self.fc_layer_offset = None

        self.master_layers = filter(lambda l: l['layer_type'] == 'master',
                                    self.subregion_layers)

        self.l1_penalty_weight = any((l['l1_penalty_weight'] > 0.
                                      for l in self.master_layers))
        self.l2_penalty_weight = any((l['l2_penalty_weight'] > 0.
                                      for l in self.master_layers))

    @property
    def n_parameters(self):
        n_param = len(self.W) + len(self.b)
        if self.fully_connected_layers is not None:
            n_param += sum((fcl.n_parameters for fcl in self.fully_connected_layers))
        return n_param

    @property
    def n_in(self):
        return sum((l['n_in'] for l in self.subregion_layers))

    @property
    def lr_multiplier(self):
        lrm = 2 * [l.get('lr_multiplier', 1.) for l in self.subregion_layers
                   if l['layer_type'] == 'master']

        if self.fully_connected_layers is not None:
            for fcl in self.fully_connected_layers:
                lrm.extend(fcl.lr_multiplier)

        return lrm

    @property
    def parameters(self):
        param = self.W + self.b
        if self.fully_connected_layers is not None:
            for fcl in self.fully_connected_layers:
                param.extend(fcl.parameters)
        return param

    @parameters.setter
    def parameters(self, value):
        assert len(value) == self.n_parameters

        if self.fully_connected_layers is None:
            conv_params = value
        else:
            n_param = self.n_parameters
            n_param_fc = sum((fcl.n_parameters for fcl in self.fully_connected_layers))
            conv_params = value[:n_param-n_param_fc]
            fc_params = value[n_param-n_param_fc:]

            idx = 0
            for fcl in self.fully_connected_layers:
                fcl.parameters = fc_params[idx:idx+fcl.n_parameters]
                idx += fcl.n_parameters

        self.W = conv_params[:len(self.W)]
        self.b = conv_params[len(self.W):]

    def update_parameters(self, values, stream=None):
        assert len(values) == self.n_parameters

        if self.fully_connected_layers is None:
            conv_params = values
        else:
            n_param = self.n_parameters
            n_param_fc = sum((fcl.n_parameters for fcl in self.fully_connected_layers))
            conv_params = values[:n_param-n_param_fc]
            fc_params = values[n_param-n_param_fc:]

            idx = 0
            for fcl in self.fully_connected_layers:
                fcl.update_parameters(fc_params[idx:idx+fcl.n_parameters])
                idx += fcl.n_parameters

        for (param, (gparam, mult)) \
          in izip(self.W + self.b, conv_params):
          param._axpbyz(1., gparam, mult, param, stream=stream)

    @property
    def l1_penalty(self):
        l1_pen = np.sum(
            [float(l['l1_penalty_weight']) * gpuarray.sum(abs(W)).get()
             for l, W in izip(self.master_layers, self.W)])

        if self.fully_connected_layers is not None:
            for fcl in self.fully_connected_layers:
                l1_pen += fcl.l1_penalty

        return l1_pen

    @property
    def l2_penalty(self):
        l2_pen = np.sum(
            [float(l['l2_penalty_weight']) * .5 * gpuarray.sum(W ** 2.).get()
             for l, W in izip(self.master_layers, self.W)])

        if self.fully_connected_layers is not None:
            for fcl in self.fully_connected_layers:
                l2_pen += fcl.l2_penalty

        return l2_pen

    def feed_forward(self, input_data, prediction=False):
        assert all((input_data[0].shape[0] == i.shape[0] for i in input_data[1:]))

        N = input_data[0].shape[0]
        activations_pooled = gpuarray.empty((N, self.n_units),
                                            self.dtype)
        argmax = gpuarray.empty(activations_pooled.shape,
                                np.uint32)

        filtermaps = []

        for input_region, layer \
            in izip(input_data, self.subregion_layers):
            W = self.W[layer['param_idx']]
            b = self.b[layer['param_idx']]
            act_fct = layer['f']

            if layer['operation'] == 'convolution':
                filtermap = pycuda_ops.convolve_sequence(input_region, W, b)
                act_fct(filtermap)
                filtermaps.append(filtermap)
                pycuda_ops.max_pool(filtermap, layer['pool_size'],
                                    layer['n_filters'],
                                    width=layer['n_in'],
                                    pooled_offset=layer['output_offset'],
                                    target=activations_pooled, argmax=argmax)
            elif layer['operation'] == 'fully_connected':
                filtermap = \
                    pycuda_ops.fully_connected_layer(input_region, W, b)
                act_fct(filtermap)
                filtermaps.append(filtermap)
                insert_columns(filtermap, activations_pooled,
                               layer['output_offset'])

        if self.fully_connected_layers is not None:
            assert len(input_data) == \
              len(self.subregion_layers) + len(self.fully_connected_layers)

            activations_fc = []
            for input_fcl, fcl, offset \
              in izip(input_data[len(self.subregion_layers):],
                      self.fully_connected_layers,
                      self.fc_layer_offset):
                afc = \
                  fcl.feed_forward(input_fcl,
                                   prediction)[0]
                activations_fc.append([afc])
                insert_columns(afc, activations_pooled,
                               offset)
        else:
            activations_fc = None

        if self.dropout and not prediction:
            # Dropout only applies to subregion layer
            dropout_mask = sample_dropout_mask(activations_pooled,
                                               columns=(0, self.fc_layer_offset[0]))
        else:
            dropout_mask = None

        return activations_pooled, argmax, filtermaps, \
            dropout_mask, activations_fc

    def backprop(self, input_data, df_output, cache=None):
        if cache is None:
            cache = self.feed_forward(input_data)

        activations_pooled, argmax, filtermaps, dropout_mask, fc_cache = cache

        if self.dropout and dropout_mask is not None:
            apply_dropout_mask(df_output, dropout_mask,
                               columns=(0, self.fc_layer_offset[0]))

        df_W = []
        df_b = []
        df_filtermaps = []

        for input_region, filtermap, layer \
          in izip(input_data, filtermaps, self.subregion_layers):
            act_df = layer['df']

            if layer['operation'] == 'convolution':
                df_filtermap = pycuda_ops.max_pool_gradient(
                    filtermap, argmax, df_output, layer['pool_size'],
                    layer['n_filters'],
                    width_pooled=layer['n_units'] / layer['n_filters'],
                    pooled_offset=layer['output_offset'])

                df_filtermaps.append(df_filtermap)

                df_conv = act_df(filtermap)
                delta = df_conv * df_filtermap
                df_b_layer = pycuda_ops.sum_delta(delta, layer['n_filters'])
                df_W_layer = pycuda_ops.convolve_sequence_gradient(
                    input_region, delta, layer['filter_width'],
                    layer['n_filters'])
            elif layer['operation'] == 'fully_connected':
                df_filtermap = act_df(filtermap)
                df_filtermaps.append(df_filtermap)

                delta = df_filtermap * df_output
                df_W_layer = \
                    pycuda_ops.fully_connected_layer_gradient(input_region, delta)
                df_b_layer = matrix_sum_out_axis(delta, 0)

            if layer['layer_type'] == 'master':
                df_W.append(df_W_layer)
                df_b.append(df_b_layer)
            else:
                df_W[layer['param_idx']] += df_W_layer
                df_b[layer['param_idx']] += df_b_layer

        for df_W_i, layer in izip(df_W, self.master_layers):
            if layer['l1_penalty_weight']:
                df_W_i -= layer['l1_penalty_weight'] * sign(df_W_i)
            if layer['l2_penalty_weight']:
                df_W_i -= layer['l2_penalty_weight'] * self.W

        if self.fully_connected_layers is not None:
            assert len(input_data) == \
              len(self.subregion_layers) + len(self.fully_connected_layers)

            grad_fc = []
            df_input_fc = []
            input_fc = input_data[-len(self.fully_connected_layers):]

            for ifc, offset, fcl, cache in \
              izip(input_fc, self.fc_layer_offset,
                   self.fully_connected_layers, fc_cache):
                df_output_fc = extract_columns(df_output,
                                               offset,
                                               offset + fcl.n_units)
                grads_layer, df_input_layer = \
                  fcl.backprop(ifc, df_output_fc,
                               cache)
                grad_fc.extend(grads_layer)
                df_input_fc.extend(df_input_layer)

            return df_W + df_b + list(grad_fc), \
              (df_filtermaps, df_input_fc)

        return df_W + df_b, df_filtermaps
