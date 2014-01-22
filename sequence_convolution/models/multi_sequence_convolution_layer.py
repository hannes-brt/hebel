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
from itertools import izip
from .. import pycuda_ops
from . import MaxPoolingLayer, SubregionLayer, SlavedSubregionLayer
from hebel import sampler
from hebel.layers import HiddenLayer
from hebel.pycuda_ops.elementwise import sign, sample_dropout_mask, \
     apply_dropout_mask
from hebel.pycuda_ops.matrix import extract_columns, insert_columns
from hebel.pycuda_ops.reductions import matrix_sum_out_axis


class MultiSequenceConvolutionLayer(HiddenLayer):
    def __init__(self, subregion_layers,
                 fully_connected_layers=None,
                 n_filters=None,
                 filter_width=None,
                 pool_size=None,
                 activation_function=None,
                 dropout=None,
                 lr_multiplier=1.,
                 l1_penalty_weight=0.,
                 l2_penalty_weight=0.,
                 dtype=np.float32,
                 weight_scale=.01):

        self.subregion_layers = subregion_layers
        self.dtype = dtype
        self.dropout = dropout

        output_offset = 0
        param_idx = 0
        for i, layer in enumerate(subregion_layers):
            # Replace defaults
            if n_filters is not None and 'n_filters' not in layer:
                layer['n_filters'] = n_filters
            if filter_width is not None and 'filter_width' not in layer:
                layer['filter_width'] = filter_width
            if pool_size is not None and 'pool_sizse' not in layer:
                layer['pool_size'] = pool_size
            if activation_function is not None and 'activation_function' not in layer:
                layer['activation_function'] = activation_function
            if 'l1_penalty_weight' not in layer:
                layer['l1_penalty_weight'] = l1_penalty_weight
            if 'l2_penalty_weight' not in layer:
                layer['l2_penalty_weight'] = l2_penalty_weight
            if 'lr_multiplier' not in layer:
                layer['lr_multiplier'] = lr_multiplier
            layer['output_offset'] = output_offset

            if not layer.has_key('weight_share'):
                _weight_scale = layer.get('weight_scale', weight_scale)
                subregion_layers[i] = SubregionLayer(layer['n_in'],
                                                     layer['n_filters'],
                                                     layer['filter_width'],
                                                     layer['pool_size'],
                                                     layer['activation_function'],
                                                     layer['l1_penalty_weight'],
                                                     layer['l2_penalty_weight'],
                                                     layer['lr_multiplier'],
                                                     weight_scale=weight_scale,
                                                     param_idx=param_idx,
                                                     output_offset=output_offset)
                param_idx += 1
            else:
                master_layer = subregion_layers[layer['weight_share']]
                subregion_layers[i] = SlavedSubregionLayer(master_layer, output_offset)

            output_offset += subregion_layers[i].n_units

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

        self.n_units = sum((layer.n_units for layer in subregion_layers))
        if self.fully_connected_layers is not None:
            self.fc_layer_offset = [self.n_units]

            for fcl in self.fully_connected_layers[:-1]:
                self.fc_layer_offset.append(self.fc_layer_offset[-1] + fcl.n_units)

            self.n_units += sum((fcl.n_units for fcl in self.fully_connected_layers))
        else:
            self.fc_layer_offset = [self.n_units]

        self.master_layers = filter(lambda l: l.is_master_layer,
                                    self.subregion_layers)

        self.l1_penalty_weight = any((l.l1_penalty_weight > 0.
                                      for l in self.master_layers))
        self.l2_penalty_weight = any((l.l2_penalty_weight > 0.
                                      for l in self.master_layers))

        self.persistent_temp_objects_config = (
            ('activations_pooled', ('batch_size', self.n_units), np.float32),
            ('argmax', ('batch_size', self.n_units), np.uint32)
        )

    def preallocate_temp_objects(self, batch_size):
        super(MultiSequenceConvolutionLayer, self).preallocate_temp_objects(batch_size)
        
        for layer in self.subregion_layers:
            layer.preallocate_temp_objects(batch_size)
        for layer in self.fully_connected_layers:
            layer.preallocate_temp_objects(batch_size)

    @property
    def W(self):
        return [l.W for l in self.master_layers]

    @property
    def b(self):
        return [l.b for l in self.master_layers]

    @property
    def n_parameters(self):
        n_param = len(self.W) + len(self.b)
        if self.fully_connected_layers is not None:
            n_param += sum((fcl.n_parameters for fcl in self.fully_connected_layers))
        return n_param

    @property
    def n_in(self):
        return sum((l.n_in for l in self.subregion_layers))

    @property
    def lr_multiplier(self):
        lrm = 2 * [l.lr_multiplier for l in self.master_layers]

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

        for layer, W, b in izip(self.master_layers,
                                conv_params[:len(self.W)],
                                conv_params[len(self.W):]):
            layer.W = W
            layer.b = b

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
            [float(l.l1_penalty_weight) * gpuarray.sum(abs(W)).get()
             for l, W in izip(self.master_layers, self.W)])

        if self.fully_connected_layers is not None:
            for fcl in self.fully_connected_layers:
                l1_pen += fcl.l1_penalty

        return l1_pen

    @property
    def l2_penalty(self):
        l2_pen = np.sum(
            [float(l.l2_penalty_weight) * .5 * gpuarray.sum(W ** 2.).get()
             for l, W in izip(self.master_layers, self.W)])

        if self.fully_connected_layers is not None:
            for fcl in self.fully_connected_layers:
                l2_pen += fcl.l2_penalty

        return l2_pen

    def feed_forward(self, input_data, prediction=False):
        assert all((input_data[0].shape[0] == i.shape[0] for i in input_data[1:]))

        N = input_data[0].shape[0]
        activations_pooled = self.get_temp_object('activations_pooled',
                                                  (N, self.n_units),
                                                  self.dtype)
        argmax = self.get_temp_object('argmax',
                                      activations_pooled.shape,
                                      np.uint32)

        filtermaps = []

        for input_region, layer \
            in izip(input_data, self.subregion_layers):
            filtermap = layer.feed_forward(input_region, prediction,
                                           activations_pooled, argmax)
            filtermaps.append(filtermap)

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

        # if self.dropout and not prediction:
        #     # Dropout only applies to subregion layer
        #     dropout_mask = sample_dropout_mask(activations_pooled,
        #                                        columns=(0, self.fc_layer_offset[0]))
        # else:
        #     dropout_mask = None

        return activations_pooled, argmax, filtermaps, activations_fc

    def backprop(self, input_data, df_output, cache=None):
        if cache is None:
            cache = self.feed_forward(input_data)

        activations_pooled, argmax, filtermaps, fc_cache = cache

        # if self.dropout and dropout_mask is not None:
        #     apply_dropout_mask(df_output, dropout_mask,
        #                        columns=(0, self.fc_layer_offset[0]))

        df_W = []
        df_b = []
        df_filtermaps = []

        for input_region, filtermap, layer \
          in izip(input_data, filtermaps, self.subregion_layers):

            ((df_W_layer, df_b_layer), df_filtermap) = \
                layer.backprop(input_region, df_output,
                                      filtermap, argmax)

            df_filtermaps.append(df_filtermap)

            if layer.is_master_layer:
                df_W.append(df_W_layer)
                df_b.append(df_b_layer)
            else:
                df_W[layer.param_idx] += df_W_layer
                df_b[layer.param_idx] += df_b_layer

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
                df_input_fc.append(df_input_layer)

            return df_W + df_b + list(grad_fc), \
              (df_filtermaps, df_input_fc)

        return df_W + df_b, df_filtermaps
