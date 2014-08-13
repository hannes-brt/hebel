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

from hebel.layers import MultiColumnLayer, Column, HiddenLayer
from . import SequenceConvolutionAndPoolLayer, SlavedSequenceConvolutionAndPoolLayer, \
    Convolution1DAndPoolLayer, SlavedConvolution1DAndPoolLayer

class MultiSequenceConvolutionLayer(MultiColumnLayer):
    def __init__(self, subregion_columns,
                 aux_data_layer=None,
                 activation_function='relu',
                 pool_size=None,
                 pooling_op='max',
                 dropout=False,
                 lr_multiplier=1.,
                 l1_penalty_weight=0.,
                 l2_penalty_weight=0.,
                 weights_scale=.01,
                 padding=(True, True)):

        columns = []

        for i_column, column in enumerate(subregion_columns):

            # Columns can be supplied in different ways
            if isinstance(column, (Column, HiddenLayer)):
                # Column is an instance of Column or HiddenLayer
                columns.append(column)

            elif isinstance(column, dict):
                # If the column is given as a dict, then slave the entire column
                column_obj = []
                for i_layer, layer in enumerate(columns[column['slaved']].hidden_layers):
                    if i_layer == 0:
                        # First layer is a sequence convolution
                        layer_obj = SlavedSequenceConvolutionAndPoolLayer(
                            layer, column.get('n_in'),
                            column.get('padding', padding))
                    else:
                        # Other layers are 1D convolutions
                        layer_obj = SlavedConvolution1DAndPoolLayer(
                            layer, column_obj[-1].n_units_per_filter,
                            column.get('padding', padding))
                    column_obj.append(layer_obj)
                column_obj = Column(column_obj)
                columns.append(column_obj)

            elif isinstance(column, (list, tuple)):
                # column given as a list-like of dicts (one for each layer)
                column_obj = []
                for i_layer, layer in enumerate(column):
                    if isinstance(layer, HiddenLayer):
                        column_obj.append(layer)
                    else:
                        if i_layer == 0:
                            if 'slaved' not in layer:
                                layer_obj = SequenceConvolutionAndPoolLayer(
                                    layer.get('n_in'),
                                    layer.get('filter_width'),
                                    layer.get('n_filters'),
                                    layer.get('pool_size', pool_size),
                                    layer.get('activation_function', activation_function),
                                    layer.get('pooling_op', pooling_op),
                                    layer.get('dropout', dropout),
                                    layer.get('weights_scale', weights_scale),
                                    layer.get('W'),
                                    layer.get('b'),
                                    layer.get('l1_penalty_weight', l1_penalty_weight),
                                    layer.get('l2_penalty_weight', l2_penalty_weight),
                                    layer.get('padding') or padding
                                )
                            else:
                                master_layer = columns[layer['slaved']].hidden_layers[i_layer]
                                layer_obj = SlavedSequenceConvolutionAndPoolLayer(
                                    master_layer,
                                    layer.get('n_in'),
                                    layer.get('padding', padding)
                                )

                        else:
                            if 'slaved' not in layer:
                                layer_obj = Convolution1DAndPoolLayer(
                                    column_obj[-1].n_units_per_filter,
                                    layer.get('filter_width'),
                                    column_obj[-1].n_filters,
                                    layer.get('n_filters'),
                                    layer.get('pool_size', pool_size),
                                    layer.get('activation_function', activation_function),
                                    layer.get('pooling_op') or pooling_op,
                                    layer.get('dropout') or dropout,
                                    layer.get('weights_scale', weights_scale),
                                    layer.get('W'),
                                    layer.get('b'),
                                    layer.get('l1_penalty_weight', l1_penalty_weight),
                                    layer.get('l2_penalty_weight', l2_penalty_weight),
                                    layer.get('padding', padding)
                                )
                            else:
                                master_layer = columns[layer['slaved']].hidden_layers[i_layer]
                                layer_obj = SlavedConvolution1DAndPoolLayer(
                                    master_layer,
                                    column_obj[-1].n_in,
                                    layer.get('padding', padding)
                                )

                        column_obj.append(layer_obj)
                column_obj = Column(column_obj)
                columns.append(column_obj)
            else:
                raise ValueError("Unknown format for column")

        if isinstance(aux_data_layer, (Column, HiddenLayer)):
            columns.append(aux_data_layer)
        elif isinstance(aux_data_layer, dict):
            layer_obj = HiddenLayer(
                aux_data_layer.get('n_in'),
                aux_data_layer.get('n_units'),
                aux_data_layer.get('activation_function', activation_function),
                aux_data_layer.get('dropout', dropout),
                aux_data_layer.get('parameters'),
                aux_data_layer.get('weights_scale'),
                aux_data_layer.get('l1_penalty_weight', l1_penalty_weight),
                aux_data_layer.get('l2_penalty_weight', l2_penalty_weight),
                aux_data_layer.get('lr_multiplier', 2 * [lr_multiplier])
            )
            columns.append(layer_obj)
        elif aux_data_layer is None:
            pass
        else:
            raise ValueError("Unknown format for aux_data_layer")

        super(MultiSequenceConvolutionLayer, self).__init__(columns, input_as_list=True)
