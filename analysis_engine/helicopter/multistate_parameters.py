# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

from flightdatautilities import units as ut
from flightdataaccessor.parameter import MappedArray

from analysis_engine.node import (
    A, MultistateDerivedParameterNode,
    M,
    P,
    S,
)
from analysis_engine.library import (
    align,
    all_of,
    any_of,
    moving_average,
    nearest_neighbour_mask_repair,
    np_ma_zeros_like,
    runs_of_ones,
    slices_and,
    slices_remove_small_slices,
)


class GearOnGround(MultistateDerivedParameterNode):
    '''
    Combination of left and right main gear signals.
    '''
    align = False
    values_mapping = {
        0: 'Air',
        1: 'Ground',
    }

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        gog_available = any_of(('Gear (L) On Ground', 'Gear (R) On Ground'), available)
        if gog_available:
            return True
        elif all_of(('Vertical Speed', 'Eng (*) Torque Avg'), available):
            return True
        else:
            return False

    def derive(self,
               gl=M('Gear (L) On Ground'),
               gr=M('Gear (R) On Ground'),
               vert_spd=P('Vertical Speed'),
               torque=P('Eng (*) Torque Avg'),
               ac_series=A('Series'),
               collective=P('Collective')):

        if gl and gr:
            delta = abs((gl.offset - gr.offset) * gl.frequency)
            if 0.75 < delta or delta < 0.25:
                # If the samples of the left and right gear are close together,
                # the best representation is to map them onto a single
                # parameter in which we accept that either wheel on the ground
                # equates to gear on ground.
                self.array = np.ma.logical_or(gl.array, gr.array)
                self.frequency = gl.frequency
                self.offset = gl.offset
                return
            else:
                # If the paramters are not co-located, then
                # merge_two_parameters creates the best combination possible.
                self.array, self.frequency, self.offset = merge_two_parameters(gl, gr)
                return
        elif gl or gr:
            gear = gl or gr
            self.array = gear.array
            self.frequency = gear.frequency
            self.offset = gear.offset
        elif vert_spd and torque:
            vert_spd_limit = 100.0
            torque_limit = 30.0
            if ac_series and ac_series.value == 'Columbia 234':
                vert_spd_limit = 125.0
                torque_limit = 22.0
                collective_limit = 15.0
                
                vert_spd_array = align(vert_spd, torque) if vert_spd.hz != torque.hz else vert_spd.array
                collective_array = align(collective, torque) if collective.hz != torque.hz else collective.array
                
                vert_spd_array = moving_average(vert_spd_array)
                torque_array = moving_average(torque.array)                
                collective_array = moving_average(collective_array)
                
                roo_vs_array = runs_of_ones(abs(vert_spd_array) < vert_spd_limit, min_samples=1)
                roo_torque_array = runs_of_ones(torque_array < torque_limit, min_samples=1)
                roo_collective_array = runs_of_ones(collective_array < collective_limit, min_samples=1)
                
                vs_and_torque = slices_and(roo_vs_array, roo_torque_array)
                grounded = slices_and(vs_and_torque, roo_collective_array)
                
                array = np_ma_zeros_like(vert_spd_array)
                for _slice in slices_remove_small_slices(grounded, count=2):
                    array[_slice] = 1
                array.mask = vert_spd_array.mask | torque_array.mask
                array.mask = array.mask | collective_array.mask
                self.array = nearest_neighbour_mask_repair(array)
                self.frequency = torque.frequency
                self.offset = torque.offset                
                
            else:
                vert_spd_array = align(vert_spd, torque) if vert_spd.hz != torque.hz else vert_spd.array
                # Introducted for S76 and Bell 212 which do not have Gear On Ground available
                
                vert_spd_array = moving_average(vert_spd_array)
                torque_array = moving_average(torque.array)
                
                grounded = slices_and(runs_of_ones(abs(vert_spd_array) < vert_spd_limit, min_samples=1), 
                                      runs_of_ones(torque_array < torque_limit, min_samples=1))
                
                array = np_ma_zeros_like(vert_spd_array)
                for _slice in slices_remove_small_slices(grounded, count=2):
                    array[_slice] = 1
                array.mask = vert_spd_array.mask | torque_array.mask
                self.array = nearest_neighbour_mask_repair(array)
                self.frequency = torque.frequency
                self.offset = torque.offset
                
        else:
            # should not get here if can_operate is correct
            raise NotImplementedError()