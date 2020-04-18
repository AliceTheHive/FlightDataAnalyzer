# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import math
import numpy as np
import six

from pprint import pformat

from flightdatautilities import aircrafttables as at, dateext, units as ut

from hdfaccess.parameter import MappedArray

from analysis_engine.node import (
    A, M, P, S, helicopter, MultistateDerivedParameterNode, derived_param_from_hdf
)

from analysis_engine.library import (
    align,
    all_of,
    any_of,
    calculate_flap,
    calculate_slat,
    clump_multistate,
    datetime_of_index,
    find_edges_on_state_change,
    including_transition,
    index_at_value,
    index_closest_value,
    first_valid_parameter,
    merge_masks,
    merge_two_parameters,
    moving_average,
    nearest_neighbour_mask_repair,
    np_ma_masked_zeros_like,
    np_ma_zeros_like,
    offset_select,
    repair_mask,
    runs_of_ones,
    second_window,
    slice_duration,
    slices_and,
    slices_and_not,
    slices_extend,
    slices_from_to,
    slices_overlap,
    slices_overlap_merge,
    slices_remove_small_gaps,
    slices_remove_small_slices,
    smooth_signal,
    step_values,
    vstack_params,
    vstack_params_where_state,
)

from analysis_engine.settings import (
    MIN_CORE_RUNNING,
    MIN_FAN_RUNNING,
    MIN_FUEL_FLOW_RUNNING,
    REVERSE_THRUST_EFFECTIVE_EPR,
    REVERSE_THRUST_EFFECTIVE_N1
)

from flightdatautilities.numpy_utils import slices_int

logger = logging.getLogger(name=__name__)

class AOAAbnormalOperation(MultistateDerivedParameterNode):
    '''
    AOA operation state. A multistate parameter that stacks all the AOA failure, signal failure,
    heater and correction parameters to identify abnormal operation of the AOA sensors.
    '''
    name = 'AOA Abnormal Operation'
    units = None

    values_mapping = {
        0: '-',
        1: 'AOA (L) Failure',
        2: 'AOA (L) Signal Failure',
        3: 'AOA (L) Primary Heater',
        4: 'AOA (R) Failure',
        5: 'AOA (R) Signal Failure',
        6: 'AOA (R) Primary Heater',
        7: 'AOA Signal Failure',
        8: 'AOA Secondary Heater',
        9: 'AOA Correction Program',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)


    def derive(self,
                 aoa_l_fail=P('AOA (L) Failure'),
                 aoa_l_signal_fail=P('AOA (L) Signal Failure'),
                 aoa_l_heater=P('AOA (L) Primary Heater'),
                 aoa_r_fail=P('AOA (R) Failure'),
                 aoa_r_signal_fail=P('AOA (R) Signal Failure'),
                 aoa_r_heater=P('AOA (R) Primary Heater'),
                 aoa_signal_fail=P('AOA Signal Failure'),
                 aoa_sec_heater=P('AOA Secondary Heater'),
                 aoa_correction=P('AOA Correction Program'),):

        parameter = first_valid_parameter(aoa_l_fail, aoa_l_signal_fail, aoa_l_heater,
                  aoa_r_fail, aoa_r_signal_fail, aoa_r_heater,
                  aoa_signal_fail, aoa_sec_heater, aoa_correction)

        if parameter:
            self.array = np_ma_zeros_like(parameter.array)

        if aoa_l_fail:
            self.array[(aoa_l_fail.array == 'Failed').filled(False)] = aoa_l_fail.name
        if aoa_l_signal_fail:
            self.array[(aoa_l_signal_fail.array == 'Failed').filled(False)] = aoa_l_signal_fail.name
        if aoa_l_heater:
            self.array[(aoa_l_heater.array != 'On').filled(False)] = aoa_l_heater.name
        if aoa_r_fail:
            self.array[(aoa_r_fail.array == 'Failed').filled(False)] = aoa_r_fail.name
        if aoa_r_signal_fail:
            self.array[(aoa_r_signal_fail.array == 'Failed').filled(False)] = aoa_r_signal_fail.name
        if aoa_r_heater:
            self.array[(aoa_r_heater.array != 'On').filled(False)] = aoa_r_heater.name
        if aoa_signal_fail:
            self.array[(aoa_signal_fail.array == 'Failed').filled(False)] = aoa_signal_fail.name
        if aoa_sec_heater:
            self.array[(aoa_sec_heater.array != 'On').filled(False)] = aoa_sec_heater.name
        if aoa_correction:
            self.array[(aoa_correction.array == 'Yes').filled(False)] = aoa_correction.name


class APEngaged(MultistateDerivedParameterNode):
    '''
    Determines if *any* of the "AP (*) Engaged" parameters are recording the
    state of Engaged.

    This is a discrete with only the Engaged state.
    '''

    name = 'AP Engaged'
    units = None
    values_mapping = {0: '-', 1: 'Engaged'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               ap1=M('AP (1) Engaged'),
               ap2=M('AP (2) Engaged'),
               ap3=M('AP (3) Engaged')):

        stacked = vstack_params_where_state(
            (ap1, 'Engaged'),
            (ap2, 'Engaged'),
            (ap3, 'Engaged'),
        )
        self.array = stacked.any(axis=0)
        self.array.mask = stacked.mask.any(axis=0)


class APChannelsEngaged(MultistateDerivedParameterNode):
    '''
    Assess the number of autopilot systems engaged.

    Airbus and Boeing = 1 autopilot at a time except when "Land" mode
    selected when 2 (Dual) or 3 (Triple) can be engaged. Airbus favours only
    2 APs, Boeing is happier with 3 though some older types may only have 2.
    '''
    name = 'AP Channels Engaged'
    units = None
    values_mapping = {0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'}

    @classmethod
    def can_operate(cls, available):
        return len(available) >= 2

    def derive(self,
               ap1=M('AP (1) Engaged'),
               ap2=M('AP (2) Engaged'),
               ap3=M('AP (3) Engaged')):
        stacked = vstack_params_where_state(
            (ap1, 'Engaged'),
            (ap2, 'Engaged'),
            (ap3, 'Engaged'),
        )
        self.array = stacked.sum(axis=0)
        self.offset = offset_select('mean', [ap1, ap2, ap3])


class APLateralMode(MultistateDerivedParameterNode):
    '''
    '''
    name = 'AP Lateral Mode'
    units = None
    # Values and states match X-Plane visualisation model documentation.
    values_mapping = {
        0: '-',
        2: 'RWY',
        4: 'RWY TRK',
        6: 'NAV',
        14: 'LOC CAPT',
        16: 'LOC',
        20: 'APP NAV',
        22: 'ROLL OUT',
        24: 'LAND',
        64: 'HDG',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               lateral_mode_selected=M('Lateral Mode Selected'),
               runway_mode_active=M('Runway Mode Active'),
               nav_mode_active=M('NAV Mode Active'),
               ils_localizer_capture_active=M('ILS Localizer Capture Active'),
               ils_localizer_track_active=M('ILS Localizer Track Active'),
               roll_go_around_mode_active=M('Roll Go Around Mode Active'),
               land_track_active=M('Land Track Active'),
               heading_mode_active=M('Heading Mode Active')):
        parameter = next(p for p in (lateral_mode_selected,
                                     runway_mode_active,
                                     nav_mode_active,
                                     ils_localizer_capture_active,
                                     ils_localizer_track_active,
                                     roll_go_around_mode_active,
                                     land_track_active,
                                     heading_mode_active) if p)
        self.array = np_ma_zeros_like(parameter.array)

        if lateral_mode_selected:
            self.array[lateral_mode_selected.array == 'Runway Mode Active'] = 'RWY'
            self.array[lateral_mode_selected.array == 'NAV Mode Active'] = 'NAV'
            self.array[lateral_mode_selected.array == 'ILS Localizer Capture Active'] = 'LOC CAPT',
        if runway_mode_active:
            self.array[runway_mode_active.array == 'Activated'] = 'RWY'
        if nav_mode_active:
            self.array[nav_mode_active.array == 'Activated'] = 'NAV'
        if ils_localizer_capture_active:
            self.array[ils_localizer_capture_active.array == 'Activated'] = 'LOC CAPT'
        if ils_localizer_track_active:
            self.array[ils_localizer_track_active.array == 'Activated'] = 'LOC'
        if roll_go_around_mode_active:
            self.array[roll_go_around_mode_active.array == 'Activated'] = 'ROLL OUT'
        if land_track_active:
            self.array[land_track_active.array == 'Activated'] = 'LAND'
        if heading_mode_active:
            self.array[heading_mode_active.array == 'Activated'] = 'HDG'


class APVerticalMode(MultistateDerivedParameterNode):
    '''
    '''
    name = 'AP Vertical Mode'
    units = None
    # Values and states match X-Plane visualisation model documentation.
    values_mapping = {
        0: '-',
        2: 'SRS',
        4: 'CLB',
        6: 'DES',
        8: 'ALT CSTR CAPT',
        10: 'ALT CSTR',
        14: 'GS CAPT',
        16: 'GS',
        18: 'FINAL',
        22: 'FLARE',
        24: 'LAND',
        26: 'DES',  # geo path, A/THR mode SPEED
        64: 'OP CLB',
        66: 'OP DES',
        68: 'ALT CAPT',
        70: 'ALT',
        72: 'ALT CRZ',
        76: 'V/S',
        86: 'EXPED CLB',
        88: 'EXPED DES',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               at_active=M('AT Active'),
               climb_mode_active=M('Climb Mode Active'),
               longitudinal_mode_selected=M('Longitudinal Mode Selected'),
               ils_glideslope_capture_active=M('ILS Glideslope Capture Active'),
               ils_glideslope_active=M('ILS Glideslope Active'),
               flare_mode=M('Flare Mode'),
               open_climb_mode=M('Open Climb Mode'),
               open_descent_mode=M('Open Descent Mode'),
               altitude_capture_mode=M('Altitude Capture Mode'),
               altitude_mode=M('Altitude Mode'),
               expedite_climb_mode=M('Expedite Climb Mode'),
               expedite_descent_mode=M('Expedite Descent Mode'),
               vert_spd_engaged=M('Vertical Speed Engaged'),
               pitch_mode=M('Pitch Mode'),):
        parameter = next(p for p in (climb_mode_active,
                                     longitudinal_mode_selected,
                                     ils_glideslope_capture_active,
                                     ils_glideslope_active,
                                     flare_mode,
                                     at_active,
                                     open_climb_mode,
                                     open_descent_mode,
                                     altitude_capture_mode,
                                     altitude_mode,
                                     expedite_climb_mode,
                                     expedite_descent_mode,
                                     vert_spd_engaged,
                                     pitch_mode) if p)
        self.array = np_ma_zeros_like(parameter.array)

        if at_active:
            self.array[at_active.array == 'Activated'] = 'DES'
        if climb_mode_active:
            self.array[climb_mode_active.array == 'Activated'] = 'CLB'
        if vert_spd_engaged:
            self.array[vert_spd_engaged.array == 'Engaged'] = 'V/S'
        if longitudinal_mode_selected:
            states = longitudinal_mode_selected.state.keys()
            if 'Climb Mode Active' in states:
                self.array[longitudinal_mode_selected.array == 'Climb Mode Active'] = 'CLB'
            if 'Descent Mode' in states:
                self.array[longitudinal_mode_selected.array == 'Descent Mode'] = 'DES'
            if 'Altitude' in states:
                self.array[longitudinal_mode_selected.array == 'Altitude'] = 'ALT CSTR'
            if 'Final Descent Mode' in states:
                self.array[longitudinal_mode_selected.array == 'Final Descent Mode'] = 'FINAL'
            if 'Flare Mode' in states:
                self.array[longitudinal_mode_selected.array == 'Flare Mode'] = 'FLARE'
            if 'Land Track Active' in states:
                self.array[longitudinal_mode_selected.array == 'Land Track Active'] = 'LAND'
            if 'Vertical Speed Engaged' in states:
                self.array[longitudinal_mode_selected.array == 'Vertical Speed Engaged'] = 'V/S'
        if ils_glideslope_capture_active:
            self.array[ils_glideslope_capture_active.array == 'Activated'] = 'GS CAPT'
        if ils_glideslope_active:
            self.array[ils_glideslope_active.array == 'Activated'] = 'GS'
        if flare_mode:
            self.array[flare_mode.array == 'Engaged'] = 'FLARE'
        if open_climb_mode:
            self.array[open_climb_mode.array == 'Activated'] = 'OP CLB'
        if open_descent_mode:
            self.array[open_descent_mode.array == 'Activated'] = 'OP DES'
        if altitude_capture_mode:
            self.array[altitude_capture_mode.array == 'Activated'] = 'ALT CAPT'
        if altitude_mode:
            self.array[altitude_mode.array == 'Activated'] = 'ALT'
        if expedite_climb_mode:
            self.array[expedite_climb_mode.array == 'Activated'] = 'EXPED CLB'
        if expedite_descent_mode:
            self.array[expedite_descent_mode.array == 'Activated'] = 'EXPED DES'
        if pitch_mode:
            states = pitch_mode.state.keys()
            if 'FLARE' in states:
                self.array[pitch_mode.array == 'FLARE'] = 'FLARE'
            if 'FINAL DES' in states:
                self.array[pitch_mode.array == 'FINAL DES'] = 'FINAL'
            if 'V/S' in states:
                self.array[pitch_mode.array == 'V/S'] = 'V/S'
            if 'DES' in states:
                self.array[pitch_mode.array == 'DES'] = 'DES'
            if 'CLB' in states:
                self.array[pitch_mode.array == 'CLB'] = 'CLB'
            if 'ALT' in states:
                self.array[pitch_mode.array == 'ALT'] = 'ALT'
            if 'ALT*' in states:
                self.array[pitch_mode.array == 'ALT*'] = 'ALT CAPT'
            if 'G/S' in states:
                self.array[pitch_mode.array == 'G/S'] = 'GS'
            if 'G/S*' in states:
                self.array[pitch_mode.array == 'G/S*'] = 'GS CAPT'


class APUOn(MultistateDerivedParameterNode):
    '''
    Combine APU (1) On and APU (2) On parameters.
    '''

    name = 'APU On'
    units = None
    values_mapping = {0: '-', 1: 'On'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self, apu_1=M('APU (1) On'), apu_2=M('APU (2) On')):
        self.array = vstack_params_where_state(
            (apu_1, 'On'),
            (apu_2, 'On'),
        ).any(axis=0)


class APURunning(MultistateDerivedParameterNode):
    '''
    Simple measure of APU status, suitable for plotting if you want an on/off
    measure. Used for fuel usage measurements.
    '''

    name = 'APU Running'
    units = None
    values_mapping = {0: '-', 1: 'Running'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self, apu_n1=P('APU N1'),
               apu_voltage=P('APU Generator AC Voltage'),
               apu_bleed_valve_open=M('APU Bleed Valve Open'),
               apu_fuel_flow=P('APU Fuel Flow'),
               apu_on=M('APU On')):
        if apu_n1:
            self.array = np.ma.where(apu_n1.array > 50.0, 'Running', '-')
        elif apu_voltage:
            # XXX: APU Generator AC Voltage > 100 volts.
            self.array = np.ma.where(apu_voltage.array > 100.0, 'Running', '-')
        elif apu_bleed_valve_open:
            self.array = apu_bleed_valve_open.array == 'Open'
        elif apu_fuel_flow:
            #Added for LFL A330-D1033RR04CCX002
            fuel_flow_threshold = max(apu_fuel_flow.array.data) / 3.0
            self.array = np.ma.where(apu_fuel_flow.array > fuel_flow_threshold, 'Running', '-')
        elif apu_on:
            self.array = np.ma.where(apu_on.array == 1, 'Running', '-')


class CargoSmokeOrFire(MultistateDerivedParameterNode):
    '''
    Merges all the cargo smoke and fire signals into one.
    '''
    name = 'Cargo (*) Smoke Or Fire'
    units = None
    values_mapping = {0: '-', 1: 'Warning'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               s_cargo_warn=P('Smoke Cargo Warning'),
               s_cargo1_warn=P('Smoke Cargo (1) Warning'),
               s_cargo2_warn=P('Smoke Cargo (2) Warning'),
               s_cargo_aft_warn=P('Smoke Cargo Aft Warning'),
               s_cargo_aft1_warn=P('Smoke Cargo Aft (1) Warning'),
               s_cargo_aft2_warn=P('Smoke Cargo Aft (2) Warning'),
               s_cargo_fwd_warn=P('Smoke Cargo Fwd Warning'),
               s_cargo_fwd1_warn=P('Smoke Cargo Fwd (1) Warning'),
               s_cargo_fwd2_warn=P('Smoke Cargo Fwd (2) Warning'),
               s_cargo_fwd3_warn=P('Smoke Cargo Fwd (3) Warning'),
               s_cargo_loweraft_warn=P('Smoke Cargo Lower Aft Warning'),
               s_cargo_loweraft1_warn=P('Smoke Cargo Lower Aft (1) Warning'),
               s_cargo_loweraft2_warn=P('Smoke Cargo Lower Aft (2) Warning'),
               s_cargo_loweraft3_warn=P('Smoke Cargo Lower Aft (3) Warning'),
               s_cargo_loweraft4_warn=P('Smoke Cargo Lower Aft (4) Warning'),
               s_cargo_lowerfwd_warn=P('Smoke Cargo Lower Fwd Warning'),
               s_cargo_lowerfwd1_warn=P('Smoke Cargo Lower Fwd (1) Warning'),
               s_cargo_lowerfwd2_warn=P('Smoke Cargo Lower Fwd (2) Warning'),
               s_cargo_lowerfwd3_warn=P('Smoke Cargo Lower Fwd (3) Warning'),
               s_cargo_lowerfwd4_warn=P('Smoke Cargo Lower Fwd (4) Warning'),
               s_cargo_rest_warn=P('Smoke Cargo Rest Warning'),
               s_cargo_rest1_warn=P('Smoke Cargo Rest (1) Warning'),
               s_cargo_rest2_warn=P('Smoke Cargo Rest (2) Warning'),
               s_cargo_upperaft1_warn=P('Smoke Cargo Upper Aft (1) Warning'),
               s_cargo_upperaft2_warn=P('Smoke Cargo Upper Aft (2) Warning'),
               s_cargo_upperaft3_warn=P('Smoke Cargo Upper Aft (3) Warning'),
               s_cargo_upperaft4_warn=P('Smoke Cargo Upper Aft (4) Warning'),
               s_cargo_upperfwd1_warn=P('Smoke Cargo Upper Fwd (1) Warning'),
               s_cargo_upperfwd2_warn=P('Smoke Cargo Upper Fwd (2) Warning'),
               s_cargo_upperfwd3_warn=P('Smoke Cargo Upper Fwd (3) Warning'),
               s_cargo_upperfwd4_warn=P('Smoke Cargo Upper Fwd (4) Warning'),

               f_cargo=P('Cargo Fire'),
               f_cargo_aft=P('Cargo Aft Fire'),
               f_cargo_aft1=P('Cargo Aft Fire (1)'),
               f_cargo_aft2=P('Cargo Aft Fire (2)'),
               f_cargo_aft3=P('Cargo Aft Fire (3)'),
               f_cargo_aft4=P('Cargo Aft Fire (4)'),
               f_cargo_fwd=P('Cargo Fwd Fire'),
               f_cargo_fwd1=P('Cargo Fwd Fire (1)'),
               f_cargo_fwd2=P('Cargo Fwd Fire (2)'),
               f_cargo_fwd3=P('Cargo Fwd Fire (3)'),
               f_cargo_fwd4=P('Cargo Fwd Fire (4)'),
               ):

        self.array = vstack_params(
            s_cargo_warn,           s_cargo1_warn,
            s_cargo2_warn,          s_cargo_aft_warn,
            s_cargo_aft1_warn,      s_cargo_aft2_warn,
            s_cargo_fwd_warn,       s_cargo_fwd1_warn,
            s_cargo_fwd2_warn,      s_cargo_fwd3_warn,
            s_cargo_loweraft_warn,  s_cargo_loweraft1_warn,
            s_cargo_loweraft2_warn, s_cargo_loweraft3_warn,
            s_cargo_loweraft4_warn, s_cargo_lowerfwd_warn,
            s_cargo_lowerfwd2_warn, s_cargo_lowerfwd3_warn,
            s_cargo_lowerfwd4_warn, s_cargo_rest_warn,
            s_cargo_rest1_warn,     s_cargo_rest2_warn,
            s_cargo_upperaft1_warn, s_cargo_upperaft2_warn,
            s_cargo_upperaft3_warn, s_cargo_upperaft4_warn,
            s_cargo_upperfwd1_warn, s_cargo_upperfwd2_warn,
            s_cargo_upperfwd3_warn, s_cargo_upperfwd4_warn,

            f_cargo,      f_cargo_aft,
            f_cargo_aft1, f_cargo_aft2,
            f_cargo_aft3, f_cargo_aft4,
            f_cargo_fwd,  f_cargo_fwd1,
            f_cargo_fwd2, f_cargo_fwd3,
            f_cargo_fwd4
        ).any(axis=0)


class Configuration(MultistateDerivedParameterNode):
    '''
    Parameter for aircraft that use configuration. Reflects the actual state
    of the aircraft. See "Flap Lever" or "Flap Lever (Synthetic)" which show
    the physical lever detents selectable by the crew.

    Multi-state with the following mapping::
        %s

    Some values are based on footnotes in various pieces of documentation:
    - 2(a) corresponds to CONF 1*
    - 3(b) corresponds to CONF 2*

    Note: Does not use the Flap Lever position. This parameter reflects the
    actual configuration state of the aircraft rather than the intended state
    represented by the selected lever position.
    ''' % pformat(at.constants.AVAILABLE_CONF_STATES)
    align_frequency = 2
    units = None

    @classmethod
    def can_operate(cls, available, model=A('Model'), series=A('Series'), family=A('Family'),
                    manufacturer=A('Manufacturer'),):

        if manufacturer and manufacturer.value not in ('Airbus', 'Embraer'):
            return False

        if family and family.value in ('A300', 'A310',):
            return False

        if not all_of(
            ('Slat Including Transition', 'Flap Including Transition', 'Manufacturer', 'Model', 'Series', 'Family'),
            available,
        ):
            return False

        if manufacturer and manufacturer.value == 'Embraer' and not all_of(('Approach And Landing', 'Taxi In'), available):
            return False

        try:
            at.get_conf_angles(model.value, series.value, family.value, manufacturer.value)
        except KeyError:
            cls.warning("No conf angles available for '%s', '%s', '%s'.",
                        model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap=M('Flap Including Transition'), slat=M('Slat Including Transition'),
               model=A('Model'), series=A('Series'), family=A('Family'), manufacturer=A('Manufacturer'),
               approach_landing=S('Approach And Landing'), taxi_in=S('Taxi In')):

        angles = at.get_conf_angles(model.value, series.value, family.value, manufacturer.value)
        values_mapping = at.constants.AVAILABLE_CONF_STATES[manufacturer.value]

        # initialize an empty masked array the same length as flap array
        self.array = MappedArray(np_ma_masked_zeros_like(flap.array, dtype=np.short),
                                 values_mapping=values_mapping)

        for (state, (s, f, a)) in six.iteritems(angles):
            condition = (flap.array == f)
            if s is not None:
                condition &= (slat.array == s)
            if manufacturer.value == 'Embraer' and state == '5':
                # Flap 4 and 5 are identical. 5 used for approach/landing. 4 used for takeoff/go-around.
                # Taxi-in included for retraction after landing. Extended to remove any gap between slices.
                taxi_in = slices_extend([x.slice for x in taxi_in], 5)
                approach_landing = [x.slice for x in approach_landing]
                for section in slices_overlap_merge(approach_landing, taxi_in):
                    self.array[section][condition[section]] = state
            else:
                self.array[condition] = state

        nearest_neighbour_mask_repair(self.array, copy=False,
                                      repair_gap_size=(30 * self.hz),
                                      direction='backward')


class ConfigurationExcludingTransition(MultistateDerivedParameterNode):
    '''
    Parameter for aircraft that use configuration. Reflects the actual state
    of the aircraft. See "Flap Lever" or "Flap Lever (Synthetic)" which show
    the physical lever detents selectable by the crew.

    Multi-state with the following mapping::
        %s

    Some values are based on footnotes in various pieces of documentation:
    - 2(a) corresponds to CONF 1*
    - 3(b) corresponds to CONF 2*

    Note: Does not use the Flap Lever position. This parameter reflects the
    actual configuration state of the aircraft rather than the intended state
    represented by the selected lever position.
    ''' % pformat(at.constants.AVAILABLE_CONF_STATES)
    align_frequency = 2
    units = None

    @classmethod
    def can_operate(cls, available, model=A('Model'), series=A('Series'), family=A('Family'),
                    manufacturer=A('Manufacturer'),):

        if manufacturer and not manufacturer.value == 'Airbus':
            return False

        if family and family.value in ('A300', 'A310',):
            return False

        if not all_of((
            'Slat Excluding Transition', 'Flap Excluding Transition', 'Manufacturer', 'Model', 'Series', 'Family'
        ), available):
            return False

        try:
            at.get_conf_angles(model.value, series.value, family.value)
        except KeyError:
            cls.warning("No conf angles available for '%s', '%s', '%s'.",
                        model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap=M('Flap Excluding Transition'), slat=M('Slat Excluding Transition'),
               model=A('Model'), series=A('Series'), family=A('Family'), manufacturer=A('Manufacturer'),


               conf=M('Configuration')):

        angles = at.get_conf_angles(model.value, series.value, family.value)
        values_mapping = at.constants.AVAILABLE_CONF_STATES[manufacturer.value]

        # initialize an empty masked array the same length as flap array
        self.array = MappedArray(np_ma_masked_zeros_like(flap.array, dtype=np.short),
                                 values_mapping=values_mapping)

        for (state, (s, f, a)) in six.iteritems(angles):
            condition = (flap.array == f)
            if s is not None:
                condition &= (slat.array == s)

            self.array[condition] = state

        nearest_neighbour_mask_repair(self.array, copy=False,
                                      repair_gap_size=(30 * self.hz),
                                      direction='backward')


class Daylight(MultistateDerivedParameterNode):
    '''
    Calculate Day or Night based upon Civil Twilight.

    FAA Regulation FAR 1.1 defines night as: "Night means the time between
    the end of evening civil twilight and the beginning of morning civil
    twilight, as published in the American Air Almanac, converted to local
    time.

    EASA EU OPS 1 Annex 1 item (76) states: 'night' means the period between
    the end of evening civil twilight and the beginning of morning civil
    twilight or such other period between sunset and sunrise as may be
    prescribed by the appropriate authority, as defined by the Member State;

    CAA regulations confusingly define night as 30 minutes either side of
    sunset and sunrise, then include a civil twilight table in the AIP.

    With these references, it was decided to make civil twilight the default.
    '''
    align = True
    # 1/4 is the minimum allowable frequency due to minimum data boundary
    # of 4 seconds.
    align_frequency = 1 / 4.0
    align_offset = 0.0
    units = None

    values_mapping = {
        0: 'Night',
        1: 'Day'
    }

    def derive(self,
               latitude=P('Latitude Smoothed'),
               longitude=P('Longitude Smoothed'),
               start_datetime=A('Start Datetime'),
               duration=A('HDF Duration')):
        # Set default to 'Day'
        array_len = int(duration.value * self.frequency)
        self.array = np.ma.ones(array_len)
        for step in range(array_len):
            curr_dt = datetime_of_index(start_datetime.value, step, 1)
            lat = latitude.array[step]
            lon = longitude.array[step]
            if lat and lon:
                if not dateext.is_day(curr_dt, lat, lon):
                    # Replace values with Night
                    self.array[step] = 0
                else:
                    continue  # leave array as 1
            else:
                # either is masked or recording 0.0 which is invalid too
                self.array[step] = np.ma.masked


class DualInput(MultistateDerivedParameterNode):
    '''
    Determines whether input by both of the pilots has occurred.

    This parameter uses the 'Pilot Flying' derived multi-state parameter to
    determine who is considered to be the pilot flying the aircraft and then
    inspects whether the other pilot has made any significant sustained input.

    For Airbus aircraft, this requires us to check the angle of the sidestick.

    Note that the AFPS defines DUAL_INPUT as 0.5 degree deflection for more
    than 3 seconds. However, the A330/A340 have poor resolution so this
    threshold was increased to 1.7 degrees. SmartCockpit Flight Controls
    Sidestick priority logic declares a 2.0 degree deflection will trigger
    "SIDE STICK PRIORITY" lights on the glareshield and the "DUAL INPUT" voice
    message is activated. Therefore, the threshold used here is 2.0 degrees for
    3 seconds so all maximum sidestick angle KPVs measured during Dual Input
    will have a minimum of 2.0 degrees.

    This is not strictly speaking a warning as we have no record that anything
    was triggered in the cockpit.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''
    units = None
    values_mapping = {0: '-', 1: 'Dual'}

    def derive(self,
               pilot=M('Pilot Flying'),
               stick_capt=P('Sidestick Angle (Capt)'),
               stick_fo=P('Sidestick Angle (FO)'),
               family=A('Family'),
               ):

        array = np_ma_zeros_like(pilot.array)
        array[pilot.array == 'Captain'] = stick_fo.array[pilot.array == 'Captain']
        array[pilot.array == 'First Officer'] = stick_capt.array[pilot.array == 'First Officer']

        array = np.ma.array(array > 2.0, mask=array.mask, dtype=int)

        '''
        Here we're removing small slices first and then removing gaps, not
        the other way around as it would cause multiple short sidestick dual input
        'nudges' to be interpreted as continuous dual input, which is incorrect, as
        Airbus AFPS specifies that in order for the warning to come on dual input
        has to be >0.5deg (we're using 2.0deg, see class docstring for details) for
        at least 3 seconds.

        Also using 2.9s to remove small slices instead of 3.0s as using 3.0s causes
        valid dual input events that are exactly 3 seconds (i.e 12 samples where angle
        is over 2 degrees at 4Hz) to be missed. Can't be 2.75 as in case the parameter
        is recorded at 8Hz it would have to be at least 2.8, this threshold makes
        it to be 'safe' for up to 10Hz.
        '''
        slices = runs_of_ones(array)
        slices = slices_remove_small_slices(slices, 2.9, self.hz)
        slices = slices_remove_small_gaps(slices, 15, self.hz)

        dual = np_ma_zeros_like(array, dtype=np.short)
        for sl in slices:
            dual[sl] = 1
        self.array = dual


class Eng_1_Fire(MultistateDerivedParameterNode):
    '''
    Combine on ground and in air fire warnings.
    '''

    name = 'Eng (1) Fire'
    units = None
    values_mapping = {0: '-', 1: 'Fire'}

    def derive(self,
               fire_gnd=M('Eng (1) Fire On Ground'),
               fire_air=M('Eng (1) Fire In Air')):

        self.array = vstack_params_where_state(
            (fire_gnd, 'Fire'),
            (fire_air, 'Fire'),
        ).any(axis=0)


class Eng_2_Fire(MultistateDerivedParameterNode):
    '''
    Combine on ground and in air fire warnings.
    '''

    name = 'Eng (2) Fire'
    units = None
    values_mapping = {0: '-', 1: 'Fire'}

    def derive(self,
               fire_gnd=M('Eng (2) Fire On Ground'),
               fire_air=M('Eng (2) Fire In Air')):

        self.array = vstack_params_where_state(
            (fire_gnd, 'Fire'),
            (fire_air, 'Fire'),
        ).any(axis=0)


class Eng_3_Fire(MultistateDerivedParameterNode):
    '''
    Combine on ground and in air fire warnings.
    '''

    name = 'Eng (3) Fire'
    units = None
    values_mapping = {0: '-', 1: 'Fire'}

    def derive(self,
               fire_gnd=M('Eng (3) Fire On Ground'),
               fire_air=M('Eng (3) Fire In Air')):

        self.array = vstack_params_where_state(
            (fire_gnd, 'Fire'),
            (fire_air, 'Fire'),
        ).any(axis=0)


class Eng_4_Fire(MultistateDerivedParameterNode):
    '''
    Combine on ground and in air fire warnings.
    '''

    name = 'Eng (4) Fire'
    units = None
    values_mapping = {0: '-', 1: 'Fire'}

    def derive(self,
               fire_gnd=M('Eng (4) Fire On Ground'),
               fire_air=M('Eng (4) Fire In Air')):

        self.array = vstack_params_where_state(
            (fire_gnd, 'Fire'),
            (fire_air, 'Fire'),
        ).any(axis=0)


class Eng_Fire(MultistateDerivedParameterNode):
    '''
    Merges all the engine fire signals into one.
    '''
    name = 'Eng (*) Fire'
    units = None
    values_mapping = {0: '-', 1: 'Fire'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=M('Eng (1) Fire'),
               eng2=M('Eng (2) Fire'),
               eng3=M('Eng (3) Fire'),
               eng4=M('Eng (4) Fire'),
               eng1_1l=M('Eng (1) Fire (1L)'),
               eng1_1r=M('Eng (1) Fire (1R)'),
               eng1_2l=M('Eng (1) Fire (2L)'),
               eng1_2r=M('Eng (1) Fire (2R)'),
               ):

        self.array = vstack_params_where_state(
            (eng1, 'Fire'), (eng2, 'Fire'),
            (eng3, 'Fire'), (eng4, 'Fire'),
            (eng1_1l, 'Fire'), (eng1_1r, 'Fire'),
            (eng1_2l, 'Fire'), (eng1_2r, 'Fire'),
        ).any(axis=0)


class Eng_Oil_Press_Warning(MultistateDerivedParameterNode):
    '''
    Combine all oil pressure (low) warning indications.
    '''

    name = 'Eng (*) Oil Press Warning'
    units = None
    values_mapping = {0: '-', 1: 'Warning'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng1=P('Eng (1) Oil Press Low'),
               eng2=P('Eng (2) Oil Press Low'),
               eng3=P('Eng (3) Oil Press Low'),
               eng4=P('Eng (4) Oil Press Low'),
               ):

        self.array = vstack_params_where_state(
            (eng1, 'Low Press'),
            (eng2, 'Low Press'),
            (eng3, 'Low Press'),
            (eng4, 'Low Press'),
        ).any(axis=0)


class EngBleedOpen(MultistateDerivedParameterNode):
    '''
    Single multistate for any engine bleed valve open.
    '''

    units = None

    name = 'Eng Bleed Open'
    values_mapping = {0: 'Closed', 1: 'Open'}

    @classmethod
    def can_operate(cls, available):
        return all_of((
            'Eng (1) Bleed',
            'Eng (2) Bleed',
        ), available)

    def derive(self,
               b1=M('Eng (1) Bleed'),
               b2=M('Eng (2) Bleed'),
               b3=M('Eng (3) Bleed'),
               b4=M('Eng (4) Bleed')):

        self.array = vstack_params_where_state((b1, 'Open'),
                                           (b2, 'Open'),
                                           (b3, 'Open'),
                                           (b4, 'Open')).any(axis=0)


class EngRunning(object):
    '''
    Abstract class for inheriting by EngRunning derived parameters.
    '''
    engnum = 0  # Replace with '2' for Eng (2)
    units = None
    values_mapping = {
        0: 'Not Running',
        1: 'Running',
    }
    # Workaround for NotImplementedError: Unknown Type raised from
    # process_flight
    node_type = MultistateDerivedParameterNode

    @classmethod
    def can_operate(cls, available):
        return 'Eng (%d) N1' % cls.engnum in available or \
               'Eng (%d) N2' % cls.engnum in available or \
               'Eng (%d) Np' % cls.engnum in available or \
               'Eng (%d) Fuel Flow' % cls.engnum in available

    def determine_running(self, eng_n1, eng_n2, eng_np, fuel_flow, ac_type):
        '''
        TODO: Include Fuel cut-off switch if recorded?
        TODO: Confirm that all engines were recording for the N2 Min / Fuel Flow
        Min parameters - theoretically there could be only three engines in the
        frame for a four engine aircraft. Use "Engine Count".
        '''
        if eng_np:
            # If it's got propellors, this overrides core engine measurements.
            return np.ma.where(eng_np.array > MIN_FAN_RUNNING, 'Running', 'Not Running')
        elif eng_n2 or fuel_flow and ac_type != helicopter:
            # Ideally have N2 and Fuel Flow with both available,
            # otherwise use just one source
            n2_running = eng_n2.array > MIN_CORE_RUNNING if eng_n2 \
                else np.zeros_like(fuel_flow.array, dtype=bool)
            fuel_flowing = fuel_flow.array > MIN_FUEL_FLOW_RUNNING if fuel_flow \
                else np.zeros_like(eng_n2.array, dtype=bool)
            data = n2_running.data | fuel_flowing.data
            mask = n2_running.mask & fuel_flowing.mask
            return np.ma.where(np.ma.array(data, mask=mask), 'Running', 'Not Running')
        else:
            # Fall back on N1
            return np.ma.where(eng_n1.array > MIN_FAN_RUNNING, 'Running', 'Not Running')


class Eng1Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 1
    name = 'Eng (1) Running'
    units = None

    def derive(self,
               eng_n1=P('Eng (1) N1'),
               eng_n2=P('Eng (1) N2'),
               eng_np=P('Eng (1) Np'),
               fuel_flow=P('Eng (1) Fuel Flow'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Eng2Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 2
    name = 'Eng (2) Running'
    units = None

    def derive(self,
               eng_n1=P('Eng (2) N1'),
               eng_n2=P('Eng (2) N2'),
               eng_np=P('Eng (2) Np'),
               fuel_flow=P('Eng (2) Fuel Flow'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Eng3Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 3
    name = 'Eng (3) Running'
    units = None

    def derive(self,
               eng_n1=P('Eng (3) N1'),
               eng_n2=P('Eng (3) N2'),
               eng_np=P('Eng (3) Np'),
               fuel_flow=P('Eng (3) Fuel Flow'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Eng4Running(EngRunning, MultistateDerivedParameterNode):
    '''
    Discrete parameter describing when the engine is running.
    '''
    engnum = 4
    name = 'Eng (4) Running'
    units = None

    def derive(self,
               eng_n1=P('Eng (4) N1'),
               eng_n2=P('Eng (4) N2'),
               eng_np=P('Eng (4) Np'),
               fuel_flow=P('Eng (4) Fuel Flow'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Eng_AllRunning(MultistateDerivedParameterNode, EngRunning):
    '''
    Discrete parameter describing when all available engines are running.
    '''
    name = 'Eng (*) All Running'
    units = None

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return 'Eng (*) N1 Min' in available
        else:
            return 'Eng (*) N1 Min' in available or \
                   'Eng (*) N2 Min' in available or \
                   'Eng (*) Np Min' in available or \
                   'Eng (*) Fuel Flow Min' in available

    def derive(self,
               eng_n1=P('Eng (*) N1 Min'),
               eng_n2=P('Eng (*) N2 Min'),
               eng_np=P('Eng (*) Np Min'),
               fuel_flow=P('Eng (*) Fuel Flow Min'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class Eng_AnyRunning(MultistateDerivedParameterNode, EngRunning):
    '''
    Discrete parameter describing when any engines are running.

    This is useful with 'Eng (*) All Running' to detect if not all engines are
    running.
    '''
    name = 'Eng (*) Any Running'
    units = None

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return 'Eng (*) N1 Max' in available
        else:
            return 'Eng (*) N1 Max' in available or \
                   'Eng (*) N2 Max' in available or \
                   'Eng (*) Np Max' in available or \
                   'Eng (*) Fuel Flow Max' in available

    def derive(self,
               eng_n1=P('Eng (*) N1 Max'),
               eng_n2=P('Eng (*) N2 Max'),
               eng_np=P('Eng (*) Np Max'),
               fuel_flow=P('Eng (*) Fuel Flow Max'),
               ac_type=A('Aircraft Type')):
        self.array = self.determine_running(eng_n1, eng_n2, eng_np, fuel_flow, ac_type)


class ThrustModeSelected(MultistateDerivedParameterNode):
    '''
    Combines Thrust Mode Selected parameters.
    '''

    units = None
    values_mapping = {
        0: '-',
        1: 'Selected',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               thrust_l=P('Thrust Mode Selected (L)'),
               thrust_r=P('Thrust Mode Selected (R)')):

        thrusts = [thrust for thrust in [thrust_l,
                                         thrust_r] if thrust]

        if len(thrusts) == 1:
            self.array = thrusts[0].array

        array = MappedArray(np_ma_zeros_like(thrusts[0].array, dtype=np.short),
                            values_mapping=self.values_mapping)

        masks = []
        for thrust in thrusts:
            masks.append(thrust.array.mask)
            array[thrust.array == 'Selected'] = 'Selected'

        array.mask = merge_masks(masks)
        self.array = array


class EventMarker(MultistateDerivedParameterNode):
    '''
    Combine Event Marker from multiple sources where recorded separately.
    '''
    values_mapping = {0: '-', 1: 'Event'}
    name = 'Event Marker'
    units = None

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               event_marker_1=M('Event Marker (1)'),
               event_marker_2=M('Event Marker (2)'),
               event_marker_3=M('Event Marker (3)'),
               event_marker_capt=M('Event Marker (Capt)'),
               event_marker_fo=M('Event Marker (FO)')):

        self.array = vstack_params_where_state(
            (event_marker_1, 'Event'),
            (event_marker_2, 'Event'),
            (event_marker_3, 'Event'),
            (event_marker_capt, 'Event'),
            (event_marker_fo, 'Event'),
        ).any(axis=0)


class Flap(MultistateDerivedParameterNode):
    '''
    Steps raw Flap angle from surface into detents.
    '''

    units = ut.DEGREE
    # Currently uses the frequency of the Flap Angle parameter - might
    # consider upsampling to 2Hz for the Kernal sizes in the calculate_flap
    # function
    ##align_frequency = 2

    @classmethod
    def can_operate(cls, available, frame=A('Frame'),
                    model=A('Model'), series=A('Series'), family=A('Family')):

        frame_name = frame.value if frame else None
        family_name = family.value if family else None

        if frame_name == 'L382-Hercules' or family_name == 'C208':
            return 'Altitude AAL' in available

        if family_name == 'Citation VLJ':
            return all_of(('HDF Duration', 'Landing', 'Takeoff'), available)

        if not all_of(('Flap Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_flap_map(model.value, series.value, family.value)
        except KeyError:
            #Q: Everyone should have a flap map - so raise error?
            cls.exception("No flap mapping available for '%s', '%s', '%s'.",
                          model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap=P('Flap Angle'),
               model=A('Model'), series=A('Series'), family=A('Family'),
               frame=A('Frame'), alt_aal=P('Altitude AAL'),
               hdf_duration=A('HDF Duration'),
               toffs=S('Takeoff'), lands=S('Landing')):

        frame_name = frame.value if frame else None
        family_name = family.value if family else None
        duration = hdf_duration.value if hdf_duration else None

        if frame_name == 'L382-Hercules':
            self.values_mapping = {0: '0', 50: '50', 100: '100'}

            self.units = ut.PERCENT  # Hercules flaps are unique in this regard!

            # Flap is not recorded, so invent one of the correct length.
            flap_herc = np_ma_zeros_like(alt_aal.array)

            # Takeoff is normally with 50% flap382
            _, toffs = slices_from_to(alt_aal.array, 0.0, 1000.0)
            flap_herc[:toffs[0].stop] = 50.0

            # Assume 50% from 2000 to 1000ft, and 100% thereafter on the approach.
            _, apps = slices_from_to(alt_aal.array, 2000.0, 0.0)
            flap_herc[apps[-1].start:] = np.ma.where(alt_aal.array[apps[-1].start:] > 1000.0, 50.0, 100.0)

            self.array = np.ma.array(flap_herc)
            self.frequency, self.offset = alt_aal.frequency, alt_aal.offset
            return

        if family_name == 'Citation VLJ' and duration:
            self.values_mapping = {0: '0', 15: '15', 30: '30'}
            self.array = np.ma.zeros(int(duration * self.frequency))
            for toff in toffs:
                self.array[toff.slice] = 15
            for land in lands:
                self.array[land.slice] = 30
            return

        if family_name == 'C208':
            self.values_mapping = {0: '0', 10: '10', 20: '20', 40: '40'}
            self.array = np_ma_zeros_like(alt_aal.array)
            self.array[alt_aal.array < 500.0] = 40
            self.frequency, self.offset = alt_aal.frequency, alt_aal.offset
            return

        if 'B737' in family_name:
            _slices = runs_of_ones(np.logical_and(flap.array>=0.9, flap.array<=2.1))
            for s in _slices:
                flap.array[s] = smooth_signal(flap.array[s], window_len=5, window='flat')
        self.values_mapping, self.array, self.frequency, self.offset = calculate_flap(
            'lever', flap, model, series, family)


class FlapLever(MultistateDerivedParameterNode):
    '''
    Rounds the Flap Lever Angle to the selected detent at the start of the
    angle movement.

    Flap is not used to synthesize Flap Lever as this could be misleading.
    Instead, all safety Key Point Values will use Flap Lever followed by Flap
    if Flap Lever is not available.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Flap Lever Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_lever_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No lever mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap_lever=P('Flap Lever Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        #self.values_mapping = at.get_lever_map(model.value, series.value, family.value)
        #self.array, self.frequency, self.offset = calculate_surface_angle(
            #'lever',
            #flap_lever,
            #self.values_mapping.keys(),
        #)
        self.values_mapping = at.get_lever_map(model.value, series.value, family.value)
        self.array = step_values(repair_mask(flap_lever.array),
                                 self.values_mapping.keys(),
                                 flap_lever.hz, step_at='move_start')



class FlapIncludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the higher of the start and endpoints of the movement
    apply. This increases the chance of needing a flap overspeed inspection,
    but provides a more cautious interpretation of the maintenance
    requirements.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Flap Angle', 'Model', 'Series', 'Family'), available):
            return all_of(('Flap', 'Model', 'Series', 'Family'), available)

        try:
            at.get_flap_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No lever mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap_angle=P('Flap Angle'), flap=M('Flap'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        self.values_mapping = at.get_flap_map(model.value, series.value, family.value)
        if flap_angle:
            self.array = including_transition(flap_angle.array, self.values_mapping, hz=self.hz)
        else:
            # if we do not have flap angle use flap, use states as values
            # will vary between frames
            array = MappedArray(np_ma_masked_zeros_like(flap.array),
                                values_mapping=self.values_mapping)
            for value, state in six.iteritems(self.values_mapping):
                array[flap.array == state] = state
            self.array = array


class FlapExcludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the higher of the start and endpoints of the movement
    apply. This increases the chance of needing a flap overspeed inspection,
    but provides a more cautious interpretation of the maintenance
    requirements.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Flap Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_flap_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No lever mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, flap_angle=P('Flap Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        family_name = family.value if family else None
        if "B737" in family_name:
            _slices = runs_of_ones(np.logical_and(flap_angle.array>=0.9, flap_angle.array<=2.1))
            for s in _slices:
                flap_angle.array[s] = smooth_signal(flap_angle.array[s], window_len=5, window='flat')
        self.values_mapping, self.array, self.frequency, self.offset = calculate_flap(
            'excluding', flap_angle, model, series, family)


class FlapLeverSynthetic(MultistateDerivedParameterNode):
    '''
    Create a synthetic representation of the Flap Lever position.
    '''

    name = 'Flap Lever (Synthetic)'
    units = ut.DEGREE
    align_frequency = 2  # force higher than most Flap frequencies

    @classmethod
    def can_operate(cls, available, manufacturer=A('Manufacturer'),
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Flap', 'Model', 'Series', 'Family', 'Manufacturer'), available):
            return False

        try:
            angles = at.get_conf_angles(model.value, series.value, family.value)
        except KeyError:
            # try lever map if no conf
            try:
                angles = at.get_lever_angles(model.value, series.value, family.value)
            except KeyError:
                cls.warning("No lever angles available for '%s', '%s', '%s'.",
                            model.value, series.value, family.value)
                return False

        can_operate = True

        slat_required = any(slat is not None for slat, flap, flaperon in
                            angles.values())
        if slat_required:
            can_operate = can_operate and 'Slat' in available

        flaperon_required = any(flaperon is not None for slat, flap, flaperon in
                                angles.values())
        if flaperon_required:
            can_operate = can_operate and 'Flaperon' in available

        return can_operate

    def derive(self, flap=M('Flap'), slat=M('Slat'), flaperon=M('Flaperon'), manufacturer=A('Manufacturer'),
               model=A('Model'), series=A('Series'), family=A('Family'),
               approach=S('Approach And Landing'), frame=A('Frame'),):
        try:
            if manufacturer.value != 'Airbus':
                raise KeyError
            angles = at.get_conf_angles(model.value, series.value, family.value)
            use_conf = True
        except KeyError:
            angles = at.get_lever_angles(model.value, series.value, family.value)
            use_conf = False

        # Get the values mapping, airbus requires some hacking:
        if use_conf:
            self.values_mapping = at.constants.LEVER_STATES
        else:
            self.values_mapping = at.get_lever_map(model.value, series.value, family.value)

        # Prepare the destination array:
        self.array = MappedArray(np_ma_masked_zeros_like(flap.array),
                                 values_mapping=self.values_mapping)

        # Update the destination array according to the mappings:
        for (state, (s, f, a)) in six.iteritems(angles):
            condition = (flap.array == str(f))
            if s is not None:
                condition &= (slat.array == str(s))
            if a is not None:
                condition &= (flaperon.array == str(a))
            if use_conf:
                state = at.constants.CONF_TO_LEVER[state]
            self.array[condition] = state

        frame_name = frame.value if frame else None
        approach_slices = slices_int(approach.get_slices()) if approach else None

        if frame_name == 'E170_EBD_047' and approach_slices is not None:
            # The Lever 4 and 5 share the same flap/slat config.
            # On approaches the config is refferred to as Lever 5
            self.array[self.array == 32] = 16  # ensure lever 4 before approach mod
            for approach_slice in approach_slices:
                self.array[approach_slice][self.array[approach_slice] == 16] = 32


class Flaperon(MultistateDerivedParameterNode):
    '''
    Where Ailerons move together and used as Flaps, these are known as
    "Flaperon" control.

    Flaperons are measured where both Left and Right Ailerons move down,
    which on the left creates possitive roll but on the right causes negative
    roll. The difference of the two signals is the Flaperon control.

    The Flaperon is stepped at the start of movement into the nearest aileron
    detents, e.g. 0, 5, 10 deg

    Note: This is used for Airbus models and does not necessarily mean as
    much to other aircraft types.
    '''
    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Aileron (L)', 'Aileron (R)', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_aileron_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No aileron/flaperon mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, al=P('Aileron (L)'), ar=P('Aileron (R)'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        # Take the difference of the two signals (which should cancel each
        # other out when rolling) and divide the range by two (to account for
        # the left going negative and right going positive when flaperons set)

        flaperon_angle = (al.array - ar.array) / 2
        self.values_mapping = at.get_aileron_map(model.value, series.value, family.value)
        snapped = step_values(flaperon_angle, self.values_mapping.keys(), al.hz, step_at='move_start')
        snap_step_at = np.nonzero(np.ediff1d(snapped))[0]
        # Move through the data with the front foot at each change point. If the stride is small enough,
        # make the change at the beginning of this stride, and keep going until it's too big a stride in
        # which case the back foot needs to move up.
        back = 0
        front = 1
        while front < len(snap_step_at) - 1:
            if (snap_step_at[front] - snap_step_at[back]) < 10:
                snapped[snap_step_at[back] + 1 : snap_step_at[front] +1] = snapped[snap_step_at[front] + 1]
            else:
                back = front
            front += 1

        self.array = snapped


class FuelQty_Low(MultistateDerivedParameterNode):
    '''
    '''
    name = "Fuel Qty (*) Low"
    units = None
    values_mapping = {
        0: '-',
        1: 'Warning',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self, fqty=M('Fuel Qty Low'),
               fqty1=M('Fuel Qty (L) Low'),
               fqty2=M('Fuel Qty (R) Low')):
        warning = vstack_params_where_state(
            (fqty, 'Warning'),
            (fqty1, 'Warning'),
            (fqty2, 'Warning'),
        )
        self.array = warning.any(axis=0)


class GearDown(MultistateDerivedParameterNode):
    '''
    This Multi-State parameter uses "majority voting" to decide whether the
    gear is up or down.

    If Gear (*) Down is not recorded, it will be created from Gear Down
    Selected which is from the cockpit lever.
    '''

    align = False
    units = None
    values_mapping = {
        0: 'Up',
        1: 'Down',
    }

    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        gear_lever = all_of(('Gear Down Selected', 'Gear Down In Transit'), available)
        return 'Gear Position' in available or gear_lever

    def derive(self,
               gear_transit=M('Gear Down In Transit'),
               gear_sel=M('Gear Down Selected'),
               gear_pos=M('Gear Position')):
        # Join all available gear parameters and use whichever are available.
        if gear_sel and gear_transit:
            gear_sel_array = align(gear_sel, gear_transit) if gear_sel.hz != gear_transit.hz else gear_sel.array
            self.array = (gear_sel_array == 'Down') & ~(gear_transit.array == 'Extending')
            self.frequency = gear_transit.frequency
            self.offset = gear_transit.offset
        else:
            self.array = gear_pos.array == 'Down'
            self.frequency = gear_pos.frequency
            self.offset = gear_pos.offset


class GearDownInTransit(MultistateDerivedParameterNode):
    units = None
    values_mapping = {
        0: '-',
        1: 'Extending',
    }

    @classmethod
    def can_operate(cls, available, model=A('Model'), series=A('Series'), family=A('Family')):
        # Can operate with a any combination of parameters available
        gears_available = all_of(('Gear Down', 'Gear Down Selected'), available) \
            or all_of(('Gear Up', 'Gear Down'), available) \
            or all_of(('Gear Down Selected', 'Gear In Transit'), available) \
            or all_of(('Gear Up', 'Gear In Transit'), available) \
            or all_of(('Gear Down', 'Gear In Transit'), available) \
            or all_of(('Gear Down Selected', 'Gear (*) Red Warning'), available) \
            or all_of(('Gear Up', 'Gear (*) Red Warning'), available) \
            or all_of(('Gear Down', 'Gear (*) Red Warning'), available) \
            or 'Gear Position' in available

        if gears_available:
            return True

        if all_of(('Model', 'Series', 'Family', 'Airborne'), available) \
           and any_of(('Gear Down Selected', 'Gear Up', 'Gear Down'), available):
            # Check dependancies before checking if transition times are available to save necessary lookups
            if model and series and family:
                try:
                    at.get_gear_transition_times(model.value, series.value, family.value)
                except KeyError:
                    cls.exception("No gear transition times available for '%s', '%s', '%s'.",
                                  model.value, series.value, family.value)
                    return False
                return True
            else:
                return False
        else:
            return False

    def derive(self,
               gear_down=M('Gear Down'),
               gear_up=M('Gear Up'),
               gear_down_sel=M('Gear Down Selected'),
               gear_in_transit=M('Gear In Transit'),
               gear_red=M('Gear (*) Red Warning'),
               gear_position=M('Gear Position'),
               airborne=S('Airborne'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        # Extend Airborne slices to make sure we don't miss any Gear parameters at liftoff
        airborne = slices_extend(airborne.get_slices(), 1)
        gear_sels = gear_ups = gear_downs = runs = []
        self.array = np_ma_zeros_like(first_valid_parameter(gear_down, gear_up, gear_down_sel, gear_position).array, dtype=int)

        # work out fallback
        extend_duration = 0
        if model and series and family:
            try:
                _, extend_duration = at.get_gear_transition_times(model.value, series.value, family.value)
            except KeyError:
                pass
        fallback = extend_duration * self.frequency

        # find stating points.
        if gear_down:
            gear_downs = find_edges_on_state_change('Down', nearest_neighbour_mask_repair(gear_down.array), phase=airborne)
        if gear_up:
            gear_ups = find_edges_on_state_change('Up', nearest_neighbour_mask_repair(gear_up.array), change='leaving', phase=airborne)
        if gear_down_sel:
            gear_sels = find_edges_on_state_change('Down', nearest_neighbour_mask_repair(gear_down_sel.array), phase=airborne)

        # create slices indicating Gear Extending
        if gear_position:
            transits = runs_of_ones(nearest_neighbour_mask_repair(gear_position.array) == 'In Transit')
            runs = [transit for transit in transits if gear_position.array[transit.start-1] == 'Up']

        elif gear_down and (gear_in_transit or gear_red):
            param, state = (gear_in_transit, 'In Transit') if gear_in_transit else (gear_red, 'Warning')
            transits = runs_of_ones(nearest_neighbour_mask_repair(param.array) == state)
            # Filter out the transits from Down to Up states. We remove transits slices
            # which overlap with the moment Gear Down goes from Down to Up.
            gear_ups = find_edges_on_state_change('Up', nearest_neighbour_mask_repair(gear_down.array), phase=airborne)
            # Find transits within a reasonable range
            duration = fallback or 20 * self.frequency
            for gear_up_edge in gear_ups:
                gear_moving_up = slice(
                    max(gear_up_edge - duration, 0),
                    min(gear_up_edge + duration, len(self.array))
                )
                transits = [
                    transit for transit in transits
                    if not slices_overlap(transit, gear_moving_up)
                ]

            if gear_red:
                # Dataframes show spurious Gear Red Warnings.
                # Only consider transits within a reasonable range of Gear Down.
                gear_moving_downs = [
                    slice(
                        max(gear_down_edge - duration, 0),
                        min(gear_down_edge + duration, len(self.array))
                    )
                    for gear_down_edge in gear_downs
                ]
                transits = [
                            transit for gear_moving_down in gear_moving_downs
                            for transit in transits
                            if slices_overlap(transit, gear_moving_down)
                        ]

            gear_down_slices = runs_of_ones(nearest_neighbour_mask_repair(gear_down.array) == 'Down')
            runs = slices_and_not(transits, gear_down_slices)

            if family and family.value == 'B737 Classic' and extend_duration:
                for idx, run in enumerate(runs):
                    if slice_duration(run, self.frequency) > extend_duration:
                        stop = run.stop
                        runs[idx] = slice(int(math.ceil(stop-fallback)), stop)

        elif gear_down_sel and (gear_in_transit or gear_red):
            param, state = (gear_in_transit, 'In Transit') if gear_in_transit else (gear_red, 'Warning')
            transits = find_edges_on_state_change(state, nearest_neighbour_mask_repair(param.array), change='leaving', phase=airborne)
            for start in gear_sels:
                stop = min([x for x in transits if x > start] or (None,))
                if stop is not None:
                    runs.append(slice(int(math.ceil(start)), stop+1))

        elif gear_up and (gear_in_transit or gear_red):
            param, state = (gear_in_transit, 'In Transit') if gear_in_transit else (gear_red, 'Warning')
            transits = runs_of_ones(nearest_neighbour_mask_repair(param.array) == state)
            # Filter out the transits from Down to Up state. We remove transits slices
            # which overlap with the moment Gear Up goes from Down to Up.
            gear_ups = find_edges_on_state_change('Up', nearest_neighbour_mask_repair(gear_up.array), phase=airborne)
            # Find transits within a reasonable range
            duration = fallback or 20 * self.frequency
            for gear_up_edge in gear_ups:
                gear_moving_up = slice(
                    max(gear_up_edge - duration, 0),
                    min(gear_up_edge + duration, len(self.array))
                )
                transits = [
                    transit for transit in transits
                    if not slices_overlap(transit, gear_moving_up)
                ]

                if gear_red:
                    # Dataframes show spurious Gear Red Warnings.
                    # Only consider transits within a reasonable range of Gear Down.
                    gear_downs = find_edges_on_state_change(
                        'Down',
                        nearest_neighbour_mask_repair(gear_up.array),
                        phase=airborne)
                    gear_moving_downs = [
                        slice(
                            max(gear_down_edge - duration, 0),
                            min(gear_down_edge + duration, len(self.array))
                        )
                        for gear_down_edge in gear_downs
                    ]
                    transits = [
                                transit for gear_moving_down in gear_moving_downs
                                for transit in transits
                                if slices_overlap(transit, gear_moving_down)
                            ]

                runs = transits

            if family and family.value == 'B737 Classic' and extend_duration:
                for idx, run in enumerate(runs):
                    if slice_duration(run, self.frequency) > extend_duration:
                        stop = run.stop
                        runs[idx] = slice(int(math.ceil(stop-fallback)), stop)

        elif gear_down and gear_down_sel:
            runs = runs_of_ones(nearest_neighbour_mask_repair(gear_down_sel.array) == 'Down')
            gear_downs = runs_of_ones(nearest_neighbour_mask_repair(gear_down.array) == 'Down')
            runs = slices_and_not(runs, gear_downs)

        elif gear_down and gear_up:
            for start, stop in zip(gear_ups, gear_downs):
                runs.append(slice(start, stop+1))

        elif gear_down and fallback:
            for stop in gear_downs:
                runs.append(slice(int(math.ceil(stop-fallback)), stop+1))

        elif gear_up and fallback:
            for start in gear_ups:
                runs.append(slice(int(math.ceil(start)), start+fallback+1))

        elif gear_down_sel and fallback:
            for start in gear_sels:
                runs.append(slice(int(math.ceil(start)), start+fallback+1))
        else:
            pass

        for run in slices_int(runs):
            self.array[run.start:run.stop] = 'Extending'


class GearUpInTransit(MultistateDerivedParameterNode):
    units = None
    values_mapping = {
        0: '-',
        1: 'Retracting',
    }

    @classmethod
    def can_operate(cls, available, model=A('Model'), series=A('Series'), family=A('Family')):
        # Can operate with a any combination of parameters available
        gears_available = all_of(('Gear Up', 'Gear Up Selected'), available) \
            or all_of(('Gear Down', 'Gear Up'), available) \
            or all_of(('Gear Up Selected', 'Gear In Transit'), available) \
            or all_of(('Gear Up', 'Gear In Transit'), available) \
            or all_of(('Gear Down', 'Gear In Transit'), available) \
            or all_of(('Gear Up Selected', 'Gear (*) Red Warning'), available) \
            or all_of(('Gear Up', 'Gear (*) Red Warning'), available) \
            or all_of(('Gear Down', 'Gear (*) Red Warning'), available) \
            or 'Gear Position' in available

        if gears_available:
            return True

        if all_of(('Model', 'Series', 'Family', 'Airborne'), available) \
           and any_of(('Gear Up Selected', 'Gear Up', 'Gear Down'), available):
            # Check dependancies before checking if transition times are available to save necessary lookups
            if model and series and family:
                try:
                    at.get_gear_transition_times(model.value, series.value, family.value)
                except KeyError:
                    cls.exception("No gear transition times available for '%s', '%s', '%s'.",
                                  model.value, series.value, family.value)
                    return False
                return True
        else:
            return False

    def derive(self,
               gear_down=M('Gear Down'),
               gear_up=M('Gear Up'),
               gear_up_sel=M('Gear Up Selected'),
               gear_in_transit=M('Gear In Transit'),
               gear_red=M('Gear (*) Red Warning'),
               gear_position=M('Gear Position'),
               airborne=S('Airborne'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        # Extend Airborne slices to make sure we don't miss any Gear parameters at liftoff
        airborne = slices_extend(airborne.get_slices(), 1)
        gear_sels = gear_ups = gear_downs = runs = []
        self.array = np_ma_zeros_like(first_valid_parameter(gear_down, gear_up, gear_up_sel, gear_position).array, dtype=int)

        # work out fallback
        retract_duration = 0
        if model and series and family:
            try:
                retract_duration, _ = at.get_gear_transition_times(model.value, series.value, family.value)
            except KeyError:
                pass
        fallback = retract_duration * self.frequency

        # find stating points.
        if gear_down:
            gear_downs = find_edges_on_state_change('Down', nearest_neighbour_mask_repair(gear_down.array), change='leaving', phase=airborne)
        if gear_up:
            gear_ups = find_edges_on_state_change('Up', nearest_neighbour_mask_repair(gear_up.array), phase=airborne)
        if gear_up_sel:
            gear_sels = find_edges_on_state_change('Up', nearest_neighbour_mask_repair(gear_up_sel.array), phase=airborne)

        # create slices indicating Gear Retracting
        if gear_position:
            transits = runs_of_ones(nearest_neighbour_mask_repair(gear_position.array) == 'In Transit')
            runs = [transit for transit in transits if gear_position.array[transit.start-1] == 'Down']

        elif gear_up and (gear_in_transit or gear_red):
            param, state = (gear_in_transit, 'In Transit') if gear_in_transit else (gear_red, 'Warning')
            transits = runs_of_ones(nearest_neighbour_mask_repair(param.array) == state)
            # Filter out the transits from Up to Down states. We remove transits slices
            # which overlap with the moment Gear Up goes from Up to Down.
            gear_downs = find_edges_on_state_change('Down', nearest_neighbour_mask_repair(gear_up.array), phase=airborne)
            # Find transits within a reasonable range
            duration = fallback or 20 * self.frequency
            for gear_down_edge in gear_downs:
                gear_moving_down = slice(
                    max(gear_down_edge - duration, 0),
                    min(gear_down_edge + duration, len(self.array))
                )
                transits = [
                    transit for transit in transits
                    if not slices_overlap(transit, gear_moving_down)
                ]

            if gear_red:
                # Dataframes show spurious Gear Red Warnings.
                # Only consider transits within a reasonable range of Gear Up.
                gear_moving_ups = [
                    slice(
                        max(gear_up_edge - duration, 0),
                        min(gear_up_edge + duration, len(self.array))
                    )
                    for gear_up_edge in gear_ups
                ]
                transits = [
                            transit for gear_moving_up in gear_moving_ups
                            for transit in transits
                            if slices_overlap(transit, gear_moving_up)
                        ]

            gear_up_slices = runs_of_ones(nearest_neighbour_mask_repair(gear_up.array) == 'Up')
            runs = slices_and_not(transits, gear_up_slices)

            if family and family.value == 'B737 Classic' and retract_duration:
                for idx, run in enumerate(runs):
                    if slice_duration(run, self.frequency) > retract_duration:
                        stop = run.stop
                        runs[idx] = slice(int(math.ceil(stop-fallback)), stop)

        elif gear_up_sel and (gear_in_transit or gear_red):
            param, state = (gear_in_transit, 'In Transit') if gear_in_transit else (gear_red, 'Warning')
            transits = find_edges_on_state_change(state, nearest_neighbour_mask_repair(param.array), change='leaving', phase=airborne)
            for start in gear_sels:
                stop = min([x for x in transits if x >= start] or (None,))
                if stop is not None:
                    _slice = slice(int(math.ceil(start)), stop+1)
                    if family and family.value == 'B737 Classic' and retract_duration and slice_duration(_slice, self.frequency) > retract_duration:
                        _slice = slice(int(math.ceil(start)), start+fallback+1)
                    runs.append(_slice)

        elif gear_down and (gear_in_transit or gear_red):
            param, state = (gear_in_transit, 'In Transit') if gear_in_transit else (gear_red, 'Warning')
            transits = runs_of_ones(nearest_neighbour_mask_repair(param.array) == state)
            # Filter out the transits from Up to Down state. We remove transits slices
            # which overlap with the moment Gear Down goes from Up to Down.
            gear_downs = find_edges_on_state_change('Down', nearest_neighbour_mask_repair(gear_down.array), phase=airborne)
            # Find transits within a reasonable range
            duration = fallback or 20 * self.frequency
            for gear_down_edge in gear_downs:
                gear_moving_down = slice(
                    max(gear_down_edge - duration, 0),
                    min(gear_down_edge + duration, len(self.array))
                )
                transits = [
                    transit for transit in transits
                    if not slices_overlap(transit, gear_moving_down)
                ]

            if gear_red:
                # Dataframes show spurious Gear Red Warnings.
                # Only consider transits within a reasonable range of Gear Up.
                gear_ups = find_edges_on_state_change(
                    'Up',
                    nearest_neighbour_mask_repair(gear_down.array),
                    phase=airborne
                )
                gear_moving_ups = [
                    slice(
                        max(gear_up_edge - duration, 0),
                        min(gear_up_edge + duration, len(self.array))
                    )
                    for gear_up_edge in gear_ups
                ]
                transits = [
                            transit for gear_moving_up in gear_moving_ups
                            for transit in transits
                            if slices_overlap(transit, gear_moving_up)
                        ]

            runs = transits

            if family and family.value == 'B737 Classic' and retract_duration:
                for idx, run in enumerate(runs):
                    if slice_duration(run, self.frequency) > retract_duration:
                        stop = run.stop
                        runs[idx] = slice(math.ceil(stop-fallback), stop)

        elif gear_up and gear_up_sel:
            runs = runs_of_ones(nearest_neighbour_mask_repair(gear_up_sel.array) == 'Up')
            gear_ups = runs_of_ones(nearest_neighbour_mask_repair(gear_up.array) == 'Up')
            runs = slices_and_not(runs, gear_ups)

        elif gear_down and gear_up:
            for start, stop in zip(gear_downs, gear_ups):
                # Ceil also stop in case gear_down and gear_up are offset by 0.5 sec
                runs.append(slice(math.ceil(start), math.ceil(stop)+1))

        elif gear_up and fallback:
            for stop in gear_ups:
                runs.append(slice(math.ceil(stop-fallback), stop+1))

        elif gear_down and fallback:
            for start in gear_downs:
                if family and family.value == 'AW169':
                    runs.append(slice(math.ceil(start-fallback), start+1))
                else:
                    runs.append(slice(math.ceil(start), start+fallback+1))

        elif gear_up_sel and fallback:
            for start in gear_sels:
                runs.append(slice(int(math.ceil(start)), start+fallback+1))
        else:
            pass

        for run in slices_int(runs):
            self.array[run.start:run.stop] = 'Retracting'


class GearUp(MultistateDerivedParameterNode):
    '''
    This Multi-State parameter uses "majority voting" to decide whether the
    gear is up or down.
    '''

    align = False
    units = None
    values_mapping = {
        0: 'Down',
        1: 'Up',
    }

    @classmethod
    def can_operate(cls, available):
        # Can operate with a any combination of parameters available
        calc_gear_up = all_of(('Gear Up Selected', 'Gear Up In Transit'), available)
        gear_pos = 'Gear Position' in available
        return calc_gear_up or gear_pos

    def derive(self,
               gear_transit=M('Gear Up In Transit'),
               gear_sel=M('Gear Up Selected'),
               gear_pos=M('Gear Position')):
        if gear_sel and gear_transit:
            gear_sel_array = align(gear_sel, gear_transit) if gear_sel.hz != gear_transit.hz else gear_sel.array
            self.array = (gear_sel_array == 'Up') & ~(gear_transit.array == 'Retracting')
            _slices = runs_of_ones(self.array == 'Down')
            _slices = slices_remove_small_gaps(_slices, 6, self.hz)
            for _slice in _slices:
                self.array[_slice] = 'Down'
        else:
            self.array = gear_pos.array == 'Up'


class GearInTransit(MultistateDerivedParameterNode):
    '''
    This Multi-State parameter uses "majority voting" to decide whether the
    gear is in transit.
    '''

    align = False
    units = None
    values_mapping = {
        0: '-',
        1: 'In Transit',
    }

    def derive(self,
               gear_down_transit=M('Gear Down In Transit'),
               gear_up_transit=M('Gear Up In Transit')):
        self.array = (gear_down_transit.array == 'Extending') | \
            (gear_up_transit.array == 'Retracting')


class GearOnGround(MultistateDerivedParameterNode):
    '''
    Combination of left and right main gear signals.
    '''
    align = False
    units = None
    values_mapping = {
        0: 'Air',
        1: 'Ground',
    }

    @classmethod
    def can_operate(cls, available):
        return any_of(('Gear (L) On Ground', 'Gear (R) On Ground'), available)

    def derive(self,
               gl=M('Gear (L) On Ground'),
               gr=M('Gear (R) On Ground')):

        # Note that this is not needed on the following frames which record
        # this parameter directly: 737-4, 737-i

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
        else:
            # should not get here if can_operate is correct
            raise NotImplementedError()


class GearDownSelected(MultistateDerivedParameterNode):
    align_frequency = 1
    units = None
    values_mapping = {
        0: 'Up',
        1: 'Down',
    }

    def derive(self, up_sel=M('Gear Up Selected')):
        self.array = up_sel.array == 'Down'


class GearUpSelected(MultistateDerivedParameterNode):
    '''
    This is the inverse of 'Gear Down Selected' which does all the hard work
    for us establishing transitions from 'Gear Down' with the assocaited Red
    Warnings.
    '''
    align = False
    units = None
    values_mapping = {
        0: 'Down',
        1: 'Up',
    }

    @classmethod
    def can_operate(cls, available):
        return all_of(('Gear Up', 'Gear Up In Transit'), available) or \
               all_of(('Gear Down', 'Gear Down In Transit'), available) or \
               'Gear Down Selected' in available

    def derive(self,
               gear_up=M('Gear Up'),
               gear_up_transit=M('Gear Up In Transit'),
               gear_down=M('Gear Down'),
               gear_down_transit=M('Gear Down In Transit'),
               gear_down_sel=M('Gear Down Selected')):

        if gear_down_sel:
            self.frequency = gear_down_sel.frequency
            self.offset = gear_down_sel.offset
            if gear_down_sel.frequency >= 1:
                # Use Gear Down Sel if the sample rate is sufficient
                self.array = 1 - (gear_down_sel.array == 'Down')
                return
            else:
                self.array = np_ma_masked_zeros_like(gear_down_sel.array)

        if gear_up and gear_up_transit:
            gear_up, gear_up_transit = self._align_params(gear_up, gear_up_transit)
            self.array = (gear_up.array == 'Up') | \
                (gear_up_transit.array == 'Retracting')
        elif gear_down and gear_down_transit:
            gear_down, gear_down_transit = self._align_params(gear_down, gear_down_transit)
            self.array = 1 - ((gear_down.array == 'Down') | \
                              (gear_down_transit.array == 'Extending'))

    def _align_params(self, *args):
        '''
        Align the parameters to 1 Hz.

        This has the same effect as align_frequency = 1.
        This is a generator, using the parameters found in args and yielding the
        aligned parameters.
        '''

        self.frequency = 1
        self.offset = args[0].offset

        for arg in args:
            try:
                aligned_arg = arg.get_aligned(self)
            except AttributeError:
                # If parameter came from an HDF, it's missing get_aligned
                arg = derived_param_from_hdf(arg, cache=self._cache)
                aligned_arg = arg.get_aligned(self)
            yield aligned_arg


class Gear_RedWarning(MultistateDerivedParameterNode):
    '''
    Merges all the Red Warning systems for Nose, Left and Right gears.
    Ensures that false warnings on the ground are ignored.
    '''
    name = 'Gear (*) Red Warning'
    units = None
    values_mapping = {0: '-', 1: 'Warning'}
    #store in hdf = False! glimpse into the future ;)

    @classmethod
    def can_operate(cls, available):
        return 'Airborne' in available and any_of((
            'Gear (L) Red Warning',
            'Gear (N) Red Warning',
            'Gear (R) Red Warning',
        ), available)

    def derive(self,
               gear_warn_l=M('Gear (L) Red Warning'),
               gear_warn_n=M('Gear (N) Red Warning'),
               gear_warn_r=M('Gear (R) Red Warning'),
               airs=S('Airborne')):

        # Join available gear parameters and use whichever are available.
        red_warning = vstack_params_where_state(
            (gear_warn_l, 'Warning'),
            (gear_warn_n, 'Warning'),
            (gear_warn_r, 'Warning'),
        )
        in_air = np.zeros(len(red_warning[0]), dtype=np.bool)
        for air in airs:
            in_air[air.slice] = 1
        # ensure that the red warnings were in the air
        ##gear_warn = M(array=red_warning.any(axis=0), values_mapping={
            ##True: 'Warning'})
        red_air = red_warning.any(axis=0) & in_air
        # creating mapped array is probably not be required due to __setattr__
        red = np.ma.zeros(len(red_air), dtype=np.short)
        red[red_air] = 1
        self.array = MappedArray(red, values_mapping=self.values_mapping)


class ILSInnerMarker(MultistateDerivedParameterNode):
    '''
    Combine ILS Marker for captain and first officer where recorded separately.
    '''
    values_mapping = {0: '-', 1: 'Present'}
    align = False
    name = 'ILS Inner Marker'
    units = None

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               ils_mkr_capt=M('ILS Inner Marker (Capt)'),
               ils_mkr_fo=M('ILS Inner Marker (FO)')):

        self.array = vstack_params_where_state(
            (ils_mkr_capt, 'Present'),
            (ils_mkr_fo, 'Present'),
        ).any(axis=0)


class ILSMiddleMarker(MultistateDerivedParameterNode):
    '''
    Combine ILS Marker for captain and first officer where recorded separately.
    '''
    values_mapping = {0: '-', 1: 'Present'}
    align = False
    name = 'ILS Middle Marker'
    units = None

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               ils_mkr_capt=M('ILS Middle Marker (Capt)'),
               ils_mkr_fo=M('ILS Middle Marker (FO)')):

        self.array = vstack_params_where_state(
            (ils_mkr_capt, 'Present'),
            (ils_mkr_fo, 'Present'),
        ).any(axis=0)


class ILSOuterMarker(MultistateDerivedParameterNode):
    '''
    Combine ILS Marker for captain and first officer where recorded separately.
    '''
    values_mapping = {0: '-', 1: 'Present'}
    align = False
    name = 'ILS Outer Marker'
    units = None

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               ils_mkr_capt=M('ILS Outer Marker (Capt)'),
               ils_mkr_fo=M('ILS Outer Marker (FO)')):

        self.array = vstack_params_where_state(
            (ils_mkr_capt, 'Present'),
            (ils_mkr_fo, 'Present'),
        ).any(axis=0)


class KeyVHFCapt(MultistateDerivedParameterNode):

    name = 'Key VHF (Capt)'
    units = None
    values_mapping = {0: '-', 1: 'Keyed'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self, key_vhf_1=M('Key VHF (1) (Capt)'),
               key_vhf_2=M('Key VHF (2) (Capt)'),
               key_vhf_3=M('Key VHF (3) (Capt)')):
        self.array = vstack_params_where_state(
            (key_vhf_1, 'Keyed'),
            (key_vhf_2, 'Keyed'),
            (key_vhf_3, 'Keyed'),
        ).any(axis=0)


class KeyVHFFO(MultistateDerivedParameterNode):

    name = 'Key VHF (FO)'
    units = None
    values_mapping = {0: '-', 1: 'Keyed'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self, key_vhf_1=M('Key VHF (1) (FO)'),
               key_vhf_2=M('Key VHF (2) (FO)'),
               key_vhf_3=M('Key VHF (3) (FO)')):
        self.array = vstack_params_where_state(
            (key_vhf_1, 'Keyed'),
            (key_vhf_2, 'Keyed'),
            (key_vhf_3, 'Keyed'),
        ).any(axis=0)


class MasterCaution(MultistateDerivedParameterNode):
    '''
    Combine Master Caution for captain and first officer.
    '''
    units = None
    values_mapping = {0: '-', 1: 'Caution'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               capt=M('Master Caution (Capt)'),
               fo=M('Master Caution (FO)'),
               capt_2=M('Master Caution (Capt) (2)'),
               fo_2=M('Master Caution (FO) (2)'),
               ):

        self.array = vstack_params_where_state(
            (capt, 'Caution'),
            (fo, 'Caution'),
            (capt_2, 'Caution'),
            (fo_2, 'Caution'),
        ).any(axis=0)


class MasterWarning(MultistateDerivedParameterNode):
    '''
    Combine master warning for captain and first officer.
    '''
    units = None
    values_mapping = {0: '-', 1: 'Warning'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               warn_capt=M('Master Warning (Capt)'),
               warn_fo=M('Master Warning (FO)')):

        self.array = vstack_params_where_state(
            (warn_capt, 'Warning'),
            (warn_fo, 'Warning'),
        ).any(axis=0)


class PackValvesOpen(MultistateDerivedParameterNode):
    '''
    Integer representation of the combined pack configuration.
    '''

    name = 'Pack Valves Open'
    units = None
    values_mapping = {
        0: 'All closed',
        1: 'One engine low flow',
        2: 'Flow level 2',
        3: 'Flow level 3',
        4: 'Both engines high flow',
    }

    @classmethod
    def can_operate(cls, available):
        '''
        '''
        # Works with both 'ECS Pack (1) On' and 'ECS Pack (2) On' ECS Pack High Flows are optional
        return all_of(['ECS Pack (1) On', 'ECS Pack (2) On'], available)

    def derive(self,
               p1=M('ECS Pack (1) On'), p1h=M('ECS Pack (1) High Flow'),
               p2=M('ECS Pack (2) On'), p2h=M('ECS Pack (2) High Flow')):
        '''
        '''
        # TODO: account properly for states/frame specific fixes
        # Sum the open engines, allowing 1 for low flow and 1+1 for high flow
        # each side.
        flow = p1.array.raw + p2.array.raw
        if p1h and p2h:
            flow = p1.array.raw * (1 + p1h.array.raw) + p2.array.raw * (1 + p2h.array.raw)
        self.array = flow
        self.offset = offset_select('mean', [p1, p1h, p2, p2h])


class PilotFlying(MultistateDerivedParameterNode):
    '''
    Determines the pilot flying the aircraft.

    For Airbus aircraft we use the captain and first officer sidestick angles
    to determine who is providing input to the aircraft.

    Reference was made to the following documentation to assist with the
    development of this algorithm:

    - A320 Flight Profile Specification
    - A321 Flight Profile Specification
    '''
    units = None
    values_mapping = {0: '-', 1: 'Captain', 2: 'First Officer'}

    def derive(self,
               stick_capt=P('Sidestick Angle (Capt)'),
               stick_fo=P('Sidestick Angle (FO)')):


        pilot_flying = MappedArray(np_ma_masked_zeros_like(stick_capt.array, dtype=np.short),
                                   values_mapping=self.values_mapping)

        if stick_capt.array.size > 61:
            # Calculate average instead of sum as it we already have a function
            # defined to work over a window and it doesn't affect the result as
            # the arrays are altered in the same way and are still comparable.
            window = 31 * self.hz  # Use 61 seconds for 30 seconds either side.
            if not window % 2:
                window += 1
            angle_capt = moving_average(np.ma.abs(stick_capt.array), window)
            angle_fo = moving_average(np.ma.abs(stick_fo.array), window)
            # Repair the array as the moving average is padded with masked
            # zeros
            angle_capt = repair_mask(angle_capt, repair_duration=31,
                                     extrapolate=True)
            angle_fo = repair_mask(angle_fo, repair_duration=31,
                                   extrapolate=True)
            # ignore moving average if no input from pilot at that time.
            # AFPS declares 0.5 degrees minimum input, but due to A330/A340
            # poor resolution, allow 1.7 degrees of movement.
            angle_capt_zerod = np.ma.where(stick_capt.array < 1.7, 0.0, angle_capt)
            angle_fo_zerod = np.ma.where(stick_fo.array < 1.7, 0.0, angle_fo)
            # mask non inputs to allow us to repair nearest neightbour later
            angle_capt_masked = np.ma.masked_where((stick_capt.array == 0.0) & (angle_capt_zerod != 0.0), angle_capt_zerod)
            angle_fo_masked = np.ma.masked_where((stick_fo.array == 0.0) & (angle_fo_zerod != 0.0), angle_fo_zerod)

            pilot_flying[angle_capt_masked > angle_fo_masked] = 'Captain'
            pilot_flying[angle_capt_masked < angle_fo_masked] = 'First Officer'
            # keep calculated masks
            pilot_flying.mask = angle_capt_masked.mask & angle_fo_masked.mask

            # repair nearest neighbour to remove small gaps of no movement
            pilot_flying = nearest_neighbour_mask_repair(pilot_flying, repair_gap_size=20*self.frequency, copy=False)
            # use second window to remove spiking between captain and first
            # officer during dual stick periods
            pilot_flying = second_window(pilot_flying.raw, self.frequency, 2).astype(np.short)

        self.array = pilot_flying


class PitchAlternateLaw(MultistateDerivedParameterNode):
    '''
    Combine Pitch Alternate Law from sources (1) and/or (2).
    '''
    units = None
    values_mapping = {0: '-', 1: 'Engaged'}

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               alt_law_1=M('Pitch Alternate Law (1)'),
               alt_law_2=M('Pitch Alternate Law (2)')):

        self.array = vstack_params_where_state(
            (alt_law_1, 'Engaged'),
            (alt_law_2, 'Engaged'),
        ).any(axis=0)


class PitchDisconnect(MultistateDerivedParameterNode):
    '''
    Combine Pitch Disconnect Parameters from sources (1) and (2).
    '''
    units = None
    values_mapping = {
        0: '-',
        1: 'Disconnect',
    }

    def derive(self, pitch_1=M('Pitch Disconnect (1)'),
                     pitch_2=M('Pitch Disconnect (2)'),):
        self.array = vstack_params_where_state(
            (pitch_1, 'Disconnect'),
            (pitch_2, 'Disconnect'),
        ).any(axis=0)


class Slat(MultistateDerivedParameterNode):
    '''
    Steps raw slat angle into detents.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Slat Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_slat_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No slat mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, slat=P('Slat Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):
        self.values_mapping, self.array, self.frequency, self.offset = calculate_slat(
            'lever',
            slat,
            model,
            series,
            family,
        )


class SlatExcludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the lower of the start and endpoints of the movement
    apply. This minimises the chance of needing a slat overspeed inspection.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Slat Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_slat_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No slat mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, slat=P('Slat Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        self.values_mapping, self.array, self.frequency, self.offset = calculate_slat(
            'excluding',
            slat,
            model,
            series,
            family,
        )


class SlatIncludingTransition(MultistateDerivedParameterNode):
    '''
    Specifically designed to cater for maintenance monitoring, this assumes
    that when moving the higher of the start and endpoints of the movement
    apply. This increases the chance of needing a slat overspeed inspection,
    but provides a more cautious interpretation of the maintenance
    requirements.
    '''

    units = ut.DEGREE

    @classmethod
    def can_operate(cls, available,
                    model=A('Model'), series=A('Series'), family=A('Family')):

        if not all_of(('Slat Angle', 'Model', 'Series', 'Family'), available):
            return False

        try:
            at.get_slat_map(model.value, series.value, family.value)
        except KeyError:
            cls.debug("No slat mapping available for '%s', '%s', '%s'.",
                      model.value, series.value, family.value)
            return False

        return True

    def derive(self, slat=P('Slat Angle'),
               model=A('Model'), series=A('Series'), family=A('Family')):

        self.values_mapping = at.get_slat_map(model.value, series.value, family.value)
        self.array=including_transition(slat.array, self.values_mapping, hz=self.hz)


class SlatFullyExtended(MultistateDerivedParameterNode):
    units = None
    values_mapping = {0: '-', 1: 'Extended'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               slat_l1=P('Slat (L1) Fully Extended'),
               slat_l2=P('Slat (L2) Fully Extended'),
               slat_l3=P('Slat (L3) Fully Extended'),
               slat_l4=P('Slat (L4) Fully Extended'),
               slat_r1=P('Slat (R1) Fully Extended'),
               slat_r2=P('Slat (R2) Fully Extended'),
               slat_r3=P('Slat (R3) Fully Extended'),
               slat_r4=P('Slat (R4) Fully Extended'),
               slat_1=P('Slat (1) Fully Extended'),
               slat_2=P('Slat (2) Fully Extended'),
               slat_3=P('Slat (3) Fully Extended'),
               slat_4=P('Slat (4) Fully Extended'),
               slat_5=P('Slat (5) Fully Extended'),
               slat_6=P('Slat (6) Fully Extended'),
               slat_7=P('Slat (7) Fully Extended'),
               slat_8=P('Slat (8) Fully Extended'),):

        extended_params = (slat_l1, slat_l2, slat_l3, slat_l4, slat_r1, slat_r2, slat_r3, slat_r4,
                           slat_1, slat_2, slat_3, slat_4, slat_5, slat_6, slat_7, slat_8)
        extended_stack = vstack_params_where_state(*[(d, 'Extended') for d in extended_params])

        array = np_ma_zeros_like(extended_stack[0], dtype=np.short)
        array = np.ma.where(extended_stack.all(axis=0), 1, array)

        # mask indexes with greater than 50% masked values
        mask = np.ma.where(extended_stack.mask.sum(axis=0).astype(float) / len(extended_stack) * 100 > 50, 1, 0)
        self.array = array
        self.array.mask = mask


class SlatPartExtended(MultistateDerivedParameterNode):
    units = None
    values_mapping = {0: '-', 1: 'Part Extended'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               slat_l1=P('Slat (L1) Part Extended'),
               slat_l2=P('Slat (L2) Part Extended'),
               slat_l3=P('Slat (L3) Part Extended'),
               slat_l4=P('Slat (L4) Part Extended'),
               slat_r1=P('Slat (R1) Part Extended'),
               slat_r2=P('Slat (R2) Part Extended'),
               slat_r3=P('Slat (R3) Part Extended'),
               slat_r4=P('Slat (R4) Part Extended'),
               slat_1=P('Slat (1) Part Extended'),
               slat_2=P('Slat (2) Part Extended'),
               slat_3=P('Slat (3) Part Extended'),
               slat_4=P('Slat (4) Part Extended'),
               slat_5=P('Slat (5) Part Extended'),
               slat_6=P('Slat (6) Part Extended'),
               slat_7=P('Slat (7) Part Extended'),
               slat_8=P('Slat (8) Part Extended')):

        extended_params = (slat_l1, slat_l2, slat_l3, slat_l4, slat_r1, slat_r2, slat_r3, slat_r4,
                           slat_1, slat_2, slat_3, slat_4, slat_5, slat_6, slat_7, slat_8)
        extended_stack = vstack_params_where_state(*[(d, 'Part Extended') for d in extended_params])

        array = np_ma_zeros_like(extended_stack[0], dtype=np.short)
        array = np.ma.where(extended_stack.all(axis=0), 1, array)

        # mask indexes with greater than 50% masked values
        mask = np.ma.where(extended_stack.mask.sum(axis=0).astype(float) / len(extended_stack) * 100 > 50, 1, 0)
        self.array = array
        self.array.mask = mask


class SlatInTransit(MultistateDerivedParameterNode):
    units = None
    values_mapping = {0: '-', 1: 'In Transit'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               slat_l1=P('Slat (L1) In Transit'),
               slat_l2=P('Slat (L2) In Transit'),
               slat_l3=P('Slat (L3) In Transit'),
               slat_l4=P('Slat (L4) In Transit'),
               slat_r1=P('Slat (R1) In Transit'),
               slat_r2=P('Slat (R2) In Transit'),
               slat_r3=P('Slat (R3) In Transit'),
               slat_r4=P('Slat (R4) In Transit'),
               slat_1=P('Slat (1) In Transit'),
               slat_2=P('Slat (2) In Transit'),
               slat_3=P('Slat (3) In Transit'),
               slat_4=P('Slat (4) In Transit'),
               slat_5=P('Slat (5) In Transit'),
               slat_6=P('Slat (6) In Transit'),
               slat_7=P('Slat (7) In Transit'),
               slat_8=P('Slat (8) In Transit')):
        transit_params = (slat_l1, slat_l2, slat_l3, slat_l4, slat_r1, slat_r2, slat_r3, slat_r4,
                          slat_1, slat_2, slat_3, slat_4, slat_5, slat_6, slat_7, slat_8)
        transit_stack = vstack_params_where_state(*[(d, 'In Transit') for d in transit_params])

        array = np_ma_zeros_like(transit_stack[0], dtype=np.short)
        array = np.ma.where(transit_stack.all(axis=0), 1, array)

        # mask indexes with greater than 50% masked values
        mask = np.ma.where(transit_stack.mask.sum(axis=0).astype(float) / len(transit_stack) * 100 > 50, 1, 0)
        self.array = array
        self.array.mask = mask


class SlatRetracted(MultistateDerivedParameterNode):
    units = None
    values_mapping = {0: '-', 1: 'Retracted'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               slat_l1=P('Slat (L1) Retracted'),
               slat_l2=P('Slat (L2) Retracted'),
               slat_l3=P('Slat (L3) Retracted'),
               slat_l4=P('Slat (L4) Retracted'),
               slat_r1=P('Slat (R1) Retracted'),
               slat_r2=P('Slat (R2) Retracted'),
               slat_r3=P('Slat (R3) Retracted'),
               slat_r4=P('Slat (R4) Retracted')):

        retracted_params = (slat_l1, slat_l2, slat_l3, slat_l4, slat_r1, slat_r2, slat_r3, slat_r4)
        retracted_stack = vstack_params_where_state(*[(d, 'Retracted') for d in retracted_params])

        array = np_ma_zeros_like(retracted_stack[0], dtype=np.short)
        array = np.ma.where(retracted_stack.all(axis=0), 1, array)

        # mask indexes with greater than 50% masked values
        mask = np.ma.where(retracted_stack.mask.sum(axis=0).astype(float) / len(retracted_stack) * 100 > 50, 1, 0)
        self.array = array
        self.array.mask = mask


class StickPusher(MultistateDerivedParameterNode):
    '''
    Where two Stick Pusher systems are recorded the results are OR'd to make
    a single parameter which operates in response to either system
    triggering.
    '''

    units = None
    values_mapping = {0: '-', 1: 'Push'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self, spl=M('Stick Pusher (L)'),
               spr=M('Stick Pusher (R)')):

        available = [par for par in [spl, spr] if par]

        if len(available) > 1:
            shake_stack = vstack_params_where_state(*[(s, 'Push') for s in available])
            self.array = shake_stack.any(axis=0)
        elif len(available) == 1:
            self.array = available[0].array


class StickShaker(MultistateDerivedParameterNode):
    '''
    This accounts for the different types of stick shaker system. Where two
    systems are recorded the results are OR'd to make a single parameter which
    operates in response to either system triggering.
    '''

    units = None
    values_mapping = {0: '-', 1: 'Shake'}

    @classmethod
    def can_operate(cls, available):
        return any_of((
            'Stick Shaker (L)',
            'Stick Shaker (R)',
            'Stick Shaker (1)',
            'Stick Shaker (2)',
            'Stick Shaker (3)',
            'Stick Shaker (4)',
        ), available)

    def derive(self, ssl=M('Stick Shaker (L)'),
               ssr=M('Stick Shaker (R)'),
               ss1=M('Stick Shaker (1)'),
               ss2=M('Stick Shaker (2)'),
               ss3=M('Stick Shaker (3)'),
               ss4=M('Stick Shaker (4)'),
               frame=A('Frame')):

        if frame and frame.value == 'B777':
            #Provision has been included for Boeing 777 type, but until this has been
            #evaluated in detail it raises an exception because there are two bits per
            #shaker, and their operation is not obvious from the documentation.
            raise ValueError

        available = [par for par in [ssl, ssr, ss1, ss2, ss3, ss4,
                                     #b777_L1, b777_L2, b777_R1, b777_R2,
                                     ] if par]
        if len(available) > 1:
            shake_stack = vstack_params_where_state(*[(s, 'Shake') for s in available])
            self.array = shake_stack.any(axis=0)
        elif len(available) == 1:
            self.array = available[0].array


class StallWarning(MultistateDerivedParameterNode):
    '''
    This accounts for the different types of stall warning system. Where two
    systems are recorded the results are OR'd to make a single parameter which
    operates in response to either system triggering.
    '''

    units = None
    values_mapping = {0: '-', 1: 'Warning'}

    @classmethod
    def can_operate(cls, available):
        return any_of((
            'Stall Warning (1)',
            'Stall Warning (2)',
        ), available)

    def derive(self,
               ss1=M('Stall Warning (1)'),
               ss2=M('Stall Warning (2)'),
               frame=A('Frame'),
               ):

        available = [par for par in [ss1, ss2] if par]
        if len(available) > 1:
            shake_stack = vstack_params_where_state(*[(s, 'Warning') for s in available])
            self.array = shake_stack.any(axis=0)
        elif len(available) == 1:
            self.array = available[0].array


class SmokeWarning(MultistateDerivedParameterNode):
    '''
    Merging all available smoke alert signals into a single parameter for
    subsequent monitoring.
    '''
    units = None
    values_mapping = {0: '-', 1: 'Smoke'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               smoke_avionics=M('Smoke Avionics Warning'),
               smoke_avionics_1=M('Smoke Avionics (1) Warning'),
               smoke_avionics_2=M('Smoke Avionics (2) Warning'),
               smoke_lav=M('Smoke Lavatory Warning'),
               smoke_lav_1=M('Smoke Lavatory (1) Warning'),
               smoke_lav_2=M('Smoke Lavatory (2) Warning'),
               smoke_cabin=M('Smoke Cabin Warning'),
               smoke_cabin_1=M('Smoke Cabin Rest (1) Warning'),
               smoke_cabin_2=M('Smoke Cabin Rest (2) Warning'),
               smoke_cargo=M('Smoke Cargo Warning'),
               smoke_cargo_fwd_1=M('Smoke Cargo Fwd (1) Warning'),
               smoke_cargo_fwd_2=M('Smoke Cargo Fwd (2) Warning'),
               smoke_cargo_aft_1=M('Smoke Cargo Aft (1) Warning'),
               smoke_cargo_aft_2=M('Smoke Cargo Aft (2) Warning'),
               smoke_cargo_rest_1=M('Smoke Cargo Rest (1) Warning'),
               smoke_cargo_rest_2=M('Smoke Cargo Rest (2) Warning'),
               smoke_lower_dec=M('Smoke Lower Deck Stowage'),
               smoke_avionic_bulk=M('Smoke Avionic Bulk'),
               smoke_ifec=M('Smoke IFEC'),
               smoke_bcrc=M('Smoke BCRC'),
               smoke_vcc=M('Smoke Autonomous VCC')):

        params = (smoke_avionics, smoke_avionics_1, smoke_avionics_2,
                  smoke_lav, smoke_lav_1, smoke_lav_2, smoke_cabin, smoke_cabin_1,
                  smoke_cabin_2, smoke_cargo, smoke_cargo_fwd_1, smoke_cargo_fwd_2,
                  smoke_cargo_aft_1, smoke_cargo_aft_2, smoke_cargo_rest_1,
                  smoke_cargo_rest_2, smoke_lower_dec, smoke_avionic_bulk, smoke_ifec,
                  smoke_bcrc, smoke_vcc)

        params_state = vstack_params_where_state(
            *[(param, 'Smoke') for param in params if param]
        )
        self.array = params_state.any(axis=0)


class SpeedbrakeDeployed(MultistateDerivedParameterNode):
    '''
    Follows same logic as when deriving speedbrake from spoiler angles. Two
    matching spoilers in deployed state indicates speedbrake, single side
    deployment indicates roll.
    '''
    units = None
    values_mapping = {0: '-', 1: 'Deployed'}

    @classmethod
    def can_operate(cls, available, family=A('Family')):
        if family and family.value == 'B787':
            return 'Speedbrake Handle' in available

        return 'Ground Spoiler Deployed' in available or \
               'Speedbrake (Tail) Deployed' in available or \
               all_of(('Spoiler (L) Deployed', 'Spoiler (R) Deployed'), available) or \
               all_of(('Ground Spoiler (L) Deployed', 'Ground Spoiler (R) Deployed'), available) or \
               all_of(('Spoiler (L) (1) Deployed', 'Spoiler (R) (1) Deployed'), available) or \
               all_of(('Spoiler (L) (2) Deployed', 'Spoiler (R) (2) Deployed'), available) or \
               all_of(('Spoiler (L) (3) Deployed', 'Spoiler (R) (3) Deployed'), available) or \
               all_of(('Spoiler (L) (4) Deployed', 'Spoiler (R) (4) Deployed'), available) or \
               all_of(('Spoiler (L) (5) Deployed', 'Spoiler (R) (5) Deployed'), available) or \
               all_of(('Spoiler (L) (6) Deployed', 'Spoiler (R) (6) Deployed'), available) or \
               all_of(('Spoiler (L) (7) Deployed', 'Spoiler (R) (7) Deployed'), available) or \
               all_of(('Ground Spoiler (L) (1) Deployed', 'Ground Spoiler (R) (1) Deployed'), available) or \
               all_of(('Ground Spoiler (L) (2) Deployed', 'Ground Spoiler (R) (2) Deployed'), available) or \
               all_of(('Spoiler (L) Outboard Deployed', 'Spoiler (R) Outboard Deployed'), available) or \
               'Spoiler' in available or \
               all_of(('Spoiler (L)', 'Spoiler (R)'), available) or \
               all_of(('Spoiler (L) (1)', 'Spoiler (R) (1)'), available) or \
               all_of(('Spoiler (L) (2)', 'Spoiler (R) (2)'), available) or \
               all_of(('Spoiler (L) (3)', 'Spoiler (R) (3)'), available) or \
               all_of(('Spoiler (L) (4)', 'Spoiler (R) (4)'), available) or \
               all_of(('Spoiler (L) (5)', 'Spoiler (R) (5)'), available) or \
               all_of(('Spoiler (L) (6)', 'Spoiler (R) (6)'), available) or \
               all_of(('Spoiler (L) (7)', 'Spoiler (R) (7)'), available) or \
               all_of(('Spoiler (L) Outboard', 'Spoiler (R) Outboard'), available)

    def derive(self, dep=M('Ground Spoiler Deployed'),
               tail=M('Speedbrake (Tail) Deployed'),
               ld=M('Spoiler (L) Deployed'),
               rd=M('Spoiler (R) Deployed'),
               gld=M('Ground Spoiler (L) Deployed'),
               grd=M('Ground Spoiler (R) Deployed'),
               l1d=M('Spoiler (L) (1) Deployed'),
               l2d=M('Spoiler (L) (2) Deployed'),
               l3d=M('Spoiler (L) (3) Deployed'),
               l4d=M('Spoiler (L) (4) Deployed'),
               l5d=M('Spoiler (L) (5) Deployed'),
               l6d=M('Spoiler (L) (6) Deployed'),
               l7d=M('Spoiler (L) (7) Deployed'),
               gl1d=M('Ground Spoiler (L) (1) Deployed'),
               gl2d=M('Ground Spoiler (L) (2) Deployed'),
               r1d=M('Spoiler (R) (1) Deployed'),
               r2d=M('Spoiler (R) (2) Deployed'),
               r3d=M('Spoiler (R) (3) Deployed'),
               r4d=M('Spoiler (R) (4) Deployed'),
               r5d=M('Spoiler (R) (5) Deployed'),
               r6d=M('Spoiler (R) (6) Deployed'),
               r7d=M('Spoiler (R) (7) Deployed'),
               gr1d=M('Ground Spoiler (R) (1) Deployed'),
               gr2d=M('Ground Spoiler (R) (2) Deployed'),
               loutd=M('Spoiler (L) Outboard Deployed'),
               routd=M('Spoiler (R) Outboard Deployed'),
               spoiler=P('Spoiler'),
               l=P('Spoiler (L)'),
               r=P('Spoiler (R)'),
               l1=P('Spoiler (L) (1)'),
               l2=P('Spoiler (L) (2)'),
               l3=P('Spoiler (L) (3)'),
               l4=P('Spoiler (L) (4)'),
               l5=P('Spoiler (L) (5)'),
               l6=P('Spoiler (L) (6)'),
               l7=P('Spoiler (L) (7)'),
               r1=P('Spoiler (R) (1)'),
               r2=P('Spoiler (R) (2)'),
               r3=P('Spoiler (R) (3)'),
               r4=P('Spoiler (R) (4)'),
               r5=P('Spoiler (R) (5)'),
               r6=P('Spoiler (R) (6)'),
               r7=P('Spoiler (R) (7)'),
               lout=P('Spoiler (L) Outboard'),
               rout=P('Spoiler (R) Outboard'),
               handle=P('Speedbrake Handle'),
               family=A('Family')):

        left = (ld, gld, l1d, l2d, l3d, l4d, l5d, l6d, l7d, gl1d, gl2d, loutd,
                l, l1, l2, l3, l4, l5, l6, l7, lout)
        right = (rd, grd, r1d, r2d, r3d, r4d, r5d, r6d, r7d, gr1d, gr2d, routd,
                 r, r1, r2, r3, r4, r5, r6, r7, rout)
        pairs = list(zip(left, right))
        state = 'Deployed'

        def is_deployed(param):
            if not param:
                return
            array = np_ma_zeros_like(
                param.array, dtype=np.bool, mask=param.array.mask)
            if state in param.name:
                if state not in param.array.state:
                    logger.warning("State '%s' not found in param '%s'", state, param.name)
                    return
                matching = param.array == state
            elif family and family.value == 'MD-11':
                matching = param.array >= 15
            else:
                matching = param.array >= 10

            for s in runs_of_ones(matching, min_samples=1):
                array[s] = True
            return array

        if family and family.value == 'B787':
            speedbrake = np.ma.zeros(len(handle.array), dtype=np.short)
            stepped_array = step_values(handle.array, [0, 20], hz=handle.hz)
            speedbrake[stepped_array == 20] = 1
            self.array = speedbrake
        else:
            combined = [a for a in (is_deployed(p) for p in (dep, tail, spoiler)) if a is not None]
            combined.extend(
                np.ma.vstack(arrays).all(axis=0) for arrays in
                ([is_deployed(p) for p in pair] for pair in pairs)
                if all(a is not None for a in arrays))

            if not combined:
                self.array = np_ma_zeros_like(
                    next(p.array for p in (dep, spoiler) + left + right if p is not None),
                    dtype=np.short, mask=True)
                return

            stack = np.ma.vstack(combined)

            array = np_ma_zeros_like(stack[0], dtype=np.short)
            array = np.ma.where(stack.any(axis=0), 1, array)

            # mask indexes with greater than 50% masked values
            mask = np.ma.where(stack.mask.sum(axis=0).astype(float) / len(stack) * 100 > 50, 1, 0)
            self.array = array
            self.array.mask = mask


class SpeedbrakeSelected(MultistateDerivedParameterNode):
    '''
    Determines the selected state of the speedbrake.

    Speedbrake Selected Values:

    - 0 -- Stowed
    - 1 -- Armed / Commanded (Spoilers Down)
    - 2 -- Deployed / Commanded (Spoilers Up)
    '''
    units = None
    values_mapping = {0: 'Stowed', 1: 'Armed/Cmd Dn', 2: 'Deployed/Cmd Up'}

    @classmethod
    def can_operate(cls, available, family=A('Family')):
        '''
        '''
        x = available
        if family and family.value == 'BD-100':
            return 'Speedbrake Handle' in x and 'Speedbrake Armed' in x
        elif family and family.value == 'Global':
            return any_of(('Speedbrake', 'Speedbrake Handle'), available)
        elif family and family.value in ('CRJ 100/200', 'B777'):
            return 'Speedbrake Handle' in x
        elif family and family.value in ('A318', 'A319', 'A320', 'A321', 'MD-11'):
            return 'Speedbrake' in x and 'Speedbrake Armed' in x
        elif family and family.value in ('A340', 'A380'):
            return ('Speedbrake Deployed' in x or
                    all_of(('Speedbrake', 'Speedbrake Handle'), x))
        else:
            return ('Speedbrake Deployed' in x or
                    ('Family' in x and 'Speedbrake Switch' in x) or
                    ('Family' in x and 'Speedbrake Handle' in x) or
                    ('Family' in x and 'Speedbrake' in x))

    @classmethod
    def derive_from_handle(cls, handle_array, deployed=1, armed=None,
                           mask_below_armed=False):
        '''
        Basic Speedbrake algorithm for Stowed and Deployed states from
        Spoiler Handle.
        '''
        array = MappedArray(np_ma_masked_zeros_like(handle_array, dtype=np.short),
                            values_mapping=cls.values_mapping)
        stowed_value = deployed
        if armed is not None:
            stowed_value = armed
            array[(handle_array >= stowed_value) & (handle_array < deployed)] = 'Armed/Cmd Dn'
        if mask_below_armed:
            array[handle_array == stowed_value] = 'Stowed'
            array[handle_array < stowed_value] = np.ma.masked
        else:
            array[handle_array < stowed_value] = 'Stowed'
        array[handle_array >= deployed] = 'Deployed/Cmd Up'
        return array

    @staticmethod
    def derive_from_armed_and_speedbrake(armed, spdbrk, threshold=1.0):
        '''
        Speedbrake operation from speedbrake armed and speedbrake, defaults
        to 1.0 which applies to A320 families of aircraft.
        '''
        armed = np.ma.where(armed.array == 'Armed', 'Armed/Cmd Dn', 'Stowed')
        array = np.ma.where(spdbrk.array > threshold,
                            'Deployed/Cmd Up',armed)
        return array

    @staticmethod
    def b737_speedbrake(spdbrk, handle):
        '''
        Speedbrake Handle Positions for 737-x:

            ========    ============
            Angle       Notes
            ========    ============
             0.0        Full Forward
             4.0        Armed
            24.0
            29.0
            38.0        In Flight
            40.0        Straight Up
            48.0        Full Up
            ========    ============

        Speedbrake Positions > 1 = Deployed
        '''
        if spdbrk and handle:
            # Speedbrake and Speedbrake Hbd100_speedbrakeandle available
            '''
            Speedbrake status taken from surface position. This allows
            for aircraft where the handle is inoperative, overwriting
            whatever the handle position is when the brakes themselves
            have deployed.

            It's not possible to identify when the speedbrakes are just
            armed in this case, so we take any significant motion as
            deployed.

            If there is no handle position recorded, the default 'Stowed'
            value is retained.
            '''
            armed = np.ma.where((2.0 < handle.array) & (handle.array < 15.0),
                                'Armed/Cmd Dn', 'Stowed')
            array = np.ma.where((handle.array >= 15.0) | (spdbrk.array > 1.0),
                                'Deployed/Cmd Up', armed)
        elif spdbrk and not handle:
            # Speedbrake only
            array = np.ma.where(spdbrk.array > 1.0,
                                'Deployed/Cmd Up', 'Stowed')
        elif handle and not spdbrk:
            # Speedbrake Handle only
            armed = np.ma.where((2.0 < handle.array) & (handle.array < 15.0),
                                'Armed/Cmd Dn', 'Stowed')
            array = np.ma.where(handle.array >= 15.0,
                                'Deployed/Cmd Up', armed)
        else:
            raise ValueError("Can't work without either Speedbrake or Handle")
        return array

    @classmethod
    def bd100_speedbrake(cls, handle_array, gnd_spoiler_armed_array):
        '''
        Speedbrake Handle non-zero is deployed, Spoiler Ground Armed is
        armed.
        '''
        # Default is stowed.
        array = MappedArray(np_ma_masked_zeros_like(handle_array, dtype=np.short),
                            values_mapping=cls.values_mapping)
        array[gnd_spoiler_armed_array != 'Armed'] = 'Stowed'
        array[gnd_spoiler_armed_array == 'Armed'] = 'Armed/Cmd Dn'
        array[handle_array >= 1] = 'Deployed/Cmd Up'
        return array

    @staticmethod
    def b787_speedbrake(handle):
        '''
        Speedbrake Handle Positions for 787, taken from early recordings.
        '''
        # Speedbrake Handle only
        speedbrake = np.ma.zeros(len(handle.array), dtype=np.short)
        stepped_array = step_values(handle.array, [0, 10, 20])
        # Assuming all values from 15 and above are Deployed. Typically a
        # maximum value of 60 is recorded when deployed with reverse thrust
        # whereas values of 30-60 are seen during the approach.
        speedbrake[stepped_array == 10] = 1
        speedbrake[stepped_array == 20] = 2
        return speedbrake

    @staticmethod
    def learjet_speedbrake(spdsw):
        '''
        Learjet 60XS has a switch with settings:
        0 = Retract
        4 = Extended
        7 = Armed
        6 = Partial

        Here we map thus:
            Retract = Stowed
            Armed = Armed/Cmd Dn
            Partial or Extended = Deployed/Cmd Up
        '''
        switch = spdsw.array
        speedbrake = np.ma.zeros(len(switch), dtype=np.short)
        speedbrake = np.ma.where(switch == 'Retract', 'Stowed',
                                 'Deployed/Cmd Up')
        speedbrake = np.ma.where(switch == 'Armed', 'Armed/Cmd Dn',
                                 speedbrake)
        return speedbrake

    def derive(self,
               deployed=M('Speedbrake Deployed'),
               armed=M('Speedbrake Armed'),
               handle=P('Speedbrake Handle'),
               spdbrk=P('Speedbrake'),
               spdsw=M('Speedbrake Switch'),
               gnd_spoiler_armed=M('Ground Spoiler Armed'),
               family=A('Family')):

        family_name = family.value if family else ''

        if family_name == 'B787':
            self.array = self.b787_speedbrake(handle)

        elif deployed:
            # Families include: A340, ...

            # We have a speedbrake deployed discrete. Set initial state to
            # stowed, then set armed states if available, and finally set
            # deployed state:
            array = np.ma.zeros(len(deployed.array), dtype=np.short)
            if armed:
                array[armed.array == 'Armed'] = 1
            array[deployed.array == 'Deployed'] = 2
            self.array = array

        elif 'B737' in family_name:
            self.array = self.b737_speedbrake(spdbrk, handle)

        elif family_name == 'B747':
            self.array = self.derive_from_handle(handle.array, deployed=5,
                                                 armed=1)

        elif family_name == 'B757':
            self.array = self.derive_from_handle(handle.array, deployed=25,
                                                 armed=12)

        elif family_name == 'B767':
            self.array = self.derive_from_handle(handle.array, deployed=45,
                                                 armed=12)

        elif family_name == 'B777':
            self.array = np.ma.where((handle.array >= 30.0),
                                     'Deployed/Cmd Up', 'Armed/Cmd Dn')
            self.array = np.ma.where((handle.array <= 1.0),
                                     'Stowed', self.array)

        elif family_name in ('A300', 'A310') and not spdbrk:
            # Have only seen Speedbrake Handle ,not Speedbrake parameter so
            # far for A300
            self.array = self.derive_from_handle(handle.array, deployed=10)

        elif family_name in ('A318', 'A319', 'A320', 'A321'):
            self.array = self.derive_from_armed_and_speedbrake(armed, spdbrk)
        elif family_name == 'MD-11':
            self.array = self.derive_from_armed_and_speedbrake(armed, spdbrk, threshold=10.0)
        elif family_name in ('A340', 'A350', 'A380'):
            self.array = np.ma.where((handle.array < -1.0),
                         'Armed/Cmd Dn', 'Stowed')
            self.array = np.ma.where((spdbrk.array > 5.0),
                                     'Deployed/Cmd Up', self.array)
        elif family_name == 'A330':
            if armed:
                array = np.ma.where((armed.array == 'Armed'),
                                         'Armed/Cmd Dn', 'Stowed')
            elif handle:
                array = np.ma.where((handle.array < -1.0),
                                         'Armed/Cmd Dn', 'Stowed')
            else:
                array = np.ma.zeros(len(spdbrk.array), dtype=np.short)
            self.array = np.ma.where((spdbrk.array > 5.0),
                                     'Deployed/Cmd Up', array)

        elif family_name == 'Learjet':
            self.array = self.learjet_speedbrake(spdsw)

        elif family_name == 'G-IV' and spdbrk and handle:
            # based on data seen for G450, clean handle signal with no armed position.
            self.array = np.ma.where((handle.array >= 1.0) | (spdbrk.array > 25.0),
                                'Deployed/Cmd Up', 'Stowed')
        elif family_name in ('G-IV',
                             'G-V',
                             'G-VI',
                             'Global',
                             'CL-600',
                             'Citation',
                             'Citation VLJ',
                             'BAE 146',
                             'ERJ-170/175',
                             'ERJ-190/195',
                             'Phenom 300') and spdbrk:
            array = np.ma.zeros(len(spdbrk.array), dtype=np.short)
            if armed:
                # G550 seen with recorded Speedbrake Armed parameter
                array[armed.array == 'Armed'] = 1
            # On the test aircraft SE-RDY the Speedbrake stored 0 at all
            # times and Speedbrake Handle was unresponsive with small numeric
            # variation. The Speedbrake (L) & (R) responded normally so we
            # simply accept over 30deg as deployed.
            self.array = np.ma.where(spdbrk.array > 2.0,
                                     'Deployed/Cmd Up',
                                     array)
        elif family_name in ['ERJ-170/175', 'ERJ-190/195'] and handle:
            self.array = np.ma.where(handle.array < -15.0,
                                     'Stowed',
                                     'Deployed/Cmd Up')

        elif family_name in ['Global', 'CRJ 100/200', 'ERJ-135/145',
                             'CL-600', 'G-IV'] and handle:
            # No valid data seen for this type to date....
            logger.warning(
                'SpeedbrakeSelected: algorithm for family `%s` is undecided, '
                'temporarily using speedbrake handle.', family_name)
            self.array = np_ma_masked_zeros_like(handle.array, dtype=np.short)

        elif family_name == 'BD-100':
            self.array = self.bd100_speedbrake(handle.array,
                                               gnd_spoiler_armed.array)
        else:
            raise NotImplementedError("No Speedbrake mapping for '%s'" % family_name)


class StableApproachStages(object):
    '''
    During the Descent and Approach, the following steps are assessed in turn
    to determine the aircraft stability:

    1. Gear is down
    2. Landing Flap is set
    3. Track is aligned to Runway (within 12 degrees or 30 if offset approach)
    4. Airspeed:
        - airspeed minus selected approach speed within -5 to +15 knots (for 3 secs)
        - or Vapp within -5 to +10 knots (for 3 secs)
        - or Vref within -5 to +35 knots (for 3 secs)
    5. Glideslope deviation within 1 dot
    6. Localizer deviation within 1 dot
    7. Vertical speed between -1100 and -200 fpm
    8. Engine Thrust greater than 40% N1 or 35% (A319/B787) or 1.09 EPR
       (for 10 secs) or 1.02 (A319, A320, A321)

    if all the above steps are met, the result is the declaration of:
    9. "Stable"

    Notes:

    Airspeed is relative to "Airspeed Selected" where available as this will
    account for the reference speed and any compensation for the wind speed.

    If Vapp is recorded, a more constraint airspeed threshold is applied.

    Where parameters are not monitored below a certain threshold (e.g. ILS
    below 200ft) the stability criteria just before 200ft is reached is
    continued through to landing. So if one was unstable due to ILS
    Glideslope down to 200ft, that stability is assumed to continue through
    to landing.

    TODO/REVIEW:
    ============
    * Check for 300ft limit if turning onto runway late and ignore stability
      criteria before this? Alternatively only assess criteria when heading is
      within 50.
    * Add hysteresis (3 second gliding windows for GS / LOC etc.)
    * Engine cycling check
    * Use Engine TPR for B787 instead of EPR if available.
    '''
    units = None
    values_mapping = {
        0: '-',  # All values should be masked anyway, this helps align values
        1: 'Gear Not Down',
        2: 'Not Landing Flap',
        3: 'Track Not Aligned',
        4: 'Aspd Not Stable',  # Q: Split into two Airspeed High/Low?
        5: 'GS Not Stable',
        6: 'Loc Not Stable',
        7: 'IVV Not Stable',
        8: 'Eng Thrust Not Stable',
        9: 'Stable',
    }

    align_frequency = 1

    def derive_stable_approach(self, apps, phases, gear, flap, tdev, aspd_rel,
                               aspd_minus_sel, vspd, gdev, ldev, eng_n1,
                               eng_epr, alt, vapp, family, model):

        # create an empty fully masked array
        self.array = np.ma.zeros(len(alt.array), dtype=np.short)
        self.array.mask = True

        self.repair = lambda ar, ap, method='interpolate': repair_mask(
            ar[ap], raise_entirely_masked=False, method=method)
        for approach in apps:
            # lookup descent from approach, dont zip as not guanenteed to have the same
            # number of descents and approaches
            phase = phases.get_last(within_slice=approach.slice, within_use='any')
            if not phase:
                continue
            # use Combined descent phase slice as it contains the data from
            # top of descent to touchdown (approach starts and finishes later)
            approach.slice = phase.slice

            # FIXME: approaches shorter than 10 samples will not work due to
            # the use of moving_average with a width of 10 samples.
            if approach.slice.stop - approach.slice.start < 10:
                continue
            # Restrict slice to 10 seconds after landing if we hit the ground
            gnd = index_at_value(alt.array, 0, approach.slice)
            if gnd and gnd + 10 < approach.slice.stop:
                stop = gnd + 10
            else:
                stop = approach.slice.stop
            _slice = slices_int(approach.slice.start, stop)

            altitude = self.repair(alt.array, _slice)
            index_at_50 = int(index_closest_value(altitude, 50))
            index_at_200 = int(index_closest_value(altitude, 200))

            if gear:
                #== 1. Gear Down ==
                stable = self._stable_1_gear_down(_slice, gear)
            if flap:
                #== 2. Landing Flap ==
                stable = self._stable_2_landing_flap(_slice, stable, altitude, flap)
            if tdev:
                #== 3. Track Deviation ==
                stable = self._stable_3_track_deviation(_slice, stable, approach, tdev)
            if aspd_rel or aspd_minus_sel:
                #== 4. Airspeed Relative ==
                stable = self._stable_4_airspeed_relative(_slice, stable, aspd_minus_sel, aspd_rel, vapp, index_at_50, altitude)
            if gdev:
                #== 5. Glideslope Deviation ==
                stable = self._stable_5_glideslope_deviation(_slice, stable, approach, gdev, index_at_200, altitude)
            if ldev:
                #== 6. Localizer Deviation ==
                stable = self._stable_6_localizer_deviation(_slice, stable, approach, ldev, index_at_200, altitude)
            if vspd:
                #== 7. Vertical Speed ==
                stable = self._stable_7_vertical_speed(_slice, stable, approach, vspd, index_at_50, altitude)
            if eng_epr or eng_n1:
                #== 8. Engine Thrust (N1/EPR) ==
                stable = self._stable_8_engine_thrust(_slice, stable, eng_epr, eng_n1, family, index_at_50, altitude)
            #== 9. Stable ==
            # Congratulations; whatever remains in this approach is stable!
            self.array[_slice][stable] = 9

        #endfor
        return

    def _stable_1_gear_down(self, _slice, gear):
        #== 1. Gear Down ==
        # prepare data for this appproach:
        gear_down = self.repair(gear.array, _slice, method='fill_start')
        # Assume unstable due to Gear Down at first
        self.array[_slice] = 1
        landing_gear_set = (gear_down == 'Down')
        stable = landing_gear_set.filled(True)  # assume stable (gear down)
        return stable

    def _stable_2_landing_flap(self, _slice, stable, altitude, flap):
        #== 2. Landing Flap ==
        # prepare data for this appproach:
        flap_lever = self.repair(flap.array, _slice, method='fill_start')
        # not due to landing gear so try to prove it wasn't due to Landing Flap
        self.array[_slice][stable] = 2
        # look for maximum flap used in approach below 1,000ft, otherwise
        # go-arounds can detect the start of flap retracting as the
        # landing flap.
        landing_flap = np.ma.where(altitude < 1000, flap_lever, np.ma.masked).max()
        if landing_flap is np.ma.masked:
            # try looking above 1000ft
            landing_flap = np.ma.where(altitude > 1000, flap_lever, np.ma.masked).max()

        if landing_flap is not np.ma.masked:
            landing_flap_set = (flap_lever == landing_flap)
            # assume stable (flap set)
            stable &= landing_flap_set.filled(True)
        else:
            # All landing flap is masked, assume stable
            logger.warning(
                'StableApproach: the landing flap is all masked in '
                'the approach.')
            stable &= True
        return stable

    def _stable_3_track_deviation(self, _slice, stable, approach, tdev):
        #== 3. Track Deviation ==
        # prepare data for this appproach:
        track_dev = self.repair(tdev.array, _slice)

        self.array[_slice][stable] = 3
        runway = approach.approach_runway
        if runway and runway.get('localizer', {}).get('is_offset'):
            # offset ILS Localizer or offset approach without ILS (IAN approach)
            STABLE_TRACK = 30  # degrees
        else:
            # use 12 to allow rolling a little over the 10 degrees when
            # aligning to runway.
            STABLE_TRACK = 12  # degrees
        stable_track_dev = abs(track_dev) <= STABLE_TRACK
        stable &= stable_track_dev.filled(True)  # assume stable (on track)
        return stable

    def _stable_4_airspeed_relative(self, _slice, stable, aspd_minus_sel, aspd_rel, vapp, index_at_50, altitude):
        #== 4. Airspeed Relative ==
        # prepare data for this appproach:
        if aspd_minus_sel:
            airspeed = self.repair(aspd_minus_sel.array, _slice)
        elif aspd_rel:
            airspeed = self.repair(aspd_rel.array, _slice)
        else:
            airspeed = None

        if airspeed is not None:
            self.array[_slice][stable] = 4
            if aspd_minus_sel:
                # Airspeed relative to selected speed
                if aspd_rel:
                    low_limit_airspeed = self.repair(aspd_rel.array, _slice)
                else:
                    low_limit_airspeed = airspeed
                STABLE_AIRSPEED_BELOW_REF = -5
                STABLE_AIRSPEED_ABOVE_REF = 15
            elif vapp:
                # Those aircraft which record a variable Vapp shall have more constraint thresholds
                low_limit_airspeed = airspeed
                STABLE_AIRSPEED_BELOW_REF = -5
                STABLE_AIRSPEED_ABOVE_REF = 10
            else:
                # Most aircraft record only Vref - as we don't know the wind correction be more lenient
                low_limit_airspeed = airspeed
                STABLE_AIRSPEED_BELOW_REF = -5
                STABLE_AIRSPEED_ABOVE_REF = 35

            stable_airspeed = (low_limit_airspeed >= STABLE_AIRSPEED_BELOW_REF) & (airspeed <= STABLE_AIRSPEED_ABOVE_REF)
            # extend the stability at the end of the altitude threshold through to landing
            stable_airspeed[altitude < 50] = stable_airspeed[index_at_50]
            stable &= stable_airspeed.filled(True)  # if no V Ref speed, values are masked so consider stable as one is not flying to the vref speed??
        return stable

    def _stable_5_glideslope_deviation(self, _slice, stable, approach, gdev, index_at_200, altitude):
        #== 5. Glideslope Deviation ==
        # prepare data for this appproach:
        glideslope = self.repair(gdev.array, _slice) if gdev else None  # optional
        if approach.gs_est:
            self.array[_slice][stable] = 5
            STABLE_GLIDESLOPE = 1.0  # dots
            stable_gs = (abs(glideslope) <= STABLE_GLIDESLOPE)
            # extend the stability at the end of the altitude threshold through to landing
            stable_gs[altitude < 200] = stable_gs[index_at_200]
            stable &= stable_gs.filled(False)  # masked values are usually because they are way outside of range and short spikes will have been repaired
        return stable

    def _stable_6_localizer_deviation(self, _slice, stable, approach, ldev, index_at_200, altitude):
        #== 6. Localizer Deviation ==
        # prepare data for this appproach:
        localizer = self.repair(ldev.array, _slice) if ldev else None  # optional

        if approach.gs_est and approach.loc_est:
            self.array[_slice][stable] = 6
            STABLE_LOCALIZER = 1.0  # dots
            stable_loc = (abs(localizer) <= STABLE_LOCALIZER)
            # extend the stability at the end of the altitude threshold through to landing
            stable_loc[altitude < 200] = stable_loc[index_at_200]
            stable &= stable_loc.filled(False)  # masked values are usually because they are way outside of range and short spikes will have been repaired
        return stable

    def _stable_7_vertical_speed(self, _slice, stable, approach, vspd, index_at_50, altitude, ):
        #== 7. Vertical Speed ==
        # prepare data for this appproach:
        # apply quite a large moving average to smooth over peaks and troughs
        vertical_speed = moving_average(self.repair(vspd.array, _slice), 11)
        runway = approach.approach_runway

        self.array[_slice][stable] = 7
        STABLE_VERTICAL_SPEED_MAX = -200
        STABLE_VERTICAL_SPEED_MIN = -1100
        if runway:
            gs_angle = runway.get('glideslope', {}).get('angle')
            # offset ILS Localizer or offset approach without ILS (IAN approach)
            if gs_angle is not None and gs_angle > 3:
                STABLE_VERTICAL_SPEED_MIN = -1500
        stable_vert = (vertical_speed >= STABLE_VERTICAL_SPEED_MIN) & (vertical_speed <= STABLE_VERTICAL_SPEED_MAX)
        # extend the stability at the end of the altitude threshold through to landing
        stable_vert[altitude < 50] = stable_vert[index_at_50]
        stable &= stable_vert.filled(True)
        return stable

    def _stable_8_engine_thrust(self, _slice, stable, eng_epr, eng_n1, family, index_at_50, altitude):
        #== 8. Engine Thrust (N1/EPR) ==
        if eng_epr:
            # use EPR if available
            engine = self.repair(eng_epr.array, _slice)
        else:
            engine = self.repair(eng_n1.array, _slice)

        self.array[_slice][stable] = 8
        # Patch this value depending upon aircraft type
        if eng_epr:
            if family and family.value in ('A319', 'A320', 'A321'):
                STABLE_EPR_MIN = 1.02  # Ratio
            else:
                STABLE_EPR_MIN = 1.09  # Ratio
            stable_engine = (engine >= STABLE_EPR_MIN)
        else:
            if family and family.value in ('B787', 'A319'):
                STABLE_N1_MIN = 35  # %
            else:
                STABLE_N1_MIN = 40  # %
            stable_engine = (engine >= STABLE_N1_MIN)
        # extend the stability at the end of the altitude threshold through to landing
        stable_engine[altitude < 50] = stable_engine[index_at_50]
        stable &= stable_engine.filled(True)
        return stable


class StableApproach(MultistateDerivedParameterNode, StableApproachStages):
    units = None

    @classmethod
    def can_operate(cls, available):
        # Many parameters are optional dependencies
        deps = ['Approach Information', 'Descent', 'Gear Down', 'Flap',
                'Vertical Speed', 'Altitude AAL',]
        return all_of(deps, available) and (
            'Eng (*) N1 Avg For 10 Sec' in available or
            'Eng (*) EPR Avg For 10 Sec' in available)

    def derive(self,
               apps=A('Approach Information'),
               phases=S('Descent'),
               gear=M('Gear Down'),
               flap=M('Flap'),
               tdev=P('Track Deviation From Runway'),
               aspd_rel=P('Airspeed Relative For 3 Sec'),
               aspd_minus_sel=P('Airspeed Minus Airspeed Selected For 3 Sec'),
               vspd=P('Vertical Speed'),
               gdev=P('ILS Glideslope'),
               ldev=P('ILS Localizer'),
               eng_n1=P('Eng (*) N1 Avg For 10 Sec'),
               eng_epr=P('Eng (*) EPR Avg For 10 Sec'),
               alt=P('Altitude AAL'),
               vapp=P('Vapp'),
               family=A('Family'),
               model=A('Model')):
        self.derive_stable_approach(apps, phases, gear, flap, tdev, aspd_rel,
                                    aspd_minus_sel, vspd, gdev, ldev, eng_n1,
                                    eng_epr, alt, vapp, family, model)


class StableApproachExcludingEngThrust(MultistateDerivedParameterNode, StableApproachStages):
    units = None

    @classmethod
    def can_operate(cls, available):
        # Many parameters are optional dependencies
        deps = ['Approach Information', 'Descent', 'Gear Down', 'Flap',
                'Vertical Speed', 'Altitude AAL',]
        return all_of(deps, available)

    def derive(self,
               apps=A('Approach Information'),
               phases=S('Descent'),
               gear=M('Gear Down'),
               flap=M('Flap'),
               tdev=P('Track Deviation From Runway'),
               aspd_rel=P('Airspeed Relative For 3 Sec'),
               aspd_minus_sel=P('Airspeed Minus Airspeed Selected For 3 Sec'),
               vspd=P('Vertical Speed'),
               gdev=P('ILS Glideslope'),
               ldev=P('ILS Localizer'),
               alt=P('Altitude AAL'),
               vapp=P('Vapp'),
               model=A('Model')):
        self.derive_stable_approach(apps, phases, gear, flap, tdev, aspd_rel,
                                    aspd_minus_sel, vspd, gdev, ldev, None,
                                    None, alt, vapp, None, model)



"""
class StickShaker(MultistateDerivedParameterNode):
    '''
    This accounts for the different types of stick shaker system. Where two
    systems are recorded the results are ORed to make a single parameter which
    operates in response to either system triggering. Hence the removal of
    automatic alignment of the signals.
    '''

    align = False
    units = None
    values_mapping = {0: '-', 1: 'Shake'}

    @classmethod
    def can_operate(cls, available):
        return ('Stick Shaker (L)' in available or \
                'Shaker Activation' in available)

    def derive(self, shake_l=M('Stick Shaker (L)'),
               shake_r=M('Stick Shaker (R)'),
               shake_act=M('Shaker Activation')):
        if shake_l and shake_r:
            self.array = np.ma.logical_or(shake_l.array, shake_r.array)
            self.frequency , self.offset = shake_l.frequency, shake_l.offset

        elif shake_l:
            # Named (L) but in fact (L) and (R) are or'd together at the DAU.
            self.array, self.frequency, self.offset = \
                shake_l.array, shake_l.frequency, shake_l.offset

        elif shake_act:
            self.array, self.frequency, self.offset = \
                shake_act.array, shake_act.frequency, shake_act.offset

        else:
            raise NotImplementedError
"""


class ThrustReversers(MultistateDerivedParameterNode):
    '''
    A single parameter with multi-state mapping as below.
    '''

    # We are interested in all stowed, all deployed or any other combination.
    # The mapping "In Transit" is used for anything other than the fully
    # established conditions, so for example one locked and the other not is
    # still treated as in transit.
    units = None
    values_mapping = {0: 'Stowed', 1: 'In Transit', 2: 'Deployed'}

    @classmethod
    def can_operate(cls, available):
        return any_of((
            'Eng (1) Thrust Reverser (L) Deployed',
            'Eng (1) Thrust Reverser (R) Deployed',
            'Eng (2) Thrust Reverser (L) Deployed',
            'Eng (2) Thrust Reverser (R) Deployed',
            'Eng (3) Thrust Reverser (L) Deployed',
            'Eng (3) Thrust Reverser (R) Deployed',
            'Eng (4) Thrust Reverser (L) Deployed',
            'Eng (4) Thrust Reverser (R) Deployed',
        ), available) or any_of((
            'Eng (1) Thrust Reverser Deployed',
            'Eng (2) Thrust Reverser Deployed',
            'Eng (3) Thrust Reverser Deployed',
            'Eng (4) Thrust Reverser Deployed',
        ), available) or any_of((
            'Eng (1) Thrust Reverser',
            'Eng (2) Thrust Reverser',
            'Eng (3) Thrust Reverser',
            'Eng (4) Thrust Reverser',
        ), available)

    def derive(self,
               e1_dep_all=M('Eng (1) Thrust Reverser Deployed'),
               e1_dep_lft=M('Eng (1) Thrust Reverser (L) Deployed'),
               e1_dep_rgt=M('Eng (1) Thrust Reverser (R) Deployed'),
               e1_ulk_all=M('Eng (1) Thrust Reverser Unlocked'),
               e1_ulk_lft=M('Eng (1) Thrust Reverser (L) Unlocked'),
               e1_ulk_rgt=M('Eng (1) Thrust Reverser (R) Unlocked'),
               e1_tst_all=M('Eng (1) Thrust Reverser In Transit'),
               e2_dep_all=M('Eng (2) Thrust Reverser Deployed'),
               e2_dep_lft=M('Eng (2) Thrust Reverser (L) Deployed'),
               e2_dep_rgt=M('Eng (2) Thrust Reverser (R) Deployed'),
               e2_ulk_all=M('Eng (2) Thrust Reverser Unlocked'),
               e2_ulk_lft=M('Eng (2) Thrust Reverser (L) Unlocked'),
               e2_ulk_rgt=M('Eng (2) Thrust Reverser (R) Unlocked'),
               e2_tst_all=M('Eng (2) Thrust Reverser In Transit'),
               e3_dep_all=M('Eng (3) Thrust Reverser Deployed'),
               e3_dep_lft=M('Eng (3) Thrust Reverser (L) Deployed'),
               e3_dep_rgt=M('Eng (3) Thrust Reverser (R) Deployed'),
               e3_ulk_all=M('Eng (3) Thrust Reverser Unlocked'),
               e3_ulk_lft=M('Eng (3) Thrust Reverser (L) Unlocked'),
               e3_ulk_rgt=M('Eng (3) Thrust Reverser (R) Unlocked'),
               e3_tst_all=M('Eng (3) Thrust Reverser In Transit'),
               e4_dep_all=M('Eng (4) Thrust Reverser Deployed'),
               e4_dep_lft=M('Eng (4) Thrust Reverser (L) Deployed'),
               e4_dep_rgt=M('Eng (4) Thrust Reverser (R) Deployed'),
               e4_ulk_all=M('Eng (4) Thrust Reverser Unlocked'),
               e4_ulk_lft=M('Eng (4) Thrust Reverser (L) Unlocked'),
               e4_ulk_rgt=M('Eng (4) Thrust Reverser (R) Unlocked'),
               e4_tst_all=M('Eng (4) Thrust Reverser In Transit'),
               e1_status=M('Eng (1) Thrust Reverser'),
               e2_status=M('Eng (2) Thrust Reverser'),
               e3_status=M('Eng (3) Thrust Reverser'),
               e4_status=M('Eng (4) Thrust Reverser')):

        deployed_params = (e1_dep_all, e1_dep_lft, e1_dep_rgt, e2_dep_all,
                           e2_dep_lft, e2_dep_rgt, e3_dep_all, e3_dep_lft,
                           e3_dep_rgt, e4_dep_all, e4_dep_lft, e4_dep_rgt,
                           e1_status, e2_status, e3_status, e4_status)

        deployed_stack = vstack_params_where_state(*[(d, 'Deployed') for d in deployed_params])

        unlocked_params = (e1_ulk_all, e1_ulk_lft, e1_ulk_rgt, e2_ulk_all,
                           e2_ulk_lft, e2_ulk_rgt, e3_ulk_all, e3_ulk_lft,
                           e3_ulk_rgt, e4_ulk_all, e4_ulk_lft, e4_ulk_rgt)

        array = np_ma_zeros_like(deployed_stack[0], dtype=np.short)
        stacks = [deployed_stack]

        if any(unlocked_params):
            unlocked_stack = vstack_params_where_state(*[(p, 'Unlocked') for p in unlocked_params])
            array = np.ma.where(unlocked_stack.any(axis=0), 1, array)
            stacks.append(unlocked_stack)

        array = np.ma.where(deployed_stack.any(axis=0), 1, array)
        array = np.ma.where(deployed_stack.all(axis=0), 2, array)

        # update with any transit params
        if any((e1_tst_all, e2_tst_all, e3_tst_all, e4_tst_all)):
            transit_stack = vstack_params_where_state(
                (e1_tst_all, 'In Transit'), (e2_tst_all, 'In Transit'),
                (e3_tst_all, 'In Transit'), (e4_tst_all, 'In Transit'),
                (e1_status, 'In Transit'), (e2_status, 'In Transit'),
                (e3_status, 'In Transit'), (e4_status, 'In Transit'),
            )
            array = np.ma.where(transit_stack.any(axis=0), 1, array)
            stacks.append(transit_stack)

        mask_stack = np.ma.concatenate(stacks, axis=0)

        # mask indexes with greater than 50% masked values
        mask = np.ma.where(mask_stack.mask.sum(axis=0).astype(float) / len(mask_stack) * 100 > 50, 1, 0)
        self.array = array
        self.array.mask = mask


class ThrustReversersEffective(MultistateDerivedParameterNode):
    units = None
    values_mapping = {0: '-', 1: 'Effective'}

    @classmethod
    def can_operate(cls, available):
        power_ok = any_of(('Eng (*) EPR Max', 'Eng (*) N1 Max'), available)
        return power_ok and all_of(('Thrust Reversers', 'Landing'), available)

    def derive(self,
               tr=M('Thrust Reversers'),
               eng_epr=P('Eng (*) EPR Max'),  # must come before N1 where available
               eng_n1=P('Eng (*) N1 Max'),
               landings=S('Landing')):

        if eng_epr:
            power = eng_epr
            threshold = REVERSE_THRUST_EFFECTIVE_EPR
        else:
            power = eng_n1
            threshold = REVERSE_THRUST_EFFECTIVE_N1

        self.array = np_ma_zeros_like(tr.array)
        high_power = np.ma.masked_less(power.array, threshold)
        high_power_slices = np.ma.clump_unmasked(high_power)
        for landing in landings:
            high_power_landing_slices = slices_and(high_power_slices, [landing.slice])
            effective_slices = clump_multistate(tr.array, 'Deployed', high_power_landing_slices)
            for sl in effective_slices:
                self.array[sl] = 'Effective'


class TAWSAlert(MultistateDerivedParameterNode):
    '''
    Merging all available TAWS alert signals into a single parameter for
    subsequent monitoring.
    '''
    name = 'TAWS Alert'
    units = None
    values_mapping = {0: '-', 1: 'Alert'}

    @classmethod
    def can_operate(cls, available):
        return any_of(['TAWS Caution Terrain',
                       'TAWS Caution',
                       'TAWS Dont Sink',
                       'TAWS Glideslope',
                       'TAWS Predictive Windshear',
                       'TAWS Pull Up',
                       'TAWS Sink Rate',
                       'TAWS Terrain',
                       'TAWS Terrain Caution',
                       'TAWS Terrain Pull Up',
                       'TAWS Terrain Warning',
                       'TAWS Too Low Flap',
                       'TAWS Too Low Gear',
                       'TAWS Too Low Terrain',
                       'TAWS Windshear Warning',
                       ],
                      available)

    def derive(self, airs=S('Airborne'),
               taws_caution_terrain=M('TAWS Caution Terrain'),
               taws_caution=M('TAWS Caution'),
               taws_dont_sink=M('TAWS Dont Sink'),
               taws_glideslope=M('TAWS Glideslope'),
               taws_predictive_windshear=M('TAWS Predictive Windshear'),
               taws_pull_up=M('TAWS Pull Up'),
               taws_sink_rate=M('TAWS Sink Rate'),
               taws_terrain_pull_up=M('TAWS Terrain Pull Up'),
               taws_terrain_caution=M('TAWS Terrain Caution'),
               taws_terrain_warning=M('TAWS Terrain Warning'),
               taws_terrain=M('TAWS Terrain'),
               taws_too_low_flap=M('TAWS Too Low Flap'),
               taws_too_low_gear=M('TAWS Too Low Gear'),
               taws_too_low_terrain=M('TAWS Too Low Terrain'),
               taws_windshear_warning=M('TAWS Windshear Warning')):

        params_state = vstack_params_where_state(
            (taws_caution_terrain, 'Caution'),
            (taws_caution, 'Caution'),
            (taws_dont_sink, 'Warning'),
            (taws_glideslope, 'Warning'),
            (taws_predictive_windshear, 'Caution'),
            (taws_predictive_windshear, 'Warning'),
            (taws_pull_up, 'Warning'),
            (taws_sink_rate, 'Warning'),
            (taws_terrain_pull_up, 'Warning'),
            (taws_terrain_caution, 'Warning'),
            (taws_terrain_warning, 'Warning'),
            (taws_terrain, 'Warning'),
            (taws_too_low_flap, 'Warning'),
            (taws_too_low_gear, 'Warning'),
            (taws_too_low_terrain, 'Warning'),
            (taws_windshear_warning, 'Warning'),
        )
        res = params_state.any(axis=0)

        self.array = np_ma_masked_zeros_like(params_state[0], dtype=np.short)
        if airs:
            for air in airs:
                self.array[air.slice] = res[air.slice]


class TAWSDontSink(MultistateDerivedParameterNode):
    name = 'TAWS Dont Sink'
    units = None
    values_mapping = {0: '-', 1: 'Warning'}

    @classmethod
    def can_operate(cls, available):
        return ('TAWS (L) Dont Sink' in available) or \
               ('TAWS (R) Dont Sink' in available)

    def derive(self, taws_l_dont_sink=M('TAWS (L) Dont Sink'),
               taws_r_dont_sink=M('TAWS (R) Dont Sink')):
        self.array = vstack_params_where_state(
            (taws_l_dont_sink, 'Warning'),
            (taws_r_dont_sink, 'Warning'),
        ).any(axis=0)


class TAWSGlideslopeCancel(MultistateDerivedParameterNode):
    name = 'TAWS Glideslope Cancel'
    units = None
    values_mapping = {0: '-', 1: 'Cancel'}

    @classmethod
    def can_operate(cls, available):
        return ('TAWS (L) Glideslope Cancel' in available) or \
               ('TAWS (R) Glideslope Cancel' in available)

    def derive(self, taws_l_gs=M('TAWS (L) Glideslope Cancel'),
               taws_r_gs=M('TAWS (R) Glideslope Cancel')):
        self.array = vstack_params_where_state(
            (taws_l_gs, 'Cancel'),
            (taws_r_gs, 'Cancel'),
        ).any(axis=0)


class TAWSTooLowGear(MultistateDerivedParameterNode):
    name = 'TAWS Too Low Gear'
    units = None
    values_mapping = {0: '-', 1: 'Warning'}

    @classmethod
    def can_operate(cls, available):
        return ('TAWS (L) Too Low Gear' in available) or \
               ('TAWS (R) Too Low Gear' in available)

    def derive(self, taws_l_gear=M('TAWS (L) Too Low Gear'),
               taws_r_gear=M('TAWS (R) Too Low Gear')):
        self.array = vstack_params_where_state(
            (taws_l_gear, 'Warning'),
            (taws_r_gear, 'Warning'),
        ).any(axis=0)


class TAWSTerrainCaution(MultistateDerivedParameterNode):
    '''
    Merge Captain and FO TAWS Terrain Caution parameters.
    '''

    name = 'TAWS Terrain Caution'
    units = None
    values_mapping = {0: '-', 1: 'Caution'}

    def derive(self, taws_terrain_cpt=M('TAWS Terrain Caution (Capt)'),
               taws_terrain_fo=M('TAWS Terrain Caution (FO)'),):

        self.array = vstack_params_where_state(
            (taws_terrain_cpt, 'Caution'),
            (taws_terrain_fo, 'Caution'),
        ).any(axis=0)


class TAWSTerrainWarning(MultistateDerivedParameterNode):
    '''
    Merge Captain and FO TAWS Terrain Warning parameters.
    '''

    name = 'TAWS Terrain Warning'
    units = None
    values_mapping = {0: '-', 1: 'Warning'}

    def derive(self, taws_terrain_cpt=M('TAWS Terrain Warning (Capt)'),
               taws_terrain_fo=M('TAWS Terrain Warning (FO)'),):

        self.array = vstack_params_where_state(
            (taws_terrain_cpt, 'Warning'),
            (taws_terrain_fo, 'Warning'),
        ).any(axis=0)


class TakeoffConfigurationWarning(MultistateDerivedParameterNode):
    '''
    Merging all available Takeoff Configuration Warning signals into a single
    parameter for subsequent monitoring.
    '''
    units = None
    values_mapping = {0: '-', 1: 'Warning'}

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self, stabilizer=M('Takeoff Configuration Stabilizer Warning'),
               parking_brake=M('Takeoff Configuration Parking Brake Warning'),
               flap=M('Takeoff Configuration Flap Warning'),
               gear=M('Takeoff Configuration Gear Warning'),
               ap=M('Takeoff Configuration AP Warning'),
               ail=M('Takeoff Configuration Aileron Warning'),
               rudder=M('Takeoff Configuration Rudder Warning'),
               spoiler=M('Takeoff Configuration Spoiler Warning')):
        params_state = vstack_params_where_state(
            (stabilizer, 'Warning'),
            (parking_brake, 'Warning'),
            (flap, 'Warning'),
            (gear, 'Warning'),
            (ap, 'Warning'),
            (ail, 'Warning'),
            (rudder, 'Warning'),
            (spoiler, 'Warning'))
        self.array = params_state.any(axis=0)


class TCASFailure(MultistateDerivedParameterNode):
    name = 'TCAS Failure'
    units = None
    values_mapping = {0: '-', 1: 'Failed'}

    @classmethod
    def can_operate(cls, available):
        return ('TCAS (L) Failure' in available) or \
               ('TCAS (R) Failure' in available)

    def derive(self, tcas_l_failure=M('TCAS (L) Failure'),
               tcas_r_failure=M('TCAS (R) Failure')):
        self.array = vstack_params_where_state(
            (tcas_l_failure, 'Failed'),
            (tcas_r_failure, 'Failed'),
        ).any(axis=0)


class TCASRA(MultistateDerivedParameterNode):
    name = 'TCAS RA'
    units = None
    values_mapping = {0: '-', 1: 'RA'}

    @classmethod
    def can_operate(cls, available):
        return ('TCAS RA (1)' in available) or \
               ('TCAS RA (2)' in available)

    def derive(self, tcas_1=M('TCAS RA (1)'),
               tcas_2=M('TCAS RA (2)')):
        self.array = vstack_params_where_state(
            (tcas_1, 'RA'),
            (tcas_2, 'RA'),
        ).any(axis=0)


class SpeedControl(MultistateDerivedParameterNode):
    units = None
    values_mapping = {0: 'Manual', 1: 'Auto'}

    @classmethod
    def can_operate(cls, available):

        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               sc0a=M('Speed Control Auto'),
               sc0m=M('Speed Control Manual'),
               sc1a=M('Speed Control (1) Auto'),
               sc1m=M('Speed Control (1) Manual'),
               sc2a=M('Speed Control (2) Auto'),
               sc2m=M('Speed Control (2) Manual')):

        self.array = vstack_params_where_state(
            (sc0a, 'Auto'), (sc0m, 'Auto'),
            (sc1a, 'Auto'), (sc1m, 'Auto'),
            (sc2a, 'Auto'), (sc2m, 'Auto'),
        ).any(axis=0).astype(np.int)


class Transmitting(MultistateDerivedParameterNode):
    '''
    Whenever the HF, VHF or Satcom transmits are used, this parameter outputs Transmit.
    '''
    units = None
    values_mapping = {0: '-', 1: 'Transmit'}

    @classmethod
    def can_operate(cls, available):
        return any(d in available for d in cls.get_dependency_names())

    def derive(self,
               hf=M('Key HF'),
               hf1=M('Key HF (1)'),
               hf2=M('Key HF (2)'),
               hf3=M('Key HF (3)'),
               hf1_capt=M('Key HF (1) (Capt)'),
               hf2_capt=M('Key HF (2) (Capt)'),
               hf3_capt=M('Key HF (3) (Capt)'),
               hf1_fo=M('Key HF (1) (FO)'),
               hf2_fo=M('Key HF (2) (FO)'),
               hf3_fo=M('Key HF (3) (FO)'),
               sc=M('Key Satcom'),
               sc1=M('Key Satcom (1)'),
               sc2=M('Key Satcom (2)'),
               vhf=M('Key VHF'),
               vhf1=M('Key VHF (1)'),
               vhf2=M('Key VHF (2)'),
               vhf3=M('Key VHF (3)'),
               vhf1_capt=M('Key VHF (1) (Capt)'),
               vhf2_capt=M('Key VHF (2) (Capt)'),
               vhf3_capt=M('Key VHF (3) (Capt)'),
               vhf1_fo=M('Key VHF (1) (FO)'),
               vhf2_fo=M('Key VHF (2) (FO)'),
               vhf3_fo=M('Key VHF (3) (FO)')):
        stacked = vstack_params_where_state(*((p, 'Keyed') for p in (
            hf, hf1, hf2, hf3, hf1_capt, hf2_capt, hf3_capt,
            hf1_fo, hf2_fo, hf3_fo, sc, sc1, sc2, vhf, vhf1, vhf2, vhf3,
            vhf1_capt, vhf2_capt, vhf3_capt, vhf1_fo, vhf2_fo, vhf3_fo,
        )))
        self.array = stacked.any(axis=0)
        self.array.mask = stacked.mask.any(axis=0)
