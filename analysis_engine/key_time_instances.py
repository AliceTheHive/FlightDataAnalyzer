from __future__ import print_function

import numpy as np
import six

from math import ceil, floor
from operator import itemgetter

from flightdatautilities import units as ut
from flightdatautilities.geometry import great_circle_distance__haversine

from analysis_engine.node import (
    A, M, P, S, KTI, App,
    helicopter, KeyTimeInstanceNode
)

from analysis_engine.library import (
    align,
    all_deps,
    all_of,
    any_of,
    coreg,
    find_edges_on_state_change,
    find_toc_tod,
    first_valid_sample,
    hysteresis,
    index_at_value,
    is_index_within_slice,
    is_index_within_slices,
    last_valid_sample,
    max_value,
    min_value,
    minimum_unmasked,
    moving_average,
    np_ma_masked_zeros_like,
    peak_curvature,
    rate_of_change,
    rate_of_change_array,
    repair_mask,
    runs_of_ones,
    slices_and,
    slices_and_not,
    slice_duration,
    slices_not,
    slices_overlap,
    slices_remove_small_gaps,
    value_at_index
)



from analysis_engine.settings import (
    CLIMB_THRESHOLD,
    HYSTERESIS_ENG_START_STOP,
    CORE_START_SPEED,
    CORE_STOP_SPEED,
    NAME_VALUES_DISTANCE,
    MIN_FAN_RUNNING,
    NAME_VALUES_CLIMB,
    NAME_VALUES_DESCENT,
    NAME_VALUES_ENGINE,
    NAME_VALUES_LEVER,
    NAME_VALUES_RANGES,
    TAKEOFF_ACCELERATION_THRESHOLD,
    TRANSITION_ALTITUDE,
    VERTICAL_SPEED_FOR_LIFTOFF,
)

from flightdatautilities.numpy_utils import slices_int


def sorted_valid_list(x):
    '''
    For list x, remove None and nan fields and return sorted list.
    Used in Liftoff and Touchdown algorithms.
    '''
    index_list = []
    for i in range(len(x)):
        if x[i] and not np.isnan(x[i]):
            index_list.append(x[i])
    return sorted(index_list)


class BottomOfDescent(KeyTimeInstanceNode):
    '''
    Bottom of a descent phase, which may be a go-around, touch and go or landing.
    '''
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               ccd=S('Climb Cruise Descent')):
        for ccd_phase in ccd:
            ccd_slice = ccd_phase.slice
            # If this slice ended in mid-cruise, the ccd slice will end in
            # None if passed in directly or be the duration if using cached
            # params from process flight
            if ccd_slice.stop is None or ccd_slice.stop == len(alt_std.array):
                continue
            self.create_kti(ccd_slice.stop)


# TODO: Determine an altitude peak per climb.
class AltitudePeak(KeyTimeInstanceNode):
    '''
    Determines the peak value of altitude above airfield level which is used to
    correctly determine the splitting point when deriving the Altitude QNH
    parameter.
    '''

    def derive(self, alt_aal=P('Altitude AAL')):
        '''
        '''
        self.create_kti(np.ma.argmax(np.ma.abs(np.ma.diff(alt_aal.array))))


##############################################################################
# Automated Systems


class APEngagedSelection(KeyTimeInstanceNode):
    '''
    AP Engaged is defined as the Autopilot entering the Engaged state.

    This works for simplex, duplex or triplex engagement options, which are
    defined by the AP Channels Engaged parameter.
    '''

    name = 'AP Engaged Selection'

    def derive(self, ap=M('AP Engaged'), phase=S('Fast')):
        # TODO: Use a phase that includes on ground too, say Acceleration
        # Start before liftoff to Turn off Runway after touchdown.
        self.create_ktis_on_state_change(
            'Engaged',
            ap.array,
            change='entering',
            phase=phase
        )


class APDisengagedSelection(KeyTimeInstanceNode):
    '''
    AP Disengaged is defined as the Autopilot leaving the Engaged state.

    This works for simplex, duplex or triplex engagement options, which are
    defined by the AP Channels Engaged parameter.
    '''

    name = 'AP Disengaged Selection'

    def derive(self, ap=M('AP Engaged'), phase=S('Fast')):
        # TODO: Use a phase that includes on ground too, say Acceleration
        # Start before liftoff to Turn off Runway after touchdown.
        self.create_ktis_on_state_change(
            'Engaged',
            ap.array,
            change='leaving',
            phase=phase
        )


class ATEngagedSelection(KeyTimeInstanceNode):
    '''

    '''

    name = 'AT Engaged Selection'

    def derive(self, at=M('AT Engaged'), phase=S('Airborne')):
        # TODO: Use a phase that includes on ground too, say Acceleration
        # Start before liftoff to Turn off Runway after touchdown.
        self.create_ktis_on_state_change(
            'Engaged',
            at.array,
            change='entering',
            phase=phase
        )


class ATDisengagedSelection(KeyTimeInstanceNode):
    '''
    '''

    name = 'AT Disengaged Selection'

    def derive(self, at=P('AT Engaged'), phase=S('Airborne')):
        # TODO: Use a phase that includes on ground too, say Acceleration
        # Start before liftoff to Turn off Runway after touchdown.
        self.create_ktis_on_state_change(
            'Engaged',
            at.array,
            change='leaving',
            phase=phase
        )


##############################################################################


class Transmit(KeyTimeInstanceNode):
    '''
    Whenever the HF, VHF or Satcom transmits are used, this KTI is triggered.
    '''

    def derive(self, transmitting=M('Transmitting')):
        self.create_ktis_on_state_change(
            'Transmit',
            transmitting.array,
            change='entering'
        )


class ClimbStart(KeyTimeInstanceNode):
    '''
    Creates KTIs where the aircraft transitions through %dft
    ''' % CLIMB_THRESHOLD

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'), liftoffs=KTI('Liftoff'),
               tocs=KTI('Top Of Climb')):

        to_scan = [[l.index, 'lift'] for l in liftoffs] + \
            [[l.index, 'toc'] for l in tocs]
        to_scan = sorted(to_scan, key=itemgetter(0))
        for i in range(len(to_scan)-1):
            if to_scan[i][1]=='lift' and to_scan[i+1][1]=='toc':
                index = index_at_value(alt_aal.array, CLIMB_THRESHOLD, slice(to_scan[i][0], to_scan[i+1][0]))
                if index:
                    self.create_kti(index)


class ClimbAccelerationStart(KeyTimeInstanceNode):
    '''
    TODO review description
    Creates KTI on first signs of acceleration.

    In order of preference it will be determined by:
    - A significant increase of Airspeed Selected with the flaps not set to 0
    - A significant rise in Airspeed prior to the first flap retraction below 5,000 ft
    - A change in Throttle Lever position for jet engines below 4,000 ft
    - A change in Eng Np Max for propeller engines below 4,000 ft
    - Defaults to 800 ft for jets and 400 ft for prop

    Alignment is performed manually! This allows to check actual params frequency.

    Dynamically create rate_of_change width from the parameter's frequency to
    avoid errors. Larger widths flatten the rate of change result.
    '''
    align = False

    @classmethod
    def can_operate(cls, available, eng_type=A('Engine Propulsion')):
        spd_sel = 'Airspeed Selected' in available
        spd = any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available) and \
              'Airspeed' in available
        jet = (eng_type and eng_type.value == 'JET' and
               'Throttle Levers' in available)
        prop = (eng_type and eng_type.value == 'PROP' and
                'Eng (*) Np Max' in available)
        alt = all_of(('Engine Propulsion', 'Altitude AAL For Flight Phases'), available)
        return all_of(('Initial Climb', 'Altitude When Climbing'), available) and \
               (spd_sel or spd or jet or prop or alt) and \
               not (eng_type and eng_type.value == 'ROTOR')

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               initial_climbs=S('Initial Climb'),
               alt_climbing=KTI('Altitude When Climbing'),
               spd_sel=P('Airspeed Selected'),
               spd_tgt=P('Airspeed Target'),
               spd_sel_fmc=P('Airspeed Selected (FMC)'),
               eng_type=A('Engine Propulsion'),
               eng_np=P('Eng (*) Np Max'),
               throttle=P('Throttle Levers'),
               spd=P('Airspeed'),
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               family=A('Family')):

        if not initial_climbs.get_first():
            return
        flap = flap_lever or flap_synth

        if self.derive_from_spd_sel(spd_tgt, spd_sel, initial_climbs, alt_climbing, spd_sel_fmc, flap, family):
            return

        if self.derive_from_spd_and_flap(flap, spd, initial_climbs, alt_climbing):
            return

        if self.derive_from_throttle(eng_type, throttle, initial_climbs, alt_climbing):
            return

        if self.derive_from_eng_np(eng_type, eng_np, initial_climbs, alt_climbing):
            return

        self.derive_from_alt_aal(alt_aal, eng_type, initial_climbs)

    def derive_from_spd_sel(self, spd_tgt, spd_sel, initial_climbs, alt_climbing, spd_sel_fmc, flap, family):
        if spd_sel is None:
            return False
        if spd_sel.frequency < 0.125:
            return False
        if flap:
            flap = flap.get_aligned(spd_sel)
        # Use first Airspeed Selected change in Initial Climb up to 4000 Ft
        _slice = slices_int(initial_climbs.get_aligned(spd_sel).get_first().slice)
        climbing_4000 = alt_climbing.get_aligned(spd_sel).get(name='4000 Ft Climbing').get_first()
        _slice = slice(_slice.start, int(climbing_4000.index) if climbing_4000 else _slice.stop)

        def index_at_first_spd_sel_change(spd_refs, _slice, flap):
            '''
            Generator that produces for each speed references the index, frequency and
            offset where the speed  jumped by at least 8 kts for a minimum of 5
            consecutive samples. This allows to filter out any short term changes.

            In case of masked data, it will determine the index of the first valid sample
            which had an increase of 8 kts for 5 samples.
            '''
            for spd_ref in spd_refs:
                if spd_ref is None:
                    continue
                array = spd_ref.array[_slice]
                if array.size < 6 or array.mask.all():
                    # Size must be at least 6 samples as the np.diff will produce one
                    # less value and we want to observe a change over 5 samples.
                    continue
                array = repair_mask(
                    array, frequency=spd_ref.hz, repair_duration=300,
                    copy=True, method='fill_start'
                )
                diff = np.ma.diff(array)
                # The threshold is 8 kt per samples, except for high frequency params
                # where we want to guarantee a threshold of 8 kt per seconds.
                diff_threshold = 8 / max(1, spd_ref.hz)
                # We want to find where the diff is above threshold and the sum of 5
                # consecutive samples is still above our threshold
                diff_sliding_window = np.lib.stride_tricks.as_strided(
                    diff, shape=(diff.size-5, 5),
                    strides=(diff.strides[0], diff.strides[0]),
                    writeable=False
                )
                # Make diff the same length as diff_sliding_window
                diff = diff[:-5]
                # Find the indices of significant increase. nonzero returns a tuple
                # with a single item as we have only 1D data.
                increase_idxs, *_ = np.ma.nonzero(diff > diff_threshold)
                # Sum over 5 consecutive samples wherever we had a significant increase
                spd_increase = np.ma.sum(diff_sliding_window[increase_idxs], axis=1)
                # Find which sliding window sum stayed above threshold, ie. the spd
                # stayed at least 8 kt above its original value.
                long_term_increases, *_ = np.ma.nonzero(spd_increase > diff_threshold)
                # Get the diff indices which were sustained for 5 samples
                indices = increase_idxs[long_term_increases]

                if indices.size:
                    index = nearest_to_flap_retraction(indices, _slice, flap)
                    yield index, spd_ref.frequency, spd_ref.offset

        def nearest_to_flap_retraction(indices, _slice, flap):
            '''
            Find the index in indices which is the nearest to the first flap retraction
            within the _slice.

            The nearest index must be before the first flap retraction.
            If for any reasons, no index before the first flap retraction can be found,
            we return the first index in indices.
            '''
            if not flap:
                return indices[0]
            flap_array = flap.array[_slice]
            flap_retraction = find_edges_on_state_change(
                flap.array[int(_slice.start)], flap_array, change='leaving',
            )
            if not flap_retraction:
                return indices[0]

            flap_retraction = int(flap_retraction[0])
            if flap_array.raw[flap_retraction] < flap_array.raw[flap_retraction + 1]:
                return indices[0]

            dist = flap_retraction - indices
            dist = np.where(dist < 0, np.inf, dist)
            index = indices[np.argmin(dist)]
            return index


        # If Airspeed Target available for B787, ignore the others.
        if family and family.value=='B787' and spd_tgt and not spd_tgt.array.mask.all():
            spd_tgt = spd_tgt.get_aligned(spd_sel)
            spd_refs = [spd_tgt]
        else:
            # Align to spd_sel
            spd_sel_fmc = spd_sel_fmc.get_aligned(spd_sel) if spd_sel_fmc else None
            spd_refs = [spd_sel, spd_sel_fmc]

        try:
            index, frequency, offset = min(
                index_at_first_spd_sel_change(spd_refs, _slice, flap)
            )
        except ValueError:
            # No index was found
            return False

        index = index + (_slice.start or 0)
        if flap:
            # Make sure Airpseed Selected increased with some flaps selected
            if flap.array[int(index)] in ('0', 'Lever 0'):
                return False

        self.frequency = frequency
        self.offset = offset
        self.create_kti(index)
        return True

    def derive_from_spd_and_flap(self, flap, spd, initial_climbs, alt_climbing):
        if spd is None or flap is None:
            return False

        # Based on airspeed increase after first flap retraction
        # Align to Airspeed.
        _slice = slices_int(initial_climbs.get_aligned(spd).get_first().slice)
        climbing_5000 = alt_climbing.get_aligned(spd).get(name='5000 Ft Climbing').get_first()
        if climbing_5000:
            _slice = slice(_slice.start, int(climbing_5000.index))

        # Find when flaps were first being retracted
        flap = flap.get_aligned(spd)
        flap_retraction = find_edges_on_state_change(
            flap.array[int(_slice.start)], flap.array, change='leaving',
            phase=[_slice]
        )
        if not flap_retraction:
            return False

        flap_retraction = int(flap_retraction[0])
        if flap.array.raw[flap_retraction] < flap.array.raw[flap_retraction + 1]:
            # This was a flap extension. Don't know what happened.
            return False

        # Find when the airspeed started to increase after first flap retraction
        spd_array = spd.array[_slice.start:flap_retraction]
        spd_avg = moving_average(spd_array, window=11)
        diff = np.ma.ediff1d(spd_avg)
        if not len(diff):
            return False

        # Scanning from right to left
        diff_reversed = diff[::-1]
        # Find the first time airspeed was significantly increasing
        first_incr_idx = np.ma.argmax(diff_reversed > 0.2)
        # We determine when the speed started to increase, which is the moment
        # when airspeed stopped to decrease.
        lowest_spd_idx = np.ma.argmax(diff_reversed[first_incr_idx:] < 0.0) + first_incr_idx

        # Substract lowest_spd_idx as we are looking from the end
        index = (flap_retraction or len(spd_array)) - lowest_spd_idx
        if spd.array[flap_retraction] - spd.array[index] < 5:
            # Too small acceleration. We bail out from here.
            return False

        # We move the index 3 seconds earlier in time, taking into account
        # it took some time for the pilots to lower the pitch before we could
        # measure an acceleration.
        index -= 3 * spd.frequency
        self.frequency = spd.frequency
        self.offset = spd.offset
        self.create_kti(index)
        return True

    def derive_from_throttle(self, eng_type, throttle, initial_climbs, alt_climbing):
        if eng_type is None or eng_type.value != 'JET':
            return False

        if throttle is None:
            return False

        # Base on first engine throttle change after liftoff.
        # Align to throttle.
        _slice = slices_int(initial_climbs.get_aligned(throttle).get_first().slice)
        climbing_4000 = alt_climbing.get_aligned(throttle).get(name='4000 Ft Climbing').get_first()
        _slice = slice(_slice.start, int(climbing_4000.index) if climbing_4000 else _slice.stop)
        # XXX: Width is too small for low frequency params.
        throttle.array = throttle.array[_slice]
        throttle_threshold = 2 / throttle.frequency
        throttle_roc = np.ma.abs(rate_of_change(throttle, 2 * (1 / throttle.frequency)))
        index = index_at_value(throttle_roc, throttle_threshold)
        if index is None:
            return False

        self.frequency = throttle.frequency
        self.offset = throttle.offset
        self.create_kti(index + (_slice.start or 0))
        return True

    def derive_from_eng_np(self, eng_type, eng_np, initial_climbs, alt_climbing):
        if eng_type is None or eng_type.value != 'PROP':
            return False

        if eng_np is None:
            return False

        # Base on first Np drop after liftoff.
        # Align to Np.
        _slice = slices_int(initial_climbs.get_aligned(eng_np).get_first().slice)
        climbing_4000 = alt_climbing.get_aligned(eng_np).get(name='4000 Ft Climbing').get_first()
        if climbing_4000:
            _slice = slice(_slice.start, int(climbing_4000.index))
        # XXX: Width is too small for low frequency params.
        eng_np.array = hysteresis(eng_np.array[_slice], 4 / eng_np.hz)
        eng_np_threshold = -0.5 / eng_np.frequency
        eng_np_roc = rate_of_change(eng_np, 2 * (1 / eng_np.frequency))
        index = index_at_value(eng_np_roc, eng_np_threshold)
        if index is None:
            return False

        self.frequency = eng_np.frequency
        self.offset = eng_np.offset
        self.create_kti(index + (_slice.start or 0))
        return True

    def derive_from_alt_aal(self, alt_aal, eng_type, initial_climbs):
        if eng_type is None:
            return False

        if eng_type.value == 'JET':
            alt = 800
        elif eng_type.value == 'PROP':
            alt = 400
        else:
            return False

        # Base on crossing altitude threshold.
        if alt_aal is None:
            return False

        self.frequency = alt_aal.frequency
        self.offset = alt_aal.offset
        ics = initial_climbs.get_aligned(alt_aal)
        if ics.get_slices() is None:
            return False

        _slice = ics.get_first().slice
        index = index_at_value(alt_aal.array, alt, _slice=_slice)
        if index is None:
            return False

        self.create_kti(index)
        return True


class ClimbThrustDerateDeselected(KeyTimeInstanceNode):
    '''
    Creates KTIs where both climb thrust derates are deselected.
    Specific to 787 operations.
    '''
    @classmethod
    def can_operate(cls, available, ac_family=A('Family')):
        return ac_family and ac_family.value == 'B787'

    def derive(self, climb_derate_1=P('AT Climb 1 Derate'),
               climb_derate_2=P('AT Climb 2 Derate'),):
        self.create_ktis_on_state_change(
            'Latched',
            climb_derate_1.array | climb_derate_2.array,
            change='leaving',
        )


class APUStart(KeyTimeInstanceNode):
    name = 'APU Start'

    def derive(self,
               apu=P('APU Running')):
        self.create_ktis_on_state_change(
            'Running',
            apu.array,
            change='entering',
        )
        # Is the APU running at the beginning of the data?
        i, v = first_valid_sample(apu.array)
        if i is not None and v == 'Running':
            self.create_kti(i)


class APUStop(KeyTimeInstanceNode):
    name = 'APU Stop'

    def derive(self,
               apu=P('APU Running')):
        array = apu.array.copy()
        self.create_ktis_on_state_change(
            'Running',
            array,
            change='leaving',
        )
        # Is the APU running at the end of the data?
        # This seems an unlikely case, but added here for cmpleteness with
        # APUStart
        i, v = last_valid_sample(apu.array)
        if i is not None and v == 'Running':
            self.create_kti(i)

class LastAPUStartBeforeLiftoff(KeyTimeInstanceNode):
    '''
    Check for the last APU start before liftoff. The index will be the last
    time the APU is started before liftoff.
    '''
    name = 'Last APU Start Before Liftoff'

    def derive(self, apu_starts=KTI('APU Start'), liftoffs=KTI('Liftoff')):
        if not liftoffs:
            return

        apu_starts_before_liftoff = []
        kti_name = 'APU Start'
        apu_start_before_liftoff = apu_starts.get_previous(
            liftoffs.get_first().index, name=kti_name)
        if not apu_start_before_liftoff:
            self.warning("Could not find '%s before Liftoff. Not generating KTI.",
                         kti_name)
        else:
            apu_starts_before_liftoff.append(apu_start_before_liftoff.index)
            self.create_kti(max(apu_starts_before_liftoff))


class EngStart(KeyTimeInstanceNode):
    '''
    Records the moment of engine start for each engine in turn.

    Engines running at the start of the valid data are assumed to start when
    the data starts.
    '''

    NAME_FORMAT = 'Eng (%(number)d) Start'
    NAME_VALUES = NAME_VALUES_ENGINE

    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               eng_1_n1=P('Eng (1) N1'),
               eng_2_n1=P('Eng (2) N1'),
               eng_3_n1=P('Eng (3) N1'),
               eng_4_n1=P('Eng (4) N1'),

               eng_1_n2=P('Eng (1) N2'),
               eng_2_n2=P('Eng (2) N2'),
               eng_3_n2=P('Eng (3) N2'),
               eng_4_n2=P('Eng (4) N2'),

               eng_1_n3=P('Eng (1) N3'),
               eng_2_n3=P('Eng (2) N3'),
               eng_3_n3=P('Eng (3) N3'),
               eng_4_n3=P('Eng (4) N3'),

               eng_1_ng=P('Eng (1) Ng'),
               eng_2_ng=P('Eng (2) Ng'),
               eng_3_ng=P('Eng (3) Ng'),
               eng_4_ng=P('Eng (4) Ng'),

               ac_type=A('Aircraft Type')):

        if eng_1_n3 or eng_2_n3:
            # This aircraft has 3-spool engines
            eng_nx_list = (eng_1_n3, eng_2_n3, eng_3_n3, eng_4_n3)
            limit = CORE_START_SPEED
        elif eng_1_n2 or eng_2_n2 and ac_type != helicopter:
            # The engines are 2-spool engines
            eng_nx_list = (eng_1_n2, eng_2_n2, eng_3_n2, eng_4_n2)
            limit = CORE_START_SPEED
        elif eng_1_ng or eng_2_ng:
            # The engines have gas generator second stages
            eng_nx_list = (eng_1_ng, eng_2_ng, eng_3_ng, eng_4_ng)
            limit = CORE_START_SPEED
        else:
            eng_nx_list = (eng_1_n1, eng_2_n1, eng_3_n1, eng_4_n1)
            limit = MIN_FAN_RUNNING

        for number, eng_nx in enumerate(eng_nx_list, start=1):
            if not eng_nx:
                continue

            started = False
            # Repair 360 seconds of masked data when detecting engine starts.
            array = hysteresis(
                repair_mask(eng_nx.array,
                            repair_duration=360 / self.frequency,
                            extrapolate=True),
                HYSTERESIS_ENG_START_STOP)
            below_slices = runs_of_ones(array < limit)

            for below_slice in below_slices:

                if ((below_slice.start != 0 and
                        slice_duration(below_slice, self.hz) < 6) or
                        below_slice.stop == len(array) or
                        eng_nx.array[below_slice.stop] is np.ma.masked):
                    # Small dip or reached the end of the array.
                    continue

                started = True
                self.create_kti(below_slice.stop,
                                replace_values={'number': number})

            if not started:
                i, v = first_valid_sample(eng_nx.array)
                if i is not None and v >= limit:
                    self.warning(
                        'Eng (%d) Start: `%s` spin up not detected, '
                        'set at the first valid data sample.' %
                        (number, eng_nx.name))
                    self.create_kti(i, replace_values={'number': number})


class FirstEngStartBeforeLiftoff(KeyTimeInstanceNode):
    '''
    Check for the first engine start before liftoff. The index will be the first
    time an engine is started and remains on before liftoff.
    '''

    def derive(self, eng_starts=KTI('Eng Start'), eng_count=A('Engine Count'),
               liftoffs=KTI('Liftoff')):
        if not liftoffs:
            return
        eng_starts_before_liftoff = []
        for x in range(eng_count.value):
            kti_name = eng_starts.format_name(number=x + 1)
            eng_start_before_liftoff = eng_starts.get_previous(
                liftoffs.get_first().index, name=kti_name)
            if not eng_start_before_liftoff:
                self.warning("Could not find '%s before Liftoff.",
                             kti_name)
                continue
            eng_starts_before_liftoff.append(eng_start_before_liftoff.index)
        if eng_starts_before_liftoff:
            self.create_kti(min(eng_starts_before_liftoff))
        else:
            # Q: Should we be creating a KTI if the first engine start cannot
            # be found? - I don't think so, so lets remove the following line and log a warning?
            self.create_kti(0)


class LastEngStartBeforeLiftoff(KeyTimeInstanceNode):
    '''
    Check for the last engine start before liftoff. The index will be the last
    time an engine is started and remains on before liftoff.
    '''

    def derive(self, eng_starts=KTI('Eng Start'), eng_count=A('Engine Count'),
               liftoffs=KTI('Liftoff')):
        if not liftoffs:
            return
        eng_starts_before_liftoff = []
        for x in range(eng_count.value):
            kti_name = eng_starts.format_name(number=x + 1)
            eng_start_before_liftoff = eng_starts.get_previous(
                liftoffs.get_first().index, name=kti_name)
            if not eng_start_before_liftoff:
                self.warning("Could not find '%s before Liftoff.",
                             kti_name)
                continue
            eng_starts_before_liftoff.append(eng_start_before_liftoff.index)
        if eng_starts_before_liftoff:
            self.create_kti(max(eng_starts_before_liftoff))


class EngStop(KeyTimeInstanceNode):
    '''
    Monitors the engine stop time. Engines still running at the end of the
    data are assumed to stop at the end of the data recording.

    We use CORE_STOP_SPEED to make sure the engine truly is stopping,
    and not just running freakishly slow.
    '''

    NAME_FORMAT = 'Eng (%(number)d) Stop'
    NAME_VALUES = NAME_VALUES_ENGINE

    @classmethod
    def can_operate(cls, available):
        return 'Eng Start' in available and (
            any_of(('Eng (%d) N1' % n for n in range(1, 5)), available) or
            any_of(('Eng (%d) N2' % n for n in range(1, 5)), available)
        )

    def derive(self,
               eng_1_n1=P('Eng (1) N1'),
               eng_2_n1=P('Eng (2) N1'),
               eng_3_n1=P('Eng (3) N1'),
               eng_4_n1=P('Eng (4) N1'),

               eng_1_n2=P('Eng (1) N2'),
               eng_2_n2=P('Eng (2) N2'),
               eng_3_n2=P('Eng (3) N2'),
               eng_4_n2=P('Eng (4) N2'),

               eng_1_n3=P('Eng (1) N3'),
               eng_2_n3=P('Eng (2) N3'),
               eng_3_n3=P('Eng (3) N3'),
               eng_4_n3=P('Eng (4) N3'),

               eng_1_ng=P('Eng (1) Ng'),
               eng_2_ng=P('Eng (2) Ng'),
               eng_3_ng=P('Eng (3) Ng'),
               eng_4_ng=P('Eng (4) Ng'),

               eng_start=KTI('Eng Start'),

               ac_type=A('Aircraft Type')):

        if eng_1_n3 or eng_2_n3:
            # This aircraft has 3-spool engines
            eng_nx_list = (eng_1_n3, eng_2_n3, eng_3_n3, eng_4_n3)
            limit = CORE_STOP_SPEED
        elif eng_1_n2 or eng_2_n2 and ac_type != helicopter:
            # The engines are 2-spool engines
            eng_nx_list = (eng_1_n2, eng_2_n2, eng_3_n2, eng_4_n2)
            limit = CORE_STOP_SPEED
        elif eng_1_ng or eng_2_ng:
            # The engines have gas generator second stages
            eng_nx_list = (eng_1_ng, eng_2_ng, eng_3_ng, eng_4_ng)
            limit = CORE_STOP_SPEED
        else:
            eng_nx_list = (eng_1_n1, eng_2_n1, eng_3_n1, eng_4_n1)
            limit = MIN_FAN_RUNNING

        for number, eng_nx in enumerate(eng_nx_list, start=1):
            if not eng_nx:
                continue

            stopped = False
            # Repair 30 seconds of masked data when detecting engine stops.
            array = hysteresis(
                repair_mask(eng_nx.array,
                            repair_duration=30 / self.frequency,
                            extrapolate=True),
                HYSTERESIS_ENG_START_STOP)
            below_slices = runs_of_ones(array < limit)

            last_idx, last_value = last_valid_sample(eng_nx.array, min_samples=5)

            for below_slice in below_slices:
                if (below_slice.start == 0 or
                        eng_nx.array[below_slice.start - 1] is np.ma.masked):
                    # start of data, or following masked data.
                    continue
                elif (slice_duration(below_slice, self.hz) < 6 and not
                      is_index_within_slice(last_idx, below_slice)):
                    # Small dip not at end of data (handled later)
                    continue

                stopped = True
                self.create_kti(below_slice.start,
                                replace_values={'number': number})

            first_eng_start = eng_start.get_first(name='Eng (%d) Start'%number)
            last_eng_stop = self.get_last(name='Eng (%d) Stop'%number)

            if first_eng_start and last_eng_stop and last_eng_stop.index < first_eng_start.index:
                stopped = False

            if not stopped:
                if last_idx is not None and last_value >= limit:
                    self.warning(
                        'Eng (%d) Stop: `%s` spin down not detected, '
                        'set at the last valid data sample.' % (number,
                                                              eng_nx.name))
                    self.create_kti(last_idx, replace_values={'number': number})


class LastEngStopAfterTouchdown(KeyTimeInstanceNode):
    '''
    Check for the last engine stop after touchdown. The index will be the last
    time an engine is stopped and remains off after liftoff.
    '''

    def derive(self, eng_stops=KTI('Eng Stop'), eng_count=A('Engine Count'),
               touchdowns=KTI('Touchdown'), duration=A('HDF Duration')):
        eng_stops_after_touchdown = []

        for x in range(eng_count.value):
            kti_name = eng_stops.format_name(number=x + 1)

            if touchdowns.get_last():
                eng_stop_after_touchdown = eng_stops.get_next(
                    touchdowns.get_last().index, name=kti_name)
            else:
                eng_stop_after_touchdown = None

            if not eng_stop_after_touchdown:
                self.warning("Could not find '%s after Touchdown.",
                             kti_name)
                continue

            eng_stops_after_touchdown.append(eng_stop_after_touchdown.index)

        if eng_stops_after_touchdown:
            self.create_kti(max(eng_stops_after_touchdown))
        else:
            # Q: Should we be creating a KTI if the last engine stop cannot
            # be found?
            self.create_kti(duration.value * self.frequency - 1)


class EnterHold(KeyTimeInstanceNode):
    def derive(self, holds=S('Holding')):
        for hold in holds:
            self.create_kti(hold.slice.start)


class ExitHold(KeyTimeInstanceNode):
    def derive(self, holds=S('Holding')):
        for hold in holds:
            self.create_kti(hold.slice.stop)


class EngFireExtinguisher(KeyTimeInstanceNode):
    def derive(self, e1f=P('Eng (1) Fire Extinguisher'),
               e2f=P('Eng (2) Fire Extinguisher'),
               airborne=S('Airborne')):
        ef = np.ma.logical_or(e1f.array, e2f.array)

        # Monitor only while airborne, in case this is triggered by pre-flight tests.
        for air in airborne:
            pull_index = np.ma.nonzero(ef[air.slice])[0]
            if len(pull_index):
                self.create_kti(pull_index[0] + air.slice.start)


class GoAround(KeyTimeInstanceNode):
    """
    In POLARIS we define a Go-Around as any descent below 3000ft followed by
    an increase of 500ft. This wide definition will identify more events than
    a tighter definition, however experience tells us that it is worth
    checking all these cases. For example, we have identified attemnpts to
    land on roads or at the wrong airport, EGPWS database errors etc from
    checking these cases.
    """
    # List the minimum acceptable parameters here
    @classmethod
    def can_operate(cls, available):
        # List the minimum required parameters. If 'Altitude Radio For Flight
        # Phases' is available, that's a bonus and we will use it, but it is
        # not required.
        return ('Descent Low Climb' in available and
                'Altitude AAL For Flight Phases' in available)

    # List the optimal parameter set here

    def derive(self, dlcs=S('Descent Low Climb'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               alt_rad=P('Altitude Radio')):

        for dlc in dlcs:
            # Check for cases where a radio altimeter is not fitted or where
            # the altimeter data is out of range, hence masked, at the lowest
            # point of the go-around.
            if alt_rad and np.ma.count(alt_rad.array[dlc.slice]):
                # Worth using the radio altimeter...
                pit = np.ma.argmin(alt_rad.array[dlc.slice])

                '''
                import matplotlib.pyplot as plt
                plt.plot(alt_aal.array[dlc.slice],'-b')
                plt.plot(alt_rad.array[dlc.slice],'-r')
                plt.show()
                '''

            else:
                # Fall back on pressure altitude. Remember the altitude may
                # have been artificially adjusted if we have no absolute
                # height reference.
                pit = np.ma.argmin(alt_aal.array[dlc.slice])
            self.create_kti(pit + dlc.start_edge)


class TopOfClimb(KeyTimeInstanceNode):
    '''
    This checks for the top of climb in each Climb/Cruise/Descent period of
    the flight.
    '''
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               ccd=S('Climb Cruise Descent')):
        for ccd_phase in ccd:
            ccd_slice = ccd_phase.slice
            try:
                n_toc = find_toc_tod(alt_std.array, ccd_slice, self.frequency, mode='toc')
            except:
                # altitude data does not have an increasing section, so quit.
                continue
            # If the data started in mid-flight the ccd slice will start with None
            if ccd_slice.start is None:
                continue
            # if this is the first point in the slice, it's come from
            # data that is already in the cruise, so we'll ignore this as well
            if n_toc == 0:
                continue
            # Record the moment (with respect to this section of data)
            self.create_kti(n_toc)


class TopOfDescent(KeyTimeInstanceNode):
    '''
    This checks for the top of descent in each Climb/Cruise/Descent period
    of the flight.
    '''
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               ccd=S('Climb Cruise Descent')):
        for ccd_phase in ccd:
            ccd_slice = ccd_phase.slice
            # If this slice ended in mid-cruise, the ccd slice will end in
            # None if passed in directly or be the duration if using cached
            # params from process flight
            if ccd_slice.stop is None or ccd_slice.stop == len(alt_std.array):
                continue
            try:
                n_tod = find_toc_tod(alt_std.array, ccd_slice, self.frequency, mode='tod')
            except ValueError:
                # altitude data does not have a decreasing section, so quit.
                continue
            # Record the moment (with respect to this section of data)
            self.create_kti(n_tod)


##############################################################################
# Flap


class FlapLeverSet(KeyTimeInstanceNode):
    '''
    Indicates where the flap was set.
    '''

    NAME_FORMAT = 'Flap %(flap)s Set'
    NAME_VALUES = NAME_VALUES_LEVER

    @classmethod
    def can_operate(cls, available):

        return any_of(('Flap Lever', 'Flap Lever (Synthetic)'), available)

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)')):

        flap = flap_lever or flap_synth
        # TODO: Simplify when we've dealt with KTI node refactoring...
        for _, state in sorted(six.iteritems(flap.values_mapping)):
            self.create_ktis_on_state_change(state, flap.array, name='flap',
                                             change='entering')


class FirstFlapExtensionWhileAirborne(KeyTimeInstanceNode):
    '''
    Records each flap extension from clean configuration.
    '''

    @classmethod
    def can_operate(cls, available):

        return 'Airborne' in available and any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        if 'Lever 0' in flap.array.state:
            retracted = flap.array == 'Lever 0'
        elif '0' in flap.array.state:
            retracted = flap.array == '0'
        mask = np.ma.getmaskarray(retracted)
        for air in airborne:
            cleans = runs_of_ones(retracted[air.slice], skip_mask=True)
            for clean in cleans:
                # Skip the case where the airborne slice ends:
                if clean.stop == air.slice.stop - air.slice.start:
                    continue
                # Skip the case where the data was corrupt
                if mask[clean.stop + air.slice.start + 1]:
                    continue
                # Subtract half a sample index as transition between indices:
                self.create_kti(clean.stop + air.slice.start - 0.5)


class FlapExtensionWhileAirborne(KeyTimeInstanceNode):
    '''
    Records every flap extension in flight.
    '''

    @classmethod
    def can_operate(cls, available):

        return 'Airborne' in available and any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        self.create_ktis_at_edges(
            flap.array.raw,  # must increase to detect extensions.
            direction='rising_edges',
            phase=airborne,
        )


class FlapLoadReliefSet(KeyTimeInstanceNode):
    '''
    Indicates where flap load relief has taken place.
    '''

    def derive(self, flr=M('Flap Load Relief')):

        self.create_ktis_on_state_change('Load Relief', flr.array, change='entering')


class FlapAlternateArmedSet(KeyTimeInstanceNode):
    '''
    Indicates where flap alternate system has been armed.
    '''

    def derive(self, faa=M('Flap Alternate Armed')):

        self.create_ktis_on_state_change('Armed', faa.array, change='entering')


class SlatAlternateArmedSet(KeyTimeInstanceNode):
    '''
    Indicates where slat alternate system has been armed.
    '''

    def derive(self, saa=M('Slat Alternate Armed')):

        self.create_ktis_on_state_change('Armed', saa.array, change='entering')


class FlapRetractionWhileAirborne(KeyTimeInstanceNode):
    '''
    '''

    @classmethod
    def can_operate(cls, available):

        return 'Airborne' in available and any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               airborne=S('Airborne')):

        flap = flap_lever or flap_synth
        self.create_ktis_at_edges(
            flap.array.raw,  # must decrease to detect retractions.
            direction='falling_edges',
            phase=airborne,
        )


class FlapRetractionDuringGoAround(KeyTimeInstanceNode):
    '''
    '''

    @classmethod
    def can_operate(cls, available):

        return 'Go Around And Climbout' in available and any_of((
            'Flap Lever',
            'Flap Lever (Synthetic)',
        ), available)

    def derive(self,
               flap_lever=M('Flap Lever'),
               flap_synth=M('Flap Lever (Synthetic)'),
               go_arounds=S('Go Around And Climbout')):

        flap = flap_lever or flap_synth
        self.create_ktis_at_edges(
            flap.array.raw,  # must decrease to detect retractions.
            direction='falling_edges',
            phase=go_arounds,
        )


##############################################################################
# Gear


class GearDownSelection(KeyTimeInstanceNode):
    '''
    Instants at which gear down was selected while airborne.
    '''

    align_frequency = 1

    def derive(self,
               gear_dn_sel=M('Gear Down Selected'),
               airborne=S('Airborne')):

        self.create_ktis_on_state_change('Down', gear_dn_sel.array,
                                         change='entering', phase=airborne)


class GearUpSelection(KeyTimeInstanceNode):
    '''
    Instants at which gear up was selected while airborne excluding go-arounds.
    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type != helicopter:
            return all_deps(cls, available)
        else:
            return all_of(('Gear Up Selected', 'Airborne'), available)

    def derive(self,
               gear_up_sel=M('Gear Up Selected'),
               airborne=S('Airborne'),
               go_arounds=S('Go Around And Climbout')):

        if not airborne:
            return

        ga = go_arounds.get_slices() if go_arounds else []

        air_not_ga = slices_and_not(airborne.get_slices(), ga)
        self.create_ktis_on_state_change('Up', gear_up_sel.array,
                                         change='entering', phase=air_not_ga)


class GearUpSelectionDuringGoAround(KeyTimeInstanceNode):
    '''
    Instants at which gear up was selected while airborne including go-arounds.
    '''

    def derive(self,
               gear_up_sel=M('Gear Up Selected'),
               go_arounds=S('Go Around And Climbout')):

        self.create_ktis_on_state_change('Up', gear_up_sel.array,
                                         change='entering', phase=go_arounds)


##############################################################################
# TAWS

class TAWSGlideslopeCancelPressed(KeyTimeInstanceNode):

    name = 'TAWS Glideslope Cancel Pressed'

    def derive(self, tgc=P('TAWS Glideslope Cancel'), airborne=S('Airborne')):
        # Monitor only while airborne, in case this is triggered pre-flight.
        self.create_ktis_on_state_change('Cancel', tgc.array,
                                         change='entering', phase=airborne)


class TAWSMinimumsTriggered(KeyTimeInstanceNode):
    name = 'TAWS Minimums Triggered'

    def derive(self, tmin=P('TAWS Minimums'), airborne=S('Airborne')):
        self.create_ktis_on_state_change('Minimums', tmin.array,
                                         change='entering', phase=airborne)


class TAWSTerrainOverridePressed(KeyTimeInstanceNode):
    name = 'TAWS Terrain Override Pressed'

    def derive(self, tmin=P('TAWS Terrain Override'), airborne=S('Airborne')):
        self.create_ktis_on_state_change('Override', tmin.array,
                                         change='entering', phase=airborne)


##############################################################################
# Flight Sequence


class TakeoffTurnOntoRunway(KeyTimeInstanceNode):
    '''
    The Takeoff flight phase is computed to start when the aircraft turns
    onto the runway, so at worst this KTI is just the start of that phase.
    Where possible we compute the sharp point of the turn onto the runway.
    '''
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type and ac_type.value == 'helicopter':
            return False
        else:
            return all_deps(cls, available)

    def derive(self, head=P('Heading Continuous'),
               toffs=S('Takeoff'),
               fast=S('Fast')):
        for toff in toffs:
            # Ideally we'd like to work from the start of the Fast phase
            # backwards, but in case there is a problem with the phases,
            # use the midpoint. This avoids identifying the heading
            # change immediately after liftoff as a turn onto the runway.
            next_fast = fast.get_next(toff.slice.start)
            start_search = next_fast.slice.start if next_fast else None
            if (start_search is None) or (start_search > toff.slice.stop):
                start_search = (toff.slice.start + toff.slice.stop) / 2
            peak_bend = peak_curvature(head.array, slice(
                start_search, toff.slice.start, -1), curve_sense='Bipolar')
            if peak_bend:
                takeoff_turn = peak_bend
            else:
                takeoff_turn = toff.slice.start
            self.create_kti(takeoff_turn)


class TakeoffAccelerationStart(KeyTimeInstanceNode):
    '''
    The start of the takeoff roll is ideally computed from the forwards
    acceleration down the runway, but a quite respectable "backstop" is
    available from the point where the airspeed starts to increase (providing
    this is from an analogue source). This allows for aircraft either with a
    faulty sensor, or no longitudinal accelerometer.
    '''
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type and ac_type.value == 'helicopter':
            return False
        else:
            if 'Acceleration Longitudinal' in available:
                return all_of(('Airspeed', 'Takeoff', 'Acceleration Longitudinal Offset Removed'), available)
            else:
                return all_of(('Airspeed', 'Takeoff'), available)

    def derive(self, speed=P('Airspeed'), takeoffs=S('Takeoff'),
               accel=P('Acceleration Longitudinal Offset Removed'),
               accel_raw=P('Acceleration Longitudinal')):
        for takeoff in takeoffs:
            start_accel = None
            if accel:
                # Ideally compute this from the forwards acceleration.
                # If they turn onto the runway already accelerating, take that as the start point.
                first_accel = accel.array[takeoff.slice.start]
                if first_accel > TAKEOFF_ACCELERATION_THRESHOLD:
                    start_accel = takeoff.slice.start
                else:
                    start_accel = slices_and(
                        runs_of_ones(accel.array >= TAKEOFF_ACCELERATION_THRESHOLD),
                        [takeoff.slice]
                    )
                    if len(start_accel) >= 1:
                        accel_sel = start_accel[0]
                        for alt_start in start_accel[1:]:
                            if min_value(accel.array,slice(accel_sel.start, alt_start.start)).value <= 0.0:
                                # The dip in acceleration is too great to be the actual
                                # takeoff acceleration point, it might be an RTO.
                                # Take the start the next slice.
                                accel_sel = alt_start
                        start_accel = accel_sel.start

            if start_accel is None or start_accel == []:
                '''
                A quite respectable "backstop" is from the rate of change of
                airspeed. We use this if the acceleration is not available or
                if, for any reason, the previous computation failed.
                Originally we used the peak_curvature algorithm to identify
                where the airspeed started to increase, but when values lower
                than a threshold were masked this ceased to work (the "knee"
                is masked out) and so the extrapolated airspeed was adopted.
                '''
                #pc = peak_curvature(speed.array[takeoff.slice])
                p, m, c = coreg(speed.array[takeoff.slice])
                if p is None:
                    continue
                start_accel = max(takeoff.slice.start - c / m, 0.0)

            # FIXME: coreg can return values larger than the parameter size (segment hash 18c141e479fc2412539b017b9340e67f78c86e6443c3cabd8f9839be0eadfbd5)
            if start_accel is not None and start_accel <= takeoff.slice.stop:
                self.create_kti(start_accel)


class TakeoffStart(KeyTimeInstanceNode):
    def derive(self,
               acc_start=KTI('Takeoff Acceleration Start'),
               throttle=P('Throttle Levers')):
        if acc_start:
            self.create_kti(acc_start.get_first().index)


class TakeoffPeakAcceleration(KeyTimeInstanceNode):
    """
    As for landing, the point of maximum acceleration, is used to identify the
    location and heading of the takeoff.
    """
    def derive(self, toffs=S('Takeoff'),
               accel=P('Acceleration Longitudinal')):
        for toff in toffs:
            index, value = max_value(accel.array, _slice=toff.slice)
            if index:  # In case all the Ay data is invalid.
                self.create_kti(index)


class Liftoff(KeyTimeInstanceNode):
    '''
    The point of liftoff is computed by working out all the available
    indications of liftoff and taking the second of these, on the assumption
    that the first indication may not be valid.

    The five indications used are:

    (a) the inertial vertical speed indicates a rate of climb (we cannot use
    barometric rate of climb as the aircraft is in ground effect and
    transient changes of pressure field as the aircraft rotates cause an
    indicated descent just prior to lift)

    (b) a normal acceleration of greater than 1.2g

    (c) radio altimeter indications greater than zero (see http://www.flightdatacommunity.com/looking-closely-at-radio-altimeters/)

    (d) altitude above airfield greater than zero. This is computed from the
    available height sources, so will work off the pressure altitude only if no
    radio altimeter is available.

    (e) change in the gear on ground (weight oon wheels) switch status where
    available.

    In the case where the gear on ground signal switches first, we use this.
    However it is common for this to switch at the end of the oleo extension
    which is why it commonly operates after other indications.

    For a more descriptive explanation of the second of many technique, refer to
    http://www.flightdatacommunity.com/when-does-the-aircraft-land/
    '''

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT', 'MID_FLIGHT', 'STOP_ONLY'):
            return False
        return 'Airborne' in available

    def derive(self,
               vert_spd=P('Vertical Speed Inertial'),
               acc_norm=P('Acceleration Normal Offset Removed'),
               vert_spd_baro=P('Vertical Speed'),
               alt_rad=P('Altitude Radio Offset Removed'),
               gog=M('Gear On Ground'),
               airs=S('Airborne'),
               frame=A('Frame'),
               ac_type=A('Aircraft Type')):

        if ac_type and ac_type.value == 'helicopter':
            for air in airs:
                self.create_kti(air.start_edge)
            return

        for air in airs:
            index_acc = index_vs = index_rad = index_gog = index_lift = None
            index_air = air.start_edge
            if index_air is None:
                continue
            back_6 = (air.slice.start - 6.0 * self.frequency)
            if back_6 < 0:
                # unlikely to have lifted off within 3 seconds of data start
                # STOP ONLY slice without a liftoff in this Airborne section
                continue
            on_6 = (air.slice.start + 6.0 * self.frequency) + 1  # For indexing
            to_scan = slice(back_6, on_6)

            if vert_spd:
                index_vs = index_at_value(
                    vert_spd.array, VERTICAL_SPEED_FOR_LIFTOFF, to_scan)
            elif vert_spd_baro:
                # Fallback to pressure rate of climb
                index_vs = index_at_value(vert_spd_baro.array,
                                          VERTICAL_SPEED_FOR_LIFTOFF,
                                          to_scan)
                # and try to augment this with another measure
                if acc_norm:
                    idx = np.ma.argmax(acc_norm.array[slices_int(to_scan)])
                    if acc_norm.array[slices_int(to_scan)][idx] > 1.2:
                        index_acc = idx + back_6

            if alt_rad:
                index_rad = index_at_value(alt_rad.array, 0.0, slice(on_6, back_6,-1))

            if gog:
                # Try using Gear On Ground switch
                edges = find_edges_on_state_change(
                    'Ground', gog.array[slices_int(to_scan)], change='leaving')
                if edges:
                    # use the last liftoff point
                    index = edges[-1] + back_6
                    # Check we were within 5ft of the ground when the switch triggered.
                    if alt_rad is None:
                        index_gog = index
                    elif alt_rad.array[int(index)] < 5.0 or \
                            alt_rad.array[int(index)] is np.ma.masked:
                        index_gog = index
                    else:
                        index_gog = None

            # We pick the second  recorded indication for the point of liftoff.
            index_list = sorted_valid_list([index_air,
                                            index_vs,
                                            index_acc,
                                            index_gog,
                                            index_rad])

            if len(index_list) > 1:
                index_lift = sorted(index_list)[1]
            else:
                index_lift = index_list[0]
            # but in any case, if we have a gear on ground signal which goes
            # off first, adopt that.
            if index_gog and index_gog < index_lift:
                index_lift = index_gog

            self.create_kti(index_lift)

            '''
            # Plotting process to view the results in an easy manner.
            import matplotlib.pyplot as plt
            name = 'Liftoff Plot %s, %d' %(frame.value, index_air)
            print(name)
            dt_pre = 5
            hz = self.frequency
            timebase=np.linspace(-dt_pre*hz, dt_pre*hz, 2*dt_pre*hz+1)
            plot_period = slice(int(floor(air.slice.start-dt_pre*hz)), int(floor(air.slice.start-dt_pre*hz+len(timebase))))
            plt.figure()
            plt.plot(0, 13.0,'vb', markersize=8)
            if vert_spd:
                plt.plot(timebase, np.ma.masked_greater(vert_spd.array[plot_period],600.0)/20.0, 'o-g')
            else:
                #plt.plot(timebase, np.ma.masked_greater(vert_spd_baro.array[plot_period],600.0)/20.0, 'o-c')
                if acc_norm:
                    plt.plot(timebase, acc_norm.array[plot_period]*10.0, 'o-c')
                if index_acc:
                    plt.plot(index_acc-air.slice.start, 15.0,'vc', markersize=8)

            if index_vs:
                plt.plot(index_vs-air.slice.start, 15,'vg', markersize=8)

            if alt_rad:
                plt.plot(timebase, np.ma.masked_greater(alt_rad.array[plot_period],30.0), 'o-r')
            if index_rad:
                plt.plot(index_rad-air.slice.start, 17.0,'vr', markersize=8)

            if gog:
                plt.plot(timebase, gog.array[plot_period]*10, 'o-k')
            if index_gog:
                plt.plot(index_gog-air.slice.start, 19.0,'vk', markersize=8)

            if vert_spd_baro:
                plt.plot(timebase, np.ma.masked_greater(vert_spd_baro.array[plot_period]/20.0, 30.0), 'o-b')

            if index_lift:
                plt.plot(index_lift-air.slice.start, -5.0,'^m', markersize=14)

            plt.title(name)
            plt.grid()
            plt.ylim(-10,30)
            filename = name
            print(name)
            # Two lines to quickly make this work.
            import os
            WORKING_DIR = tempfile.gettempdir()
            output_dir = os.path.join(WORKING_DIR, 'Liftoff_graphs')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            plt.savefig(os.path.join(output_dir, filename + '.png'))
            plt.show()
            plt.clf()
            plt.close()
            '''


class LowestAltitudeDuringApproach(KeyTimeInstanceNode):
    '''
    For any approach phase that did not result in a landing, the lowest point
    is taken as key, from which the position, heading and height will be
    taken as KPVs.

    This KTI is essential to collect the related KPVs which inform the
    approach attribute, and thereafter compute the smoothed track.
    '''

    def derive(self,
               alt_aal=P('Altitude AAL'),
               alt_rad=P('Altitude Radio'),
               approaches=S('Approach And Landing')):

        height = minimum_unmasked(alt_aal.array, alt_rad.array)
        for approach in approaches:
            index = np.ma.argmin(height[approach.slice])
            self.create_kti(approach.start_edge + index)


class InitialClimbStart(KeyTimeInstanceNode):
    # The Takeoff flight phase is computed to run up to the start of the
    # initial climb, so this KTI is just at the end of that phase.
    def derive(self, toffs=S('Takeoff')):
        for toff in toffs:
            if toff.stop_edge:
                self.create_kti(toff.stop_edge)


class LandingStart(KeyTimeInstanceNode):
    # The Landing flight phase is computed to start passing through 50ft
    # (nominally), so this KTI is just at the start of that phase.
    def derive(self, landings=S('Landing')):
        for landing in landings:
            if landing.start_edge:
                self.create_kti(landing.start_edge)


class TouchAndGo(KeyTimeInstanceNode):
    """
    In POLARIS we define a Touch and Go as a Go-Around that contacted the ground.
    """
    @classmethod
    def can_operate(cls, available):
        return all_of(('Altitude AAL', 'Go Around And Climbout'), available)

    def derive(self, alt_aal=P('Altitude AAL'), go_around_and_climbouts=S('Go Around And Climbout'),
               gog=P('Gear On Ground')):
        if gog:
            aligned_gog = align(gog, alt_aal)
        else:
            aligned_gog = np.ma.array([0]*len(alt_aal.array), mask=[1]*len(alt_aal.array))
        for ga in go_around_and_climbouts:
            ga_index = int(ga.start_edge or ga.slice.start)
            while ga_index < ga.stop_edge:
                if alt_aal.array[ga_index] == 0.0 or aligned_gog[ga_index] == 'Ground':
                    self.create_kti(ga_index)
                    break
                ga_index += 1


class Touchdown(KeyTimeInstanceNode):
    '''
    Touchdown is notoriously difficult to identify precisely, and a
    suggestion from a Boeing engineer was to add a longitudinal acceleration
    term as there is always an instantaneous drag when the mainwheels touch.

    This was added in the form of three triggers, one detecting the short dip
    in Ax, a second for the point of onset of deceleration and a third for
    braking deceleration.

    So, we look for the weight on wheels switch if this is the first indication,
    or the second indication of:
    * Zero feet AAL (normally derived from the radio altimeter)
    * Sudden rise in normal acceleration bumping the ground
    * Significant product of two samples of normal acceleration (correlating to a sudden drop in descent rate)
    * A transient reduction in longitudinal acceleration as the wheels first spin up
    * A large reduction in longitudinal acceleration when braking action starts

    http://www.flightdatacommunity.com/when-does-the-aircraft-land/
    '''
    # List the minimum acceptable parameters here

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'), seg_type=A('Segment Type')):
        # if we have 'Acceleration Longitudinal' hold off until
        # 'Acceleration Longitudinal Offset Removed' can be processed
        if 'Acceleration Longitudinal' in available and \
           'Acceleration Longitudinal Offset Removed' not in available:
            return False
        if ac_type and ac_type.value == 'helicopter':
            return 'Airborne' in available
        elif seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT', 'MID_FLIGHT', 'START_ONLY'):
            return False
        else:
            return all_of(('Altitude AAL', 'Landing'), available)

    def derive(self, acc_norm=P('Acceleration Normal'),
               acc_long=P('Acceleration Longitudinal Offset Removed'),
               alt=P('Altitude AAL'),
               alt_rad=P('Altitude Radio'),
               gog=M('Gear On Ground'),
               lands=S('Landing'),
               flap=P('Flap'),
               manufacturer=A('Manufacturer'),
               family=A('Family'),
               # helicopter
               airs=S('Airborne'),
               ac_type=A('Aircraft Type'),
               # Only used in can_operate
               accel=P('Acceleration Longitudinal')):
        if ac_type and ac_type.value == 'helicopter':
            for air in airs:
                self.create_kti(air.stop_edge)
            return

        # The preamble here checks that the landing we are looking at is
        # genuine, it's not just because the data stopped in mid-flight. We
        # reduce the scope of the search for touchdown to avoid triggering in
        # mid-cruise, and it avoids problems for aircraft where the gear
        # signal changes state on raising the gear (OK, if they do a gear-up
        # landing it won't work, but this will be the least of the problems).

        dt_pre = 10.0  # Seconds to scan before estimate.
        dt_post = 3.0  # Seconds to scan after estimate.
        hz = alt.frequency

        for land in lands:

            # initialise within loop as we dont want to carry indexes into the next landing
            index_gog = index_wheel_touch = index_brake = index_decel = None
            index_dax = index_z = index_az = index_daz = None
            peak_ax = peak_az = delta = 0.0

            # We have to have an altitude signal, so this forms an initial
            # estimate of the touchdown point.
            index_alt = index_at_value(alt.array, 0.0, land.slice)

            if gog:
                # Try using Gear On Ground switch
                edges = find_edges_on_state_change(
                    'Ground', gog.array[land.slice])
                if edges:
                    # use the first contact with ground as touchdown point
                    # (i.e. we ignore bounces)
                    index = edges[0] + land.slice.start
                    # Check we were within 10ft of the ground when the switch triggered.
                    if not alt or alt.array[int(index)] < 10.0:
                        index_gog = index

            if manufacturer and manufacturer.value == 'Saab' and \
               family and family.value == '2000':
                # This covers aircraft with automatic flap retraction on
                # landing but no gear on ground signal. The Saab 2000 is a
                # case in point.
                land_flap = np.ma.array(data=flap.array.data[land.slice],
                                        mask=flap.array.mask[land.slice])
                flap_change_idx = index_at_value(land_flap, land_flap[0] - 1)
                if flap_change_idx:
                    index_gog = int(flap_change_idx) + land.slice.start

            index_ref = min([x for x in (index_alt, index_gog) if x is not None])

            # With an estimate from the height and perhaps gear switch, set
            # up a period to scan across for accelerometer based
            # indications...
            period_end = int(ceil(index_ref + dt_post * hz))
            period_start = max(floor(index_ref - dt_pre * hz), 0)
            if alt_rad:
                # only look for 5ft altitude if Radio Altitude is recorded,
                # due to Altitude STD accuracy and ground effect.
                alt_rad_start = index_at_value(alt.array, 5, _slice=slice(period_end, period_start, -1))
                if alt_rad_start is not None:
                    period_start = alt_rad_start
            period = slice(period_start, period_end)

            if acc_long:
                drag = np.ma.copy(acc_long.array[slices_int(period)])
                drag = np.ma.where(drag > 0.0, 0.0, drag)

                # Look for inital wheel contact where there is a sudden spike in Ax.

                touch = np_ma_masked_zeros_like(drag)
                for i in range(2, len(touch)-2):
                    # Looking for a downward pointing "V" shape over half the
                    # Az sample rate. This is a common feature at the point
                    # of wheel touch.
                    touch[i-2] = max(0.0,drag[i-2]-drag[i]) * max(0.0,drag[i+2]-drag[i])
                peak_ax = np.max(touch)
                # Only use this if the value was significant.
                if peak_ax>0.0005:
                    ix_ax2 = np.argmax(touch)
                    ix_ax = ix_ax2
                    # See if this was the second of a pair, with the first a little smaller.
                    if np.ma.count(touch[:ix_ax2]) > 0:
                        # I have some valid data to scan
                        ix_ax1 = np.argmax(touch[:ix_ax2])
                        if touch[ix_ax1] > peak_ax*0.2:
                            # This earlier touch was a better guess.
                            peak_ax = touch[ix_ax1]
                            ix_ax = ix_ax1

                    index_wheel_touch = ix_ax+1+period.start

                # Look for the onset of braking
                index_brake = np.ma.argmin(rate_of_change_array(drag, hz))
                if index_brake:
                    index_brake += period.start

                # Look for substantial deceleration

                index_decel = index_at_value(drag, -0.1)
                if index_decel:
                    index_decel += period.start

            if acc_norm:
                lift = acc_norm.array[slices_int(period)]
                mean = np.mean(lift)
                lift = np.ma.masked_less(lift-mean, 0.0)
                bump = np_ma_masked_zeros_like(lift)

                # A firm touchdown is typified by at least two large Az samples.
                for i in range(1, len(bump)-1):
                    bump[i-1]=lift[i]*lift[i+1]
                peak_az = np.max(bump)
                if peak_az > 0.1:
                    index_az = np.argmax(bump)+period.start
                else:
                    # In the absence of a clear touchdown, contact can be
                    # indicated by an increase in g of more than 0.075
                    for i in range(0, len(lift)-1):
                        if lift[i] and lift[i+1]:
                            delta=lift[i+1]-lift[i]
                            if delta > 0.075:
                                index_daz = i+1+period.start
                                break

            # Pick the first of the two normal accelerometer measures to
            # avoid triggering a touchdown from a single faulty sensor:
            index_z_list = [x for x in (index_az, index_daz) if x is not None]
            if index_z_list:
                index_z = min(index_z_list)

            # ...then collect the valid estimates of the touchdown point...
            index_list = sorted_valid_list([index_alt,
                                            index_gog,
                                            index_wheel_touch,
                                            index_brake,
                                            index_decel,
                                            index_dax,
                                            index_z])

            # ...to find the best estimate...
            # If we have lots of measures, bias towards the earlier ones.
            #index_tdn = np.median(index_list[:4])
            if len(index_list) == 0:
                # No clue where the aircraft landed. Give up.
                return
            elif len(index_list) == 1:
                # Only one identifier - pick this
                index_tdn = index_list[0]
            else:
                # Normal selection is just the second one!
                index_tdn = index_list[1]
                # ensure detected touchdown point is not after Gear on Ground indicates on ground
                if index_gog:
                    index_tdn = min(index_tdn, index_gog)

            # self.create_kti(index_tdn)
            self.info("Touchdown: Selected index: %s @ %sHz. Complete list (index_alt: %s, "\
                      "index_gog: %s, index_wheel_touch: %s,  index_brake: %s, "\
                      "index_decel: %s, index_dax: %s, index_z: %s)",
                      index_tdn, self.frequency, index_alt, index_gog, index_wheel_touch,
                      index_brake, index_decel, index_dax, index_z)
            self.create_kti(index_tdn)

            '''
            # Plotting process to view the results in an easy manner.
            import matplotlib.pyplot as plt
            import os
            name = 'Touchdown with values Ax=%.4f, Az=%.4f and dAz=%.4f' %(peak_ax, peak_az, delta)
            self.info(name)
            tz_offset = index_ref - period.start
            timebase=np.linspace(-tz_offset, dt_post*hz, tz_offset+(dt_post*hz)+1)
            plot_period = slice(int(floor(index_ref-tz_offset)), int(floor(index_ref-tz_offset+len(timebase))))
            plt.figure()
            if alt:
                plt.plot(timebase, alt.array[plot_period], 'o-r')
            if acc_long:
                plt.plot(timebase, acc_long.array[plot_period]*100, 'o-m')
            if acc_norm:
                plt.plot(timebase, acc_norm.array[plot_period]*10, 'o-g')
            if gog:
                plt.plot(timebase, gog.array[plot_period]*10, 'o-k')
            if index_gog:
                plt.plot(index_gog-index_ref, 2.0,'ok', markersize=8)
            if index_brake:
                plt.plot(index_brake-index_ref, 2.5,'oy', markersize=8)
            if index_wheel_touch:
                plt.plot(index_wheel_touch-index_ref, 3.0,'om', markersize=8)
            if index_decel:
                plt.plot(index_decel-index_ref, 3.0,'oc', markersize=8)
            if index_az:
                plt.plot(index_az-index_ref, 4.0,'og', markersize=8)
            if index_dax:
                plt.plot(index_dax-index_ref, 5.5,'dm', markersize=8)
            if index_daz:
                plt.plot(index_daz-index_ref, 5.0,'dg', markersize=8)
            if index_alt:
                plt.plot(index_alt-index_ref, 1.0,'or', markersize=8)
            if index_tdn:
                plt.plot(index_tdn-index_ref, -2.0,'db', markersize=10)
            plt.title(name)
            plt.grid()
            filename = 'One-ax-Touchdown-%s' % os.path.basename(self._h.file_path) if getattr(self, '_h', None) else 'Plot'
            print(name)
            WORKING_DIR = 'C:\\Temp'
            output_dir = os.path.join(WORKING_DIR, 'Touchdown_graphs')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            plt.savefig(os.path.join(output_dir, filename + '.png'))
            #plt.show()
            plt.clf()
            plt.close()
            '''


class OffshoreTouchdown(KeyTimeInstanceNode):
    '''
    Subset of touchdowns which occur offshore
    '''

    def derive(self, touchdowns=KTI('Touchdown'),
               offshore=S('Offshore')):
        offshore_slices = offshore.get_slices()
        for tdwn in touchdowns:
            if is_index_within_slices(tdwn.index, offshore_slices):
                self.create_kti(tdwn.index)


class OnshoreTouchdown(KeyTimeInstanceNode):
    '''
    Subset of touchdowns which occur offshore
    '''

    def derive(self, touchdowns=KTI('Touchdown'),
               offshore=S('Offshore')):
        offshore_slices = offshore.get_slices()
        for tdwn in touchdowns:
            if not is_index_within_slices(tdwn.index, offshore_slices):
                self.create_kti(tdwn.index)


class LandingDecelerationEnd(KeyTimeInstanceNode):
    '''
    Whereas peak acceleration at takeoff is a good measure of the start of
    the takeoff roll, the peak deceleration on landing often occurs very late
    in the landing when the brakes are applied harshly for a moment, for
    example when stopping to make a particular turnoff. For this reason we
    prefer to use the end of the steep reduction in airspeed as a measure of
    the end of the landing roll.
    '''
    def derive(self, speed=P('Airspeed'), landings=S('Landing')):
        for landing in landings:
            end_decel = peak_curvature(speed.array, landing.slice, curve_sense='Concave')
            # Create the KTI if we have found one, otherwise point to the end
            # of the data, as sometimes recordings stop in mid-landing phase
            if end_decel:
                self.create_kti(end_decel)
            else:
                self.create_kti(landing.stop_edge)


class LandingTurnOffRunway(KeyTimeInstanceNode):
    # See Takeoff Turn Onto Runway for description.
    def derive(self, apps=App('Approach Information')):
        for app in apps:
            if app.turnoff:
                self.create_kti(app.turnoff)


################################################################################


class AltitudeWhenClimbing(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain altitudes when the aircraft is climbing.
    '''
    NAME_FORMAT = '%(altitude)d Ft Climbing'
    NAME_VALUES = NAME_VALUES_CLIMB

    def derive(self,
               takeoff=S('Takeoff'),
               initial_climb=S('Initial Climb'),
               climb=S('Climb'),
               alt_aal=P('Altitude AAL'),
               alt_std=P('Altitude STD Smoothed')):

        climbs = list(takeoff) + list(initial_climb) + list(climb)
        climb_slices = slices_remove_small_gaps([c.slice for c in climbs])
        for climb_slice in climb_slices:
            for alt_threshold in self.NAME_VALUES['altitude']:
                # Will trigger a single KTI per height (if threshold is crossed)
                # per climbing phase.
                if alt_threshold <= TRANSITION_ALTITUDE:
                    # Use height above airfield.
                    alt = alt_aal.array
                else:
                    # Use standard altitudes.
                    alt = alt_std.array

                index = index_at_value(alt, alt_threshold, climb_slice)
                if index:
                    self.create_kti(index, altitude=alt_threshold)


class AltitudeWhenDescending(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain heights when the aircraft is descending.
    '''
    NAME_FORMAT = '%(altitude)d Ft Descending'
    NAME_VALUES = NAME_VALUES_DESCENT

    def derive(self, descending=S('Descent'),
               alt_aal=P('Altitude AAL'),
               alt_std=P('Altitude STD Smoothed')):
        alt_aal=repair_mask(alt_aal.array, frequency=alt_aal.frequency, copy=True)
        alt_std=repair_mask(alt_std.array, frequency=alt_std.frequency, copy=True)
        for descend in descending:
            for alt_threshold in self.NAME_VALUES['altitude']:
                # Will trigger a single KTI per height (if threshold is
                # crossed) per descending phase. The altitude array is
                # scanned backwards to make sure we trap the last instance at
                # each height.
                if alt_threshold <= TRANSITION_ALTITUDE:
                    # Use height above airfield.
                    alt = alt_aal
                else:
                    # Use standard altitudes.
                    alt = alt_std

                index = index_at_value(alt, alt_threshold,
                                       slice(descend.slice.stop,
                                             descend.slice.start, -1))
                if index:
                    self.create_kti(index, altitude=alt_threshold)


class AltitudeBeforeLevelFlightWhenClimbing(KeyTimeInstanceNode):

    NAME_FORMAT = '%(altitude)d Ft Before Level Flight Climbing'
    NAME_VALUES = {'altitude': [1000, 2000]}

    def derive(self,
               aal=P('Altitude STD Smoothed'),
               level_flight=S('Level Flight'),
               climbing=S('Climb')):

        if not level_flight:
            # no level flights so exit here
            return

        ordered_level = level_flight.get_ordered_by_index().get_slices()
        not_level = [slice(0, ordered_level[0].start)] + \
            slices_not(ordered_level)

        for n, level in enumerate(slices_int(ordered_level)):
            climb_descent = not_level[n]
            level_height = np.ma.median(aal.array[level])
            if level_height < 3000:
                continue
            climb_slice = None
            for climb in climbing:
                if slices_overlap(climb.slice, climb_descent):
                    climb_slice = climb_descent
                    break
            else:
                continue  # Must be following a descent

            for height in self.NAME_VALUES['altitude']:
                index = index_at_value(aal.array, level_height-height,
                                       _slice=slice(climb_slice.stop,
                                                    climb_slice.start,
                                                    -1))
                if index:
                    self.create_kti(index, replace_values={'altitude': height})


class AltitudeBeforeLevelFlightWhenDescending(KeyTimeInstanceNode):

    NAME_FORMAT = '%(altitude)d Ft Before Level Flight Descending'
    NAME_VALUES = {'altitude': [1000, 2000]}

    def derive(self,
               aal=P('Altitude STD Smoothed'),
               level_flight=S('Level Flight'),
               descending=S('Descending')):

        if not level_flight:
            # no level flights so exit here
            return

        ordered_level = level_flight.get_ordered_by_index().get_slices()
        not_level = [slice(0, ordered_level[0].start)] + \
            slices_not(ordered_level)

        for n, level in enumerate(slices_int(ordered_level)):
            climb_descent = not_level[n]
            level_height = np.ma.median(aal.array[level])
            if level_height < 3000:
                continue
            descent_slice = None
            for descent in descending:
                if slices_overlap(descent.slice, climb_descent):
                    descent_slice = climb_descent
                    break
            else:
                continue  # Must be following a climb

            for height in self.NAME_VALUES['altitude']:
                index = index_at_value(aal.array, level_height+height,
                                       _slice=slice(descent_slice.stop,
                                                    descent_slice.start,
                                                    -1))
                if index:
                    self.create_kti(index, replace_values={'altitude': height})


"""

Altitudes split with 5000ft and below related to airfield, and above this
standard pressure altitudes. Therefore Altitude STD Descending is redundant.

class AltitudeSTDWhenDescending(KeyTimeInstanceNode):
    '''
    Creates KTIs at certain Altitude STD heights when the aircraft is
    descending.
    '''
    name = 'Altitude STD When Descending'
    NAME_FORMAT = '%(altitude)d Ft Descending'
    NAME_VALUES = NAME_VALUES_DESCENT

    def derive(self, descending=S('Descending'),
               alt_aal=P('Altitude AAL'),
               alt_std=P('Altitude STD Smoothed')):

        for descend in descending:
            for alt_threshold in self.NAME_VALUES['altitude']:
                # Will trigger a single KTI per height (if threshold is
                # crossed) per descending phase. The altitude array is
                # scanned backwards to make sure we trap the last instance at
                # each height.
                if alt_threshold <= 5000:
                    # Use height above airfield.
                    alt = alt_aal.array
                else:
                    # Use standard altitudes.
                    alt = alt_std.array

                index = index_at_value(alt, alt_threshold,
                                       slice(descend.slice.stop,
                                             descend.slice.start, -1))
                if index:
                    self.create_kti(index, altitude=alt_threshold)
"""

class MinsToTouchdown(KeyTimeInstanceNode):
    #TODO: TESTS
    NAME_FORMAT = "%(time)d Mins To Touchdown"
    NAME_VALUES = {'time': [5, 4, 3, 2, 1]}

    def derive(self, touchdowns=KTI('Touchdown'),
               liftoffs=KTI('Liftoff')):
        for touchdown in touchdowns:
            for t in self.NAME_VALUES['time']:
                index = touchdown.index - (t * 60 * self.frequency)
                # Make sure the aircraft was airborne at that time.
                if index and index > liftoffs.get_previous(touchdown.index).index:
                    self.create_kti(index, time=t)


class SecsToTouchdown(KeyTimeInstanceNode):
    #TODO: TESTS
    NAME_FORMAT = "%(time)d Secs To Touchdown"
    NAME_VALUES = {'time': [90, 30, 20]}

    def derive(self, touchdowns=KTI('Touchdown'),
               liftoffs=KTI('Liftoff')):
        for touchdown in touchdowns:
            for t in self.NAME_VALUES['time']:
                index = touchdown.index - (t * self.frequency)
                # Make sure the aircraft was airborne at that time.
                if index and index > liftoffs.get_previous(touchdown.index).index:
                    self.create_kti(index, time=t)


class DistanceToTouchdown(KeyTimeInstanceNode):
    NAME_FORMAT = "%(distance)s NM To Touchdown"
    NAME_VALUES = {'distance': [0.8, 1.0, 1.5, 2.0]}

    def derive(self, dtl=P('Distance To Landing'),
               touchdowns=KTI('Touchdown')):
        last_tdwn_idx = 0
        for touchdown in touchdowns:
            for d in self.NAME_VALUES['distance']:
                index = index_at_value(dtl.array, d,
                                       slice(int(floor(touchdown.index)), last_tdwn_idx, -1))
                if index:
                    # may not have travelled far enough to find distance threshold.
                    self.create_kti(index, distance=d)
            last_tdwn_idx = touchdown.index


class Autoland(KeyTimeInstanceNode):
    '''
    All requried autopilots engaged at touchdown. Many Boeing aircraft require
    all three AutoPilot channels to be engaged.
    '''
    TRIPLE_FAMILIES = (
        'B737 Classic',
        'B737 NG',
        'B757',
        'B767',
    )

    @classmethod
    def can_operate(cls, available):
        return all_of(('AP Channels Engaged', 'Touchdown'), available)

    def derive(self, ap=M('AP Channels Engaged'), touchdowns=KTI('Touchdown'),
               family=A('Family')):
        family = family.value if family else None
        for td in touchdowns:
            if ap.array[int(td.index)] == 'Dual' and family not in self.TRIPLE_FAMILIES:
                self.create_kti(td.index)
            elif ap.array[int(td.index)] == 'Triple':
                self.create_kti(td.index)
            else:
                # in Single OR Dual and Triple was required
                continue


#################################################################
# ILS & IAN Established Markers (primarily for development)

class LocalizerEstablishedStart(KeyTimeInstanceNode):
    def derive(self, ilss=S('ILS Localizer Established')):
        for ils in ilss:
            self.create_kti(ils.slice.start)

class LocalizerEstablishedEnd(KeyTimeInstanceNode):
    def derive(self, ilss=S('ILS Localizer Established')):
        for ils in ilss:
            self.create_kti(ils.slice.stop)

class GlideslopeEstablishedStart(KeyTimeInstanceNode):
    def derive(self, ilss=S('ILS Glideslope Established')):
        for ils in ilss:
            self.create_kti(ils.slice.start)

class GlideslopeEstablishedEnd(KeyTimeInstanceNode):
    def derive(self, ilss=S('ILS Glideslope Established')):
        for ils in ilss:
            self.create_kti(ils.slice.stop)


class IANFinalApproachEstablishedStart(KeyTimeInstanceNode):
    name = 'IAN Final Approach Established Start'
    def derive(self, ilss=S('IAN Final Approach Established')):
        for ils in ilss:
            self.create_kti(ils.slice.start)

class IANFinalApproachEstablishedEnd(KeyTimeInstanceNode):
    name = 'IAN Final Approach Established End'
    def derive(self, ilss=S('IAN Final Approach Established')):
        for ils in ilss:
            self.create_kti(ils.slice.stop)

class IANGlidepathEstablishedStart(KeyTimeInstanceNode):
    name = 'IAN Glidepath Established Start'
    def derive(self, ilss=S('IAN Glidepath Established')):
        for ils in ilss:
            self.create_kti(ils.slice.start)

class IANGlidepathEstablishedEnd(KeyTimeInstanceNode):
    name = 'IAN Glidepath Established End'
    def derive(self, ilss=S('IAN Glidepath Established')):
        for ils in ilss:
            self.create_kti(ils.slice.stop)

#################################################################
# Rejected Takeoff Start Markers (primarily for development)

class RejectedTakeoffStart(KeyTimeInstanceNode):
    def derive(self, rejs=S('Rejected Takeoff')):
        for rej in rejs:
            self.create_kti(rej.slice.start)

#################################################################


class MovementStart(KeyTimeInstanceNode):
    """
    Aircraft stops moving.
    """
    def derive(self, stationary=S('Stationary')):
        for st in stationary:
            if st.slice.stop:
                # don't create KTI at the end of data
                self.create_kti(st.slice.stop)


class MovementStop(KeyTimeInstanceNode):
    """
    Aircraft stops moving.
    """
    def derive(self, stationary=S('Stationary')):
        for st in stationary:
            if st.slice.start:
                # don't create KTI at the start of data
                self.create_kti(st.slice.start)


class OffBlocks(KeyTimeInstanceNode):
    '''
    Simple KTI derived from the first point of heading change, so probably
    pushback or start of data.
    '''

    def derive(self, mobile=S('Mobile')):
        if len(mobile):
            self.create_kti(mobile[0].slice.start or 0)


class OnBlocks(KeyTimeInstanceNode):
    '''
    Simple KTI derived from the last point of heading change.
    '''

    def derive(self, mobile=S('Mobile'), hdg=P('Heading')):
        if len(mobile):
            self.create_kti(mobile[0].slice.stop or len(hdg.array))


class FirstEngFuelFlowStart(KeyTimeInstanceNode):
    '''
    Start of fuel flow in any engine.
    '''
    def derive(self, ff=S('Eng (*) Fuel Flow')):
        ix = np.ma.argmax(ff.array > 0)
        if ix or ff.array[ix] != 0:
            self.create_kti(ix)


class LastEngFuelFlowStop(KeyTimeInstanceNode):
    '''
    Stop of fuel flow in all engines.
    '''
    def derive(self, ff=S('Eng (*) Fuel Flow')):
        slices = runs_of_ones(ff.array == 0)
        if slices:
            ix = slices[-1].start
            self.create_kti(ix)


class DistanceFromLocationMixin(object):

    def calculate(
            self, datum_lat, datum_lon, lat, lon, distance, direction='forward',
            repair_mask_duration=None, _slice=slice(None, None, None)):
        assert direction in ('forward', 'backward'), 'Unsupported direction: "%s"' % direction

        if lat.array.mask.all() or lon.array.mask.all():
            return # Either Latitude or Longitude are entirely masked

        # We can only handle single liftoffs or touchdowns at this time:
        lat_array = lat.array
        lon_array = lon.array
        if repair_mask:
            lat_array = repair_mask(lat_array, repair_duration=repair_mask_duration)
            lon_array = repair_mask(lon_array, repair_duration=repair_mask_duration)

        distances = great_circle_distance__haversine(lat_array, lon_array, [datum_lat], [datum_lon], units=ut.NM)
        if direction == 'backward':
            back_slice = slice(_slice.stop, _slice.start, -1)
            index = index_at_value(distances, distance, back_slice, endpoint='nearest')
        else:
            index = index_at_value(distances, distance, _slice)

        if index:
            # Check result is valid, as it may be the nearest but not an acceptable solution.
            error = abs(value_at_index(distances, index) - distance)
            # Allow 1/20th of a mile to reject wrong runway cases (normally > 1/10th NM apart).
            if error<0.05:
                self.create_kti(index, replace_values={'distance': distance})


class DistanceFromTakeoffAirport(KeyTimeInstanceNode, DistanceFromLocationMixin):
    '''
    Creates KTIs at certain distances from the departure airport.

    Note that we avoid using liftoff, as the distance is measured from the airport
    reference point, to avoid difference according to the runway in use.
    '''
    NAME_FORMAT = '%(distance)d NM From Takeoff Airport'
    NAME_VALUES = NAME_VALUES_DISTANCE

    def derive(self,
               lon=P('Longitude Smoothed'),
               lat=P('Latitude Smoothed'),
               airs=S('Airborne'),
               apt=A('FDR Takeoff Airport')):

        if not apt.value:
            return

        if len(airs)!=1:
            return # Only going to handle simple cases for now.

        apt_lat = apt.value.get('latitude')
        apt_lon = apt.value.get('longitude')
        for distance in self.NAME_VALUES['distance']:
            self.calculate(
                apt_lat, apt_lon, lat, lon, distance,
                direction='forward', repair_mask_duration=60,
                _slice=airs[0].slice)


class DistanceFromLandingAirport(KeyTimeInstanceNode, DistanceFromLocationMixin):
    '''
    Creates KTIs at certain distances from the arrival airport.

    The inclusion of Holding allows us to make a better estimate of the time to landing
    for cases where the aircraft sits in a holding pattern for some time before being
    cleared to land. Holding patterns are well within the 150/250nm range currently envisaged,
    but this would need to be reviewed if lower ranges were to be added.
    '''
    NAME_FORMAT = '%(distance)d NM From Landing Airport'
    NAME_VALUES = NAME_VALUES_DISTANCE

    def derive(self,
               lon=P('Longitude Smoothed'),
               lat=P('Latitude Smoothed'),
               airs=S('Airborne'),
               apt=A('FDR Landing Airport')):

        if not apt.value:
            return

        if len(airs)!=1:
            return # Only going to handle simple cases for now.

        apt_lat = apt.value.get('latitude')
        apt_lon = apt.value.get('longitude')
        for distance in self.NAME_VALUES['distance']:
            self.calculate(
                apt_lat, apt_lon, lat, lon, distance,
                direction='forward', repair_mask_duration=60,
                _slice=airs[0].slice)


class DistanceFromThreshold(KeyTimeInstanceNode, DistanceFromLocationMixin):
    '''
    Creates KTIs at certain distances from the arrival threshold.
    '''
    NAME_FORMAT = '%(distance)d NM From Threshold'
    NAME_VALUES = NAME_VALUES_RANGES

    def derive(self,
               lon=P('Longitude Smoothed'),
               lat=P('Latitude Smoothed'),
               airs=S('Airborne'),
               rwy=A('FDR Landing Runway')):

        if not rwy.value:
            return # Empty handed; nothing we can do.

        if len(airs)!=1:
            return # Only going to handle simple cases for now.

        for distance in self.NAME_VALUES['distance']:
            self.calculate(
                rwy.value['start']['latitude'],
                rwy.value['start']['longitude'],
                lat, lon, distance, direction='backward', _slice=airs[0].slice)
