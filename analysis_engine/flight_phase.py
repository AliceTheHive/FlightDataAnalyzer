from math import ceil
import numpy as np

from operator import itemgetter
from scipy.ndimage import filters
from scipy.signal import medfilt

from flightdatautilities import units as ut

from analysis_engine.node import (
    A, M, P, S, KPV, KTI, aeroplane, aeroplane_only,
    App, FlightPhaseNode, helicopter
)

from analysis_engine.library import (
    all_deps,
    all_of,
    any_of,
    any_one_of,
    bearing_and_distance,
    cycle_finder,
    filter_slices_duration,
    find_climb_cruise_descent,
    find_low_alts,
    find_nearest_slice,
    first_order_washout,
    first_valid_sample,
    heading_diff,
    index_at_value,
    integrate,
    is_index_within_slice,
    is_index_within_slices,
    is_slice_within_slice,
    last_valid_sample,
    mask_isolated_valid_samples,
    max_value,
    peak_curvature,
    rate_of_change,
    rate_of_change_array,
    repair_mask,
    runs_of_ones,
    shift_slice,
    shift_slices,
    slices_above,
    slices_and,
    slices_and_not,
    slice_duration,
    slices_extend_duration,
    slices_from_to,
    slices_not,
    slices_or,
    slices_overlap,
    slices_overlap_merge,
    slices_remove_small_gaps,
    slices_remove_small_slices
)

from analysis_engine.settings import (
    AIRBORNE_THRESHOLD_TIME,
    AIRSPEED_THRESHOLD,
    BOUNCED_LANDING_THRESHOLD,
    BOUNCED_MAXIMUM_DURATION,
    GROUNDSPEED_FOR_MOBILE,
    HEADING_RATE_FOR_FLIGHT_PHASES_FW,
    HEADING_RATE_FOR_FLIGHT_PHASES_RW,
    HEADING_RATE_FOR_MOBILE,
    HEADING_RATE_FOR_STRAIGHT_FLIGHT,
    HEADING_RATE_FOR_TAXI_TURNS,
    HEADING_TURN_OFF_RUNWAY,
    HEADING_TURN_ONTO_RUNWAY,
    HOLDING_MAX_GSPD,
    HOLDING_MIN_TIME,
    HOLDING_MIN_TURN,
    HOLDING_ROT,
    HYSTERESIS_FPALT_CCD,
    ILS_CAPTURE,
    INITIAL_CLIMB_THRESHOLD,
    LANDING_ROLL_END_SPEED,
    LANDING_THRESHOLD_HEIGHT,
    LEVEL_FLIGHT_MIN_DURATION,
    ROTORSPEED_THRESHOLD,
    TAKEOFF_ACCELERATION_THRESHOLD,
    VERTICAL_SPEED_FOR_CLIMB_PHASE,
    VERTICAL_SPEED_FOR_DESCENT_PHASE,
    VERTICAL_SPEED_FOR_LEVEL_FLIGHT,

    LANDING_COLLECTIVE_PERIOD,
    LANDING_HEIGHT,
    LANDING_TRACEBACK_PERIOD
)

from flightdatautilities.numpy_utils import slices_int

class Airborne(FlightPhaseNode):
    '''
    Periods where the aircraft is in the air, includes periods where on the
    ground for short periods (touch and go).

    TODO: Review whether depending upon the "dips" calculated by Altitude AAL
    would be more sensible as this will allow for negative AAL values longer
    than the remove_small_gaps time_limit.
    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'), seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT'):
            return False
        else:
            return 'Altitude AAL For Flight Phases' in available

    def derive(self,
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast')):
        # Remove short gaps in going fast to account for aerobatic manoeuvres
        speedy_slices = slices_remove_small_gaps(fast.get_slices(),
                                                 time_limit=60, hz=fast.frequency)

        # Just find out when altitude above airfield is non-zero.
        for speedy in speedy_slices:
            # Stop here if the aircraft never went fast.
            if speedy.start is None and speedy.stop is None:
                break

            start_point = speedy.start or 0
            stop_point = speedy.stop or len(alt_aal.array)
            # Restrict data to the fast section (it's already been repaired)
            working_alt = alt_aal.array[slices_int(start_point, stop_point)]

            # Stop here if there is inadequate airborne data to process.
            if working_alt is None or np.ma.ptp(working_alt)==0.0:
                continue
            airs = slices_remove_small_gaps(
                    np.ma.clump_unmasked(np.ma.masked_less_equal(working_alt, 1.0)),
                    time_limit=40, # 10 seconds was too short for Herc which flies below 0  AAL for 30 secs.
                    hz=alt_aal.frequency)
            # Make sure we propogate None ends to data which starts or ends in
            # midflight.
            for air in airs:
                begin = air.start
                if begin + start_point == 0: # Was in the air at start of data
                    begin = None
                else:
                    # Fine tune the liftoff to avoid transient non-zero altitudes
                    # during the takeoff run causing premature airborne state.
                    up = index_at_value(working_alt[air], 50.0)
                    if up:
                        # We did climb through 50ft, now scan back to the last time we climbed
                        # through 1ft, thereby ignoring transients, and refine the beginning point.
                        end = begin + up
                        begin = end - index_at_value(working_alt[slices_int(end, None, -1)], 1.0) + 1
                end = air.stop
                if end + start_point >= len(alt_aal.array): # Was in the air at end of data
                    end = None
                if begin is None or end is None:
                    self.create_phase(shift_slice(slice(begin, end),
                                                      start_point))
                else:
                    duration = end - begin
                    if (duration / alt_aal.hz) > AIRBORNE_THRESHOLD_TIME:
                        self.create_phase(shift_slice(slice(begin, end),
                                                          start_point))


class GoAroundAndClimbout(FlightPhaseNode):
    '''
    We already know that the Key Time Instance has been identified at the
    lowest point of the go-around, and that it lies below the 3000ft
    approach thresholds. The function here is to expand the phase 500ft before
    to the first level off after (up to 2000ft above minimum altitude).

    Uses find_low_alts to exclude level offs and level flight sections, therefore
    approach sections may finish before reaching 2000 ft above the go around.
    '''

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type'), ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return False
        correct_seg_type = seg_type and seg_type.value not in ('GROUND_ONLY', 'NO_MOVEMENT')
        return 'Altitude AAL For Flight Phases' in available and correct_seg_type

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               level_flights=S('Level Flight')):
        # Find the ups and downs in the height trace.
        low_alt_slices = find_low_alts(
            alt_aal.array, alt_aal.frequency, 3000,
            start_alt=500, stop_alt=2000,
            level_flights=level_flights.get_slices() if level_flights else None,
            relative_start=True,
            relative_stop=True,
        )
        self.create_phases(s for s in low_alt_slices if
                           alt_aal.array[int(s.start)] and
                           alt_aal.array[int(s.stop - 1)])


class Holding(FlightPhaseNode):
    """
    Holding is a process which involves multiple turns in a short period
    during the descent, normally in the same sense.

    First we compute a rate of turn parameter, then we seek periods with
    a high average rate of turn over a long period. Short turns are
    rejected and those with too small a turn or with an overall
    significant groundspeed. All these tests are to try to reject
    abnormal but not holding flight patterns.

    The final step is to trim the endpoints to the start and end of the
    first and last turns.
    """

    can_operate = aeroplane_only

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               head=P('Heading Continuous'),
               alt_max=KPV('Altitude Max'),
               lat=P('Latitude Smoothed'), lon=P('Longitude Smoothed')):

        all_turn_bands = []
        hold_bands = []

        turn_rate = rate_of_change(head, 150)
        # We scan the entire descent from highest altitude onwards.
        max_alt_to_end = [slices_int(alt_max[0].index, len(alt_aal.array))]
        above_approach = slices_above(alt_aal.array, 2000)[1]
        to_scan = slices_and(max_alt_to_end, above_approach)
        frequency = alt_aal.frequency

        for this_scan in to_scan:
            turn_right_bands = slices_extend_duration(
                np.ma.clump_unmasked(np.ma.masked_less(turn_rate[slices_int(this_scan)], HOLDING_ROT)),
                frequency, 5,
            )
            all_turn_bands.extend(shift_slices(
                slices_remove_small_gaps(turn_right_bands, time_limit=HOLDING_MIN_TIME, hz=frequency),
                this_scan.start,
            ))
            turn_left_bands = slices_extend_duration(
                np.ma.clump_unmasked(np.ma.masked_greater(turn_rate[slices_int(this_scan)], -HOLDING_ROT)),
                frequency, 5,
            )
            all_turn_bands.extend(shift_slices(
                slices_remove_small_gaps(turn_left_bands, time_limit=HOLDING_MIN_TIME, hz=frequency),
                this_scan.start,
            ))

        # Filter out slices where start or stop is None
        all_turn_bands = [turn for turn in all_turn_bands if turn.start and turn.stop]
        if len(all_turn_bands) > 1: # must have more than one turn
            for turn_band in all_turn_bands:
                # Reject short periods and check that the average groundspeed was
                # low. The index is reduced by one sample to avoid overruns, and
                # this is fine because we are not looking for great precision in
                # this test.
                hold_sec = slice_duration(turn_band, self.hz)
                if hold_sec < HOLDING_MIN_TIME:
                    continue
                start = int(turn_band.start or 0)
                stop = len(lat.array) - 1 if turn_band.stop is None else int(turn_band.stop - 1)
                _, hold_dist = bearing_and_distance(
                    lat.array[start], lon.array[start],
                    lat.array[stop], lon.array[stop])
                if hold_dist is not None:
                    average_speed = ut.convert(hold_dist / hold_sec, ut.METER_S, ut.KT)
                    if average_speed > HOLDING_MAX_GSPD:
                        continue
                angle_of_turn = head.array[stop] - head.array[start]
                if np.ma.abs(angle_of_turn) < HOLDING_MIN_TURN:
                    continue
                # So this lasted long enough, turned through a large angle and had a low
                # overall speed. Now fine tune the endpoints.
                rot_thld = HOLDING_ROT / 2.0
                diff = np.ma.abs(np.ma.ediff1d(head.array))
                # Put the start where the unfiltered rate of turn first rises...
                if diff[start] < rot_thld:
                    trim_start = index_at_value(diff, rot_thld, _slice=turn_band)
                else:
                    trim_start = index_at_value(diff, rot_thld, _slice=slice(start, 0, -1))
                # ...and the same for the endpoint.
                if diff[stop + 1] < rot_thld:
                    trim_end = index_at_value(diff, rot_thld, _slice=slice(stop + 1, start, -1))
                else:
                    trim_end = index_at_value(diff, rot_thld, _slice=slice(stop + 1, len(diff)))
                hold_bands.append(slices_int(trim_start, trim_end))

            self.create_phases(hold_bands)


class EngHotelMode(FlightPhaseNode):
    '''
    Some turbo props use the Engine 2 turbine to provide power and air whilst
    the aircraft is on the ground, a brake is applied to prevent the
    propellers from rotating
    '''

    @classmethod
    def can_operate(cls, available, family=A('Family'), ac_type=A('Aircraft Type')):
        return ac_type == aeroplane and all_deps(cls, available) and family.value in ('ATR-42', 'ATR-72') # Not all aircraft with Np will have a 'Hotel' mode


    def derive(self, eng2_np=P('Eng (2) Np'),
               eng1_n1=P('Eng (1) N1'),
               eng2_n1=P('Eng (2) N1'),
               groundeds=S('Grounded'),
               prop_brake=M('Propeller Brake')):
        pos_hotel = (eng2_n1.array > 45) & (eng2_np.array <= 0) & (eng1_n1.array < 40) & (prop_brake.array == 'On')
        hotel_mode = slices_and(runs_of_ones(pos_hotel), groundeds.get_slices())
        self.create_phases(hotel_mode)


class ApproachAndLanding(FlightPhaseNode):
    '''
    Approaches from 3000ft to lowest point in the approach (where a go around
    is performed) or down to and including the landing phase.

    Uses find_low_alts to exclude level offs and level flight sections, therefore
    approach sections may start below 3000 ft.

    Q: Suitable to replace this with BottomOfDescent and working back from
    those KTIs rather than having to deal with GoAround AND Landings?
    '''
    # Force to remove problem with desynchronising of approaches and landings
    # (when offset > 0.5)
    align_offset = 0

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'), seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT', 'START_ONLY'):
            return False
        elif ac_type == helicopter:
            return all_of(('Approach', 'Landing'), available)
        else:
            return 'Altitude AAL For Flight Phases' in available

    def _derive_aircraft(self, alt_aal, level_flights, landings):
        # Prepare to extract the slices
        level_flights = level_flights.get_slices() if level_flights else None

        low_alt_slices = slices_remove_small_slices(find_low_alts(
            alt_aal.array, alt_aal.frequency, 3000,
            stop_alt=0,
            level_flights=level_flights,
            cycle_size=500.0), 5, alt_aal.hz)

        for low_alt in low_alt_slices:
            if not alt_aal.array[int(low_alt.start)]:
                # Exclude Takeoff.
                continue

            # Include the Landing period
            if landings:
                landing = landings.get_last()
                if is_index_within_slice(landing.slice.start, low_alt):
                    low_alt = slice(low_alt.start, landing.slice.stop)
            self.create_phase(low_alt)

    def _derive_helicopter(self, apps, landings):
        phases = []
        for new_phase in apps:
            phases = slices_or(phases, [new_phase.slice])
        for new_phase in landings:
            # The phase is extended to make sure we enclose the endpoint
            # so that the touchdown point is included in this phase.
            phases = slices_or(phases, [slice(new_phase.slice.start, new_phase.slice.stop+2)])
        self.create_phases(phases)

    def derive(self,
               ac_type=A('Aircraft Type'),
               # aircraft
               alt_aal=P('Altitude AAL For Flight Phases'),
               level_flights=S('Level Flight'),
               # helicopter
               apps=S('Approach'),
               # shared
               landings=S('Landing')):

        if ac_type == helicopter:
            self._derive_helicopter(apps, landings)
        else:
            self._derive_aircraft(alt_aal, level_flights, landings)


class Approach(FlightPhaseNode):
    """
    This separates out the approach phase excluding the landing.

    Includes all approaches such as Go Arounds, but does not include any
    climbout afterwards.

    Landing starts at 50ft, therefore this phase is until 50ft.
    Uses find_low_alts to exclude level offs and level flight sections.
    """
    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type'), ac_type=A('Aircraft Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT', 'START_ONLY'):
            return False
        elif ac_type == helicopter:
            return all_of(('Altitude AGL', 'Altitude STD'), available)
        else:
            return 'Altitude AAL For Flight Phases' in available

    def _derive_aircraft(self, alt_aal, level_flights, landings):
        level_flights = level_flights.get_slices() if level_flights else None
        low_alts = find_low_alts(alt_aal.array, alt_aal.frequency, 3000,
                                 start_alt=3000, stop_alt=50,
                                 stop_mode='descent',
                                 level_flights=level_flights)
        for low_alt in low_alts:
            # Select landings only.
            if alt_aal.array[int(low_alt.start)] and \
               alt_aal.array[int(low_alt.stop)] and \
               alt_aal.array[int(low_alt.start)] > alt_aal.array[int(low_alt.stop)]:
                self.create_phase(low_alt)

    def _derive_helicopter(self, alt_agl, alt_std):
        _, apps = slices_from_to(alt_agl.array, 500, 100, threshold=1.0)
        for app, next_app in zip(apps, apps[1:]+[slice(None)]):
            begin = peak_curvature(alt_std.array,
                                   _slice=slice(app.start, app.start - 300 * alt_std.frequency, -1),
                                   curve_sense='Convex')
            end = index_at_value(alt_agl.array, 0.0,
                                 _slice=slice(app.stop, next_app.start),
                                 endpoint='first_closing')
            if begin and end:
                self.create_phase(slice(begin, end))

    def derive(self,
               ac_type=A('Aircraft Type'),
               # aircraft
               alt_aal=P('Altitude AAL For Flight Phases'),
               level_flights=S('Level Flight'),
               landings=S('Landing'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               alt_std=P('Altitude STD')):
        if ac_type == helicopter:
            self._derive_helicopter(alt_agl, alt_std)
        else:
            self._derive_aircraft(alt_aal, level_flights, landings)


class BouncedLanding(FlightPhaseNode):
    '''
    Bounced landing, defined as from first moment on ground to the final moment on the ground.

    Note: Airborne includes rejection of short segments, so the bounced period is within
    an airborne phase.
    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return ac_type == aeroplane and \
               all_of(('Altitude AAL For Flight Phases', 'Airborne'), available)

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               airs=S('Airborne'), gog=P('Gear On Ground')):
        gnds = np.ma.clump_masked(np.ma.masked_less(alt_aal.array,
                                                    BOUNCED_LANDING_THRESHOLD))
        if gog:
            gnds = slices_or(gnds, runs_of_ones(gog.array == 'Ground'))

        for air in airs:
            for gnd in gnds:
                if not is_slice_within_slice(gnd, air.slice):
                    continue
                dt = (air.slice.stop - gnd.stop) / alt_aal.frequency
                if dt > BOUNCED_MAXIMUM_DURATION or \
                   dt <= 0.0:
                    continue
                self.create_phase(slice(gnd.stop, air.slice.stop - 1))


class ClimbCruiseDescent(FlightPhaseNode):
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               airs=S('Airborne')):
        for air in airs:
            try:
                alts = repair_mask(alt_std.array[air.slice], repair_duration=None)
            except:
                # Short segments may be wholly masked. We ignore these.
                continue

            section_slices = find_climb_cruise_descent(alts)
            section_slices = shift_slices(section_slices, air.slice.start or 0)
            self.create_sections(section_slices)


"""
class CombinedClimb(FlightPhaseNode):
    '''
    Climb phase from liftoff or go around to top of climb
    '''
    def derive(self,
               toc=KTI('Top Of Climb'),
               ga=KTI('Go Around'),
               lo=KTI('Liftoff'),
               touchdown=KTI('Touchdown')):

        end_list = [x.index for x in toc.get_ordered_by_index()]
        start_list = [y.index for y in [lo.get_first()] + ga.get_ordered_by_index()]
        assert len(start_list) == len(end_list)

        slice_idxs = zip(start_list, end_list)
        for slice_tuple in slice_idxs:
            self.create_phase(slice(*slice_tuple))
"""


class Climb(FlightPhaseNode):
    '''
    This phase goes from 1000 feet (top of Initial Climb) in the climb to the
    top of climb
    '''
    def derive(self,
               toc=KTI('Top Of Climb'),
               eot=KTI('Climb Start')): # AKA End Of Initial Climb
        # First we extract the kti index values into simple lists.
        toc_list = []
        for this_toc in toc:
            toc_list.append(this_toc.index)

        # Now see which follows a takeoff
        for this_eot in eot:
            eot = this_eot.index
            # Scan the TOCs
            closest_toc = None
            for this_toc in toc_list:
                if (eot < this_toc and
                    ((closest_toc and this_toc < closest_toc)
                     or
                     closest_toc is None)):
                    closest_toc = this_toc
            # Build the slice from what we have found.
            if ((closest_toc - eot) / self.hz) <= 2:
                continue # skip brief climbs
            self.create_phase(slice(eot, closest_toc))


class Climbing(FlightPhaseNode):
    def derive(self, vert_spd=P('Vertical Speed For Flight Phases'),
               airs=S('Airborne')):
        # Climbing is used for data validity checks and to reinforce regimes.
        for air in airs:
            climbing = np.ma.masked_less(vert_spd.array[air.slice],
                                         VERTICAL_SPEED_FOR_CLIMB_PHASE)
            climbing_slices = slices_remove_small_gaps(
                np.ma.clump_unmasked(climbing), time_limit=30.0, hz=vert_spd.hz)
            self.create_phases(shift_slices(climbing_slices, air.slice.start))


class Cruise(FlightPhaseNode):
    def derive(self,
               ccds=S('Climb Cruise Descent'),
               tocs=KTI('Top Of Climb'),
               tods=KTI('Top Of Descent'),
               air_spd=P('Airspeed')):
        # We may have many phases, tops of climb and tops of descent at this
        # time.
        # The problem is that they need not be in tidy order as the lists may
        # not be of equal lengths.

        # ensure traveling greater than 50 kts in cruise
        # scope = slices_and(slices_above(air_spd.array, 50)[1], ccds.get_slices())
        scope = ccds.get_slices()
        for ccd in scope:
            toc = tocs.get_first(within_slice=ccd)
            if toc:
                begin = toc.index
            else:
                begin = ccd.start

            tod = tods.get_last(within_slice=ccd)
            if tod:
                end = tod.index
            else:
                end = ccd.stop

            # Some flights just don't cruise. This can cause headaches later
            # on, so we always cruise for at least one second !
            if end is not None and end <= (begin or 0):
                end = (begin or 0) + 1

            self.create_phase(slice(begin, end))


class InitialCruise(FlightPhaseNode):
    '''
    This is a period from five minutes into the cruise lasting for 30
    seconds, and is used to establish average conditions for fuel monitoring
    programmes.
    '''

    align_frequency = 1.0
    align_offset = 0.0

    can_operate = aeroplane_only

    def derive(self, cruises=S('Cruise')):
        if not cruises:
            return
        cruise = cruises[0].slice
        if cruise.stop - cruise.start > 330:
            self.create_phase(slice(cruise.start+300, cruise.start+330))

"""
class CombinedDescent(FlightPhaseNode):
    def derive(self,
               tod_set=KTI('Top Of Descent'),
               bod_set=KTI('Bottom Of Descent'),
               liftoff=KTI('Liftoff'),
               touchdown=KTI('Touchdown')):

        end_list = [x.index for x in bod_set.get_ordered_by_index()]
        start_list = [y.index for y in tod_set.get_ordered_by_index()]
        assert len(start_list) == len(end_list)

        slice_idxs = zip(start_list, end_list)
        for slice_tuple in slice_idxs:
            self.create_phase(slice(*slice_tuple))
"""

class Descending(FlightPhaseNode):
    """
    Descending faster than 500fpm towards the ground
    """
    def derive(self, vert_spd=P('Vertical Speed For Flight Phases'),
               airs=S('Airborne')):
        # Vertical speed limits of 500fpm gives good distinction with level
        # flight.
        for air in airs:
            descending = np.ma.masked_greater(vert_spd.array[air.slice],
                                              VERTICAL_SPEED_FOR_DESCENT_PHASE)
            desc_slices = slices_remove_small_slices(np.ma.clump_unmasked(descending))
            self.create_phases(shift_slices(desc_slices, air.slice.start))


class Descent(FlightPhaseNode):
    def derive(self,
               tod_set=KTI('Top Of Descent'),
               bod_set=KTI('Bottom Of Descent')):

        start_list = [y.index for y in tod_set.get_ordered_by_index()]
        end_list = [x.index for x in bod_set.get_ordered_by_index()]
        assert len(start_list) == len(end_list)

        slice_idxs = zip(start_list, end_list)
        for slice_tuple in slice_idxs:
            self.create_phase(slice(*slice_tuple))


class DescentToFlare(FlightPhaseNode):
    '''
    Descent phase down to 50ft.
    '''

    def derive(self,
            descents=S('Descent'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        #TODO: Ensure we're still in the air
        for descent in descents:
            end = index_at_value(alt_aal.array, 50.0, descent.slice)
            if end is None:
                end = descent.slice.stop
            self.create_phase(slice(descent.slice.start, end))


class DescentLowClimb(FlightPhaseNode):
    '''
    Finds where the aircaft descends below the INITIAL_APPROACH_THRESHOLD and
    then climbs out again - an indication of a go-around.

    TODO: Consider refactoring this based on the Bottom Of Descent KTIs and
    just check the altitude at each BOD.
    '''

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type'), ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return False

        correct_seg_type = seg_type and seg_type.value not in ('GROUND_ONLY', 'NO_MOVEMENT')
        return 'Altitude AAL For Flight Phases' in available and correct_seg_type

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               level_flights=S('Level Flight')):
        level_flights = level_flights.get_slices() if level_flights else None
        low_alt_slices = find_low_alts(alt_aal.array, alt_aal.frequency,
                                       500,
                                       3000,
                                       level_flights=level_flights)
        for low_alt in low_alt_slices:
            if (alt_aal.array[int(low_alt.start)] and
                alt_aal.array[int(low_alt.stop - 1)]):
                self.create_phase(low_alt)


class Fast(FlightPhaseNode):
    '''
    Data will have been sliced into single flights before entering the
    analysis engine, so we can be sure that there will be only one fast
    phase. This may have masked data within the phase, but by taking the
    notmasked edges we enclose all the data worth analysing.

    Therefore len(Fast) in [0,1]

    TODO: Discuss whether this assertion is reliable in the presence of air data corruption.
    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    seg_type=A('Segment Type')):
        if ac_type == helicopter:
            return 'Nr' in available
        else:
            return seg_type and seg_type.value == 'START_AND_STOP' and 'Airspeed' in available

    def derive(self, airspeed=P('Airspeed'), rotor_speed=P('Nr'),
               ac_type=A('Aircraft Type'), eng_prop=A('Engine Propulsion')):
        """
        Did the aircraft go fast enough to possibly become airborne?

        # We use the same technique as in index_at_value where transition of
        # the required threshold is detected by summing shifted difference
        # arrays. This has the particular advantage that we can reject
        # excessive rates of change related to data dropouts which may still
        # pass the data validation stage.
        value_passing_array = (airspeed.array[0:-2]-AIRSPEED_THRESHOLD) * \
            (airspeed.array[1:-1]-AIRSPEED_THRESHOLD)
        test_array = np.ma.masked_outside(value_passing_array, 0.0, -100.0)
        """

        if ac_type == helicopter:
            nr = repair_mask(rotor_speed.array, repair_duration=600,
                             raise_entirely_masked=False, copy=True)
            fast = np.ma.masked_less(nr, ROTORSPEED_THRESHOLD)
            fast_slices = np.ma.clump_unmasked(fast)
        else:
            if eng_prop and eng_prop.value == 'PROP':
                spd_threshold = 60
            else:
                spd_threshold = AIRSPEED_THRESHOLD

            ias = airspeed.array.copy()
            # Mask isolated (single sample) of "valid" data surrounded by masked data
            mask_isolated_valid_samples(ias)
            ias = repair_mask(ias, repair_duration=600,
                          raise_entirely_masked=False)
            fast = np.ma.masked_less(ias, spd_threshold)
            fast_slices = np.ma.clump_unmasked(fast)
            fast_slices = slices_remove_small_gaps(fast_slices, time_limit=30,
                                                   hz=self.frequency)

        fast_slices = slices_remove_small_slices(fast_slices, time_limit=10,
                                                     hz=self.frequency)
        self.create_phases(fast_slices)


class FinalApproach(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               airs=S('Airborne')):
        # Airborne dependency added as we should not be approaching if never airborne
        self.create_phases(alt_aal.slices_from_to(1000, 50))


class GearExtending(FlightPhaseNode):
    '''
    Gear extending and retracting are section nodes, as they last for a
    finite period. Based on the Gear Red Warnings.

    For some aircraft no parameters to identify the transit are recorded, so
    a nominal period is included in Gear Down Selected Calculations to
    allow for exceedance of gear transit limits.
    '''

    def derive(self, down_in_transit=M('Gear Down In Transit'), airs=S('Airborne')):

        in_transit = (down_in_transit.array == 'Extending')
        gear_extending = slices_and(runs_of_ones(in_transit), airs.get_slices())
        self.create_phases(gear_extending)


class GearExtended(FlightPhaseNode):
    '''
    Simple phase translation of the Gear Down parameter.
    '''
    def derive(self, gear_down=M('Gear Down')):
        repaired = repair_mask(gear_down.array, gear_down.frequency,
                               repair_duration=120, extrapolate=True,
                               method='fill_start')
        gear_dn = runs_of_ones(repaired == 'Down')
        # remove single sample changes from this phase
        # note: order removes gaps before slices for Extended
        slices_remove_small_bits = lambda g: slices_remove_small_slices(
            slices_remove_small_gaps(g, count=2), count=2)
        self.create_phases(slices_remove_small_bits(gear_dn))


class GearRetracting(FlightPhaseNode):
    '''
    Gear extending and retracting are section nodes, as they last for a
    finite period. Based on the Gear Red Warnings.

    For some aircraft no parameters to identify the transit are recorded, so
    a nominal period is included in Gear Up Selected Calculations to
    allow for exceedance of gear transit limits.
    '''

    def derive(self, up_in_transit=M('Gear Up In Transit'), airs=S('Airborne')):

        in_transit = (up_in_transit.array == 'Retracting')
        gear_retracting = slices_and(runs_of_ones(in_transit), airs.get_slices())
        self.create_phases(gear_retracting)


class GearRetracted(FlightPhaseNode):
    '''
    Simple phase translation of the Gear Down parameter to show gear Up.
    '''
    def derive(self, gear_up=M('Gear Up')):
        repaired = repair_mask(gear_up.array, gear_up.frequency,
                               repair_duration=120, extrapolate=True,
                               method='fill_start')
        gear_up = runs_of_ones(repaired == 'Up')
        # remove single sample changes from this phase
        # note: order removes slices before gaps for Retracted
        slices_remove_small_bits = lambda g: slices_remove_small_gaps(
            slices_remove_small_slices(g, count=2), count=2)
        self.create_phases(slices_remove_small_bits(gear_up))


def scan_ils(beam, ils_dots, height, scan_slice, frequency,
             hdg=None, hdg_ldg=None, duration=10):
    '''
    Scans ils dots and returns last slice where ils dots fall below ILS_CAPTURE and remain below 2.5 dots
    if beam is glideslope slice will not extend below 200ft.

    :param beam: 'localizer' or 'glideslope'
    :type beam: str
    :param ils_dots: localizer deviation in dots
    :type ils_dots: Numpy array
    :param height: Height above airfield
    :type height: Numpy array
    :param scan_slice: slice to be inspected
    :type scan_slice: python slice
    :param frequency: input signal sample rate
    :type frequency: float
    :param hdg: aircraft heaing
    :type hdg: Numpy array
    :param hdg_ldg: Heading on the landing roll
    :type hdg_ldg: list fo Key Point Values
    :param duration: Minimum duration for the ILS to be established
    :type duration: float, default = 10 seconds.
    '''
    if beam not in ['localizer', 'glideslope']:
        raise ValueError('Unrecognised beam type in scan_ils')

    if beam=='localizer' and hdg_ldg:
        # Restrict the scan to approaches facing the runway This restriction
        # will be carried forward into the glideslope calculation by the
        # restricted duration of localizer established, hence does not need
        # to be repeated.
        hdg_landing = None
        for each_ldg in hdg_ldg:
            if is_index_within_slice(each_ldg.index, scan_slice):
                hdg_landing = each_ldg.value
                break

        if hdg_landing:
            diff = np.ma.abs(heading_diff(hdg[scan_slice] % 360, hdg_landing))
            facing = shift_slices(
                np.ma.clump_unmasked(np.ma.masked_greater(diff, 90.0)),
                scan_slice.start)
            scan_slice = slices_and([scan_slice], facing)[-1]


    if np.ma.count(ils_dots[scan_slice]) < duration*frequency:
        # less than duration seconds of valid data within slice
        return None

    # Find the range of valid ils dots withing scan slice
    valid_ends = np.ma.flatnotmasked_edges(ils_dots[scan_slice])
    if valid_ends is None:
        return None
    valid_slice = slice(*(valid_ends+scan_slice.start))
    if np.ma.count(ils_dots[valid_slice])/float(len(ils_dots[valid_slice])) < 0.4:
        # less than 40% valid data within valid data slice
        return None

    # get abs of ils dots as its used everywhere and repair small masked periods
    ils_abs = repair_mask(np.ma.abs(ils_dots), frequency=frequency, repair_duration=5)

    # ----------- Find loss of capture

    last_valid_idx, last_valid_value = last_valid_sample(ils_abs[scan_slice])

    if last_valid_value < 2.5:
        # finished established ? if established in first place
        ils_lost_idx = scan_slice.start + last_valid_idx + 1
    else:
        # find last time went below 2.5 dots
        last_25_idx = index_at_value(ils_abs, 2.5, slice(scan_slice.stop, scan_slice.start, -1))
        if last_25_idx is None:
            # never went below 2.5 dots
            return None
        else:
            # Round down to nearest integer
            ils_lost_idx = int(last_25_idx)

    if beam == 'glideslope':
        # If Glideslope find index of height last passing 200ft and use the
        # smaller of that and any index where the ILS was lost
        idx_200 = index_at_value(height, 200, slice(scan_slice.stop,
                                                scan_slice.start, -1),
                             endpoint='closing')
        if idx_200 is not None:
            ils_lost_idx = min(ils_lost_idx, idx_200) + 1

        if np.ma.count(ils_dots[slices_int(scan_slice.start, ils_lost_idx)]) < 5:
            # less than 5 valid values within remaining section
            return None

    # ----------- Find start of capture

    # Find where to start scanning for the point of "Capture", Look for the
    # last time we were within 2.5dots
    scan_start_idx = index_at_value(ils_abs, 2.5, slice(ils_lost_idx-1, scan_slice.start-1, -1))

    first_valid_idx, first_valid_value = first_valid_sample(
        ils_abs[slices_int(scan_slice.start, ils_lost_idx)]
    )

    ils_capture_idx = None
    if scan_start_idx or (first_valid_value > ILS_CAPTURE):
        # Look for first instance of being established
        if not scan_start_idx:
            scan_start_idx = index_at_value(ils_abs, ILS_CAPTURE, slice(scan_slice.start, ils_lost_idx))
        if scan_start_idx is None:
            # didnt start established and didnt move within 2.5 dots
            return None
        half_dot = np.ma.masked_greater(ils_abs, 0.5)
        est = np.ma.clump_unmasked(half_dot)
        est_slices = slices_and(est, (slice(scan_start_idx, ils_lost_idx),))
        est_slices = slices_remove_small_slices(est_slices, duration, hz=frequency)
        if est_slices:
            ils_capture_idx = est_slices[0].start
        else:
            return None
    elif first_valid_value < ILS_CAPTURE:
        # started established
        ils_capture_idx = scan_slice.start + first_valid_idx
    if first_valid_idx is None:
        # no valid data
        return None

    if ils_capture_idx is None or ils_lost_idx is None:
        return None
    else:
        # OK, we have seen an ILS signal, but let's make sure we did respond
        # to it. The test here is to make sure that we didn't just pass
        # through the beam (L>R or R>L or U>D or D>U) without making an
        # effort to correct the variation.
        ils_slice = slice(ils_capture_idx, ils_lost_idx)
        width = 5.0
        if frequency < 0.5:
            width = 10.0
        ils_rate = rate_of_change_array(ils_dots[slices_int(ils_slice)],
                                        frequency, width=width,
                                        method='regression')
        top = max(ils_rate)
        bottom = min(ils_rate)
        if top*bottom > 0.0:
            # the signal never changed direction, so went straight through
            # the beam without getting established...
            return None
        else:
            # Hurrah! We did capture the beam
            return ils_slice


class IANFinalApproachCourseEstablished(FlightPhaseNode):
    name = 'IAN Final Approach Established'

    def derive(self,
               ian_final=P('IAN Final Approach Course'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               apps=S('Approach Information'),
               ils_freq=P('ILS Frequency'),
               app_src_capt=M('Displayed App Source (Capt)'),
               app_src_fo=M('Displayed App Source (FO)')):

        def create_ils_phases(slices):
            for _slice in slices:
                ils_slice = scan_ils('localizer', ian_final.array, alt_aal.array,
                                     _slice, ian_final.frequency)
                if ils_slice is not None:
                    self.create_phase(ils_slice)

        # Displayed App Source required to ensure that IAN is being followed
        in_fmc = (app_src_capt.array == 'FMC') | (app_src_fo.array == 'FMC')
        ian_final.array[~in_fmc] = np.ma.masked

        for app in apps:
            if app.loc_est:
                # Mask IAN data for approaches where ILS is established
                ian_final.array[app.slice] = np.ma.masked
                continue

            if not np.ma.count(ian_final.array[app.slice]):
                # No valid IAN Final Approach data for this approach.
                continue
            valid_slices = np.ma.clump_unmasked(ian_final.array[app.slice])
            valid_slices = slices_remove_small_gaps(valid_slices, count=5)
            last_valid_slice = shift_slice(valid_slices[-1], app.slice.start)
            create_ils_phases([last_valid_slice])


class IANGlidepathEstablished(FlightPhaseNode):
    name = 'IAN Glidepath Established'

    def derive(self,
               ian_glidepath=P('IAN Glidepath'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               apps=App('Approach Information'),
               app_src_capt=P('Displayed App Source (Capt)'),
               app_src_fo=P('Displayed App Source (FO)')):

        def create_ils_phases(slices):
            for _slice in slices:
                ils_slice = scan_ils('glideslope', ian_glidepath.array,
                                     alt_aal.array, _slice,
                                     ian_glidepath.frequency)
                if ils_slice is not None:
                    self.create_phase(ils_slice)

        # Assumption here is that Glidepath is not as tightly coupled with
        # Final Approach Course as Glideslope is to Localiser

        # Displayed App Source required to ensure that IAN is being followed
        in_fmc = (app_src_capt.array == 'FMC') | (app_src_fo.array == 'FMC')
        ian_glidepath.array[~in_fmc] = np.ma.masked

        for app in apps:
            if app.gs_est:
                # Mask IAN data for approaches where ILS is established
                ian_glidepath.array[app.slice] = np.ma.masked
                continue

            if not np.ma.count(ian_glidepath.array[app.slice]):
                # No valid ian glidepath data for this approach.
                continue
            valid_slices = np.ma.clump_unmasked(ian_glidepath.array[app.slice])
            valid_slices = slices_remove_small_gaps(valid_slices, count=5)
            last_valid_slice = shift_slice(valid_slices[-1], app.slice.start)
            create_ils_phases([last_valid_slice])


class ILSLocalizerEstablished(FlightPhaseNode):
    name = 'ILS Localizer Established'

    def derive(self, apps=App('Approach Information')):
        for app in apps:
            if app.loc_est:
                self.create_phase(app.loc_est)

'''
class ILSApproach(FlightPhaseNode):
    name = "ILS Approach"
    """
    Where a Localizer Established phase exists, extend the start and end of
    the phase back to 3 dots (i.e. to beyond the view of the pilot which is
    2.5 dots) and assign this to ILS Approach phase. This period will be used
    to determine the range for the ILS display on the web site and for
    examination for ILS KPVs.
    """
    def derive(self, ils_loc = P('ILS Localizer'),
               ils_loc_ests = S('ILS Localizer Established')):
        # For most of the flight, the ILS will not be valid, so we scan only
        # the periods with valid data, ignoring short breaks:
        locs = np.ma.clump_unmasked(repair_mask(ils_loc.array))
        for loc_slice in locs:
            for ils_loc_est in ils_loc_ests:
                est_slice = ils_loc_est.slice
                if slices_overlap(loc_slice, est_slice):
                    before_established = slice(est_slice.start, loc_slice.start, -1)
                    begin = index_at_value(np.ma.abs(ils_loc.array),
                                                     3.0,
                                                     _slice=before_established)
                    end = est_slice.stop
                    self.create_phase(slice(begin, end))
'''


class ILSGlideslopeEstablished(FlightPhaseNode):
    name = "ILS Glideslope Established"

    def derive(self, apps=App('Approach Information')):
        for app in apps:
            if app.gs_est:
                self.create_phase(app.gs_est)


class InitialApproach(FlightPhaseNode):
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               app_lands=S('Approach')):
        for app_land in app_lands:
            # We already know this section is below the start of the initial
            # approach phase so we only need to stop at the transition to the
            # final approach phase.
            ini_app = np.ma.masked_where(alt_AAL.array[app_land.slice]<1000,
                                         alt_AAL.array[app_land.slice])
            if not len(ini_app):
                continue
            phases = np.ma.clump_unmasked(ini_app)
            for phase in phases:
                begin = phase.start
                pit = np.ma.argmin(ini_app[phase]) + begin
                if ini_app[pit] < ini_app[begin] :
                    self.create_phases(shift_slices([slice(begin, pit)],
                                                    app_land.slice.start))


class InitialClimb(FlightPhaseNode):
    '''
    Phase from end of Takeoff (35ft) to start of climb (1000ft)
    '''
    def derive(self,
               takeoffs=S('Takeoff'),
               climb_starts=KTI('Climb Start'),
               tocs=KTI('Top Of Climb'),
               alt=P('Altitude STD'),
               ac_type=A('Aircraft Type')):

        # If max alt is above 1000 ft we don't need to consider the ToC
        # point in our calculations for aeroplanes.
        if alt and np.ma.max(alt.array) > 1000 and ac_type != helicopter:
            to_scan = [[t.stop_edge, 'takeoff'] for t in takeoffs] + \
                [[c.index, 'climb'] for c in climb_starts]
        else:
            to_scan = [[t.stop_edge, 'takeoff'] for t in takeoffs] + \
                [[c.index, 'climb'] for c in climb_starts]+ \
                [[c.index, 'climb'] for c in tocs]
        to_scan = sorted(to_scan, key=itemgetter(0))
        for i in range(len(to_scan)-1):
            if to_scan[i][1]=='takeoff' and to_scan[i+1][1]=='climb':
                begin = to_scan[i][0]
                end = to_scan[i+1][0]
                self.create_phase(slice(begin, end), begin=begin, end=end)


class LevelFlight(FlightPhaseNode):
    '''
    Level flight for at least 20 seconds.

    This now excludes extended touch and go operations which are level, but
    below 5ft above the runway. We have seen almost a minute on the runway,
    so this algorithm does not include a time limit for such actions.
    '''
    def derive(self,
               airs=S('Airborne'),
               vrt_spd=P('Vertical Speed For Flight Phases'),
               alt_aal=P('Altitude AAL')):

        for air in airs:
            limit = VERTICAL_SPEED_FOR_LEVEL_FLIGHT
            level_flight = np.ma.masked_outside(vrt_spd.array[air.slice], -limit, limit)
            level_flight_slices = np.ma.clump_unmasked(level_flight)
            above_runway = np.ma.masked_less(alt_aal.array[air.slice], 5.0)
            above_runway_slices = np.ma.clump_unmasked(above_runway)
            level_slices = slices_and(level_flight_slices, above_runway_slices)
            level_slices = slices_remove_small_slices(level_slices,
                                                      time_limit=LEVEL_FLIGHT_MIN_DURATION,
                                                      hz=vrt_spd.frequency)
            self.create_phases(shift_slices(level_slices, air.slice.start))



class StraightAndLevel(FlightPhaseNode):
    '''
    Building on Level Flight, this checks for straight flight. We use heading
    rate as more sensitive than roll attitude and sticking to the core three
    parameters.
    '''
    def derive(self,
               levels=S('Level Flight'),
               hdg=P('Heading')):

        for level in levels:
            limit = HEADING_RATE_FOR_STRAIGHT_FLIGHT
            rot = rate_of_change_array(hdg.array[level.slice], hdg.frequency, width=30)
            straight_flight = np.ma.masked_outside(rot, -limit, limit)
            straight_slices = np.ma.clump_unmasked(straight_flight)
            straight_and_level_slices = slices_remove_small_slices(
                straight_slices, time_limit=LEVEL_FLIGHT_MIN_DURATION,
                hz=hdg.frequency)
            self.create_phases(shift_slices(straight_and_level_slices, level.slice.start))


class Grounded(FlightPhaseNode):
    '''
    Includes start of takeoff run and part of landing run.
    Was "On Ground" but this name conflicts with a recorded 737-6 parameter name.
    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return all_of(('Airborne', 'Airspeed'), available)
        else:
            return 'HDF Duration' in available

    def _derive_aircraft(self, speed, hdf_duration, air):
        data_end = hdf_duration.value * self.frequency if hdf_duration else None
        if air:
            gnd_phases = slices_not(air.get_slices(), begin_at=0, end_at=data_end)
            if not gnd_phases:
                # Either all on ground or all in flight.
                median_speed = np.ma.median(speed.array)
                if median_speed > AIRSPEED_THRESHOLD:
                    gnd_phases = [slice(None,None,None)]
                else:
                    gnd_phases = [slice(0,data_end,None)]
        else:
            # no airborne so must be all on ground
            gnd_phases = [slice(0,data_end,None)]

        self.create_phases(gnd_phases)

    def _derive_helicopter(self, air, airspeed):
        '''
        Needed for AP Engaged KPV.
        '''
        all_data = slice(0, len(airspeed.array))
        self.create_sections(slices_and_not([all_data], air.get_slices()))

    def derive(self,
               airspeed=P('Airspeed'),
               air=S('Airborne'),
               ac_type=A('Aircraft Type'),
               hdf_duration=A('HDF Duration')):
        if ac_type == helicopter:
            self._derive_helicopter(air, airspeed)
        else:
            self._derive_aircraft(airspeed, hdf_duration, air)


class Taxiing(FlightPhaseNode):
    '''
    This finds the first and last signs of movement to provide endpoints to
    the taxi phases.

    If groundspeed is available, only periods where the groundspeed is over
    5kts are considered taxiing.

    With all mobile and moving periods identified, we then remove all the
    periods where the aircraft is either airborne, taking off, landing or
    carrying out a rejected takeoff. What's left are the taxiing on the
    ground periods.
    '''

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type')):
        default = all_of(('Mobile', 'Takeoff', 'Landing', 'Airborne'), available)
        ground_only = seg_type and seg_type.value == 'GROUND_ONLY' and \
            'Mobile' in available
        return default or ground_only

    def derive(self, mobiles=S('Mobile'), gspd=P('Groundspeed'),
               toffs=S('Takeoff'), lands=S('Landing'),
               rtos=S('Rejected Takeoff'),
               airs=S('Airborne')):
        # XXX: There should only be one Mobile slice.
        if gspd:
            # Limit to where Groundspeed is above GROUNDSPEED_FOR_MOBILE.
            taxiing_slices = np.ma.clump_unmasked(np.ma.masked_less
                                                  (np.ma.abs(gspd.array),
                                                   GROUNDSPEED_FOR_MOBILE))
            taxiing_slices = slices_and(mobiles.get_slices(), taxiing_slices)
        else:
            taxiing_slices = mobiles.get_slices()

        if toffs:
            taxiing_slices = slices_and_not(taxiing_slices, toffs.get_slices())
        if lands:
            taxiing_slices = slices_and_not(taxiing_slices, lands.get_slices())
        if rtos:
            taxiing_slices = slices_and_not(taxiing_slices, rtos.get_slices())
        if airs:
            taxiing_slices = slices_and_not(taxiing_slices, airs.get_slices())

        self.create_phases(taxiing_slices)


class Mobile(FlightPhaseNode):
    '''
    This finds the first and last signs of movement to provide endpoints to
    the taxi phases. As Heading Rate is derived directly from heading, this
    phase is guaranteed to be operable for very basic aircraft.
    '''

    @classmethod
    def can_operate(cls, available):
        return 'Heading Rate' in available

    def derive(self,
               rot=P('Heading Rate'),
               gspd=P('Groundspeed'),
               airs=S('Airborne'),
               fast=S('Fast'),
               #power=P('Eng (*) Any Running'),
               ):

        turning = np.ma.masked_less(np.ma.abs(rot.array), HEADING_RATE_FOR_MOBILE)
        movement = np.ma.flatnotmasked_edges(turning)
        start, stop = movement if movement is not None else (None, None)

        if gspd is not None:
            moving = np.ma.masked_less(np.ma.abs(gspd.array), GROUNDSPEED_FOR_MOBILE)
            mobile = np.ma.flatnotmasked_edges(moving)
            if mobile is not None:
                start = min(start, mobile[0]) if start else mobile[0]
                stop = max(stop, mobile[1]) if stop else mobile[1]
        elif fast:
            fast_start = fast.get_first().slice.start
            fast_stop = fast.get_last().slice.stop
            start = min(start, fast_start) if start else fast_start
            stop = max(stop, fast_stop) if stop else fast_stop

        if airs and airs is not None:
            start = min(start, airs[0].slice.start) if start is not None else airs[0].slice.start
            stop = max(stop, airs[-1].slice.stop) if stop else airs[-1].slice.stop

        self.create_phase(slice(start, stop))


class Stationary(FlightPhaseNode):
    """
    Phases of the flight when the aircraft remains stationary.

    This is useful in fuel monitoring.
    """
    def derive(self,
               gspd=P('Groundspeed')):
        not_moving = runs_of_ones(gspd.array < GROUNDSPEED_FOR_MOBILE)
        self.create_phases(slices_remove_small_gaps(not_moving, time_limit=5, hz=gspd.frequency))


class Landing(FlightPhaseNode):
    '''
    This flight phase starts at 50 ft in the approach and ends as the
    aircraft turns off the runway. Subsequent KTIs and KPV computations
    identify the specific moments and values of interest within this phase.

    We use Altitude AAL (not "for Flight Phases") to avoid small errors
    introduced by hysteresis, which is applied to avoid hunting in level
    flight conditions, and thereby make sure the 50ft startpoint is exact.
    '''
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'), seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT'):
            return False
        elif ac_type == helicopter:
            return all_of(('Altitude AGL', 'Collective', 'Airborne'), available)
        else:
            return 'Altitude AAL For Flight Phases' in available

    def _derive_aircraft(self, head, alt_aal, fast, mobile):
        phases = []
        for speedy in fast:
            # See takeoff phase for comments on how the algorithm works.

            # AARRGG - How can we check if this is at the end of the data
            # without having to go back and test against the airspeed array?
            # TODO: Improve endpoint checks. DJ
            # Answer:
            #  duration=A('HDF Duration')
            #  array_len = duration.value * self.frequency
            #  if speedy.slice.stop >= array_len: continue

            if (speedy.slice.stop is None or \
                speedy.slice.stop >= len(alt_aal.array) - 2):
                break

            landing_run = speedy.slice.stop + 2
            datum = head.array[landing_run]

            first = landing_run - (300 * alt_aal.frequency)
            # Limit first to be the latest of 5 mins or maximum altitude
            # during fast slice to account for short flights
            first = max(first, max_value(alt_aal.array, _slice=slice(speedy.slice.start, landing_run)).index)+2
            landing_begin = index_at_value(alt_aal.array,
                                           LANDING_THRESHOLD_HEIGHT,
                                           slice(landing_run, first, -1))
            if landing_begin is None:
                # we are not able to detect a landing threshold height,
                # therefore invalid section
                continue

            # The turn off the runway must lie within eight minutes of the
            # landing. (We did use 5 mins, but found some landings on long
            # runways where the turnoff did not happen for over 6 minutes
            # after touchdown).
            last = landing_run + (480 * head.frequency)

            # A crude estimate is given by the angle of turn
            landing_end = index_at_value(np.ma.abs(head.array-datum),
                                         HEADING_TURN_OFF_RUNWAY,
                                         slice(landing_run, last))
            if landing_end is None:
                # The data ran out before the aircraft left the runway so use
                # end of mobile or remainder of data.
                landing_end = mobile.get_last().slice.stop if mobile else len(head.array)-1

            # ensure any overlap with phases are ignored (possibly due to
            # data corruption returning multiple fast segments)
            new_phase = [slice(landing_begin, landing_end)]
            phases = slices_or(phases, new_phase)
        self.create_phases(phases)

    def _derive_helicopter(self, alt_agl, coll, airs):
        phases = []
        for air in airs:
            tdn = air.stop_edge
            # Scan back to find either when we descend through LANDING_HEIGHT or had peak hover height.
            to_scan = tdn - alt_agl.frequency*LANDING_TRACEBACK_PERIOD
            landing_begin = index_at_value(alt_agl.array, LANDING_HEIGHT,
                                           _slice=slice(tdn, to_scan , -1),
                                           endpoint='exact')

            # Scan forwards to find lowest collective shortly after touchdown.
            to_scan = tdn + coll.frequency*LANDING_COLLECTIVE_PERIOD
            landing_end = tdn  + np.ma.argmin(coll.array[slices_int(tdn,to_scan)])
            if landing_begin and landing_end:
                new_phase = [slice(landing_begin, landing_end)]
                phases = slices_or(phases, new_phase)
        self.create_phases(phases)

    def derive(self,
               ac_type=A('Aircraft Type'),
               # aircraft
               head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast'),
               mobile=S('Mobile'),
               # helicopter
               alt_agl=P('Altitude AGL'),
               coll=P('Collective'),
               airs=S('Airborne')):
        if ac_type == helicopter:
            self._derive_helicopter(alt_agl, coll, airs)
        else:
            self._derive_aircraft(head, alt_aal, fast, mobile)


class LandingRoll(FlightPhaseNode):
    '''
    FDS developed this node to support the UK CAA Significant Seven
    programme. This phase is used when computing KPVs relating to the
    deceleration phase of the landing.

    "CAA to go with T/D to 60 knots with the T/D defined as less than 2 deg
    pitch (after main gear T/D)."

    The complex index_at_value ensures that if the aircraft does not flare to
    2 deg, we still capture the highest attitude as the start of the landing
    roll, and the landing roll starts as the aircraft passes 2 deg the last
    time, i.e. as the nosewheel comes down and not as the flare starts.
    '''
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return 'Landing' in available and any_of(('Airspeed True', 'Groundspeed'), available) \
               and ac_type == aeroplane

    def derive(self, pitch=P('Pitch'), gspd=P('Groundspeed'),
               aspd=P('Airspeed True'), lands=S('Landing')):
        if gspd:
            speed = gspd.array
        else:
            speed = aspd.array
        for land in lands:
            # Airspeed True on some aircraft do not record values below 61
            end = index_at_value(speed, LANDING_ROLL_END_SPEED, land.slice)
            if end is None:
                # due to masked values, use the land.stop rather than
                # searching from the end of the data
                end = land.slice.stop
            begin = None
            if pitch:
                begin = index_at_value(pitch.array, 2.0,
                                       slice(end,land.slice.start,-1),
                                       endpoint='nearest')
            if begin is None:
                # due to masked values, use land.start in place
                begin = land.slice.start

            self.create_phase(slice(int(begin), ceil(end)), begin=int(begin), end=ceil(end))


class TakeoffRunwayHeading(FlightPhaseNode):
    '''
    Return phases where the aircraft is traveling on the same heading as the
    takeoff runway with a deviation of +/-10 degrees.
    '''
    def derive(self,
               hdg=P('Heading'),
               groundeds=S('Grounded'),
               toffs=S('Takeoff Roll') ):
        diff = 10
        overflow = 360
        for toff in toffs:
            for gnd in groundeds:
                if not slices_overlap(toff.slice, gnd.slice):
                    continue
                gnd_hdg = hdg.array[gnd.slice]
                if not np.ma.count(gnd_hdg):
                    continue
                rwy_hdg = np.ma.mean(hdg.array[toff.slice])
                if np.ma.is_masked(rwy_hdg):
                    continue
                max_hdg = (rwy_hdg + diff) % overflow
                min_hdg = (rwy_hdg - diff) % overflow
                min_hdg, max_hdg = min(min_hdg, max_hdg), max(min_hdg, max_hdg)
                if (max_hdg - min_hdg) > diff * 2:
                    match = ((gnd_hdg >= 0) & (gnd_hdg <= min_hdg) | (gnd_hdg >= max_hdg) & (gnd_hdg <= overflow))
                else:
                    match = (gnd_hdg >= min_hdg) & (gnd_hdg <= max_hdg)
                self.create_phases(shift_slices(runs_of_ones(match), gnd.slice.start))


class RejectedTakeoff(FlightPhaseNode):
    '''
    Rejected Takeoff based on Acceleration Longitudinal Offset Removed
    exceeding the TAKEOFF_ACCELERATION_THRESHOLD and not being followed by
    a liftoff.

    If the aircraft has the TOGA parameter, the TAKEOFF_ACCELERATION_THRESHOLD is
    ignored, and the time that TOGA is recored as active is used instead.

    Note: We cannot use Liftoff, Taxi Out or Airborne in this computation in
    case the rejected takeoff was followed by a taxi back to stand.

    For START_AND_STOP segments:
    Filter potential RTO phases where the aircraft's heading is the same as
    the takeoff heading (+/-10 degrees).

    For GROUND_ONLY segments:
    The takeoff heading is not available to filter potential RTO phases to the
    heading of the takeoff runway. This may increase the number of
    invalid RTO events.
    '''

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type')):
        req =  all_of(('Acceleration Longitudinal Offset Removed',
                       'Eng (*) All Running',
                       'Grounded',
                       'Segment Type'
                       ), available)
        if seg_type and seg_type.value == 'START_AND_STOP':
            return req and 'Takeoff Runway Heading' in available
        elif seg_type and seg_type.value == 'GROUND_ONLY':
            return req
        else:
            return False

    def derive(self,
               accel_lon=P('Acceleration Longitudinal Offset Removed'),
               eng_running=M('Eng (*) All Running'),
               groundeds=S('Grounded'),
               eng_n1=P('Eng (*) N1 Max'),
               toff_acc=KTI('Takeoff Acceleration Start'),
               toff_rwy_hdg=S('Takeoff Runway Heading'),
               seg_type=A('Segment Type'),
               toga=M('Takeoff And Go Around')):

        # We need all engines running to be a realistic attempt to get airborne
        runnings = runs_of_ones(eng_running.array=='Running')
        if not runnings:
            # No engine start, so can't have a rejected takeoff.
            return
        hz = accel_lon.frequency
        # If Takeoff Acceleration Start KTI exists, include only slices
        # before this KTI and shorten the slice containing the it.
        if toff_acc:
            toff_idx = toff_acc.get_first().index
            running_on_grounds = slices_and(
                runnings,
                [slice(runnings[0].start, toff_idx),]
            )
        else:
            # We ignore the last slice in groundeds as this is usually the
            # during the taxi in
            if seg_type.value == 'START_AND_STOP':
                gnd_slices = groundeds.get_slices()[:-1]
            else:
                gnd_slices = groundeds.get_slices()
            running_on_grounds = slices_and(runnings, gnd_slices)

        # Narrow the RTO search to when the aircraft is traveling on the
        # same heading as the takeoff runway.
        if seg_type.value == 'START_AND_STOP':
            rwy_hdgs = slices_remove_small_slices(toff_rwy_hdg.get_slices(),
                                                  time_limit=5, hz=hz)
            running_on_grounds = slices_and(running_on_grounds, rwy_hdgs)

        if eng_n1 is not None:
            n1_max_above_50 = runs_of_ones(
                repair_mask(eng_n1.array, frequency=hz, repair_duration=None) >
                50
            )
            rto_list=[]
            accel_above_thres = runs_of_ones(
                repair_mask(accel_lon.array, frequency=hz, repair_duration=None) >=
                TAKEOFF_ACCELERATION_THRESHOLD
            )
            if toga is not None:
                rto_list=[]
                potential_rtos = slices_remove_small_gaps(runs_of_ones(toga.array))
                # on some frames, TOGA is registered as a button press lasting 1 sample, so we remove anything that's
                # less than 2 seconds, this can still break on frames that have TOGA recorded at 0.5Hz, but currently
                # we have no such frames
                potential_rtos = slices_remove_small_slices(potential_rtos, 2, hz=self.hz)
                potential_rtos = slices_and(potential_rtos, n1_max_above_50)
                # for rto in potential_rtos:
                #     # If there is any decelleration during toga, find the first section of deceleration.
                #     accel_lon_rto = accel_lon.array[rto]
                #     if len(accel_lon_rto[accel_lon_rto < 0]):
                #         rto_list.append(rto)
            else:
                # list of potential RTO's which may include the takeoff as well.
                potential_rtos = slices_and(accel_above_thres, n1_max_above_50)
                potential_rtos = slices_remove_small_gaps(potential_rtos, hz=hz)
            for rto in potential_rtos:
                for running_on_ground in running_on_grounds:
                    # The RTO slice can only be within the 'Grounded' phase.
                    # If RTO slice size changes (decreases) when AND'd with
                    # running_on_ground this Acceleration/N1 Max combination
                    # should be the part of the takeoff.
                    if slices_and([rto], [running_on_ground]) == [rto]:
                        if len(rto_list) > 0 and\
                        (rto.start - rto_list[-1].stop)/hz < 60.0:
                            continue
                        rto_list.append(rto)
            if rto_list:
                self.create_phases(rto_list)
        else:
            for running_on_ground in running_on_grounds:
                accel_lon_ground = accel_lon.array[slices_int(running_on_ground)]
                accel_lon_slices = runs_of_ones(
                    accel_lon_ground >= TAKEOFF_ACCELERATION_THRESHOLD
                )

                trough_index = 0
                for peak in accel_lon_slices:
                    if trough_index and peak.start < trough_index:
                        continue
                    # Look for the deceleration characteristic of a rejected
                    # takeoff.
                    trough_index = index_at_value(
                        accel_lon_ground,
                        -TAKEOFF_ACCELERATION_THRESHOLD/2.0,
                        _slice=slice(peak.stop, None)
                    )
                    # trough_index will be None for every takeoff. Then, if it
                    # looks like a rejection, we check the two accelerations
                    # happened fairly close together.
                    if trough_index and \
                       (trough_index - peak.start)/accel_lon.hz < 60.0:
                        self.create_phase(
                            slice(peak.start + running_on_ground.start,
                                  trough_index + running_on_ground.start)
                        )


class Takeoff(FlightPhaseNode):
    """
    This flight phase starts as the aircraft turns onto the runway and ends
    as it climbs through 35ft. Subsequent KTIs and KPV computations identify
    the specific moments and values of interest within this phase.

    We use Altitude AAL (not "for Flight Phases") to avoid small errors
    introduced by hysteresis, which is applied to avoid hunting in level
    flight conditions, and make sure the 35ft endpoint is exact.
    """

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'), seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT', 'STOP_ONLY'):
            return False
        else:
            return all_of(('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast', 'Airborne'), available)

    def derive(self,
               head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast'),
               airs=S('Airborne'), # If never airborne, didnt takeoff.
               ):
        # Note: This algorithm works across the entire data array, and
        # not just inside the speedy slice, so the final indexes are
        # absolute and not relative references.

        for speedy in fast:
            # This basic flight phase cuts data into fast and slow sections.

            # We know a takeoff should come at the start of the phase,
            # however if the aircraft is already airborne, we can skip the
            # takeoff stuff.
            if not speedy.slice.start:
                break

            # The aircraft is part way down its takeoff run at the start of
            # the section.
            takeoff_run = speedy.slice.start

            #-------------------------------------------------------------------
            # Find the start of the takeoff phase from the turn onto the runway.

            # The heading at the start of the slice is taken as a datum for now.
            datum = head.array[int(takeoff_run)]

            # Track back to the turn
            # If he took more than 5 minutes on the runway we're not interested!
            first = max(0, takeoff_run - (300 * head.frequency))
            # Repair small gaps incase transition is masked.
            # XXX: This could be optimized by repairing and calling abs on
            # the 5 minute window of the array. Shifting the index manually
            # will be less pretty than using index_at_value.
            head_abs_array = np.ma.abs(repair_mask(
                head.array, frequency=head.frequency, repair_duration=30) - datum)
            takeoff_begin = index_at_value(head_abs_array,
                                           HEADING_TURN_ONTO_RUNWAY,
                                           slice(takeoff_run, first, -1))

            # Where the data starts in line with the runway, default to the
            # start of the data
            if takeoff_begin is None:
                takeoff_begin = first

            #-------------------------------------------------------------------
            # Find the end of the takeoff phase as we climb through 35ft.

            # If it takes more than 5 minutes, he's certainly not doing a normal
            # takeoff !
            last = takeoff_run + (300 * alt_aal.frequency)
            # Limit last to be the maximum altitude within 5 minutes
            last = max_value(alt_aal.array, _slice=slice(takeoff_run, last)).index
            takeoff_end = index_at_value(alt_aal.array, INITIAL_CLIMB_THRESHOLD,
                                         slice(last, takeoff_run, -1))

            if takeoff_end is None or takeoff_end <= 0:
                continue

            #-------------------------------------------------------------------
            # Create a phase for this takeoff
            self.create_phase(slice(takeoff_begin, takeoff_end))


class TakeoffRoll(FlightPhaseNode):
    '''
    Sub-phase originally written for the correlation tests but has found use
    in the takeoff KPVs where we are interested in the movement down the
    runway, not the turnon or liftoff.

    If pitch is not avaliable to detect rotation we use the end of the takeoff.
    '''

    @classmethod
    def can_operate(cls, available):
        return all_of(('Takeoff', 'Takeoff Acceleration Start'), available)

    def derive(self, toffs=S('Takeoff'),
               acc_starts=KTI('Takeoff Acceleration Start'),
               pitch=P('Pitch')):
        for toff in toffs:
            begin = toff.slice.start # Default if acceleration term not available.
            if acc_starts: # We don't bother with this for data validation, hence the conditional
                acc_start = acc_starts.get_last(within_slice=toff.slice)
                if acc_start:
                    begin = acc_start.index
            chunk = slice(begin, toff.slice.stop)
            if pitch:
                pwo = first_order_washout(pitch.array[slices_int(chunk)], 3.0, pitch.frequency)
                two_deg_idx = index_at_value(pwo, 2.0)
                if two_deg_idx is None:
                    roll_end = toff.slice.stop
                    self.warning('Aircraft did not reach a pitch of 2 deg or Acceleration Start is incorrect')
                else:
                    roll_end = two_deg_idx + begin
                self.create_phase(slice(begin, roll_end))
            else:
                self.create_phase(chunk)


class TakeoffRollOrRejectedTakeoff(FlightPhaseNode):
    '''
    For monitoring configuration warnings especially, this combines actual
    and rejected takeoffs into a single phase node for convenience.
    '''
    @classmethod
    def can_operate(cls, available):
        return any_of(cls.get_dependency_names(), available)

    def derive(self,
               trolls=S('Takeoff Roll'),
               rtoffs=S('Rejected Takeoff'),
               helo_toffs=S('Transition Hover To Flight')):
        self.create_phases(
            [s.slice for n in (trolls, rtoffs, helo_toffs) if n for s in n],
            name= "Takeoff Roll Or Rejected Takeoff")


class TakeoffRotation(FlightPhaseNode):
    '''
    This is used by correlation tests to check control movements during the
    rotation and lift phases.
    '''

    can_operate = aeroplane_only

    align_frequency = 1

    def derive(self, lifts=S('Liftoff')):
        if not lifts:
            return
        lift_index = lifts.get_first().index
        start = lift_index - 10
        end = lift_index + 15
        self.create_phase(slice(start, end))


class TakeoffRotationWow(FlightPhaseNode):
    '''
    Used by correlation tests which need to use only the rotation period while the mainwheels are on the ground. Specifically, AOA.
    '''
    name = 'Takeoff Rotation WOW'

    can_operate = aeroplane_only

    def derive(self, toff_rots=S('Takeoff Rotation')):
        for toff_rot in toff_rots:
            self.create_phase(slice(toff_rot.slice.start,
                                    toff_rot.slice.stop-15))


################################################################################
# Takeoff/Go-Around Ratings


class Takeoff5MinRating(FlightPhaseNode):
    '''
    For engines, the period of high power operation is normally a maximum of
    5 minutes from the start of takeoff.

    For all aeroplanes we use the Takeoff Acceleration Start to indicate the
    start of the Takeoff 5 Minute Rating

    For turbo prop aircraft we look for NP stabalising at least 5% less than
    liftoff NP

    For Jet aircraft we look for 5 minutes following Takeoff Acceleration Start.
    '''
    align_frequency = 1

    @classmethod
    def can_operate(cls, available, eng_type=A('Engine Propulsion'), ac_type=A('Aircraft Type')):
        if eng_type and eng_type.value == 'PROP':
            return all_of(('Takeoff Acceleration Start', 'Liftoff', 'Eng (*) Np Avg', 'Engine Propulsion', 'HDF Duration'), available)
        elif ac_type == helicopter:
            return all_of(('Liftoff', 'HDF Duration'), available)
        else:
            return all_of(('Takeoff Acceleration Start', 'HDF Duration'), available)

    def get_metrics(self, angle):
        window_sizes = [2,4,8,16,32]
        metrics = np.ma.ones(len(angle)) * 1000000
        for l in window_sizes:
            maxy = filters.maximum_filter1d(angle, l)
            miny = filters.minimum_filter1d(angle, l)
            m = (maxy - miny) / l
            metrics = np.minimum(metrics, m)

        metrics = medfilt(metrics,3)
        metrics = 200.0 * metrics

        return metrics

    def derive(self, toffs=KTI('Takeoff Acceleration Start'),
               lifts=KTI('Liftoff'),
               eng_np=P('Eng (*) Np Avg'),
               duration=A('HDF Duration'),
               eng_type=A('Engine Propulsion'),
               ac_type=A('Aircraft Type')):
        '''
        '''
        five_minutes = 300 * self.frequency
        max_idx = duration.value * self.frequency
        if eng_type and eng_type.value == 'PROP':
            filter_median_window = 11
            enp_filt = medfilt(eng_np.array, filter_median_window)
            enp_filt = np.ma.array(enp_filt)
            g = self.get_metrics(enp_filt)
            enp_filt.mask = g > 40
            flat_slices = np.ma.clump_unmasked(enp_filt)
            for accel_start in toffs:
                rating_end = toff_slice_avg = None
                if not lifts.get_next(accel_start.index):
                    continue
                toff_idx = lifts.get_next(accel_start.index).index
                for flat in flat_slices:
                    if is_index_within_slice(toff_idx, flat):
                        toff_slice_avg = np.ma.average(enp_filt[flat])
                    elif toff_slice_avg is not None:
                        flat_avg = np.ma.average(enp_filt[flat])
                        if abs(toff_slice_avg - flat_avg) >= 5:
                            rating_end = flat.start
                            break
                    else:
                        continue
                if rating_end is None:
                    rating_end = accel_start.index + (five_minutes)
                self.create_phase(slice(accel_start.index, min(rating_end, max_idx)))
        elif ac_type == helicopter:

            start_idx = end_idx = 0
            for lift in lifts:
                start_idx = start_idx or lift.index
                end_idx = lift.index + five_minutes
                next_lift = lifts.get_next(lift.index)
                if next_lift and next_lift.index < end_idx:
                    end_idx = next_lift.index + five_minutes
                    continue
                self.create_phase(slice(start_idx, min(end_idx, max_idx)))
                start_idx = 0
        else:
            for toff in toffs:
                self.create_phase(slice(toff.index, min(toff.index + five_minutes, max_idx)))


class GoAround5MinRating(FlightPhaseNode):
    '''
    For engines, the period of high power operation is normally 5 minutes from
    the start of takeoff. Also applies in the case of a go-around.
    '''
    align_frequency = 1

    @classmethod
    def can_operate(cls, available, eng_type=A('Engine Propulsion')):
        if eng_type and eng_type.value == 'PROP':
            return all_of(('Go Around', 'Eng (*) Np Avg', 'HDF Duration', 'Engine Propulsion'), available)
        else:
            return all_of(('Go Around', 'HDF Duration'), available)

    def get_metrics(self, angle):
        window_sizes = [2,4,8,16,32]
        metrics = np.ma.ones(len(angle)) * 1000000
        for l in window_sizes:
            maxy = filters.maximum_filter1d(angle, l)
            miny = filters.minimum_filter1d(angle, l)
            m = (maxy - miny) / l
            metrics = np.minimum(metrics, m)

        metrics = medfilt(metrics,3)
        metrics = 200.0 * metrics

        return metrics

    def derive(self, gas=KTI('Go Around'),
               eng_np=P('Eng (*) Np Avg'),
               duration=A('HDF Duration'),
               eng_type=A('Engine Propulsion')):

        five_minutes = 300 * self.frequency
        #define max index to prevent out of bounds exception
        max_idx = duration.value * self.frequency

        if eng_type and eng_type.value == 'PROP':
            # smoothen the signal
            filter_median_window = 11 # 11 to get one median per section in window
            enp_filt = medfilt(eng_np.array, filter_median_window)
            enp_filt = np.ma.array(enp_filt)

            #generate RoC array and find 'flat' slices
            g = self.get_metrics(enp_filt)
            enp_filt.mask = g > 12.5
            flat_slices = np.ma.clump_unmasked(enp_filt)

            for ga in gas:
                ga_flat_slices = [s for s in flat_slices if s.stop >= ga.index]
                rating_end = ga_slice_avg = None

                if not is_index_within_slices(ga.index, ga_flat_slices):
                    ga_slice_avg = np.ma.average(enp_filt[find_nearest_slice(ga.index, ga_flat_slices)])

                for flat in ga_flat_slices:
                    if ga_slice_avg is None and is_index_within_slice(ga.index, flat):
                        ga_slice_avg = np.ma.average(enp_filt[flat])

                    elif ga_slice_avg is not None:
                        flat_avg = np.ma.average(enp_filt[flat])
                        if abs(ga_slice_avg - flat_avg) >= 0.30:
                            rating_end = flat.start
                            break
                    else:
                        continue

                if rating_end is None or rating_end > (ga.index + five_minutes):
                    rating_end = ga.index + five_minutes
                self.create_phase(slice(ga.index, min(rating_end, max_idx)))
        else:
            for ga in gas:
                startpoint = ga.index
                endpoint = ga.index + five_minutes

                if startpoint < endpoint:
                    self.create_phase(slice(startpoint, min(endpoint, max_idx)))


class MaximumContinuousPower(FlightPhaseNode):
    '''
    '''

    align_frequency = 1

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return all_of(('Airborne', 'Takeoff 5 Min Rating'), available)
        else:
            return all_deps(cls, available)

    def derive(self,
               airborne=S('Airborne'),
               to_ratings=S('Takeoff 5 Min Rating'),
               ga_ratings=S('Go Around 5 Min Rating')):

        ga_slices = ga_ratings.get_slices() if ga_ratings else []
        ratings = to_ratings.get_slices() + ga_slices
        mcp = slices_and_not(airborne.get_slices(), ratings)
        self.create_phases(mcp)


################################################################################


class TaxiIn(FlightPhaseNode):
    """
    This takes the period from the end of landing to either the last engine
    stop after touchdown or the end of the mobile section.
    """
    def derive(self, gnds=S('Mobile'), lands=S('Landing'),
               last_eng_stops=KTI('Last Eng Stop After Touchdown')):
        land = lands.get_last()
        if not land:
            return
        for gnd in gnds:
            if slices_overlap(gnd.slice, land.slice):
                # Mobile may or may not stop before Landing for helicopters.
                taxi_start = min(gnd.slice.stop, land.slice.stop)
                taxi_stop = max(gnd.slice.stop, land.slice.stop)
                # Use Last Eng Stop After Touchdown if available.
                if last_eng_stops:
                    last_eng_stop = last_eng_stops.get_next(taxi_start)
                    if last_eng_stop and last_eng_stop.index > taxi_start:
                        taxi_stop = min(last_eng_stop.index,
                                        taxi_stop)
                if taxi_start != taxi_stop:
                    self.create_phase(slice(taxi_start, taxi_stop),
                                      name="Taxi In")


class TaxiOut(FlightPhaseNode):
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in.
    """
    @classmethod
    def can_operate(cls, available):
        return all_of(('Mobile', 'Takeoff'), available)

    def derive(self, gnds=S('Mobile'), toffs=S('Takeoff'),
               first_eng_starts=KTI('First Eng Start Before Liftoff')):
        if toffs:
            toff = toffs[0]
            for gnd in gnds:
                # If takeoff starts at begining of data there was no taxi out phase
                if slices_overlap(gnd.slice, toff.slice) and toff.slice.start > 1:
                    taxi_start = gnd.slice.start + 1
                    taxi_stop = toff.slice.start - 1
                    if first_eng_starts:
                        first_eng_start = first_eng_starts.get_next(taxi_start)
                        if first_eng_start and first_eng_start.index < taxi_stop:
                            taxi_start = max(first_eng_start.index,
                                             taxi_start)
                    if taxi_stop > taxi_start:
                        self.create_phase(slice(taxi_start, taxi_stop),
                                          name="Taxi Out")


################################################################################
# TCAS periods of alert

class TCASOperational(FlightPhaseNode):
    """
    There are different validity flags with different aircraft, and we need to make sure
    that TCAS does not operate on the ground. This phase merges the alternative sources
    to avoid repetition elsewhere.

    TCAS determines the approximate altitude of each aircraft above the ground.
    If this difference is less than 900 feet, TCAS considers the reporting aircraft
    to be on the ground.
    """

    name = 'TCAS Operational'
    frequency = 1.0

    @classmethod
    def can_operate(cls, available):
        return 'Altitude AAL' in available

    def derive(self, alt_aal=P('Altitude AAL'),
               tcas_cc=M('TCAS Combined Control'),
               tcas_status=P('TCAS Status'),
               tcas_valid=P('TCAS Valid'),
               tcas_fail=P('TCAS Failure')):

        operating = slices_remove_small_gaps(runs_of_ones(alt_aal.array >= 900.0),
                                             time_limit=10, hz=alt_aal.frequency)
        invalid_slices = []
        possible_ras = []

        if not operating:
            # No point in looking further if the aircraft didn't fly.
            return

        if tcas_cc:
            # Build a list of the valid sections of Combined Control data...
            good_slices = []
            # and we'll need a list of RA segments to check later...
            for op in operating:
                samples = np.ma.sum(tcas_cc.array[op].any_of(
                    'Clear of Conflict',
                    'Clear Of Conflict',  # will be run against historic data with inconsistent state names
                    'Drop Track',
                    'Altitude Lost',
                    'Up Advisory Corrective',
                    'Down Advisory Corrective',
                    'Preventive',
                    ignore_missing=True,
                ))
                if samples / float(len(tcas_cc.array[op])) > 0.1:
                    continue  # TCAS not working properly.

                ras_local = tcas_cc.array[op].any_of(
                    'Up Advisory Corrective',
                    'Down Advisory Corrective',
                    'Preventive',
                    ignore_missing=True,
                )
                ra_slices = shift_slices(runs_of_ones(ras_local), op.start)
                possible_ras.extend(ra_slices)
                # We discard the (common) RAs near the airfield
                if ras_local[0]:
                    invalid_slices.append(ra_slices[0]) # RA at takeoff not valid
                if ras_local[-1]:
                    invalid_slices.append(ra_slices[-1]) # RA at landing not valid

                # invalid conditions
                ras_local = tcas_cc.array[op].any_of(
                    'Not Used',
                    'Spare',
                    ignore_missing=True,
                )
                invalid_slices.extend(shift_slices(runs_of_ones(ras_local), op.start))
                # Overlay the original mask
                mask_local = np.ma.getmaskarray(tcas_cc.array[op])
                invalid_slices.extend(shift_slices(runs_of_ones(mask_local), op.start))

                good_slices.extend(slices_and_not([op], invalid_slices))
            operating = good_slices

        if tcas_status:
            try:
                state_name = next(s for s in ('TCAS Active', 'Normal Operation')
                                  if s in tcas_status.state)
            except StopIteration:
                pass
            else:
                tcas_bad = runs_of_ones(~(tcas_status.array == state_name))
                tcas_good = slices_not(slices_overlap_merge(tcas_bad, possible_ras), begin_at=0, end_at=len(tcas_status.array))
                operating = slices_and(operating, tcas_good)

        if tcas_valid:
            tcas_bad = runs_of_ones(~(tcas_valid.array == 'Valid'))
            tcas_good = slices_not(slices_overlap_merge(tcas_bad, possible_ras), begin_at=0, end_at=len(tcas_valid.array))
            operating = slices_and(operating, tcas_good)

        if tcas_fail:
            # With Altitude AAL defaulting to 2Hz and TCAS Fail at once per 4-sec frame, the interval is 8 samples.
            tcas_bad = slices_remove_small_gaps(runs_of_ones((tcas_fail.array == 'Failed')), count=8)
            tcas_good = slices_not(slices_overlap_merge(tcas_bad, possible_ras), begin_at=0, end_at=len(tcas_fail.array))
            operating = slices_and(operating, tcas_good)

        self.create_phases(operating)


class TCASTrafficAdvisory(FlightPhaseNode):
    """
    TCAS Traffic Advisory phase
    """

    name = 'TCAS Traffic Advisory'

    @classmethod
    def can_operate(cls, available):
        return any_one_of(('TCAS TA', 'TCAS All Threat Traffic', 'TCAS Traffic Alert', 'TCAS TA (1)'), available) \
            and 'TCAS Operational' in available

    def derive(self, tcas_ops=S('TCAS Operational'),
               tcas_ta1=M('TCAS TA'),
               tcas_ta2=M('TCAS All Threat Traffic'),
               tcas_ta3=M('TCAS Traffic Alert'),
               tcas_ta4=M('TCAS TA (1)'),
               tcas_ras=S('TCAS Resolution Advisory'),
               ):

        # Accept the TCAS TA parameter from the variously named options
        tas = [tcas_ta1, tcas_ta2, tcas_ta3, tcas_ta4]
        tcas_ta = next((item for item in tas if item is not None), None)

        all_slices = []
        for tcas_op in tcas_ops:
            array = tcas_ta.array[tcas_op.slice]
            if not len(array):
                continue
            tas_local = array.any_of('TA', 'Alert', ignore_missing=True)
            ta_slices = shift_slices(runs_of_ones(tas_local), tcas_op.slice.start)
            ta_slices = slices_remove_small_slices(ta_slices,
                                                   time_limit=4.0,
                                                   hz=tcas_ta.frequency)
            all_slices.extend(ta_slices)

        if tcas_ras:
            # We will be removing some items from the list after iterating across that list.
            to_pop = []
            for n, each_slice in enumerate(all_slices):
                # Extend to ensure overlap
                for tcas_ra in slices_extend_duration(tcas_ras.get_slices(), tcas_ops.frequency, 5.0):
                    if slices_overlap(each_slice, tcas_ra):
                        to_pop.append(n)
            for pop in to_pop[::-1]:
                all_slices.pop(pop)

        self.create_phases(all_slices)

class TCASResolutionAdvisory(FlightPhaseNode):
    '''
    This uses the Combined Control parameter only because the TCAS RA signals are only
    present on aircraft with Combined Control as well, and the TCAS RA signals include
    the Clear of Conflict period, making the duration of the phase inconsistent.
    '''

    @classmethod
    def can_operate(cls, available):
        return all_of(('TCAS Combined Control', 'TCAS Operational'), available) or \
               all_of(('TCAS RA', 'TCAS Operational'), available)

    name = 'TCAS Resolution Advisory'

    def derive(self, tcas_cc=M('TCAS Combined Control'),
               tcas_ops=S('TCAS Operational'),
               tcas_ra=M('TCAS RA')):

        for tcas_op in tcas_ops:
            # We can be sloppy about error conditions because these have been taken
            # care of in the TCAS Operational definition.
            if tcas_cc:
                ra_slices = runs_of_ones(tcas_cc.array[tcas_op.slice].any_of(
                    'Up Advisory Corrective',
                    'Down Advisory Corrective',
                    'Preventive',
                    'Drop Track',
                    ignore_missing=True,
                ))

                ra_slices = shift_slices(ra_slices, tcas_op.slice.start)
                hz = tcas_cc.frequency

            else:
                # Operating with only a single TCAS RA signal, as recorded on some aircraft.
                ra_slices = runs_of_ones(tcas_ra.array[tcas_op.slice].any_of(
                    'RA',
                    ignore_missing=True,
                ))
                ra_slices = shift_slices(ra_slices, tcas_op.slice.start)
                hz = tcas_ra.frequency

            # Where data is corrupted, single samples are a common source of error
            # time_limit rejects single samples, but 4+ sample events are retained.
            ra_slices = slices_remove_small_slices(ra_slices,
                                                   time_limit=4.0,
                                                   hz=hz)
            self.create_phases(ra_slices)


class TurningInAir(FlightPhaseNode):
    """
    Rate of Turn is greater than +/- HEADING_RATE_FOR_FLIGHT_PHASES in the air
    """
    def derive(self, rate_of_turn=P('Heading Rate'),
               airborne=S('Airborne'),
               ac_type=A('Aircraft Type')):

        if ac_type == helicopter:
            rate = HEADING_RATE_FOR_FLIGHT_PHASES_RW
        else:
            rate = HEADING_RATE_FOR_FLIGHT_PHASES_FW

        turning = np.ma.masked_inside(repair_mask(rate_of_turn.array), -rate, rate)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, air.slice)
                    for air in airborne]):
                # If the slice is within any airborne section.
                self.create_phase(turn_slice, name="Turning In Air")


class TurningOnGround(FlightPhaseNode):
    """
    Turning on ground is computed during the two taxi phases. This
    avoids high speed turnoffs where the aircraft may be travelling at high
    speed at, typically, 30deg from the runway centreline. The landing
    phase turnoff angle is nominally 45 deg, so avoiding this period.

    Rate of Turn is greater than +/- HEADING_RATE_FOR_TAXI_TURNS (%.2f) on the ground
    """ % HEADING_RATE_FOR_TAXI_TURNS
    def derive(self, rate_of_turn=P('Heading Rate'), taxi=S('Taxiing')): # Q: Use Mobile?
        turning = np.ma.masked_inside(repair_mask(rate_of_turn.array),
                                      -HEADING_RATE_FOR_TAXI_TURNS,
                                      HEADING_RATE_FOR_TAXI_TURNS)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, txi.slice)
                    for txi in taxi]):
                self.create_phase(turn_slice, name="Turning On Ground")


# NOTE: Python class name restriction: '2DegPitchTo35Ft' not permitted.
class TwoDegPitchTo35Ft(FlightPhaseNode):
    '''
    '''

    name = '2 Deg Pitch To 35 Ft'

    def derive(self, takeoff_rolls=S('Takeoff Roll'), takeoffs=S('Takeoff')):
        for takeoff in takeoffs:
            for takeoff_roll in takeoff_rolls:
                if not is_slice_within_slice(takeoff_roll.slice, takeoff.slice):
                    continue

                if takeoff.slice.stop - takeoff_roll.slice.stop > 1:
                    self.create_section(slice(takeoff_roll.slice.stop, takeoff.slice.stop),
                                    begin=takeoff_roll.stop_edge,
                                    end=takeoff.stop_edge)
                else:
                    self.warning('%s not created as slice less than 1 sample' % self.name)


class ShuttlingApproach(FlightPhaseNode):
    '''
    Flight phase for the shuttling approach
    '''

    def derive(self, approaches=App('Approach Information')):
        for approach in approaches:
            if approach.type == 'SHUTTLING':
                self.create_section(approach.slice, name='Shuttling Approach')


class AirborneRadarApproach(FlightPhaseNode):
    '''
    Flight phase for airborne radar approaches (ARDA/AROA)
    '''

    def derive(self, approaches=App('Approach Information')):
        for approach in approaches:
            if approach.type == 'AIRBORNE_RADAR':
                self.create_section(approach.slice, name='Airborne Radar Approach')


class BaroDifference(FlightPhaseNode):
    """
    Baro Correction difference between Capt and FO phase

    Minimum 10 sec duration.
    """

    @classmethod
    def can_operate(cls, available, manufacturer=A('Manufacturer')):
        baro_setting_sel = True
        if manufacturer and manufacturer.value == 'Airbus':
            # Airbus Baro Correction does not show 1013 when selecting QNH STD.
            # Can only be used in the presence of other parameters telling us when STD is selected.
            baro_setting_sel = any((
                any_of(('Baro Setting Selection', 'Baro Correction (ISIS)'), available),
                all_of(('Baro Setting Selection (Capt)', 'Baro Setting Selection (FO)'), available)
            ))
        two_baro =  all_of(('Baro Correction (Capt)', 'Baro Correction (FO)'), available)
        return two_baro and baro_setting_sel

    def derive(self, baro_cpt=P('Baro Correction (Capt)'),
               baro_fo=P('Baro Correction (FO)'),
               baro_sel=M('Baro Setting Selection'),
               baro_sel_cpt=M('Baro Setting Selection (Capt)'),
               baro_sel_fo=M('Baro Setting Selection (FO)'),
               baro_cor_isis=P('Baro Correction (ISIS)'),
               fasts=S('Fast')):

        diff = abs(baro_cpt.array - baro_fo.array)
        if not np.ma.count(diff):
            return
        if baro_sel:
            diff[baro_sel.array == 'ALT STD'] = 0
        elif baro_sel_cpt and baro_sel_fo:
            diff[(baro_sel_cpt.array == 'STD') | (baro_sel_fo.array == 'STD')] = 0
        elif baro_cor_isis:
            diff[np.isclose(baro_cor_isis.array, 1013)] = 0

        _, diff_slices = slices_above(diff, 1.0)
        diff_slices = slices_and(diff_slices, fasts.get_slices())
        diff_slices = filter_slices_duration(diff_slices, 10, frequency=self.hz)
        if self.hz < 0.25:  # handle slices from superframe parameters extending beyond end of flight
            diff_slices = [slice(s.start, s.stop - 1 if s.stop > s.start else s.stop) for s in diff_slices]
        self.create_sections(diff_slices)

