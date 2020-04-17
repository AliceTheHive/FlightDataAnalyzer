import numpy as np

from analysis_engine.node import (
    A, M, P, S, FlightPhaseNode, helicopter, helicopter_only
)

from analysis_engine.library import (
    all_deps,
    all_of,
    filter_slices_duration,
    index_at_value,
    mask_outside_slices,
    moving_average,
    runs_of_ones,
    shift_slice,
    shift_slices,
    repair_mask,
    slices_above,
    slices_and,
    slices_and_not,
    slices_below,
    slices_from_to,
    slices_overlap,
    slices_remove_small_gaps,
    slices_remove_small_slices
)

from analysis_engine.settings import (
    AIRBORNE_THRESHOLD_TIME_RW,
    AUTOROTATION_SPLIT,
    HOVER_GROUNDSPEED_LIMIT,
    HOVER_HEIGHT_LIMIT,
    HOVER_MIN_DURATION,
    HOVER_MIN_HEIGHT,
    HOVER_TAXI_HEIGHT,
    ROTOR_TRANSITION_ALTITUDE,
    ROTOR_TRANSITION_SPEED_HIGH,
    ROTOR_TRANSITION_SPEED_LOW,
    TAKEOFF_PERIOD
)

from flightdatautilities.numpy_utils import slices_int


class Airborne(FlightPhaseNode):
    '''
    Periods where the aircraft is in the air.
    We do not use Altitude AGL as the smoothing function causes values close to the
    ground to be elevated.

    On the AS330 Puma, the Gear On Ground signal is only sampled once per frame
    so is only used to confirm validity of the radio altimeter signal and for
    preliminary data validation flight phase computation.
    '''

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT'):
            return False
        return all_of(('Gear On Ground',), available)

    def derive(self,
               alt_rad=P('Altitude Radio'),
               alt_agl=P('Altitude AGL'),
               gog=M('Gear On Ground'),
               rtr=S('Rotors Turning')):
        # When was the gear in the air?
        gear_off_grounds = runs_of_ones(gog.array == 'Air')

        if alt_rad and alt_agl and rtr:
            # We can do a full analysis.
            # First, confirm that the rotors were turning at this time:
            gear_off_grounds = slices_and(gear_off_grounds, rtr.get_slices())

            # When did the radio altimeters indicate airborne?
            airs = slices_remove_small_gaps(
                np.ma.clump_unmasked(np.ma.masked_less_equal(alt_agl.array, 1.0)),
                time_limit=AIRBORNE_THRESHOLD_TIME_RW, hz=alt_agl.frequency)
            # Both is a reliable indication of being in the air.
            for air in airs:
                for goff in gear_off_grounds:
                    # Providing they relate to each other :o)
                    if slices_overlap(air, goff):
                        start_index = max(air.start, goff.start)
                        end_index = min(air.stop, goff.stop)

                        better_begin = index_at_value(
                            alt_rad.array, 1.0,
                            _slice=slice(max(start_index-5*alt_rad.frequency, 0),
                                         start_index+5*alt_rad.frequency)
                        )
                        if better_begin:
                            begin = better_begin
                        else:
                            begin = start_index

                        better_end = index_at_value(
                            alt_rad.array, 1.0,
                            _slice=slice(max(end_index+5*alt_rad.frequency, 0),
                                         end_index-5*alt_rad.frequency, -1))
                        if better_end:
                            end = better_end
                        else:
                            end = end_index

                        duration = end - begin
                        if (duration / alt_rad.hz) > AIRBORNE_THRESHOLD_TIME_RW:
                            self.create_phase(slice(begin, end))
        else:
            # During data validation we can select just sensible flights;
            # short hops make parameter validation tricky!
            self.create_phases(
                slices_remove_small_gaps(
                    slices_remove_small_slices(gear_off_grounds, time_limit=30)))


class Autorotation(FlightPhaseNode):
    '''
    Look for at least 1% difference between the highest power turbine speed
    and the rotor speed.
    This is bound to happen in a descent, and we define the autorotation
    period as from the initial onset
    to the final establishment of normal operation.

    Note: For Autorotation KPV: Detect maximum Nr during the Autorotation phase.
    '''

    can_operate = helicopter_only

    def derive(self, max_n2=P('Eng (*) N2 Max'),
               nr=P('Nr'), descs=S('Descending')):
        for desc in descs:
            # Look for split in shaft speeds.
            delta = nr.array[desc.slice] - max_n2.array[desc.slice]
            split = np.ma.masked_less(delta, AUTOROTATION_SPLIT)
            split_ends = np.ma.clump_unmasked(split)
            if split_ends:
                self.create_phase(shift_slice(slice(split_ends[0].start,
                                                    split_ends[-1].stop ),
                                              desc.slice.start))


class HelicopterEngineCheck(FlightPhaseNode):
    '''
    We define an engine power assurance check as a period of operation
    with rotors turning, more than 10% collective and more than
    20% torque asymmetry.

    The radio altitude check ensures this is not carried out in flight,
    but allows for the aircraft to lift into the hover or become light
    on the wheels.
    '''

    def derive(self, coll=P('Collective'), rtrs=S('Rotors Turning'),
               rad_alt=P('Altitude Radio'),
               q_diff=P('Torque Asymmetry'),
               running=M('Eng (*) All Running'),
               nr=P('Nr')):

        lows = slices_below(rad_alt.array, 50.0)[1]
        coll_raised = slices_above(coll.array, 20.0)[1]
        all_running = runs_of_ones(running.array)
        tests = slices_and(lows, coll_raised)
        tests = slices_and(tests, rtrs.get_slices())
        tests = slices_and(tests, all_running)
        for test in tests:
            # A real test engages both engines then splits off one at a time.
            # This complex test rejects cases where the rotor is started or
            # stopped with only one engine running.
            if np.ma.max(q_diff.array[slices_int(test)]) > 20.0:
                self.create_phase(test)


class Hover(FlightPhaseNode):
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return ac_type == helicopter and \
               all_of(('Altitude AGL', 'Airborne', 'Groundspeed'), available)

    def derive(self, alt_agl=P('Altitude AGL'),
               airs=S('Airborne'),
               gspd=P('Groundspeed'),
               trans_hfs=S('Transition Hover To Flight'),
               trans_fhs=S('Transition Flight To Hover')):

        low_flights = []
        hovers = []

        for air in airs:
            lows = slices_below(alt_agl.array[air.slice], HOVER_HEIGHT_LIMIT)[1]
            for low in lows:
                if np.ma.min(alt_agl.array[shift_slice(low, air.slice.start)]) <= HOVER_MIN_HEIGHT:
                    low_flights.extend([shift_slice(low, air.slice.start)])

        repaired_gspd = repair_mask(gspd.array, frequency=gspd.hz,
                                    repair_duration=8, method='fill_start')

        slows = slices_below(repaired_gspd, HOVER_GROUNDSPEED_LIMIT)[1]
        low_flights = slices_and(low_flights, slows)
        # Remove periods identified already as transitions.
        for low_flight in low_flights:
            if trans_fhs:
                for trans_fh in trans_fhs:
                    if slices_overlap(low_flight, trans_fh.slice):
                        low_flight = slice(trans_fh.slice.stop, low_flight.stop)

            if trans_hfs:
                for trans_hf in trans_hfs:
                    if slices_overlap(low_flight, trans_hf.slice):
                        low_flight = slice(low_flight.start, trans_hf.slice.start)

            hovers.extend([low_flight])

        # Exclude transition periods and trivial periods of operation.
        self.create_phases(filter_slices_duration(hovers, HOVER_MIN_DURATION, frequency=alt_agl.frequency))


class HoverTaxi(FlightPhaseNode):
    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return ac_type == helicopter and \
               all_of(('Altitude AGL', 'Airborne', 'Hover'), available)

    def derive(self, alt_agl=P('Altitude AGL'),
               airs=S('Airborne'),
               hovers=S('Hover'),
               trans_hfs=S('Transition Hover To Flight'),
               trans_fhs=S('Transition Flight To Hover')):

        air_taxis = []
        taxis = []

        if airs:
            for air in airs:
                lows = slices_below(alt_agl.array[air.slice], HOVER_TAXI_HEIGHT)[1]
                taxis = shift_slices(lows, air.slice.start)
        # Remove periods identified already as transitions.
        if taxis:
            for taxi in slices_and_not(taxis, [h.slice for h in hovers]):
                if trans_fhs:
                    for trans_fh in trans_fhs:
                        if slices_overlap(taxi, trans_fh.slice):
                            taxi = slice(trans_fh.slice.stop, taxi.stop)

                if trans_hfs:
                    for trans_hf in trans_hfs:
                        if slices_overlap(taxi, trans_hf.slice):
                            taxi = slice(taxi.start, trans_hf.slice.start)

                air_taxis.extend([taxi])

        self.create_phases(air_taxis)


class NoseDownAttitudeAdoption(FlightPhaseNode):
    '''
    ABO H-175 helideck takeoff profile requires helicopters to reach
    -10 degrees pitch after reaching 20ft radio altitude and initiation of nose
    down attitude. This phase represents the duration of time between nose down
    attitude initiation and -10 degrees pitch. The phase does not exclude pitch
    values prior to 20ft radio altitude as insufficient altitude prior to nose
    down attitude adoption needs to be picked up by a KPV to generate events.
    Likewise, if a pitch of -10 degrees is never found, the minimum is used.
    '''
    align_frequency = 16
    can_operate = helicopter_only

    @classmethod
    def can_operate(cls, available, family=A('Family')):
        return family and family.value == 'H175' and \
               all_deps(cls, available)

    def derive(self,
               pitch=P('Pitch'),
               climbs=S('Initial Climb'),
               offshore=S('Offshore')):

        for climb_offshore in slices_and(climbs.get_slices(), offshore.get_slices()):
            climb_offshore = slices_int(climb_offshore)
            masked_pitch = mask_outside_slices(pitch.array, [climb_offshore])

            pitch_index = np.ma.argmax(masked_pitch <= -10) or np.ma.argmin(masked_pitch)

            scaling_factor = abs(masked_pitch[pitch_index]) / 10

            window_threshold = -10.00 * scaling_factor
            min_window_threshold = -8.00 * scaling_factor
            window_size = 32
            window_threshold_step = 0.050 * scaling_factor

            diffs = np.ma.ediff1d(masked_pitch[climb_offshore.start:pitch_index])
            diffs_exist = diffs.data.size >= 2

            big_diff_index = -1

            while diffs_exist:
                sig_pitch_threshold = window_threshold / window_size

                for i, d in enumerate(diffs):
                    # Look for the first big negative pitch spike
                    if diffs[slices_int(i, i+window_size)].sum() < window_threshold:

                        # Find the first significant negative value within the
                        # spike and make that the starting point of the phase
                        big_diff_index = np.ma.argmax(diffs[i:i+window_size] < sig_pitch_threshold) + i
                        break

                # Bail on match or total failure
                if big_diff_index != -1 or window_size < 2:
                    break

                # Shrink window size instead of looking for insignificant
                # spikes and scale window/pitch thresholds accordingly
                if window_threshold >= min_window_threshold:
                    window_size //= 2; min_window_threshold /= 2; window_threshold /= 2; window_threshold_step /= 2
                    sig_pitch_threshold *= 2
                else:
                    window_threshold += window_threshold_step

            if big_diff_index != -1:
                self.create_section(slice(climb_offshore.start + big_diff_index,
                                          pitch_index))

            # Worst case fallback, this should happen extremely rarely
            # and would trigger all events related to this phase
            else:
                self.create_section(slice(climb_offshore.start, climb_offshore.stop))


class RotorsTurning(FlightPhaseNode):
    '''
    Used to suppress nuisance warnings on the ground.

    Note: Rotors Running is the Multistate parameter, while Rotors Turning is the flight phase.
    '''

    can_operate = helicopter_only

    def derive(self, rotors=M('Rotors Running')):
        self.create_sections(runs_of_ones(rotors.array == 'Running'))


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
    def can_operate(cls, available, seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT', 'STOP_ONLY'):
            return False
        else:
            return all_of(('Altitude AGL', 'Liftoff'), available)

    def derive(self,
               alt_agl=P('Altitude AGL'),
               lifts=S('Liftoff')):
        for lift in lifts:
            begin = max(lift.index - TAKEOFF_PERIOD * alt_agl.frequency, 0)
            end = min(lift.index + TAKEOFF_PERIOD * alt_agl.frequency, len(alt_agl.array) - 1)
            self.create_phase(slice(begin, end))


class TransitionHoverToFlight(FlightPhaseNode):
    '''
    The pilot normally makes a clear nose down pitching motion to initiate the
    transition from the hover, and with airspeed built, will raise the nose and
    initiate a clear climb to mark the end of the transition phase and start of the climb.
    '''

    can_operate = helicopter_only

    def derive(self, alt_agl=P('Altitude AGL'),
               ias=P('Airspeed'),
               airs=S('Airborne'),
               pitch_rate=P('Pitch Rate')):
        for air in airs:
            lows = np.ma.clump_unmasked(np.ma.masked_greater(alt_agl.array[air.slice],
                                                             ROTOR_TRANSITION_ALTITUDE))
            for low in lows:
                trans_slices = slices_from_to(ias.array[air.slice][low],
                                              ROTOR_TRANSITION_SPEED_LOW,
                                              ROTOR_TRANSITION_SPEED_HIGH,
                                              threshold=1.0)[1]
                if trans_slices:
                    for trans in trans_slices:
                        base = air.slice.start + low.start
                        ext_start = int(base  + trans.start - 20*ias.frequency)
                        if alt_agl.array[ext_start]==0.0:
                            trans_start = index_at_value(ias.array, 0.0,
                                                         _slice=slice(base+trans.start, ext_start, -1),
                                                         endpoint='first_closing')
                        else:
                            trans_start = np.ma.argmin(pitch_rate.array[ext_start:base+trans.start]) + ext_start
                        self.create_phase(slice(trans_start, trans.stop+base))


class TransitionFlightToHover(FlightPhaseNode):
    '''
    Forward flight to hover transitions are weakly defined from a flight parameter
    perspective, so we only reply upon airspeed changes.
    '''

    can_operate = helicopter_only

    def derive(self, alt_agl=P('Altitude AGL'),
               ias=P('Airspeed'),
               airs=S('Airborne'),
               pitch_rate=P('Pitch Rate')):
        for air in airs:
            trans_slices = slices_from_to(ias.array[air.slice],
                                          ROTOR_TRANSITION_SPEED_HIGH,
                                          ROTOR_TRANSITION_SPEED_LOW,
                                          threshold=1.0)[1]

            if trans_slices:
                for trans in shift_slices(trans_slices, air.slice.start):
                    trans_end = index_at_value(ias.array, 0.0,
                                                 _slice=slice(trans.stop, trans.stop+20*ias.frequency),
                                                 endpoint='first_closing')
                    self.create_phase(slice(trans.start, trans_end+1))


class OnDeck(FlightPhaseNode):
    '''
    Flight phase for helicopters that land on the deck of a moving vessel.

    Testing for motion will separate moving vessels from stationary decks, chosen as a better
    option than testing the location against Google Earth for land/sea.

    Also, movement was not practical as helicopters taxi at similar speeds to a ship sailing!

    Note that this qualifies Grounded which is still asserted when On Deck.
    '''

    can_operate = helicopter_only

    def derive(self, gnds=S('Grounded'),
               pitch=P('Pitch'), roll=P('Roll')):

        decks = []
        for gnd in gnds:
            # The fourier transform for pitching motion...
            p = pitch.array[gnd.slice]
            if np.all(p.mask):
                continue
            n = float(len(p)) # Scaling the result to be independet of data length.
            fft_p = np.abs(np.fft.rfft(p - moving_average(p))) / n

            # similarly for roll
            r = roll.array[gnd.slice]
            if np.all(r.mask):
                continue
            fft_r = np.abs(np.fft.rfft(r - moving_average(r))) / n

            # What was the maximum harmonic seen?
            fft_max = np.ma.max(fft_p + fft_r)

            # Values of less than 0.1 were on the ground, and 0.34 on deck for the one case seen to date.
            if fft_max > 0.2:
                decks.append(gnd.slice)
        if decks:
            self.create_sections(decks)
