#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import logging
import pytz
import numpy as np

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from math import floor

from analysis_engine import hooks, settings
from analysis_engine.datastructures import Segment
from analysis_engine.node import P
from analysis_engine.library import (align,
                                     blend_parameters,
                                     calculate_timebase,
                                     hash_array,
                                     min_value,
                                     mask_outside_slices,
                                     normalise,
                                     repair_mask,
                                     rate_of_change,
                                     runs_of_ones,
                                     shift_slices,
                                     slices_and_not,
                                     slices_multiply,
                                     slices_of_runs,
                                     slices_remove_small_gaps,
                                     slices_remove_small_slices,
                                     straighten_headings,
                                     vstack_params)

from hdfaccess.file import hdf_file
from hdfaccess.utils import segment_boundaries, write_segment

from flightdatautilities.filesystem_tools import sha_hash_file
from flightdatautilities.numpy_utils import (
    slices_int,
    py2round,
)
from six import string_types


# Timestamp infomation
PRECISE = 'PRECISE'
FALLBACK_NO_PARAM = 'FALLBACK_NO_PARAM'
FALLBACK_VALIDATION = 'FALLBACK_VALIDATION'


logger = logging.getLogger(name=__name__)


class AircraftMismatch(ValueError):
    pass


class TimebaseError(ValueError):
    pass


def validate_aircraft(aircraft_info, hdf):
    """
    """
    #if 'Aircraft Ident' in hdf and so on:
    # TODO: Implement validate_aircraft.
    logger.warning("Validate Aircraft not implemented")
    if True:
        return True
    else:
        raise AircraftMismatch("Tail does not match identification %s" %
                               aircraft_info['Tail Number'])


def _segment_type_and_slice(speed_array, speed_frequency,
                            heading_array, heading_frequency,
                            start, stop, eng_arrays,
                            aircraft_info, thresholds, hdf,
                            vspeed=None):
    """
    Uses the Heading to determine whether the aircraft moved about at all and
    the airspeed to determine if it was a full or partial flight.

    NO_MOVEMENT: When the aircraft is in the hanger,
    the altitude and airspeed can be tested and record values which look like
    the aircraft is in flight; however the aircraft is rarely moved and the
    heading sensor is a good indication that this is a hanger test.

    GROUND_ONLY: If the heading changed, the airspeed needs to have been above
    the threshold speed for flight for a minimum amount of time, currently 3
    minutes to determine. If not, this segment is identified as GROUND_ONLY,
    probably taxiing, repositioning on the ground or a rejected takeoff.

    START_ONLY: If the airspeed started slow but ended fast, we had a partial
    segment for the start of a flight.

    STOP_ONLY:  If the airspeed started fast but ended slow, we had a partial
    segment for the end of a flight.

    MID_FLIGHT: The airspeed started and ended fast - no takeoff or landing!

    START_AND_STOP: The airspeed started and ended slow, implying a complete
    flight.

    segment_type is one of:
    * 'NO_MOVEMENT' (didn't change heading)
    * 'GROUND_ONLY' (didn't go fast)
    * 'START_AND_STOP'
    * 'START_ONLY'
    * 'STOP_ONLY'
    * 'MID_FLIGHT'
    """

    speed_start = int(start * speed_frequency)
    speed_stop = int(stop * speed_frequency)
    speed_array = speed_array[speed_start:speed_stop]

    heading_start = int(start * heading_frequency)
    heading_stop = int(stop * heading_frequency)
    heading_array = heading_array[heading_start:heading_stop]

    slow_start = slow_stop = fast_for_long = ground_only = threshold_exceedance = None

    # Check Heading change to see if the aircraft moved.
    if eng_arrays is not None:
        heading_array = np.ma.masked_where(eng_arrays[heading_start:heading_stop] < settings.MIN_FAN_RUNNING, heading_array)
    hdiff = np.ma.abs(np.ma.diff(heading_array)).sum()
    did_move = hdiff > settings.HEADING_CHANGE_TAXI_THRESHOLD

    # Did the aircraft climb and descend?
    vspd_threshold_exceedance = None
    if vspeed:
        vert_spd_array = vspeed.array[int(start * vspeed.frequency):int(stop * vspeed.frequency)]
        vspd_threshold_exceedance = \
            (np.ma.sum(vert_spd_array > thresholds['vertical_speed_max']) / vspeed.frequency) > thresholds['min_duration'] or \
            (np.ma.sum(vert_spd_array < thresholds['vertical_speed_min']) / vspeed.frequency) > thresholds['min_duration']

    # Did the aircraft fly fast at some time?
    # remove small gaps between valid data, e.g. brief data spikes
    unmasked_slices = slices_remove_small_gaps(
        np.ma.clump_unmasked(speed_array), 2, speed_frequency)
    # remove small slices to find 'consistent' valid data
    unmasked_slices = slices_remove_small_slices(
        unmasked_slices, 40, speed_frequency)
    if unmasked_slices:
        # Check speed
        slow_start = speed_array[unmasked_slices[0].start] < thresholds['speed_threshold']
        slow_stop = speed_array[unmasked_slices[-1].stop - 1] < thresholds['speed_threshold']
        threshold_exceedance = np.ma.sum(
            speed_array > thresholds['speed_threshold']) / speed_frequency
        fast_for_long = threshold_exceedance > thresholds['min_duration']

    # Additional checks for start and stop of helicopters, in case the data did not have neat Nr endpoints.
    if aircraft_info and aircraft_info['Aircraft Type'] == 'helicopter':
        # if any gear params use them
        gog = next((p for p in (hdf.get(n) for n in ('Gear On Ground', 'Gear (R) On Ground', 'Gear (L) On Ground')) if p), None)
        col = next((p for p in (hdf.get(n) for n in ('Collective', 'Collective (1)', 'Collective (2)')) if p), None)
        if gog:
            # Backstop check for helicopters that pop into the hover without a heading change.
            if not did_move:
                air_slices = slices_remove_small_slices(runs_of_ones(gog.array == 'Air'))
                did_move = len(air_slices) > 0
            # Use Gear on Ground as a backup, as the rotor speed may be
            # 90+% at beginning or end of segment.
            gog_window_samples = int(120 * gog.frequency)
            gog_min_samples = 4 * gog.frequency

            # Did the helicopter stay on the ground?
            if not slow_start:
                gog_start_idx = int(start * gog.frequency)
                gog_start_slices = sorted(slices_of_runs(
                    gog.array[gog_start_idx:gog_start_idx + gog_window_samples],
                    min_samples=gog_min_samples, flat=True))
                slow_start = (gog.array[gog_start_slices[0].start] == 'Ground')

            if not slow_stop:
                gog_stop_idx = int(stop * gog.frequency)
                gog_stop_slices = sorted(slices_of_runs(
                    gog.array[gog_stop_idx - gog_window_samples:gog_stop_idx],
                    min_samples=gog_min_samples, flat=True))
                slow_stop = (gog.array[gog_stop_slices[-1].stop - 1] == 'Ground')

        elif col:
            # If we can't use gear on ground, we test the collective.
            # If the collective is below the COLLECTIVE_ON_GROUND_THRESHOLD then the helicopter
            # is likely be on the ground and we can test for slow_start and slow_stop.
            col = next((p for p in (hdf.get(n) for n in ('Collective', 'Collective (1)', 'Collective (2)')) if p), None)
            col_window_sample = int(120 * col.frequency)
            col_min_sample = int(4 * col.frequency)
            speedy = np.ma.where(speed_array > thresholds['speed_threshold'])[0]
            if len(speedy):
                col_start = int((speed_start + speedy[0]) * (col.frequency / speed_frequency))
                col_stop = int((speed_start + speedy[-1]) * (col.frequency / speed_frequency))
                try:  # shift start backwards to earlier low Collective value
                    col_start -= col_window_sample + np.ma.where(col.array[col_start - col_window_sample:col_start])[0][0]
                except IndexError:
                    pass
                if col_start < unmasked_slices[0].start:
                    col_start = unmasked_slices[0].start
                try: # shift stop forwards to later low Collective value
                    col_stop += np.ma.where(col.array[col_stop:col_stop + col_window_sample])[0][-1]
                except IndexError:
                    pass
            elif unmasked_slices:
                # With some of the pre-split data, the collective appears to be padded
                # starting and finishing inside the speed_array.
                # Use unmasked speed slices as the start/end indices for collective window
                col_start = int(unmasked_slices[0].start * (col.frequency / speed_frequency))
                col_stop = int((unmasked_slices[-1].stop - 1) * (col.frequency / speed_frequency))
            else:
                col_start = int(start * col.frequency)
                col_stop = int(stop * col.frequency)

            # We only use the collective settings to improve analysis if the rotor speed is high at the end of the data segment.
            if not slow_start:
                slow_start = runs_of_ones(
                    col.array[col_start:col_start+col_window_sample] < settings.COLLECTIVE_ON_GROUND_THRESHOLD,
                    min_samples=col_min_sample
                )
            if not slow_stop:
                slow_stop = runs_of_ones(
                    col.array[col_stop-col_window_sample:col_stop] < settings.COLLECTIVE_ON_GROUND_THRESHOLD,
                    min_samples=col_min_sample
                )

    if not did_move or (not fast_for_long and eng_arrays is None):
        # added check for not fast for long and no engine params to avoid
        # lots of Herc ground runs
        logger.debug("Aircraft did not move.")
        segment_type = 'NO_MOVEMENT'
        # e.g. hanger tests, esp. if speed changes!
    elif slow_start and slow_stop and fast_for_long:
        logger.debug(
            "speed started below threshold, rose above and stopped below.")
        segment_type = 'START_AND_STOP'
    elif slow_start and threshold_exceedance:
        logger.debug("speed started below threshold and stopped above.")
        segment_type = 'START_ONLY'
    elif slow_stop and threshold_exceedance:
        logger.debug("speed started above threshold and stopped below.")
        segment_type = 'STOP_ONLY'
    elif not fast_for_long and vspd_threshold_exceedance:
        logger.debug("No airspeed but has vertical speed.")
        segment_type = 'INVALID'
    elif not fast_for_long:
        logger.debug("speed was below threshold.")
        segment_type = 'GROUND_ONLY'  # e.g. RTO, re-positioning A/C
        #Q: report a go_fast?
    else:
        logger.debug("speed started and stopped above threshold.")
        segment_type = 'MID_FLIGHT'
    logger.info("Segment type is '%s' between '%s' and '%s'.",
                segment_type, start, stop)

    # ARINC 717 data has frames or superframes. ARINC 767 will be split
    # on a minimum boundary of 4 seconds for the analyser.
    boundary = 64 if hdf.superframe_present else 4
    segment = slice(start, stop)

    supf_start_secs, supf_stop_secs, array_start_secs, array_stop_secs = segment_boundaries(segment, boundary)

    return segment_type, segment, array_start_secs


def _get_normalised_split_params(hdf, align_param=None):
    '''
    Get split parameters (currently engine power and Groundspeed) from hdf,
    normalise them on a scale from 0-1.0 and return the minimum.

    :param hdf: hdf_file object.
    :type hdf: hdfaccess.file.hdf_file
    :returns: Minimum of normalised split parameters along with its frequency.
        Will return None, None if no split parameters are available.
    :rtype: (None, None) or (np.ma.masked_array, float)
    '''
    params = []

    split_params = (
        'Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',
        'Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',
        'Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np',
        'Eng (1) Fuel Flow', 'Eng (2) Fuel Flow', 'Eng (3) Fuel Flow', 'Eng (4) Fuel Flow',
        'Groundspeed', 'Groundspeed (1)', 'Groundspeed (2)'
    )
    for param_name in split_params:
        try:
            param = hdf[param_name]
        except KeyError:
            continue
        if align_param:
            # Align all other parameters to first available.  #Q: Why not force
            # to 1Hz?
            param.array = align(param, align_param)
        else:
            align_param = param
        params.append(param)

    if not len(params):
        return None, None
    # If there is at least one split parameter available.
    # normalise the parameters we'll use for splitting the data
    stacked_params = vstack_params(*params)
    # We normalise each in turn to the range 0-1 so they have equal weight
    normalised_params = [normalise(i) for i in stacked_params]
    # Using a true minimum leads to bias to a zero value. We take the average
    # to allow each parameter equal weight, then (later) seek the minimum.
    split_params_min = np.ma.average(normalised_params, axis=0)
    return split_params_min, align_param.frequency


def _get_eng_params(hdf, align_param=None):
    '''
    Get eng parameters from hdf, and return the minimum.

    :param hdf: hdf_file object.
    :type hdf: hdfaccess.file.hdf_file
    :returns: Minimum of normalised split parameters along with its frequency.
        Will return None, None if no split parameters are available.
    :rtype: (None, None) or (np.ma.masked_array, float)
    '''
    params = []

    eng_params = (
        'Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',
        'Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',
        'Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np',
        'Eng (1) Fuel Flow', 'Eng (2) Fuel Flow', 'Eng (3) Fuel Flow', 'Eng (4) Fuel Flow'
    )

    for param_name in eng_params:
        try:
            param = hdf[param_name]
        except KeyError:
            continue
        if align_param:
            # Align all other parameters to provided param or first available.
            param.array = align(param, align_param)
        else:
            align_param = param
        params.append(param)

    if not len(params):
        return None, None
    # If there is at least one split parameter available.
    stacked_params = vstack_params(*params)
    split_params_avg = np.ma.average(stacked_params, axis=0)
    return split_params_avg, align_param.frequency


def _rate_of_turn(heading):
    '''
    Create rate of turn from heading.

    :param heading: Heading parameter.
    :type heading: Parameter
    '''
    heading.array = repair_mask(straighten_headings(heading.array),
                                repair_duration=None)
    rate_of_turn = np.ma.abs(rate_of_change(heading, 8))
    rate_of_turn_masked = \
        np.ma.masked_greater(rate_of_turn,
                             settings.HEADING_RATE_SPLITTING_THRESHOLD)
    return rate_of_turn_masked


def _split_on_eng_params(slice_start_secs, slice_stop_secs, split_params_min,
                         split_params_frequency):
    '''
    Find split using engine parameters.

    :param slice_start_secs: Start of slow slice in seconds.
    :type slice_start_secs: int or float
    :param slice_stop_secs: Stop of slow slice in seconds.
    :type slice_stop_secs: int or float
    :param split_params_min: Minimum of engine parameters.
    :type split_params_min: np.ma.MaskedArray
    :param split_params_frequency: Frequency of split_params_min.
    :type split_params_frequency: int or float
    :returns: Split index in seconds and value of split_params_min at this
        index.
    :rtype: (int or float, int or float)
    '''
    slice_start = slice_start_secs * split_params_frequency
    slice_stop = slice_stop_secs * split_params_frequency
    split_params_slice = slice(int(np.round(slice_start, 0)), int(np.round(slice_stop, 0)))
    split_index, split_value = min_value(split_params_min,
                                         _slice=split_params_slice)

    if split_index is None:
        return split_index, split_value

    eng_min_slices = runs_of_ones(split_params_min[split_params_slice] == split_value)

    if not eng_min_slices:
        return py2round(split_index / split_params_frequency), split_value

    split_index = eng_min_slices[0].start + \
        ((eng_min_slices[0].stop - eng_min_slices[0].start) // 2) + slice_start
    return py2round(split_index / split_params_frequency), split_value


def _split_on_dfc(slice_start_secs, slice_stop_secs, dfc_frequency,
                  dfc_half_period, dfc_diff, eng_split_index=None):
    '''
    Find split using 'Frame Counter' parameter.

    :param slice_start_secs: Start of slow slice in seconds.
    :type slice_start_secs: int or float
    :param slice_stop_secs: Stop of slow slice in seconds.
    :type slice_stop_secs: int or float
    :param dfc_frequency: Frequency of 'Frame Counter' parameter.
    :type dfc_frequency: int or float
    :param dfc_half_period: Gap between values in the diff.
    :type dfc_half_period: int or float
    :param dfc_diff: Diff of 'Frame Counter' parameter.
    :type dfc_diff: np.ma.MaskedArray
    :param eng_split_index: Split index based on minimum of engine parameters.
    :type eng_split_index: int or float
    :returns: Split index based on 'Frame Counter' jumps or None if no jumps
        occur.
    :rtype: int or float or None
    '''
    dfc_slice = slice(slice_start_secs * dfc_frequency,
                      int(floor(slice_stop_secs * dfc_frequency)) + 1)
    unmasked_edges = np.ma.flatnotmasked_edges(dfc_diff[slices_int(dfc_slice)])
    if unmasked_edges is None:
        return None
    unmasked_edges = unmasked_edges.astype(float)
    unmasked_edges /= dfc_frequency
    if eng_split_index:
        # Split on the jump closest to the engine parameter minimums.
        dfc_jump = unmasked_edges[np.ma.argmin(np.ma.abs(
            (eng_split_index - slice_start_secs) - unmasked_edges))]
    else:
        # Split on the first DFC jump.
        dfc_jump = unmasked_edges[0]
    dfc_index = py2round(dfc_jump + slice_start_secs + dfc_half_period)
    # account for rounding of dfc index exceeding slow slice
    if dfc_index > slice_stop_secs:
        split_index = slice_stop_secs
    elif dfc_index < slice_start_secs:
        split_index = slice_start_secs
    else:
        split_index = dfc_index
    return split_index


def _split_on_rot(slice_start_secs, slice_stop_secs, heading_frequency,
                  rate_of_turn):
    '''
    :param slice_start_secs: Start of slow slice in seconds.
    :type slice_start_secs: int or float
    :param slice_stop_secs: Stop of slow slice in seconds.
    :type slice_stop_secs: int or float
    :param heading_frequency: Frequency of Heading.
    :type heading_frequency: int or float
    :param rate_of_turn: Rate of turn array created from Heading diff.
    :type rate_of_turn: np.ma.MaskedArray
    :returns: Split index based on minimal rate of turn.
    :rtype: int or float or None
    '''
    rot_slice = slice(slice_start_secs * heading_frequency,
                      slice_stop_secs * heading_frequency)
    midpoint = (rot_slice.stop - rot_slice.start) // 2
    stopped_slices = np.ma.clump_unmasked(rate_of_turn[slices_int(rot_slice)])
    if not stopped_slices:
        return

    middle_stop = min(stopped_slices, key=lambda s: abs(s.start - midpoint))

    # Split half-way within the stop slice.
    stop_duration = middle_stop.stop - middle_stop.start
    rot_split_index = \
        rot_slice.start + middle_stop.start + (stop_duration // 2)
    # Get the absolute split index at 1Hz.
    split_index = py2round(rot_split_index / heading_frequency)
    return split_index


def split_segments(hdf, aircraft_info):
    '''
    TODO: DJ suggested not to use decaying engine oil temperature.

    Notes:
     * We do not want to split on masked superframe data if mid-flight (e.g.
       short section of corrupt data) - repair_mask without defining
       repair_duration should fix that.
     * Use turning alongside engine parameters to ensure there is no movement?
     XXX: Beware of pre-masked minimums to ensure we don't split on padded
     superframes

    TODO: Use L3UQAR num power ups for difficult cases?
    '''

    segments = []
    speed, vspeed, thresholds = _get_speed_parameter(hdf, aircraft_info)
    min_split_duration = thresholds['min_split_duration']

    # Look for heading first
    try:
        # Fetch Heading if available
        heading = hdf.get_param('Heading', valid_only=True)
    except KeyError:
        # try Heading True, otherwise fail loudly with a KeyError
        heading = hdf.get_param('Heading True', valid_only=True)

    eng_arrays, _ = _get_eng_params(hdf, align_param=heading)

    # Look for speed
    try:
        speed_array = repair_mask(speed.array, repair_duration=None,
                                  repair_above=thresholds['speed_threshold'])
    except ValueError:
        # speed array is masked, most likely under min threshold so it did
        # not go fast.
        logger.warning("speed is entirely masked. The entire contents of "
                       "the data will be a GROUND_ONLY slice.")
        return [_segment_type_and_slice(
            speed.array, speed.frequency, heading.array,
            heading.frequency, 0, hdf.duration, eng_arrays,
            aircraft_info, thresholds, hdf, vspeed)]

    speed_secs = len(speed_array) / speed.frequency

    # if Segment Split parameter is in hdf file someone has already done the hard work for us
    if 'Segment Split' in hdf:
        seg_split = hdf['Segment Split']
        split_flags = np.ma.where(seg_split.array == 'Split')
        start = 0

        if split_flags:
            for split_idx in split_flags[0]:
                split_idx = split_idx / seg_split.frequency
                segments.append(_segment_type_and_slice(
                    speed_array, speed.frequency, heading.array,
                    heading.frequency, start, split_idx, eng_arrays,
                    aircraft_info, thresholds, hdf, vspeed))
                start = split_idx
                logger.info("Split Flag found at at index '%d'.", split_idx)
            # Add remaining data to a segment.
            segments.append(_segment_type_and_slice(
                speed_array, speed.frequency, heading.array, heading.frequency,
                start, speed_secs, eng_arrays, aircraft_info, thresholds, hdf,
                vspeed))
        else:
            # if no split flags use whole file.
            logger.info("'Segment Split' found but no Splits found, using whole file.")
            segments.append(_segment_type_and_slice(
                speed_array, speed.frequency, heading.array, heading.frequency,
                start, speed_secs, eng_arrays, aircraft_info, thresholds, hdf,
                vspeed))
        return segments

    slow_array = np.ma.masked_less_equal(speed_array,
                                         thresholds['speed_threshold'])

    speedy_slices = np.ma.clump_unmasked(slow_array)
    if len(speedy_slices) <= 1:
        logger.info("There are '%d' sections of data where speed is "
                    "above the splitting threshold. Therefore there can only "
                    "be at maximum one flights worth of data. Creating a "
                    "single segment comprising all data.", len(speedy_slices))
        # Use the first and last available unmasked values to determine segment
        # type.
        return [_segment_type_and_slice(
            speed_array, speed.frequency, heading.array,
            heading.frequency, 0, speed_secs, eng_arrays,
            aircraft_info, thresholds, hdf, vspeed)]

    # suppress transient changes in speed around 80 kts
    slow_slices = slices_remove_small_slices(np.ma.clump_masked(slow_array), 10, speed.frequency)

    # Skip dropouts in HDF5 files (first identified on H175 helicopter)
    frame_count = hdf.get('FDRS Frame Counter')
    if frame_count:
        drops = np.ma.clump_masked(np.ma.where(frame_count.array == 0.0, np.ma.masked, 1.0)) # 1.0 is any value that's not masked
        dropouts = slices_multiply(slices_remove_small_gaps(drops, hz=frame_count.frequency),
                                   speed.frequency / frame_count.frequency)
        slow_slices = slices_and_not(slow_slices, dropouts)

    rate_of_turn = _rate_of_turn(heading)

    split_params_min, split_params_frequency \
        = _get_normalised_split_params(hdf, heading)
    if split_params_min is not None and not split_params_min.mask.all():
        split_params_min = repair_mask(split_params_min,
                                       frequency=split_params_frequency,
                                       repair_duration=None)

    if hdf.reliable_frame_counter:
        dfc = hdf['Frame Counter']
        dfc_diff = np.ma.diff(dfc.array)
        # Mask 'Frame Counter' incrementing by 1.
        dfc_diff = np.ma.masked_equal(dfc_diff, 1)
        # Mask 'Frame Counter' overflow where the Frame Counter transitions
        # from 4095 to 0.
        # Q: This used to be 4094, are there some Frame Counters which
        # increment from 1 rather than 0 or something else?
        dfc_diff = np.ma.masked_equal(dfc_diff, -4095)
        # Gap between difference values.
        dfc_half_period = (1 / dfc.frequency) / 2
    else:
        logger.info("'Frame Counter' will not be used for splitting since "
                    "'reliable_frame_counter' is False.")
        dfc = None

    start = 0
    last_fast_index = None
    for slow_slice in slow_slices:
        if slow_slice.start == 0:
            # Do not split if slow_slice is at the beginning of the data.
            # Since we are working with masked slices, masked padded superframe
            # data will be included within the first slow_slice.
            continue
        last_slow_slice = slow_slice.stop == len(speed_array)

        if last_fast_index is not None:
            fast_duration = (slow_slice.start -
                             last_fast_index) / speed.frequency
            if fast_duration < settings.MINIMUM_FAST_DURATION:
                logger.info("Disregarding short period of fast speed %s",
                            fast_duration)
                continue

        # Get start and stop at 1Hz.
        slice_start_secs = slow_slice.start / speed.frequency
        slice_stop_secs = slow_slice.stop / speed.frequency

        slow_duration = slice_stop_secs - slice_start_secs
        if slow_duration < min_split_duration:
            logger.info("Disregarding period of speed below '%s' "
                        "since '%s' is shorter than MINIMUM_SPLIT_DURATION "
                        "('%s').", thresholds['speed_threshold'], slow_duration,
                        min_split_duration)
            continue

        last_fast_index = slow_slice.stop

        # Find split based on minimum of engine parameters.
        if split_params_min is not None:
            eng_split_index, eng_split_value = _split_on_eng_params(
                slice_start_secs,
                slice_stop_secs,
                split_params_min,
                split_params_frequency)
        else:
            eng_split_index, eng_split_value = None, None

        # Split using 'Frame Counter'.
        if dfc is not None:
            dfc_split_index = _split_on_dfc(
                slice_start_secs, slice_stop_secs, dfc.frequency,
                dfc_half_period, dfc_diff, eng_split_index=eng_split_index)
            if dfc_split_index:
                if last_slow_slice and slice_stop_secs-dfc_split_index < min_split_duration:
                    dfc_split_index = slice_stop_secs
                segments.append(_segment_type_and_slice(
                    speed_array, speed.frequency, heading.array,
                    heading.frequency, start, dfc_split_index, eng_arrays,
                    aircraft_info, thresholds, hdf, vspeed))
                start = dfc_split_index
                logger.info("'Frame Counter' jumped within slow_slice '%s' "
                            "at index '%d'.", slow_slice, dfc_split_index)
                continue
            else:
                logger.info("'Frame Counter' did not jump within slow_slice "
                            "'%s'.", slow_slice)

        # Split using minimum of engine parameters.
        if eng_split_value is not None and \
           eng_split_value < settings.MINIMUM_SPLIT_PARAM_VALUE:
            logger.info("Minimum of normalised split parameters ('%s') was "
                        "below  ('%s') within "
                        "slow_slice '%s' at index '%d'.",
                        eng_split_value, settings.MINIMUM_SPLIT_PARAM_VALUE,
                        slow_slice, eng_split_index)
            if last_slow_slice and slice_stop_secs-eng_split_index < min_split_duration:
                eng_split_index = slice_stop_secs
            segments.append(_segment_type_and_slice(
                speed_array, speed.frequency, heading.array, heading.frequency,
                start, eng_split_index, eng_arrays, aircraft_info, thresholds,
                hdf, vspeed))
            start = eng_split_index
            continue
        else:
            logger.info("Minimum of normalised split parameters ('%s') was "
                        "not below MINIMUM_SPLIT_PARAM_VALUE ('%s') within "
                        "slow_slice '%s' at index '%s'.",
                        eng_split_value, settings.MINIMUM_SPLIT_PARAM_VALUE,
                        slow_slice, eng_split_index)

        # Split using rate of turn. Q: Should this be considered in other
        # splitting methods.
        if rate_of_turn is None or dfc:
            continue

        rot_split_index = _split_on_rot(slice_start_secs, slice_stop_secs,
                                        heading.frequency, rate_of_turn)
        if rot_split_index:
            if last_slow_slice and slice_stop_secs-rot_split_index < min_split_duration:
                rot_split_index = slice_stop_secs
            segments.append(_segment_type_and_slice(
                speed_array, speed.frequency, heading.array, heading.frequency,
                start, rot_split_index, eng_arrays, aircraft_info, thresholds,
                hdf, vspeed))
            start = rot_split_index
            logger.info("Splitting at index '%s' where rate of turn was below "
                        "'%s'.", rot_split_index,
                        settings.HEADING_RATE_SPLITTING_THRESHOLD)
            continue
        else:
            logger.info(
                "Aircraft did not stop turning during slow_slice "
                "('%s'). Therefore a split will not be made.", slow_slice)

        #Q: Raise error here?
        logger.warning("Splitting methods failed to split within slow_slice "
                       "'%s'.", slow_slice)

    # Add remaining data to a segment.
    if start < speed_secs:
        segments.append(_segment_type_and_slice(
            speed_array, speed.frequency, heading.array, heading.frequency,
            start, speed_secs, eng_arrays, aircraft_info, thresholds, hdf,
            vspeed))

    '''
    import matplotlib.pyplot as plt
    to_plot = [speed_array, heading.array, split_params_min]
    if dfc:
        to_plot.append(dfc.array / 4096.0)
    for look in to_plot:
        try:
            plt.plot(np.linspace(0, speed_secs, len(look)), look/np.ptp(look))
        except:
            pass
    for seg in segments:
        plt.plot([seg[1].start, seg[1].stop], [-0.5,+1])
    plt.show()
    '''

    return segments


def _get_speed_parameter(hdf, aircraft_info):

    thresholds = {}
    if aircraft_info.get('Engine Propulsion', None) == 'ROTOR':

        try:
            # Preferred source of rotor speed data
            parameter = hdf['Nr']
        except:
            # Alternative if dual sources available
            parameter = blend_parameters((hdf['Nr (1)'], hdf['Nr (2)']))
            parameter = P(name='Nr', array=parameter, data_type=parameter.dtype)

        thresholds['speed_threshold'] = settings.ROTORSPEED_THRESHOLD
        thresholds['min_duration'] = settings.ROTORSPEED_THRESHOLD_TIME
        # Very short dips in rotor speed before recording stops.
        thresholds['min_split_duration'] = settings.ROTOR_MINIMUM_SPLIT_DURATION
        # Let's try one minute on the ground as worth splitting.
        # Set to 30 sec as this gives two splits and two keeps in the test data set
        # TODO: add to settings
        thresholds['hash_min_samples'] = settings.AIRSPEED_HASH_MIN_SAMPLES
    else:
        parameter = hdf['Airspeed']
        thresholds['speed_threshold'] = settings.AIRSPEED_THRESHOLD
        thresholds['min_split_duration'] = settings.MINIMUM_SPLIT_DURATION
        thresholds['hash_min_samples'] = settings.AIRSPEED_HASH_MIN_SAMPLES
        thresholds['min_duration'] = settings.AIRSPEED_THRESHOLD_TIME
    vspeed = hdf.get('Vertical Speed')
    thresholds['vertical_speed_max'] = settings.VERTICAL_SPEED_FOR_CLIMB_PHASE
    thresholds['vertical_speed_min'] = settings.VERTICAL_SPEED_FOR_DESCENT_PHASE
    return parameter, vspeed, thresholds


def _mask_invalid_years(array, latest_year):
    '''
    Mask years which are in the future, not 2 or 4 digits or were made before the first recording FDR was invented.

    FDR date based on the Aeronautical Research Laboratory system named the "Red Egg", made by the British firm of
    S. Davall & Son in 1960.
    '''
    # ignore 4 digit years in the future
    array[array > latest_year] = np.ma.masked
    # mask out 3 digits up to the date of the first real FDR
    array[(array >= 100) & (array < 1960)] = np.ma.masked
    # mask out any 2 digit years in future
    two_digits = (array >= 0) & (array < 100)
    in_future = two_digits & (array > latest_year % 100)
    array[in_future] = np.ma.masked
    return array


def get_dt_arrays(hdf, fallback_dt, validation_dt, valid_slices=[]):
    now = datetime.utcnow().replace(tzinfo=pytz.utc)

    if fallback_dt:
        fallback_dts = []
        for secs in range(0, int(hdf.duration)):
            fallback_dts.append(fallback_dt + timedelta(seconds=secs))

    onehz = P(frequency=1)
    dt_arrays = []
    precise = True
    dt_parameter_origin = {}
    for name in ('Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'):
        param = hdf.get(name)
        if param:
            if name == 'Year':
                year = getattr(validation_dt, 'year', None) or now.year
                param.array = _mask_invalid_years(param.array, year)
            # do not interpolate date/time parameters to avoid rollover issues
            array = align(param, onehz, interpolate=False)
            if len(array) == 0 or np.ma.count(array) == 0:
                logger.warning(f"No valid values returned for {name}")
            elif (np.ma.all(array == 0)) and not (name == 'Hour' and len(array) < 3600): # Hour can be 0 for up to 1 hour (after midnight)
                # Other than the year 2000 or possibly 2100, no date values
                # can be all 0's
                logger.warning(f"Only zero values returned for {name}")
            else:
                # values returned, continue
                dt_arrays.append(array)
                dt_parameter_origin[name]=PRECISE # parameter is good
                continue

        if fallback_dt:
            precise = False
            array = [getattr(dt, name.lower()) for dt in fallback_dts]
            logger.warning("%s not available, using range from %d to %d from fallback_dt %s",
                           name, array[0], array[-1], fallback_dt)
            dt_arrays.append(array)
            dt_parameter_origin[name]=FALLBACK_VALIDATION if param else FALLBACK_NO_PARAM
            continue
        else:
            raise TimebaseError(f"Required parameter '{name}' not available")

    validated_dt_arrays = [mask_outside_slices(np.ma.array(a), valid_slices) for a in dt_arrays] if valid_slices else dt_arrays

    return validated_dt_arrays, precise, dt_parameter_origin


def has_constant_time(hdf):
    """
    Test if the time is not changing in the time base parameters.

    In some cases we have access to initial datetime for the whole recording.
    It manifests itself by "flat" time characteristics: the time is recorded in
    the flights, but it is constant. In those cases we assume that the initial
    time is the actual time of the recorder power-on.  We can use it as the
    more precise fallback_dt.
    """
    minutes = hdf.get('Minute')
    if minutes is None:
        return False  # We don't know

    samples = minutes.array.size
    duration = samples * minutes.hz
    return samples > 5 and duration > 5 and np.ptp(minutes.array) == 0


def calculate_fallback_dt(hdf, fallback_dt=None, validation_dt=None,
                          fallback_relative_to_start=True,
                          frame_doubled=False, dt_origin_kwargs={}):
    """
    Check the time parameters in the HDF5 file and update the fallback_dt.

    Takes into account adjustment of fallback datetime if it's relative to
    the end of data and if there's a constant timebase within the data it
    will use this as the fallback datetime.
    """
    if fallback_dt and not fallback_relative_to_start:
        # fallback_dt is relative to the end of the data; remove the data
        # duration to make it relative to the start of the data
        secs = hdf.duration
        fallback_dt -= timedelta(seconds=secs)
        logger.info("Reduced fallback_dt by %ddays %dhr %dmin to %s",
                    secs // 86400, secs % 86400 // 3600,
                    secs % 86400 % 3600 // 60, fallback_dt)

    if not frame_doubled or not has_constant_time(hdf):
        # we don't need to do any further corrections
        return fallback_dt

    # Only use this for certain recorders where we use timestamps from headers
    # to provide a fallback which happens to be a constant value.
    try:
        dt_arrays, _, dt_parameter_origin = get_dt_arrays(hdf, fallback_dt, validation_dt)
        timebase = calculate_timebase(*dt_arrays)
        fallback_changes = [v for v in dt_parameter_origin.values() if 'fallback' in v]
    except (KeyError, ValueError):
        # The time parameters are not available/operational
        return fallback_dt
    else:
        now = datetime.utcnow().replace(tzinfo=pytz.utc)
        assert timebase <= now, (
            f"Fallback time '{fallback_dt}' in the future is not allowed. Current time is '{now}'.")
        if validation_dt is not None:
            assert timebase <= validation_dt, (
                "Fallback time '%s' ahead of validation time is not allowed. "
                "Validation time is '%s'." % (timebase, validation_dt))
        logger.warning("Time doesn't change, using the starting time as the fallback_dt")
        return timebase


def get_valid_dt_slices(hdf, min_threshold_count=360):
    aspd = hdf.get('Airspeed')

    if not aspd:
        return []

    aspd = align(aspd, P(hz=1))
    aspd_gt_threshold = np.ma.where(aspd > settings.AIRSPEED_THRESHOLD)[0].size

    return np.ma.clump_unmasked(aspd) if aspd_gt_threshold > min_threshold_count else []


def _calculate_start_datetime(hdf, fallback_dt, validation_dt):
    """
    Calculate start datetime.

    :param hdf: Flight data HDF file
    :type hdf: hdf_access object
    :param fallback_dt: Used to replace elements of datetimes which are not
        available in the hdf file (e.g. YEAR not being recorded)
    :type fallback_dt: datetime

    HDF params used:
    :Year: Optional (defaults to 1970)
    :Month: Optional (defaults to 1)
    :Day: Optional (defaults to 1)
    :Hour: Required
    :Minute: Required
    :Second: Required

    If required parameters are not available and fallback_dt is not provided,
    a TimebaseError is raised
    """
    now = datetime.utcnow().replace(tzinfo=pytz.utc)
    if fallback_dt is not None:
        if (fallback_dt.tzinfo is None or fallback_dt.tzinfo.utcoffset(fallback_dt) is None):
            # Assume fallback_dt is UTC
            fallback_dt = fallback_dt.replace(tzinfo=pytz.utc)

        # Even if fallback_dt is UTC, there's a chance that the timezone was
        # wrong, so if we're still in the future, let's see if it's less than
        # 12 hours, and if it is, try to fix it
        if fallback_dt >= now and (fallback_dt - now).seconds / 3600 < 12:
            fallback_dt -= fallback_dt - now

    # align required parameters to 1Hz
    dt_arrays, precise_timestamp, timestamp_configuration = get_dt_arrays(hdf, fallback_dt, validation_dt,
                                                                          valid_slices=get_valid_dt_slices(hdf))

    length = max(len(a) for a in dt_arrays)
    if length > 1:
        # ensure all arrays are the same length
        for n, arr in enumerate(dt_arrays):
            if len(arr) == 1:
                # repeat to the correct size
                arr = np.repeat(arr, length)
                dt_arrays[n] = arr
            elif len(arr) != length:
                raise ValueError('After align, all arrays should be the same length')
            else:
                pass

    # establish timebase for start of data
    if has_constant_time(hdf):
        timebase = fallback_dt
    else:
        try:
            timebase = calculate_timebase(*dt_arrays)
        except (KeyError, ValueError) as err:
            raise TimebaseError(f'Error with timestamp values: {err}')

    if timebase > now:
        # Flight Data Analysis in the future is a challenge, lets see if we can correct this first...
        if 'Day' not in hdf:
            # unlikely to have year, month or day.
            # Scenario: that fallback_dt is of the current day but recorded time is in the future of the fallback time,
            # therefore resulting in a futuristic date.
            a_day_before = timebase - relativedelta(days=1)
            if a_day_before < now:
                logger.info(f'Timebase was in the future, using a DAY before satisfies requirements: {a_day_before}')
                return a_day_before, False, timestamp_configuration
            # continue to take away a Year
        if 'Year' not in hdf:
            # remove a year from the timebase
            a_year_before = timebase - relativedelta(years=1)
            if a_year_before < now:
                logger.info(f'Timebase was in the future, using a YEAR before satisfies requirements: {a_year_before}')
                return a_year_before, False, timestamp_configuration

        raise TimebaseError(f"Timebase '{timebase}' is in the future.")

    if settings.MAX_TIMEBASE_AGE and timebase < (now - timedelta(days=settings.MAX_TIMEBASE_AGE)):
        # Only allow recent timebases.
        raise TimebaseError(f"Timebase '{timebase}' older than the allowed '{settings.MAX_TIMEBASE_AGE}' days.")

    logger.info(f'Valid timebase identified as {timebase}')
    return timebase, precise_timestamp, timestamp_configuration


def append_segment_info(hdf_segment_path, segment_type, segment_slice, part,
                        fallback_dt=None, validation_dt=None, aircraft_info={}):
    """
    Get information about a segment such as type, hash, etc. and return a named tuple.

    If a valid timestamp can't be found, it creates start_dt as epoch(0) i.e. datetime(1970,1,1,1,0). Go-fast dt and
    Stop dt are relative to this point in time.

    :param hdf_segment_path: path to HDF segment to analyse
    :type hdf_segment_path: string
    :param segment_slice: Slice of this segment relative to original file.
    :type segment_slice: slice
    :param part: Numeric part this segment was in the original data file (1 indexed)
    :type part: Integer
    :param fallback_dt: Used to replace elements of datetimes which are not available in the hdf file (e.g. YEAR not
        being recorded)
    :type fallback_dt: datetime
    :returns: Segment named tuple
    :rtype: Segment
    """
    # build information about a slice
    with hdf_file(hdf_segment_path) as hdf:
        speed, _, thresholds = _get_speed_parameter(hdf, aircraft_info)
        duration = hdf.duration
        try:
            start_datetime, precise_timestamp, timestamp_configuration = _calculate_start_datetime(
                hdf, fallback_dt, validation_dt)
        except TimebaseError:
            # Warn the user and store the fake datetime. The code on the other side should check the datetime and avoid
            # processing this file
            logger.exception('Unable to calculate timebase, using 1970-01-01 00:00:00+0000!')
            start_datetime = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
            precise_timestamp = False
            timestamp_configuration = None
        stop_datetime = start_datetime + timedelta(seconds=duration)
        hdf.start_datetime = start_datetime

    if segment_type in ('START_AND_STOP', 'START_ONLY', 'STOP_ONLY'):
        # we went fast, so get the index
        spd_above_threshold = \
            np.ma.where(speed.array > thresholds['speed_threshold'])
        go_fast_index = spd_above_threshold[0][0] / speed.frequency
        go_fast_datetime = \
            start_datetime + timedelta(seconds=int(go_fast_index))
        # Identification of raw data speed hash
        speed_hash_sections = runs_of_ones(
            speed.array.data > thresholds['speed_threshold'])
        speed_hash = hash_array(
            speed.array.data, speed_hash_sections, thresholds['hash_min_samples'])
    #elif segment_type == 'GROUND_ONLY':
        ##Q: Create a groundspeed hash?
        #pass
    else:
        go_fast_index = None
        go_fast_datetime = None
        # if not go_fast, create hash from entire file
        speed_hash = sha_hash_file(hdf_segment_path)
    segment = Segment(
        segment_slice,
        segment_type,
        part,
        hdf_segment_path,
        speed_hash,
        start_datetime,
        go_fast_datetime,
        stop_datetime,
        precise_timestamp,
        timestamp_configuration,
    )
    return segment


def split_hdf_to_segments(hdf_path, aircraft_info, fallback_dt=None,
                          validation_dt=None, fallback_relative_to_start=True,
                          draw=False, dest_dir=None, pre_file_kwargs={},
                          dt_origin_kwargs={}):
    """
    Main method - analyses an HDF file for flight segments and splits each
    flight into a new segment appropriately.

    :param hdf_path: path to HDF file
    :type hdf_path: string
    :param aircraft_info: Information which identify the aircraft, specfically
        with the keys 'Tail Number', 'MSN'...
    :type aircraft_info: Dict
    :param fallback_dt: A datetime which is as close to the end of the data
        file as possible. Used to replace elements of datetimes which are not
        available in the hdf file (e.g. YEAR not being recorded)
    :type fallback_dt: datetime
    :param draw: Whether to use matplotlib to plot the flight
    :type draw: Boolean
    :param dest_dir: Destination directory, if None, the source file directory
        is used
    :type dest_dir: str
    :param pre_file_kwargs: Pre-file analysis keyword arguments.
    :type pre_file_kwargs: dict
    :returns: List of Segments
    :rtype: List of Segment recordtypes ('slice type part duration path hash')
    """
    logger.debug(f"Processing file: {hdf_path}")

    if dest_dir is None:
        dest_dir = os.path.dirname(hdf_path)

    if draw:
        from analysis_engine.plot_flight import plot_essential
        plot_essential(hdf_path)

    if isinstance(validation_dt, string_types):
        validation_dt = datetime.strptime(validation_dt, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=pytz.utc)

    with hdf_file(hdf_path) as hdf:

        # Confirm aircraft tail for the entire datafile
        logger.debug("Validating aircraft matches that recorded in data")
        validate_aircraft(aircraft_info, hdf)

        # now we know the Aircraft is correct, go and do the PRE FILE ANALYSIS
        hook = hooks.PRE_FILE_ANALYSIS
        if hook:
            logger.debug("Performing PRE_FILE_ANALYSIS action '%s' with options: %s",
                         getattr(hook, 'func_name', getattr(hook, '__name__')),
                         pre_file_kwargs)
            hook(hdf, aircraft_info, **pre_file_kwargs)
        else:
            logger.info("No PRE_FILE_ANALYSIS actions to perform")

        # ARINC 717 data has frames or superframes. ARINC 767 will be split
        # on a minimum boundary of 4 seconds for the analyser.
        if hdf.arinc == '717':
            boundary = 64 if hdf.superframe_present else 4
        elif hdf.arinc == '767':
            boundary = max(int(1 / min(hdf.frequencies)), 4)
        else:
            # For CSV and similar single frequency formats, there is no boundary constraint.
            boundary = 4

        segment_tuples = split_segments(hdf, aircraft_info)
        frame_doubled = aircraft_info.get('Frame Doubled', False)

        fallback_dt = calculate_fallback_dt(hdf, fallback_dt, validation_dt,
                                            fallback_relative_to_start,
                                            frame_doubled, dt_origin_kwargs)

    # process each segment (into a new file) having closed original hdf_path
    segments = []
    previous_stop_dt = None
    for part, (segment_type, segment_slice, start_padding) in enumerate(segment_tuples,
                                                         start=1):
        # write segment to new split file (.001)
        basename = os.path.basename(hdf_path)
        dest_basename = os.path.splitext(basename)[0] + '.%03d.hdf5' % part
        dest_path = os.path.join(dest_dir, dest_basename)
        logger.debug(f"Writing segment {part}: {dest_path}")

        write_segment(hdf_path, segment_slice, dest_path, boundary,
                      submasks=('arinc', 'invalid_states', 'padding', 'saturation'))

        # adjust fallback time to account for any padding added at start of segment
        segment_start_dt = fallback_dt - timedelta(seconds=start_padding)

        segment = append_segment_info(
            dest_path, segment_type, segment_slice, part,
            fallback_dt=segment_start_dt, validation_dt=validation_dt,
            aircraft_info=aircraft_info)
        if previous_stop_dt and segment.start_dt < previous_stop_dt - timedelta(0, 4):
            # In theory, this should not happen - but be warned of superframe
            # padding?
            logger.warning(
                f"Segment start_dt '{segment.start_dt}' comes before the previous segment ended '{previous_stop_dt}'")
        previous_stop_dt = segment.stop_dt

        if fallback_dt:
            # move the fallback_dt on to be relative to start of next segment slice
            fallback_dt += timedelta(seconds=(segment_slice.stop - segment_slice.start))
        segments.append(segment)
        if draw:
            plot_essential(dest_path)

    if draw:
        # show all figures together
        from matplotlib.pyplot import show
        show()
        #close('all') # closes all figures

    return segments


def parse_cmdline():
    import argparse

    now = datetime.utcnow().replace(tzinfo=pytz.utc)

    def valid_date(s):
        try:
            return datetime.strptime(s, '%Y-%m-%d %H:%M').replace(tzinfo=pytz.utc)
        except ValueError:
            raise argparse.ArgumentTypeError('not a valid date and time: %s' % s)

    parser = argparse.ArgumentParser(description='Split a data file into flight segments.')
    parser.add_argument('file', type=str,
                        help='Path of file to process.')
    parser.add_argument('-tail', '--tail', dest='tail_number', type=str, default='G-FDSL',
                        help='Aircraft Tail Number for processing.')
    parser.add_argument(
        '-t', '--fallback-datetime', type=valid_date, default=now, metavar='DATETIME',
        help='Date and time at beginning of data, used if parameters unreliable (%%Y-%%m-%%d %%H:%%M)',
    )
    parser.add_argument(
        '-d', '--validation-datetime', type=valid_date, default=now, metavar='DATETIME',
        help='Date and time used to validate time parameters, usually upload time (%%Y-%%m-%%d %%H:%%M)',
    )
    parser.add_argument('-L', '--log-level', default=None, help='Log level')
    parser.add_argument('-q', '--quiet', action='store_true', help="Don't output messages")

    args = parser.parse_args()

    if args.log_level:
        log_level = args.log_level.upper()
        # Convert log level name to value
        if not hasattr(logging, log_level):
            raise parser.error(f'Invalid log level `{log_level}`')
        args.log_level_number = getattr(logging, log_level)
    else:
        args.log_level_number = logging.WARNING

    if args.quiet:
        args.log_level_number = 1000

    return args, parser


def main():
    args, parser = parse_cmdline()

    if not args.quiet:
        print('FlightDataSplitter (c) Copyright 2020 Flight Data Services, Ltd.')
        print('  - Powered by POLARIS')
        print('  - http://www.flightdatacommunity.com')
        print()

    import os
    from analysis_engine.utils import get_aircraft_info
    from flightdatautilities.filesystem_tools import copy_file

    logger = logging.getLogger()
    logger.setLevel(args.log_level_number)

    ac_info = get_aircraft_info(args.tail_number)
    hdf_copy = copy_file(args.file, postfix='_split')
    logger.info("Working on copy: %s", hdf_copy)
    segments = split_hdf_to_segments(
        hdf_copy,
        ac_info,
        fallback_dt=args.fallback_datetime,
        validation_dt=args.validation_datetime,
        fallback_relative_to_start=False,
        draw=False)

    # Rename the segment filenames to be able to use glob()
    for segment in segments:
        dir, fn = os.path.split(segment.path)
        name, ext = os.path.splitext(fn)
        new_fn = '-'.join((name, 'SEGMENT', segment.type)) + ext
        new_path = os.path.join(dir, new_fn)
        os.rename(segment.path, new_path)
        segment.path = new_path

    if not args.quiet:
        import pprint
        pprint.pprint(segments)


if __name__ == '__main__':
    main()
