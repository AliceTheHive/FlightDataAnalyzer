# -*- coding: utf-8 -*-
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
##############################################################################

from __future__ import print_function

import itertools
import logging
import math
import numpy as np
from numpy.ma.extras import _ezclump as ezclump
import pytz

from builtins import zip
from collections import OrderedDict, namedtuple
from copy import copy, deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from math import ceil, copysign, cos, floor, log, radians, sin, sqrt
from operator import itemgetter
from scipy import interpolate as scipy_interpolate, optimize
from scipy.ndimage import filters
from scipy.signal import medfilt
from six.moves import zip_longest

from hdfaccess.parameter import MappedArray

from flightdatautilities import aircrafttables as at, units as ut
from flightdatautilities.geometry import cross_track_distance, great_circle_distance__haversine
from flightdatautilities.numpy_utils import (
    py2round,
    slices_int,
)

from analysis_engine.settings import (
    BUMP_HALF_WIDTH,
    HYSTERESIS_FPALT_CCD,
    ILS_CAPTURE,
    ILS_CAPTURE_ROC,
    ILS_ESTABLISHED_DURATION,
    ILS_LOC_SPREAD,
    ILS_GS_SPREAD,
    ALTITUDE_RADIO_MIN_OVERFLOW,
    REPAIR_DURATION,
    RUNWAY_HEADING_TOLERANCE,
    RUNWAY_ILSFREQ_TOLERANCE,
    SLOPE_FOR_TOC_TOD,
    TRUCK_OR_TRAILER_INTERVAL,
    TRUCK_OR_TRAILER_PERIOD,
    WRAPPING_PARAMS,
)

# There is no numpy masked array function for radians, so we just multiply thus:
deg2rad = radians(1.0)

logger = logging.getLogger(name=__name__)

Value = namedtuple('Value', 'index value')


class InvalidDatetime(ValueError):
    pass


class IntegrationError(ValueError):
    '''
    Exception raised when node is always expected to be derived.

    Useful to safeguard the processing of important nodes. We want to fail
    early instead of having to check which nodes were derived.
    '''
    pass


def all_deps(cls, available):
    return all(x in available for x in cls.get_dependency_names())


def any_deps(cls, available):
    return any(x in available for x in cls.get_dependency_names())


def all_of(names, available):
    '''
    Returns True if all of the names are within the available list.
    i.e. names is a subset of available
    '''
    return all(name in available for name in names)


def any_of(names, available):
    '''
    Returns True if any of the names are within the available list.

    NB: Was called "one_of" but that implies ONLY one name is available.
    '''
    return any(name in available for name in names)


def any_one_of(names, available):
    '''
    Returns True if any ONE, and only one, of the names is within the available list.
    '''
    return sum(name in available for name in names) == 1


def air_track(lat_start, lon_start, lat_end, lon_end, spd, hdg, alt_aal, frequency):
    """
Computation of the air track for cases where recorded latitude and longitude
are not available but the origin and destination airport locations are known.

Note that as the data is computed for each half of the flight from the origin
and destination coordinates, with errors due to wind and earth curvature
appearing as a jump in the middle of the flight. Either groundspeed or
airspeed may be used.

:param lat_start: Fixed latitude point at the origin.
:type lat_start: float, latitude degrees.
:param lon_start: Fixed longitude point at the origin.
:type lon_start: float, longitude degrees.
:param lat_end: Fixed latitude point at the destination.
:type lat_end: float, latitude degrees.
:param lon_end: Fixed longitude point at the destination.
:type lon_end: float, longitude degrees.
:param spd: Speed (air or ground) in knots
:type spd: Numpy masked array.
:param hdg: Heading (ideally true) in degrees.
:type hdg: Numpy masked array.
:param frequency: Frequency of the groundspeed and heading data
:type frequency: Float (units = Hz)

:returns
:param lat_track: Latitude of computed ground track
:type lat_track: Numpy masked array
:param lon_track: Longitude of computed ground track
:type lon_track: Numpy masked array.

:error conditions
:Fewer than 5 valid data points, returns None, None
:Invalid mode fails with ValueError
:Mismatched array lengths fails with ValueError
"""
    def compute_track(lat_start, lon_start, lat_end, lon_end, spd, hdg, alt_aal, frequency):

        def measure_jump(arr, n):
            # For an array, the jump from n-1 to n rejecting mean slope is:
            d = (arr[n-2] - 3*arr[n-1] + 3*arr[n] - arr[n+1])/2.0
            return d

        lat = np_ma_zeros_like(spd)
        lon = np_ma_zeros_like(spd)
        half_len = int(len(spd)/2.0)
        lat[0]=lat_start
        lon[0]=lon_start
        lat[-1]=lat_end
        lon[-1]=lon_end

        spd_north = spd * np.ma.cos(hdg_rad)
        spd_east = spd * np.ma.sin(hdg_rad)

        scale = ut.multiplier(ut.KT, ut.METER_S)

        # Compute displacements in metres north and east of the starting point.
        north_from_start = integrate(spd_north[:half_len], frequency, scale=scale)
        east_from_start = integrate(spd_east[:half_len], frequency, scale=scale)
        bearings = np.ma.array(np.rad2deg(np.arctan2(east_from_start, north_from_start)))
        distances = np.ma.array(np.ma.sqrt(north_from_start**2 + east_from_start**2))
        lat[:half_len],lon[:half_len] = latitudes_and_longitudes(
            bearings, distances, {'latitude':lat_start, 'longitude':lon_start})

        south_from_end = integrate(spd_north[half_len:], frequency, scale=scale, direction='reverse')
        west_from_end = integrate(spd_east[half_len:], frequency, scale=scale, direction='reverse')
        bearings = (np.ma.array(np.rad2deg(np.arctan2(west_from_end, south_from_end)))+180.0) % 360.0
        distances = np.ma.array(np.ma.sqrt(south_from_end**2 + west_from_end**2))
        lat[half_len:],lon[half_len:] = latitudes_and_longitudes(
            bearings, distances, {'latitude':lat_end, 'longitude':lon_end})


        # There will be a jump in the middle of the data caused by
        # accumulated errors. We remove this to produce a smooth track.
        dlat = measure_jump(lat, half_len)
        dlon = measure_jump(lon, half_len)

        # Compute the area under first half the altitude profile curve to weight the correction
        profile = integrate(alt_aal[:half_len], frequency)
        lat[:half_len] += profile * (dlat/(profile[-1] * 2.0))
        lon[:half_len] += profile * (dlon/(profile[-1] * 2.0))

        # Compute the area under second half the altitude profile curve
        profile = integrate(alt_aal[half_len:], frequency, direction='backwards')
        lat[half_len:] -= profile * (dlat/(profile[0] * 2.0))
        lon[half_len:] -= profile * (dlon/(profile[0] * 2.0))

        return lat, lon

    # First check that the gspd/hdg arrays are sensible.
    if len(spd) != len(hdg):
        raise ValueError('Ground_track requires equi-length speed and '
                         'heading arrays')

    # It's not worth doing anything if there is too little data
    if np.ma.count(spd) < 5:
        return None, None

    # Prepare arrays for the outputs
    lat = np_ma_masked_zeros_like(spd)
    lon = np_ma_masked_zeros_like(spd)

    # Do some spadework to prepare the ground
    repair_mask(spd, repair_duration=None)
    repair_mask(hdg, repair_duration=None)
    # Get longest slice as we want the flight not a high speed RTO
    valid_slices = np.ma.clump_unmasked(np.ma.masked_less_equal(spd, 50.0))
    valid_slice = max(valid_slices, key=lambda p: p.stop - p.start)
    hdg_rad = hdg[valid_slice] * deg2rad

    lat[valid_slice], lon[valid_slice] = compute_track(lat_start, lon_start,
                                                       lat_end, lon_end,
                                                       spd[valid_slice],
                                                       hdg[valid_slice],
                                                       alt_aal[valid_slice],
                                                       frequency)

    repair_mask(lat, repair_duration=None, extrapolate=True)
    repair_mask(lon, repair_duration=None, extrapolate=True)
    return lat, lon


def is_power2(number):
    """
    States if a number is a power of two. Forces floats to Int.
    Ref: http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
    """
    if number % 1:
        return False
    num = int(number)
    return num > 0 and ((num & (num - 1)) == 0)


def is_power2_fraction(number):
    '''
    TODO: Tests

    :type number: int or float
    :returns: if the number is either a power of 2 or a fraction, e.g. 4, 2, 1, 0.5, 0.25
    :rtype: bool
    '''
    if number < 1:
        number = 1 / number
    return is_power2(number)


def straighten_parameter_array(param, copy_array=False):
    '''
    Unwrap the parameters which have wraparound characteristics (angles).

    :param param: parameter to be unwrapped
    :type param: Parameter
    :param copy_array: should the array be copied if no straightening was
        taking place
    :type copy_array: bool
    :returns: array of unwrapped values
    :rtype: np.ma.array
    '''
    if param.name in WRAPPING_PARAMS:
        if param.name == 'Altitude STD (Fine)':
            return straighten_altitudes(param.array, None, 5000)
        elif param.name.startswith('Longitude'):
            return straighten_longitude(param.array)
        else:
            return straighten_headings(param.array)

    if copy_array:
        return param.array.copy()
    else:
        return param.array


def wrap_array(param_name, array):
    '''
    Restore the straightened array to normal range.

    :param param_name: name of the parameter to be wrapped
    :type param: string
    :param array: the array to be wrapped
    :type array: np.ma.array
    :returns: array of wrapped values
    :rtype: np.ma.array
    '''
    if param_name in WRAPPING_PARAMS:
        if param_name == 'Altitude STD (Fine)':
            # FIXME: do we need to wrap the values again?
            return array
        elif param_name.startswith('Longitude'):
            # wrap around 180
            return (array + 180) % 360 - 180
        else:
            # wrap around 360
            return array % 360
    return array


def align(slave, master, interpolate=True):
    """
    This function takes two parameters which will have been sampled at
    different rates and with different measurement offsets in time, and
    aligns the slave parameter's samples to match the master parameter. In
    this way the master and aligned slave data may be processed without
    timing errors.

    The values of the returned array will be those of the slave parameter,
    aligned to the master and adjusted by linear interpolation. The initial
    or final values will be masked zeros if they lie outside the timebase of
    the slave parameter (i.e. we do not extrapolate). The offset and hz for
    the returned masked array will be those of the master parameter.

    MappedArray slave parameters (discrete/multi-state) will not be
    interpolated, even if interpolate=True.

    Anything other than discrete or multi-state will result in interpolation
    of the data across each sample period.

    WARNING! Not tested with ASCII arrays.

    :param slave: The parameter to be aligned to the master
    :type slave: Parameter objects
    :param master: The master parameter
    :type master: Parameter objects
    :param interpolate: Whether to interpolate parameters (multistates exempt)
    :type interpolate: Bool

    :raises AssertionError: If the arrays and sample rates do not equate to the same overall data duration.

    :returns: Slave array aligned to master.
    :rtype: np.ma.array
    """
    aligned = align_args(
        straighten_parameter_array(slave),
        slave.frequency,
        slave.offset,
        master.frequency,
        master.offset,
        interpolate=interpolate,
    )
    return wrap_array(slave.name, aligned)


def align_args(slave_array, slave_frequency, slave_offset, master_frequency, master_offset=0, interpolate=True):
    '''
    align implementation abstracted from Parameter class interface.

    TODO: Upscaling MappedArrays should result in transitions at midpoints rather
          than simply repeating the array as the exact time is unknown, e.g.
          ['-', '-', 'Up'] * 4 == ['-', '-', '-', '-',
                                   '-', '-' 'Up', 'Up', # currently '-', '-', '-', '-'
                                   'Up', 'Up', 'Up', 'Up']
          This is arguably more accurate, as the state could have changed
          anywhere between the sample where the state changed and the previous
          one.

    :type slave_array: np.ma.masked_array
    :type slave_frequency: int or float
    :type slave_offset: int or float
    :type master_frequency: int or float
    :type master_offset: int or float
    :type interpolate: bool
    :returns: Slave array aligned to master.
    :rtype: array of same type as slave_array
    '''

    original_array = slave_array
    if slave_array.dtype.type is np.string_:
        # by converting string arrays to multistate arrays we can use
        # established aligning process
        slave_array = string_array_to_mapped_array(slave_array)

    if isinstance(slave_array, MappedArray):  # Multi-state array.
        # force disable interpolate!
        mappings = slave_array.values_mapping
        slave_array = slave_array.raw
        interpolate = False
        _dtype = slave_array.dtype
    elif isinstance(slave_array, np.ma.MaskedArray):
        _dtype = float
    else:
        raise ValueError('Cannot align slave array of unknown type.')

    if len(slave_array) == 0:
        # No elements to align, avoids exception being raised in the loop below.
        return slave_array
    if slave_frequency == master_frequency and slave_offset == master_offset:
        # No alignment is required, return the slave's array unchanged.
        if isinstance(original_array, MappedArray):
            return original_array
        else:
            return slave_array

    # Get the sample rates for the two parameters
    wm = master_frequency
    ws = slave_frequency
    slowest = min(wm, ws)

    # The timing offsets comprise of word location and possible latency.
    # Express the timing disparity in terms of the slave parameter sample interval
    delta = (master_offset - slave_offset) * slave_frequency

    # If the slowest sample rate is less than 1 Hz, we extend the period and
    # so achieve a lowest rate of one per period.
    if slowest < 1:
        wm /= slowest
        ws /= slowest

    # Check the values are in ranges we have tested
    assert is_power2(wm) or not wm % 5, \
           "master @ %sHz; wm=%s" % (master_frequency, wm)
    assert is_power2(ws) or not ws % 5, \
           "slave @ %sHz; ws=%s" % (slave_frequency, ws)

    # Trap 5, 10 or 20Hz parameters that have non-zero offsets (this case is not currently covered)
    if master_offset and not wm % 5:
        raise ValueError('Align: Master offset non-zero at sample rate %sHz' % master_frequency)
    if slave_offset and not ws % 5:
        raise ValueError('Align: Slave offset non-zero at sample rate %sHz' % slave_frequency)

    # Compute the sample rate ratio:
    r = wm / float(ws)

    # Here we create a masked array to hold the returned values that will have
    # the same sample rate and timing offset as the master
    len_aligned = int(len(slave_array) * r)
    if len_aligned != (len(slave_array) * r):
        raise ValueError("Array length problem in align. Probable cause is flight cutting not at superframe boundary")

    slave_aligned = np.ma.zeros(int(len(slave_array) * r), dtype=_dtype)

    # Where offsets are equal, the slave_array recorded values remain
    # unchanged and interpolation is performed between these values.
    # - and we do not interpolate mapped arrays!
    if not delta and interpolate and (is_power2(slave_frequency) and
                                      is_power2(master_frequency)):
        slave_aligned.mask = True
        if master_frequency > slave_frequency:
            # populate values and interpolate
            slave_aligned[0::int(r)] = slave_array[0::1]
            # Interpolate and do not extrapolate masked ends or gaps
            # bigger than the duration between slave samples (i.e. where
            # original slave data is masked).
            # If array is fully masked, return array of masked zeros
            dur_between_slave_samples = 1.0 / slave_frequency
            return repair_mask(
                slave_aligned,
                frequency=master_frequency,
                repair_duration=dur_between_slave_samples,
                raise_entirely_masked=False,
            )

        else:
            # step through slave taking the required samples
            return slave_array[0::int(1/r)]
    # wm & ws used for indexing from now on ensure they are integars
    wm = int(wm)
    ws = int(ws)
    # Each sample in the master parameter may need different combination parameters
    for i in range(wm):
        bracket = (i / r) + delta
        # Interpolate between the hth and (h+1)th samples of the slave array
        h = int(floor(bracket))
        h1 = h + 1

        # Compute the linear interpolation coefficients, b & a
        b = bracket - h

        # Cunningly, if we are interpolating (working with mapped arrays e.g.
        # discrete or multi-state parameters), by reverting to 1,0 or 0,1
        # coefficients we gather the closest value in time to the master
        # parameter.
        if not interpolate:
            b = py2round(b)

        # Either way, a is the residual part.
        a = 1 - b

        if h < 0:
            if h<-ws:
                raise ValueError('Align called with excessive timing mismatch')
            # slave_array values do not exist in aligned array
            if ws==1:
                slave_aligned[i+wm::wm] = a*slave_array[h+ws:-ws:ws] + b*slave_array[h1+ws::ws]
            else:
                slave_aligned[i+wm::wm] = a*slave_array[h+ws:-ws:ws] + b*slave_array[h1+ws:1-ws:ws]
            # We can't interpolate the inital values as we are outside the
            # range of the slave parameters.
            # Treat ends as "padding"; Value of 0 and Masked.
            slave_aligned[i] = 0
            slave_aligned[i] = np.ma.masked
        elif h1 >= ws:
            slave_aligned[i:-wm:wm] = a*slave_array[h:-ws:ws] + b*slave_array[h1::ws]
            # At the other end, we run out of slave parameter values so need to
            # pad to the end of the array.
            # Treat ends as "padding"; Value of 0 and Masked.
            slave_aligned[i-wm] = 0
            slave_aligned[i-wm] = np.ma.masked
        else:
            # Sheer bliss. We can compute slave_aligned across the whole
            # range of the data without having to take special care at the
            # ends of the array.
            slave_aligned[i::wm] = a*slave_array[h::ws] + b*slave_array[h1::ws]

    if isinstance(original_array, MappedArray) or original_array.dtype.type is np.string_:
        # return back to mapped array
        mapped_array = MappedArray(np.ma.zeros(len(slave_aligned)).astype(_dtype), values_mapping=mappings)
        mapped_array[:] = slave_aligned[:]
        slave_aligned = mapped_array

    if original_array.dtype.type is np.string_:
        # return back to string array
        slave_aligned = mapped_array_to_string_array(slave_aligned)

    return slave_aligned


def align_slices(slave, master, slices):
    '''
    :param slave: The node to align the slices to.
    :type slave: Node
    :param master: The node which the slices are currently aligned to.
    :type master: Node
    :param slices: Slices to align or None values to skip.
    :type slices: [slice or None]
    :returns: Slices aligned to slave.
    :rtype: [slice or None]
    '''
    if slave.frequency == master.frequency and slave.offset == master.offset:
        return slices
    multiplier = slave.frequency / master.frequency
    offset = (master.offset - slave.offset) * slave.frequency
    aligned_slices = []
    for s in slices:
        if s is None:
            aligned_slices.append(s)
            continue
        aligned_slices.append(slice(
            int(ceil((s.start * multiplier) + offset)) if s.start else None,
            int(ceil((s.stop * multiplier) + offset)) if s.stop else None,
            s.step))
    return aligned_slices


def align_slice(slave, master, _slice):
    '''
    :param slave: The node to align the slice to.
    :type slave: Node
    :param master: The node which the slice is currently aligned to.
    :type master: Node
    :param _slice: Slice to align.
    :type _slice: slice or None
    :returns: Slice aligned to slave.
    :rtype: slice or None
    '''
    return align_slices(slave, master, [_slice])[0]


def string_array_to_mapped_array(array):
    compressed = array.compressed()
    values, counts = np.unique(compressed, return_counts=True)
    mappings = {k: v for k, v in zip(range(len(values)+1), [''] + list(values))}
    mapped_array = MappedArray(np.ma.zeros(len(array)).astype('int'), values_mapping=mappings)
    mapped_array[:] = array[:]

    return mapped_array


def mapped_array_to_string_array(array):
    max_string_length = max(len(x) for x in array.values_mapping.values())
    string_array = np.ma.empty(len(array), dtype='S%s'%max_string_length)
    string_array.fill('')
    for state in array.values_mapping.values():
        string_array[array==state] = state
    string_array.mask = array.mask
    return string_array


def ambiguous_runway(rwy):
    # There are a number of runway related KPVs that we only create if we
    # know the actual runway we landed on. Where there is ambiguity the
    # runway attribute may be truncated, or the identifier, if present, will
    # end in a "*" character.
    return (rwy is None or rwy.value is None or not 'identifier' in rwy.value or
            rwy.value['identifier'].endswith('*'))


def bearing_and_distance(lat1, lon1, lat2, lon2):
    """
    Simplified version of bearings and distances for a single pair of
    locations. Gives bearing and distance of point 2 from point 1.

    :param lat1, lon1: Latitude/Longitude of starting point in degrees
    :param lat2, lon2: Latitude/Longitude of ending point in degrees

    :returns bearing, distance: Bearing in degrees, Distance in metres.
    :rtype: Two Numpy masked arrays
    """
    if any(np.ma.is_masked(x) for x in (lat1, lon1, lat2, lon2)):
        return None, None
    brg, dist = bearings_and_distances(np.ma.array(lat2), np.ma.array(lon2),
                                       {'latitude':lat1, 'longitude':lon1})
    return brg.item(), dist.item()


def bearings_and_distances(latitudes, longitudes, reference):
    """
    Returns the bearings and distances of a track with respect to a fixed point.

    Usage:
    brg[], dist[] = bearings_and_distances(lat[], lon[], {'latitude':lat_ref, 'longitude':lon_ref})

    :param latitudes: The latitudes of the track.
    :type latitudes: Numpy masked array.
    :param longitudes: The latitudes of the track.
    :type longitudes: Numpy masked array.
    :param reference: The location of the second point.
    :type reference: dict with {'latitude': lat, 'longitude': lon} in degrees.

    :returns bearings, distances: Bearings in degrees, Distances in metres.
    :rtype: Two Numpy masked arrays

    Navigation formulae have been derived from the scripts at
    http://www.movable-type.co.uk/scripts/latlong.html
    Copyright 2002-2011 Chris Veness, and altered by Flight Data Services to
    suit the POLARIS project.
    """

    lat_array = latitudes * deg2rad
    lon_array = longitudes * deg2rad
    lat_ref = radians(reference['latitude'])
    lon_ref = radians(reference['longitude'])

    dlat = lat_array - lat_ref
    dlon = lon_array - lon_ref

    a = np.ma.sin(dlat/2)**2 + \
        np.ma.cos(lat_array) * np.ma.cos(lat_ref) * np.ma.sin(dlon/2)**2
    dists = 2 * np.ma.arctan2(np.ma.sqrt(a), np.ma.sqrt(1.0 - a))
    dists *= 6371000 # Earth radius in metres


    y = np.ma.sin(dlon) * np.ma.cos(lat_array)
    x = np.ma.cos(lat_ref) * np.ma.sin(lat_array) \
        - np.ma.sin(lat_ref) * np.ma.cos(lat_array) * np.ma.cos(dlon)
    brgs = np.ma.arctan2(y,x)

    joined_mask = np.logical_or(latitudes.mask, longitudes.mask)
    brg_array = np.ma.array(data=np.rad2deg(brgs) % 360,
                            mask=joined_mask)
    dist_array = np.ma.array(data=dists,
                             mask=joined_mask)

    return brg_array, dist_array

"""
Landing stopping distances.

def braking_action(gspd, landing, mu):
    scale = ut.multiplier(ut.KT, ut.METER_S)
    dist = integrate(gspd.array[landing.slice], gspd.hz, scale=scale)
    #decelerate = np.power(gspd.array[landing.slice]*scale,2.0)\
        #/(2.0*GRAVITY_METRIC*mu)
    mu = np.power(gspd.array[landing.slice]*scale,2.0)\
        /(2.0*GRAVITY_METRIC*dist)
    limit_point = np.ma.argmax(mu)
    ##limit_point = np.ma.argmax(dist+decelerate)
    ##braking_distance = dist[limit_point] + decelerate[limit_point]
    return limit_point, mu[limit_point]
"""

def bump(acc, index, start=None, end=None):
    """
    This scans an acceleration array for a short period either side of the
    moment of interest. Too wide and we risk monitoring flares and
    post-liftoff motion. Too short and we may miss a local peak.

    Optionally one can provide a start index and/or end index. Defaults
    is to monitor for 3 seconds before and after `index`.

    :param acc: An acceleration parameter
    :type acc: A Parameter object
    :param index: A Key Time Instance or simple Index
    :type index: int or float
    :param start: A Key Time Instance or simple Index
    :type index: int or float
    :param end: A Key Time Instance or simple Index
    :type index: int or float

    :returns: The peak acceleration within +/- 3 seconds of the KTI
    :type: Acceleration, from the acc.array.
    """
    dt = BUMP_HALF_WIDTH # Half width of range to scan across for peak acceleration.
    from_index = int(start if start is not None else max(ceil(index - dt * acc.hz), 0))
    to_index = int(end if end is not None else min(int(index + dt * acc.hz)+1, len(acc.array)))
    bump_accel = acc.array[from_index:to_index]

    # Taking the absoulte value makes no difference for normal acceleration
    # tests, but seeks the peak left or right for lateral tests.
    bump_index = np.ma.argmax(np.ma.abs(bump_accel))

    peak = bump_accel[bump_index]
    return from_index + bump_index, peak


def calculate_slat(mode, slat_angle, model, series, family):
    '''
    Retrieves flap map and calls calculate_surface_angle with detents.

    :type mode: str
    :type slat_angle: np.ma.array
    :type model: Attribute
    :type series: Attribute
    :type family: Attribute
    :rtype: dict, np.ma.array, int or float
    '''
    values_mapping = at.get_slat_map(model.value, series.value, family.value)
    array, frequency, offset = calculate_surface_angle(mode, slat_angle,
                                                       list(values_mapping))
    return values_mapping, array, frequency, offset


def calculate_flap(mode, flap_angle, model, series, family):
    '''
    Retrieves flap map and calls calculate_surface_angle with detents.

    :type mode: str
    :type flap_angle: POLARIS parameter
    :type model: Attribute
    :type series: Attribute
    :type family: Attribute
    :returns: values mapping, array and frequency
    :rtype: dict, np.ma.array, int or float
    '''
    values_mapping = at.get_flap_map(model.value, series.value, family.value)
    array, frequency, offset = calculate_surface_angle(mode, flap_angle,
                                                       list(values_mapping))
    '''
    import matplotlib.pyplot as plt
    plt.plot(flap_angle.array)
    plt.plot(array)
    plt.show()
    '''
    return values_mapping, array, frequency, offset


def calculate_surface_angle(mode, param, detents):
    '''
    Steps a surface angle into detents using one of the following modes:
     - 'including' - Including transition is equivalent to 'move_start' on rising values and 'move_stop' on falling values.
     - 'excluding' - Excluding transition is equivalent to 'move_stop' on rising values and 'move_start' on falling values.
     - 'lever' - Lever is equivalent to 'move_start' on both rising and falling values.

    :param mode: 'including', 'excluding' or 'lever'.
    :type mode: str
    :type param: Parameter
    :type detents: [int or float]
    :returns: array and frequency
    :rtype: np.ma.array, int or float
    '''
    freq_multiplier = 4 if param.frequency < 2 else 2
    freq = param.frequency * freq_multiplier
    # No need to re-align if high frequency.
    offset = param.offset / freq_multiplier
    angle = align_args(
        param.array,
        param.frequency,
        param.offset,
        freq,
        offset,
    )

    mask = angle.mask.copy()
    # Repair the array to avoid extreme values affecting the algorithm.
    # Extrapolation included to avoid problems of corrupt data at end of flight and/or testing.
    angle = repair_mask(angle, repair_duration=None, extrapolate=True)
    thresh_main_metrics = 4.0
    thresh_angle_range = 0.4
    filter_median_window = 5

    result = np_ma_zeros_like(angle, mask=angle.mask)

    # ---- Pre-processing ----------------------------------------------
    angle = np.ma.masked_array(medfilt(angle, filter_median_window))

    metrics = np.full(len(angle), np.Inf)
    for l in np.array([2, 3, 4, 5, 6, 8, 12, 16]):
        maxy = filters.maximum_filter1d(angle, l)
        miny = filters.minimum_filter1d(angle, l)
        m = np.log(maxy - miny +  1) / np.log(l)
        metrics = np.minimum(metrics, m)

    metrics = medfilt(metrics,3)
    metrics *= 200.0

    c1 = metrics < angle
    c2 = metrics < thresh_main_metrics
    angle.mask = (c1 | c2)

    # ---- First pass: flat transitions are not transitions ------------
    flat_slices = np.ma.clump_masked(angle)
    tran_slices = np.ma.clump_unmasked(angle)

    for s in tran_slices:
        if np.abs(np.max(angle[s]) - np.min(angle[s])) < thresh_angle_range:
            angle.mask[s] = True

    # ---- Second pass: re-assign flap ----------------------------------
    flat_slices = np.ma.clump_masked(angle)
    tran_slices = np.ma.clump_unmasked(angle)

    for s in flat_slices:
        avg = np.mean(angle.data[s])
        index = np.argmin(np.abs(avg - detents))
        result[s] = detents[index]

    for s in tran_slices:

        lo = angle[s][0]
        hi = angle[s][-1]

        detents_lo = detents[np.argmin(abs(lo - detents))]
        detents_hi = detents[np.argmin(abs(hi - detents))]

        if mode == 'lever':
            result[s] = detents_hi
            continue

        if detents_hi > detents_lo:
            result[s] = detents_lo if mode == 'excluding' else detents_hi
        elif detents_hi < detents_lo:
            result[s] = detents_hi if mode == 'excluding' else detents_lo
        else:
            # If equal fill with either hi or lo.
            result[s] = detents_lo

    # ---- Set-up the output --------------------------------------------
    # Re-apply the original mask.
    result.mask = mask
    return result, freq, offset


def calculate_timebase(years, months, days, hours, mins, secs):
    """
    Calculates the timestamp most common in the array of timestamps. Returns
    timestamp calculated for start of array by applying the offset of the
    most common timestamp.

    Accepts arrays and numpy arrays at 1Hz.

    WARNING: If at all times, one or more of the parameters are masked, you
    willnot get a valid timestamp and an exception will be raised.

    Note: if uneven arrays are passed in, they are assumed by zip that the
    start is valid and the uneven ends are invalid and skipped over.

    Supports years as a 2 digits - e.g. "11" is "2011"

    :param years, months, days, hours, mins, secs: Appropriate 1Hz time elements
    :type years, months, days, hours, mins, secs: iterable of numeric type
    :returns: best calculated datetime at start of array
    :rtype: datetime
    :raises: InvalidDatetime if no valid timestamps provided
    """
    base_dt = None
    # Calculate current year here and pass into
    # convert_two_digit_to_four_digit_year to save calculating year for every
    # second of flight
    current_year = str(datetime.utcnow().year)
    # OrderedDict so if all values are the same, max will consistently take the
    # first val on repeated runs
    clock_variation = OrderedDict()

    if not len(years) == len(months) == len(days) == \
       len(hours) == len(mins) == len(secs):
        raise ValueError("Arrays must be of same length")

    for step, (yr, mth, day, hr, mn, sc) in enumerate(np.transpose((years, months, days, hours, mins, secs))):

        #TODO: Try using numpy datetime functions for speedup?
        #try:
            #date = np.datetime64('%d-%d-%d' % (yr, mth, day), 'D')
        #except np.core._mx_datetime_parser.RangeError  :
            #continue
        # same for time?

        if yr is not None and yr is not np.ma.masked and yr < 100:
            yr = convert_two_digit_to_four_digit_year(yr, current_year)

        try:
            dt = datetime(int(yr), int(mth), int(day), int(hr), int(mn), int(sc), tzinfo=pytz.utc)
        except (ValueError, TypeError, np.ma.core.MaskError):
            # ValueError is raised if values are out of range, e.g. 0..59.
            # Q: Should we validate these parameters and switch to fallback_dt
            #    if it fails?
            continue
        if not base_dt:
            base_dt = dt # store reference datetime
        # calc diff from base
        diff = dt - base_dt - timedelta(seconds=step)
        ##print("%02d - %s %s" % (step, dt, diff))
        try:
            clock_variation[diff] += 1
        except KeyError:
            # new difference
            clock_variation[diff] = 1
    if base_dt:
        # return most regular difference
        clock_delta = max(clock_variation, key=clock_variation.get)
        return base_dt + clock_delta
    else:
        # No valid datestamps found
        raise InvalidDatetime("No valid datestamps found")


def convert_two_digit_to_four_digit_year(yr, current_year):
    """
    Everything below the current year is assume to be in the current
    century, everything above is assumed to be in the previous
    century.
    if current year is 2012

    13 = 1913
    12 = 2012
    11 = 2011
    01 = 2001
    """
    # convert to 4 digit year
    century = int(current_year[:2]) * 100
    yy = int(current_year[2:])
    if yr > yy:
        return century - 100 + yr
    else:
        return century + yr


def coreg(y, indep_var=None, force_zero=False):
    """
    Combined correlation and regression line calculation.

    correlate, slope, offset = coreg(y, indep_var=x, force_zero=True)

    :param y: dependent variable
    :type y: numpy float array - NB: MUST be float
    :param indep_var: independent variable
    :type indep_var: numpy float array. Where not supplied, a linear scale is created.
    :param force_zero: switch to force the regression offset to zero
    :type force_zero: logic, default=False

    :returns:
    :param correlate: The modulus of Pearson's correlation coefficient

    Note that we use only the modulus of the correlation coefficient, so that
    we only have to test for positive values when checking the strength of
    correlation. Thereafter the slope is used to identify the sign of the
    correlation.

    :type correlate: float, in range 0 to +1,
    :param slope: The slope (m) in the equation y=mx+c for the regression line
    :type slope: float
    :param offset: The offset (c) in the equation y=mx+c
    :type offset: float

    Example usage:

    corr,m,c = coreg(air_temp.array, indep_var=alt_std.array)

    corr > 0.5 shows weak correlation between temperature and altitude
    corr > 0.8 shows good correlation between temperature and altitude
    m is the lapse rate
    c is the temperature at 0ft
    """
    n = len(y)
    if n < 2:
        logger.warning('Function coreg called with data of length %s' % n)
    if indep_var is None:
        x = np.ma.arange(n, dtype=np.float64)
    else:
        x = indep_var
        if len(x) != n:
            raise ValueError('Function coreg called with arrays of differing '
                             'length')

    # Need to propagate masks into both arrays equally.
    mask = np.ma.logical_or(x.mask, y.mask)
    x = np.ma.array(data=x.data, mask=mask, dtype=np.float64)
    y = np.ma.array(data=y.data, mask=mask, dtype=np.float64)

    if x.ptp() == 0.0 or y.ptp() == 0.0:
        # raise ValueError('Function coreg called with invariant independent variable')
        return None, None, None

    # n is the number of useful data pairs for analysis.
    n = np.float64(np.ma.count(x))
    sx = np.ma.sum(x)
    sxy = np.ma.sum(x*y)
    sy = np.ma.sum(y)
    sx2 = np.ma.sum(x*x)
    sy2 = np.ma.sum(y*y)

    # Correlation
    try: # in case sqrt of a negative number is attempted
        p = np.abs((n*sxy - sx*sy)/(np.sqrt(n*sx2-sx*sx)*np.sqrt(n*sy2-sy*sy)))
    except ValueError:
        return None, None, None

    if np.isinf(p) or np.isnan(p):
        # known to be caused by negative sqrt in np.sqrt(n*sx2-sx*sx)
        return None, None, None

    # Regression
    if force_zero:
        m = sxy/sx2
        c = 0.0
    else:
        m = (sxy-sx*sy/n)/(sx2-sx*sx/n)
        c = sy/n - m*sx/n

    return p, m, c


def create_phase_inside(array, hz, offset, phase_start, phase_end):
    '''
    This function masks all values of the reference array outside of the phase
    range phase_start to phase_end, leaving the valid phase inside these times.

    :param array: input data
    :type array: masked array
    :param a: sample rate for the input data (sec-1)
    :type hz: float
    :param offset: fdr offset for the array (sec)
    :type offset: float
    :param phase_start: time into the array where we want to start seeking the threshold transit.
    :type phase_start: float
    :param phase_end: time into the array where we want to stop seeking the threshold transit.
    :type phase_end: float
    :returns: input array with samples outside phase_start and phase_end masked.
    '''
    return _create_phase_mask(array,  hz, offset, phase_start, phase_end, 'inside')


def create_phase_outside(array, hz, offset, phase_start, phase_end):
    '''
    This function masks all values of the reference array inside of the phase
    range phase_start to phase_end, leaving the valid phase outside these times.

    :param array: input data
    :type array: masked array
    :param a: sample rate for the input data (sec-1)
    :type hz: float
    :param offset: fdr offset for the array (sec)
    :type offset: float
    :param phase_start: time into the array where we want to start seeking the threshold transit.
    :type phase_start: float
    :param phase_end: time into the array where we want to stop seeking the threshold transit.
    :type phase_end: float
    :returns: input array with samples outside phase_start and phase_end masked.
    '''
    return _create_phase_mask(array, hz, offset, phase_start, phase_end, 'outside')


def _create_phase_mask(array, hz, offset, a, b, which_side):
    # Create Numpy array of same size as array data
    length = len(array)
    m = np.arange(length)

    if a > b:
        a, b = b, a # Swap them over to make sure a is the smaller.

    # Convert times a,b to indices ia, ib and check they are within the array.
    ia = int((a - offset) * hz)
    if ia < (a - offset) * hz:
        ia += 1
    if ia < 0 or ia > length:
        raise ValueError('Phase mask index out of range')

    ib = int((b - offset) * hz) + 1
    if ib < 0 or ib > length:
        raise ValueError('Phase mask index out of range')

    # Populate the arrays to be False where the flight phase is valid.
    # Adjustments ensure phase is intact and not overwritten by True data.
    if which_side == 'inside':
        m[:ia]  = True
        m[ia:ib] = False
        m[ib:]  = True
    else:
        m[:ia]  = False
        m[ia:ib] = True
        m[ib:]  = False

    # Return the masked array containing reference data and the created mask.
    return np.ma.MaskedArray(array, mask=m)


def cycle_counter(array, min_step, max_time, hz, offset=0):
    '''
    Counts the number of consecutive cycles.

    Each cycle must have a period of not more than ``cycle_time`` seconds, and
    have a variation greater than ``min_step``.

    Note: Where two events with the same cycle count arise in the same array,
    the latter is recorded as it is normally the later in the flight that will
    be most hazardous.

    :param array: Array of data to count cycles within.
    :type array: numpy.ma.core.MaskedArray
    :param min_step: Minimum step, below which fluctuations will be removed.
    :type min_step: float
    :param max_time: Maximum time for a complete valid cycle in seconds.
    :type max_time: float
    :param hz: The sample rate of the array.
    :type hz: float
    :param offset: Index offset to start of the provided array.
    :type offset: int
    :returns: A tuple containing the index of the array element at the end of
        the highest number of cycles and the highest number of cycles in the
        array. Note that the value can be a float as we count a half cycle for
        each change over the minimum step.
    :rtype: (int, float)
    '''
    idxs, vals = cycle_finder(array, min_step=min_step)

    if idxs is None:
        return Value(None, None)

    index, half_cycles = None, 0
    max_index, max_half_cycles = None, 0

    # Determine the half cycle times and look for the most cycling:
    half_cycle_times = np.ediff1d(idxs) / hz
    for n, half_cycle_time in enumerate(half_cycle_times):
        # If we are within the max time, keep track of the half cycle:
        if half_cycle_time < max_time:
            half_cycles += 1
            index = idxs[n + 1]
        # Otherwise check if this is the most cycling and reset:
        elif 0 < half_cycles >= max_half_cycles:
            max_index, max_half_cycles = index, half_cycles
            half_cycles = 0
    else:
        # Finally check whether the last loop had most cycling:
        if 0 < half_cycles >= max_half_cycles:
            max_index, max_half_cycles = index, half_cycles

    # Ignore single direction movements (we only want full cycles):
    if max_half_cycles < 2:
        return Value(None, None)

    return Value(offset + max_index, max_half_cycles / 2.0)


def cycle_select(array, min_step, max_time, hz, offset=0):
    '''
    Selects the value difference in the array when cycling.

    Each cycle must have a period of not more than ``cycle_time`` seconds, and
    have a variation greater than ``min_step``.  The selected value is the
    largest peak-to-peak value of a returning cycle.

    :param array: Array of data to count cycles within.
    :type array: numpy.ma.core.MaskedArray
    :param min_step: Minimum step, below which fluctuations will be removed.
    :type min_step: float
    :param cycle_time: Maximum time for a complete valid cycle in seconds.
    :type cycle_time: float
    :param hz: The sample rate of the array.
    :type hz: float
    :param offset: Index offset to start of the provided array.
    :type offset: int
    :returns: A tuple containing the index of the array element at the peak of
        the highest difference and the highest difference between a peak and the
        troughs either side.
    :rtype: (int, float)
    '''
    idxs, vals = cycle_finder(array, min_step=min_step)

    if idxs is None:
        return Value(None, None)

    max_index, max_value = None, 0

    # Determine the half cycle times and ptp values for the half cycles:
    half_cycle_times = np.ediff1d(idxs) / hz
    half_cycle_diffs = abs(np.ediff1d(vals))
    if len(half_cycle_diffs)<2:
        return Value(None, None)
    full_cycle_pairs = zip(half_cycle_times[1:]+half_cycle_times[:-1],
                           [min(half_cycle_diffs[n],half_cycle_diffs[n+1]) \
                            for n in range(len(half_cycle_diffs)-1)])
    for n, (cycle_time, value) in enumerate(full_cycle_pairs):
        # If we are within the max time and have max difference, keep it:
        if cycle_time < max_time and value >= max_value:
            max_index, max_value = idxs[n + 1], value

    if max_index is None:
        return Value(None, None)

    return Value(offset + max_index, max_value)


def cycle_finder(array, min_step=0.0, include_ends=True):
    '''
    Simple implementation of a peak detection algorithm with small cycle
    remover.

    :param array: time series data
    :type array: Numpy masked array
    :param min_step: Optional minimum step, below which fluctuations will be removed.
    :type min_step: float
    :param include_ends: Decides whether the first and last points of the array are to be included as possible turning points
    :type include_ends: logical

    :returns: A tuple containing the list of peak indexes, and the list of peak values.
    '''

    if len(array) == 0:
        # Nothing to do, so return None.
        return None, None

    # Find the peaks and troughs by difference products which change sign.
    x = np.ma.ediff1d(array, to_begin=0.0)
    # Stripping out only the nonzero values ensures we don't get confused with
    # invariant data.
    y = np.ma.nonzero(x)[0]
    z = x[y] # np.ma.nonzero returns a tuple of indices
    peak = -z[:-1] * z[1:] # Here we compute the change in direction.
    # And these are the indeces where the direction changed.
    idxs = y[np.nonzero(np.ma.maximum(peak, 0.0))]
    vals = array.data[idxs] # So these are the local peak and trough values.

    # Optional inclusion of end points.
    if include_ends and np.ma.count(array):
        # We can only extend over the range of valid data, so find the first
        # and last valid samples.
        first, last = np.ma.flatnotmasked_edges(array)
        idxs = np.insert(idxs, 0, first)
        vals = np.insert(vals, 0, array.data[first])
        # If the end two are in line, scrub the middle one.
        try:
            if (vals[2] - vals[1]) * (vals[1] - vals[0]) >= 0.0:
                idxs = np.delete(idxs, 1)
                vals = np.delete(vals, 1)
        except:
            # If there are few vals in the array, there's nothing to tidy up.
            pass
        idxs = np.append(idxs, last)
        vals = np.append(vals, array.data[last])
        try:
            if (vals[-3] - vals[-2]) * (vals[-2] - vals[-1]) >= 0.0:
                idxs = np.delete(idxs, -2)
                vals = np.delete(vals, -2)
        except:
            pass # as before.

    # This section progressively removes reversals smaller than the step size of
    # interest, hence the arrays shrink until just the desired answer is left.
    dvals = np.ediff1d(vals)
    while len(dvals) > 0 and np.min(abs(dvals)) < min_step:
        sort_idx = np.argmin(abs(dvals))
        last = len(dvals)
        if sort_idx == 0:
            idxs = np.delete(idxs, 0)
            vals = np.delete(vals, 0)
            dvals = np.delete(dvals, 0)
        elif sort_idx == last-1:
            idxs = np.delete(idxs, last)
            vals = np.delete(vals, last)
            dvals = np.delete(dvals, last-1) # One fewer dval than val.
        else:
            idxs = np.delete(idxs, slice(sort_idx, sort_idx + 2))
            vals = np.delete(vals, slice(sort_idx, sort_idx + 2))
            dvals[sort_idx - 1] += dvals[sort_idx] + dvals[sort_idx + 1]
            dvals = np.delete(dvals, slice(sort_idx, sort_idx + 2))
    if len(dvals) == 0:
        # All the changes have disappeared, so return the
        # single array peak index and value.
        return idxs, vals
    else:
        return idxs, vals


def cycle_match(idx, cycle_idxs, dist=None):
    '''
    Finds the previous and next cycle indexes either side of idx plus
    an allowable distance. For use after "cycle_finder".

    cycle_idxs are generally a "down up down up down" affair where you are
    searching for the indexes either side of an "up" for instance. If no
    index is available before or after a match, a None is returned in its
    place. If no matching index found within cycle_idxs, ValueError is raised.

    :param idx: Index to find cycle around
    :type idx: Float (may have been identified following interpolation)
    :param cycle_idxs: Indexes from cycle finder
    :type cycle_idxs: List of indexes (normally Numpy array of floats)
    :param dist: If None, uses a quarter of the minimum dist between cycles
    :type dist: Float or None
    :returns: previous and next cycle indexes
    :rtype: float, float
    '''
    if dist is None:
        dist = np.min(np.diff(cycle_idxs)) / 4.0

    min_idx = np.argmin(np.abs(np.array(cycle_idxs) - idx))
    if abs(cycle_idxs[min_idx] - idx) < dist:
        prev = cycle_idxs[min_idx-1] if min_idx > 0 else None
        post = cycle_idxs[min_idx+1] if min_idx < len(cycle_idxs)-1 else None
        return prev, post
    raise ValueError("Did not find a match for index '%d' within cycles %s" % (
        idx, cycle_idxs))


def datetime_of_index(start_datetime, index, frequency=1):
    '''
    Returns the datetime of an index within the flight at a particular
    frequency.

    :param start_datetime: Start datetime of the flight available as the 'Start Datetime' attribute.
    :type start_datetime: datetime
    :param index: Index within the flight.
    :type index: int
    :param frequency: Frequency of the index.
    :type frequency: int or float
    :returns: Datetime at index.
    :rtype: datetime
    '''
    index_in_seconds = index / frequency
    offset = timedelta(seconds=index_in_seconds)
    return (start_datetime or 0) + offset


def delay(array, period, hz=1.0):
    '''
    This function introduces a time delay. Used in validation testing where
    correlation is improved by allowing for the delayed response of one
    parameter when compared to another.

    :param array: Masked array of floats
    :type array: Numpy masked array
    :param period: Time delay(sec)
    :type period: int/float
    :param hz: Frequency of the data_array
    :type hz: float

    :returns: array with data shifted back in time, and initial values masked.
    '''
    n = int(period * hz)
    result = np_ma_masked_zeros_like(array)
    if len(result[n:])==len(array[:-n]):
        result[n:] = array[:-n]
        return result
    else:
        if n==0:
            return array
        else:
            return result


def positive_index(container, index):
    '''
    Always return a positive index but without going to the other
    end of the array for indexes greater than -1 which arise from alignment.
    :param container: Container with a length.
    :param index: Positive or negative index within the array.
    :type index: int or float
    :returns: Positive index.
    :rtype: int or float
    '''
    if index is None:
        return index

    if index <= -1:
        index += len(container)
    elif -1 < index < 0:
        index = 0
    elif index >= len(container):
        index = len(container) - 1

    return index


def power_ceil(x):
    '''
    :returns: The closest power of 2 greater than or equal to x.
    '''
    return 2**(ceil(log(x, 2)))


def power_floor(x):
    '''
    :returns: The closest power of 2 less than or equal to x.
    '''
    #Q: not sure why casting to an int is necessary
    return int(2**(floor(log(x, 2))))


def next_unmasked_value(array, index, stop_index=None):
    '''
    Find the next unmasked value from index in the array.

    :type array: np.ma.masked_array
    :param index: Start index of search.
    :type index: int
    :param stop_index: Optional stop index of search, otherwise searches to the end of the array.
    :type stop_index: int or None
    :returns: Next unmasked value or None.
    :rtype: Value or None
    '''
    array.mask = np.ma.getmaskarray(array)

    # Normalise indices.
    index = positive_index(array, index)
    stop_index = positive_index(array, stop_index)

    try:
        unmasked_index = np.where(np.invert(array.mask[slices_int(index, stop_index)]))[0][0]
    except IndexError:
        return None
    else:
        unmasked_index += index
        return Value(index=unmasked_index, value=array[unmasked_index])


def prev_unmasked_value(array, index, start_index=None):
    '''
    Find the previous unmasked value from index in the array.

    :type array: np.ma.masked_array
    :param index: Stop index of search (searches backwards).
    :type index: int
    :param start_index: Optional start index of search, otherwise searches to the beginning of the array.
    :type start_index: int or None
    :returns: Previous unmasked value or None.
    :rtype: Value or None
    '''
    array.mask = np.ma.getmaskarray(array)

    # Normalise indices.
    index = positive_index(array, index)
    start_index = positive_index(array, start_index)

    try:
        unmasked_index = np.where(np.invert(array.mask[slices_int(start_index,index + 1)]))[0][-1]
    except IndexError:
        return None
    else:
        if start_index:
            unmasked_index += start_index
        return Value(index=unmasked_index, value=array[unmasked_index])


def closest_unmasked_value(array, index, start_index=None, stop_index=None):
    '''
    Find the closest unmasked value from index in the array, optionally
    specifying a search range with start_index and stop_index.

    :type array: np.ma.masked_array
    :param index: Start index of search.
    :type index: int
    :param start_index: Optional start index of search.
    :type start_index: int
    '''
    array.mask = np.ma.getmaskarray(array)

    # Normalise indices.
    index = int(floor(positive_index(array, index)))

    if not array.mask[index]:
        return Value(index, array[index])

    if start_index and stop_index:
        # Fix invalid index ranges, I assume slice.step was -1.
        start_index = min(start_index, stop_index)
        stop_index = max(start_index, stop_index)

    # Normalise indices.
    start_index = positive_index(array, start_index)
    stop_index = positive_index(array, stop_index)

    prev_value = prev_unmasked_value(array, index, start_index=start_index)
    next_value = next_unmasked_value(array, index, stop_index=stop_index)
    if prev_value and next_value:
        if abs(prev_value.index - index) < abs(next_value.index - index):
            return prev_value
        else:
            return next_value
    elif prev_value:
        return prev_value
    elif next_value:
        return next_value
    else:
        return None


def clump_multistate(array, state, _slices=[slice(None)], condition=True):
    '''
    This tests a multistate array and returns a classic POLARIS list of slices.

    Masked values will not be included in the slices. If this troubles you,
    repair the masked data (maintaining the previous value or taking the
    nearest neighbour value) using nearest_neighbour_mask_repair before
    passing the array into this function.

    :param array: data to scan
    :type array: multistate numpy masked array
    :param state: state to be tested
    :type state: string
    :param _slices: slice or list of slices over which to scan the array.
    :type _slices: slice list
    :param condition: selection of true or false (i.e. inverse) test to apply.
    :type condition: boolean

    :returns: list of slices.
    '''
    if not _slices:
        return []

    if not state in array.state:
        return None

    if isinstance(_slices, slice):
        # single slice provided
        _slices = [_slices]

    if condition:
        state_match = runs_of_ones(array == state)
    else:
        state_match = runs_of_ones(array != state)

    return slices_and(_slices, state_match)


def unique_values(array):
    '''
    Count the number of unique valid values found within an array.

    Hint: If you get "TypeError: array cannot be safely cast to required type"
    and your data is integer type, try casting it to int type:

    unique_values(flap.array.data.astype(int))

    :param array: Array to count occurrences of values within
    :type array: np.array
    :returns: [(val, count), (val2, count2)]
    :rtype: List of tuples
    '''
    if not np.ma.count(array):
        return {}
    counts = np.bincount(array.compressed())
    vals = np.nonzero(counts)[0]
    if hasattr(array, 'values_mapping'):
        keys = [array.values_mapping[x] for x in vals]
    else:
        keys = vals
    return dict(zip(keys, counts[vals]))


def modulo(value1, value2):
    '''
    Accurate modulo operation avoiding IEEE 754 floating point errors.

    >>> 6 % 0.2
    0.19999999999999968
    >>> Decimal( 6) % Decimal(0.2) # IEEE 754 error already present
    Decimal('0.1999999999999996780353228587')
    >>> Decimal('6') % Decimal('0.2')
    Decimal('0.0')

    :type value1: float or int
    :type value2: float or int
    :rtype: float
    '''
    return float(Decimal(str(value1)) % Decimal(str(value2)))


def most_common_value(array, threshold=None):
    '''
    Find the most repeating non-negative valid value within an array. Works
    with mapped arrays too as well as arrays of strings.

    :param array: Array to count occurrences of values within
    :type array: np.ma.array
    :param threshold: return None if number of most common value occurrences is
        less than this
    :type threshold: float
    :returns: most common value in array
    '''
    compressed = array.compressed()
    values, counts = np.unique(compressed, return_counts=True)
    if not counts.size:
        return None

    i = np.argmax(counts)
    value, count = values[i], counts[i]

    if threshold is not None and count < compressed.size * threshold:
        return None

    if hasattr(array, 'values_mapping'):
        return array.values_mapping[value]
    else:
        return value


def compress_iter_repr(iterable, cast=None, join='+'):
    '''
    Groups list or tuple iterables and finds repeating values. Useful for
    building compressed lists of repeating values.

    Uses the objects repr if possible, otherwise it's cast to %s (__str__)

    'cast' keyword argument can force casting to another type, e.g. int

    >>> print(compress_iter_repr([0,0,1,0,2,2,2]))
    [0]*2 + [1] + [0] + [2]*3
    >>> print(compress_iter_repr(['a', 'a']))
    ['a']*2

    :param iterable: iterable to compress
    :type iterable: list or tuple
    :param cast: function to apply to value before calling repr, e.g. str, int
    :type cast: function or None
    :param join: to adjust the space between + join symbols e.g. join=' + '
    :type join: str
    '''
    prev_v = None
    res = []
    count = 0
    for v in iterable:
        if cast:
            v = cast(v)
        if v != prev_v or \
           not np.ma.is_masked(v) and np.ma.is_masked(prev_v) or \
           np.ma.is_masked(v) and not np.ma.is_masked(prev_v):
            if np.ma.is_masked(prev_v):
                res.append((prev_v, count))
            elif prev_v != None:
                # FIXME: NameError for count
                res.append((prev_v, count))
            count = 1
        else:# v == prev_v:
            count += 1
        prev_v = v
    else:
        # end of loop, add last item to res
        if prev_v != None:
            res.append((prev_v, count))
    entries = []
    for val, cnt in res:
        v = val.__repr__() if hasattr(val, '__repr__') else val
        c = '*%d' % cnt if cnt > 1 else ''
        entries.append('[%s]%s' % (v, c))
    return join.join(entries)


def filter_vor_ils_frequencies(array, navaid):
    '''
    This function passes valid ils or vor frequency data and masks all other data.

    To quote from Flightgear ~(where the clearest explanation can be found)
    "The VOR uses frequencies in the the Very High Frequency (VHF) range, it
    uses channels between 108.0 MHz and 117.95 MHz. It is spaced with 0.05
    MHz intervals (so 115.00; 115.05; 115.10 etc). The range 108...112 is
    shared with ILS frequencies. To differentiate between them VOR has an
    even number on the 0.1 MHz frequency and the ILS has an uneven number on
    the 0,1 MHz frequency.

    So 108.0; 108.05; 108.20; 108.25; 108.40; 108.45 would be VOR stations.
    and 108.10; 108.15; 108.30; 108.35; 108.50; 108.55 would be ILS stations.

    :param array: Masked array of radio frequencies in MHz
    :type array: Floating point Numpy masked array
    :param navaid: Type of navigation aid
    :type period: string, 'ILS' or 'VOR' only accepted.

    :returns: Numpy masked array. The requested navaid type frequencies will be passed as valid. All other frequencies will be masked.
    '''
    vor_range = np.ma.masked_outside(array, 108.0, 117.95)
    ils_range = np.ma.masked_greater(vor_range, 111.95)

    # This finds the four sequential frequencies, so fours has values:
    #   0 = .Even0, 1 = .Even5, 2 = .Odd0, 3 = .Odd5
    # The round function is essential as using floating point values leads to inexact values.
    fours = np.ma.round(array * 20) % 4

    # Remove frequencies outside the operating range.
    if navaid == 'ILS':
        return np.ma.masked_where(fours < 2.0, ils_range)
    elif navaid == 'VOR':
        return np.ma.masked_where(fours > 1.0, vor_range)
    else:
        raise ValueError('Navaid of unrecognised type %s' % navaid)


def find_app_rwy(app_info, this_loc):
    """
    This function scans through the recorded approaches to find which matches
    the current localizer established phase. This is required because we
    cater for multiple ILS approaches on a single flight.
    """
    for approach in app_info:
        # line up an approach slice
        if slices_overlap(this_loc.slice, approach.slice):
            # we've found a matching approach where the localiser was established
            break
    else:
        logger.warning("No approach found within slice '%s'.",this_loc)
        return None, None

    runway = approach.approach_runway
    if not runway:
        logger.warning("Approach runway information not available.")
        return approach, None

    return approach, runway


def index_of_first_start(bool_array, _slice=slice(0, None), min_dur=0.0,
                         frequency=1):
    '''
    Find the first starting index of a state change.

    Using bool_array allows one to select the filter before hand,
    e.g. index_of_first_start(state.array == 'state', this_slice)

    Similar to "find_edges_on_state_change" but allows a minumum
    duration (in samples)

    Note: applies -0.5 offset to interpolate state transition, so use
    value_at_index() for the returned index to ensure correct values
    are returned from arrays.
    '''
    if _slice.step and _slice.step < 0:
        raise ValueError("Reverse step not supported")
    runs = runs_of_ones(bool_array[slices_int(_slice)])
    if min_dur:
        runs = filter_slices_duration(runs, min_dur, frequency=frequency)
    if runs:
        return runs[0].start + (_slice.start or 0) - 0.5  # interpolate offset
    else:
        return None


def index_of_last_stop(bool_array, _slice=slice(0, None), min_dur=1,
                       frequency=1):
    '''
    Find the first stopping index of a state change.

    Using bool_array allows one to select the filter before hand,
    e.g. index_of_first_stop(state.array != 'state', this_slice)

    Similar to "find_edges_on_state_change" but allows a minumum
    duration (in samples)

    Note: applies +0.5 offset to interpolate state transition, so use
    value_at_index() for the returned index to ensure correct values
    are returned from arrays.
    '''
    if _slice.step and _slice.step < 0:
        raise ValueError("Reverse step not supported")
    runs = runs_of_ones(bool_array[_slice])
    if min_dur:
        runs = filter_slices_duration(runs, min_dur, frequency=frequency)
    if runs:
        return runs[-1].stop + (_slice.start or 0) - 0.5
    else:
        return None


def find_low_alts(array, frequency, threshold_alt,
                  start_alt=None, stop_alt=None,
                  level_flights=None, cycle_size=500,
                  stop_mode='climbout', relative_start=False,
                  relative_stop=False):
    '''
    This function allows us to find the minima of cycles which dip below
    threshold_alt, along with initial takeoff and landing.
    The start_alt and stop_alt are extrapolated from
    the minimum index. A stop_alt of 0 will find the lowest point of
    descent. A negative stop_alt will seek backwards to find altitudes
    in the descent, e.g. from 3000-50ft descending.

    :param array: Altitude AAL data array
    :type array: numpy masked array
    :param threshold_alt: Altitude which must have dipped below.
    :type threshold_alt: float or int
    :param start_alt: Start of descent altitude. A value of None will be substituted with threshold_alt.
    :type start_alt: float or int or None
    :param stop_alt: Stop altitude during climbout or descent. A value of None will be substituted with threshold_alt. If stop_mode is climbout, this altitude will be found after the minimum. If stop mode is climbout the altitude is searched for backwards through the descent.
    :type stop_alt: float or int or None
    :param level_flights: Level flight slices to exclude for descent and climbout. The index_at_value_or_level_off algorithm often includes sections of level flight.
    :type level_flights: [slice]
    :param cycle_size: Minimum cycle size to detect as a dip.
    :type cycle_size: int or float
    :param stop_mode: If stop_mode is 'climbout', the stop_alt will be searched for after the minimum altitude (during climbout). If stop_mode is 'descent', the stop_alt will be searched for before the minimum altitude (during descent).
    :param relative_start: Start alt is relative to minimum dip altitude.
    :type relative_start: bool
    :param relative_stop: Stop alt is relative to minimum dip altitude.
    :type relative_stop: bool
    :type stop_mode: int or float
    :returns: List of takeoff, landing and dip slices.
    :rtype: [slices]
    '''
    if relative_start and start_alt is None:
        raise ValueError('relative_start requires start_alt')
    if relative_stop and stop_alt is None:
        raise ValueError('relative_stop requires stop_alt')

    start_alt = start_alt if start_alt is not None else threshold_alt
    stop_alt = stop_alt if stop_alt is not None else threshold_alt

    low_alt_slices = []
    low_alt_cycles = []

    pk_to_pk = np.ma.ptp(array)
    if pk_to_pk > cycle_size:
        min_step = cycle_size
    elif pk_to_pk > 0.0:
        min_step = pk_to_pk * 0.5
    else:
        return []

    pk_idxs, pk_vals = cycle_finder(array, min_step)
    # Convert cycles to slices.
    if not pk_vals[0]:
        if stop_alt != 0:
            # Do not include takeoffs when looking for climbout of 0.
            low_alt_cycles.append(
                (pk_idxs[0], pk_idxs[1], pk_idxs[0], pk_vals[0]))
        pk_idxs = pk_idxs[1:]
        pk_vals = pk_vals[1:]

    if not pk_vals[-1]:
        low_alt_cycles.append(
            (pk_idxs[-2], pk_idxs[-1], pk_idxs[-1], pk_vals[-1]))
        pk_idxs = pk_idxs[:-1]
        pk_vals = pk_vals[:-1]

    for dip_start, dip_min_index, dip_min_value, dip_stop in zip(pk_idxs[::2],
                                                                 pk_idxs[1::2],
                                                                 pk_vals[1::2],
                                                                 pk_idxs[2::2]):
        if dip_min_value >= threshold_alt:
            # Altitude must dip below threshold_alt.
            continue
        low_alt_cycles.append(
            (dip_start, dip_stop, dip_min_index, dip_min_value))

    for start_index, stop_index, dip_min_index, dip_min_value in low_alt_cycles:

        start_value = (start_alt + dip_min_value) if relative_start and start_alt else start_alt
        stop_value = (stop_alt + dip_min_value) if relative_stop and stop_alt else stop_alt

        # Extrapolate climbout.
        if stop_value == 0:
            # Get min index of array in slice.
            stop_index = min_value(array, _slice=slice(start_index,
                                                       stop_index)).index
        elif abs(stop_value) <= dip_min_value:
            # Cycle min value above stop_value.
            stop_index = dip_min_index
        elif stop_mode == 'climbout':
            # Search forwards during climbout.
            pk_idxs, pk_vals = cycle_finder(array[dip_min_index:stop_index],
                                            min_step=cycle_size)

            if pk_idxs is not None and len(pk_idxs) >= 2:
                pk_idxs += dip_min_index
                stop_index = index_at_value_or_level_off(
                    array, frequency,
                    stop_value, slice(pk_idxs[0], pk_idxs[1]))
        elif stop_mode == 'descent':
            # Search backwards during descent.
            pk_idxs, pk_vals = cycle_finder(array[start_index:dip_min_index],
                                            min_step=cycle_size)

            if pk_idxs is not None and len(pk_idxs) >= 2:
                pk_idxs += start_index
                stop_index = index_at_value_or_level_off(
                    array, frequency, abs(stop_value),
                    slice(pk_idxs[-1], pk_idxs[-2], -1))
        else:
            raise NotImplementedError(
                "Unknown stop_mode '%s' within find_low_alts." % stop_mode)

        # Extrapolate descent.
        pk_idxs, pk_vals = cycle_finder(array[start_index:dip_min_index],
                                        min_step=cycle_size)

        if pk_idxs is not None and len(pk_idxs) >= 2:
            pk_idxs += start_index
            start_index = index_at_value_or_level_off(
                array, frequency, start_value,
                slice(pk_idxs[-1], pk_idxs[-2], -1))

        low_alt_slice = slice(start_index, stop_index)

        if level_flights:
            # Exclude level flight.
            excluding_level_flight = slices_and_not([low_alt_slice], level_flights)
            low_alt_slice = find_nearest_slice(
                max(dip_min_index-1, low_alt_slice.start), excluding_level_flight)

        if low_alt_slice:
            low_alt_slices.append(low_alt_slice)

    return sorted(low_alt_slices)


def find_level_off(array, frequency, _slice):
    '''
    :param array: Altitude array
    :type array: np.ma.array
    :param frequency: Altitude array sampling frequency
    :type frequency: float
    :param _slice: Slice of the array to be processed
    :type _slice: Python slice

    :returns: index of level flight phase.
    :type: integer
    '''
    if _slice.step == -1:
        my_slice = slice(_slice.stop, _slice.start, 1)
    else:
        my_slice = _slice

    data = array[my_slice]
    if data[0] < data[-1]:
        # Treat as climbing
        index = find_toc_tod(data, slice(None, None), frequency, mode='toc')
        return my_slice.start + index
    else:
        # Treat as descending
        index = find_toc_tod(data, slice(None, None), frequency, mode='tod')
        return my_slice.start + index


def find_toc_tod(alt_data, ccd_slice, frequency, mode=None):
    '''
    Find the Top Of Climb or Top Of Descent from an altitude trace.

    This code separates the climb cruise descent into half width sections,
    section_2, and then divides again into top and bottom halves, section_4.
    This means that the code can be fairly easily extended to detect bottoms
    of climb and descent if required.

    :param alt_data: Altitude array usually above FL100
    :type alt_data: np.ma.array
    :param ccd_slice: "cruise climb descent" slice of data, although similar will do
    :type ccd_slice: slice
    :param frequency: alt_data sampling frequency
    :type frequency: float
    :param mode: Either 'Climb' or 'Descent' to define which to select.
    :type mode: String
    :returns: Index of location identified within slice, relative to start of alt_data
    :rtype: Int
    '''
    # Find the maximum, minimim and midpoint altitudes and their indexes in
    # this slice
    peak = max_value(alt_data, ccd_slice)
    if mode == 'toc':
        section_2 = slice(ccd_slice.start or 0, int(peak.index)+1)
        slope = SLOPE_FOR_TOC_TOD / frequency
    elif mode == 'tod':
        # Descent case
        section_2 = slice(int(peak.index) or 0, ccd_slice.stop or len(alt_data))
        slope = -SLOPE_FOR_TOC_TOD / frequency
    else:
        raise ValueError('Invalid mode in find_toc_tod')

    # Find the midpoint to separate tops and bottoms of climbs and descents.
    trough = min_value(alt_data, section_2)
    mid_alt = (peak.value + trough.value) / 2.0
    try:
        mid_index = int(index_at_value(alt_data[section_2], mid_alt, endpoint='nearest') + (section_2.start or 0))
    except:
        # no peak found
        return np.ma.argmax(alt_data)

    if mode == 'tod':
        # The first part is where the point of interest lies
        if section_2.start == mid_index:
            return np.ma.argmax(alt_data[section_2]) + (section_2.start or 0)
        section_4 = slice(section_2.start, mid_index)
    else:
        # we need to scan the second part of this half.
        section_4 = slice(mid_index, section_2.stop+1)

    # Establish a simple monotonic timebase
    timebase = np.arange(len(alt_data[section_4]))
    # Then scale this to the required altitude data slope
    ramp = timebase * slope

    test_slope = alt_data[section_4] - ramp
    return np.ma.argmax(test_slope) + section_4.start


def find_edges(array, _slice=slice(None), direction='rising_edges'):
    '''
    Edge finding low level routine, called by create_ktis_at_edges (and
    historically create_kpvs_at_edges). Also useful within algorithms
    directly.

    :param array: array of values to scan for edges
    :type array: Numpy masked array
    :param _slice: slice to be examined
    :type _slice: slice
    :param direction: Optional edge direction for sensing. Default 'rising_edges'
    :type direction: string, one of 'rising_edges', 'falling_edges' or 'all_edges'.

    :returns edge_list: Indexes for the appropriate edge transitions.
    :type edge_list: list of floats.

    Note: edge_list values are always integer+0.5 as it is assumed that the
    transition took place (with highest probability) midway between the two
    recorded states.
    '''
    if ((_slice.start is not None or _slice.stop is not None) and
        len(array[_slice]) < 2):
        # Slice of array is too small for finding edges.
        # Avoid unnecessarily copying whole array if slice is default.
        return []

    if direction == 'rising_edges':
        method = 'fill_start'
    else:
        method = 'fill_stop'
    array = repair_mask(array, method=method, repair_duration=None, copy=True)
    # Find increments. Extrapolate at start to keep array sizes straight.
    deltas = np.ma.ediff1d(array[_slice], to_begin=array[_slice][0])
    deltas[0]=0 # Ignore the first value
    if direction == 'rising_edges':
        edges = np.ma.nonzero(np.ma.maximum(deltas, 0))
    elif direction == 'falling_edges':
        edges = np.ma.nonzero(np.ma.minimum(deltas, 0))
    elif direction == 'all_edges':
        edges = np.ma.nonzero(deltas)
    else:
        raise ValueError('Edge direction not recognised')

    # edges is a tuple catering for multi-dimensional arrays, but we
    # are only interested in 1-D arrays, hence selection of the first
    # element only.
    # The -0.5 shifts the value midway between the pre- and post-change
    # samples.
    edge_list = edges[0] + int(_slice.start or 0) - 0.5
    return list(edge_list)


def find_edges_on_state_change(state, array, change='entering', phase=None, min_samples=1):
    '''
    Version of find_edges tailored to suit multi-state parameters.

    :param state: multistate parameter condition e.g. 'Ground'
    :type state: text, from the states for that parameter.
    :param array: the multistate parameter array
    :type array: numpy masked array with state attributes.

    :param change: Condition for detecting edge. Default 'entering', 'leaving' and 'entering_and_leaving' alternatives
    :type change: text
    :param phase: Flight phase or list of slices within which edges will be detected.
    :type phase: list of slices, default=None
    :param min_samples: Minimum number of samples within desired state.
                        Usually 1 or 2 for warnings, more for multistates which remain at their state for longer.
    :type min_samples: int

    :returns: list of indexes

    :raises: ValueError if change not recognised
    :raises: KeyError if state not recognised
    '''
    def state_changes(state, array, change, _slice=slice(0, -1), min_samples=1):
        '''
        min_samples of 3 means 3 or more samples must be in the state for it to be returned.
        '''
        _slice = slices_int(_slice)
        length = len(array[_slice])
        # The offset allows for phase slices and puts the transition midway
        # between the two conditions as this is the most probable time that
        # the change took place.
        offset = _slice.start - 0.5
        state_periods = runs_of_ones(array[_slice] == state)
        # ignore small periods where slice is in state, then remove small
        # gaps where slices are not in state
        # we are taking 1 away from min_samples here as
        # slices_remove_small_slices removes slices of less than count
        state_periods = slices_remove_small_slices(state_periods, count=min_samples - 1)
        state_periods = slices_remove_small_gaps(state_periods, count=min_samples)
        edge_list = []
        for period in state_periods:
            if change == 'entering':
                if period.start > 0:
                    edge_list.append(period.start + offset)

            elif change == 'leaving':
                if period.stop < length:
                    edge_list.append(period.stop + offset)

            elif change == 'entering_and_leaving':
                if period.start > 0:
                    edge_list.append(period.start + offset)
                if period.stop < length:
                    edge_list.append(period.stop + offset)
            else:
                raise  ValueError("Change '%s'in find_edges_on_state_change not recognised" % change)

        return edge_list

    if phase is None:
        return state_changes(state, array, change, min_samples=min_samples)

    edge_list = []
    for period in phase:
        period = getattr(period, 'slice', period)
        edges = state_changes(state, array, change, _slice=period,
                              min_samples=min_samples)
        edge_list.extend(edges)
    return edge_list


def first_valid_parameter(*parameters, **kwargs):
    '''
    Returns the first valid parameter from the provided list.

    A parameter is valid if it has an array that is not fully masked.

    Optionally, a list of phases can be provided during which the parameter
    must be valid. We only require the parameter to be valid during one of the
    phases.

    :param *parameters: parameters to check for validity/availability.
    :type *parameters: Parameter
    :param phases: a list of slices where the parameter must be valid.
    :type phases: list of slice
    :returns: the first valid parameter from those provided.
    :rtype: Parameter or None
    '''
    phases = kwargs.get('phases')
    if phases:
        phases = slices_int(phases)
    else:
        phases = [slice(None)]
    for parameter in parameters:
        if not parameter:
            continue
        for phase in phases:
            if not parameter.array[phase].mask.all():
                return parameter
    else:
        return None


def first_valid_sample(array, start_index=0):
    '''
    Returns the first valid sample of data from a point in an array.

    :param array: array of values to scan
    :type array: Numpy masked array
    :param start_index: optional initial point for the scan. Must be positive.
    :type start_index: integer

    :returns index: index for the first valid sample at or after start_index.
    :type index: Integer or None
    :returns value: the value of first valid sample.
    :type index: Float or None
    '''
    # Trap to ensure we don't stray into the far end of the array and that the
    # sliced array is not empty.
    if not 0 <= start_index < len(array):
        return Value(None, None)

    clumps = np.ma.clump_unmasked(array[start_index:])
    if clumps:
        index = clumps[0].start + start_index
        return Value(index, array[index])
    else:
        return Value(None, None)


def last_valid_sample(array, end_index=None, min_samples=None):
    '''
    Returns the last valid sample of data before a point in an array.

    :param array: array of values to scan
    :type array: Numpy masked array
    :param end_index: optional initial point for the scan. May be negative.
    :type end_index: integer

    :returns index: index for the last valid sample at or before end_index.
    :type index: Integer or None
    :returns value: the value of last valid sample.
    :type index: Float or None
    '''
    if end_index is None:
        end_index = len(array)
    elif end_index > len(array):
        return Value(None, None)

    clumps = np.ma.clump_unmasked(array[:end_index+1])
    if min_samples:
        clumps = slices_remove_small_slices(clumps, count=min_samples)
    if clumps:
        index = clumps[-1].stop - 1
        return Value(index, array[index])
    else:
        return Value(None, None)


def first_order_lag(param, time_constant, hz, gain=1.0, initial_value=None):
    '''
    Computes the transfer function
            x.G
    y = -----------
         (1 + T.s)
    where:
    x is the input function
    G is the gain
    T is the timeconstant
    s is the Laplace operator
    y is the output

    Basic example:
    first_order_lag(param, time_constant=5) is equivalent to
    array[index] = array[index-1] * 0.8 + array[index] * 0.2.

    :param param: input data (x)
    :type param: masked array
    :param time_constant: time_constant for the lag function (T)(sec)
    :type time_constant: float
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param gain: gain of the transfer function (non-dimensional)
    :type gain: float
    :param initial_value: initial value of the transfer function at t=0
    :type initial_value: float
    :returns: masked array of values with first order lag applied
    '''
    #input_data = np.copy(array.data)

    # Scale the time constant to allow for different data sample rates.
    tc = time_constant * hz

    # Trap the condition for stability
    if tc < 0.5:
        raise ValueError('Lag timeconstant too small')

    x_term = []
    x_term.append (gain / (1.0 + 2.0*tc)) #b[0]
    x_term.append (gain / (1.0 + 2.0*tc)) #b[1]
    x_term = np.array(x_term)

    y_term = []
    y_term.append (1.0) #a[0]
    y_term.append ((1.0 - 2.0*tc)/(1.0 + 2.0*tc)) #a[1]
    y_term = np.array(y_term)

    return masked_first_order_filter(y_term, x_term, param, initial_value)


def masked_first_order_filter(y_term, x_term, param, initial_value):
    """
    This provides access to the scipy filter function processed across the
    unmasked data blocks, with masked data retained as masked zero values.
    This is a better option than masking all subsequent values which would be
    the mathematically correct thing to do with infinite response filters.

    :param y_term: Filter denominator terms.
    :type param: list
    :param x_term: Filter numerator terms.
    :type x_term: list
    :param param: input data array
    :type param: masked array
    :param initial_value: Value to be used at the start of the data
    :type initial_value: float (or may be None)
    """
    # import locally to speed up imports of library.py
    from scipy.signal import lfilter, lfilter_zi
    z_initial = lfilter_zi(x_term, y_term) # Prepare for non-zero initial state
    # The initial value may be set as a command line argument, mainly for testing
    # otherwise we set it to the first data value.

    result = np.ma.zeros(len(param))  # There is no zeros_like method.
    good_parts = np.ma.clump_unmasked(param)
    for good_part in good_parts:

        if initial_value is None:
            initial_value = param[good_part.start]
        # Tested version here...
        answer, z_final = lfilter(x_term, y_term, param[good_part], zi=z_initial*initial_value)
        result[good_part] = np.ma.array(answer)

    # The mask should last indefinitely following any single corrupt data point
    # but this is impractical for our use, so we just copy forward the original
    # mask.
    bad_parts = np.ma.clump_masked(param)
    for bad_part in bad_parts:
        # The mask should last indefinitely following any single corrupt data point
        # but this is impractical for our use, so we just copy forward the original
        # mask.
        result[bad_part] = np.ma.masked

    return result


def first_order_washout(param, time_constant, hz, gain=1.0, initial_value=None):
    '''
    Computes the transfer function
         x.G.(T.s)
    y = -----------
         (1 + T.s)
    where:
    x is the input function
    G is the gain
    T is the timeconstant
    s is the Laplace operator
    y is the output

    :param param: input data (x)
    :type param: masked array
    :param time_constant: time_constant for the lag function (T)(sec)
    :type time_constant: float
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param gain: gain of the transfer function (non-dimensional)
    :type gain: float
    :param initial_value: initial value of the transfer function at t=0
    :type initial_value: float
    :returns: masked array of values with first order lag applied
    '''
    #input_data = np.copy(param.data)

    # Scale the time constant to allow for different data sample rates.
    tc = time_constant * hz

    # Trap the condition for stability
    if tc < 0.5:
        raise ValueError('Lag timeconstant too small')

    x_term = []
    x_term.append (gain*2.0*tc  / (1.0 + 2.0*tc)) #b[0]
    x_term.append (-gain*2.0*tc / (1.0 + 2.0*tc)) #b[1]
    x_term = np.array(x_term)

    y_term = []
    y_term.append (1.0) #a[0]
    y_term.append ((1.0 - 2.0*tc)/(1.0 + 2.0*tc)) #a[1]
    y_term = np.array(y_term)

    return masked_first_order_filter(y_term, x_term, param, initial_value)


def runway_distance_from_end(runway, *args, **kwds):
    """
    Distance from the end of the runway to any point. The point is first
    snapped onto the runway centreline and then the distance from the runway
    end is taken. This is a convenient startingpoint for measuring runway
    landing distances.

    Note: If high accuracy is required, compute the latitude and longitude
    using the value_at_index function rather than just indexing into the
    latitude and longitude array. Alternatively use KPVs 'Latitude Smoothed At
    Touchdown' and 'Longitude Smoothed At Touchdown' which are the most
    accurate locations we have available for touchdown.

    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']
    *args if supplied are the latitude and longitude of a point.
    :param lat: Latitude of the point of interest
    :type lat: float
    :param lon: Longitude of the point of interest
    :type lon: float

    **kwds if supplied are a point in the runway dictionary
    :param point: dictionary name of the point of reference, e.g. 'glideslope'
    :type point: String

    :return distance: Distance from runway end to the point of interest, along runway centreline.
    :type distance: float (units=metres)
    """
    if args:
        new_lat, new_lon = runway_snap(runway, args[0], args[1])
    else:
        try:
            # if kwds['point'] in ['localizer', 'glideslope', 'start']:
            new_lat, new_lon = runway_snap(runway, runway[kwds['point']]['latitude'], runway[kwds['point']]['longitude'])
        except (KeyError, ValueError):
            logger.warning ('Runway_distance_from_end: Unrecognised or missing'\
                            ' keyword %s for runway id %s',
                            kwds['point'], runway['id'])
            return None

    if new_lat and new_lon:
        return great_circle_distance__haversine(new_lat, new_lon,
                                                runway['end']['latitude'],
                                                runway['end']['longitude'])
    else:
        return None


def runway_deviation(array, runway={}, heading=None):
    '''
    Computes an array of heading deviations from the selected runway
    centreline calculated from latitude/longitude coordinates. For use with
    True Heading.

    If you use heading, it allows one to use magnetic heading
    comparisons.

    NOTE: Uses heading supplied in preference to coordinates.

    :param array: array or Value of TRUE heading values
    :type array: Numpy masked array (usually already sliced to relate to the landing in question).
    :param runway: runway details.
    :type runway: dict (runway.value if this is taken from an attribute).
    :param heading: heading to use, in preference to runway.
    :type heading: Int/Float

    :returns dev: array of heading deviations
    :type dev: Numpy masked array.
    '''
    if heading is not None:
        rwy_hdg = float(heading)
    else:
        rwy_hdg = runway_heading(runway)
    dev = (array - rwy_hdg) % 360
    return np.ma.where(dev>180.0, dev-360.0, dev)


def heading_diff(heading1, heading2):
    '''
    Difference between heading1 and heading2. If heading1 is greater than
    heading2, the returned difference will be negative.

    :type heading1: int or float or MaskedArray
    :type heading2: int or float or MaskedArray
    :rtype: value of int or float else masked array
    '''
    # validate heading values are between 0 and 360
    array_input = False
    for heading in (heading1, heading2):
        if isinstance(heading,  np.ma.MaskedArray):
            array_input = True
            assert (0 <= heading).all() & (heading < 360).all()
        else:
            assert (0 <= heading < 360)

    diff = heading2 - heading1
    abs_diff = abs(diff)
    array = 360 - abs_diff
    array = np.ma.where(abs_diff <= 180, diff, array)
    array = np.ma.where(np.ma.logical_and(abs_diff > 180, heading2 > heading1),
                        abs_diff - 360, array)
    if array_input:
        return array
    else:
        return array.flatten()[0]


def runway_distances(runway):
    '''
    Projection of the ILS antenna positions onto the runway
    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']
    ['localizer']['latitude'] ILS localizer antenna position
    ['localizer']['longitude']
    ['glideslope']['latitude'] ILS glideslope antenna position
    ['glideslope']['longitude']

    :return
    :param start_loc: distance from start of runway to localizer antenna
    :type start_loc: float, units = metres
    :param gs_loc: distance from projected position of glideslope antenna on runway centerline to the localizer antenna
    :type gs_loc: float, units = metres
    :param end_loc: distance from end of runway to localizer antenna
    :type end_loc: float, units = metres
    :param pgs_lat: projected position of glideslope antenna on runway centerline
    :type pgs_lat: float, units = degrees latitude
    :param pgs_lon: projected position of glideslope antenna on runway centerline
    :type pgs_lon: float, units = degrees longitude
    '''
    start_lat = runway['start']['latitude']
    start_lon = runway['start']['longitude']
    end_lat = runway['end']['latitude']
    end_lon = runway['end']['longitude']
    lzr_lat = runway['localizer']['latitude']
    lzr_lon = runway['localizer']['longitude']
    gs_lat = runway['glideslope']['latitude']
    gs_lon = runway['glideslope']['longitude']

    #a = great_circle_distance__haversine(gs_lat, gs_lon, lzr_lat, lzr_lon)
    #b = great_circle_distance__haversine(gs_lat, gs_lon, start_lat, start_lon)
    #c = great_circle_distance__haversine(end_lat, end_lon, lzr_lat, lzr_lon)
    #d = great_circle_distance__haversine(start_lat, start_lon, lzr_lat, lzr_lon)

    #r = (1.0+(a**2 - b**2)/d**2)/2.0
    #g = r*d

    # =========================================================================
    # We have a problem that some runway coordinates were imported into the
    # database with the latitude and longitude reversed. This only applies to
    # localizer and glideslope coordinates. The traps that follow identify
    # the error and correct it locally, allowing for manual confirmation of
    # the error and correction of the database at a later stage.
    if (start_lat-lzr_lat)**2 > (start_lat-lzr_lon)**2 and \
       (end_lat-lzr_lat)**2 > (end_lat-lzr_lon)**2:
        lzr_lat = runway['localizer']['longitude']
        lzr_lon = runway['localizer']['latitude']
        logger.warning('Reversing lat and long for localizer on runway %d' %runway['id'])

    if (start_lat-gs_lat)**2 > (start_lat-gs_lon)**2 and \
       (end_lat-gs_lat)**2 > (end_lat-gs_lon)**2:
        gs_lat = runway['glideslope']['longitude']
        gs_lon = runway['glideslope']['latitude']
        logger.warning('Reversing lat and long for glideslope on runway %d' %runway['id'])
    # =========================================================================

    start_2_loc = great_circle_distance__haversine(start_lat, start_lon, lzr_lat, lzr_lon)
    # The projected glideslope antenna position is given by this formula
    pgs_lat, pgs_lon = runway_snap(runway, gs_lat, gs_lon)
    gs_2_loc = great_circle_distance__haversine(pgs_lat, pgs_lon, lzr_lat, lzr_lon)
    end_2_loc = great_circle_distance__haversine(end_lat, end_lon, lzr_lat, lzr_lon)

    return start_2_loc, gs_2_loc, end_2_loc, pgs_lat, pgs_lon  # Runway distances to start, glideslope and end.


def runway_length(runway):
    '''
    Calculation of only the length for runways with no glideslope details
    and possibly no localizer information. In these cases we assume the
    glideslope is near end of runway and the beam is 700ft wide at the
    threshold.

    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']

    :return
    :param start_end: distance from start of runway to end
    :type start_loc: float, units = metres.

    :error conditions
    :runway without adequate information fails with ValueError
    '''

    try:
        start_lat = runway['start']['latitude']
        start_lon = runway['start']['longitude']
        end_lat = runway['end']['latitude']
        end_lon = runway['end']['longitude']

        return great_circle_distance__haversine(start_lat, start_lon, end_lat, end_lon)
    except:
        raise ValueError("runway_length unable to compute length of runway id='%s'" %runway['id'])


def runway_touchdown(runway):
    '''
    Distance from runway start to centre of touchdown markings.
    ref: https://www.icao.int/safety/Implementation/Library/Manual%20Aerodrome%20Stds.pdf
    page 68, taking lenth to beginning of marking plus half the minimum length of stripe.

    :param runway: Runway location details dictionary.
    :type runway: Dictionary

    :return
    :param tdn_dist: distance from start of runway to touchdown point
    :type tdn_dist: float, units = metres.
    :param aiming_point: aiming point on runways
    :type aiming_point: dict, with keys 'latitude' and 'longitude' both units = degrees
    '''
    length = runway_length(runway)
    if length:
        if length < 800:
            tdn_dist = 165
        elif length < 1200:
            tdn_dist = 265
        elif length < 2400:
            tdn_dist = 322.5
        elif length > 2400:
            tdn_dist = 422.5

        r = tdn_dist / length

        lat = (1.0-r) * runway['start']['latitude'] + r * runway['end']['latitude']
        lon = (1.0-r) * runway['start']['longitude'] + r * runway['end']['longitude']
        return tdn_dist, {'latitude':lat, 'longitude':lon}

    else:
        # Helipads
        if runway:
            locn = runway['start']
        else:
            locn = None
        return 0.0, locn


def runway_heading(runway):
    '''
    Computation of the runway heading from endpoints.
    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']

    :return
    :param rwy_hdg: true heading of runway centreline.
    :type rwy_hdg: float, units = degrees, facing from start to end.

    :error conditions
    :runway without adequate information fails with ValueError
    '''
    try:
        end_lat = runway['end']['latitude']
        end_lon = runway['end']['longitude']

        brg, dist = bearings_and_distances(np.ma.array(end_lat),
                                           np.ma.array(end_lon),
                                           runway['start'])
        rwy_brg = float(brg.data)
        try:
            rwy_data = float(runway['magnetic_heading'])
        except KeyError:
            return rwy_brg
        diff = (rwy_brg - rwy_data + 180) % 360 - 180
        if abs(diff) > 90.0:
            rwy_brg = (rwy_brg + 180) % 360.0
        return rwy_brg
    except:
        if runway:
            raise ValueError("runway_heading unable to resolve heading for runway: %s" % runway)
        else:
            raise ValueError("runway_heading unable to resolve heading; no runway")


def runway_snap_dict(runway, lat, lon):
    """
    This function snaps any location onto the closest point on the runway centreline.

    :param runway: Dictionary containing the runway start and end points.
    :type dict
    :param lat: Latitude of the point to snap
    :type lat: float
    :param lon: Longitude of the point to snap
    :type lon: float

    :returns dictionary {['latitude'],['longitude']}
    :type dict.
    """
    lat, lon = runway_snap(runway, lat, lon)
    to_return = {}
    to_return['latitude'] = lat
    to_return['longitude'] = lon
    return to_return


def runway_snap(runway, lat, lon):
    """
    This function snaps any location onto the closest point on the runway centreline.

    :param runway: Dictionary containing the runway start and end points.
    :type dict
    :param lat: Latitude of the point to snap
    :type lat: float
    :param lon: Longitude of the point to snap
    :type lon: float

    :returns new_lat, new_lon: Amended position now on runway centreline.
    :type float, float.
    http://www.flightdatacommunity.com/breaking-runways/
    """
    try:
        start_lat = runway['start']['latitude']
        start_lon = runway['start']['longitude']
        end_lat = runway['end']['latitude']
        end_lon = runway['end']['longitude']
    except:
        # Can't do the sums without endpoints.
        return None, None

    # =========================================================================
    # We have a problem that some runway coordinates were imported into the
    # database with the latitude and longitude reversed. This only applies to
    # localizer and glideslope coordinates. The traps that follow identify
    # the error and correct it locally, allowing for manual confirmation of
    # the error and correction of the database at a later stage.
    if (start_lat-lat)**2 > (start_lat-lon)**2 and \
       (end_lat-lat)**2 > (end_lat-lon)**2:
        x = lat
        lat = lon
        lon = x
        logger.warning('Reversing lat and long in runway_snap')
    # =========================================================================

    a = great_circle_distance__haversine(lat, lon, end_lat, end_lon)
    b = great_circle_distance__haversine(lat, lon, start_lat, start_lon)
    d = great_circle_distance__haversine(start_lat, start_lon, end_lat, end_lon)

    if not a or not b:
        return lat, lon

    if max(a,b,d)>20000:
        return None, None
        # was raise ValueError('Runway snap unrealistic distance')

    if d:
        r = (1.0+(a**2 - b**2)/d**2)/2.0

        # The projected glideslope antenna position is given by this formula
        new_lat = end_lat + r*(start_lat - end_lat)
        new_lon = end_lon + r*(start_lon - end_lon)

        return new_lat, new_lon

    else:
        return None, None

def groundspeed_from_position(lat, lon, hz):
    '''
    Calculation of Groundspeed from latitude and longitude

    :param lat: Latitude Prepared array
    :type lat: numpy array
    :param lon: Longitude Prepared array
    :type lon: numpy array

    :returns
    :param gs: groundspeed in knots
    :type gs: numpy array
    '''
    deg_to_metres = 1852*60 # 1852 m/nm & 60 nm/deg latitude
    dlat = rate_of_change_array(lat, hz, 2.0) * deg_to_metres
    # There is no masked array cos function, but the mask will be
    # carried forward as part of the latitude array anyway.
    dlon = rate_of_change_array(lon, hz, 2.0) * deg_to_metres *\
        np.cos(np.radians(lat))

    gs = np.ma.sqrt(dlat*dlat+dlon*dlon) / 0.514 # kn per m/s
    return gs

def ground_track(lat_fix, lon_fix, gspd, hdg, frequency, mode):
    """
    Computation of the ground track assuming no slipping.
    :param lat_fix: Fixed latitude point at one end of the data.
    :type lat_fix: float, latitude degrees.
    :param lon_fix: Fixed longitude point at the same time as lat_fix.
    :type lat_fix: float, longitude degrees.
    :param gspd: Groundspeed in knots
    :type gspd: Numpy masked array.
    :param hdg: True heading in degrees.
    :type hdg: Numpy masked array.
    :param frequency: Frequency of the groundspeed and heading data (used for integration scaling).
    :type frequency: Float (units = Hz)
    :param mode: type of calculation to be completed.
    :type mode: String, either 'takeoff' or 'landing' accepted.

    :returns
    :param lat_track: Latitude of computed ground track
    :type lat_track: Numpy masked array
    :param lon_track: Longitude of computed ground track
    :type lon_track: Numpy masked array.

    :error conditions
    :Fewer than 5 valid data points, returns None, None
    :Invalid mode fails with ValueError
    :Mismatched array lengths fails with ValueError
    """

    # We are going to extend the lat/lon_fix point by the length of the gspd/hdg arrays.
    # First check that the gspd/hdg arrays are sensible.
    if len(gspd) != len(hdg):
        raise ValueError('Ground_track requires equi-length groundspeed and '
                         'heading arrays')

    # Dummy masked array to join the invalid data arrays
    result=np.ma.array(np.zeros_like(gspd))
    result.mask = np.ma.logical_or(np.ma.getmaskarray(gspd),
                                   np.ma.getmaskarray(hdg))
    # It's not worth doing anything if there is too little data
    if np.ma.count(result) < 5:
        return None, None

    # Force a copy of the result array, as the repair_mask functions will
    # otherwise overwrite the result mask.
    result = np.ma.copy(result)

    repair_mask(gspd, repair_duration=None)
    repair_mask(hdg, repair_duration=None)

    if mode == 'takeoff':
        direction = 'backwards'
    elif mode == 'landing':
        direction = 'forwards'
    else:
        raise ValueError('Ground_track only recognises takeoff or landing '
                         'modes')

    hdg_rad = hdg * deg2rad
    delta_north = gspd * np.ma.cos(hdg_rad)
    delta_east = gspd * np.ma.sin(hdg_rad)

    scale = ut.multiplier(ut.KT, ut.METER_S)
    north = integrate(delta_north, frequency, scale=scale, direction=direction)
    east = integrate(delta_east, frequency, scale=scale, direction=direction)

    bearing = np.ma.array(np.rad2deg(np.arctan2(east, north)))
    distance = np.ma.array(np.ma.sqrt(north**2 + east**2))
    distance.mask = result.mask

    gt_lat, gt_lon = latitudes_and_longitudes(bearing, distance,
                                        {'latitude':lat_fix,
                                         'longitude':lon_fix})

    return gt_lat, gt_lon


def ground_track_precise(lat, lon, speed, hdg, frequency):
    """
    Computation of the ground track.
    :param lat: Latitude for the duration of the ground track.
    :type lat: Numpy masked array, latitude degrees.
    :param lon: Longitude for the duration of the ground track.
    :type lat: Numpy masked array, longitude degrees.

    :param gspd: Groundspeed for the duration of the ground track.
    :type gspd: Numpy masked array in knots.
    :param hdg: True heading for the duration of the ground track.
    :type hdg: Numpy masked array in degrees.

    :param frequency: Frequency of the array data (required for integration scaling).
    :type frequency: Float (units = Hz)

    :returns
    :param lat_track: Latitude of computed ground track
    :type lat_track: Numpy masked array
    :param lon_track: Longitude of computed ground track
    :type lon_track: Numpy masked array.

    :error conditions
    :Mismatched array lengths fails with ValueError
    """

    # Build arrays to return the computed track.
    lat_return = np_ma_masked_zeros_like(lat)
    lon_return = np_ma_masked_zeros_like(lat)
    lat_model = np_ma_masked_zeros_like(lat)
    lon_model = np_ma_masked_zeros_like(lat)

    # We are going to extend the lat/lon_fix point by the length of the gspd/hdg arrays.
    # First check that the gspd/hdg arrays are sensible.
    if (len(speed) != len(hdg)) or (len(speed) != len(lat)) or (len(speed) != len(lon)):
        raise ValueError('Ground_track_precise requires equi-length arrays')

    # We use the period where the speed was over 1kn and just leave the position at lower speeds unchanged.
    spd_above_1 = np.ma.masked_less_equal(np.ma.abs(speed), 1.0)
    track_slices = slices_remove_small_slices(np.ma.clump_unmasked(spd_above_1))

    # import matplotlib.pyplot as plt
    # plt.plot(lon, lat, '-k')

    hdg_hyst_chg = np.ma.ediff1d(hysteresis(hdg, 10.0))
    all_straights = np.ma.clump_unmasked(np.ma.masked_not_equal(hdg_hyst_chg, 0.0))

    for track_slice in track_slices:
        straights = slices_remove_small_slices(slices_and(all_straights, [track_slice]))
        curves = slices_remove_small_slices(
            slices_not(
                straights, begin_at=track_slice.start, end_at=track_slice.stop),
            count=1,
        )

        for straight in straights:
            # We compute an average ground track from heading and groundspeed.
            # This is computed in each direction, then blended progressively so that it meets the
            # endpoints exactly thereby cancelling out the errors in integrating the ground track.
            lat_model[straight], lon_model[straight] = av_gnd_trk(lat[straight], lon[straight], speed[straight], hdg[straight], frequency)
            # plt.plot(lon_model[straight], lat_model[straight], 'o-b')

        for curve in curves:
            # If this has to match the ILS track we need to blend it in nicely.
            if curve.start == 0:
                lat_model[curve] = gtp_blend_curve(lat[curve])
                lon_model[curve] = gtp_blend_curve(lon[curve])
            # We just use the prepared track because during turns the prepared track errors are usually not obvious
            else:
                lat_model[curve] = lat[curve]
                lon_model[curve] = lon[curve]
            # plt.plot(lon_model[curve], lat_model[curve], 'o-r')

    # plt.show()
    # We have computed the straight and curved moving sections. Where the aircraft was barely moving,
    # we leave it stationary at the point where the last movement stopped.
    lat_return = repair_mask(lat_model, repair_duration=None, extrapolate=True, method='fill_start', raise_entirely_masked=False)
    lon_return = repair_mask(lon_model, repair_duration=None, extrapolate=True, method='fill_start', raise_entirely_masked=False)

    return lat_return, lon_return

def gtp_blend_curve(array):
    '''
    The first data point for a ground track with ILS localizer correction is set from the localizer
    track computation, so we blend in this point across the curve turning off the runway.

    We can do this for the first track leaving the stand as it makes no difference and saves
    the complexity of knowing which piece of track we are calculating.
    '''
    delta = array[0]-array[1]
    out = array + np.linspace(delta, 0.0, num=len(array))
    out[0] -= delta
    return out

def av_gnd_trk(lat, lon, gspd, hdg, hz):
    '''
    Computation of an average ground track from the start forwards, and the end backwards, averaged.

    Helper function for ground_track_precise calling ground_track twice per segment.
    '''
    lat1, lon1 = ground_track(lat[0], lon[0], gspd, hdg, hz, 'landing')
    lat2, lon2 = ground_track(lat[-1], lon[-1], gspd, hdg, hz, 'takeoff')
    scale = np.linspace(1.0, 0.0, num=len(lat), endpoint=True)
    my_lat = lat1 * scale + lat2 * (1.0-scale)
    my_lon = lon1 * scale + lon2 * (1.0-scale)
    return my_lat, my_lon


def hash_array(array, sections, min_samples):
    '''
    Creates a sha256 hash from the array's tostring() method .
    '''
    checksum = sha256()
    for section in sections:
        if section.stop - section.start < min_samples:
            continue
        checksum.update(array[section].tostring())

    return checksum.hexdigest()


def hysteresis(array, hysteresis):
    """
    Applies hysteresis to an array of data. The function applies half the
    required level of hysteresis forwards and then backwards to provide a
    phase neutral result.

    :param array: Input data for processing
    :type array: Numpy masked array
    :param hysteresis: Level of hysteresis to apply.
    :type hysteresis: Float
    """
    if np.ma.count(array) == 0: # No unmasked elements.
        return array

    if hysteresis == 0.0 or hysteresis is None:
        return array

    if hysteresis < 0.0:
        raise ValueError("Hysteresis called with negative threshold")
        return array

    quarter_range = hysteresis / 4.0
    # Length is going to be used often, so prepare here:
    length = len(array)
    half_done = np.zeros(length)
    result = np.zeros(length)
    length = length-1 #  To be used for array indexing next

    # get a list of the unmasked data - allow for array.mask = False (not an array)
    if array.mask is np.False_:
        notmasked = np.arange(length+1)
    else:
        notmasked = np.ma.where(~array.mask)[0]
    # The starting point for the computation is the first notmasked sample.
    old = array[notmasked[0]]
    for index in notmasked:
        new = array[index]

        if new - old > quarter_range:
            old = new  - quarter_range
        elif new - old < -quarter_range:
            old = new + quarter_range
        half_done[index] = old

    # Repeat the process in the "backwards" sense to remove phase effects.
    for index in notmasked[::-1]:
        new = half_done[index]
        if new - old > quarter_range:
            old = new  - quarter_range
        elif new - old < -quarter_range:
            old = new + quarter_range
        result[index] = old

    # At the end of the process we reinstate the mask, although the data
    # values may have affected the result.
    return np.ma.array(result, mask=array.mask)


def ils_established(array, _slice, hz, point='established'):
    '''
    Helper function for ILS established computations
    :param array: ILS localizer or glideslope data array
    :type array: numpy masked array in dots
    :param _slice: slice of this array to scan
    :type _slice: Python slice
    :param hz: frequency of the array
    :type hz: float
    :param hz: point of interest 'established', 'immediate', 'end'
    :type hz: float
    '''
    if point in ('established', 'end'):
        time_required = ILS_ESTABLISHED_DURATION
    elif point == 'immediate':
        time_required = 1.0

    scope = slices_int(_slice)
    if scope.start == scope.stop:
        return None

    # When is the ILS signal within ILS_CAPTURE (0.5 dots)?
    captures = np.ma.clump_unmasked(
        np.ma.masked_greater(
            np.ma.abs(array[scope]),
            ILS_CAPTURE)
    )
    # When is the signal changing at less than ILS_CAPTURE_RATE (+/- 0.1 dots/second)?
    ils_rate = rate_of_change_array(array[scope], hz)
    low_rocs = np.ma.clump_unmasked(np.ma.masked_greater(np.ma.abs(ils_rate), ILS_CAPTURE_ROC))

    # We want both conditions to be true at the same time, so AND the two conditions
    capture_slices = slices_and(captures, low_rocs)
    if point == 'end':
        valid = slices_remove_small_gaps(
            np.ma.clump_unmasked(array[scope]),
            time_limit=time_required,
            hz=hz
        )
        capture_slices = slices_and(capture_slices, valid)

    for capture in capture_slices:
        # and now check that this period is longer than the required period.
        if slice_duration(capture, hz) >= time_required:
            if point == 'end':
                # get index passing 1 dot
                return index_at_value(np.ma.abs(array), 1.0, slice(_slice.start + (_slice.step or 1)*capture.stop, _slice.stop))
            else:
                return _slice.start + (_slice.step or 1)*capture.start
    return None


def ils_glideslope_align(runway):
    '''
    Projection of the ILS glideslope antenna onto the runway centreline
    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']
    ['glideslope']['latitude'] ILS glideslope antenna position
    ['glideslope']['longitude']

    :returns dictionary containing:
    ['latitude'] ILS glideslope position aligned to start and end of runway
    ['longitude']

    :error: if there is no glideslope antenna in the database for this runway, returns None
    '''
    try:
        new_lat, new_lon = runway_snap(runway,
                                       runway['glideslope']['latitude'],
                                       runway['glideslope']['longitude'])
        return {'latitude':new_lat, 'longitude':new_lon}
    except KeyError:
        return None


def ils_localizer_align(runway):
    '''
    Projection of the ILS localizer antenna onto the runway centreline
    :param runway: Runway location details dictionary.
    :type runway: Dictionary containing:
    ['start']['latitude'] runway start position
    ['start']['longitude']
    ['end']['latitude'] runway end position
    ['end']['longitude']
    ['localizer']['latitude'] ILS localizer antenna position
    ['localizer']['longitude']

    :returns dictionary containing:
    ['latitude'] ILS localizer position aligned to start and end of runway
    ['longitude']
    '''
    try:
        new_lat, new_lon = runway_snap(runway,
                                   runway['localizer']['latitude'],
                                   runway['localizer']['longitude'])
    except KeyError:
        new_lat, new_lon = runway['end']['latitude'], runway['end']['longitude']
        logger.warning('Localizer not found for this runway, so endpoint substituted')

    return {'latitude':new_lat, 'longitude':new_lon}


def integrate(array, frequency, initial_value=0.0, scale=1.0,
              direction="forwards", contiguous=False, extend=False,
              repair=False):
    """
    Trapezoidal integration

    Usage example:

    feet_to_land = integrate(airspeed[:touchdown], scale=ut.multiplier(ut.KT, ut.FPS), direction='reverse')

    :param array: Integrand.
    :type array: Numpy masked array.
    :param frequency: Sample rate of the integrand.
    :type frequency: Float
    :param initial_value: Initial value for the integral
    :type initial_value: Float
    :param scale: Scaling factor, default = 1.0
    :type scale: float
    :param direction: Optional integration sense, default = 'forwards'
    :type direction: String - ['forwards', 'backwards', 'reverse']
    :param contiguous: Option to restrict the output to the single longest contiguous section of data
    :type contiguous: Logical
    :param extend: Option to extend by half intervals at either end of the array.
    :type extend: Logical
    :param repair: Option to repair mask before integration.
    :type repair: Logical

    Notes: Reverse integration does not include a change of sign, so positive
    values have a negative slope following integration using this function.
    Backwards integration DOES include a change of sign, so positive
    values have a positive slope following integration using this function.

    Normal integration over n points will result in n-1 trapezoidal intervals
    being summed. This can be extended to provide n intervals by extending
    the first values by an integration step if required. The effect is to
    make the initial value the preceding value to the integral.

    :returns integral: Result of integration by time
    :type integral: Numpy masked array.
    """

    if np.ma.count(array)==0:
        return np_ma_masked_zeros_like(array)

    if repair:
        integrand = repair_mask(array,
                                repair_duration=None,
                                raise_entirely_masked=False,
                                extrapolate=True,
                                copy=True)
    elif contiguous:
        blocks = np.ma.clump_unmasked(array)
        longest_index = None
        longest_slice = 0
        for n, block in enumerate(blocks):
            slice_length = block.stop-block.start
            if slice_length > longest_slice:
                longest_slice = slice_length
                longest_index = n
        integrand = np_ma_masked_zeros_like(array)
        integrand[blocks[longest_index]] = array[blocks[longest_index]]
    else:
        integrand = array

    if direction.lower() == 'forwards':
        d = +1
        s = +1
    elif direction.lower() == 'reverse':
        d = -1
        s = +1
    elif direction.lower() == 'backwards':
        d = -1
        s = -1
    else:
        raise ValueError("Invalid direction '%s'" % direction)

    k = (scale * 0.5)/frequency
    to_int = k * (integrand + np.roll(integrand, d))
    edges = np.ma.flatnotmasked_edges(to_int)
    # In some cases to_int and the rolled version may result in a completely masked result.
    if edges is None:
        return np_ma_masked_zeros_like(array)

    if direction == 'forwards':
        if edges[0] == 1:
            to_int[0] = initial_value
        else:
            to_int[edges[0]] = initial_value
    else:
        if edges[1] == -1:
            to_int[-1] = initial_value * s
        else:
            to_int[edges[1]] = initial_value * s
            # Note: Sign of initial value will be reversed twice for backwards case.

    result=np.ma.zeros(len(integrand))

    result[::d] = np.ma.cumsum(to_int[::d] * s)


    # Original version used this half sample shifted result; never used.
    ##if extend:
        ##result += integrand[0]*s*k
        ##result[-1] += integrand[-1]*s*k

    if extend:
        first_value = integrand.compressed()[0]
        result += first_value * 2. * s * k

    return result

def integ_value(array,
                _slice=slice(None),
                start_edge=None,
                stop_edge=None,
                frequency=1.0,
                scale=1.0):
    """
    Get the integral value in the array and its index.

    :param array: masked array
    :type array: np.ma.array
    :param _slice: Slice to apply to the array and return min value relative to
    :type _slice: slice
    :param start_edge: Index for precise start timing
    :type start_edge: Float, between _slice.start-1 and slice_start
    :param stop_edge: Index for precise end timing
    :type stop_edge: Float, between _slice.stop and slice_stop+1

    :returns: Value named tuple of index and value.
    """
    if stop_edge:
        index = stop_edge
    elif _slice.stop:
        index = _slice.stop - 1
    else:
        index = len(array) - 1

    try:
        value = integrate(array[slices_int(_slice)],
                          frequency=frequency,
                          scale=scale,
                          repair=True,
                          extend=True)[-1]
    except IndexError:
        # Arises from _slice outside array boundary.
        index = None
        value = None
    return Value(index, value)

def interpolate(array, extrapolate=True):
    """
    This will replace all masked values in an array with linearly
    interpolated values between unmasked point pairs, and extrapolate first
    and last unmasked values to the ends of the array by default.

    See Derived Parameter Node 'Magnetic Deviation' for the prime example of
    use.

    In the special case where all source data is masked, the algorithm
    returns an unmasked array of zeros.

    :param array: Array of data with masked values to be interpolated over.
    :type array: numpy masked array
    :param extrapolate: Option to extrapolate the first and last masked values
    :type extrapolate: Bool

    :returns interpolated: array of all valid data
    :type interpolated: Numpy masked array, with all masks False.
    """
    # Where do we need to use the raw data?
    blocks = np.ma.clump_masked(array)
    last = len(array)
    if len(blocks)==1:
        if blocks[0].start == 0 and blocks[0].stop == last:
            logger.warning('No unmasked data to interpolate')
            return np_ma_zeros_like(array)

    for block in blocks:
        # Setup local variables
        a = block.start
        b = block.stop

        if a == 0:
            if extrapolate:
                array[:b] = array[b]
            else:
                # leave masked values at start untouched
                continue
        elif b == last:
            if extrapolate:
                array[a:] = array[a-1]
            else:
                # leave masked values at end untouched
                continue
        else:
            join = np.linspace(array[a - 1], array[b], num=b - a + 2)
            array[a:b] = join[1:-1]

    return array


def interpolate_coarse(array):
    '''
    Interpolate a coarse array which changes infrequently, e.g.
    [56, 56, 56, 57] -> [56, 56.333, 56.666, 57]

    :param array: Coarse array.
    :type array: numpy masked array
    :returns: Array interpolated between changing values.
    :rtype: numpy masked array
    '''
    array = array.copy()
    array.mask[1:] = np.ma.masked_equal(np.ma.diff(array.data), 0).mask | array.mask[1:]
    # Q: Use interpolate function instead?
    return repair_mask(array, repair_duration=None)


def interleave(param_1, param_2):
    """
    Interleaves two parameters (usually from different sources) into one
    masked array. Maintains the mask of each parameter.

    :param param_1:
    :type param_1: Parameter object
    :param param_2:
    :type param_2: Parameter object

    """
    # Check the conditions for merging are met
    if param_1.frequency != param_2.frequency:
        raise ValueError('Attempt to interleave parameters at differing sample '
                         'rates')

    dt = param_2.offset - param_1.offset
    # Note that dt may suffer from rounding errors,
    # hence rounding the value before comparison.
    if 2 * abs(round(dt, 6)) != 1 / param_1.frequency:
                raise ValueError('Attempt to interleave parameters that are '
                                 'not correctly aligned')

    merged_array = np.ma.zeros((2, len(param_1.array)))
    if dt > 0:
        merged_array = np.ma.column_stack((param_1.array, param_2.array))
    else:
        merged_array = np.ma.column_stack((param_2.array, param_1.array))

    return np.ma.ravel(merged_array)

"""
Superceded by blend routines.

def interleave_uneven_spacing(param_1, param_2):
    '''
    This interleaves samples that are not quote equi-spaced.
       |--------dt---------|
       |   x             y |
       |          m        |
       |   |------a------| |
       |     o         o   |
       |   |b|         |b| |

    Over a period dt two samples x & y will be merged to an equi-spaced new
    parameter "o". x & y are a apart, while samples o are displaced by b from
    their original positions.

    There is a second case where the samples are close together and the
    interpolation takes place not between x > y, but across the y > x interval.
    Hence two sections of code. Also, we don't know at the start whether x is
    parameter 1 or 2, so there are two options for the basic interleaving stage.
    '''
    # Check the conditions for merging are met
    if param_1.frequency != param_2.frequency:
        raise ValueError('Attempt to interleave parameters at differing sample rates')

    mean_offset = (param_2.offset + param_1.offset) / 2.0
    #result_offset = mean_offset - 1.0/(2.0 * param_1.frequency)
    dt = 1.0/param_1.frequency

    merged_array = np.ma.zeros((2, len(param_1.array)))

    if mean_offset - dt > 0:
        # The larger gap is between the two first samples
        merged_array = np.ma.column_stack((param_1.array,param_2.array))
        offset_0 = param_1.offset
        offset_1 = param_2.offset
        a = offset_1 - offset_0
    else:
        # The larger gap is between the second and third samples
        merged_array = np.ma.column_stack((param_2.array,param_1.array))
        offset_0 = param_2.offset
        offset_1 = param_1.offset
        a = dt - (offset_1 - offset_0)
    b = (dt - a)/2.0

    straight_array = np.ma.ravel(merged_array)
    if a < dt:
        straight_array[0] = straight_array[1] # Extrapolate a little at start
        x = straight_array[1::2]
        y = straight_array[2::2]
    else:
        x = straight_array[0::2]
        y = straight_array[1::2]
    # THIS WON'T WORK !!!
    x = (y - x)*(b/a) + x
    y = (y-x) * (1.0 - b) / a + x

    #return straight_array
    return None # to force a test error until this is fixed to prevent extrapolation
"""
"""
def interpolate_params(*params):
    '''
    Q: Should we mask indices which are being interpolated in masked areas of
       the input arrays.
    '''
    param_frequencies = [param.frequency for param in params]
    max_frequency = max(param_frequencies)
    out_frequency = sum(param_frequencies)

    data_arrays = []
    index_arrays = []

    for param in sorted(params, key=attrgetter('frequency')):
        multiplier = out_frequency / param.frequency
        offset = (param.offset * multiplier)
        # Will not create interpolation points for masked indices.
        unmasked_indices = np.where(~param.array.mask)[0]
        index_array = unmasked_indices.astype(np.float_) * multiplier + offset
        # Take only unmasked values to match size with index_array.
        data_arrays.append(param.array.data[unmasked_indices])
        index_arrays.append(index_array)
    # param assigned within loop has the maximum frequency.

    data_array = np.concatenate(data_arrays)
    index_array = np.concatenate(index_arrays)
    record = np.rec.fromarrays([index_array, data_array],
                               names='indices,values')
    record.sort()
    # Masked values will be NaN.
    interpolator = interp1d(record.indices, record.values, bounds_error=False,
                            fill_value=np.NaN)
    # Ensure first interpolated value is within range.
    out_offset = np.min(record.indices)
    out_indices = np.arange(out_offset, len(param.array) * multiplier,
                            param.frequency / out_frequency)
    interpolated_array = interpolator(out_indices)
    masked_array = np.ma.masked_array(interpolated_array,
                                      mask=np.isnan(interpolated_array))
    return masked_array, out_frequency, out_offset
"""


def index_of_datetime(start_datetime, index_datetime, frequency, offset=0):
    '''
    :param start_datetime: Start datetime of data file.
    :type start_datetime: datetime
    :param index_datetime: Datetime of which to calculate the index.
    :type index_datetime: datetime
    :param frequency: Frequency of index.
    :type frequency: float or int
    :param offset: Optional offset of the parameter in seconds.
    :type offset: float
    :returns: The index of index_datetime relative to start_datetime and frequency.
    :rtype: int or float
    '''
    difference = index_datetime - start_datetime
    return (difference.total_seconds() - offset) * frequency


def pairwise(iterable):
    '''
    https://docs.python.org/2/library/itertools.html#recipes
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    '''
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def is_index_within_slice(index, _slice):
    '''
    :type index: int or float
    :type _slice: slice
    :returns: whether index is within the slice.
    :rtype: bool
    '''
    negative_step = _slice.step is not None and _slice.step < 0
    if _slice.start is None and _slice.stop is None:
        return True
    elif _slice.start is None:
        if negative_step:
            return index > _slice.stop
        else:
            return index < _slice.stop
    elif _slice.stop is None:
        if negative_step:
            return index <= _slice.start
        else:
            return index >= _slice.start
    if negative_step:
        return _slice.start >= index > _slice.stop
    else:
        return _slice.start <= index < _slice.stop


def are_indexes_within_slices(indexes, slices):
    '''
    :type indexes: list of int or float
    :type slices: slice list
    :returns: whether any of the indexes is within any of the slices.
    :rtype: bool
    '''
    for _slice in slices:
        for index in indexes:
            if is_index_within_slice(index, _slice):
                return True
    return False


def is_index_within_slices(index, slices):
    '''
    :type index: int or float
    :type slices: slice
    :returns: whether index is within any of the slices.
    :rtype: bool
    '''
    for _slice in slices:
        if is_index_within_slice(index, _slice):
            return True
    return False


def filter_slices_length(slices, length):
    '''
    :param slices: List of slices to filter.
    :type slices: [slice]
    :param length: Minimum length of slices.
    :type length: int or float
    :rtype: [slice]
    '''
    return [s for s in slices if (s.stop - s.start) >= length]


def filter_slices_duration(slices, duration, frequency=1):
    '''
    Q: Does this need to be updated to use Sections?
    :param slices: List of slices to filter.
    :type slices: [slice]
    :param duration: Minimum duration of slices in seconds.
    :type duration: int or float
    :param frequency: Frequency of slice start and stop.
    :type frequency: int or float
    :returns: List of slices greater than duration.
    :rtype: [slice]
    '''
    return filter_slices_length(slices, duration * frequency)


def find_slices_containing_index(index, slices):
    '''
    :type index: int or float
    :type slices: a list of slices to search through

    :returns: the first slice which contains index or None
    :rtype: [slice]
    '''
    return [s for s in slices if is_index_within_slice(index, s)]


def find_nearest_slice(index, slices):
    '''
    :type index: int or float
    :type slices: a list of slices to search through

    :returns: the first slice which contains index or None
    :rtype: [slice]
    '''
    if len(slices) == 0:
        return None

    if len(slices) == 1:
        return slices[0]

    slice_start_diffs = []
    slice_stop_diffs = []
    for _slice in slices:
        slice_start_diffs.append(abs(index - _slice.start))
        slice_stop_diffs.append(abs(index - _slice.stop))
    min_slice_start_diff = min(slice_start_diffs)
    min_slice_stop_diff = min(slice_stop_diffs)
    min_diff = min(min_slice_start_diff, min_slice_stop_diff)
    try:
        index = slice_start_diffs.index(min_diff)
    except ValueError:
        index = slice_stop_diffs.index(min_diff)
    return slices[index]


def is_slice_within_slice(inner_slice, outer_slice, within_use='slice'):
    '''
    inner_slice is considered to not be within outer slice if its start or
    stop is None.

    :type inner_slice: slice
    :type outer_slice: slice
    :returns: Whether inner_slice is within the outer_slice.
    :rtype: bool
    '''

    def entire_slice_within_slice():
        if outer_slice.start is None and outer_slice.stop is None:
            return True
        elif inner_slice.start is None and outer_slice.start is not None:
            return False
        elif inner_slice.stop is None and outer_slice.stop is not None:
            return False
        elif inner_slice.start is None and outer_slice.start is None:
            return inner_slice.stop < outer_slice.stop
        elif outer_slice.stop is None and outer_slice.stop is None:
            return inner_slice.start >= outer_slice.start
        else:
            start_within = outer_slice.start <= inner_slice.start <= outer_slice.stop
            stop_within = outer_slice.start <= inner_slice.stop <= outer_slice.stop
            return start_within and stop_within

    if within_use == 'slice':
        return entire_slice_within_slice()
    elif within_use == 'start':
        return is_index_within_slice(inner_slice.start, outer_slice)
    elif within_use == 'stop':
        return is_index_within_slice(inner_slice.stop, outer_slice)
    elif within_use == 'any':
        return slices_overlap(inner_slice, outer_slice)


def find_slices_overlap(first_slice, second_slice):
    '''
    Return slice representing common elements of slice1 and slice2.

    :param slice1: First slice
    :type slice1: Python slice
    :param slice2: Second slice
    :type slice2: Python slice

    :returns slice
    '''
    def slice_step(sl):
        return sl.step if sl.step is not None else 1

    if first_slice.step is not None and first_slice.step < 1 \
       or second_slice.step is not None and second_slice.step < 1:
        raise ValueError("Negative step is not supported")

    if slice_step(first_slice) != slice_step(second_slice):
        raise ValueError("Use the same step to find slice overlap")
    else:
        step = first_slice.step

    if second_slice.start is None or \
       (first_slice.start > second_slice.start if first_slice.start else False):
        start = first_slice.start
    else:
        start = second_slice.start

    if second_slice.stop is None or first_slice.stop < second_slice.stop:
        stop = first_slice.stop
    else:
        stop = second_slice.stop

    if start < stop:
        return slice(start, stop, step)
    else:
        return None


def slices_overlap(first_slice, second_slice):
    '''
    Logical check for an overlap existing between two slices.
    Requires more than one value overlapping

    :param slice1: First slice
    :type slice1: Python slice
    :param slice2: Second slice
    :type slice2: Python slice

    :returns boolean
    '''
    if first_slice.step is not None and first_slice.step < 1 \
       or second_slice.step is not None and second_slice.step < 1:
        raise ValueError("Negative step not supported")
    return ((second_slice.stop is None) or ((first_slice.start or 0) < second_slice.stop)) and \
           ((first_slice.stop is None) or ((second_slice.start or 0) < first_slice.stop))


def slices_overlap_merge(first_list, second_list, extend_start=0, extend_stop=0, limit_start=None, limit_end=None):
    '''
    Where slices from the second list overlap the first, the first slice is
    extended to the limit of the second list slice.
    If there is no overlap, the first slice is returned unchanged.

    :param first_list: First list of slices
    :type first_list: List of slices
    :param second_list: Second list of slices
    :type second_list: List of slices
    :param extend_start: Increment at start of the resulting slices_above
    :type extend_stop: Integer
    :param extend_stop: Increment at stop end of the resulting slices_above
    :type extend_stop: Integer
    :param limit_start: Maximum length of overlap to merge into at start
    :type limit_start: Integer
    :param limit_stop: Maximum length of overlap to merge into at end
    :type limit_stop: Integer
    '''
    result_list = []

    for first_slice in first_list:
        overlap = False
        for second_slice in second_list:
            if slices_overlap(first_slice, second_slice):
                overlap = True
                begin = max(min(first_slice.start, second_slice.start) - extend_start, 0)
                if limit_start:
                    begin = max(begin, first_slice.start - limit_start)
                end = max(first_slice.stop, second_slice.stop) + extend_stop
                if limit_end:
                    end = min(end, first_slice.stop + limit_end)
                result_list.append(slice(begin, end))
                break

        if not overlap:
            if not extend_start and not extend_stop:
                result_list.append(first_slice)
            else:
                result_list.append(slice(first_slice.start - extend_start,
                                         first_slice.stop + extend_stop))

    return result_list


def slices_and(first_list, second_list):
    '''
    This is a simple AND function to allow two slice lists to be merged. This
    function accepts reverse sequence input slices, but the output is always
    forward ordered.

    :param first_list: First list of slices
    :type first_list: List of slices
    :param second_list: Second list of slices
    :type second_list: List of slices

    :returns: List of slices where first and second lists overlap.
    '''
    def fwd(_slice):
        if (_slice.step is not None and _slice.step < 0):
            return slice(_slice.stop+1, max(_slice.start+1,0), -_slice.step)
        else:
            return _slice

    result_list = []
    for first_slice in first_list:
        for second_slice in second_list:
            slice_1 = fwd(first_slice)
            slice_2 = fwd(second_slice)

            if slices_overlap(slice_1, slice_2):
                slice_start = max((s.start for s in (slice_1, slice_2)
                                   if s.start is not None))
                if slice_1.stop is None:
                    slice_stop = slice_2.stop
                elif slice_2.stop is None:
                    slice_stop = slice_1.stop
                else:
                    slice_stop = min(slice_1.stop, slice_2.stop)
                result_list.append(slice(slice_start,slice_stop))
    return result_list


def slices_and_not(first, second):
    '''
    It is surprisingly common to need one condition but not a second.
    Airborne but not Approach And Landing, for example. This little routine
    makes this simple.

    :param first: First list of slices- values to be included
    :type first: [slice]
    :param second: Second list of slices- values to be excluded
    :type second: [slice]

    :returns: List of slices in the first but outside the second lists.
    '''
    if not first:
        return []

    return slices_and(first,
                      slices_not(second,
                                 begin_at=min([s.start for s in first]),
                                 end_at=max([s.stop for s in first])))


def slices_not(slice_list, begin_at=None, end_at=None):
    '''
    Inversion of a list of slices. Currently does not cater for reverse slices.

    :param slice_list: list of slices to be inverted.
    :type slice_list: list of Python slices.
    :param begin_at: optional starting index value, slices before this will be ignored
    :param begin_at: integer
    :param end_at: optional ending index value, slices after this will be ignored
    :param end_at: integer

    :returns: list of slices. If begin or end is specified, the range will extend to these points. Otherwise the scope is within the end slices.
    '''
    if not slice_list:
        return [slice(begin_at, end_at)]

    start_slices = [s.start for s in slice_list]
    a = None if None in start_slices else min(start_slices)

    stop_slices  = [s.stop for s in slice_list]
    b =  None if None in stop_slices else min(stop_slices)

    c = max([s.step or 1 for s in slice_list])
    if c>1:
        raise ValueError("slices_not does not cater for non-unity steps")

    try:
        startpoint = a if b is None else min(a,b)
    except TypeError:
        startpoint = None

    if begin_at is not None \
       and startpoint is not None \
       and begin_at < startpoint:
        startpoint = begin_at
    if startpoint is None:
        startpoint = 0

    starts = [s for s in start_slices if s is not None]
    c = max(starts) if starts else None

    stops = [s for s in stop_slices if s]
    d = max(stops) if stops else None

    endpoint = max(c,d) if c and d else c or d

    try:
        endpoint = end_at if end_at > endpoint else endpoint
    except TypeError:
        endpoint = end_at or endpoint

    workspace = np.ma.zeros(int(endpoint))
    for each_slice in slice_list:
        workspace[slices_int(each_slice)] = 1
    workspace=np.ma.masked_equal(workspace, 1)

    return shift_slices(
        np.ma.clump_unmasked(workspace[slices_int(startpoint,endpoint)]),
        startpoint
    )


def slices_or(*slice_lists):
    '''
    Logical OR function for lists of slices.

    :param slice_lists: Lists of slices to be combined.
    :type slice_lists: [[slice]]
    :returns: List of slices combined.
    :rtype: list
    '''
    slices = []
    if all(len(s) == 0 or s == [None] for s in slice_lists):
        return slices

    recheck = False
    for slice_list in slice_lists:
        for input_slice in slice_list:
            if input_slice is None:
                continue
            modified = False
            for output_slice in copy(slices):
                if slices_overlap(input_slice, output_slice):
                    if input_slice.start is None or output_slice.start is None:
                        slice_start = None
                    else:
                        slice_start = min(input_slice.start, output_slice.start)
                    if input_slice.stop is None or output_slice.stop is None:
                        slice_stop = None
                    else:
                        slice_stop = max(input_slice.stop, output_slice.stop)

                    input_slice = slice(slice_start, slice_stop)
                    try:
                        slices.remove(output_slice)
                    except:
                        # This has already been handled at a lower level, so quit.
                        break
                    slices.append(input_slice)
                    modified = True
                    recheck = True

            if not modified:
                slices.append(input_slice)

        if recheck:
            # The new slice may overlap two, so repeat the process.
            slices = slices_or(slices)

    return slices


def slices_remove_overlaps(slices):
    '''
    removes overlapping slices from list, keeps longest slice.
    '''
    result = []
    for s in sorted(slices, key=lambda s: slice_duration(s, 1), reverse=True):
        if not any(slices_overlap(s, r) for r in result):
            result.append(s)
    return result


def slices_remove_small_gaps(slice_list, time_limit=10, hz=1, count=None):
    '''
    Routine to remove small gaps in a list of slices. Typically when a list
    of flight phases have been computed and we don't want to drop out for
    trivial periods, this will create a single slice across what were two
    slices with a small gap.

    :param slice_list: list of slices to be processed
    :type slice_list: list of Python slices.
    :param time_limit: Tolerance below which slices will be joined.
    :type time_limit: integer (sec)
    :param hz: sample rate for the parameter
    :type hz: float

    :param count: Tolerance based on count, not time
    :type count: integer (default = None)

    :returns: slice list.
    '''
    if slice_list is None or len(slice_list) < 2:
        return slice_list
    elif any(s.start is None for s in slice_list) and any(s.stop is None for s in slice_list):
        return [slice(None, None, slice_list[0].step)]

    sample_limit = count if count is not None else time_limit * hz
    slice_list = sorted(slice_list, key=lambda s: s.start or 0)
    new_list = [slice_list[0]]
    for each_slice in slice_list[1:]:
        if each_slice.start and new_list[-1].stop  and \
           each_slice.start - new_list[-1].stop < sample_limit:
            new_list[-1] = slice(new_list[-1].start, each_slice.stop)
        else:
            new_list.append(each_slice)
    return new_list


def slices_remove_small_slices(slices, time_limit=10, hz=1, count=None):
    '''
    Routine to remove small slices in a list of slices.

    :param slice_list: list of slices to be processed
    :type slice_list: list of Python slices.

    :param time_limit: Tolerance below which slice will be rejected.
    :type time_limit: integer (sec)
    :param hz: sample rate for the parameter
    :type hz: float

    :param count: Tolerance based on count, not time
    :type count: integer (default = None)

    :returns: slice list.
    '''
    if slices is None or slices == []:
        return slices
    sample_limit = count if count is not None else time_limit * hz
    return [s for s in slices if s.stop - s.start > sample_limit]


def slices_find_small_slices(slices, time_limit=10, hz=1, count=None):
    '''
    Find small slices in a list of slices.

    :param slice_list: list of slices to be processed
    :type slice_list: list of Python slices.

    :param time_limit: Tolerance below which slice will be rejected.
    :type time_limit: integer (sec)
    :param hz: sample rate for the parameter
    :type hz: float

    :param count: Tolerance based on count, not time
    :type count: integer (default = None)

    :returns: slice list.
    '''
    if slices is None or slices == []:
        return slices
    sample_limit = count if count is not None else time_limit * hz
    return [s for s in slices if s.stop - s.start < sample_limit]


def trim_slices(slices, seconds, frequency, hdf_duration):
    '''
    Trims slices by a number of seconds and excludes slices which are too small
    after trimming. Does not work with reverse slices.

    :param slices: Slices to trim.
    :type slices: [slice]
    :param seconds: Seconds to trim.
    :type seconds: int or float
    :param frequency: Frequency of slice indices.
    :type frequency: int or float
    :param hdf_duration: Duration of data within the HDF file in seconds
    :type hdf_duration: int
    :returns: Trimmed slices.
    :rtype: list
    '''
    trim_duration = seconds * frequency
    trimmed_slices = []
    for _slice in slices:
        trimmed_slice = slice((_slice.start or 0) + trim_duration,
                              (_slice.stop or hdf_duration) - trim_duration)
        if slice_duration(trimmed_slice, frequency) <= 0:
            continue
        trimmed_slices.append(trimmed_slice)
    return trimmed_slices


def valid_slices_within_array(array, sections=None):
    '''
    returns slices of unmasked data, optionally within section slices.
    '''
    array_band = mask_outside_slices(array, [x.slice for x in sections])
    return np.ma.clump_unmasked(array_band)


"""
def section_contains_kti(section, kti):
    '''
    Often want to check that a KTI value is inside a given slice.
    '''
    if len(kti)!=1 or len(section)!=2:
        return False
    return section.slice.start <= kti[0].index <= section.slice.stop
"""


def latitudes_and_longitudes(bearings, distances, reference):
    """
    Returns the latitudes and longitudes of a track given true bearing and
    distances with respect to a fixed point.

    Usage:
    lat[], lon[] = latitudes_and_longitudes(brg[], dist[],
                   {'latitude':lat_ref, 'longitude', lon_ref})

    :param bearings: The bearings of the track in degrees.
    :type bearings: Numpy masked array.
    :param distances: The distances of the track in metres.
    :type distances: Numpy masked array.
    :param reference: The location of the reference point in degrees.
    :type reference: dict with {'latitude': lat, 'longitude': lon} in degrees.

    :returns latitude, longitude: Latitudes and Longitudes in degrees.
    :type latitude, longitude: Two Numpy masked arrays

    Navigation formulae have been derived from the scripts at
    http://www.movable-type.co.uk/scripts/latlong.html
    Copyright 2002-2011 Chris Veness, and altered by Flight Data Services to
    suit the POLARIS project.
    """
    lat_ref = radians(reference['latitude'])
    lon_ref = radians(reference['longitude'])
    brg = bearings * deg2rad
    dist = distances.data / 6371000.0 # Scale to earth radius in metres

    lat = np.arcsin(sin(lat_ref)*np.ma.cos(dist) +
                   cos(lat_ref)*np.ma.sin(dist)*np.ma.cos(brg))
    lon = np.arctan2(np.ma.sin(brg)*np.ma.sin(dist)*np.ma.cos(lat_ref),
                      np.ma.cos(dist)-sin(lat_ref)*np.ma.sin(lat))
    lon += lon_ref

    joined_mask = np.logical_or(bearings.mask, distances.mask)
    lat_array = np.ma.array(data = np.rad2deg(lat),mask = joined_mask)
    lon_array = np.ma.array(data = np.rad2deg(lon),mask = joined_mask)
    return lat_array, lon_array


def localizer_scale(runway):
    """
    Compute the ILS localizer scaling factor from runway or nominal data.
    """
    try:
        # Compute the localizer scale factor (degrees per dot)
        # Half the beam width is 2.5 dots full scale
        scale = (runway['runway']['localizer']['beam_width']/2.0) / 2.5
    except:
        try:
            length = runway_length(runway)
        except:
            length = None

        if length is None:
            length = ut.convert(8000, ut.FT, ut.METER)  # Typical length

        # Normal scaling of a localizer gives 700ft width at the threshold,
        # so half of this is 350ft=106.68m. This appears to be full 2-dots
        # scale.
        scale = np.degrees(np.arctan2(106.68, length)) / 2.0
    return scale


def mask_inside_slices(array, slices):
    '''
    Mask slices within array.

    :param array: Masked array to mask.
    :type array: np.ma.masked_array
    :param slices: Slices to mask.
    :type slices: list of slice
    :returns: Array with masks applied.
    :rtype: np.ma.masked_array
    '''
    mask = np.zeros(len(array), dtype=np.bool_) # Create a mask of False.
    for slice_ in slices:
        start_ = int(math.floor(slice_.start or 0))
        if slice_.stop:
            stop_ = int(math.ceil(slice_.stop) + 1)
        else:
            stop_ = None
        mask[slice(start_, stop_)] = True
        # Original version below did not operate correctly with floating point slice ends. Amended temporarily until more comprehensive solution available.
        # Was: mask[slice_] = True
    return np.ma.array(array, mask=np.ma.mask_or(mask, array.mask))


def mask_outside_slices(array, slices):
    '''
    Mask areas outside of slices within array.

    :param array: Masked array to mask.
    :type array: np.ma.masked_array
    :param slices: The areas outside these slices will be masked..
    :type slices: list of slice
    :returns: Array with masks applied.
    :rtype: np.ma.masked_array
    '''
    mask = np.ones(len(array), dtype=np.bool_) # Create a mask of True.
    for slice_ in slices:
        start_ = int(math.ceil(slice_.start or 0))
        stop_ = int(math.floor(slice_.stop or -1))
        mask[slice(start_, stop_)] = False
        # Original version below did not operate correctly with floating point slice ends. Amended temporarily until more comprehensive solution available.
        # Was: mask[slice_] = False
    return np.ma.array(array, mask=np.ma.mask_or(mask, array.mask))


def mask_edges(mask):
    '''
    Return a mask with sections masked only at the beginning and end of the
    data.

    :type mask: np.array
    :returns: Mask with only the edges masked.
    :rtype: np.array
    '''
    if isinstance(mask, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(mask)

    edge_mask = np.zeros_like(mask, dtype=np.bool)

    if not len(mask) or mask.all() or (not mask[0] and not mask[-1]):
        return edge_mask

    masked_slices = runs_of_ones(mask)

    if masked_slices[0].start == 0:
        edge_mask[masked_slices[0]] = True

    if masked_slices[-1].stop == len(mask):
        edge_mask[masked_slices[-1]] = True

    return edge_mask


def mask_isolated_valid_samples(array):
    '''
    Mask single isolated valid samples surrounded by masked data.

    The operation is done in place, ie. it will mutate the array.

    :param array: Masked array to mask.
    :type array: np.ma.MaskedArray
    :returns: Masked array with single valid samples masked.
    :rtype: np.ma.MaskedArray
    '''
    m = np.ma.getmask(array)
    if m is not np.ma.nomask:
        # Find valid sample surrounded by invalid samples
        # mask would be True, False, True
        short_spike = (m[2:] & m[:-2]) & ~m[1:-1]
        array[1:-1][short_spike] = np.ma.masked
    return array


def max_continuous_unmasked(array, _slice=slice(None)):
    """
    Returns the max_slice
    """
    if _slice.step and _slice.step != 1:
        raise ValueError("Step not supported")
    clumps = np.ma.clump_unmasked(array[slices_int(_slice)])
    if not clumps or clumps == [slice(0,0,None)]:
        return None

    _max = None
    for clump in clumps:
        dur = clump.stop - clump.start
        if not _max or _max.stop-_max.start < dur:
            _max = clump
    offset = _slice.start or 0
    return slice(_max.start + offset, _max.stop + offset)


def max_abs_value(array, _slice=slice(None), start_edge=None, stop_edge=None):
    """
    Get the value of the maximum absolute value in the array.
    Return value is NOT the absolute value (i.e. may be negative)

    Note, if all values are masked, it will return the value at the first index
    (which will be masked!)

    :param array: masked array
    :type array: np.ma.array
    :param _slice: Slice to apply to the array and return max absolute value relative to
    :type _slice: slice
    :param start_edge: Index for precise start timing
    :type start_edge: Float, between _slice.start-1 and slice_start
    :param stop_edge: Index for precise end timing
    :type stop_edge: Float, between _slice.stop and slice_stop+1

    :returns: Value named tuple of index and value.
    """
    #slice_start = start_edge if start_edge is not None else _slice.start
    #slice_stop = stop_edge if stop_edge is not None else _slice.stop
    #max_val_slice = slice(slice_start, slice_stop, _slice.step)
    index, value = max_value(np.ma.abs(array), _slice,
                          start_edge=start_edge, stop_edge=stop_edge)

    if value is None:
        return Value(None, None)
    else:
        return Value(index, array[int(index)]) # Recover sign of the value.


def max_value(array, _slice=slice(None), start_edge=None, stop_edge=None):
    """
    Get the maximum value in the array and its index relative to the array and
    not the _slice argument.

    :param array: masked array
    :type array: np.ma.array
    :param _slice: Slice to apply to the array and return max value relative to
    :type _slice: slice
    :param start_edge: Index for precise start timing
    :type start_edge: Float, between _slice.start-1 and slice_start
    :param stop_edge: Index for precise end timing
    :type stop_edge: Float, between _slice.stop and slice_stop+1

    :returns: Value named tuple of index and value.
    """
    #slice_start = start_edge if start_edge is not None else _slice.start
    #slice_stop = stop_edge if stop_edge is not None else _slice.stop
    #max_val_slice = slice(slice_start, slice_stop, _slice.step)
    index, value = _value(array, _slice, np.ma.argmax,
                          start_edge=start_edge, stop_edge=stop_edge)

    return Value(index, value)


def median_value(array, _slice=None, start_edge=None, stop_edge=None):
    '''
    Calculate the median value within an optional slice of the array and return
    both the endpoint index and the median.

    :param array: Data to calculate the median value of.
    :type array: np.ma.masked_array
    :param _slice: Optional subsection of the data to calculate the average value within.
    :type _slice: slice
    :returns: The midpoint index and the average value.
    :rtype: Value named tuple of index and value.
    '''
    start = _slice.start or 0 if _slice else 0
    stop = _slice.stop or len(array) if _slice else len(array)
    midpoint = start + ((stop - start) // 2)
    if _slice:
        array = array[_slice]
    return Value(midpoint, np.ma.median(array))


def merge_masks(masks, min_unmasked=1):
    '''
    :type masks: [mask]
    :type min_unmasked: int
    :returns: Array of merged masks.
    :rtype: np.array(dtype=np.bool_)
    '''
    if len(masks) == 1:
        return masks[0]
    # Q: What if min_unmasked is less than one?
    mask_sum = np.sum(np.array(masks), axis=0)
    return mask_sum >= min_unmasked


def min_value(array, _slice=slice(None), start_edge=None, stop_edge=None):
    """
    Get the minimum value in the array and its index.

    :param array: masked array
    :type array: np.ma.array
    :param _slice: Slice to apply to the array and return min value relative to
    :type _slice: slice
    :param start_edge: Index for precise start timing
    :type start_edge: Float, between _slice.start-1 and slice_start
    :param stop_edge: Index for precise end timing
    :type stop_edge: Float, between _slice.stop and slice_stop+1

    :returns: Value named tuple of index and value.
    """
    #slice_start = start_edge if start_edge is not None else _slice.start
    #slice_stop = stop_edge if stop_edge is not None else _slice.stop
    #min_val_slice = slice(slice_start, slice_stop, _slice.step)
    index, value = _value(array, _slice, np.ma.argmin,
                          start_edge=start_edge, stop_edge=stop_edge)

    return Value(index, value)


def average_value(array, _slice=slice(None), start_edge=None, stop_edge=None):
    '''
    Calculate the average value within an optional slice of the array and return
    both the midpoint index and the average.

    :param array: Data to calculate the average value of.
    :type array: np.ma.masked_array
    :param _slice: Optional subsection of the data to calculate the average value within.
    :type _slice: slice
    :returns: The midpoint index and the average value.
    :rtype: Value named tuple of index and value.
    '''
    start = _slice.start or 0 if _slice else 0
    stop = _slice.stop or len(array) if _slice else len(array)
    midpoint = start + ((stop - start) // 2)
    if _slice:
        array = array[slices_int(_slice)]
    return Value(midpoint, np.ma.mean(array))


def std_dev_value(array, _slice=slice(None), start_edge=None, stop_edge=None):
    """
    Get the standard deviation in the array and its index.

    :param array: Data to calculate the standard deviation of.
    :type array: np.ma.masked_array
    :param _slice: Optional subsection of the data to calculate the standard deviation within.
    :type _slice: slice
    :returns: The midpoint index and the standard deviation.
    :rtype: Value named tuple of index and value.
    """
    start = _slice.start or 0 if _slice else 0
    stop = _slice.stop or len(array) if _slice else len(array)
    midpoint = start + ((stop - start) // 2)
    if _slice:
        array = array[_slice]
    return Value(midpoint, np.ma.std(array))


def minimum_unmasked(array1, array2):
    """
    Get the minimum value between two arrays. Differs from the Numpy minimum
    in that is there are masked values in one array, these are ignored and
    data from the other array is used.

    :param array_1: masked array
    :type array_1: np.ma.array
    :param array_2: masked array
    :type array_2: np.ma.array
    """
    a1_masked = np.ma.getmaskarray(array1)
    a2_masked = np.ma.getmaskarray(array2)
    neither_masked = np.logical_not(np.logical_or(a1_masked,a2_masked))
    one_masked = np.logical_xor(a1_masked,a2_masked)
    # Data for a1 is good when only one is masked and the mask is on a2.
    a1_good = np.logical_and(a2_masked, one_masked)

    return np.ma.where(neither_masked, np.ma.minimum(array1, array2),
                       np.ma.where(a1_good, array1, array2))


def merge_two_parameters(param_one, param_two):
    '''
    Use: merge_two_parameters is intended for discrete and multi-state
    parameters. Use blend_two_parameters for analogue parameters.

    This process merges two parameter objects. They must be recorded at the
    same frequency. They are interleaved without smoothing, and then the
    offset and frequency are computed as though the finished item was
    equispaced.

    If the two parameters are recorded less than half the sample interval
    apart, a value error is raised as the synthesized parameter cannot
    realistically be described by an equispaced result.

    :param param_one: Parameter object
    :type param_one: Parameter
    :param param_two: Parameter object
    :type param_two: Parameter

    :returns array, frequency, offset
    '''
    assert param_one.frequency  == param_two.frequency
    assert len(param_one.array) == len(param_two.array)

    delta = (param_one.offset - param_two.offset) * param_one.frequency
    off = (param_one.offset+param_two.offset-(1/(2.0*param_one.frequency)))/2.0
    if -0.75 < delta < -0.25:
        # merged array should be monotonic (always increasing in time)
        array = merge_sources(param_one.array, param_two.array)
        return array, param_one.frequency * 2, off
    elif 0.25 < delta < 0.75:
        array = merge_sources(param_two.array, param_one.array)
        return array, param_two.frequency * 2, off
    else:
        raise ValueError("merge_two_parameters called with offsets too similar. %s : %.4f and %s : %.4f" \
                         % (param_one.name, param_one.offset, param_two.name, param_two.offset))

def merge_sources(*arrays):
    '''
    This simple process merges the data from multiple sensors where they are
    sampled alternately. Unlike blend_alternate_sensors or the parameter
    level option blend_two_parameters, this procedure does not make any
    allowance for the two sensor readings being different.

    :param array: sampled data from an alternate signal source
    :type array: masked array
    :returns: masked array with merging algorithm applied.
    :rtype: masked array
    '''
    result = np.ma.empty((len(arrays[0]),len(arrays)))
    for dim, array in enumerate(arrays):
        result[:,dim] = array
    return np.ma.ravel(result)


def blend_equispaced_sensors(array_one, array_two):
    '''
    This process merges the data from two sensors where they are sampled
    alternately. Where one sensor is invalid, the process substitutes from
    the other sensor where possible, maintaining a higher level of data
    validity.

    :param array_one: sampled data from one signal source
    :type array_one: masked array
    :param array_two: sampled data from one signal source
    :type array_two: masked array
    :returns: masked array with merging algorithm applied.
    :rtype: masked array
    '''
    assert len(array_one) == len(array_two)
    both = merge_sources(array_one, array_two)
    both_mask = np.ma.getmaskarray(both)

    av_other = np_ma_masked_zeros_like(both)
    av_other[1:-1] = (both[:-2] + both[2:])/2.0
    av_other[0] = both[1]
    av_other[-1] = both[-2]
    av_other_mask = np.ma.getmaskarray(av_other)

    best = (both + av_other)/2.0
    best_mask = np.ma.getmaskarray(best)

    # We build up the best available data starting from the worst case, where
    # we have no valid data, so return a masked zero
    result = np_ma_masked_zeros_like(both)

    # If the other channel is valid, use the average of the before and after
    # samples of the other channel.
    result = np.ma.where(av_other_mask, result, av_other)

    # Better - if the channel sampled at the right moment is valid, use this.
    result = np.ma.where(both_mask, result, both)

    # Best option is this channel averaged with the mean of the other channel
    # before and after samples.
    result = np.ma.where(best_mask, result, best)

    return result


def blend_nonequispaced_sensors(array_one, array_two, padding):
    '''
    Where there are timing differences between the two samples, this
    averaging process computes the average value between alternate pairs of
    samples. This has the effect of removing sensor mismatch and providing
    equispaced data points. The disadvantage is that in the presence of one
    sensor malfunction, all resulting data is invalid.

    :param array_one: sampled data from one signal source
    :type array_one: masked array
    :param array_two: sampled data from one signal source
    :type array_two: masked array
    :param padding: where to put the padding value in the array
    :type padding: String "Precede" or "Follow"
    :returns: masked array with merging algorithm applied.
    :rtype: masked array
    '''
    assert len(array_one) == len(array_two)

    both = merge_sources(array_one, array_two)
    mask_array = np.ma.getmaskarray(both)
    first_mask = mask_array[0]
    last_mask = mask_array[-1]
    mask_array[0] = False
    mask_array[-1] = False

    mean_array = np.divide(np.roll(both, -1) + np.roll(both, 1), 2.0)
    both[mask_array] = mean_array[mask_array]
    if first_mask:
        both[0] = np.ma.masked
    if last_mask:
        both[-1] = np.ma.masked

    # A simpler technique than trying to append to the averaged array.
    av_pairs = np.ma.empty_like(both)
    if padding == 'Follow':
        av_pairs[:-1] = (both[:-1]+both[1:])/2
        av_pairs[-1] = av_pairs[-2]
        av_pairs[-1] = np.ma.masked
    else:
        av_pairs[1:] = (both[:-1]+both[1:])/2
        av_pairs[0] = av_pairs[1]
        av_pairs[0] = np.ma.masked
    return av_pairs

def blend_two_parameters(param_one, param_two, mode=None):
    '''
    Use: blend_two_parameters is intended for analogue parameters. Use
    merge_two_parameters for discrete and multi-state parameters.

    This process merges the data from two sensors where they are sampled
    alternately. Often pilot and co-pilot attitude and air data signals are
    stored in alternate locations to provide the required sample rate while
    allowing errors in either to be identified for investigation purposes.

    For FDM, only a single parameter is required, but mismatches in the two
    sensors can lead to, taking pitch attitude as an example, apparent "nodding"
    of the aircraft and errors in the derived pitch rate.

    This process merges two parameter arrays of the same frequency.
    Smoothes and then computes the offset and frequency appropriately.

    Two alternative processes are used, depending upon whether the samples
    are equispaced or not.

    :param param_one: Parameter object
    :type param_one: Parameter
    :param param_two: Parameter object
    :type param_two: Parameter
    :param mode: Function control for ILS merging; may be 'localizer' or 'glideslope'
    :type mode: string

    :returns array, frequency, offset
    :type array: Numpy masked array
    :type frequency: Float (Hz)
    :type offset: Float (sec)

    '''
    def _interp(a, freq, off):
        '''
        Simple linear interpolation to double sample rate when one of the
        arrays to merge is invariant or corrupt. This allows the subsequent
        analysis processes to proceed without failing due to differing sample
        rates.
        '''
        b = np.ma.empty(2 * a.size,)
        if off >= 0.5/freq:
            b[1::2] = a
            b[2::2] = (a[:-1]+a[1:]) / 2.0
            b[0] = b[1]
            off -= 0.5/freq
        else:
            b[0::2] = a
            b[1:-1:2] = (a[:-1]+a[1:]) / 2.0
            b[-1] = b[-2]
        return b, 2.0 * freq, off

    if param_one is None and param_two is None:
        raise ValueError('blend_two_parameters called with both parameters = None')
    if param_one is None:
        return _interp(param_two.array, param_two.frequency, param_two.offset)

    if param_two is None:
        return _interp(param_one.array, param_one.frequency, param_one.offset)

    assert param_one.frequency == param_two.frequency, \
        'The frequency of blended parameters must be the same: ' \
        '%s %sHz, %s %sHz' % (param_one.name, param_one.frequency,
                              param_two.name, param_two.frequency)

    # Parameters for blending should not be aligned.
    #assert param_one.offset != param_two.offset

    # A common problem is that one sensor may be unserviceable, and has been
    # identified already by parameter validity testing. Trap this case and
    # deal with it first, raising a warning and dropping back to the single
    # reliable source of information.
    # Note that for ILS sources, it is commonplace for this to arise, hence the
    # special mode options listed below.
    a = np.ma.count(param_one.array)
    b = np.ma.count(param_two.array)
    if not mode:
        if a+b == 0:
            logger.warning("Neither '%s' or '%s' has valid data available.",
                           param_one.name, param_two.name)
            # Return empty space of the right shape...
            return np_ma_masked_zeros_like(len(param_one.array)*2), 2.0*param_one.frequency, param_one.offset

        if a < b*0.8:
            logger.warning("Little valid data available for %s (%d valid samples), using %s (%d valid samples).", param_one.name, float(a)/len(param_one.array)*100, param_two.name, float(b)/len(param_two.array)*100)
            return _interp(param_two.array, param_two.frequency, param_two.offset)

        elif b < a*0.8:
            logger.warning("Little valid data available for %s (%d valid samples), using %s (%d valid samples).", param_two.name, float(b)/len(param_two.array)*100, param_one.name, float(a)/len(param_one.array)*100)
            return _interp(param_one.array, param_one.frequency, param_one.offset)

    # A second problem is where both sensor may appear to be serviceable but
    # one is invariant. If the parameters were similar, a/(a+b)=0.5 so we are
    # looking for one being less than 20% of its normal level.
    c = float(np.ma.ptp(param_one.array))
    d = float(np.ma.ptp(param_two.array))

    if c+d == 0.0:
        logger.warning("No variation in %s or %s, returning %s.", param_one.name, param_two.name, param_one.name)
        return _interp(param_one.array, param_one.frequency, param_one.offset)

    if c/(c+d) < 0.1:
        logger.warning("No variation in %s, using only %s.", param_one.name, param_two.name)
        return _interp(param_two.array, param_two.frequency, param_two.offset)

    elif d/(c+d) < 0.1:
        logger.warning("No variation in %s, using only %s.", param_two.name, param_one.name)
        return _interp(param_one.array, param_one.frequency, param_one.offset)


    frequency = param_one.frequency * 2.0

    # Special treatment of two-source ILS signals
    if mode:
        # The tolerance is set by:
        if mode == 'localizer':
            tol = ILS_LOC_SPREAD
        elif mode == 'glideslope':
            tol = ILS_GS_SPREAD
        else:
            raise ValueError('ILS mode not recognized')

        masked_array = np_ma_masked_zeros_like(param_one.array)

        temp = np.ma.where(np.ma.abs(param_one.array) < np.ma.abs(param_two.array) - tol,
                           x=masked_array ,
                           y=param_one.array)
        param_two.array = np.ma.where(np.ma.abs(param_two.array) < np.ma.abs(param_one.array) - tol,
                                      x=masked_array,
                                      y=param_two.array)

        param_one.array = temp

    # Are the parameters equispaced?
    # We force ILS signals through this path as the benefit of gaining the best signal
    # (i.e. producing a valid result if one singal is inoperative) outweighs
    # the timing errors involved.
    if (abs(param_one.offset - param_two.offset) * frequency == 1.0):
        # Equispaced process
        if param_one.offset < param_two.offset:
            offset = param_one.offset
            array = blend_equispaced_sensors(param_one.array, param_two.array)
        else:
            offset = param_two.offset
            array = blend_equispaced_sensors(param_two.array, param_one.array)

    else:
        # Non-equispaced process
        offset = (param_one.offset + param_two.offset)/2.0
        padding = 'Follow'

        if offset >= 1.0/frequency:
            offset = offset - 1.0/frequency
            padding = 'Precede'

        if param_one.offset <= param_two.offset:
            # merged array should be monotonic (always increasing in time)
            array = blend_nonequispaced_sensors(param_one.array, param_two.array, padding)
        else:
            array = blend_nonequispaced_sensors(param_two.array, param_one.array, padding)

    return array, frequency, offset


def blend_parameters(params, offset=0.0, frequency=1.0, small_slice_duration=4, mode='linear',
                     validity='any_one', tolerance=None):
    '''
    This most general form of the blend options allows for multiple sources
    to be blended together even though the spacing, validity and even sample
    rate may be different. Furthermore the offset and frequency of the output
    parameter can be selected if required.

    :param params: list of parameters to be merged, can be None if not available
    :type params: [Parameter]
    :param offset: the offset of the resulting parameter in seconds
    :type offset: float
    :param frequency: the frequency of the resulting parameter
    :type frequency: float
    :param small_slice_duration: duration of short slices to ignore in seconds (only applicable in mode cubic)
    :type small_slice_duration: float
    :param mode: type of interpolation to use - default 'linear' or 'cubic'. Anything else raises exception.
    :type mode: str
    :param validity: requirements for validity in the data - from ['all', 'all_but_one', 'any_one']
    :type validity: str
    :param tolerance: tolerance for merging to take place
    :type tolerance: float or None (for no test to be applied)
    '''

    assert frequency > 0.0
    assert mode in {'linear', 'cubic'}, 'blend_parameters mode must be either linear or cubic.'

    # accept as many params as required
    params = [p for p in params if p is not None]
    assert len(params), "No parameters to merge"

    # Find out about the parameters we have to deal with...
    min_ip_freq = min(p.frequency for p in params)

    if tolerance:
        test_array = np.ma.zeros((len(params), int(len(params[0].array) * min_ip_freq / params[0].frequency)))
        for n, p in enumerate(params):
            test_array[n, :] = resample(p.array, p.frequency, min_ip_freq)

    if mode == 'linear':
        return blend_parameters_linear(params, frequency, tolerance=tolerance, offset=offset)

    # mode is cubic

    # Find out about the parameters we have to deal with...
    min_ip_freq = min(p.frequency for p in params)

    p_valid_slices = []

    # Prepare a place for the output signal
    length = int(len(params[0].array) * frequency / params[0].frequency)
    result = np_ma_masked_zeros(length)
    # Ensure mask is expanded for slicing.
    result.mask = np.ma.getmaskarray(result)

    # Slices of valid data are scaled to the lowest timebase and then or'd
    # to find out when any valid data is available.
    for seq, param in enumerate(params):
        # We can only work on non-trivial slices which have four or more
        # samples as below this level it's not possible to compute a cubic
        # spline.
        nts=slices_remove_small_slices(np.ma.clump_unmasked(param.array),
                                       count=max(param.frequency * small_slice_duration, 4))
        nts=slices_remove_small_gaps(nts)
        if nts:
            param.array = repair_mask(mask_outside_slices(param.array, nts))

        # Now scale these non-trivial slices into the lowest timebase for
        # collation.
        p_valid_slices.append(slices_multiply(nts, min_ip_freq / param.frequency))

    all_masks = np.array([np.ma.getmaskarray(p.array)[::int(p.frequency // min_ip_freq)] for p in params])
    num_valid = len(params) - np.sum(all_masks, axis=0)

    if validity == 'all':
        bad = num_valid < len(params)
    elif validity == 'all_but_one':
        bad = num_valid < max(len(params) - 1, 1)
    elif validity == 'any_one':
        bad = num_valid == 0
    else:
        raise ValueError('Unrecognised validity mode in blend_parameters')

    any_valid = runs_of_ones(~bad)

    if any_valid is None:
        # No useful chunks of data to process, so give up now.
        return

    # Now we can work through each period of valid data.
    for this_valid in any_valid:
        result_slice = slice_multiply(this_valid, frequency/min_ip_freq)
        if result_slice.stop - result_slice.start < 1:
            # Less than 1 sample of valid data.
            continue
        result[result_slice] = blend_parameters_cubic(
            frequency, offset, params, result_slice, tolerance=tolerance)
        # The endpoints of a cubic spline are generally unreliable, so trim
        # them back.
        result[result_slice][0] = np.ma.masked
        result[result_slice][-1] = np.ma.masked
    return result


def resample_mask(mask, orig_hz, resample_hz):
    '''
    Resample a boolean array by simply repeating or stepping.
    TODO: Tests

    :param mask: mask array or scalar
    :type: np.array(dtype=bool)
    :param orig_hz: original frequency of the mask array
    :type orig_hz: int or float
    :param resample_hz: target frequency
    :type resample_hz: int or float
    :returns: resampled mask array
    :rtype: np.array(dtype=bool)
    '''
    if np.isscalar(mask) or resample_hz == orig_hz:
        return mask
    elif not is_power2_fraction(orig_hz) or not is_power2_fraction(orig_hz):
        raise ValueError("Both orig_hz '%f' and resample_hz '%f' must be powers of two.", orig_hz, resample_hz)

    if resample_hz > orig_hz:
        resampled = mask.repeat(resample_hz / orig_hz)
    elif resample_hz < orig_hz:
        resampled = mask[::orig_hz / resample_hz]

    return resampled


def blend_parameters_linear(params, frequency, tolerance=None, offset=0):
    '''
    This provides linear interpolation to support the generic routine
    blend_parameters.

    :param params: list of parameters to blend linearly
    :type params: [Parameter]
    :param frequency: target frequency in Hz
    :type frequency: int or float
    :param offset: target offset in seconds
    :type offset: int or float
    :returns: linearly blended parameter array
    :rtype: np.ma.array
    '''

     # Make space for the computed curves
    weights = []
    aligned = []

    # Find out about the parameters we have to deal with...
    min_ip_freq = min(p.frequency for p in params)

    # Compute the individual splines
    for param in params:
        aligned.append(align_args(param.array, param.frequency, param.offset, frequency, offset))
        # Remember the sample rate as this dictates the weighting
        weights.append(param.frequency)

    result = np.ma.average(aligned, axis=0, weights=weights)

    tol_mask = None
    if tolerance:
        test_array = np.ma.zeros((len(params), int(len(params[0].array) * min_ip_freq / params[0].frequency)))
        for n, p in enumerate(params):
            test_array[n, :] = resample(p.array, p.frequency, min_ip_freq)
        tol_mask = np.ma.masked_greater(np.ma.ptp(test_array, axis=0), tolerance)
        result.mask = np.ma.logical_or(np.ma.getmaskarray(result), np.ma.getmaskarray(tol_mask))

    return result


def blend_parameters_cubic(frequency, offset, params, result_slice, tolerance=None):
    '''
    :param frequency: the frequency of the output parameter
    :type frequency: float
    :param offset: the offset of the output parameter
    :type offset: float
    :param params: list of input parameters to be merged, can be None if not available
    :type params: [Parameter]
    :param result_slice: the slice where results will be returned
    :type result_slice: slice

    This provides cubic spline interpolation to support the generic routine
    blend_parameters. It uses cubic spline interpolation for each of the
    component parameters, then applies weighting to reflect both the
    frequency of samples of the parameter and it's mask. The multiple cubic
    splines are then summed at the points where new samples are required.

    To be used with care as this both gives a smoother transition at sample
    boundaries, but suffers from overswing which can cause problems.
    '''

    new_t = np.linspace(result_slice.start / frequency,
                        result_slice.stop / frequency,
                        num=(result_slice.stop - result_slice.start),
                        endpoint=False) + offset

    # Make space for the computed curves
    curves = []
    weights = []
    resampled_masks = []

    # Compute the individual splines
    for seq, param in enumerate(params):
        # The slice and timebase for this parameter...
        my_slice = slice_multiply(result_slice, param.frequency / frequency)
        resampled_masks.append(
            resample(np.ma.getmaskarray(param.array)[my_slice],
                     param.frequency, frequency))
        timebase = np.linspace(my_slice.start/param.frequency,
                               my_slice.stop/param.frequency,
                               num=my_slice.stop-my_slice.start,
                               endpoint=False) + param.offset
        my_time = np.ma.array(
            data=timebase, mask=np.ma.getmaskarray(param.array[my_slice]))
        if len(my_time.compressed()) < 4:
            continue
        my_curve = scipy_interpolate.splrep(
            my_time.compressed(), param.array[my_slice].compressed(), s=0)
        # my_curve is the spline knot array, now compute the values for
        # the output timebase.
        curves.append(
            scipy_interpolate.splev(new_t, my_curve, der=0, ext=0))

        # Compute the weights
        weights.append(blend_parameters_weighting(
            param.array[my_slice], frequency/param.frequency))

    if curves:
        a = np.vstack(tuple(curves))
        result = np.ma.average(a, axis=0, weights=weights)
        if tolerance:
            result.mask = np.ma.masked_greater(np.ma.ptp(a, axis=0), tolerance).mask
    else:
        result = np_ma_masked_zeros_like(new_t)

    return result

def blend_parameters_weighting(array, wt):
    '''
    A small function to relate masks to weights. Used by
    blend_parameters_cubic above.

    :param array: array to compute weights for
    :type array: numpy masked array
    :param wt: weighting factor =  ratio of sample rates
    :type wt: float
    '''
    mask = np.ma.getmaskarray(array)
    param_weight = (1.0 - mask)
    result_weight = np_ma_masked_zeros_like(np.ma.arange(floor(len(param_weight) * wt)))
    final_weight = np_ma_masked_zeros_like(np.ma.arange(floor(len(param_weight) * wt)))
    result_weight[0] = param_weight[0] / wt
    result_weight[-1] = param_weight[-1] / wt

    for i in range(1, len(param_weight) - 1):
        if param_weight[i] == 0.0:
            result_weight[int(i * wt)] = 0.0
            continue
        if param_weight[i - 1] == 0.0 or param_weight[i + 1] == 0.0:
            result_weight[int(i * wt)] = 0.1 # Low weight to tail of valid data. Non-zero to avoid problems of overlapping invalid sections.
            continue
        result_weight[int(i * wt)] = 1.0 / wt

    for i in range(1, len(result_weight) - 1):
        if result_weight[i-1]==0.0 or result_weight[i + 1] == 0.0:
            final_weight[i]=result_weight[i]/2.0
        else:
            final_weight[i]=result_weight[i]
    final_weight[0]=result_weight[0]
    final_weight[-1]=result_weight[-1]

    return repair_mask(final_weight, repair_duration=None)



def most_points_cost(coefs, x, y):
    '''
    This cost function computes a value which is minimal for points clost to
    a "best fit" line. It differs from normal least squares optimisation in
    that points a long way from the line have almost the same error as points
    a little way off the line.

    The function is used as a form of correlation function where we are
    looking to find the largest number of points on a certain line, with less
    regard to points that lie off that line.

    :param coefs: line coefficients, m and c, to be adjusted to minimise this cost function.
    :type coefs: list of floats, containing [m, c]
    :param x: independent variable
    :type x: numpy masked array
    :param y: dependent variable
    :type y: numpy masked array

    :returns: cost function; most negative value represents best fit.
    :type: float
    '''
    # Wrote "assert len(x) == len(y)" but can't find how to test this, so verbose equivalent is...
    if len(x) != len(y):
        raise ValueError('most_points_cost called with x & y of unequal length')
    if len(x) < 2:
        raise ValueError('most_points_cost called with inadequate samples')
    # Conventional y=mx+c equation for the "best fit" line
    m=coefs[0]
    c=coefs[1]

    # We compute the distance of each point from the line
    d = np.ma.sqrt((m*x+c-y)**2.0/(m**2.0+1))
    # and work out the maximum distance
    d_max = np.ma.max(d)
    if d_max == 0.0:
        raise ValueError('most_points_cost called with colinear data')
    # The error for each point is computed as a nonlinear function of the
    # distance, tailored to make points on the line give a small error, and
    # those away from the line progressively greater, but reaching a limit
    # value of 0 so that points at a great distance do not contribute more to
    # the weighted error.

    # width sets the width of the channel created by this function. Larger
    # values make the channel wider, but this opens up the function to
    # settling on minima away from the optimal line. Too narrow a width and,
    # again, the function can latch onto few points and determine a local
    # minimum. The value of 0.003 was chosen from analysis of fuel flow vs
    # altitude plots where periods of level flight in the climb create low
    # fuel readings which are not part of the climb performance we are trying
    # to detect. Values 3 times greater or smaller gave similar results,
    # while values 10 times greater or smaller led to erroneous results.
    width=0.003
    e = 1.0 -1.0/((d/d_max)**2 + width)
    mean_error = np.ma.sum(e)/len(d)
    return mean_error


def moving_average(array, window=9, weightings=None):
    """
    Moving average over an array with window of n samples. Weightings allows
    customisation of the importance of each position's value in the average.

    Requires odd lengthed moving windows so that the result is positioned
    centrally in the window offset.

    :param array: Masked Array
    :type array: np.ma.array
    :param window: Size of moving average window to use
    :type window: Integer
    :param weightings: Apply uneven weightings across the window - the same length as window.
    :type weightings: array-like object consisting of floats

    deprecated...
    :param pad: Pad the returned array to the same length of the input, using masked 0's
    :type pad: Boolean

    Ref: http://argandgahandapandpa.wordpress.com/2011/02/24/python-numpy-moving-average-for-data/
    """
    # Make sure we are working with a valid window size.
    assert window%2 == 1, 'Window %d is not an odd number' % window
    if len(array)==0:
        return None

    if weightings is None:
        weightings = np.repeat(1.0, window) / window
    elif len(weightings) != window:
        raise ValueError("weightings argument (len:%d) must equal window (len:%d)" % (
            len(weightings), window))

    # We repair the mask otherwise masked elements will expand to cover the
    # whole window dimension.
    copy_array = np.ma.copy(array) # To avoid corrupting the mask.
    repaired = repair_mask(copy_array ,
                           repair_duration=None,
                           raise_duration_exceedance=False,
                           extrapolate=True)
    stretch = int(window/2)
    stretched_data = np.ma.concatenate([[repaired[0]] * stretch,
                                        repaired,
                                        [repaired[-1]] * stretch])

    averaged = np.convolve(stretched_data.data, weightings, 'valid')
    result = np.ma.array(data=averaged,
                         mask=np.ma.getmaskarray(array).copy())
    return result


def nearest_neighbour_mask_repair(array, copy=True, repair_gap_size=None, direction='both'):
    """
    Repairs gaps in data by replacing it with the nearest neighbour from
    either side until the gaps are filled. Designed for lots of fairly short
    gaps.

    Restrict the gap repairing using repair_gap_size which determines how
    many samples in gaps to fill over.

    NOTE: The start and end are extrapolated with the first / last valid
    sample in all cases.

    WARNING: Currently wraps, so masked items at start will be filled with
    values from end of array.

    TODO: Avoid wrapping from start /end and use first value to preceed values. (extrapolate)

    Ref: http://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays
    """
    if direction not in ('both', 'forward', 'backward'):
        raise ValueError('Unexpected direction value provided: %s' % direction)
    if copy:
        array = array.copy()
    def next_neighbour(start=1):
        """
        Generates incrementing positive and negative pairs from start
        e.g. start = 1
        yields 1,-1, 2,-2, 3,-3,...
        """
        x = start
        while True:
            if direction in ('both', 'forward'):
                yield x
            if direction in ('both', 'backward'):
                yield -x
            x += 1
    # if first or last masked, repair now (extrapolate)
    start, stop = np.ma.notmasked_edges(array)
    if start > 0:
        array[:start] = array[start]
    if stop+1 < len(array):
        array[stop+1:] = array[stop]

    neighbours = next_neighbour()
    if isinstance(array, MappedArray):
        a_copy = array.raw.copy()
    else:
        a_copy = array.copy()
    for n, shift in enumerate(neighbours):
        if not np.any(array.mask) or repair_gap_size and n >= repair_gap_size:
            break
        a_shifted = np.roll(a_copy, shift=shift)
        idx = ~a_shifted.mask * array.mask
        array[idx] = a_shifted[idx]
    return array


def normalise(array, normalise_max=1.0, scale_max=None, copy=True, axis=None):
    """
    Normalise an array between 0 and normalise_max.

    :param normalise_max: Upper limit of normalised result. Default range is between 0 and 1.
    :type normalise_max: float
    :param scale_max: Maximum value to normalise against. If None, the maximum value will be sourced from the array.
    :type scale_max: int or float or None
    :param copy: Returns a copy of the array, leaving input array untouched
    :type copy: bool
    :param axis: default to normalise across all axis together. Only supports None, 0 and 1!
    :type axis: int or None
    :returns: Array containing normalised values.
    :rtype: np.ma.masked_array
    """
    if copy:
        array = array.copy()
    scaling = normalise_max / (scale_max or array.max(axis=axis))
    try:
        if np.isinf(scaling):
            return np_ma_masked_zeros_like(array)
    except:
        # Multi-dimensional process gives scaling = np.ma.masked
        pass

    if axis == 1:
        # transpose
        scaling = scaling.reshape(scaling.shape[0],-1)
    array *= scaling
    ##array *= normalise_max / array.max() # original single axis version
    return array

def np_ma_concatenate(arrays):
    """
    Derivative of the normal concatenate function which handles mapped discrete arrays.
    :param arrays: list of arrays, which may have mapped values.
    :type arrays: list of numpy masked arrays

    :returns: single numpy masked array, which may have mapped values.

    :raises: ValueError if mapped arrays carry different mappings.
    """
    if len(arrays) == 0:
        return None # Nothing to concatenate !

    if hasattr(arrays[0], 'values_mapping'):
        # Handle mapped arrays here.
        mapping = arrays[0].values_mapping
        for each_array in arrays[1:len(arrays)+1]:
            if each_array.values_mapping != mapping:
                raise ValueError('Attempt to concatenate differing multistate arrays')
        array = np.ma.concatenate(arrays)
        array.values_mapping = mapping
        return array
    else:
        # Numeric only arrays.
        return np.ma.concatenate(arrays)


def np_ma_zeros_like(array, mask=False, dtype=float):
    """
    The Numpy masked array library does not have equivalents for some array
    creation functions. These are provided with similar names which may be
    replaced should the Numpy library be extended in future.

    :param array: array of length to be replicated.
    :type array: A Numpy masked array - can be masked or not.

    TODO: Confirm operation with normal Numpy array. The reference to array.data probably fails.

    :returns: Numpy masked array of unmasked zero values, length same as input array.
    """
    return np.ma.array(
        np.zeros_like(array.data),
        mask=mask.copy() if isinstance(mask, np.ndarray) else mask,
        dtype=dtype)


def np_ma_ones_like(array, **kwargs):
    """
    Creates a masked array filled with ones. See also np_ma_zeros_like.

    :param array: array of length to be replicated.
    :type array: A Numpy array - can be masked or not.

    :returns: Numpy masked array of unmasked 1.0 float values, length same as input array.
    """
    return np_ma_zeros_like(array, **kwargs) + 1.0


def np_ma_ones(length):
    """
    Creates a masked array filled with ones.

    :param length: length of the array to be created.
    :type length: integer.

    :returns: Numpy masked array of unmasked 1.0 float values, length as specified.
    """
    return np_ma_zeros_like(np.ma.arange(length)) + 1.0


def np_ma_masked_zeros(length):
    """
    Creates a masked array filled with masked values. The unmasked data
    values are all zero. The very klunky code here is to circumvent Numpy's
    normal response which is to return random data values where it knows the
    data is masked. In this case we want to ensure zero values as we may be
    lifting the mask in due course and we don't want to reveal random data.

    See also np_ma_zeros_like.

    :param length: array length to be replicated.
    :type length: int

    :returns: Numpy masked array of masked 0.0 float values of length equal to
    input.
    """
    return np.ma.array(data=np.zeros(length), mask=True)


def np_ma_masked_zeros_like(array, dtype=float):
    """
    Creates a masked array filled with masked values. The unmasked data
    values are all zero. The very klunky code here is to circumvent Numpy's
    normal response which is to return random data values where it knows the
    data is masked. In this case we want to ensure zero values as we may be
    lifting the mask in due course and we don't want to reveal random data.

    See also np_ma_zeros_like.

    :param array: array of length to be replicated.
    :type array: A Numpy array - can be masked or not.

    :returns: Numpy masked array of masked 0.0 float values, length same as
    input array.
    """
    return np.ma.array(data=np.zeros(len(array), dtype=dtype),
                       mask=np.ones(len(array), dtype=np.bool))


def truck_and_trailer(data, ttp, overall, trailer, curve_sense, _slice):
    '''
    See peak_curvature procedure for details of parameters.

    http://www.flightdatacommunity.com/truck-and-trailer/
    '''
    # Trap for invariant data
    if np.ma.ptp(data) == 0.0:
        return None

    # Set up working arrays
    x = np.arange(ttp) + 1 #  The x-axis is always short and constant
    sx = np.sum(x)
    r = sx/float(x[-1]) #
    trucks = len(data) - ttp + 1 #  How many trucks fit this array length?

    sy = np.empty(trucks) #  Sigma y
    sy[0]=np.sum(data[0:ttp]) #  Initialise this array with just y values

    sxy = np.empty(trucks) #  Sigma x.y
    sxy[0]=np.sum(data[0:ttp]*x[0:ttp]) #  Initialise with xy products

    for back in range(trucks-1):
        # We compute the values for the least squares formula, using the
        # numerator only (the denominator is constant and we're not really
        # interested in the answer).

        # As we move the back of the truck forward, the trailer front is a
        # little way ahead...
        front = back + ttp
        sy[back+1] = sy[back] - data [back] + data[front]
        sxy[back+1] = sxy[back] - sy[back] + ttp*data[front]

    m = np.empty(trucks) # Resulting least squares slope (best fit y=mx+c)
    m = sxy - r*sy

    #  How many places can the truck and trailer fit into this data set?
    places=len(data) - overall + 1
    #  The angle between the truck and trailer at each place it can fit
    angle=np.empty(places)

    for place in range(places):
        angle[place] = m[place+trailer] - m[place]

    # Normalise array and prepare for masking operations
    if np.max(np.abs(angle)) == 0.0:
        return None # All data in a straight line, so no curvature to find.


    # Default curve sense of Concave has a positive angle. The options are
    # adjusted to allow us to use positive only tests hereafter.
    if curve_sense == 'Bipolar':
        angle_max = np.max(np.abs(angle))
        angles = np.ma.abs(angle/angle_max)
    elif curve_sense == 'Convex':
        angle_max = np.min(angle)
        if angle_max>=0.0:
            return None # No concave angles.
        angles = np.ma.array(angle/angle_max)
    else:  # curve_sense == 'Concave'
        angle_max = np.max(angle)
        if angle_max<=0.0:
            return None # No concave angles.
        angles=np.ma.array(angle/angle_max)

    # Find peak - using values over 50% of the highest allows us to operate
    # without knowing the data characteristics.
    peak_slice=np.ma.clump_unmasked(np.ma.masked_less(angles,0.5))

    if peak_slice:
        index = peak_index(angles.data[peak_slice[0]])+\
            peak_slice[0].start+(overall/2.0)-0.5
        return index*(_slice.step or 1) + (_slice.start or 0)
    else:
        # Data curved in wrong sense or too weakly to find corner point.
        return None


def offset_select(mode, param_list):
    """
    This little piece of code finds the offset from a list of possibly empty
    parameters. This is used in the collated engine parameters where
    allowance is made for four engines, but only two or three may be
    installed and we don't know which order the parameters are recorded in.

    :param mode: which type of offset to compute.
    :type mode: string 'mean', 'first', 'last'

    :return: offset
    :type: float
    """
    least = None
    for p in param_list:
        if p:
            if not least:
                least = p.offset
                most = p.offset
                total = p.offset
                count = 1
            else:
                least = min(least, p.offset)
                most = max(most, p.offset)
                total = total + p.offset
                count += 1
    if mode == 'mean':
        return total / float(count)
    if mode == 'first':
        return least
    if mode == 'last':
        return most
    raise ValueError ("offset_select called with unrecognised mode")


def overflow_correction(test_array, param, air_slices):
    '''
    Overflow Correction, used only on Altitude Radio signals.
    This is used to correct for data overflows and to tidy up the resulting data,
    in case there are offsets following the removal of overflow jumps.

    :param test_array: array of data from a single radio altimeter, with arinc masking
    :type test_array: Numpy array
    :param: The 'Altitude Radio (*)' parameter
    :type param: flight data parameter
    :param air_slices: airborne periods
    :type air_slices: List of slices
    '''
    oflow = ALTITUDE_RADIO_MIN_OVERFLOW
    array = overflow_correction_array(test_array.astype(type(oflow)), oflow)
    good_slices = slices_int(slices_remove_small_gaps(
        slices_remove_small_slices(np.ma.clump_unmasked(array),
                                   time_limit=10, hz=param.frequency),
        time_limit=15, hz=param.frequency))
    if not good_slices:
        return array

    # Due to aircraft power-up sequences, the first few samples of data is
    # sometimes corrupt.
    end = good_slices[0].stop
    begin = min(good_slices[0].start + 4, end - 1)
    good_slices[0] = slice(begin, end)
    # Build a list of liftoff and touchdown indexes
    fast_idxs = [f.start for f in air_slices]
    fast_idxs.extend([f.stop for f in air_slices])
    # If there is only one good_slice, we want the landing to take priority
    fast_idxs = sorted(fast_idxs, reverse=True)
    # Find the highest point in all this data
    top_all = np.ma.max(array)

    for good_slice in good_slices:
        # Find the highest point in this slice
        top = np.ma.max(array[good_slice])

        if are_indexes_within_slices(fast_idxs, [good_slice]):
            # We know the rad alt should read zero at liftoff or touchdown
            # Find the index within this good slice
            for f in fast_idxs:
                if good_slice.start < f < good_slice.stop:
                    break

            delta = oflow * np.rint(array[int(f)] / oflow)
            if delta and np.ma.min(array[good_slice] - delta) >= -20:
                # if the lowest point is less than -20, we can safely assume something went wrong,
                # so we don't apply it, otherwise make the adjustment
                array[good_slice] -= delta

        elif slices_and_not([good_slice], air_slices):
            # Rad alt should read zero when the aircraft is too slow to be airborne
            delta = oflow * np.rint(np.ma.median(array[good_slice]) / oflow)
            if delta and np.ma.min(array[good_slice] - delta) >= -20:
                array[good_slice] -= delta

        elif np.ma.ptp(array[good_slice]) < 20.0 or top is np.ma.masked:
            # This data comes from an airborne phase so should vary and should not be masked.
            # If it is in variant or masked, we can best ignore the whole block.
            array[good_slice] = np.ma.masked

        else:
            # An airborne period where we can make the top align with the rest of the data
            delta = oflow * np.rint((top - top_all) / oflow)
            if delta and np.ma.min(array[good_slice] - delta) >= -20:
                array[good_slice] -= delta

    return array

def overflow_correction_array(array, oflow):
    '''
    Overflow correction based on power of two jumps only.

    :param array: array of data to be corrected for overflows
    :type array: numpy masked array
    :param oflow: lowest power of 2 overflow to be corrected
    :type oflow: float
    '''
    jump = np.ma.ediff1d(array, to_begin=0.0)
    abs_jump = np.ma.abs(jump)
    steps = -np.ma.where(abs_jump > oflow * 0.9, 2**np.rint(np.ma.log2(abs_jump)) * np.sign(jump), 0)
    max_step = np.ma.max(np.ma.abs(steps))

    if max_step < oflow: # Nothing to do, so return unchanged
        return array
    for index in np.ma.nonzero(steps)[0]:
        if abs(jump[index]) < max_step * 0.9:
            steps[index] = 0.0

    # This applies the accumulated step changes in one line
    array += np.ma.cumsum(steps)

    return array


def peak_curvature(array, _slice=slice(None), curve_sense='Concave',
                   gap = TRUCK_OR_TRAILER_INTERVAL,
                   ttp = TRUCK_OR_TRAILER_PERIOD):
    """
    :param array: Parameter to be examined
    :type array: Numpy masked array
    :param _slice: Range of index values to be scanned.
    :type _slice: Python slice. May be indexed in reverse to scan backwards in time.
    :param curve_sense: Optional operating mode. Default 'Concave' has
                        positive curvature (concave upwards when plotted).
                        Alternatives 'Convex' for curving downwards and
                        'Bipolar' to detect either sense.
    :type curve_sense: string

    :returns peak_curvature: The index where the curvature first peaks in the required sense.
    :rtype: integer

    Note: Although the range to be inspected may be restricted by slicing,
    the peak curvature index relates to the whole array, not just the slice.

    This routine uses a "Truck and Trailer" algorithm to find where a
    parameter changes slope. In the case of FDM, we are looking for the point
    where the airspeed starts to increase (or stops decreasing) on the
    takeoff and landing phases. This is more robust than looking at
    longitudinal acceleration and complies with the POLARIS philosophy that
    we should provide analysis with only airspeed, altitude and heading data
    available.
    """
    curve_sense = curve_sense.title()
    if curve_sense not in ('Concave', 'Convex', 'Bipolar'):
        raise ValueError('Curve Sense %s not supported' % curve_sense)
    if gap%2 - 1:
        gap -= 1  #  Ensure gap is odd
    trailer = int(ttp)+int(gap)
    overall = 2*int(ttp) + int(gap)

    input_data = array[slices_int(_slice)]
    if np.ma.count(input_data)==0:
        return None

    valid_slices = np.ma.clump_unmasked(input_data)
    for valid_slice in valid_slices:
        # check the contiguous valid data is long enough.
        if (valid_slice.stop - valid_slice.start) <= 3:
            # No valid segment data is not long enough to process
            continue
        elif np.ma.ptp(input_data[slices_int(valid_slice)]) == 0:
            # No variation to scan in current valid slice.
            continue
        elif valid_slice.stop - valid_slice.start > overall:
            # Use truck and trailer as we have plenty of data
            data = array[slices_int(_slice)][slices_int(valid_slice)]
            # The normal path is to go and process this data.
            corner = truck_and_trailer(data, int(ttp), overall, trailer, curve_sense, _slice)  #Q: What is _slice going to do if we've already subsliced it?
            if corner:
                # Found curve
                return corner + valid_slice.start
            # Look in next slice
            continue
        else:
            if _slice.step not in (None, 1, -1):
                raise ValueError("Index returned cannot handle big steps!")
            # Simple methods for small data sets.
            data = input_data[slices_int(valid_slice)]
            curve = data[2:] - 2.0*data[1:-1] + data[:-2]
            if curve_sense == 'Concave':
                curve_index, val = max_value(curve)
                if val <= 0:
                    # No curve or Curved wrong way
                    continue
            elif curve_sense == 'Convex':
                curve_index, val = min_value(curve)
                if val >= 0:
                    # No curve or Curved wrong way
                    continue
            else:  #curve_sense == 'Bipolar':
                curve_index, val = max_abs_value(curve)
                if val == 0:
                    # No curve
                    continue
            if _slice.step and _slice.step < 0 and (_slice.start and _slice.stop):
                # stepping backwards through data, change index
                return _slice.start - curve_index -1
            elif _slice.step and _slice.step < 0:
                # stepping backwards without slices
                return len(array) - (curve_index + 1 + valid_slice.start + (_slice.start or 0))
            else:
                # step == 0 or 1, return index + 1 to get to the middle of the curve
                return curve_index + 1 + valid_slice.start + (_slice.start or 0)
        #endif
    else:  #endfor
        # did not find curve in valid data
        return None


def peak_index(a):
    '''
    Scans an array and returns the peak, where possible computing the local
    maximum assuming a quadratic curve over the top three samples.

    :param a: array
    :type a: list of floats

    '''
    if len(a) == 0:
        raise ValueError('No data to scan for peak')
    elif len(a) == 1:
        return 0
    elif len(a) == 2:
        return np.argmax(a)
    else:
        loc=np.argmax(a)
        if loc == 0:
            return 0
        elif loc == len(a)-1:
            return len(a)-1
        else:
            denominator = (2.0*a[loc-1]-4.0*a[loc]+2.0*a[loc+1])
            if abs(denominator) < 0.001:
                return loc
            else:
                peak=(a[loc-1]-a[loc+1])/denominator
                return loc+peak


def rate_of_change_array(to_diff, hz, width=None, method='two_points'):
    '''
    Lower level access to rate of change algorithm. See rate_of_change for
    description.

    The regression method was added to provide greater smoothing over an
    extended period. This is required where the parameter being
    differentiated has poor quantisation, e.g. Altitude STD with 32ft steps.

    :param to_diff: input data
    :type to_diff: Numpy masked array
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param width: the differentiation time period (sec)
    :type width: float
    :param method: selects 'two_point' simple differentiation or 'regression'
    type method: string

    :returns: masked array of values with differentiation applied

    '''
    if width is None:
        width = 2 / hz

    hw = int(width * hz / 2.0)
    hw2 = hw * 2

    if hw < 1:
        raise ValueError('Rate of change called with inadequate width.')

    if len(to_diff) <= 2 * hw:
        logger.info("Rate of change called with short data segment. Zero rate "
                    "returned")
        return np_ma_zeros_like(to_diff)

    if method == 'two_points':
        input_mask = np.ma.getmaskarray(to_diff)
        # Set up an array of masked zeros for extending arrays.
        slope = np.ma.copy(to_diff)
        slope[hw:-hw] = (to_diff[2*hw:] - to_diff[:-2*hw]) / hw2 * hz
        slope[:hw] = (to_diff[1:hw+1] - to_diff[0:hw]) * hz
        slope[-hw:] = (to_diff[-hw:] - to_diff[-hw-1:-1])* hz
        slope.mask = np.logical_or(input_mask, np.ma.getmaskarray(slope))
        for i in range(-hw,0):
            slope.mask[:i] = np.logical_or(input_mask[-i:], slope.mask[:i])
        for i in range(1,hw+1):
            slope.mask[i:] = np.logical_or(input_mask[:-i], slope.mask[i:])
        return slope

    elif method == 'regression':
        # Neat solution; works well, but for height data smoothing the raw
        # values works better and for pitch and roll attitudes the
        # improvement was small and would result in more masked results than
        # the preceding technique.

        # The fit will be for equi-spaced samples around the midpoint.
        x = np.arange(-hw, hw+1)
        # Scaling is given by:
        sx2_hz = np.sum(x*x)/hz
        # We extended data array to allow for convolution overruns.
        z = np.array([to_diff[0]]*hw+list(to_diff)+[to_diff[-1]]*hw)
        # The compute the least squares fit for each point over the required
        # range and re-scale to allow for width and sample rate.
        return np.convolve(z,-x,'same')[hw:-hw]/sx2_hz

    else:
        raise ValueError('Rate of change called with unrecognised method')


def rate_of_change(diff_param, width, method='two_points'):
    '''
    @param to_diff: Parameter object with .array attr (masked array)

    Differentiation using the xdot(n) = (x(n+hw) - x(n-hw))/w formula.
    Half width hw=w/2 and this provides smoothing over a w second period,
    without introducing a phase shift.

    The mask array is manipulated to make all samples enclosed by the
    differentiation range masked; that is, although only two points in the
    original array are used in the computation, if the width covers 4
    samples, then 9 final result values are masked, corresponding to four
    steps before and four steps after the midpoint.

    :param diff_param: input Parameter
    :type diff_param: Parameter object
    :type diff_param.array : masked array
    :param diff_param.frequency : sample rate for the input data (sec-1)
    :type diff_param.frequency: float
    :param width: the differentiation time period (sec)
    :type width: float
    :param method: selects 'two_point' simple differentiation or 'regression'
    type method: string

    :returns: masked array of values with differentiation applied
    '''
    hz = diff_param.frequency
    to_diff = diff_param.array
    return rate_of_change_array(to_diff, hz, width, method=method)


"""
def runs_of_ones_array(bits, min_len=0, max_len=None):
    '''
    :type bits: np.ma.masked_array
    :type min_len: int
    :type max_len: int or None
    :returns: Start, stop and duration of ones.
    :rtype: (int, int, int)
    '''
    # make sure all runs of ones are well-bounded
    bounded = np.ma.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.ma.diff(bounded)
    run_starts, = np.ma.where(difs > 0)
    run_ends, = np.ma.where(difs < 0)
    run_dur = run_ends - run_starts
    filtered = run_dur >= min_len
    if max_len:
        filtered &= run_dur <= max_len
    # return duration and starting locations
    return run_starts[filtered], run_ends[filtered], run_dur[filtered]


def repair(array, fill_with='starts'):
    if not np.ma.count(array):
        return array
    #TODO: Move to repair_mask...
    #data = array
    starts, ends, durs = runs_of_ones_array(array.mask)
    repeats = np.diff(np.ma.hstack(([0], ends)))
    # use starts-1 or just ends
    if fill_with == 'starts':
        fill = np.repeat(array[starts-1], repeats)
    else:
        fill = np.repeat(array[ends], repeats)
    pad = np.zeros(len(array) - len(fill))
    vals = np.hstack((fill, pad))
    # fill data, where masked, using repeated vals
    np.copyto(array, vals, where=array.mask, casting='unsafe')
    return array


>>> repair(y)
array([ 0,  1,  1,  1,  1,  1,  6,  7,  7,  9, 10, 11, 11, 11, 11, 11, 16,
       17, 17, 19])
>>> y
masked_array(data = [0 1 -- -- -- -- 6 7 -- 9 10 11 -- -- -- -- 16 17 -- 19],
             mask = [False False  True  True  True  True False False  True False False False
  True  True  True  True False False  True False],
       fill_value = 999999)

>>> repair(y, 'ends')
array([ 0,  1,  6,  6,  6,  6,  6,  7,  9,  9, 10, 11, 16, 16, 16, 16, 16,
       17, 19, 19])

"""


def fill_masked_edges(array, fill_value):
    '''
    :type array: np.ma.array
    :type fill_value: float or int
    :returns: Array with masked beginning and end masked.
    :rtype: np.ma.masked_array
    '''
    if array.mask.all() or not array.mask.any():
        # Array is entirely masked or entirely unmasked.
        return array

    masked_slices = np.ma.clump_masked(array)
    if masked_slices[0].start == 0:
        # Fill the start of the array.
        array[masked_slices[0]] = fill_value
    if masked_slices[-1].stop == len(array):
        # Fill the end of the array.
        array[masked_slices[-1]] = fill_value

    return array


def repair_mask(array, frequency=1, repair_duration=REPAIR_DURATION,
                copy=False, extrapolate=False, repair_above=None,
                method='interpolate', raise_duration_exceedance=False,
                raise_entirely_masked=True):
    '''
    This repairs short sections of data ready for use by flight phase algorithms
    It is not intended to be used for key point computations, where invalid data
    should remain masked.

    :param copy: If True, returns modified copy of array, otherwise modifies the array in-place.
    :param method: Repair method to apply in masked sections, either interpolate, fill_start (fill with value at start of masked section), fill_stop (fill with stop of masked section).
    :param raise_entirely_masked: If True, an exception is raised if the incoming data is entirely masked.
    :param repair_duration: If None, any length of masked data will be repaired.
    :param raise_duration_exceedance: If False, no warning is raised if there are masked sections longer than repair_duration. They will remain unrepaired.
    :param extrapolate: If True, data is extrapolated at the start and end of the array.
    :param repair_above: If value provided only masked ranges where first and last unmasked values are this value will be repaired.
    :raises ValueError: If the entire array is masked.
    '''
    if array.mask.all():
        # Cannot repair entierly masked array.
        if raise_entirely_masked:
            raise ValueError("Array cannot be repaired as it is entirely masked")
        else:
            return array
    elif not array.mask.any():
        # Array entirely unmasked - nothing to do.
        return array

    if copy:
        array = array.copy()

    if repair_duration:
        repair_samples = repair_duration * frequency
    else:
        repair_samples = None

    masked_sections = np.ma.clump_masked(array)

    for section in masked_sections:
        length = section.stop - section.start
        if repair_samples and length > repair_samples:
            if raise_duration_exceedance:
                raise ValueError("Length of masked section '%s' exceeds "
                                 "repair duration '%s'." % (length * frequency,
                                                            repair_duration))
            else:
                continue # Too long to repair
        elif section.start == 0:
            if extrapolate or method == 'fill_stop':
                # TODO: Does it make sense to subtract 1 from the section stop??
                #array.data[section] = array.data[section.stop - 1]
                array[section] = array[section.stop]
            else:
                continue # Can't interpolate if we don't know the first sample

        elif section.stop == len(array):
            if extrapolate or method == 'fill_start':
                array[section] = array[section.start - 1]
            else:
                continue # Can't interpolate if we don't know the last sample

        else:
            start_value = array[section.start - 1]
            stop_value = array[section.stop]
            if method == 'interpolate':
                if (repair_above is None or
                    (start_value > repair_above and stop_value > repair_above)):
                    # XXX: Find neater solution, potentially
                    # scipy.interpolate.InterpolatedUnivariateSpline, as
                    # optimisation.
                    array.data[section] = np.linspace(start_value, stop_value,
                                                      length+2)[1:-1]
                    array.mask[section] = False
            elif method == 'fill_start':
                array[section] = start_value
            elif method == 'fill_stop':
                array[section] = stop_value
            else:
                raise NotImplementedError('Repair method %s not implemented.',
                                          method)

    return array


def resample(array, orig_hz, resample_hz):
    '''
    Upsample or downsample an array for it to match resample_hz.
    Offset is maintained because the first sample is always returned.
    '''
    if orig_hz == resample_hz:
        return array
    modifier = resample_hz / float(orig_hz)
    if modifier > 1:
        return np.ma.repeat(array, modifier)
    else:
        # Only convert complete blocks of data.
        endpoint = floor(len(array)*modifier)/modifier
        return array[:int(endpoint):int(1 / modifier)]


def round_to_nearest(array, step):
    """
    Rounds to nearest step value, so step 5 would round as follows:
    1 -> 0
    3.3 -> 5
    7.5 -> 10
    10.5 -> 10 # np.round drops to nearest even number(!)

    :param array: Array to be rounded
    :type array: np.ma.array
    :param step: Value to round to
    :type step: int or float
    """
    step = float(step) # must be a float
    return np.ma.round(array / step) * step


def rms_noise(array, ignore_pc=None):
    '''
    :param array: input parameter to measure noise level
    :type array: numpy masked array
    :param ignore_pc: percent to ignore (see below)
    :type integer: % value in range 0-100

    :returns: RMS noise level
    :type: Float, units same as array

    :exception: Should all the difference terms include masked values, this
    function will return None.

    This computes the rms noise for each sample compared with its neighbours.
    In this way, a steady cruise at 30,000 ft will yield no noise, as will a
    steady climb or descent.

    The rms noise may be used to examine parameter reasonableness, in which
    case the occasional spike is not considered background noise levels. The
    ignore_pc value allows the highest spike readings to be ignored and the
    rms is then the level for the normal operation of the parameter.
    '''
    # The difference between one sample and the ample to the left is computed
    # using the ediff1d algorithm, then by rolling it right we get the answer
    # for the difference between this sample and the one to the right.
    if len(array.data)==0 or np.ma.ptp(array.data)==0.0:
        #logging.warning('rms noise test has no variation in signal level')
        return None
    diff_left = np.ma.ediff1d(array, to_end=0)
    diff_right = np.ma.array(data=np.roll(diff_left.data,1),
                             mask=np.roll(diff_left.mask,1))
    local_diff = (diff_left - diff_right)/2.0
    diffs = local_diff[1:-1]
    if np.ma.count(diffs) == 0:
        return None
    elif ignore_pc is None or ignore_pc/100.0*len(array)<1.0:
        to_rms = diffs
    else:
        monitor = slice(0, int(floor(len(diffs) * (1-ignore_pc/100.0))))
        to_rms = np.ma.sort(np.ma.abs(diffs))[monitor]
    return sqrt(np.ma.mean(np.ma.power(to_rms, 2))) # RMS in one line !


def runs_of_ones(bits, min_samples=None, skip_mask=False):
    '''
    Q: This function used to have a min_len kwarg which was a result of its
    implementation. If there is a use case for only returning sections greater
    than a minimum length, would it be better to specify time based on a
    frequency rather than samples?
    TODO: Update to return Sections?
    :returns: S
    :rtype: [slice]
    '''
    if not len(bits) or not np.ma.count(bits):
        return []
    if skip_mask:
        bits = repair_mask(bits)
    runs = np.ma.clump_unmasked(np.ma.masked_not_equal(bits, 1))
    if min_samples:
        runs = slices_remove_small_slices(runs, count=min_samples)

    return runs


def slices_of_runs(array, min_samples=None, flat=False):
    '''
    Provides a list of slices of runs of each value in the array.

    If the provided array is a mapped array used for multi-states, we return the
    string representation of the value. This function works as a generator which
    yields a tuple of the value and a list of slices of each run of that value
    in the provided array.

    Note that the masked constant value is excluded from the returned data.

    :param array: a numpy masked array or mapped array.
    :type array: np.ma.array
    :param flat: yield a flat list of unordered slices
    :type flat: bool
    '''
    mapped_array = hasattr(array, 'values_mapping')
    for value in np.ma.sort(np.ma.unique(array.raw if mapped_array else array)):
        if value is np.ma.masked:
            continue
        if mapped_array:
            value = array.values_mapping[value]
        runs = runs_of_ones(array == value, min_samples=min_samples)
        if flat:
            for run in runs:
                yield run
        else:
            if min_samples and runs == []:
                value = None
            yield value, runs


def shift_slice(this_slice, offset):
    """
    This function shifts a slice by an offset. The need for this arises when
    a phase condition has been used to limit the scope of another phase
    calculation.

    :type this_slice: slice
    :type offset: int or float
    :rtype: slice
    """
    if not offset:
        return this_slice

    start = None if this_slice.start is None else this_slice.start + offset
    stop = None if this_slice.stop is None else this_slice.stop + offset

    if start is None or stop is None or (stop - start) >= 1:
        ### This traps single sample slices which can arise due to rounding of
        ### the iterpolated slices.
        return slice(start, stop, this_slice.step)
    else:
        return None


def shift_slices(slicelist, offset):
    """
    This function shifts a list of slices by a common offset, retaining only
    the valid (not None) slices.

    :type slicelist: [slice]
    :type offset: int or float
    :rtype [slice]

    """
    if offset:
        newlist = []
        for each_slice in slicelist:
            if each_slice and offset:
                new_slice = shift_slice(each_slice,offset)
                if new_slice: newlist.append(new_slice)
        return newlist
    else:
        return slicelist


def slice_duration(_slice, hz):
    '''
    Gets the duration of a slice in taking the frequency into account. While
    the calculation is simple, there were instances within the code of slice
    durations being compared against values in seconds without considering
    the frequency of the slice indices.

    :param _slice: Slice to calculate the duration of.
    :type _slice: slice
    :param hz: Frequency of slice.
    :type hz: float or int
    :returns: Duration of _slice in seconds.
    :rtype: float
    '''
    if _slice.stop is None:
        raise ValueError("Slice stop '%s' is unsupported by slice_duration.",
                         _slice.stop)
    return (_slice.stop - (_slice.start or 0)) / float(hz)


def slices_duration(slices, hz):
    '''
    Gets the total duration of a list of slices.

    :param slices: Slices to calculate the total duration of.
    :type slices: [slice]
    :param hz: Frequency of slices.
    :type hz: int or float
    :returns: Total duration of all slices.
    :rtype: float
    '''
    return sum([slice_duration(_slice, hz) for _slice in slices])


def slices_contract(slices, length, max_index=None):
    '''
    Contract as in decrease in size. Slices which are smaller than the contraction are removed.
    Negative steps are not supported.

    :param slices: Slices to contract.
    :type slices: [slice]
    :param length: Length to contract.
    :type length: int or float
    :param max_index: Max index for contracting None slices.
    :type max_index: int or None
    :returns: Contracted slices.
    :rtype: slice
    '''
    if length == 0:
        return slices

    if max_index is not None and max_index <= 0:
        raise ValueError('')

    constricted_slices = []
    for s in slices:
        if s.step and s.step < 0:
            raise NotImplementedError('Negative steps are not supported.')

        slice_start = s.start + length if s.start else length

        slice_stop = s.stop or max_index
        if slice_stop:
            slice_stop -= length

            if slice_start >= slice_stop:
                # Slice has been contracted 'out of existence'.
                continue

        constricted_slices.append(slice(slice_start, slice_stop, s.step))

    return constricted_slices


def slices_contract_duration(slices, frequency, seconds, max_index=None):
    '''
    :param slices: Slices to contract.
    :type slices: [slice]
    :param frequency: Frequency of slices.
    :type frequency: int or float
    :param seconds: Seconds to contract the slices by.
    :type seconds: int or float
    :param max_index: Max index for contracting None slices.
    :type max_index: int or None
    :returns: Contract slices.
    :rtype: [slice]
    '''
    return slices_contract(slices, seconds / float(frequency), max_index=max_index)


def slices_extend(slices, length):
    '''
    :param slices: Slices to extend.
    :type slices: [slice]
    :param length: Length of extension in samples.
    :type length: int or float
    :returns: Extended slices.
    :rtype: [slice]
    '''
    if length == 0:
        return slices

    extended_slices = []
    for s in slices:
        if s.step and s.step < 0:
            raise NotImplementedError('Negative step is not supported.')

        extended_slices.append(slice(
            max(s.start - length, 0) if s.start else 0,
            s.stop + length if s.stop else None,
            s.step,
        ))
    return slices_or(extended_slices)


def slices_extend_duration(slices, frequency, seconds):
    '''
    :param slices: Slices to extend.
    :type slices: [slice]
    :param frequency: Frequency of slices.
    :type frequency: int or float
    :param seconds: Seconds to extend the slices by.
    :type seconds: int or float
    :returns: Extended slices.
    :rtype: [slice]
    '''
    return slices_extend(slices, seconds / float(frequency))


def slices_after(slices, index):
    '''
    Gets slices truncated to only contain sections after an index.

    :param slices: Slices to truncate.
    :type slices: [slice]
    :param index: Cutoff index.
    :type index: int or float
    :returns: Truncated slices.
    :rtype: [slice]
    '''
    truncated_slices = []
    for _slice in slices:
        if _slice.stop is None:
            raise ValueError(
                'Slice stop being None is not supported in slices_after.')
        if _slice.start > _slice.stop:
            raise ValueError(
                'Reverse slices are not supported in slices_after.')
        if _slice.stop < index:
            # Entire slice is before index.
            continue
        if (_slice.start or 0) < index:
            _slice = slice(index, _slice.stop)
        truncated_slices.append(_slice)
    return truncated_slices


def slices_before(slices, index):
    '''
    Gets slices truncated to only contain sections before an index.

    :param slices: Slices to truncate.
    :type slices: [slice]
    :param index: Cutoff index.
    :type index: int or float
    :returns: Truncated slices.
    :rtype: [slice]
    '''
    truncated_slices = []
    for _slice in slices:
        if _slice.stop is None:
            raise ValueError(
                'Slice stop being None is not supported in slices_before.')
        if _slice.start > _slice.stop:
            raise ValueError(
                'Reverse slices are not supported in slices_after.')
        if _slice.start > index:
            # Entire slice is before index.
            continue
        if _slice.stop > index:
            _slice = slice(_slice.start, index)
        truncated_slices.append(_slice)
    return truncated_slices


def slice_midpoint(_slice):
    '''
    Gets the midpoint of a slice. Slice stop of None is not supported.

    :param _slice:
    :type _slice: slice
    :returns: The midpoint of the slice.
    :rtype: float
    '''
    difference = _slice.stop - (_slice.start or 0)
    return _slice.stop - (float(difference) / 2)


def slice_multiply(_slice, f):
    '''
    :param _slice: Slice to rescale
    :type _slice: slice
    :param f: Rescale factor
    :type f: float

    :returns: slice rescaled by factor f
    :rtype: integer
    '''
    """
    Original version replaced by less tidy version to maintain start=0 cases
    and to ensure rounding for reductions in frequency does not extend into
    earlier samples than those intended.
    """
    return slice(None if _slice.start is None else int(ceil(_slice.start * f)),
                 None if _slice.stop is None else int(_slice.stop * f),
                 None if _slice.step is None else int(_slice.step * f))


def slices_multiply(slices, f):
    '''
    :param slices: Iterable of slices to rescale
    :type slices: Iterable
    :param f: Rescale factor
    :type f: float

    :returns: List of slices rescaled by factor f
    :rtype: integer
    '''
    return [slice_multiply(s, f) for s in slices]


def slice_round(_slice):
    '''
    Round the slice start and stop indices to the nearest integer boundary.

    As numpy arrays floor float indices, this can produce inaccurate results,
    especially when the parameter frequency is low.

    :type _slice: slice
    :rtype: slice
    '''
    start = _slice.start
    stop = _slice.stop
    if start is not None:
        start = int(np.round(start))
    if stop is not None:
        stop = int(np.round(stop))
    return slice(start, stop, _slice.step)


def slices_split(slices, index):
    '''
    Split slices at index.

    Returns a new list of slices with the matching one of them split at index
    (if applicable).

    :param slices: Slices to split
    :type slices: list of slices
    :param index: Index at which to split slices
    :type index: int

    :returns: Split slices
    :rtype: list of slices
    '''
    result = []
    for sl in slices:
        if is_index_within_slice(index, sl):
            result.append(slice(sl.start, index))
            result.append(slice(index, sl.stop))
        else:
            result.append(sl)

    return result


def slices_round(slices):
    '''
    Round an iterable of slices.

    :type _slice: iterable of slices
    :rtype: [slice]
    '''
    return [slice_round(s) for s in slices]


def slice_samples(_slice):
    '''
    Gets the number of samples in a slice.

    :param _slice: Slice to count sample length.
    :type _slice: slice
    :returns: Number of samples in _slice.
    :rtype: integer
    '''
    step = 1 if _slice.step is None else _slice.step

    if _slice.start is None or _slice.stop is None:
        return 0
    else:
        return (abs(_slice.stop - _slice.start) - 1) // abs(step) + 1


def slices_above(array, value):
    '''
    Get slices where the array is above value. Repairs the mask to avoid a
    large number of slices being created.

    :param array:
    :type array: np.ma.masked_array
    :param value: Value to create slices above.
    :type value: float or int
    :returns: Slices where the array is above a certain value.
    :rtype: list of slice
    '''
    if len(array) == 0:
        return array, []
    repaired_array = repair_mask(array)
    if repaired_array is None: # Array length is too short to be repaired.
        return array, []
    band = np.ma.masked_less(repaired_array, value)
    slices = np.ma.clump_unmasked(band)
    return repaired_array, slices


def slices_below(array, value):
    '''
    Get slices where the array is below value. Repairs the mask to avoid a
    large number of slices being created.

    :param array:
    :type array: np.ma.masked_array
    :param value: Value to create slices below.
    :type value: float or int
    :returns: Slices where the array is below a certain value.
    :rtype: list of slice
    '''
    if len(array) == 0:
        return array, []
    repaired_array = repair_mask(array)
    if repaired_array is None: # Array length is too short to be repaired.
        return array, []
    band = np.ma.masked_greater(repaired_array, value)
    slices = np.ma.clump_unmasked(band)
    return repaired_array, slices


def slices_between(array, min_, max_):
    '''
    Get slices where the array's values are between min_ and max_. Repairs
    the mask to avoid a large number of slices being created.

    :param array:
    :type array: np.ma.masked_array
    :param min_: Minimum value within slices.
    :type min_: float or int
    :param max_: Maximum value within slices.
    :type max_: float or int
    :returns: Slices where the array is above a certain value.
    :rtype: list of slice
    '''
    if np.ma.count(array) == 0:
        return array, []
    try:
        repaired_array = repair_mask(array)
    except ValueError:
        # data is entirely masked or too short to be repaired
        return array, []
    # Slice through the array at the top and bottom of the band of interest
    band = np.ma.masked_outside(repaired_array, min_, max_)
    # Remove the equality cases as we don't want these. (The common issue
    # here is takeoff and landing cases where 0ft includes operation on the
    # runway. As the array samples here are not coincident with the parameter
    # being tested in the KTP class, by doing this we retain the last test
    # parameter sample before array parameter saturated at the end condition,
    # and avoid testing the values when the array was unchanging.
    band = np.ma.masked_equal(band, min_)
    band = np.ma.masked_equal(band, max_)
    # Group the result into slices - note that the array is repaired and
    # therefore already has small masked sections repaired, so no allowance
    # is needed here for minor data corruptions.
    slices = np.ma.clump_unmasked(band)
    return repaired_array, slices


def slices_from_to(array, from_, to, threshold=0.1):
    '''
    Get slices of the array where values are between from_ and to, and either
    ascending or descending depending on whether from_ is greater than or less
    than to. Dips and peaks into the range will create slices where the
    direction of the array matches from_ and to.

    :param array:
    :type array: np.ma.masked_array
    :param from_: Value from.
    :type from_: float or int
    :param to: Value to.
    :type to: float or int
    :param threshold: Minimum threshold to detect dips and peaks into the range defined as a ratio from 0 (no threshold) to 1 (full range).
    :type threshold: float or int
    :returns: Slices of the array where values are between from_ and to and either ascending or descending depending on comparing from_ and to.
    :rtype: list of slice
    '''
    if from_ == to:
        raise ValueError('From and to values should not be equal.')
    elif threshold > 1 or threshold < 0:
        raise ValueError('threshold must be within the range 0-1.')
    elif len(array) == 0:
        return array, []

    # Find the minimum and maximum of the range without knowing direction.
    range_min = min(from_, to)
    range_max = max(from_, to)

    # Find the threshold min and max.
    threshold = (range_max - range_min) * threshold
    threshold_max = range_max - threshold
    threshold_min = range_min + threshold

    rep_array, slices = slices_between(array, from_, to)
    rep_array.mask = np.ma.getmaskarray(rep_array)

    filtered_slices = []
    for _slice in slices:

        if len(array[_slice]) == 1:
            # If only a single sample matches, we must compare the sample
            # before and after to ascertain direction.
            test_slice = slice(_slice.start - 1, _slice.stop + 1)
            if sum(np.invert(rep_array.mask[test_slice])) != 3:
                # Skip slice without 3 unmasked samples.
                continue
        else:
            test_slice = _slice

        # Calculate the following variables to work out if _slice is linear,
        # dip or peak.
        # Check if the start and stop of the slice are either masked or
        # at an array boundary.
        starts_within_range = (test_slice.start == 0 or
                               rep_array.mask[test_slice.start - 1])
        stops_within_range = (test_slice.stop == len(array) or
                              rep_array.mask[test_slice.stop])
        start_value = array[test_slice.start]
        stop_value = array[test_slice.stop - 1]
        # Check if the start and stop are in the upper or lower half of the
        # range to attempt to discern direction.
        start_max = abs(start_value - range_max) < abs(start_value - range_min)
        stop_max = abs(stop_value - range_max) < abs(stop_value - range_min)
        # The section is considered to be linear if it transitions from one
        # half of the range to the other between the start and stop. Sections
        # not matching this condition are considered to have no clear direction.
        linear = sum([start_max, stop_max]) == 1

        slice_start = _slice.start
        slice_stop = _slice.stop

        if starts_within_range and stops_within_range and not linear:
            # data is curved but does not enter range at start or stop.
            # we could dissect further, but it's already complicated.
            continue
        elif linear:
            # linear
            if from_ > to and not start_max:
                # descending wrong direction
                continue
            elif from_ < to and not stop_max:
                # climbing wrong direction
                continue
        elif ((starts_within_range and stop_max) or
              (stops_within_range and start_max) or
              (start_max and stop_max)):
            # treat as dip
            value = min_value(rep_array, _slice=test_slice)
            if value.value > threshold_max:
                # value does not dip below threshold
                continue
            if from_ > to:
                # descending
                slice_stop = value.index
            else:
                # climbing
                slice_start = value.index
        elif ((starts_within_range and not stop_max) or
              (stops_within_range and not start_max) or
              (not start_max and not stop_max)):
            # treat as peak
            value = max_value(rep_array, _slice=test_slice)
            if value.value < threshold_min:
                # value does not peak above threshold
                continue
            if from_ > to:
                # descending
                slice_start = value.index
            else:
                # climbing
                slice_stop = value.index
        if int(slice_start) != int(slice_stop):
            filtered_slices.append(slice(slice_start, slice_stop))

    return rep_array, filtered_slices


def slices_from_ktis(kti_1, kti_2):
    '''
    From two KTIs or KTI lists, this function identifies the pairs of times
    which relate to a section of the flight and return a list of slices ready
    for creation of a KPV using the existing "create_kpv..." family of
    methods. This routine forms the basis of the fuel usage measurement
    functions.

    :param kti_1: Key Time Instance or list of KTIs at start of period of interest
    :type kti_1: KeyTimeInstance node(s)
    :param kti_2: Key Time Instance or list of KTIs at end of period of interest
    :type kti_2: KeyTimeInstance node(s)

    :returns: list of slices
    '''
    # If either list is void, we won't find any valid periods.
    if kti_1 is None or kti_2 is None:
        return []

    # Inelegant way of ensuring we are dealing with lists of KTIs
    if not isinstance(kti_1, list):
        kti_1=[kti_1]
    if not isinstance(kti_2, list):
        kti_2=[kti_2]

    # Unpack the KTIs to get the indexes, and mark which were
    # start (0) and end (1) values.
    unpk = sorted(itertools.chain(
        ((t.index, 0) for t in kti_1),
        ((t.index, 1) for t in kti_2)))
    # Prepare the ground...
    previous = None
    slices = []
    # Now scan the list looking for an end immediately following a start.
    for item in unpk:
        if item[1]:
            if previous is None or previous[1]:
                continue
            else:
                # previous[1] was 0 and item[1] = 1
                slices.append(slice(previous[0], item[0]+1))
        previous = item
    return slices


def level_off_index(array, frequency, seconds, variance, _slice=None,
                   include_window=True):
    '''
    Find the first index within _slice where array levels off.

    :type array: np.ma.masked_array
    :type frequency: int or float
    :type seconds: int or float
    :type variance: int or float
    :type _slice: slice
    :param include_window: Include the window in the returned level off index.
    :type include_window: bool
    :raises ValueError: If the sample window (frequency * samples) is not a whole number.
    :returns: Index where the array first levels off within slice within the specified variance.
    :rtype:
    '''
    if _slice:
        array = array[slices_int(_slice)]

    samples = int(ceil(frequency * seconds))

    if samples > array.size:
        return None

    sliding_window = np.lib.stride_tricks.as_strided(
        array, shape=(len(array) - samples, samples),
        strides=array.strides * 2)

    unmasked_slices = np.ma.clump_unmasked(array)

    for unmasked_slice in filter_slices_length(unmasked_slices, samples):
        # Exclude masked data within the sliding_window.
        stop = unmasked_slice.stop - (samples - 1)
        ptp = np.ptp(sliding_window[unmasked_slice.start:stop], axis=-1)
        stable = ptp < variance

        if not stable.any():
            # All data is not level.
            continue

        index = np.argmax(stable) + unmasked_slice.start

        if _slice and _slice.start:
            index += _slice.start

        if include_window:
            index += samples

        return index

    return None


"""
Spline function placeholder

At some time we are likely to want to add interpolation, and this scrap of
code was used to prove the principle. Easy to do and the results are really
close to the recorded data in the case used for testing.

See 'Pitch rate computation at 4Hz and 1Hz with interpolation.xls'

import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

y=np.array([0.26,0.26,0.79,0.35,-0.26,-0.04,1.23,4.57,4.75,1.93,0.44,1.14,0.97,1.14,0.79])
x=np.array(range(236,251,1))
f = interp.interp1d(x, y, kind='cubic')
xnew = np.linspace(236,250,57)
plt.plot(x,y,'o',xnew,f(xnew),'-')
plt.legend(['data', 'cubic'], loc='best')
plt.show()
for i in xnew:
    print(f(i))
"""
def step_local_cusp(array, span):
    """
    A small function developed for the step function to find local cusps
    where data has changed from sloping to steady. Cusp defined as the point
    closest to the start of the data where the local slope is half the slope
    from the first sample.

    Unlike the peak curvature algorithm, this does not require a significant
    number of samples to operate (flap travelling times between detents can
    be short). Unlike top of climb algorithm, this uses the local data to
    determine the slope characteristic.

    :param array: Masked array to examine. Always start from beginning of array
                 (must be passed in using reverse indexing if backwards operation needed).
    :type array: np.ma.array

    :returns: index to cusp from start of data. Zero if no cusp found, or if
    the slope increases significantly after the start of the range to test.
    :rtype: integer
    """
    local_array=array[span]
    if len(local_array)==0:
        return None
    elif len(local_array)<3:
        return 0
    else:
        v0 = local_array[0]
        v_1=v0
        for n, v in enumerate(local_array[1:]):
            slope_0 = abs(v-v0)/float(n+1)
            slope_n = abs(v-v_1)
            # The condition reverses for reversed slices.
            if span.step==-1:
                if slope_n < slope_0/2.0:
                    return n+1
                if slope_n > slope_0*2.0:
                    return 0
            else:
                if slope_n < slope_0/2.0:
                    return n
                if slope_n > slope_0*2.0:
                    return 0
            v_1=v
        return 0


def including_transition(array, steps, hz=1, mode='include'):
    r'''
    Snaps signal to step values including transition, e.g.:
          _____
      ___|/   \|
    _|/        \|__

    :type array: np.ma.array
    :param steps: Steps to align the signal to.
    :type steps: [int]
    :param threshold: Threshold of difference between two flap settings to apply the next flap setting.
    :type threshold: float

    threshold = 0.01 makes the system too late and too conservative.
    '''
    steps = sorted(steps)
    mid_steps = [steps[0] - 10.0]

    for step_1, step_2 in zip(steps[:-1], steps[1:]):
        mid_steps.append((step_1 + step_2) / 2.0)
    mid_steps.append(steps[-1] + 10.0)

    change = rate_of_change_array(array, hz)
    # change will show non null values a bit too soon as the function is looking
    # over multiple samples (a window). We want to check the edges of the periods
    # where there is a change and overwrite it with the value given by a
    # simple ediff1d
    significant_roc = np.ma.abs(change) > 0.03
    edges = np_ma_zeros_like(significant_roc, dtype=np.bool)
    edges[1:] = significant_roc[1:] ^ significant_roc[:-1]
    sample_change = np.ma.ediff1d(array, to_begin=0.0)
    change[edges] = sample_change[edges]

    # first raise the array to the next step if it exceeds the previous step
    # plus a minimal threshold (step as early as possible)
    output = np_ma_masked_zeros_like(array)

    for mid_1, flap, mid_2 in zip(mid_steps[:-1], steps, mid_steps[1:]):
        # Slice the data into bands that are between the midpoint flap values
        bands = slices_and(runs_of_ones(array > mid_1), runs_of_ones(array <= mid_2))
        for band in bands:
            # Find where the data did not change in this band...
            partial = np.ma.where(np.ma.abs(change[band.start:band.stop]) < 0.03, flap, np.ma.masked) # threshold of 0.03 to account for slight changes/flutter
            if len(partial) == 1:
                if change[band.start] > 0:
                    output[band.start] = flap
                else:
                    output[min(band.stop - 1, len(array) - 1)] = flap
                continue

            if np.ma.count(partial):
                # Unchanged data can be included in our output flap array directly
                output[band] = partial

            # See if it passed through the flap setting of interest.
            # Widen band of interest in case values were rapidly changing.
            # mid_1, flap = 27.5, 30. If array = [..., 27.0, 31.0, 33.0, ...] the band would
            # only capture the values starting at 31.0 and we would miss the point where
            # we crossed 30 degrees of flap.
            wider_band = slice(max(band.start - 1, 0), min(band.stop + 1, len(array)))
            index = index_at_value(array[wider_band], flap)
            if index is not None:
                if array[wider_band][-1] > array[wider_band.start]:
                    # Going up
                    output[floor(index + wider_band.start)] = flap
                else:
                    # Going down
                    output[ceil(index + wider_band.start)] = flap

            elif not np.ma.count(partial):
                # The data crept into this band without reaching the flap setting.
                # Find the local extrema.
                array_derivative_sign_change = np.diff(np.sign(sample_change[band]))
                idx_non_zero = np.nonzero(array_derivative_sign_change)
                if idx_non_zero:
                    output[band.start + idx_non_zero[0]] = flap
                else:
                    # This would mean that the data continuously increased or decreased
                    # in this band but never crossed the flap setting.
                    # Unless masked data, this seems impossible. Ignore.
                    pass


    for gap in np.ma.clump_masked(output):
        before = output[max(gap.start - 1, 0)]
        after = output[min(gap.stop, len(output) - 1)]
        if mode == 'include':
            output[gap] = max(before, after)
        else:
            output[gap] = min(before, after)

    return output


def step_values(array, steps, hz=1, step_at='midpoint', rate_threshold=0.5):
    """
    Rounds each value in the array to the nearest step, depending upon the
    step_at method.

    Primarily written for flap movements, although slat, ailerons and others
    can make good use of this stepping system.

    Maintains the original array's mask.

    NOTE: If "skip" is to be supported again, simply merge step changes which
    occur within 3 to 5 samples of each other!


    step_at options
    ===============

    midpoint:
    * simply change flap in the middle of transitions

    move_start:
    * transition at the start of movements (like flap lever)

    move_stop:
    * transition at the end of movements

    including_transition:
    * all transition movements are included as the next step (early on
      increase, late on decrease). Normally more safety cautious.

    excluding_transition:
    * transition movements are excluded from the next step until transtion has
      finished. Used by those wishing for minimal time at the next step level
      e.g. this is likely to reduce flap overspeed measurements.


    :param array: Masked array to step
    :type array: np.ma.array
    :param steps: Steps to round to nearest value
    :type steps: list of integers
    :param hz: If 'midpoint' steps not in use, required for rate of change calc
    :type hz: float
    :param step_at: Step conversion mode
    :type step_at: String, default='midpoint', options are listed above.
    :param rate_threshold: rate of change threshold for non-moving control
    :type rate_threshold: float, default 0.5 is suitable for flap operation.
    :returns: Stepped masked array
    :rtype: np.ma.array
    """
    step_at = step_at.lower()
    step_at_options = ('midpoint', 'move_start', 'move_stop',
                       'including_transition', 'excluding_transition')
    if step_at not in step_at_options:
        raise ValueError("Incorrect step_at choice argument '%s'" % step_at)

    steps = sorted(steps)  # ensure steps are in ascending order
    stepping_points = np.ediff1d(steps, to_end=[0])/2.0 + steps
    stepped_array = np_ma_zeros_like(array, mask=array.mask)
    low = None
    for level, high in zip(steps, stepping_points):
        if low is None:
            matching = (-high < array) & (array <= high)
        else:
            matching = (low < array) & (array <= high)
        stepped_array[matching] = level
        low = high
    # all the remaining values are above the top step level
    stepped_array[low < array] = level
    stepped_array.mask = np.ma.getmaskarray(array)
    if step_at == 'midpoint':
        # our work here is done
        return stepped_array

    '''
    A note about how this works:

    We've found the midpoints of each transition and we have an array which
    has forced the array to the nearest steps. We now need to move forward or
    backward from the midpoint to find the start of the transition
    'move_start' or the end of the transition 'move_stop'.

    Where possible, we use the rate of change of the parameter to determine
    where the transition to the next step starts / stops. Sometimes this
    isn't very effective (for very progressive state changes with low rate of
    change), in which case we seek for where the state crossed the next step
    value (see next paragraph). Failing both of these options, we use the
    flap midpoint determined as the first step to ensure we don't go beyond
    two steps changes worth.

    Depending on the direction of travel (increasing / decreasing) determines
    how close to the next setting we will get (5% of the difference between
    flap settings under for increasing steps, 5% over for decreasing). This
    is why you may see slightly early transitions, however this value was
    found to be the perfect balance of accounting for parameters that do not
    sit at the desired value and accounting for the slight transition delay.

    If increasing we step early for 'including_transition' and step late for
    decreasing so that the entire next step and the transition period are
    included as the next step.

    The opposite happens for 'excluding_transition' so that the transitions
    are ignored until the next step is fully established.
    '''
    # create new array, initialised with first flap setting
    new_array = np.ones_like(array.data) * first_valid_sample(stepped_array).value

    # create a list of tuples with index of midpoint change and direction of
    # travel
    flap_increase = find_edges(stepped_array, direction='rising_edges')
    flap_decrease = find_edges(stepped_array, direction='falling_edges')

    transitions = [(idx, 'increase') for idx in flap_increase] + \
                  [(idx, 'decrease') for idx in flap_decrease]

    if not transitions:
        logger.warning("No changes between steps could be found in step_values.")
        return np.ma.array(new_array)

    # sort based on index
    sorted_transitions = sorted(transitions, key=lambda v: v[0])
    flap_changes = [idx for idx, direction in sorted_transitions]

    roc = rate_of_change_array(array, hz)

    for prev_midpoint, (flap_midpoint, direction), next_midpoint in\
        zip_longest( [0] + flap_changes[0:-1], sorted_transitions,
                     flap_changes[1:]):
        prev_flap = prev_unmasked_value(stepped_array, int(floor(flap_midpoint)),
                                        start_index=int(floor(prev_midpoint)))
        stop_index = int(ceil(next_midpoint)) if next_midpoint else None
        next_flap = next_unmasked_value(stepped_array, int(ceil(flap_midpoint)),
                                        stop_index=stop_index).value
        is_masked = (array[int(floor(flap_midpoint))] is np.ma.masked or
                     array[int(ceil(flap_midpoint))] is np.ma.masked)
        if is_masked:
            new_array[int(floor(prev_midpoint)):int(floor(prev_flap.index))] = prev_flap.value
            prev_midpoint = prev_flap.index

        if direction == 'increase':
            # looking for where positive change reduces to this value
            roc_to_seek_for = 0.1
        else:
            # looking for negative roc reduces to this value
            roc_to_seek_for = -0.1

        # allow a change to be 5% before the flap is reached
        flap_tolerance = (abs(prev_flap.value - next_flap) * 0.05)

        if (is_masked and direction == 'decrease'
            or step_at == 'move_start'
            or direction == 'increase' and step_at == 'including_transition'
            or direction == 'decrease' and step_at == 'excluding_transition'):
            #TODO: support within 0.1 rather than 90%
            # prev_midpoint (scan stop) should be after the other scan transition...
            scan_rev = slice(flap_midpoint, prev_midpoint, -1)
            ### 0.975 is within 0.1 of flap 40 and 0.
            ##idx = index_at_value_or_level_off(array, prev_flap, scan_rev,
                                              ##abs_threshold=0.2)
            if direction == 'decrease':
                flap_tolerance *= -1

            roc_idx = index_at_value(roc, roc_to_seek_for, scan_rev)
            val_idx = index_at_value(array, prev_flap.value + flap_tolerance, scan_rev)
            idxs = [x for x in (roc_idx, val_idx) if x]
            idx = max(idxs) if idxs else flap_midpoint

        elif (is_masked and direction == 'increase'
              or step_at == 'move_stop'
              or direction == 'increase' and step_at == 'excluding_transition'
              or direction == 'decrease' and step_at == 'including_transition'):
            scan_fwd = slice(int(flap_midpoint), next_midpoint, +1)

            if direction == 'increase':
                flap_tolerance *= -1

            roc_idx = index_at_value(roc, roc_to_seek_for, scan_fwd)
            val_idx = index_at_value(array, next_flap + flap_tolerance, scan_fwd)
            # Rate of change is preferred when the parameter flattens out,
            # value is used when transitioning between two states and the
            # parameter does not level.
            idxs = [x for x in (val_idx, roc_idx) if x is not None]
            idx = (idxs and min(idxs)) or flap_midpoint

        # floor +1 to ensure transitions start at the next sample
        new_array[int(floor(idx))+1:] = next_flap

    # Mask edges of array to avoid extrapolation.
    finished_array = np.ma.array(new_array, mask=mask_edges(array))

    '''
    import matplotlib.pyplot as plt
    plt.plot(array)
    plt.plot(finished_array)
    plt.show()
    '''

    return finished_array


def touchdown_inertial(land, roc, alt):
    """
    For aircraft without weight on wheels switches, or if there is a problem
    with the switch for this landing, we do a local integration of the
    inertial rate of climb to estimate the actual point of landing. This is
    referenced to the available altitude signal, Altitude AAL, which will
    have been derived from the best available source. This technique leads on
    to the rate of descent at landing KPV which can then make the best
    calculation of the landing ROD as we know more accurately the time where
    the mainwheels touched.

    :param land: Landing period
    :type land: slice
    :param roc: inertial rate of climb
    :type roc: Numpy masked array
    :param alt: altitude aal
    :type alt: Numpy masked array

    :returns: index, rod
    :param index: index within landing period
    :type index: integer
    :param rod: rate of descent at touchdown
    :type rod: float, units fpm
    """
    # Time constant of 6 seconds.
    tau = 1/6.0
    # Make space for the integrand
    startpoint = land.start_edge
    endpoint = land.stop_edge
    sm_ht = np_ma_zeros_like(roc.array[startpoint:endpoint])
    # Repair the source data (otherwise we propogate masked data)
    my_roc = repair_mask(roc.array[startpoint:endpoint])
    my_alt = repair_mask(alt.array[startpoint:endpoint])

    # Start at the beginning...
    sm_ht[0] = alt.array[startpoint]
    #...and calculate each with a weighted correction factor.
    for i in range(1, len(sm_ht)):  # FIXME: Slow - esp. when landing covers a large period - perhaps second check that altitude is sensible?
        sm_ht[i] = (1.0-tau)*sm_ht[i-1] + tau*my_alt[i-1] + my_roc[i]/60.0/roc.hz


    '''
    # Plot for ease of inspection during development.
    from analysis_engine.plot_flight import plot_parameter
    plot_parameter(alt.array[startpoint:endpoint], show=False)
    plot_parameter(roc.array[startpoint:endpoint]/100.0, show=False)
    #plot_parameter(on_gnd.array[startpoint:endpoint], show=False)
    plot_parameter(sm_ht)
    '''

    # Find where the smoothed height touches zero and hence the rod at this
    # point. Note that this may differ slightly from the touchdown measured
    # using wheel switches.
    index = index_at_value(sm_ht, 0.0)
    if index:
        roc_tdn = my_roc[index]
        return Value(index + startpoint, roc_tdn)
    else:
        return Value(None, None)


def track_linking(pos, local_pos):
    """
    Obtain corrected tracks from takeoff phase, final approach and landing
    phase and possible intermediate approach and go-around phases, and
    compute error terms to align the recorded lat&long with each partial data
    segment.

    Takes an array of latitude or longitude position data and the equvalent
    array of local position data from ILS localizer and synthetic takeoff
    data.

    :param pos: Flight track data (latitude or longitude) in degrees.
    :type pos: np.ma.masked_array, masked from data validity tests.
    :param local_pos: Position data relating to runway or ILS.
    :type local_pos: np.ma.masked_array, masked where no local data computed.

    :returns: Position array using local_pos data where available and interpolated pos data elsewhere.

    TODO: Include last valid sample style functions to avoid trap of adjusting at a masked value.
    """
    # Where do we need to use the raw data?
    blocks = np.ma.clump_masked(local_pos)
    last = len(local_pos)

    for block in blocks:
        # Setup local variables
        a = block.start
        b = block.stop
        adj_a = 0.0
        adj_b = 0.0
        link_a = 0
        link_b = 0

        # Look at the first edge
        if a==0:
            link_a = 1
        else:
            adj_a = (local_pos[a-1] - pos[a-1]) or 0.0

        # now the other end
        if b==last:
            link_b = 1
        else:
            adj_b = (local_pos[b] - pos[b]) or 0.0

        fix_a = adj_a + link_a*adj_b
        fix_b = adj_b + link_b*adj_a

        if link_a ==1 or link_b == 1:
            fix = np.linspace(fix_a, fix_b, num=b-a)
        else:
            fix = np.linspace(fix_a, fix_b, num=b-a+2)[1:-1]
        local_pos[block] = pos[block] + fix
    return local_pos


def smooth_track_cost_function(lat_s, lon_s, lat, lon, ac_type, hz):
    # Summing the errors from the recorded data is easy.
    from_data = np.sum((lat_s - lat)**2)+np.sum((lon_s - lon)**2)

    # The errors from a straight line are computed swiftly using convolve.
    slider=np.array([-1,2,-1])
    from_straight = np.sum(np.convolve(lat_s,slider,'valid')**2) + \
        np.sum(np.convolve(lon_s,slider,'valid')**2)

    if ac_type and ac_type.value=='helicopter':
        weight = 100 # As helicopters fly more slowly so we don't need such smoothing.
    elif hz == 1.0:
        weight = 1000
    elif hz == 0.5:
        weight = 300
    elif hz == 0.25:
        weight = 100
    else:
        raise ValueError('Lat/Lon sample rate not recognised in smooth_track_cost_function.')

    cost = from_data + weight*from_straight
    return cost


def smooth_signal(array, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        array: the input signal array to be smoothed
        window_len: the dimension of the smoothing window and should be an odd
                    integer
        window: the type of window from:
            'flat' - window will produce a moving average smoothing.
            'hanning'
            'hamming'
            'bartlett'
            'blackman'

    output:
        the smoothed signal

    Method from http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """
    if array.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if array.size < window_len:
        return array

    if window_len < 3:
        return array

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', "
                         "'bartlett', 'blackman'")

    s=np.r_[array[window_len-1:0:-1], array, array[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len, np.float64)
    else:
        win_func = getattr(np, window, None)
        w = win_func(window_len)
    out = np.convolve(w/w.sum(), s, mode='valid')
    # Trim the extra elements of the array to make the returned array the same
    # length. The example suggest using:
    #    "return y[(window_len/2-1):-(window_len/2)]"
    # This left the array size 1 element too big , the excess has been trim
    # off the end of the array.
    if out.size > array.size:
        extra = out.size - array.size
        extra_start = window_len//2-1
    return np.ma.MaskedArray(out[extra_start:-(extra-extra_start)], array.mask)


def smooth_track(lat, lon, ac_type, hz):
    """
    Input:
    lat = Recorded latitude array
    lon = Recorded longitude array
    ac_type = aircraft type (aeroplane or helicopter)
    hz = sample rate

    Returns:
    lat_last = Optimised latitude array
    lon_last = optimised longitude array
    Cost = cost function, used for testing satisfactory convergence.
    """

    if len(lat) <= 5:
        return lat, lon, 0.0 # Polite return of data too short to smooth.

    lat_s = np.ma.copy(lat)
    lon_s = np.ma.copy(lon)

    # Set up a weighted array that will slide past the data.
    r = 0.7
    # Values of r alter the speed to converge; 0.7 seems best.
    slider = np.ma.ones(5)*r/4
    slider[2] = 1-r

    cost_0 = float('inf')
    cost = smooth_track_cost_function(lat_s, lon_s, lat, lon, ac_type, hz)

    while cost < cost_0:  # Iterate to an optimal solution.
        lat_last = np.ma.copy(lat_s)
        lon_last = np.ma.copy(lon_s)

        # Straighten out the middle of the arrays, leaving the ends unchanged.
        lat_s.data[2:-2] = np.convolve(lat_last,slider,'valid')
        lon_s.data[2:-2] = np.convolve(lon_last,slider,'valid')

        cost_0 = cost
        cost = smooth_track_cost_function(lat_s, lon_s, lat, lon, ac_type, hz)

    if cost>0.1:
        logger.warning("Smooth Track Cost Function closed with cost %f.3",cost)

    return lat_last, lon_last, cost_0

def straighten_altitudes(fine_array, coarse_array, limit, copy=False):
    '''
    Like straighten headings, this takes an array and removes jumps, however
    in this case it is the fine altimeter rollovers that get corrected.

    In the original format, we kept the signal in step with the coarse
    altimeter signal without relying upon that for accuracy, but now the fine
    signal is straightened before removing spikes and the alignment is
    carried out in match_altitudes.
    '''
    return straighten(fine_array, coarse_array, limit, copy)

def match_altitudes(fine, coarse):
    '''
    This function is specific to old altimetry systems which had fine and
    coarse potentiometers. The coarse pot had a range of 135,000ft (yes!) and
    the fine pot covered 5,000ft. The difficulty is that there is no
    certainty what the coarse value will be when the fine pot rolls over
    (unlike digital systems where the coarse and fine parts originate from
    the same binary value).

    The fine part is straightened early in the processing so that spikes can
    be corrected using the normal validation processes, but as we start from
    an arbitrary turn of the potentiometer, we can be any multiple of 5000 ft
    out from the true altitude.

    This function compares the two, then uses the correlation function to
    determine the best fit height adjustment. This is snapped onto the
    nearest 5000ft value and used to correct the altitude(fine) based
    readings.

    The process works in valid data blocks as the offset will have been reset
    during the calculation of the fine part in the presence of data spikes.
    '''

    fine.mask = np.ma.getmaskarray(coarse) | np.ma.getmaskarray(fine)
    chunks = np.ma.clump_unmasked(fine)
    big_chunks = slices_remove_small_slices(chunks, count=2)
    result = np_ma_masked_zeros_like(fine)
    for chunk in big_chunks:
        av_diff = np.average(fine.data[chunk] - coarse.data[chunk])
        correction = round(av_diff/5000.0)*5000.0
        result[chunk] = fine[chunk]-correction
    return result

def straighten_headings(heading_array, copy=True):
    '''
    We always straighten heading data before checking for spikes.
    It's easier to process heading data in this format.

    :param heading_array: array/list of numeric heading values
    :type heading_array: iterable
    :returns: Straightened headings
    :rtype: Generator of type Float
    '''
    return straighten(heading_array, None, 360.0, copy)

def straighten(array, estimate, limit, copy):
    '''
    Basic straightening routine, used by both heading and altitude signals.

    :param array: array of numeric of overflowing values
    :type array: numpy masked array
    :param limit: limit value for overflow.
    :type limit: float
    :returns: Straightened parameter
    :rtype: numpy masked array
    '''
    if copy:
        array = array.copy()
    last_value = None
    for clump in np.ma.clump_unmasked(array):
        start_value = array[clump.start]
        if estimate is not None and estimate[clump.start]:
            # make sure we are close to the estimate at the start of each block
            offset = estimate[clump.start] - start_value
            if offset>0.0:
                start_value += int(floor(offset / limit + 0.5) * limit)
            else:
                start_value += int(ceil(offset / limit - 0.5) * limit)
        else:
            if last_value is not None:
                # shift array section to be consistent with previous
                start_value += limit * np.round((last_value - start_value) / limit)

        diff = np.ediff1d(array[clump])
        diff -= limit * np.trunc(diff * 2.0 / limit)
        array[clump][0] = start_value
        array[clump][1:] = np.cumsum(diff) + start_value
        last_value = array[clump][-1]
    return array


def straighten_overflows(array, min_val, max_val, threshold=4):
    '''
    Straighten a signal which overflows when the value exceeds a specified
    range. Threshold is specified as a fraction of the full range
    (total_range / threshold, e.g. 200 / 4 = 50). The threshold is used to
    avoid missing data and data spikes causing unwanted overflows.

    :param array: Signal to straighten.
    :type array: np.ma.masked_array
    :param min_val: Minimum value of parameter before overflow occurs.
    :type min_val: int or float
    :param max_val: Maximum value of parameter before overflow occurs.
    :type max_val: int or float
    :param threshold: Threshold specified as a fraction of the total range.
    :type threshold: int or float
    :returns: Straightened signal.
    :rtype: np.ma.masked_array
    '''
    straight = array.copy()

    if max_val <= min_val:
        raise ValueError('Invalid range: %s to %s', min_val, max_val)

    total_range = max_val - min_val

    partition = total_range / threshold
    upper = max_val - partition
    lower = min_val + partition

    last_value = None
    for unmasked_slice in np.ma.clump_unmasked(array):
        overflow = 0
        first_value = array[unmasked_slice.start]
        if last_value is not None:
            # Check if overflow occurred within masked region.
            if last_value < lower and first_value > upper:
                overflow -= 1
            elif last_value > upper and first_value < lower:
                overflow += 1
            # correct the rest of the array
            straight[unmasked_slice.start:] += overflow * total_range

        # locate overflows and shift the arrays
        diff = np.ediff1d(array[unmasked_slice])
        abs_diff = np.abs(diff)

        for idx in np.where(abs_diff > total_range - partition)[0]:
            if (idx + 1 < len(abs_diff) and
                (abs_diff[idx - 1] > partition * 2 or
                 abs_diff[idx + 1] > partition * 2)):
                # ignore data spike
                continue
            sign = np.sign(diff[idx])
            overflow -= sign
            straight[unmasked_slice.start + idx + 1:] -= sign * total_range

        last_value = array[unmasked_slice.stop - 1]

    return straight


def straighten_longitude(array):
    return straighten_overflows(array, -180, 180, threshold=32)


def subslice(orig, new):
    """
    a = slice(2,10,2)
    b = slice(2,2)
    c = subslice(a, b)
    assert range(100)[c] == range(100)[a][b]

    See tests for capabilities.
    """
    step = (orig.step or 1) * (new.step or 1)

    # FIXME: asks DJ
    # Inelegant fix for one special case. Sorry, Glen.
    if new.start == 0:
        start = orig.start
    else:
        start = (orig.start or 0) + (new.start or orig.start or 0) * (orig.step or 1)

    stop = orig.stop if new.stop is None else \
        (orig.start or 0) + (new.stop or orig.stop or 0) * (orig.step or 1) # the bit after "+" isn't quite right!!

    return slice(start, stop, None if step == 1 else step)


def index_at_distance(distance, index_ref, latitude_ref, longitude_ref, latitude, longitude, hz):
    '''
    This routine computes the index into arrays latitude and longitude that
    is a specified distance from the reference point.

    :param distance: Distance from the reference point required.
    :type distance: int, units nautical miles
    :param index_ref: Index into the latitude and longitude arrays at reference point
    :type index_ref: int. Note: This is only a help to speed the algorithm; accuracy is not important.
    :param latitude_ref: Latitude of the reference point
    :type latitude_ref: float, degrees latitude
    :param longitude_ref: Longitude of the reference point
    :type longitude_ref: float, degrees longitude
    :param latitude: Latitude of the aircraft track
    :type latitude: np.ma.array
    :param longitude: Longitude of the aircraft track
    :type longitude: np.ma.array
    :param hz: Sample rate of latitude and longitude arrays
    :type hz: float

    :returns: Index into the latitude and longitude arrays
    :rtype: float
    :raises: ValueError
    '''

    def distance_error(index, *args):
        radius = args[0]
        latitude_ref = args[1]
        longitude_ref = args[2]
        latitude = args[3]
        longitude = args[4]

        rad = distance_at_index(index[0], latitude, longitude, latitude_ref, longitude_ref)
        # and simply squaring the error allows robust minimum searching.
        error = (rad - radius) ** 2.0
        return error

    # It is natural to hold the distance as an integer, but locally we need a float
    _distance = float(distance)

    # We start by guessing an index based on some sample results.
    #   the distance estimate
    abs_d = abs(_distance)
    if abs_d < 10:
        # Flown at low speed; about a minute for the last two miles
        secs = abs_d * 30.0
    else:
        # More distant ranges at higher speeds
        secs = abs_d * 15
    guess = copysign(secs * hz, _distance)
    end_data = np.ma.flatnotmasked_edges(latitude)[1]-60
    estimate = max(0, min(index_ref + guess, end_data))

    # By constraining the boundaries we ensure the iteration does not
    # stray outside the available array.
    boundaries = [(0, end_data)]

    kti = optimize.fmin_l_bfgs_b(
        distance_error, estimate,
        fprime=None,
        args = (
            abs(_distance),
            latitude_ref,
            longitude_ref,
            latitude,
            longitude,
            index_ref,
        ),
        approx_grad=True,
        epsilon=0.005,
        factr=1e8,
        bounds=boundaries, maxfun=100)

    solution_index = kti[0][0]
    # To find out if this was a good answer, we compute the actual range and compare the error.
    solution_range = distance_at_index(solution_index, latitude, longitude, latitude_ref, longitude_ref)
    _dist_err = abs(abs(distance) - solution_range)

    error = kti[2]['warnflag']
    if solution_index == boundaries[0][0] or solution_index == boundaries[0][1]:
        logger.warning('Attempted to scan further than data permits.')
        return
    elif _dist_err > 0.02:
        # This can happen if the flight is too short (i.e. less than 250nm if that is the distance requested)
        # It can also arise if the wrong runway has been identified, so the actual position is the closest point
        # on the approach to the correct (but unidentified) runway.
        logger.warning('Converged on the wrong minimum.')
        return
    elif error:
        logger.warning('Returned early from iteration algorithm: %d', error)
        # ...but the solution is still good, so return this...
        return solution_index
    else:
        return solution_index


def distance_at_index(i, latitude, longitude, latitude_ref, longitude_ref):
    try:
        lon_i = value_at_index(longitude, i) or 0.0
    except ValueError:
        lon_i = 0.0
    try:
        lat_i = value_at_index(latitude, i) or 0.0
    except ValueError:
        lat_i = 0.0

    return  great_circle_distance__haversine(lat_i, lon_i, latitude_ref, longitude_ref, units=ut.NM)


def index_closest_value(array, threshold, _slice=slice(None)):
    '''
    This function seeks the moment when the parameter in question gets
    closest to a threshold. It works both forwards and backwards in time. See
    index_at_value for further details.
    '''
    return index_at_value(array, threshold, _slice, endpoint='closing')


def index_at_value(array, threshold, _slice=slice(None), endpoint='exact'):
    '''
    This function seeks the moment when the parameter in question first crosses
    a threshold. It works both forwards and backwards in time. To scan backwards
    pass in a slice with a negative step. This is really useful for finding
    things like the point of landing.

    For example, to find 50ft Rad Alt on the descent, use something like:
       idx_50 = index_at_value(alt_rad, 50.0, slice(on_gnd_idx,0,-1))

    :param array: input data
    :type array: masked array
    :param threshold: the value that we expect the array to cross in this slice.
    :type threshold: float
    :param _slice: slice where we want to seek the threshold transit.
    :type _slice: slice
    :param endpoint: type of end condition being sought.
    :type endpoint: string 'exact' requires array to pass through the threshold,
    while 'closing' seeks the last point where the array is closing on the
    threshold and 'nearest' seeks the point nearest to the threshold.
    'first_closing' seeks the first point where the array reaches the closest value.

    :returns: interpolated time when the array values crossed the threshold. (One value only).
    :returns type: Float or None
    '''
    assert endpoint in ['exact', 'closing', 'nearest', 'first_closing']
    step = _slice.step or 1
    max_index = len(array)

    # Arrange the limits of our scan, ensuring that we stay inside the array.
    if step == 1:
        begin = max(int(py2round(_slice.start or 0)), 0)
        end = min(int(py2round(_slice.stop or max_index)), max_index)
        left, right = slice(begin, end - 1, step), slice(begin + 1, end,step)

    elif step == -1:
        begin = min(int(py2round(_slice.start or max_index)), max_index-1)
        # Indexing from the end of the array results in an array length
        # mismatch. There is a failing test to cover this case which may work
        # with array[:end:-1] construct, but using slices appears insoluble.
        end = max(int(_slice.stop or 0),0)
        left = slice(begin, end, step)
        right = slice(begin - 1, end - 1 if end > 0 else None, step)

    else:
        raise ValueError('Step length not 1 in index_at_value')

    if begin == end:
        logger.warning('No range for seek function to scan across')
        return None

    # When the data being tested passes the value we are seeking, the
    # difference between the data and the value will change sign.
    # Therefore a negative value indicates where value has been passed.
    value_passing_array = (array[left] - threshold) * (array[right] - threshold)
    test_array = np.ma.masked_greater(value_passing_array, 0.0)

    if len(test_array) == 0:
        # Q: Does this mean that value_passing_array is also empty?
        return None

    if (_slice.stop == _slice.start) and (_slice.start is not None):
        # No range to scan across. Special case of slice(None, None, None)
        # covers the whole array so is allowed.
        return None

    elif not np.ma.count(test_array) or np.ma.all(np.isnan(test_array)):
        # The parameter does not pass through threshold in the period in
        # question, so return empty-handed.
        if endpoint in ['closing', 'first_closing']:
            # Rescan the data to find the last point where the array data is
            # closing.
            diff = np.ma.ediff1d(array[slices_int(_slice)])
            if _slice.step is not None and _slice.step >= 0:
                start_index = _slice.start
                stop_index = _slice.stop
            else:
                start_index = _slice.stop
                stop_index = _slice.start
            value = closest_unmasked_value(array, _slice.start or 0,
                                           start_index=start_index,
                                           stop_index=stop_index)
            if value:
                value = value.value
            else:
                return None

            if endpoint == 'closing':
                if threshold >= value:
                    diff_where = np.ma.where(diff < 0)
                else:
                    diff_where = np.ma.where(diff > 0)
            elif endpoint == 'first_closing':
                if threshold >= value:
                    diff_where = np.ma.where(diff <= 0)
                else:
                    diff_where = np.ma.where(diff >= 0)
            else:
                raise 'Unrecognised command in index_at_value'

            try:
                return (_slice.start or 0) + (step * diff_where[0][0])
            except IndexError:
                if start_index is None or stop_index is None:
                    return len(array) - 1
                else:
                    if step==1:
                        return (_slice.stop - step)
                    else:
                        return _slice.stop

        elif endpoint == 'nearest':
            closing_array = abs(array-threshold)
            return begin + step * np.ma.argmin(closing_array[slices_int(_slice)])
        else:
            return None  #TODO: raise exception when not found?
    else:
        n, dummy = np.ma.flatnotmasked_edges(test_array)
        a = array[begin + (step * n)]
        b = array[begin + (step * (n + 1))]
        # Force threshold to float as often passed as an integer.
        # Also check for b=a as otherwise we get a divide by zero condition.
        if (a is np.ma.masked or b is np.ma.masked or np.isnan(a) or np.isnan(b) or a == b):
            r = 0.5
        else:
            r = (float(threshold) - a) / (b - a)

    return (begin + step * (n + r))


def index_at_value_or_level_off(array, frequency, value, _slice, abs_threshold=None):
    '''
    Find the index closest to the value unless it doesn't get within 10% of
    that value or the value +/- the abs_threshold, in which case find the
    point of level off.

    Designed for finding sections around Go Arounds where the
    _slice region defines the area to search within.

    Negative step in slice supported.

    :param array: Normally an Altitude based array.
    :type array: np.ma.array
    :param value: Value to seek to
    :type value: Float
    :param _slice: Constraint within array to search until
    :type _slice: slice
    :param abs_threshold: The absolute threshold which the value must be within.
    :type abs_threshold: float
    :returns: Index at closest value or at level off
    :rtype: Int
    '''
    index = index_at_value(array, value, _slice, 'nearest')
    # did we get within 90% of the threshold?
    if abs_threshold is None:
        abs_threshold = value * 0.1
    if index is not None and abs(value_at_index(array, index) - value) < abs_threshold:
        return index
    else:
        # we never got quite close enough to 2000ft above the
        # minimum go around altitude. Find the top of the climb.
        return find_level_off(array, frequency, _slice)

def _value(array, _slice, operator, start_edge=None, stop_edge=None):
    """
    Applies logic of min_value and max_value across the array slice.
    """
    start_result = np.nan
    stop_result = np.nan
    slice_start = _slice.start
    slice_stop = _slice.stop

    # have we got sections or slices
    if slice_start and slice_start % 1:
        start_edge = slice_start
        slice_start = int(ceil(slice_start))
    if slice_stop and slice_stop % 1:
        stop_edge = slice_stop
        slice_stop = int(floor(slice_stop))

    values = []

    search_slice = slices_int(slice_start, slice_stop, _slice.step)

    if _slice.step and _slice.step < 0:
        raise ValueError("Negative step not supported")
    if np.ma.count(array[search_slice]):
        # get start_edge and stop_edge values if required
        if start_edge:
            start_result = value_at_index(array, start_edge)
            if start_result is not None and start_result is not np.ma.masked:
                values.append((start_result, start_edge))
        # floor the start position as it will have been floored during the slice
        value_index = int(operator(array[search_slice]) + int(floor(search_slice.start or 0)) * (search_slice.step or 1))
        value = array[value_index]
        values.append((value, value_index))
        if stop_edge:
            stop_result = value_at_index(array, stop_edge)
            if stop_result is not None and stop_result is not np.ma.masked:
                values.append((stop_result, stop_edge))
        result_idx = operator(np.ma.array(values)[:,0])
        return Value(values[result_idx][1], values[result_idx][0])

    else:
        return Value(None, None)


def value_at_time(array, hz, offset, time_index):
    '''
    Finds the value of the data in array at the time given by the time_index.

    :param array: input data
    :type array: masked array
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param offset: fdr offset for the array (sec)
    :type offset: float
    :param time_index: time into the array where we want to find the array value.
    :type time_index: float
    :returns: interpolated value from the array
    :raises ValueError: From value_at_index if time_index is outside of array range.
    '''
    # Timedelta truncates to 6 digits, therefore round offset down.
    time_into_array = time_index - round(offset - 0.0000005, 6)
    location_in_array = time_into_array * hz

    # Trap overruns which arise from compensation for timing offsets.
    diff = location_in_array - len(array)
    if location_in_array < 0:
        location_in_array = 0
    if diff > 0:
        location_in_array = len(array) - 1

    return value_at_index(array, location_in_array)


def value_at_datetime(start_datetime, array, hz, offset, value_datetime):
    '''
    Finds the value of the data in array at the time given by value_datetime.

    :param start_datetime: Start datetime of data.
    :type start_datetime: datetime
    :param array: input data
    :type array: masked array
    :param hz: sample rate for the input data (sec-1)
    :type hz: float
    :param offset: fdr offset for the array (sec)
    :type offset: float
    :param value_datetime: Datetime to fetch the value for.
    :type value_datetime: datetime
    :returns: interpolated value from the array
    :raises ValueError: From value_at_index if value_datetime is outside of array range.
    '''
    value_timedelta = value_datetime - start_datetime
    seconds = value_timedelta.total_seconds()
    return value_at_time(array, hz, offset, seconds)


def value_at_index(array, index, interpolate=True):
    '''
    Finds the value of the data in array at a given index.

    Samples outside the array boundaries are permitted, as we need this to
    allow for offsets within the data frame.

    :param array: input data
    :type array: masked array or list
    :param index: index into the array where we want to find the array value.
    :type index: float
    :param interpolate: whether to interpolate the value if index is float.
    :type interpolate: boolean
    :returns: interpolated value from the array
    '''

    if index is None or index < 0.0:
        return array[0]
    elif index > len(array) - 1:
        return array[-1]

    low = int(index)
    if low == index:
        # I happen to have arrived at exactly the right value by a fluke...
        return None if np.ma.is_masked(array[low]) else array[low]
    else:
        high = low + 1
        r = index - low
        if isinstance(array, np.ma.masked_array):
            low_value = array.data[low]
            high_value = array.data[high]
            # Crude handling of masked values. TODO: Must be a better way !
            if array.mask.any(): # An element is masked
                if array.mask[low]:
                    if array.mask[high]:
                        return None
                    else:
                        return high_value
                else:
                    if array.mask[high]:
                        return low_value
        else:
            low_value = array[low]
            high_value = array[high]
            if low_value is None and high_value is None:
                return None
            elif low_value is None and high_value is not None:
                return high_value
            elif high_value is None and low_value is not None:
                return low_value
        # If not interpolating and no mask or masked samples:
        if not interpolate:
            return array[int(index + 0.5)]
        # In the cases of no mask, or neither sample masked, interpolate.
        return r * high_value + (1 - r) * low_value


def vstack_params(*params):
    '''
    Create a multi-dimensional masked array with a dimension per param.

    :param params: Parameter arguments as required. Allows some None values.
    :type params: np.ma.array or Parameter object or None
    :returns: Each parameter stacked onto a new dimension
    :rtype: np.ma.array
    :raises: ValueError if all params are None (concatenation of zero-length sequences is impossible)
    '''
    return np.ma.vstack([getattr(p, 'array', p) for p in params if p is not None])


def vstack_params_filtered(window, *params, **kw):
    '''
    Create a multi-dimensional masked array with a dimension per param.

    The arrays are averaged in `window` seconds.

    :param params: Parameter arguments as required. Allows some None values.
    :type params: np.ma.array or Parameter object or None
    :returns: Each parameter stacked onto a new dimension
    :rtype: np.ma.array
    :raises: ValueError if all params are None (concatenation of zero-length sequences is impossible)
    '''
    method = kw.get('method')
    if method not in ('moving_average', 'second_window'):
        raise ValueError(
            'Only `moving_average`, `second_window` filtering methods are '
            'currently supported')

    params = [p for p in params if p is not None]

    if method == 'moving_average':
        window = int(window * params[0].frequency)
        if not window % 2:
            window += 1
        if window > 1:
            arrays = [moving_average(p.array, window) for p in params]
        else:
            [p.array for p in params]
    elif method == 'second_window':
        arrays = [
            second_window(p.array, p.hz, window, extend_window=True) for p in params]

    return np.ma.vstack(arrays)


def vstack_params_avg(window, *params):
    return vstack_params_filtered(window, *params, method='moving_average')


def vstack_params_sw(window, *params):
    return vstack_params_filtered(window, *params, method='second_window')


def vstack_params_where_state(*param_states):
    '''
    Create a multi-dimensional masked array with a dimension for each param,
    where the state is equal to that provided.

    res = vstack_params_where_state(
        (tcas_adv_up, 'Up'),
        (tcas_combined_control, 'Down'),
        )
    # looks like this:
    [[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 'Up'
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 'Down'


    :param param_states: tuples containing params or array and multistate value to match with. Allows None parameters.
    :type param_states: np.ma.array or Parameter object or None
    :returns: Each parameter stacked onto a new dimension
    :rtype: np.ma.array
    :raises: ValueError if all params are None (concatenation of zero-length sequences is impossible)
    '''
    param_arrays = []
    for param, state in param_states:
        if param is None:
            continue
        if state in param.array.state:
            array = getattr(param, 'array', param)
            param_arrays.append(array == state)
        else:
            logger.warning("State '%s' not found in param '%s'", state, param.name)
    return np.ma.vstack(param_arrays)


def second_window(array, frequency, seconds, extend_window=False):
    '''
    Only include values which are maintained for a number of seconds, shorter
    exceedances are excluded.

    Only supports odd numbers of seconds when frequency is 1.

    e.g. [0, 1, 2, 3, 2, 1, 2, 3] -> [0, 1, 2, 2, 2, 2, 2, 2]

    :param array: ...
    :type array: np.ma.masked_array
    :param frequncy: frequency of the array data
    :type frequency: float or int
    :param seconds: window size in seconds
    :type seconds: float or int
    :param extend_window: extend window to next boundary.
    :type extend_window: bool
    '''
    min_window_size = 2.0 / frequency
    if modulo(seconds, min_window_size) != 0:
        if extend_window:
            seconds = seconds - seconds % min_window_size + min_window_size
        else:
            raise ValueError('%s seconds is not valid for the frequency %s Hz.\n'
                             'Value of seconds must be a multiple of 2 / frequency.'
                             % (seconds, frequency))

    samples = int(frequency * seconds)

    window_array = np_ma_masked_zeros_like(array)

    if array.size < samples:
        # Array size is less than the window sample size.
        return window_array

    # Make a view of a sliding window using a different stride.
    # For 3 samples, the array [1, 2, 3, 4, 5, 6, 7, 8, 9] will become
    # [[1, 2, 3],
    #  [2, 3, 4],
    #  [3, 4, 5],
    #  [4, 5, 6],
    #  [5, 6, 7],
    #  [6, 7, 8],
    #  [7, 8, 9]]

    sliding_window = np.lib.stride_tricks.as_strided(
        array, shape=(len(array) - samples, samples + 1),
        strides=array.strides * 2)

    # Calculate min and max over the last axis (for each sliding window)
    # Optimization: convert array to list for indexing (161 ms -> 124 ms)
    min_ = np.min(sliding_window, axis=-1).tolist()
    max_ = np.max(sliding_window, axis=-1).tolist()

    unmasked_slices = np.ma.clump_unmasked(array)

    # Optimization: local namespace (17 ms -> 6 ms)
    window_array_data = window_array.data

    for unmasked_slice in filter_slices_length(unmasked_slices, samples):
        start = unmasked_slice.start
        last_value = array[start]
        window_stop = unmasked_slice.stop - samples
        # We set already the mask to False where we are going to insert
        # values. That is from start up to stop-sample_idx
        # Much faster than setting the value in the masked array item by item
        window_array.mask[start:window_stop] = False
        for idx in range(start, window_stop):
            # Clip the last value between sliding window min and max
            last_value = min(max(last_value, min_[idx]), max_[idx])
            # Set this value in the data object of the masked array.
            # Much faster than using window_array[start+i] = last_value
            window_array_data[idx] = last_value

    '''
    import matplotlib.pyplot as plt
    plt.plot(array)
    plt.plot(window_array)
    plt.show()
    '''

    return window_array

#---------------------------------------------------------------------------
# Air data calculations adapted from AeroCalc V0.11 to suit POLARIS Numpy
# data format. For increased speed, only standard POLARIS units used.
#
# AeroCalc is Copyright (c) 2008, Kevin Horton and used under open source
# license with permission. For copyright notice and disclaimer, please see
# airspeed.py source code in AeroCalc.
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Initialise constants used by the air data algorithms
#---------------------------------------------------------------------------
P0 = 1013.25       # Pressure at sea level, mBar
Rhoref = 1.2250    # Density at sea level, kg/m**3
A0 = 340.2941      # Speed of sound at sea level, m/s
T0 = 288.15        # Sea level temperature 15 C = 288.15 K
L0 = -0.0019812    # Lapse rate C/ft
g = 9.80665        # Acceleration due to gravity, m/s**2
Rd = 287.05307     # Gas constant for dry air, J/kg K
H1 = 36089.0       # Transition from Troposphere to Stratosphere

# Values at 11km:
T11 =  T0 + ut.convert(11000 * L0, ut.METER, ut.FT)
PR11 = (T11 / T0) ** ((-g) / (Rd * L0))
P11 = PR11 * P0

#---------------------------------------------------------------------------
# Added function to correct for non-ISA temperatures. See blog article.
#---------------------------------------------------------------------------

def alt_dev2alt(alt, dev):
    return np.ma.where(
        alt < H1,
        alt * (T0+dev)/T0,
        np.ma.masked)

def from_isa(alt, sat):
    '''
    Temperature variation from ISA standard at given altitude and temperature.
    '''
    if alt < H1:
        return sat - (15.0+L0*alt)
    else:
        return sat - (15.0+L0*H1)

#---------------------------------------------------------------------------
# Computation modules use AeroCalc structure and are called from the Derived
# Parameters as required.
#---------------------------------------------------------------------------

def alt2press(alt_ft):
    press = P0  * alt2press_ratio(alt_ft)
    return press

def alt2press_ratio(alt_ft):
    return np.ma.where(
        alt_ft <= H1,
        _alt2press_ratio_gradient(alt_ft),
        _alt2press_ratio_isothermal(alt_ft),
    )

def air_density(alt, sat):
    pr = alt2press_ratio(alt)
    tr = (sat+T0-15.0)/T0
    dr = pr/tr
    return dr*Rhoref

def cas2dp(cas_kt):
    """
    Convert corrected airspeed to pressure rise (includes allowance for
    compressibility)
    """
    if np.ma.max(cas_kt) > 661.48:
        raise ValueError('Supersonic airspeed compuations not included')
    cas_mps = ut.convert(np.ma.masked_greater(cas_kt, 661.48), ut.KT, ut.METER_S)
    p = ut.convert(P0, ut.MILLIBAR, ut.PASCAL)
    return P0 * (((Rhoref * cas_mps*cas_mps)/(7.* p) + 1.)**3.5 - 1.)

def cas_alt2mach(cas, alt_ft):
    """
    Return the mach that corresponds to a given CAS and altitude.
    """
    dp = cas2dp(cas)
    p = alt2press(alt_ft)
    dp_over_p = dp / p
    mach = dp_over_p2mach(dp_over_p)
    return mach

def dp_over_p2mach(dp_over_p):
    """
    Return the mach number for a given delta p over p. Supersonic results masked as invalid.
    """
    mach_squared = 5.0 * ((dp_over_p + 1.0) ** (2.0/7.0) - 1.0)
    mach = np.ma.sqrt(mach_squared)
    return np.ma.masked_greater_equal(mach, 1.0)

def _dp2speed(dp, P, Rho):

    p = P*100 # pascal not mBar inside the calculation
    # dp / P not changed as we use mBar for pressure dp.
    speed_mps = np.ma.sqrt(((7. * p) * (1. / Rho)) * (
        np.ma.power((dp / P + 1.), 2./7.) - 1.))
    speed_kt = ut.convert(speed_mps, ut.METER_S, ut.KT)

    # Mask speeds over 661.48 kt
    return np.ma.masked_greater(speed_kt, 661.48)


def dp2cas(dp):
    return np.ma.masked_greater(_dp2speed(dp, P0, Rhoref), 661.48)


def dp2tas(dp, alt_ft, sat):
    P = alt2press(alt_ft)
    press_ratio = alt2press_ratio(alt_ft)
    temp_ratio = ut.convert(sat, ut.CELSIUS, ut.KELVIN) / 288.15
    # FIXME: FloatingPointError: underflow encountered in multiply
    density_ratio = press_ratio / temp_ratio
    Rho = Rhoref * density_ratio
    tas = _dp2speed(dp, P, Rho)
    return tas

def alt2sat(alt_ft):
    """ Convert altitude to temperature using lapse rate"""
    return np.ma.where(alt_ft <= H1, 15.0 + L0 * alt_ft, -56.5)

def machtat2sat(mach, tat, recovery_factor=0.995):
    """
    Return the ambient temp, given the mach number, indicated temperature and the
    temperature probe's recovery factor.

    Recovery factor is taken from the BF Goodrich Model 101 and 102 Total
    Temperature Sensors data sheet. As "...the world's leading supplier of
    total temperature sensors" it is likely that a sensor of this type, or
    comparable, will be installed on monitored aircraft.
    """
    # Default fill of zero produces runtime divide by zero errors in Numpy.
    # Hence force fill to >0.
    denominator = np.ma.array(1.0 + (0.2*recovery_factor) * mach * mach)
    ambient_temp = ut.convert(tat, ut.CELSIUS, ut.KELVIN) / denominator
    sat = ut.convert(ambient_temp, ut.KELVIN, ut.CELSIUS)
    return sat

def machsat2tat(mach, sat, recovery_factor=0.995):
    """
    The inverse of machtat2sat, using the same assumptions.
    """
    numerator = np.ma.array(1.0 + (0.2*recovery_factor) * mach * mach)
    ambient_temp = ut.convert(sat, ut.CELSIUS, ut.KELVIN)
    tat = ut.convert(ambient_temp * numerator, ut.KELVIN, ut.CELSIUS)
    return tat

def _alt2press_ratio_gradient(H):
    # From http://www.aerospaceweb.org/question/atmosphere/q0049.shtml
    # Faster to compute than AeroCalc formulae, and pass AeroCalc tests.
    return np.ma.power(1 - H/145442.0, 5.255876)

def _alt2press_ratio_isothermal(H):
    # FIXME: FloatingPointError: overflow encountered in exp
    return 0.223361 * np.ma.exp((36089.0-H)/20806.0)


def press2alt(P):
    """
    Return the altitude corresponding to the pressure.

    Pressure is assumed to be in psi, and height is returned in feet.
    """
    Pmb = ut.convert(P, ut.PSI, ut.MILLIBAR)
    H = np.ma.where(
        Pmb > P11,
        _press2alt_gradient(Pmb),
        _press2alt_isothermal(Pmb),
    )
    return H

def _press2alt_gradient(Pmb):
    return 145442 * (1.0 - np.ma.power(Pmb/P0, 1.0/5.255876))

def _press2alt_isothermal(Pmb):
    return 36089 - np.ma.log((Pmb/P0)/0.223361)*20806

##### TODO: Add memoize caching... Currently breaks tests!
####from flightdatautilities.cache import memoize
####@memoize
def lookup_table(obj, name, _am, _as, _af, _et=None, _es=None):
    '''
    Fetch a lookup table by name for the specified aircraft.

    Handles logging on the passed object if the lookup table is not found.

    :param obj: the node class or instance calling this function.
    :type obj: Node
    :param name: the name of the table to lookup.
    :type name: string
    :param _am: the aircraft model attribute.
    :type _am: Attribute
    :param _as: the aircraft series attribute.
    :type _as: Attribute
    :param _af: the aircraft family attribute.
    :type _af: Attribute
    :param _et: the engine type attribute.
    :type _et: Attribute
    :param _es: the engine series attribute.
    :type _es: Attribute
    :returns: the instantiated velocity speed table.
    :rtype: VelocitySpeed or None
    '''
    attributes = (_am, _as, _af, _et, _es)
    attributes = [(a.value if a else None) for a in attributes]
    try:
        _vs = at.get_vspeed_map(*attributes)()
    except KeyError:
        pass
    else:
        if name in _vs.tables or name in _vs.fallback:
            return _vs
    message = 'No %s table available for '
    message += ', '.join("'%s'" for i in range(len(attributes)))
    obj.warning(message, name, *attributes)
    return None

def filter_runway_heading(r, h):
    rh = r.get('magnetic_heading')
    if not rh:
        logger.warning('No heading information available for runway #%d.', r['id'])
        return
    h1 = h - RUNWAY_HEADING_TOLERANCE
    h2 = h + RUNWAY_HEADING_TOLERANCE
    q1 = h1 + 360 <= rh <= 360 or 0 <= rh <= h if h1 < 0 else h1 <= rh <= h
    q2 = h <= rh <= 360 or 0 <= rh <= h2 % 360 if h2 > 360 else h <= rh <= h2
    return q1 or q2

def nearest_runway(airport, heading, ilsfreq=None, latitude=None, longitude=None, hint=None):
    '''
    Helper to find the nearest runway in the database.

    Attempt to choose a runway depending on provided hints.

    Runway identifier suffixes::

       C = Center
       L = Left
       R = Right
       S = STOL (Short Takeoff & Landing)
       T = True Heading (Not magnetic heading)

    :param airport: The airport to find runways for.
    :type airport: Attribute
    :param heading: The magnetic heading of the aircraft.
    :type heading: str
    :param ilsfreq: The localizer frequency the aircraft is tuned to.
    :type ilsfreq: str
    :param latitude: The latitude to search for.
    :type latitude: str
    :param longitude: The longitude to search for.
    :type longitude: str
    :param hint: Assistance for determining runway selection when imprecise.
    :type hint: str
    :returns: The nearest runway found.
    :rtype: Runway
    :raises: ValueError
    :raises: IndexError
    '''

    def _filter_ilsfreq(r, f):
        rf = r.get('localizer').get('frequency')
        f0 = ilsfreq - RUNWAY_ILSFREQ_TOLERANCE
        f1 = ilsfreq + RUNWAY_ILSFREQ_TOLERANCE
        return (f0 <= rf <= f1) if rf is not None else False

    if not airport:
        return None

    try:
        runways = airport['runways']
    except KeyError:
        logger.warning('No runway information available for airport #%d.', airport['id'])
        return None

    # 1. Attempt to identify the runway by magnetic heading:
    assert 0 <= heading <= 360, u'Heading must be between 0° and 360° degrees.'
    runways = [runway for runway in runways if filter_runway_heading(runway, heading)]
    if len(runways) == 0:
        logger.warning('No runways found at airport #%d for heading %03.1f degrees.', airport['id'], heading)
        return None
    if len(runways) == 1:
        return runways[0]

    # 2. Attempt to identify the runway by localizer frequency:
    if ilsfreq is not None:
        ilsfreq = int(ilsfreq * 1000)  # Convert from MHz to kHz
        if not (108100 <= ilsfreq <= 111950):
            ilsfreq = None
            logger.warning("Localizer frequency '%s' is out-of-range.", ilsfreq)
        elif not (ilsfreq // 100 % 10 % 2):
            ilsfreq = None
            logger.warning("Localiser frequency '%s' must have odd 100 kHz digit.", ilsfreq)
        else:
            x = [runway for runway in runways if _filter_ilsfreq(runway, ilsfreq)]
            if len(x) == 1:
                logger.info("Runway '%s' selected: Identified by ILS.", x[0]['identifier'])
                return x[0]
            elif len(x) == 0:
                logger.warning("ILS '%s' frequency provided, no matching runway found at '%s'.", ilsfreq, airport['id'])
            else:
                logger.warning("ILS '%s' frequency provided, multiple matching runways found at '%s'.", ilsfreq, airport['id'])

    # 3. If hint provided (i.e. not precise positioning) try narrowing down the
    #    runway by heading - if more than one runway within 10 degrees, likely
    #    parallel runways so continue with other means of runway detection.
    if hint is not None:
        # TODO: Compare true heading with one calculated from runway end points?
        assert hint in ('takeoff', 'landing', 'approach')
        for limit in (20, 10):
            x = [runway for runway in runways if abs(float(runway['magnetic_heading']) - heading) < limit]
            if len(x) == 1:
                logger.info("Runway '%s' selected: Only runway within %d degrees of provided heading.", x[0]['identifier'], limit)
                return x[0]

    # 4. Attempt to identify by nearest runway:
    if latitude is not None and longitude is not None:
        assert np.all(-90 <= latitude) and np.all(latitude <= 90), 'Latitude must be between -90 and 90 degrees.'
        assert np.all(-180 < longitude) and np.all(longitude <= 180), 'Longitude must be between -180 and 180 degrees.'
        p3y, p3x = latitude, longitude
        distance = float('inf')
        runway = None
        close_runways = []
        for r in runways:
            p1x = r['start']['longitude']
            p1y = r['start']['latitude']
            p2x = r['end']['longitude']
            p2y = r['end']['latitude']
            args = (p1y, p1x, p2y, p2x, p3y, p3x)
            if not any(args):
                continue
            abs_dxt = np.average(np.abs(cross_track_distance(*args)))
            # if we find more than one runway less than 30m away (30m is half the average runway width), that means we
            # have landed on a runway crossing. W create a list of such runways and then pick the one which magnetic
            # heading is closer to the aircraft heading - since we're looking at the lowest point, even with crosswind
            # the aircraft will likely be post de-crab at this stage.
            if abs_dxt < 30:
                close_runways.append(r)
            if abs_dxt < distance:
                distance = abs_dxt
                runway = r

        if len(close_runways) > 1:
            hdg_diff = float('inf')
            for r in close_runways:
                diff = abs(heading - r.get('magnetic_heading'))
                if diff < hdg_diff:
                    hdg_diff = diff
                    runway = r

        if runway:
            logger.info("Runway '%s' selected: Closest to provided coordinates.", runway['identifier'])
            return runway

    # 4. Fall back to not identifying which parallel runway:
    idents = map(lambda runway: runway['identifier'], runways)

    # Check that the runway identifiers don't conflict, otherwise guess:
    ident_to_int = lambda x: int(x.rstrip('CLRST'), 10)
    groups = list(set(map(ident_to_int, idents)))
    if len(groups) > 1:
        # Ensure that we can find out if there was a runway conflict:
        args = [airport['id'], heading, ilsfreq, latitude, longitude, hint]
        message = 'Runways with conflicting identifer headings found: %s [%s].'
        details = (', '.join(map(str, idents)), ', '.join(map(str, args)))
        logger.info(message, *details)
        # Determine nearest identifiers to the heading provided:
        nearest = min(groups, key=lambda x: abs(x - heading))
        # Filter out runways that are not the nearest to the heading:
        runways = [r for r in runways if ident_to_int(r['identifier']) == nearest]
        # Prevent addition of * if only a single runway:
        if len(runways) == 1:
            return runways[0]

    # Choose an order of runway identifiers based on the provided hint:
    order = {'takeoff': 'CLRST', 'landing': 'RCLST', 'approach': 'CRLST'}[hint or 'landing']
    # Order the runway objects by the selected ordering:
    runway = sorted(runways, key=lambda x: order.find(x['identifier'][-1]))[0]

    runway = deepcopy(runway)
    runway['identifier'] = runway['identifier'].rstrip('CLRST') + '*'  # FIXME: Copy to avoid modifying!

    if not runway.get('end'):
        raise ValueError('Runway %s at airport #%d has no end coordinates.'
                         % (runway['identifier'], airport['id']))

    return runway


def mb2ft(mb):
    '''Convert millibars into feet'''
    return (1-np.power((mb/1013.25),0.190284))*145366.45


def valid_between(param_1,lower,upper):
    return 1-np.ma.getmaskarray(np.ma.masked_outside(param_1,lower,upper))


def find_rig_approach(condition_defs, phase_map, approach_map,
                      Vy,
                      this_hdf_file,
                      param_arrays,
                      debug=False,
                      lat_oil_rig=None,
                      lon_oil_rig=None):

    '''
    Core part of rig approach detection algorithm.

    param_arrays is a dictionary of parameter name : array, where the keys are:
    ['Airspeed'] - airspeed in knots
    ['Altitude ADH'] - altitude with respect to the deck height in feet
    ['Distance To Landing'] - track distnace to next landing in NM
    ['Heading'] - heading (continuous) in degrees
    ['Latitude'] - latitude in degrees
    ['Longitude'] - longitude in degrees
    ['Roll'] - roll attitude in degrees
    ['Vertical Speed'] - vertical speed in ft/min
    '''
    def build_map_array(condition_names, phase_map):
        n_conditions = len(condition_names)
        n_phases = len(phase_map)
        phase_mappings = np.ma.zeros((n_phases, n_conditions))

        for ix_phase, phase in enumerate(phase_map):
            for ix_condition, condition  in enumerate(condition_names):
                # This code just checks that the required conditions are available.
                # an easy (trust me!) mistake to make.
                for check in phase_map[phase]:
                    if check not in condition_names:
                        raise ValueError('%s requires missing: %s' %(phase, check))
                # Now we know it's going to work, populate this line of the array.
                if condition in phase_map[phase]:
                    phase_mappings[ix_phase, ix_condition] = 1
        return phase_mappings

    def document_condition(message, index, phase_mappings, d, conditions, causes, condition_names, name):
        tell_me = {'begin': ' starts when this is first met: ',
                   'end'  : ' ends when this is no longer met: ',
                   'never': ' is never true. Closest still has not met ',
                   }
        missed_conditions = np.ma.logical_and(1-conditions.transpose()[index], phase_mappings[d])
        causes[index] = []
        for n, valid in enumerate(missed_conditions):
            if valid:
                causes[index].append(name+tell_me[message]+condition_names[n])

    def head_diff(h, h0):
        # Allows us to use Heading Continuous to avoid 360 deg jumps, while still viewing the
        # data as values around zero for the approach heading.
        turns = int((h[-1]-h0+180)/360)
        return (h-h0) - (360.0*turns)

    # Local names
    u = param_arrays['Airspeed']
    lat = param_arrays['Latitude']
    lon = param_arrays['Longitude']
    heading = param_arrays['Heading']
    height = param_arrays['Altitude ADH']
    distance = param_arrays['Distance To Landing']

    duration = len(u)
    two_miles = index_at_value(distance, 2.0, _slice=slice(None, None, -1))

    # Use the latitude and longitude of the oil rig if provided.
    # Otherwise use the landing latitude/longitude
    if lon_oil_rig is not None and lat_oil_rig is None:
        lat_oil_rig = lat[-1]
    elif lat_oil_rig is not None and lon_oil_rig is None:
        lon_oil_rig = lon[-1]
    elif lat_oil_rig is None and lon_oil_rig is None:
        lat_oil_rig = lat[-1]
        lon_oil_rig = lon[-1]

    head_two_miles, _ = bearing_and_distance(
        closest_unmasked_value(lat, int(two_miles)).value,
        closest_unmasked_value(lon, int(two_miles)).value,
        lat_oil_rig,
        lon_oil_rig
    )
    param_arrays['head_off_two_miles'] = head_diff(heading, head_two_miles)
    param_arrays['head_final'] = np.ma.median(heading[-30:])
    param_arrays['head_off_final'] = head_diff(heading, param_arrays['head_final'])

    # Debug plots
    if debug:
        y_loc = -450
        import matplotlib.pyplot as plt
        plt.switch_backend('TkAgg')
        plt.title(this_hdf_file)

        plt.plot(lon, lat)
        plt.plot(lon[two_miles],lat[two_miles], marker='o', color='r')
        plt.plot(lon[-1],lat[-1], marker='o', color='g')
        plt.show()
        plt.clf()

        plt.plot(u, label='u')
        plt.plot(heading, label='heading')
        plt.plot(param_arrays['Groundspeed'], label='groundspeed')
        plt.plot(height, label='height')
        plt.plot(distance, label='d')
        plt.legend()
        plt.show()
        plt.clf()

        plt.plot(height)
        plt.plot(u)
        plt.plot(param_arrays['head_off_two_miles'])


    #========================================================================
    # Define all the conditional elements individually
    #========================================================================
    # It is convenient to have a list of names available
    condition_names = [c for c in condition_defs]

    # We build the list of true and false results for each condition across the entire data set...
    arrays = [f(param_arrays) for f in condition_defs.values()]

    conditions=np.ma.concatenate(arrays)
    # ...then reshape the conditions array to be the number of named conditions by the length of the arrays supplied.
    num_condition_names = len(condition_names)
    conditions = conditions.reshape(num_condition_names,
                                    len(conditions)//num_condition_names)

    #========================================================================
    # Define the phases and which conditions have to be met for each phase
    #========================================================================
    phase_mappings = build_map_array(condition_names, phase_map)

    #========================================================================
    # Define the types of approach by the phases they contain
    #========================================================================
    approach_type_mappings = build_map_array(phase_map, approach_map).transpose()

    #========================================================================
    # Work out which approach type we are in...
    #========================================================================

    #...first work out which phases are satisfied

    # How many conditions are satisfied for each phase at each sample point?
    number_of_conditions_met = np.ma.dot(phase_mappings, conditions).transpose()
    # How many conditions were specified for each phase?
    phase_mapping_totals = np.sum(phase_mappings, axis=1)
    # So when were all the condiitons for each phase fully satisfied?
    complete_phases = np.ma.masked_where(number_of_conditions_met < phase_mapping_totals,
                                         number_of_conditions_met)

    # Track the cause for incomplete phase conditions in the sample before the start
    # and after the end of each satisfied phase - we will need this later for documentation and debugging
    causes = {}
    for d, name in enumerate(phase_map):
        phases_meet_all_conditions = np.ma.clump_unmasked(complete_phases.transpose()[d])
        if phases_meet_all_conditions:
            for phase_meets_all_conditions in phases_meet_all_conditions:
                if debug:
                    print(name, phases_meet_all_conditions)
                begin = phase_meets_all_conditions.start
                if begin > 1:
                    document_condition('begin', begin-1, phase_mappings, d, conditions, causes, condition_names, name)

                end = phase_meets_all_conditions.stop
                if end < duration:
                    document_condition('end', end, phase_mappings, d, conditions, causes, condition_names, name)

                if debug:
                    plt.plot([begin,end],[y_loc,y_loc],label=name)
                    y_loc+=100

        else:
            # We didn't enter that phase satisfactorily. Report why (for the first closest match only) to aid debugging.
            nearly_idx = np.ma.argmax(complete_phases.transpose()[d].data)
            document_condition('never', nearly_idx, phase_mappings, d, conditions, causes, condition_names, name)

    if debug:
        import pprint
        pprint.pprint(causes)


    #...second work out which approach types are satisfied <<< VERY SIMILAR TO ABOVE >>>
    number_of_phases_met = np.ma.dot(complete_phases.mask, approach_type_mappings)
    approach_type_totals = np.sum(approach_type_mappings, axis=0)
    complete_approaches = np.ma.masked_where(number_of_phases_met < approach_type_totals,
                                             number_of_phases_met)

    longest_approach_type = None
    longest_approach_durn = 0
    half_mile = index_at_value(distance, 0.5, _slice=slice(None, None, -1))
    if debug:
        print(half_mile)

    # If ARDA/AROA is detected, this takes priority and we return this instead of checking if
    # this is indeed the longest slice. Else, we check if the conditions for the Standard
    # Approach are met and return this instead. If the conditions are not met for both the
    # ARDA/AROA and Standard Approach, we return None
    is_arda_aroa = False
    for d, name in enumerate(approach_map):
        slices_this_approach_type = np.ma.clump_masked(complete_approaches.transpose()[d])
        if slices_this_approach_type:
            max_durn = 0
            max_durn_slice = None
            for this_slice in slices_this_approach_type:
                # I'm only going to be interested in approaches that encompass the half mile point
                # this avoids lengthy periods in the circuit from "beating" last minute dashes
                # for the rig.
                if not is_index_within_slice(half_mile, this_slice):
                    continue
                if name == 'AIRBORNE_RADAR':
                    longest_approach_type = name
                    longest_approach_durn = slices_duration([this_slice], 1.0)
                    longest_approach_slice = this_slice
                    is_arda_aroa = True
                    break
                elif is_arda_aroa == False:
                    longest_approach_durn = slices_duration([this_slice], 1.0)
                    longest_approach_type = name
                    longest_approach_slice = this_slice
            if debug:
                if max_durn_slice:
                    print(name,'longest period', max_durn, 'sec, from', max_durn_slice.start,'to', max_durn_slice.stop)
                else:
                    print(name, 'not met at half mile point')

        if is_arda_aroa:
            break

    if debug:
        print('\n'+'Longest slice overall was type', longest_approach_type, longest_approach_durn, 'sec')

        try:
            print('Start from', longest_approach_slice.start, 'as phase',causes[longest_approach_slice.start-1][0])
        except:
            print('started')

        try:
            print('End at', longest_approach_slice.stop, 'as phase',causes[longest_approach_slice.stop][0])
        except:
            print('finished')
        print('---------------------------------')
        legend = plt.legend(loc='upper right')
        plt.show()

    if longest_approach_type:
        return longest_approach_type, longest_approach_durn, longest_approach_slice
    else:
        return None, None, None


def max_maintained_value(array, seconds, frequency, _slice=slice(None)):
    '''
    Return the indice and the value of the maximum value maintained for at least
    the given duration within the slice. The index will be relative to the start
    of the array, not to the start of the slice.

    seconds * frequency gives the number of samples which must be greater or equal to 2.

    :param array: ...
    :type array: np.ma.masked_array
    :param seconds: window size in seconds
    :type seconds: float or int
    :param frequncy: frequency of the array data
    :type frequency: float or int
    :param _slice: slice for the data of interest.
    :type _slice: slice
    '''
    array = array[slices_int(_slice)]
    if not array.size:
        return None, None
    sample_duration = 1 / frequency
    if modulo(seconds, sample_duration) != 0:
        seconds = seconds - seconds % sample_duration + sample_duration

    samples = int(frequency * seconds)
    if samples < 2:
        raise ValueError('Too small duration %s for frequency %s Hz.\n'
                         'Value of seconds * frequency must be at least 2.'
                         % (seconds, frequency))

    def max_maintained_value_in_valid_array(array, samples):
        for valid_slice in np.ma.clump_unmasked(array):
            # Make a view of a sliding window using a different stride.
            # For 3 samples, the array [1, 2, 3, 4, 5, 6, 7, 8, 9] will become
            # [[1, 2, 3],
            #  [2, 3, 4],
            #  [3, 4, 5],
            #  [4, 5, 6],
            #  [5, 6, 7],
            #  [6, 7, 8],
            #  [7, 8, 9]]
            valid_array = array[valid_slice]
            if valid_array.size < samples:
                continue
            sliding_window = np.lib.stride_tricks.as_strided(
                valid_array, shape=(len(valid_array) - samples + 1, samples),
                strides=valid_array.strides * 2)

            # Calculate min over the last axis (for each sliding window)
            min_ = np.min(sliding_window, axis=-1)
            idx, value =  min_.argmax(), min_.max()
            yield idx + valid_slice.start, value

    idx, value = max(
        max_maintained_value_in_valid_array(array, samples),
        key=itemgetter(1),
        default=(None, None)
    )
    if idx is not None:
        idx += _slice.start or 0
    return idx, value


def find_climb_cruise_descent(alt_std):
    '''
    Find slices defining a climb-cruise-descent for the given pressure altitude array

    :param alt_std: pressure altitude array
    :type alt_std: np.ma.masked_array
    :returns: List of slices defining each a climb-cruise-descent
    :rtype: List of slices
    '''
    slices = list()

    # We squash the altitude signal above 10,000ft so that changes of
    # altitude to create a new flight phase have to be 10 times
    # greater; 500ft changes below 10,000ft are significant, while
    # above this 5,000ft is more meaningful.
    alt_squash = np.ma.where(
        alt_std > 10000, (alt_std - 10000) / 10.0 + 10000, alt_std)
    pk_idxs, pk_vals = cycle_finder(alt_squash,
                                    min_step=HYSTERESIS_FPALT_CCD)

    if pk_vals is not None:
        n = 0
        n_vals = len(pk_vals)
        while n < n_vals - 1:
            pk_val = pk_vals[n]
            pk_idx = pk_idxs[n]
            next_pk_val = pk_vals[n + 1]
            next_pk_idx = pk_idxs[n + 1]
            if pk_val > next_pk_val:
                # descending
                slices.append(slice(None, next_pk_idx))
                n += 1
            else:
                # ascending
                # We are going upwards from n->n+1, does it go down
                # again?
                if n + 2 < n_vals:
                    if pk_vals[n + 2] < next_pk_val:
                        # Hurrah! make that phase
                        slices.append(slice(pk_idx,
                                                pk_idxs[n + 2]))
                        n += 2
                else:
                    slices.append(slice(pk_idx, None))
                    n += 1

        return slices


def maintain_altitude(distance, hz=1.0):
    '''
    Check if the altitude was maintained given the `distance array` between
    Current Altitude and Target Altitude.

    :param distance: The array of distances between Target Altitude and
        Current Altitude
    :type distance: numpy.array
    :param hz: Array sampling frequency
    :type hz: float

    :returns: If Current Altitude was within 50 ft of Target Altitude for 20
        consecutive secs.
    :rtype: bool
    '''
    within_50ft = np.abs(distance) < 50
    clumps = ezclump(within_50ft)
    return any((clump.stop - clump.start) > 20 * hz for clump in clumps)

