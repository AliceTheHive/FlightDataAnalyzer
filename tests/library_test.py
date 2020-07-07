# -*- coding: utf-8 -*-
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
##############################################################################

'''
'''

##############################################################################
# Imports
import csv
import mock
import numpy as np
import os
import pytz
import types
import unittest
import yaml

from  collections import Counter
from datetime import datetime
from math import sqrt
from mock import patch
from numpy.ma.testutils import assert_array_almost_equal, assert_array_equal, assert_array_less, assert_equal
from time import process_time

from analysis_engine.flight_attribute import LandingRunway

# A set of masked array test utilities from Pierre GF Gerard-Marchant
# http://www.java2s.com/Open-Source/Python/Math/Numerical-Python/numpy/numpy/ma/testutils.py.htm
from flightdatautilities import units as ut
from flightdatautilities.array_operations import load_compressed
import flightdatautilities.masked_array_testutils as ma_test

from analysis_engine.library import (
    air_density,
    all_of,
    alt_dev2alt,
    alt2sat,
    alt2press,
    alt2press_ratio,
    ambiguous_runway,
    any_of,
    any_one_of,
    air_track,
    align,
    align_slice,
    align_slices,
    average_value,
    bearing_and_distance,
    bearings_and_distances,
    blend_equispaced_sensors,
    blend_nonequispaced_sensors,
    blend_two_parameters,
    blend_parameters,
    blend_parameters_weighting,
    bump,
    calculate_flap,
    calculate_slat,
    calculate_timebase,
    cas2dp,
    cas_alt2mach,
    closest_unmasked_value,
    clump_multistate,
    compress_iter_repr,
    convert_two_digit_to_four_digit_year,
    coreg,
    create_phase_inside,
    create_phase_outside,
    cycle_counter,
    cycle_finder,
    cycle_match,
    cycle_select,
    datetime_of_index,
    delay,
    dp_over_p2mach,
    dp2cas,
    dp2tas,
    Decimal,
    fill_masked_edges,
    filter_slices_duration,
    filter_slices_length,
    filter_vor_ils_frequencies,
    find_climb_cruise_descent,
    find_edges,
    find_edges_on_state_change,
    find_slices_overlap,
    find_toc_tod,
    find_low_alts,
    find_nearest_slice,
    first_order_lag,
    first_order_washout,
    first_valid_parameter,
    first_valid_sample,
    from_isa,
    groundspeed_from_position,
    ground_track,
    ground_track_precise,
    gtp_blend_curve,
    hash_array,
    heading_diff,
    hysteresis,
    ils_established,
    ils_glideslope_align,
    ils_localizer_align,
    including_transition,
    index_at_distance,
    index_at_value,
    index_closest_value,
    index_of_datetime,
    index_of_first_start,
    index_of_last_stop,
    integrate,
    integ_value,
    interleave,
    interpolate,
    interpolate_coarse,
    is_index_within_slice,
    is_index_within_slices,
    is_power2,
    is_slice_within_slice,
    InvalidDatetime,
    last_valid_sample,
    latitudes_and_longitudes,
    level_off_index,
    localizer_scale,
    lookup_table,
    MappedArray,
    maintain_altitude,
    mask_inside_slices,
    mask_outside_slices,
    max_abs_value,
    max_continuous_unmasked,
    max_value,
    max_maintained_value,
    machsat2tat,
    machtat2sat,
    match_altitudes,
    mb2ft,
    min_value,
    minimum_unmasked,
    median_value,
    merge_masks,
    merge_sources,
    merge_two_parameters,
    modulo,
    most_common_value,
    most_points_cost,
    moving_average,
    nearest_runway,
    nearest_neighbour_mask_repair,
    next_unmasked_value,
    normalise,
    np_ma_concatenate,
    np_ma_ones_like,
    np_ma_masked_zeros,
    np_ma_masked_zeros_like,
    np_ma_zeros_like,
    offset_select,
    overflow_correction,
    overflow_correction_array,
    peak_curvature,
    peak_index,
    positive_index,
    power_ceil,
    power_floor,
    press2alt,
    prev_unmasked_value,
    repair_mask,
    resample,
    rms_noise,
    rate_of_change,
    rate_of_change_array,
    round_to_nearest,
    runs_of_ones,
    runway_deviation,
    runway_distance_from_end,
    runway_distances,
    runway_heading,
    runway_length,
    runway_snap,
    runway_snap_dict,
    runway_touchdown,
    second_window,
    shift_slice,
    shift_slices,
    slice_duration,
    slice_round,
    slice_multiply,
    slice_samples,
    slices_and,
    slices_and_not,
    slices_above,
    slices_after,
    slices_before,
    slices_below,
    slices_between,
    slices_contract,
    slices_contract_duration,
    slices_duration,
    slices_extend,
    slices_extend_duration,
    slices_from_to,
    slices_from_ktis,
    slices_int,
    slices_multiply,
    slices_not,
    slices_or,
    slices_of_runs,
    slices_overlap,
    slices_overlap_merge,
    slices_remove_overlaps,
    slices_remove_small_gaps,
    slices_remove_small_slices,
    slices_round,
    smooth_signal,
    smooth_track,
    straighten,
    straighten_altitudes,
    straighten_headings,
    straighten_longitude,
    subslice,
    step_local_cusp,
    step_values,
    timedelta,
    track_linking,
    trim_slices,
    unique_values,
    Value,
    value_at_datetime,
    value_at_index,
    value_at_time,
    vstack_params,
    vstack_params_where_state,
    wrap_array,
)

from analysis_engine.node import (A, P, S, load, M, KTI, KeyTimeInstance, Section)

from analysis_engine.test_utils import buildsections

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')


class TestAllOf(unittest.TestCase):
    def test_all_of(self):
        available = ['Altitude AAL', 'Airspeed', 'Groundspeed']
        self.assertTrue(all_of(['Airspeed', 'Groundspeed'], available))
        self.assertFalse(all_of(['NOT PRESENT', 'Groundspeed'], available))
        self.assertTrue(all_of(['Groundspeed'], available))
        self.assertFalse(all_of(['NOT PRESENT', 'ALSO NOT THERE'], available))


class TestAnyOf(unittest.TestCase):
    def test_any_of(self):
        available = ['Altitude AAL', 'Airspeed', 'Groundspeed']
        self.assertTrue(any_of(['Airspeed', 'Groundspeed'], available))
        self.assertTrue(any_of(['NOT PRESENT', 'Groundspeed'], available))
        self.assertTrue(any_of(['Groundspeed'], available))
        self.assertFalse(any_of(['NOT PRESENT', 'ALSO NOT THERE'], available))

class TestAnyOneOf(unittest.TestCase):
    def test_any_one_of(self):
        self.assertTrue(any_one_of(['Airspeed', 'Groundspeed'], ['Airspeed']))
        self.assertTrue(any_one_of(['NOT PRESENT', 'Groundspeed'], ['Groundspeed']))
        self.assertFalse(any_one_of(['Groundspeed'], []))
        self.assertFalse(any_one_of(['Groundspeed'], ['Airspeed', 'Groudspeed']))
        self.assertFalse(any_one_of(['NOT PRESENT', 'ALSO NOT THERE'], ['Altitude STD']))

class TestAirTrack(unittest.TestCase):

    def test_air_track_basic(self):
        spd = np.ma.array([260,260,260,260,260,260,260], dtype=float)
        hdg = np.ma.array([0,0,0,90,90,90,270], dtype=float)
        alt = np_ma_ones_like(hdg)
        alt[0] = 0.0
        alt[-1] = 0.0
        lat, lon = air_track(0.0, 0.0, 0.0035, 0.0035, spd, hdg, alt, 1.0)
        np.testing.assert_array_almost_equal(0.0035, lat[-1], decimal=4)
        np.testing.assert_array_almost_equal(0.0035, lon[-1], decimal=4)

    def test_air_track_arrays_too_short(self):
        spd = np.ma.array([60,60])
        hdg = np.ma.array([0,0])
        alt = np.ma.array([0,0])
        # (lat_start, lon_start, lat_end, lon_end, spd, hdg, alt_aal, frequency
        lat, lon = air_track(0.0, 0.0, 1.0, 1.0, spd, hdg, alt, 1.0)
        self.assertEqual(lat, None)
        self.assertEqual(lon, None)

    def test_air_track_masked_end(self):
        spd = load_compressed(
            os.path.join(test_data_path, 'air_track_spd.npz'))
        hdg = load_compressed(
            os.path.join(test_data_path, 'air_track_hdg.npz'))
        alt = np_ma_ones_like(hdg)
        alt[0] = 0.0
        alt[-1] = 0.0
        lat, lon = air_track(25.751953125, -80.332374, 9.052734375, -79.453464,
                             spd, hdg, alt, 1.0)
        self.assertEqual(lat[0], 25.751953125)
        self.assertEqual(lon[0], -80.332374000000002)
        self.assertAlmostEqual(lat[5000], 17.177, places=3)
        self.assertAlmostEqual(lon[5000], -79.5, places=3)
        self.assertAlmostEqual(lat[-1], 9.05, places=2)
        self.assertAlmostEqual(lon[-1], -79.45, places=2)


class TestIsPower2(unittest.TestCase):
    def test_is_power2(self):
        self.assertEqual([i for i in range(2000) if is_power2(i)],
                         [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        self.assertFalse(is_power2(-2))
        self.assertFalse(is_power2(2.2))


class TestAlignSlices(unittest.TestCase):
    def test_align_slices_unchanged(self):
        slave = P('slave')
        master = P('master')
        slices = [slice(None, 20), slice(20, 40), slice(40, None),
                  slice(5, 10, 3), None, slice(None, None)]
        result = align_slices(slave, master, slices)
        self.assertEqual(result, slices)

    def test_align_slices(self):
        slave = P('slave', frequency=2, offset=0.75)
        master = P('master', frequency=1, offset=0.25)
        slices = [slice(None, 20), slice(20, 40), slice(40, None),
                  slice(5, 10, 3), None, slice(None, None)]
        result = align_slices(slave, master, slices)
        self.assertEqual(result, [slice(None, 39, None), slice(39, 79, None),
                                  slice(79, None, None), slice(9, 19, 3), None,
                                  slice(None, None, None)])


class TestFindSlicesOverlap(unittest.TestCase):
    def test_find_slices_overlap(self):
        slice1 = slice(None, 100)
        slice2 = slice(90, None)
        expected = slice(90, 100)
        res = find_slices_overlap(slice1, slice2)
        self.assertEqual(res, expected)

        slice1 = slice(50, 100)
        slice2 = slice(90, 150)
        expected = slice(90, 100)
        res = find_slices_overlap(slice1, slice2)
        self.assertEqual(res, expected)

        slice1 = slice(50, 90)
        slice2 = slice(100, 150)
        expected = None
        res = find_slices_overlap(slice1, slice2)
        self.assertEqual(res, expected)

        slice1 = slice(50, 100, None)
        slice2 = slice(90, 150, 1)
        expected = slice(90, 100)
        res = find_slices_overlap(slice1, slice2)
        self.assertEqual(res, expected)

        slice1 = slice(50, 100, -1)
        slice2 = slice(90, 150, 1)
        self.assertRaises(ValueError, find_slices_overlap, slice1, slice2)

        slice1 = slice(50, 100)
        slice2 = slice(90, 150, 2)
        self.assertRaises(ValueError, find_slices_overlap, slice1, slice2)


class TestFindLowAlts(unittest.TestCase):
    def test_find_low_alts_1(self):
        # Example flight with 3 approaches.
        array = load(os.path.join(test_data_path, 'alt_aal_goaround.nod')).array
        level_flights = [slice(1629.0, 2299.0),
                         slice(3722.0, 4708.0),
                         slice(4726.0, 4807.0),
                         slice(5009.0, 5071.0),
                         slice(5168.0, 6883.0),
                         slice(8433.0, 9058.0)]

        low_alts = find_low_alts(array, 1.0, 500, 3000, 2000,
                                 level_flights=level_flights)
        self.assertEqual(len(low_alts), 5)
        self.assertAlmostEqual(low_alts[0].start, 0, places=0)
        self.assertAlmostEqual(low_alts[0].stop, 499, places=0)
        self.assertAlmostEqual(low_alts[1].start, 3425, places=0)
        self.assertAlmostEqual(low_alts[1].stop, 3722, places=0)
        self.assertAlmostEqual(low_alts[2].start, 4807, places=0)
        self.assertAlmostEqual(low_alts[2].stop, 5009, places=0)
        self.assertAlmostEqual(low_alts[3].start, 6883, places=0)
        self.assertAlmostEqual(low_alts[3].stop, 7258, places=0)
        self.assertAlmostEqual(low_alts[4].start, 10362, places=0)
        self.assertAlmostEqual(low_alts[4].stop, 10815, places=0)

        low_alts = find_low_alts(array, 1.0, 500, 500, 2000,
                                        level_flights=level_flights,
                                        relative_start=True,
                                        relative_stop=True)
        self.assertEqual(len(low_alts), 5)
        self.assertAlmostEqual(low_alts[0].start, 0, places=0)
        self.assertAlmostEqual(low_alts[0].stop, 499, places=0)
        self.assertAlmostEqual(low_alts[1].start, 3586, places=0)
        self.assertAlmostEqual(low_alts[1].stop, 3722, places=0)
        self.assertAlmostEqual(low_alts[2].start, 4895, places=0)
        self.assertAlmostEqual(low_alts[2].stop, 5009, places=0)
        self.assertAlmostEqual(low_alts[3].start, 7124, places=0)
        self.assertAlmostEqual(low_alts[3].stop, 7265, places=0)
        self.assertAlmostEqual(low_alts[4].start, 10522, places=0)
        self.assertAlmostEqual(low_alts[4].stop, 10815, places=0)

        low_alts = find_low_alts(array, 1.0, 500, 3000,
                                 level_flights=level_flights)
        self.assertEqual(len(low_alts), 5)
        self.assertAlmostEqual(low_alts[0].start, 0, places=0)
        self.assertAlmostEqual(low_alts[0].stop, 455, places=0)
        self.assertAlmostEqual(low_alts[1].start, 3425, places=0)
        self.assertAlmostEqual(low_alts[1].stop, 3643, places=0)
        self.assertAlmostEqual(low_alts[2].start, 4807, places=0)
        self.assertAlmostEqual(low_alts[2].stop, 4955, places=0)
        self.assertAlmostEqual(low_alts[3].start, 6883, places=0)
        self.assertAlmostEqual(low_alts[3].stop, 7182, places=0)
        self.assertAlmostEqual(low_alts[4].start, 10362, places=0)
        self.assertAlmostEqual(low_alts[4].stop, 10815, places=0)

        low_alts = find_low_alts(array, 1.0, 500, level_flights=level_flights)
        self.assertEqual(len(low_alts), 5)
        self.assertAlmostEqual(low_alts[0].start, 0, places=0)
        self.assertAlmostEqual(low_alts[0].stop, 455, places=0)
        self.assertAlmostEqual(low_alts[1].start, 3611, places=0)
        self.assertAlmostEqual(low_alts[1].stop, 3643, places=0)
        self.assertAlmostEqual(low_alts[2].start, 4919, places=0)
        self.assertAlmostEqual(low_alts[2].stop, 4955, places=0)
        self.assertAlmostEqual(low_alts[3].start, 7150, places=0)
        self.assertAlmostEqual(low_alts[3].stop, 7182, places=0)
        self.assertAlmostEqual(low_alts[4].start, 10522, places=0)
        self.assertAlmostEqual(low_alts[4].stop, 10815, places=0)


        # Slices will include level flight without passing in slices.
        low_alts = find_low_alts(array, 1.0, 500, 3000, 2000)
        self.assertEqual(len(low_alts), 5)
        self.assertAlmostEqual(low_alts[0].start, 0, places=0)
        self.assertAlmostEqual(low_alts[0].stop, 499, places=0)
        self.assertAlmostEqual(low_alts[1].start, 3425, places=0)
        self.assertAlmostEqual(low_alts[1].stop, 4384, places=0)
        self.assertAlmostEqual(low_alts[2].start, 4805, places=0)
        self.assertAlmostEqual(low_alts[2].stop, 5130, places=0)
        self.assertAlmostEqual(low_alts[3].start, 6648, places=0)
        self.assertAlmostEqual(low_alts[3].stop, 7258, places=0)
        self.assertAlmostEqual(low_alts[4].start, 10362, places=0)
        self.assertAlmostEqual(low_alts[4].stop, 10815, places=0)


        # Support negative climbout_alt (search backwards).
        # Slices will include level flight without passing in slices.
        low_alts = find_low_alts(array, 1.0, 500, 3000, 50, stop_mode='descent',
                                 level_flights=level_flights)
        self.assertEqual(len(low_alts), 5)
        self.assertAlmostEqual(low_alts[0].start, 0, places=0)
        self.assertAlmostEqual(low_alts[0].stop, 1629, places=0)
        self.assertAlmostEqual(low_alts[1].start, 3425, places=0)
        self.assertAlmostEqual(low_alts[1].stop, 3632, places=0)
        self.assertAlmostEqual(low_alts[2].start, 4807, places=0)
        self.assertAlmostEqual(low_alts[2].stop, 4943, places=0)
        self.assertAlmostEqual(low_alts[3].start, 6883, places=0)
        self.assertAlmostEqual(low_alts[3].stop, 7171, places=0)
        self.assertAlmostEqual(low_alts[4].start, 10362, places=0)
        self.assertAlmostEqual(low_alts[4].stop, 10556, places=0)

        # 0 climbout_alt (lowest point of descent).
        low_alts = find_low_alts(array, 1.0, 500, 3000, 0,
                                 level_flights=level_flights)
        self.assertEqual(len(low_alts), 4)
        self.assertAlmostEqual(low_alts[0].start, 3425, places=0)
        self.assertAlmostEqual(low_alts[0].stop, 3632, places=0)
        self.assertAlmostEqual(low_alts[1].start, 4807, places=0)
        self.assertAlmostEqual(low_alts[1].stop, 4941, places=0)
        self.assertAlmostEqual(low_alts[2].start, 6883, places=0)
        self.assertAlmostEqual(low_alts[2].stop, 7171, places=0)
        self.assertAlmostEqual(low_alts[3].start, 10362, places=0)
        self.assertAlmostEqual(low_alts[3].stop, 10569, places=0)


    @unittest.skip('Known failure case')
    def test_find_low_alts_2(self):
        # Example flight with noisy alt aal
        array = load_compressed(os.path.join(test_data_path, 'find_low_alts_alt_aal_1.npz'))

        level_flights = [
            slice(1856.0, 2392.0),
            slice(4062.0, 4382.0),
            slice(4432.0, 4584.0),
            slice(4606.0, 4856.0),
            slice(5210.0, 5562.0),
            slice(5576.0, 5700.0),
            slice(5840.0, 5994.0),
            slice(6152.0, 6598.0),
            slice(7268.0, 7768.0),
            slice(8908.0, 9124.0),
            slice(9752.0, 9898.0),
            slice(9944.0, 10210.0),
            slice(10814.0, 11098.0),
            slice(11150.0, 11332.0),
            slice(11352.0, 11676.0),
            slice(12122.0, 12346.0),
            slice(12814.0, 12998.0),
            slice(13028.0, 13194.0),
            slice(13432.0, 13560.0),
            slice(13716.0, 13888.0),
            slice(13904.0, 14080.0),
            slice(14122.0, 14348.0),
            slice(14408.0, 14570.0),
            slice(14596.0, 14786.0),
            slice(15092.0, 15356.0),
            slice(15364.0, 15544.0),
            slice(15936.0, 16066.0),
            slice(16078.0, 16250.0),
            slice(16258.0, 16512.0),
            slice(16632.0, 16782.0),
            slice(16854.0, 16982.0),
            slice(17924.0, 18112.0),
            slice(18376.0, 18514.0),
            slice(18654.0, 20582.0),
            slice(21184.0, 21932.0),
        ]

        low_alts = find_low_alts(array, 3000, stop_alt=0,
                                 level_flights=level_flights)
        # With a default 500ft change in level for a low altitude excursion,
        # I think there really are 7 such periods in this test, hence why I
        # have left this failing. DJ.
        self.assertEqual(len(low_alts), 4)
        self.assertAlmostEqual(low_alts[0].start, 3037, places=0)
        self.assertAlmostEqual(low_alts[0].stop, 4062, places=0)
        self.assertAlmostEqual(low_alts[1].start, 6051, places=0)
        self.assertAlmostEqual(low_alts[1].stop, 6152, places=0)
        self.assertAlmostEqual(low_alts[2].start, 8176, places=0)
        self.assertAlmostEqual(low_alts[2].stop, 8908, places=0)
        self.assertAlmostEqual(low_alts[3].start, 21932, places=0)
        self.assertAlmostEqual(low_alts[3].stop, 22316, places=0)


class TestFindNearestSlice(unittest.TestCase):
    def test_find_nearest_slice(self):
        slices = [slice(2, 10)]
        nearest_slice = find_nearest_slice(5, slices)
        self.assertEqual(nearest_slice, slices[0])
        slices = [slice(2, 10), slice(20,30)]
        nearest_slice = find_nearest_slice(18, slices)
        self.assertEqual(nearest_slice, slices[1])

    def test_find_nearest_slice_nothing_to_search(self):
        slices = []
        nearest_slice = find_nearest_slice(5, slices)
        self.assertEqual(nearest_slice, None)

class TestAlignSlice(unittest.TestCase):
    @mock.patch('analysis_engine.library.align_slices')
    def test_align_slice(self, align_slices):
        slave = mock.Mock()
        master = mock.Mock()
        _slice = mock.Mock()
        align_slices.return_value = [mock.Mock()]
        result = align_slice(slave, master, _slice)
        align_slices.assert_called_once_with(slave, master, [_slice])
        self.assertEqual(result, align_slices.return_value[0])


class TestAlign(unittest.TestCase):
    def test_align_returns_same_array_if_aligned(self):
        slave = P('slave', np.ma.arange(10))
        master = P('master', np.ma.arange(30))
        aligned = align(slave, master)
        self.assertEqual(id(slave.array), id(aligned))

    def test_align_section_param(self):
        alt_aal = P('Altitude AAL', np.ma.arange(0, 5), frequency=1, offset=1)
        fast = S('Fast', frequency=4, offset=0.5)
        aligned = align(alt_aal, fast)
        self.assertEqual(len(aligned), 20)
        np.testing.assert_array_equal(aligned,
                                      [0, 0, 0, 0.25, 0.5, 0.75, 1, 1.25,
                                       1.5, 1.75, 2, 2.25, 2.5, 2.75, 3,
                                       3.25, 3.5, 3.75, 4, 4])

    def test_align_basic(self):
        class DumParam():
            def __init__(self):
                self.name = 'Dummy'
                self.data_type = None
                self.offset = None
                self.frequency = None
                self.array = []

        first = DumParam()
        first.frequency = 4
        first.offset = 0.1
        first.array = np.ma.arange(8)

        second = DumParam()
        second.frequency = 4
        second.offset = 0.2
        second.array = np.ma.arange(8)

        result = align(second, first) #  sounds more natural so order reversed 20/11/11
        np.testing.assert_array_equal(result.data,
                                      [0.0, 0.6, 1.6, 2.6, 3.6,
                                       4.6, 5.6, 6.6000000000000005])
        # first value is masked as it cannot be calculated
        np.testing.assert_array_equal(result.mask,
                    [ True, False, False, False, False, False, False, False])

    def test_align_value_error_raised(self):
        class DumParam():
            def __init__(self):
                self.name = 'Dummy'
                self.data_type = None
                self.offset = None
                self.frequency = None
                self.array = []

        first = DumParam()
        first.frequency = 1
        first.offset = 0.1
        first.array = np.ma.arange(3)

        second = DumParam()
        second.frequency = 2
        second.offset = 0.2
        second.array = np.ma.arange(5)

        self.assertRaises(ValueError, align, second, first)

    def test_align_discrete(self):
        first = P(frequency=1, offset=0.0,
                  array=np.ma.array([0,0,1,1,0,1,0,1], dtype=float))
        second = M(frequency=1, offset=0.7,
                   array=np.ma.array([0,0,1,1,0,1,0,1], dtype=float))

        result = align(second, first)
        np.testing.assert_array_equal(result.data, [0,0,0,1,1,0,1,0])
        np.testing.assert_array_equal(result.mask,
                    [ True, False, False, False, False, False, False, False])

    def test_align_multi_state(self):
        first = P(frequency=1, offset=0.6,
                  array=np.ma.array([11,12,13,14,15], dtype=float))
        second = M(frequency=1, offset=0.0,
                   array=np.ma.array([0,1,2,3,4], dtype=int))

        result = align(second, first)
        # check dtype is int
        self.assertEqual(result.dtype, int)
        np.testing.assert_array_equal(result.data, [1, 2, 3, 4, 0])
        np.testing.assert_array_equal(result.mask, [0, 0, 0, 0, 1])

    def test_align_multistate_no_action(self):
        first = P(frequency=1, offset=0.0,
                      array=np.ma.array([1.0, 2.0, 3.0]))
        array = np.ma.array([1, 2, 3])
        values_mapping = {1: 'one', 2: 'two', 3: 'three'}
        second = M('Test Node', array, values_mapping=values_mapping,
                   frequency=1.0, offset=0.0)
        result = align(second, first)
        self.assertTrue(isinstance(result, MappedArray))
        self.assertEqual(result[0], 'one')

    def test_align_multi_state__float(self):
        first = P(frequency=1, offset=0.6,
                  array=np.ma.array([11,12,13,14,15], dtype=float))
        second = M(frequency=1, offset=0.0,
                   array=np.ma.array([0,1,2,3,4], dtype=float))

        result = align(second, first)
        # check dtype is int
        self.assertEqual(result.dtype, float)
        np.testing.assert_array_equal(result.data, [1, 2, 3, 4, 0])
        np.testing.assert_array_equal(result.mask, [0, 0, 0, 0, 1])

    def test_align_parameters_without_interpolation(self):
        # Both are parameters, but interpolation forced off
        first = P(frequency=2, offset=0.2,
                  array=np.ma.array([11,12,13,14,15], dtype=float))
        second = P(frequency=1, offset=0.0,
                   array=np.ma.array([1, 2, 3.5, 4, 5], dtype=float))

        result = align(second, first, interpolate=False)
        # Check dtype returned is a float
        self.assertEqual(result.dtype, float)

        # Slave at 2Hz 0.2 offset explained:
        # 0.2 offset: 1 taken from 0.0 second already recorded
        # 0.7 offset: 2 taken from 1.0 second (1.0 is closer than 0.0 second)
        # 1.2 offset: 2 taken from 1.0 second (1.0 is closer than 2.0 second)
        # 1.7 offset: 3.5 taken from 2.0 second (2.0 is closest to 1.7 second)
        # ...
        np.testing.assert_array_equal(result.data, [1, 2, 2, 3.5, 3.5, 4, 4, 5, 0, 0])
        np.testing.assert_array_equal(result.mask, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    def test_align_same_hz_delayed(self):
        # Both arrays at 1Hz, master behind slave in time
        class DumParam():
            def __init__(self):
                self.name = 'Dummy'
                self.data_type = None
                self.offset = None
                self.frequency = 1
                self.array = []
        master = P(array=np.ma.array([0,1,2,3], dtype=float),
                   frequency=1,
                   offset=0.5)
        slave = P(array=np.ma.array([10,11,12,13], dtype=float),
                  frequency=1,
                  offset=0.2)
        result = align(slave, master)
        # last sample should be masked
        np.testing.assert_array_almost_equal(result.data, [10.3,11.3,12.3,0])
        np.testing.assert_array_equal(result.mask, [0, 0, 0, 1])

    def test_align_same_hz_advanced(self):
        # Both arrays at 1Hz, master ahead of slave in time
        class DumParam():
            def __init__(self):
                self.name = 'Dummy'
                self.data_type = None
                self.offset = None
                self.frequency = 1
                self.array = []
        master = P(array=np.ma.array([0,1,2,3],dtype=float),
                   frequency=1,
                   offset=0.2)
        slave = P(array=np.ma.array([10,11,12,13],dtype=float),
                  frequency=1,
                  offset=0.5)
        result = align(slave, master)
        np.testing.assert_array_almost_equal(result.data, [0,10.7,11.7,12.7])
        np.testing.assert_array_equal(result.mask, [1, 0, 0, 0])

    def test_align_increasing_hz_delayed(self):
        # Master at higher frequency than slave
        master = P(array=np.ma.array([0,1,2,3,4,6,6,7],dtype=float),
                   frequency=4,
                   offset=0.15)
        slave = P(array=np.ma.array([10,11,12,13],dtype=float),
                  frequency=2,
                  offset=0.1)
        result = align(slave, master)
        np.testing.assert_array_almost_equal(result.data, [10.1,10.6,11.1,11.6,
                                                           12.1,12.6, 0, 0])
        np.testing.assert_array_equal(result.mask, [0,0,0,0,0,0,1,1])

    def test_align_increasing_hz_advanced(self):
        # Master at higher frequency than slave
        master = P(array=np.ma.array([0,1,2,3,4,6,6,7,
                                      0,1,2,3,4,6,6,7],dtype=float),
                   frequency=8,
                   offset=0.1)
        slave = P(array=np.ma.array([10,11,12,13],dtype=float),
                  frequency=2,
                  offset=0.15)
        result = align(slave, master)
        # First sample of slave hasn't been sampled at initial master (at 0.1
        # seconds) so is masked

        # Last three samples of aligned slave have no final value to
        # extrapolate to so are also masked.
        np.testing.assert_array_almost_equal(result.data, [ 0.0,10.15,10.4,10.65,
                                                           10.9,11.15,11.4,11.65,
                                                           11.9,12.15,12.4,12.65,
                                                           12.9, 0.0 , 0.0,0.0 ])

    def test_align_decreasing_hz_delayed(self):
        # Master at lower frequency than slave
        master = P(array=np.ma.array([10,11,12,13],dtype=float),
                  frequency=1,
                  offset=0.25)
        slave = P(array=np.ma.array([0,1,2,3,4,6,6,7],dtype=float),
                   frequency=2,
                   offset=0.0)
        result = align(slave, master)
        np.testing.assert_array_almost_equal(result.data, [0.5, 2.5, 5.0, 6.5])
        np.testing.assert_array_equal(result.mask, [0,0,0,0])

    def test_align_decreasing_hz_delayed_big_delay_in_master(self):
        # Master at lower frequency than slave
        master = P(array=np.ma.array([10,11,12,13],dtype=float),
                  frequency=1,
                  offset=-0.75)
        slave = P(array=np.ma.array([0,1,2,3,4,6,6,7],dtype=float),
                   frequency=2,
                   offset=0.0)
        result = align(slave, master)
        np.testing.assert_array_almost_equal(result.data, [0.0, 0.5, 2.5, 5.0])
        np.testing.assert_array_equal(result.mask, [1,0,0,0])

    def test_align_decreasing_hz_delayed_excessive_delay_in_master(self):
        # Master at lower frequency than slave
        master = P(array=np.ma.array([10,11,12,13],dtype=float),
                  frequency=1,
                  offset=-1.5)
        slave = P(array=np.ma.array([0,1,2,3,4,6,6,7],dtype=float),
                   frequency=2,
                   offset=0.0)
        self.assertRaises(ValueError, align, slave, master)

    def test_align_mask_propogation(self):
        """
        This is a "pretty bad case scenario" for masking. We essentially "lose"
        the valid last recorded sample of 13 as we cannot interpolate
        across it to the end of the data.

        The first value masked as slave offset means it is sampled after the
        Master's first.

        The 3rd sample (12) is masked so values from 11 through 13 are
        masked.

        The last sample (13) is offset with invalid data to the left and
        padded data to the right, so there are no valid samples either side
        of the recorded sample to interpolate an aligned value, so this is
        also masked.
        """
        # Master at higher frequency than slave
        master = P(array=np.ma.array([0,1,2,3,4,6,6,7,
                                   0,1,2,3,4,6,6,7],dtype=float),
                   frequency=8,
                   offset=0.1)
        slave = P(array=np.ma.array([10,11,12,13],
                               mask=[ 0, 0, 1, 0],
                               dtype=float),
                  frequency=2,
                  offset=0.15)

        result = align(slave, master)
        answer = np.ma.array(data = [10.0,10.15,10.4,10.65,
                                     10.9,0,0,0,
                                     0,0,0,0,
                                     0,13.0,13.0,13.0],
                             mask = [True,False,False,False,
                                     False, True, True, True,
                                     True , True, True, True,
                                     True , True, True, True])
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_align_mask_propogation_same_offsets(self):
        # Master at higher frequency than slave, but using repair_mask
        master = P(array=np.ma.array([0,1,2,3,4,6,6,7,
                                   0,1,2,3,4,6,6,7],dtype=float),
                   frequency=8,
                   offset=0.2)
        slave = P(array=np.ma.array([10,11,12,13],
                               mask=[ 0, 0, 1, 0],
                               dtype=float),
                  frequency=2,
                  offset=0.2)

        result = align(slave, master)
        answer = np.ma.array(data = [
            # good, interpolated to 11
            10.00, 10.25, 10.50, 10.75,
            # good, unreliable
            11.00,  0.00,  0.00,  0.00,
            # masked, unreliable
            12.00,  0.00,  0.00,  0.00,
            # good, no extrapolation AKA masked padding
            13.00,  0.00,  0.00,  0.00],
                             mask = [
            False,False,False,False,
            False, True, True, True,
            True , True, True, True,
            False, True, True, True])
        ma_test.assert_masked_array_approx_equal(result, answer)


    def test_align_atr_problem_replicated(self):
        # AeroTech Research data showed up a specific problem simuated by this test.

        def zero_ends_error(array, air_time):
            result = np_ma_masked_zeros_like(array)
            result[air_time] = repair_mask(array[air_time])
            return result

        class DumParam():
            def __init__(self):
                self.name = 'Dummy'
                self.data_type = None
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.array([1.0]*48) # 6 seconds
        master.frequency = 8
        master.offset = 0.00390625
        slave = DumParam()
        slave.array = np.ma.array([12,12,669684.84,668877.65,12,12],dtype=float)
        slave.array[2:4] = np.ma.masked
        slave.frequency = 1
        slave.offset = 0.66796875
        result = zero_ends_error(align(slave, master),slice(6,42))
        answer = np.ma.array(data = [12.0]*48,
                             mask = [True]*6+[False]*8+[True]*24+[False]*4+[True]*6)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_align_atr_problem_corrected(self):
        # AeroTech Research data showed up a specific problem simuated by this test.

        def zero_ends_correct(array, air_time):
            result = np_ma_masked_zeros_like(array)
            result[air_time] = repair_mask(array[air_time], frequency=8)
            return result

        class DumParam():
            def __init__(self):
                self.name = 'Dummy'
                self.data_type = None
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.array([1.0]*48) # 6 seconds
        master.frequency = 8
        master.offset = 0.00390625
        slave = DumParam()
        slave.array = np.ma.array([12,12,669684.84,668877.65,12,12],dtype=float)
        slave.array[2:4] = np.ma.masked
        slave.frequency = 1
        slave.offset = 0.66796875
        result = zero_ends_correct(align(slave, master),slice(6,42))
        answer = np.ma.array(data = [12.0]*48,
                             mask = [True]*6+[False]*36+[True]*6)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_align_increasing_hz_extreme(self):
        # Master at higher frequency than slave
        master = P(array=np.ma.array([0,1,2,3,4,6,6,7,
                                      0,1,2,3,4,6,6,7],dtype=float),
                   frequency=8,
                   offset = 0.1)
        slave = P(array=np.ma.array([10,11],dtype=float),
                  frequency=1,
                  offset=0.95)
        result = align(slave, master)
        expected = [ 0.0 ,  0.0  ,  0.0,  0.0  ,
                     0.0 ,  0.0  ,  0.0, 10.025,
                    10.15, 10.275, 10.4, 10.525,
                    10.65, 10.775, 10.9,  0.0  ]
        np.testing.assert_array_almost_equal(result.data, expected)
        np.testing.assert_array_equal(result.mask,
                    [True,  True,  True,  True,
                     True,  True,  True,  False,
                     False, False, False, False,
                     False, False, False, True  ])

    def test_align_across_frame_increasing(self):
        master = P(array=np.ma.zeros(64, dtype=float),
                   frequency=8,
                   offset=0.1)
        slave = P(array=np.ma.array([10,11], dtype=float),
                   frequency=0.25,
                   offset=3.95)
        result = align(slave, master)
        # Build the correct answer...
        answer = np.ma.ones(64)
        # data is masked up to first slave sample
        answer[:31] = 0
        answer[:31] = np.ma.masked
        # increment between 10 and 11
        answer[31:] *= 10
        answer[31] += 0.00625
        for i in range(31):
            answer [31+i+1] = answer[31+i] + 1/32.0
        # last value is after the slave offset therefore masked
        answer[-1] = 0
        answer[-1] = np.ma.masked

        # ...and check the resulting array in one hit.
        ma_test.assert_masked_array_approx_equal(result, answer)


    def test_align_across_frame_decreasing(self):
        # Master at higher frequency than slave
        class DumParam():
            def __init__(self):
                self.name = 'Dummy'
                self.data_type = None
                self.offset = None
                self.frequency = 1
                self.array = []
        master = DumParam()
        master.array = np.ma.zeros(4, dtype=float)
        # It is necessary to force the data type, as otherwise the array is cast
        # as integer and the result comes out rounded down as well.
        master.frequency = 0.5
        master.offset = 1.5
        slave = DumParam()
        # Fill a two-frame sample with linear data
        slave.array = np.ma.arange(32,dtype=float)
        slave.frequency = 4
        slave.offset = 0.1
        result = align(slave, master)
        # Build the correct answer...
        answer=np.ma.array([5.6,13.6,21.6,29.6])
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_align_superframe_master(self):
        class DumParam():
            def __init__(self):
                self.name = 'Dummy'
                self.data_type = None
                self.offset = None
                self.frequency = 1
                self.offset = 0.0
                self.array = []
        master = DumParam()
        master.array = np.ma.array([1,2])
        master.frequency = 1/64.0
        slave = DumParam()
        slave.array = np.ma.arange(128)
        slave.frequency = 1
        result = align(slave, master)
        expected = [0,64]
        np.testing.assert_array_equal(result.data,expected)

    def test_align_superframe_slave(self):
        class DumParam():
            def __init__(self):
                self.name = 'Dummy'
                self.data_type = None
                self.offset = None
                self.frequency = 1
                self.offset = 0.0
                self.array = []
        master = DumParam()
        master.array = np.ma.arange(64)
        master.frequency = 2
        slave = DumParam()
        slave.array = np.ma.array([1,3,6,9])
        slave.frequency = 1/8.0
        result = align(slave, master)
        expected = [
        1.    ,  1.125 ,  1.25  ,  1.375 ,  1.5   ,  1.625 ,  1.75  ,
        1.875 ,  2.    ,  2.125 ,  2.25  ,  2.375 ,  2.5   ,  2.625 ,
        2.75  ,  2.875 ,  3.    ,  3.1875,  3.375 ,  3.5625,  3.75  ,
        3.9375,  4.125 ,  4.3125,  4.5   ,  4.6875,  4.875 ,  5.0625,
        5.25  ,  5.4375,  5.625 ,  5.8125,  6.    ,  6.1875,  6.375 ,
        6.5625,  6.75  ,  6.9375,  7.125 ,  7.3125,  7.5   ,  7.6875,
        7.875 ,  8.0625,  8.25  ,  8.4375,  8.625 ,  8.8125,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ]
        np.testing.assert_array_equal(result.data,expected)

    def test_align_superframe_slave_extreme(self):
        master = P(array=np.ma.arange(1024), frequency=8)
        slave = P(array=np.ma.array([0, 512]), frequency=1/64.0)
        result = align(slave, master)
        expected = list(range(0, 512)) + [0]*512
        np.testing.assert_array_equal(result.data,expected)


    def test_align_superframes_both(self):
        master = P(array = np.ma.arange(16),
                   frequency = 1/8.0)
        slave = P(array=np.ma.array([100, 104, 108, 112]),
                  frequency = 1/32.0)
        result = align(slave, master)
        expected_data = list(range(100, 112)) + [0] * 4
        expected = np.ma.array(data=expected_data,
                               mask=[0]*12+[1]*4)
        assert_array_almost_equal(result, expected)

    def test_align_8_hz_half_hz(self):
        # Same offset, so every 16th sample (ratio between master and slave)
        # will be that from the slave array.
        master = P(array=np.ma.arange(22576),
                   frequency=8.0)
        slave = P(array=np.ma.arange(1411),
                  frequency=0.5)
        result = align(slave, master)
        expected_16th_samples = np.ma.arange(1411)
        # all slave values are in the result
        np.testing.assert_array_equal(result[::16], expected_16th_samples)
        # last 15 samples are masked
        np.testing.assert_array_equal(result.mask[-15:], [True]*15)
        # test a chunk is interpolated
        np.testing.assert_array_equal(result.data[:16], np.arange(16, dtype=float)/16)

    def test_align_superframe_to_onehz_multistate(self):
        # Slave once per superframe, master at 1Hz, Multi-State
        onehz = M(frequency=1, offset=0.0,
                  array=np.ma.array([0,0,1,1,0,1,0,1], dtype=float))
        slave = P(frequency=1.0/64, offset=0.0,
                   array=np.ma.array([1, 65, 129, 193], dtype=float))
        result = align(slave, onehz)
        expected = np.ma.array(list(range(1, 193)) + [0] * 64)
        np.testing.assert_array_equal(result.data, expected)

    def test_align_fully_masked_array(self):
        # fully masked arrays are passed in at higher frequency but fully masked
        # build masked array
        expected = np.ma.zeros(16)
        expected.mask = True

        # Test with same offset - returns all 0s
        master = P(frequency=8, offset=0.02)
        slave = P(frequency=2, offset=0.02,
                  array=np.ma.array([10, 11, 12, 13], mask=True))
        result = align(slave, master)
        #Original implementation returned zero array. Now completes first
        #step of alignment before returning. This is not significant as the
        #array is completely masked, so testing the data component is
        #superfluous. np.testing.assert_array_equal(result.data,
        #expected.data)
        np.testing.assert_array_equal(result.mask, expected.mask)

        # Example with different offset - returns all 0s
        master = P(frequency=8, offset=0.05)
        slave = P(frequency=2, offset=0.01,
                  array=np.ma.array([10, 11, 12, 13], mask=True))
        result = align(slave, master)
        # np.testing.assert_array_equal(result.data, expected.data)
        np.testing.assert_array_equal(result.mask, expected.mask)

    def test_align_downsample_same_offset(self):
        # Sample every other value.
        slave = P(array=np.ma.array([
            22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
            21.9990234375, 21.9912109375, 21.9833984375, 21.9755859375,
            21.9677734375, 21.9599609375, 21.9521484375, 21.9443359375,
            21.9365234375, 21.9287109375, 21.9208984375, 21.9130859375,
            21.9052734375, 21.8974609375, 21.8896484375, 21.8818359375,
            21.875, 21.875, 21.875, 21.875, 21.875, 21.875, 21.875, 21.875,
            21.875, 21.875, 21.875, 21.875, 21.875, 21.875]), frequency=4,
                  offset=0)
        master = P(frequency=2, offset=0)
        result = align(slave, master)
        ma_test.assert_masked_array_almost_equal (
            result,
            np.ma.masked_array(
                [22.0, 22.0, 22.0, 22.0, 22.0, 21.9990234375, 21.9833984375,
                 21.9677734375, 21.9521484375, 21.9365234375, 21.9208984375,
                 21.9052734375, 21.8896484375, 21.875, 21.875, 21.875, 21.875,
                 21.875, 21.875, 21.875]))

    def test_align_5hz(self):
        master = P('master', array=[1,2,3], frequency=1.0, offset=0.0)
        slave = P('slave', np.ma.arange(15), frequency=5.0, offset=0.0)
        result = align(slave, master)
        assert_array_equal(result, [0, 5, 10])

    def test_align_5hz_reverse(self):
        master = P('master', np.ma.arange(15.0), frequency=5.0, offset=0.0)
        slave = P('slave', array=[1,2,3], frequency=1.0, offset=0.0)
        result = align(slave, master)
        expected = (master.array/5.0)+1.0
        expected[11:]=np.ma.masked
        assert_array_almost_equal(result, expected)

    def test_align_10hz(self):
        master = P('master', array=[1,2], frequency=1.0, offset=0.0)
        slave = P('slave', np.ma.arange(20), frequency=10.0, offset=0.0)
        result = align(slave, master)
        assert_array_equal(result, [0, 10])

    def test_align_10hz_reverse(self):
        master = P('master', np.ma.arange(20.0), frequency=10.0, offset=0.0)
        slave = P('slave', array=[2,3], frequency=1.0, offset=0.0)
        result = align(slave, master)
        expected = (master.array/10.0)+2.0
        expected[21:]=np.ma.masked
        assert_array_almost_equal(result, expected)

    def test_align_15hz(self):
        master = P('master', array=[1,2,3], frequency=1.0, offset=0.0)
        slave = P('slave', np.ma.arange(45), frequency=15.0, offset=0.0)
        result = align(slave, master)
        assert_array_equal(result, [0, 15, 30])

    def test_align_15hz_reverse(self):
        master = P('master', np.ma.arange(30), frequency=15.0, offset=0.0)
        slave = P('slave', [2,3], frequency=1.0, offset=0.0)
        result = align(slave, master)
        expected = (master.array/15.0) + 2.0
        result[15:] = np.ma.masked
        assert_array_almost_equal(result, expected)

    def test_align_20hz(self):
        master = P('master', array=[1,2], frequency=1.0, offset=0.0)
        slave = P('slave', np.ma.arange(40), frequency=20.0, offset=0.0)
        result = align(slave, master)
        assert_array_equal(result, [0, 20])

    def test_align_20hz_reverse(self):
        master = P('master', np.ma.arange(100.0), frequency=20.0, offset=0.0)
        slave = P('slave', array=[6,5,4,3,2], frequency=1.0, offset=0.0)
        result = align(slave, master)
        expected = 6.0-(master.array/20.0)
        expected[81:]=np.ma.masked
        assert_array_almost_equal(result, expected)

    def test_align_5_10_20_offset_master(self):
        master = P('master', np.ma.arange(100.0), frequency=20.0, offset=0.1)
        slave = P('slave', array=[6,5,4,3,2], frequency=1.0, offset=0.0)
        self.assertRaises(ValueError, align, slave, master)

    def test_align_5_10_20_offset_slave(self):
        master = P('master', np.ma.arange(10.0), frequency=2.0, offset=0.0)
        slave = P('slave', array=[6,5,4,3,2], frequency=5.0, offset=0.3)
        self.assertRaises(ValueError, align, slave, master)

    def test_align_multi_state_5_10(self):
        first = P(frequency=10, offset=0.0,
                  array=np.ma.array([11,12,13,14,15,16,17,18,19,20], dtype=float))
        second = M(frequency=5, offset=0.0,
                   array=np.ma.array([1,3,4,5,6], dtype=int))

        result = align(second, first)
        # check dtype is int
        self.assertEqual(result.dtype, int)
        np.testing.assert_array_equal(result.data, [1,3,3,4,4,5,5,6,0,0])
        np.testing.assert_array_equal(result.mask, [0,0,0,0,0,0,0,0,1,1])

    def test_align_multi_state_10_15(self):
        first = P(frequency=15, offset=0.0,
                  array=np.ma.array([11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], dtype=float))
        second = M(frequency=10, offset=0.0,
                   array=np.ma.array([1,3,4,5,6,7,8,9,10,11], dtype=int))

        result = align(second, first)
        # check dtype is int
        self.assertEqual(result.dtype, int)
        np.testing.assert_array_equal(result.data, [1,3,3,4,5,5,6,7,7,8,9,9,10,11,0])
        np.testing.assert_array_equal(result.mask, [0] * 14 + [1])

    def test_align_multi_state_15_25(self):
        first = P(frequency=25, offset=0.0, array=np.ma.arange(25, dtype=int))
        second = M(frequency=15, offset=0.0,
                   array=np.ma.array([1,3,4,5,6,7,8,9,10,11,12,13,14,15,16], dtype=int))

        result = align(second, first)
        # check dtype is int
        self.assertEqual(result.dtype, int)
        np.testing.assert_array_equal(result.data, [1,3,3,4,4,5,6,6,7,7,8,9,9,10,10,11,12,12,13,13,14,15,15,16,0])
        np.testing.assert_array_equal(result.mask, [0] * 24  + [1])

    def test_align_multi_state_25_15(self):
        first = P(frequency=15, offset=0.0, array=np.ma.arange(15, dtype=np.int))
        second = M(frequency=25, offset=0.0, array=np.ma.arange(25, dtype=np.int))
        result = align(second, first)
        self.assertEqual(result.dtype, int)
        np.testing.assert_array_equal(result.data, [0,2,3,5,7,8,10,12,13,15,17,18,20,22,23])
        np.testing.assert_array_equal(result.mask, [0] * 15)

class TestAlignStringArrays(unittest.TestCase):
    def test_offset(self):
        first = P(frequency=1.0, offset=0.6,
                  array=np.ma.array([11,12,13,14,15], dtype=float))
        second = P(frequency=1.0, offset=0.0,
                   array=np.ma.array([b"0",b"1",b"2",b"3",b"4"]))

        expected = np.ma.array((b"1", b"2", b"3", b"4", b"0"), mask=[0,0,0,0,1])
        result = align(second, first)

        np.testing.assert_array_equal(result, expected)

    def test_align_multi_state_5_10(self):
        first = P(frequency=10, offset=0.0,
                  array=np.ma.array([11,12,13,14,15,16,17,18,19,20], dtype=float))
        second = P(frequency=5.0, offset=0.0,
                   array=np.ma.array([b"1",b"3",b"4",b"5",b"6"]))

        result = align(second, first)
        expected = np.ma.array([b"1",b"3",b"3",b"4",b"4",b"5",b"5",b"6",b"0",b"0"],
                               mask=[0,0,0,0,0,0,0,0,1,1])

        np.testing.assert_array_equal(result, expected)


class TestAmbiguousRunway(unittest.TestCase):
    def test_valid(self):
        landing_runway = LandingRunway()
        landing_runway.set_flight_attr({'identifier': '27L'})
        self.assertFalse(ambiguous_runway(landing_runway))

    def test_invalid_unresolved_runways(self):
        landing_runway = LandingRunway()
        landing_runway.set_flight_attr({'identifier': '27*'})
        self.assertTrue(ambiguous_runway(landing_runway))

    def test_invalid_no_runway_A(self):
        self.assertTrue(ambiguous_runway(None))

    def test_invalid_no_runway_B(self):
        landing_runway = LandingRunway()
        landing_runway.set_flight_attr(None)
        self.assertTrue(ambiguous_runway(landing_runway))

    def test_invalid_no_runway_C(self):
        landing_runway = LandingRunway()
        landing_runway.set_flight_attr({})
        self.assertTrue(ambiguous_runway(landing_runway))


class TestBearingsAndDistances(unittest.TestCase):
    def test_known_bearing_and_distance(self):
        fareham = {'latitude':50.856146,'longitude':-1.183182}
        goodyear_lon = np.ma.array([-112.359])
        goodyear_lat = np.ma.array([33.459])
        brg,dist = bearings_and_distances(goodyear_lat, goodyear_lon, fareham)
        self.assertAlmostEqual(dist[0],8482000, delta=2000)
        self.assertAlmostEqual(brg[0],306.78, delta=0.02)

    # With an atan(x/y) formula giving the bearings, it's easy to get this
    # wrong, as I did originally, hence the three tests for bearings ! The
    # important thing to remember is we are looking for the bearing from the
    # reference point to the array points, not the other way round.
    def test_quarter_bearings(self):
        origin = {'latitude':0.0,'longitude':0.0}
        latitudes = np.ma.array([.1,.1,-.1,-.1])
        longitudes = np.ma.array([-.1,.1,.1,-.1])
        # Bearings changed from +/-180 to 0:360 when this function was used
        # to populate runway magnetic headings in the airport database.
        compass = np.ma.array([360-45,45,135,360-135])
        brg, dist = bearings_and_distances(latitudes, longitudes, origin)
        ma_test.assert_masked_array_approx_equal(brg, compass)

    def test_ordinal_bearings(self):
        origin = {'latitude':0.0,'longitude':0.0}
        latitudes = np.ma.array([1,0,-1,0])
        longitudes = np.ma.array([0,1,0,-1])
        compass = np.ma.array([0,90,180,270])
        brg, dist = bearings_and_distances(latitudes, longitudes, origin)
        ma_test.assert_masked_array_approx_equal(brg, compass)

    def test_known_bearings(self):
        origin = {'latitude':60.280151,'longitude':5.222579}
        latitudes = np.ma.array([60.2789,60.30662494,60.289,60.28875])
        longitudes = np.ma.array([5.223,5.21370074,5.2272,5.2636])
        compass = np.ma.array([170,351,14,67])
        brg, dist = bearings_and_distances(latitudes, longitudes, origin)
        for i in range(4):
            self.assertLess(abs(compass[i]-brg[i]),1.0)

    def test_mask(self):
        origin = {'latitude':0.0,'longitude':0.0}
        latitudes = np.ma.array([.1,.1,-.1,-.1])
        latitudes[0]=np.ma.masked
        longitudes = np.ma.array([-.1,.1,.1,-.1])
        longitudes[2]=np.ma.masked
        # Bearings changed from +/-180 to 0:360 when this function was used
        # to populate runway magnetic headings in the airport database.
        compass = np.ma.array([135,45,360-45,360-135])
        compass.mask=[True,False,True,False]
        brg, dist = bearings_and_distances(latitudes, longitudes, origin)
        ma_test.assert_masked_array_approx_equal(brg, compass)
        self.assertEqual(dist[0].mask,True)
        self.assertEqual(dist[2].mask,True)

    def test_bearings_and_back_again(self):
        # One should be able to go back and forth between bearings and
        # distances and latitudes and longitudes without any loss of
        # prescision.
        latitude = np.ma.array([49.0128, 53.6304])
        longitude = np.ma.array([2.55, 9.98823])
        reference = {'latitude':49.0128, 'longitude':2.55}
        brgs, dists = bearings_and_distances(latitude, longitude, reference)
        b, d = bearing_and_distance(49.0128, 2.55, 53.61, 9.9)
        end_lats, end_lons = latitudes_and_longitudes(brgs, dists, reference)
        self.assertAlmostEqual(end_lats[0], 49.0128)
        self.assertAlmostEqual(end_lons[0], 2.55)
        self.assertAlmostEqual(end_lats[1], 53.6304)
        self.assertAlmostEqual(end_lons[1], 9.98823)


class TestLatitudesAndLongitudes(unittest.TestCase):
    def test_known_bearing_and_distance(self):
        # Amended Nov 2013 to greatly increase distance and hence improve quality of test.
        fareham = {'latitude': 50.856146,'longitude': -1.183182}
        goodyear_dist = np.ma.array([8482000.0])
        goodyear_brg = np.ma.array([306.78])
        lat,lon = latitudes_and_longitudes(goodyear_brg, goodyear_dist, fareham)
        self.assertAlmostEqual(lat[0],33.44929,delta=0.01)
        self.assertAlmostEqual(lon,-112.359,delta=0.01)
        # TODO - Test with array and masks (for Brg/Dist also?)


class TestLocalizerScale(unittest.TestCase):
    def test_basic_operation(self):
        rwy = A(name='test',
                value=[{'runway': {'end':
                                   {'latitude': 25.262131,
                                    'longitude': 55.347572},
                                   'localizer': {'beam_width': 4.5,
                                                 'frequency': 111300.0},
                                   'start': {'latitude': 25.243322,
                                             'longitude': 55.381519},
                                   }}])
        result = localizer_scale(rwy.value[0])
        self.assertAlmostEqual(result, 0.9)

    def test_no_beam_width(self):
        rwy = A(name='test',
                value=[{'runway': {'end':
                                   {'latitude': 25.262131,
                                    'longitude': 55.347572},
                                   'start': {'latitude': 25.243322,
                                             'longitude': 55.381519},
                                   }}])
        result = localizer_scale(rwy.value[0])
        self.assertGreater(result, 1.2)
        self.assertLess(result, 1.3)

    def test_no_beam_width_or_length(self):
        rwy = A(name='test',
                value=[None])
        result = localizer_scale(rwy.value[0])
        self.assertGreater(result, 1.2)
        self.assertLess(result, 1.3)


class TestBlendEquispacedSensors(unittest.TestCase):
    def test_blend_alternate_sensors_basic(self):
        array_1 = np.ma.array([0, 0, 1, 1],dtype=float)
        array_2 = np.ma.array([5, 5, 6, 6],dtype=float)
        result = blend_equispaced_sensors (array_1, array_2)
        np.testing.assert_array_equal(result.data, [2.5,2.5,2.5,2.75,3.25,3.5,3.5,3.5])
        np.testing.assert_array_equal(result.mask, [False,False,False,False,
                                                   False,False,False,False])

    def test_blend_alternate_sensors_masked(self):
        array_1 = np.ma.array(data = [0, 0, 1, 1, 2, 2],dtype=float,
                              mask = [0, 1, 0, 0, 0, 1])
        array_2 = np.ma.array(data = [5, 5, 6, 6, 7, 7],dtype=float,
                              mask = [0, 0, 1, 0, 0, 1])
        result = blend_equispaced_sensors (array_1, array_2)
        np.testing.assert_array_equal(result.data,[2.5,5.0,5.0,5.0,1.0,1.0,
                                                   1.0,3.75,4.25,7.0,0.0,0.0])
        np.testing.assert_array_equal(result.mask, [False,False,False,False,False,False,
                                                   False,False,False,False,True,True])


class TestBlendNonequispacedSensors(unittest.TestCase):
    def test_blend_alternate_sensors_basic(self):
        array_1 = np.ma.array([0, 0, 1, 1],dtype=float)
        array_2 = np.ma.array([5, 5, 6, 6],dtype=float)
        result = blend_nonequispaced_sensors (array_1, array_2, 'Follow')
        np.testing.assert_array_equal(result.data, [2.5,2.5,2.5,3,3.5,3.5,3.5,3.5])
        np.testing.assert_array_equal(result.mask, [False,False,False,False,
                                                   False,False,False,True])

    def test_blend_alternate_sensors_both_mask(self):
        array_1 = np.ma.array([0, 0, 1, 1],dtype=float)
        array_2 = np.ma.array([5, 5, 6, 6],dtype=float)
        array_1[2] = np.ma.masked
        array_2[2] = np.ma.masked
        result = blend_nonequispaced_sensors (array_1, array_2, 'Follow')
        expected = np.ma.array(data=[2.5, 2.5, 2.5, 9, 9, 9, 3.5, 9],
                               mask = [0,   0,   0, 1, 1, 1,   0, 1])
        ma_test.assert_masked_array_equal(expected, result)

    def test_blend_alternate_sensors_one_mask(self):
        array_1 = np.ma.array([0, 0, 1, 1],dtype=float)
        array_2 = np.ma.array([5, 5, 6, 6],dtype=float)
        array_1[2] = np.ma.masked
        result = blend_nonequispaced_sensors (array_1, array_2, 'Follow')
        expected = np.ma.array(data=[2.5, 2.5, 2.5, 5.25, 5.75, 3.5, 3.5, 9],
                               mask = [0,   0,   0,    0,    0,   0,   0, 1])
        ma_test.assert_masked_array_equal(expected, result)

    def test_blend_alternate_sensors_reverse(self):
        array_1 = np.ma.array([0, 0, 1, 1],dtype=float)
        array_2 = np.ma.array([5, 5, 6, 6],dtype=float)
        result = blend_nonequispaced_sensors (array_1, array_2, 'Precede')
        np.testing.assert_array_equal(result.data, [2.5,2.5,2.5,2.5,3,3.5,3.5,3.5])
        np.testing.assert_array_equal(result.mask, [True,False,False,False,
                                                    False,False,False,False])

    def test_blend_example(self):
        # segment 681b27f775450719e1db6c6d5d14ab259949019bec2aff451b40d2eb9fb0ef1e
        param_one = load_compressed(os.path.join(test_data_path, '681b27f77545_blend_param_one.npz'))
        param_two = load_compressed(os.path.join(test_data_path, '681b27f77545_blend_param_two.npz'))
        result = blend_nonequispaced_sensors (param_one, param_two, 'Follow')
        np.testing.assert_array_equal(
            np.ma.clump_masked(result),
            [slice(21006, 21008, None),]
        )
        np.testing.assert_array_equal(
            runs_of_ones(result > 0.0),
            [slice(469, 910, None),
             slice(915, 922, None),
             slice(967, 20384, None),
             slice(20387, 20404, None)])


class TestBump(unittest.TestCase):

    def setUp(self):
        self.accel = P('Test',
                       np.ma.array([2,0,0,0,1,0,0,0,0,0,0,0,0,3,0,0],
                                   dtype=float),
                       frequency=2.0,
                       offset=0.0)

    def test_bump(self):
        instant = 6.5

        self.assertEqual(bump(self.accel, instant)[0], 4.0)
        self.assertEqual(bump(self.accel, instant)[1], 1.0)

    def test_bump_start(self):
        instant = 6.5

        self.assertEqual(bump(self.accel, instant, start=0)[0], 0.0)
        self.assertEqual(bump(self.accel, instant, start=0)[1], 2.0)

    def test_bump_end(self):
        instant = 6.5

        self.assertEqual(bump(self.accel, instant, end=14.5)[0], 13.0)
        self.assertEqual(bump(self.accel, instant, end=14.5)[1], 3.0)


class TestIncludingTransition(unittest.TestCase):
    flap_map_1 = {0: '0', 15: '15', 30: '30', 45: '45'}
    flap_map_2 = {0: '0', 1: '1', 5: '5', 15: '15', 20: '20', 25: '25', 30: '30'}
    flap_map_3 = {0: '0', 15: '15', 30: '30'}
    flap_map_4 = {0: '0', 1: '1', 2: '2', 5: '5', 10: '10', 15: '15', 25: '25', 30: '30', 40: '40'}
    hz=2

    @staticmethod
    def _plot(array, flap_inc):
        from analysis_engine.plot_flight import plot_parameter
        plot_parameter(array, show=False)
        plot_parameter(flap_inc)

    def test_including_transition_1(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_1.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        self.assertTrue(np.ma.all(flap_inc[0:11] == 0))
        self.assertTrue(np.ma.all(flap_inc[11:29] == 15))

    @unittest.skip('Not Implemented')
    def test_including_transition_2(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_2.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        #self._plot(array, flap_inc)
        #sample is too short
        self.assertEqual(flap_inc.tolist(), [15] * 23 + [0] * 14)

    def test_including_transition_3(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_3.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        self.assertTrue(np.ma.all(flap_inc[0:11] == 0))
        self.assertTrue(np.ma.all(flap_inc[11:33] == 15))

    def test_including_transition_4(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_4.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        self.assertTrue(np.ma.all(flap_inc[0:11] == 15))
        self.assertTrue(np.ma.all(flap_inc[11:29] == 30))

    def test_including_transition_5(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_5.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        self.assertEqual(flap_inc.tolist(), [30] * 24)

    def test_including_transition_6(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_6.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        self.assertTrue(np.ma.all(flap_inc[40:49] == 0))
        self.assertTrue(np.ma.all(flap_inc[24:40] == 15))
        self.assertTrue(np.ma.all(flap_inc[0:24] == 30))

    @unittest.skip('Not Implemented')
    def test_including_transition_7(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_7.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        # this test doesn't represent real world example - it's a very short sample
        # of data. Current algorithm won't work properly on bits of data without
        # any transition / starting at random angle
        self.assertEqual(flap_inc.tolist(), [0] * 23)

    def test_including_transition_8(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_8.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        self.assertEqual(flap_inc.tolist(), [0] * 40)

    def test_including_transition_9(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_9.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        self.assertTrue(np.ma.all(flap_inc[0:11] == 0))
        self.assertTrue(np.ma.all(flap_inc[11:29] == 15))

    def test_including_transition_10(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_10.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        self.assertTrue(np.ma.all(flap_inc[31:40] == 0))
        self.assertTrue(np.ma.all(flap_inc[2:31] == 15))

    def test_including_transition_11(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_11.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        self.assertTrue(np.ma.all(flap_inc[0:11] == 0))
        self.assertTrue(np.ma.all(flap_inc[11:29] == 15))

    @unittest.skip('Not Implemented')
    def test_including_transition_12(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_12.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        # this example wont work with the new algorithm as the flaps didn't
        # transition to position 15 and this is expected - prepending the array
        # with few values lower than 15 works just fine, therefore in a real-world
        # case the results are as expected
        self.assertEqual(flap_inc.tolist(), [15] * 16 + [30] * 17)

    def test_including_transition_13(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_13.npz'))
        flap_inc = including_transition(array, self.flap_map_1)
        self.assertTrue(np.ma.all(flap_inc[44:53] == 0))
        self.assertTrue(np.ma.all(flap_inc[26:44] == 15))
        self.assertTrue(np.ma.all(flap_inc[0:26] == 30))

    def test_including_transition_14(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_14.npz'))
        flap_inc = including_transition(array, self.flap_map_2)
        self.assertTrue(np.ma.all(flap_inc[0:11] == 0))
        self.assertTrue(np.ma.all(flap_inc[11:31] == 1))
        self.assertTrue(np.ma.all(flap_inc[31:61] == 5))

    def test_including_transition_15(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_15.npz'))
        flap_inc = including_transition(array, self.flap_map_2)
        self.assertTrue(np.ma.all(flap_inc[55:72] == 0))
        self.assertTrue(np.ma.all(flap_inc[37:55] == 1))
        self.assertTrue(np.ma.all(flap_inc[0:37] == 5))

    def test_including_transition_16(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_16.npz'))
        flap_inc = including_transition(array, self.flap_map_2)
        self.assertTrue(np.ma.all(flap_inc[0:11] == 0))
        self.assertTrue(np.ma.all(flap_inc[11:45] == 1))

    def test_including_transition_17(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_17.npz'))
        flap_inc = including_transition(array, self.flap_map_2)
        self.assertTrue(np.ma.all(flap_inc[0:12] == 1))
        self.assertTrue(np.ma.all(flap_inc[12:46] == 5))

    def test_including_transition_18(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_18.npz'))
        flap_inc = including_transition(array, self.flap_map_2)
        self.assertTrue(np.ma.all(flap_inc[0:11] == 5))
        self.assertTrue(np.ma.all(flap_inc[11:23] == 15))
        self.assertTrue(np.ma.all(flap_inc[23:43] == 20))

    def test_including_transition_19(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_19.npz'))
        flap_inc = including_transition(array, self.flap_map_2)
        self.assertTrue(np.ma.all(flap_inc[0:16] == 20))
        self.assertTrue(np.ma.all(flap_inc[16:65] == 25))

    def test_including_transition_20(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_20.npz'))
        flap_inc = including_transition(array, self.flap_map_2)
        # allow small variance in transition points
        self.assertTrue(np.ma.all(flap_inc[87:106] == 0))
        self.assertTrue(np.ma.all(flap_inc[69:87] == 1))
        self.assertTrue(np.ma.all(flap_inc[44:69] == 5))
        self.assertTrue(np.ma.all(flap_inc[32:44] == 15))
        self.assertTrue(np.ma.all(flap_inc[26:32] == 20))
        self.assertTrue(np.ma.all(flap_inc[18:26] == 25))
        self.assertTrue(np.ma.all(flap_inc[0:18] == 30))

    def test_including_transition_21(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_21.npz'))
        flap_inc = including_transition(array, self.flap_map_3)
        self.assertTrue(flap_inc.mask[:4].all())
        # no flap 1 as it's not in the flap map
        self.assertTrue(np.ma.all(flap_inc[5:3031] == 0))
        self.assertTrue(np.ma.all(flap_inc[3899:8790] == 0))
        self.assertTrue(np.ma.all(flap_inc[9271:10224] == 0))
        self.assertTrue(np.ma.all(flap_inc[3031:3899] == 15))
        self.assertTrue(np.ma.all(flap_inc[8790:8848] == 15))
        self.assertTrue(np.ma.all(flap_inc[9256:9271] == 15))
        self.assertTrue(np.ma.all(flap_inc[8848:9256] == 30))

    def test_including_transition_22(self):
        array = load_compressed(os.path.join(test_data_path, 'flap_including_transition_flap_angle.npz'))
        flap_inc = including_transition(array, self.flap_map_4, self.hz)
        self.assertTrue(np.ma.all(flap_inc[0:91] == 0))
        self.assertTrue(np.ma.all(flap_inc[543:2369] == 0))
        self.assertTrue(np.ma.all(flap_inc[2737:3488] == 0))
        self.assertTrue(np.ma.all(flap_inc[3993:4184] == 0))
        self.assertTrue(np.ma.all(flap_inc[91:98] == 1))
        self.assertTrue(np.ma.all(flap_inc[535:543] == 1))
        self.assertTrue(np.ma.all(flap_inc[2369:2427] == 1))
        self.assertTrue(np.ma.all(flap_inc[2729:2737] == 1))
        self.assertTrue(np.ma.all(flap_inc[3488:3534] == 1))
        self.assertTrue(np.ma.all(flap_inc[3985:3993] == 1))
        self.assertTrue(np.ma.all(flap_inc[98:106] == 2))
        self.assertTrue(np.ma.all(flap_inc[527:535] == 2))
        self.assertTrue(np.ma.all(flap_inc[2427:2435] == 2))
        self.assertTrue(np.ma.all(flap_inc[2721:2729] == 2))
        self.assertTrue(np.ma.all(flap_inc[3534:3542] == 2))
        self.assertTrue(np.ma.all(flap_inc[3977:3985] == 2))
        self.assertTrue(np.ma.all(flap_inc[106:527] == 5))
        self.assertTrue(np.ma.all(flap_inc[2435:2534] == 5))
        self.assertTrue(np.ma.all(flap_inc[2711:2721] == 5))
        self.assertTrue(np.ma.all(flap_inc[3542:3636] == 5))
        self.assertTrue(np.ma.all(flap_inc[3969:3977] == 5))
        self.assertTrue(np.ma.all(flap_inc[2540:2555] == 15))
        self.assertTrue(np.ma.all(flap_inc[2657:2707] == 15))
        self.assertTrue(np.ma.all(flap_inc[3641:3654] == 15))
        self.assertTrue(np.ma.all(flap_inc[3962:3964] == 15))
        self.assertTrue(np.ma.all(flap_inc[2555:2556] == 25))
        self.assertTrue(np.ma.all(flap_inc[2651:2657] == 25))
        self.assertTrue(np.ma.all(flap_inc[3656:3658] == 25))
        self.assertTrue(np.ma.all(flap_inc[3960:3962] == 25))
        self.assertTrue(np.ma.all(flap_inc[2557:2564] == 30))
        self.assertTrue(np.ma.all(flap_inc[2650:2651] == 30))
        self.assertTrue(np.ma.all(flap_inc[3658:3666] == 30))
        self.assertTrue(np.ma.all(flap_inc[3959:3960] == 30))
        self.assertTrue(np.ma.all(flap_inc[2564:2650] == 40))
        self.assertTrue(np.ma.all(flap_inc[3666:3959] == 40))


    def test_including_transition_23(self):
        array = load_compressed(os.path.join(test_data_path, 'flap_including_transition_flap_angle_2.npz'))
        flap_inc = including_transition(array, self.flap_map_4)
        self.assertTrue(flap_inc.mask[0])
        self.assertTrue(np.ma.all(flap_inc[2:2756] == 0))
        self.assertTrue(np.ma.all(flap_inc[3642:11798] == 0))
        self.assertTrue(np.ma.all(flap_inc[12702:13696] == 0))
        self.assertTrue(np.ma.all(flap_inc[2756:3642] == 1))
        self.assertTrue(np.ma.all(flap_inc[11798:12129] == 1))
        self.assertTrue(np.ma.all(flap_inc[12686:12702] == 1))
        self.assertTrue(np.ma.all(flap_inc[12129:12133] == 2))
        self.assertTrue(np.ma.all(flap_inc[12682:12686] == 2))
        self.assertTrue(np.ma.all(flap_inc[12133:12245] == 5))
        self.assertTrue(np.ma.all(flap_inc[12674:12682] == 5))
        self.assertTrue(np.ma.all(flap_inc[12251:12308] == 15))
        self.assertTrue(np.ma.all(flap_inc[12664:12669] == 15))
        self.assertTrue(np.ma.all(flap_inc[12308:12314] == 25))
        self.assertTrue(np.ma.all(flap_inc[12656:12664] == 25))
        self.assertTrue(np.ma.all(flap_inc[12314:12656] == 30))

    def test_including_transition_24(self):
        '''
        Test that all flap settings are included from 0 to 30.
        '''
        array = load_compressed(os.path.join(test_data_path, 'flap_including_transition_flap_angle_3.npz'))
        flap_inc = including_transition(array, self.flap_map_4)
        self.assertTrue(np.ma.all(flap_inc[0:73] == 0))
        self.assertTrue(np.ma.all(flap_inc[411:2131] == 0))
        self.assertTrue(np.ma.all(flap_inc[2408:2624] == 0))
        self.assertTrue(np.ma.all(flap_inc[73:411] == 1))
        self.assertTrue(np.ma.all(flap_inc[2131:2143] == 1))
        self.assertTrue(np.ma.all(flap_inc[2395:2408] == 1))
        self.assertTrue(np.ma.all(flap_inc[2143:2145] == 2))
        self.assertTrue(np.ma.all(flap_inc[2393:2395] == 2))
        self.assertTrue(np.ma.all(flap_inc[2145:2163] == 5))
        self.assertTrue(np.ma.all(flap_inc[2389:2393] == 5))
        self.assertTrue(np.ma.all(flap_inc[2174:2192] == 15))
        self.assertTrue(np.ma.all(flap_inc[2384:2386] == 15))
        self.assertTrue(np.ma.all(flap_inc[2192:2203] == 25))
        self.assertTrue(np.ma.all(flap_inc[2380:2384] == 25))
        self.assertTrue(np.ma.all(flap_inc[2203:2380] == 30))

    def test_including_transition_25(self):
        '''
        Test that including transition does not jump prematurely to 1.
        '''
        array = load_compressed(os.path.join(test_data_path, 'flap_including_transition_flap_angle_4.npz'))
        flap_inc = including_transition(array, self.flap_map_4, hz=self.hz)
        self.assertTrue(np.ma.all(flap_inc[557:2663] == 0))
        self.assertTrue(np.ma.all(flap_inc[2952:3136] == 0))
        self.assertTrue(np.ma.all(flap_inc[0:159] == 1))
        self.assertTrue(np.ma.all(flap_inc[544:557] == 1))
        self.assertTrue(np.ma.all(flap_inc[2663:2708] == 1))
        self.assertTrue(np.ma.all(flap_inc[2939:2952] == 1))
        self.assertTrue(np.ma.all(flap_inc[159:170] == 2))
        self.assertTrue(np.ma.all(flap_inc[541:544] == 2))
        self.assertTrue(np.ma.all(flap_inc[2708:2710] == 2))
        self.assertTrue(np.ma.all(flap_inc[2937:2939] == 2))
        self.assertTrue(np.ma.all(flap_inc[170:541] == 5))
        self.assertTrue(np.ma.all(flap_inc[2710:2776] == 5))
        self.assertTrue(np.ma.all(flap_inc[2934:2937] == 5))
        self.assertTrue(np.ma.all(flap_inc[2776:2780] == 10))
        self.assertTrue(np.ma.all(flap_inc[2931:2934] == 10))
        self.assertTrue(np.ma.all(flap_inc[2780:2784] == 15))
        self.assertTrue(np.ma.all(flap_inc[2929:2931] == 15))
        self.assertTrue(np.ma.all(flap_inc[2784:2790] == 25))
        self.assertTrue(np.ma.all(flap_inc[2926:2929] == 25))
        self.assertTrue(np.ma.all(flap_inc[2790:2926] == 30))


class TestCalculateSurfaceAngle(unittest.TestCase):
    flap_map_1 = {0: '0', 15: '15', 30: '30', 45: '45'}
    flap_map_2 = {0: '0', 1: '1', 5: '5', 15: '15', 20: '20', 25: '25', 30: '30'}
    flap_map_3 = {0: '0', 15: '15', 30: '30'}

    slat_map_1 = {0: '0', 15: '15', 25: '25'}
    slat_map_2 = {0: '0', 50: '50', 100: '100'}

    @staticmethod
    def _call_calculate(func, parameter):
        args = [parameter, mock.Mock(), mock.Mock(), mock.Mock()]
        return (
            func('excluding', *args)[1],
            func('including', *args)[1],
            func('lever', *args)[1],
        )

    @classmethod
    def _calculate(cls, func, array, hz=2, align=False):
        parameter = P(array=array, frequency=hz)
        if align:
            return cls._call_calculate(func, parameter)

        with patch('analysis_engine.library.align_args') as align_args:
            align_args.return_value = array
            return cls._call_calculate(func, parameter)

    @classmethod
    @patch('analysis_engine.library.at')
    def _calculate_flap(cls, array, map, at, align=False):
        at.get_flap_map.return_value = map
        return cls._calculate(calculate_flap, array, align=align)

    @classmethod
    @patch('analysis_engine.library.at')
    def _calculate_slat(cls, array, map, at, align=False):
        at.get_slat_map.return_value = map
        return cls._calculate(calculate_slat, array, align=align)

    def test_calculate_flap_1(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_1.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        early = [0] * 11 + [15] * 18
        late = [0] * 20 + [15] * 9
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_2(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_2.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        early = [15] * 11 + [0] * 26
        late = [15] * 28 + [0] * 9
        self.assertEqual(flap_exc.tolist(), early)
        self.assertEqual(flap_inc.tolist(), late)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_3(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_3.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        early = [0] * 8 + [15] * 25
        late = [0] * 24 + [15] * 9
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_4(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_4.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array,  self.flap_map_1)
        early = [15] * 11 + [30] * 18
        late = [15] * 20 + [30] * 9
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_5(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_5.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        flat = [30] * 24
        self.assertEqual(flap_exc.tolist(), flat)
        self.assertEqual(flap_inc.tolist(), flat)
        self.assertEqual(flap_lev.tolist(), flat)

    def test_calculate_flap_6(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_6.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        early = [30] * 11 + [0] * 38
        late = [30] * 41 + [0] * 8
        self.assertEqual(flap_exc.tolist(), early)
        self.assertEqual(flap_inc.tolist(), late)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_7(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_7.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        flat = [0] * 23
        self.assertEqual(flap_exc.tolist(), flat)
        self.assertEqual(flap_inc.tolist(), flat)
        self.assertEqual(flap_lev.tolist(), flat)

    def test_calculate_flap_8(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_8.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        flat = [0] * 40
        self.assertEqual(flap_exc.tolist(), flat)
        self.assertEqual(flap_inc.tolist(), flat)
        self.assertEqual(flap_lev.tolist(), flat)

    def test_calculate_flap_9(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_9.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        early = [0] * 11 + [15] * 18
        late = [0] * 20 + [15] * 9
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_10(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_10.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        early = [15] * 11 + [0] * 29
        late = [15] * 31 + [0] * 9
        self.assertEqual(flap_exc.tolist(), early)
        self.assertEqual(flap_inc.tolist(), late)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_11(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_11.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        early = [0] * 11 + [15] * 18
        late = [0] * 20 + [15] * 9
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_12(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_12.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        early = [15] * 11 + [30] * 22
        late = [15] * 24 + [30] * 9
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_13(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_13.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_1)
        early = [30] * 11 + [0] * 42
        late = [30] * 44 + [0] * 9
        self.assertEqual(flap_exc.tolist(), early)
        self.assertEqual(flap_inc.tolist(), late)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_14(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_14.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_2)
        early = [0] * 11 + [1] * 20 + [5] * 30
        late = [0] * 17 + [1] * 35 + [5] * 9
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_15(self):
        # This test case is a little problematic as the flap setting of 1 is skipped for Flap Excluding Transition.
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_15.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_2)
        early = [5] * 11 + [1] * 27 + [0] * 34
        late = [5] * 38 + [0] * 34
        self.assertEqual(flap_exc.tolist(), early)
        self.assertEqual(flap_inc.tolist(), late)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_16(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_16.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_2)
        early = [0] * 11 + [1] * 34
        late = [0] * 17 + [1] * 28
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_17(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_17.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_2)
        early = [1] * 11 + [5] * 35
        late = [1] * 37 + [5] * 9
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_18(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_18.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_2)
        early = [5] * 11 + [20] * 32
        late = [5] * 31 + [20] * 12
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_19(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_19.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_2)
        early = [20] * 35 + [25] * 30
        late = [20] * 41 + [25] * 24
        self.assertEqual(flap_exc.tolist(), late)
        self.assertEqual(flap_inc.tolist(), early)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_20(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_20.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_2)
        early = [30] * 11 + [1] * 59 + [0] * 36
        late = [30] * 70 + [0] * 36
        self.assertEqual(flap_exc.tolist(), early)
        self.assertEqual(flap_inc.tolist(), late)
        self.assertEqual(flap_lev.tolist(), early)

    def test_calculate_flap_21(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_flap_21.npz'))
        flap_exc, flap_inc, flap_lev = self._calculate_flap(array, self.flap_map_3)

        self.assertTrue(flap_exc.mask[:4].all())
        self.assertTrue(np.ma.all(flap_exc[4:3040] == 0))
        self.assertTrue(np.ma.all(flap_exc[3040:3886] == 15))
        self.assertTrue(np.ma.all(flap_exc[3886:8799] == 0))
        self.assertTrue(np.ma.all(flap_exc[8799:8858] == 15))
        self.assertTrue(np.ma.all(flap_exc[8858:9243] == 30))
        self.assertTrue(np.ma.all(flap_exc[9243:-6] == 0))
        self.assertTrue(flap_exc.mask[-6:].all())

        self.assertTrue(np.ma.all(flap_inc[:3031] == 0))
        self.assertTrue(np.ma.all(flap_inc[3031:3899] == 15))
        self.assertTrue(np.ma.all(flap_inc[3899:8790] == 0))
        self.assertTrue(np.ma.all(flap_inc[8790:8848] == 15))
        self.assertTrue(np.ma.all(flap_inc[8848:9271] == 30))
        self.assertTrue(np.ma.all(flap_inc[9271:] == 0))

        self.assertTrue(np.ma.all(flap_lev[:3031] == 0))
        self.assertTrue(np.ma.all(flap_lev[3031:3886] == 15))
        self.assertTrue(np.ma.all(flap_lev[3886:8790] == 0))
        self.assertTrue(np.ma.all(flap_lev[8790:8848] == 15))
        self.assertTrue(np.ma.all(flap_lev[8848:9243] == 30))
        self.assertTrue(np.ma.all(flap_lev[9243:] == 0))

    def test_calculate_slat_1(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_slat_1.npz'))
        slat_exc, slat_inc, slat_lev = self._calculate_slat(array, self.slat_map_1)
        early = [0] * 11 + [15] * 34
        late = [0] * 36 + [15] * 9
        self.assertEqual(slat_exc.tolist(), late)
        self.assertEqual(slat_inc.tolist(), early)
        self.assertEqual(slat_lev.tolist(), early)

    def test_calculate_slat_2(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_slat_2.npz'))
        slat_exc, slat_inc, slat_lev = self._calculate_slat(array, self.slat_map_1)
        early = [15] * 11 + [0] * 33
        late = [15] * 35 + [0] * 9
        self.assertEqual(slat_exc.tolist(), early)
        self.assertEqual(slat_inc.tolist(), late)
        self.assertEqual(slat_lev.tolist(), early)

    def test_calculate_slat_3(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_slat_3.npz'))
        slat_exc, slat_inc, slat_lev = self._calculate_slat(array, self.slat_map_2)
        early = [0] * 11 + [50] * 31
        late = [0] * 33 + [50] * 9
        self.assertEqual(slat_exc.tolist(), late)
        self.assertEqual(slat_inc.tolist(), early)
        self.assertEqual(slat_lev.tolist(), early)

    def test_calculate_slat_4(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_slat_4.npz'))
        slat_exc, slat_inc, slat_lev = self._calculate_slat(array, self.slat_map_2)
        early = [100] * 11 + [50] * 19
        late = [100] * 21 + [50] * 9
        self.assertEqual(slat_exc.tolist(), early)
        self.assertEqual(slat_inc.tolist(), late)
        self.assertEqual(slat_lev.tolist(), early)

    def test_calculate_slat_5(self):
        array = load_compressed(os.path.join(test_data_path, 'calculate_slat_5.npz'))
        slat_exc, slat_inc, slat_lev = self._calculate_slat(array, self.slat_map_2)
        early = [50] * 11 + [0] * 31
        late = [50] * 33 + [0] * 9
        self.assertEqual(slat_exc.tolist(), early)
        self.assertEqual(slat_inc.tolist(), late)
        self.assertEqual(slat_lev.tolist(), early)


class TestCalculateTimebase(unittest.TestCase):
    last_year = datetime.now().year - 1
    def test_calculate_timebase_zero_year(self):
        # 6th second is the first valid datetime(2020,12,25,23,59,0)
        years = [None] * 6 + [0] * 19  # 6 sec offset
        months = [None] * 5 + [12] * 20
        days = [None] * 4 + [24] * 5 + [25] * 16
        hours = [None] * 3 + [23] * 7 + [0] * 15
        mins = [None] * 2 + [59] * 10 + [1] * 13
        secs = [None] * 1 + list(range(55, 60)) + list(range(19))  # 6th second in next hr
        start_dt = calculate_timebase(years, months, days, hours, mins, secs)

        #>>> datetime(2020,12,25,0,1,19) - timedelta(seconds=25)
        #datetime.datetime(2020, 12, 25, 0, 0, 50)
        self.assertEqual(start_dt, datetime(2000, 12, 25, 0, 0, 54, tzinfo=pytz.utc))

    @unittest.skip("Validation of year moved outside of calculate_timebase - invalid test?")
    def test_calculate_timebase_future_year(self):
        # a few valid years followed by many invalid
        years = [self.last_year] * 15 + [2999] * 10
        months = [None] * 5 + [12] * 20
        days = [None] * 4 + [24] * 5 + [25] * 16
        hours = [None] * 3 + [23] * 7 + [0] * 15
        mins = [None] * 2 + [59] * 10 + [1] * 13
        secs = [None] * 1 + list(range(55, 60)) + list(range(19))  # 6th second in next hr
        start_dt = calculate_timebase(years, months, days, hours, mins, secs)

        #>>> datetime(2020,12,25,0,1,19) - timedelta(seconds=25)
        #datetime.datetime(2020, 12, 25, 0, 0, 50)
        self.assertEqual(start_dt, datetime(self.last_year, 12, 24, 23, 58, 54, tzinfo=pytz.utc))

    def test_calculate_timebase(self):
        # 6th second is the first valid datetime(2020,12,25,23,59,0)
        years = [None] * 6 + [self.last_year] * 19  # 6 sec offset
        months = [None] * 5 + [12] * 20
        days = [None] * 4 + [24] * 5 + [25] * 16
        hours = [None] * 3 + [23] * 7 + [0] * 15
        mins = [None] * 2 + [59] * 10 + [1] * 13
        secs = [None] * 1 + list(range(55, 60)) + list(range(19))  # 6th second in next hr
        start_dt = calculate_timebase(years, months, days, hours, mins, secs)

        #>>> datetime(2020,12,25,0,1,19) - timedelta(seconds=25)
        #datetime.datetime(2020, 12, 25, 0, 0, 50)
        self.assertEqual(start_dt, datetime(self.last_year, 12, 25, 0, 0, 54, tzinfo=pytz.utc))

    def test_no_valid_datetimes_raises_valueerror(self):
        years = [None] * 25
        months = [None] * 25
        days = [None] * 4 + [24] * 5 + [25] * 16
        hours = [None] * 3 + [23] * 7 + [0] * 15
        mins = [None] * 2 + [59] * 10 + [1] * 13
        secs = [None] * 1 + list(range(55, 60)) + list(range(19))  # 6th second in next hr
        self.assertRaises(InvalidDatetime, calculate_timebase, years, months, days, hours, mins, secs)

    def test_uneven_length_arrays(self):
        "Tests that the uneven drabs raises ValueError"
        # You should always pass in complete arrays at the moment!
        years = [None] * 1 + [2020] * 10  # uneven
        months = [None] * 5 + [12] * 20
        days = [None] * 4 + [24] * 5 + [25] * 16
        hours = [None] * 3 + [23] * 7 + [0] * 1 # uneven
        mins = [None] * 2 + [59] * 10 + [1] * 13
        secs = [None] * 1 + list(range(55, 60)) + list(range(19))
        self.assertRaises(ValueError, calculate_timebase,
                          years, months, days, hours, mins, secs)

    def test_no_change_in_dt_picks_it_as_start(self):
        # also tests using numpy masked arrays
        years = np.ma.array([self.last_year] * 20)  # 6 sec offset
        months = np.ma.array([12] * 20)
        days = np.ma.array([25] * 20)
        hours = np.ma.array([23] * 20)
        mins = np.ma.array([0] * 20)
        secs = np.ma.array([0] * 20) # 6th second in next hr
        start_dt = calculate_timebase(years, months, days, hours, mins, secs)
        self.assertEqual(start_dt, datetime(self.last_year,12,25,23,0,0, tzinfo=pytz.utc))

    def test_real_data_params_2_digit_year(self):
        years = load_compressed(os.path.join(test_data_path, 'year.npz'))
        months = load_compressed(os.path.join(test_data_path, 'month.npz'))
        days = load_compressed(os.path.join(test_data_path, 'day.npz'))
        hours = load_compressed(os.path.join(test_data_path, 'hour.npz'))
        mins = load_compressed(os.path.join(test_data_path, 'minute.npz'))
        secs = load_compressed(os.path.join(test_data_path, 'second.npz'))
        start_dt = calculate_timebase(years, months, days, hours, mins, secs)
        self.assertEqual(start_dt, datetime(2011, 12, 30, 8, 20, 36, tzinfo=pytz.utc))

    def test_real_data_params_no_year(self):
        months = load_compressed(os.path.join(test_data_path, 'month.npz'))
        days = load_compressed(os.path.join(test_data_path, 'day.npz'))
        hours = load_compressed(os.path.join(test_data_path, 'hour.npz'))
        mins = load_compressed(os.path.join(test_data_path, 'minute.npz'))
        secs = load_compressed(os.path.join(test_data_path, 'second.npz'))
        years = np.array([2012]*len(months)) # fixed year
        start_dt = calculate_timebase(years, months, days, hours, mins, secs)
        self.assertEqual(start_dt, datetime(2012, 12, 30, 8, 20, 36, tzinfo=pytz.utc))

    def test_masked_values_are_ignored(self):
        # If masked values are interpreted as 0, the start_dt is incorrectly calculated as 2001
        start_dt = calculate_timebase(*load_compressed(os.path.join(test_data_path, 'calculate_timebase_01.npz')))
        self.assertEqual(start_dt, datetime(2020, 1, 31, 0, 1, 37, tzinfo=pytz.utc))

    @unittest.skip("Implement if this is a requirement, currently "
                   "all parameters are aligned before this is being used.")
    def test_using_offset_for_seconds(self):
        # TODO: check offset milliseconds are applied to the timestamps
        self.assertFalse(True)


class TestConvertTwoDigitToFourDigitYear(unittest.TestCase):
    def test_convert_two_digit_to_four_digit_year(self):
        # WARNING - this test will fail next year(!)
        self.assertEqual(convert_two_digit_to_four_digit_year(99, '2012'), 1999)
        self.assertEqual(convert_two_digit_to_four_digit_year(13, '2012'), 1913)
        self.assertEqual(convert_two_digit_to_four_digit_year(12, '2012'), 2012) # will break next year
        self.assertEqual(convert_two_digit_to_four_digit_year(11, '2012'), 2011)
        self.assertEqual(convert_two_digit_to_four_digit_year(1, '2012'), 2001)


class TestCoReg(unittest.TestCase):
    def test_correlation_basic(self):
        x=np.ma.array([0,1,2,4,5,7], dtype=float)
        y=np.ma.array([2,4,5,3,6,8], dtype=float)
        correlate, slope, offset = coreg(y, indep_var=x)
        self.assertAlmostEqual(correlate, 0.818447591071135)
        self.assertAlmostEqual(slope, 0.669856459330144)
        self.assertAlmostEqual(offset, 2.54545454545455)

    def test_correlation_masked(self):
        x=np.ma.array([0,1,2,4,5,7], mask=[0,0,1,0,0,0], dtype=float)
        y=np.ma.array([2,4,5,3,6,8], mask=[0,0,0,0,1,0], dtype=float)
        correlate, slope, offset = coreg(y, indep_var=x)
        self.assertAlmostEqual(correlate, 0.841685056859012)
        self.assertAlmostEqual(slope, 0.7)
        self.assertAlmostEqual(offset, 2.15)

    def test_correlation_raises_error_unequal(self):
        x=np.ma.array([0,1,2,4,5,7], dtype=float)
        y=np.ma.array([-2,-4,-5,-3,-6], dtype=float)
        self.assertRaises(ValueError, coreg, y, indep_var=x)

    '''
    Now only a logger warning message.
    def test_correlation_raises_error_too_short(self):
        y=np.ma.array([1], dtype=float)
        self.assertRaises(ValueError, coreg, y)
    '''

    def test_correlation_constant_arrays(self):
        x=np.ma.array([0,0,0,0,0,0], dtype=float)
        y=np.ma.arange(6)
        self.assertEqual(coreg(x), (None, None, None))
        self.assertEqual(coreg(x, indep_var=y), (None, None, None))
        self.assertEqual(coreg(y, indep_var=x), (None, None, None))

    def test_correlation_constant_arrays_when_masked(self):
        x=np.ma.array([0,0,1,2,3,0], dtype=float)
        y=np.ma.array([1,2,3,4,5,6], mask=[0,0,1,1,1,0])
        self.assertEqual(coreg(x, indep_var=y), (None, None, None))
        self.assertEqual(coreg(y, indep_var=x), (None, None, None))

    def test_correlation_monotonic_independent_variable(self):
        y=np.ma.array([2,4,5,3,6,8], dtype=float)
        correlate, slope, offset = coreg(y)
        self.assertAlmostEqual(correlate, 0.841281820819169)
        self.assertAlmostEqual(slope, 0.971428571428571)
        self.assertAlmostEqual(offset, 2.23809523809524)

    def test_correlation_only_return(self):
        y=np.ma.array([2,4,5,3,6,8], dtype=float)
        correlate,d1,d2 = coreg(y)  # You need to cater for the three return arguments.
        self.assertAlmostEqual(correlate, 0.841281820819169)

    def test_correlation_forced_zero(self):
        y=np.ma.array([2,4,5,3,6,8], dtype=float)
        correlate, slope, offset = coreg(y, force_zero=True)
        self.assertAlmostEqual(slope, 1.58181818181818)
        self.assertAlmostEqual(offset, 0.0)

    def test_correlation_negative_slope(self):
        x=np.ma.array([0,1,2,4,5,7], dtype=float)
        y=np.ma.array([-2,-4,-5,-3,-6,-8], dtype=float)
        correlate, slope, offset = coreg(y,indep_var=x)
        self.assertAlmostEqual(correlate, 0.818447591071135)
        self.assertAlmostEqual(slope, -0.669856459330144)
        self.assertAlmostEqual(offset, -2.54545454545455)


class TestPositiveValue(unittest.TestCase):
    def test_positive_value(self):
        array = np.arange(10)
        self.assertEqual(positive_index(array, 0), 0)
        self.assertEqual(positive_index(array, 5), 5)
        self.assertEqual(positive_index(array, -3), 7)
        self.assertEqual(positive_index(array, -0.3), 0)
        self.assertEqual(positive_index(array, 10), 9)


class TestPowerFloor(unittest.TestCase):
    def test_power_floor(self):
        results = []
        for x in range(1, 128):
            results.append(power_floor(x))
        expected = ([1] +
                    [2] * 2 +
                    [4] * 4 +
                    [8] * 8 +
                    [16] * 16 +
                    [32] * 32 +
                    [64] * 64)
        self.assertEqual(results, expected)


class TestPowerCeil(unittest.TestCase):
    def test_power_ceil(self):
        self.assertEqual(power_ceil(0.2), 0.25)
        self.assertEqual(power_ceil(0.25), 0.25)
        self.assertEqual(power_ceil(2.4), 4)
        self.assertEqual(power_ceil(35), 64)


class TestNextUnmaskedValue(unittest.TestCase):
    def test_next_unmasked_value(self):
        # Entirely unmasked
        array = np.ma.arange(10)
        self.assertEqual(next_unmasked_value(array, 7), Value(7, 7))
        # Entirely masked.
        array.mask = True
        self.assertEqual(next_unmasked_value(array, 3), None)
        array.mask = False
        array.mask[3:8] = True
        self.assertEqual(next_unmasked_value(array, 4), Value(8, 8))


class TestPrevUnmaskedValue(unittest.TestCase):
    def test_prev_unmasked_value(self):
        # Entirely unmasked
        array = np.ma.arange(10)
        self.assertEqual(prev_unmasked_value(array, 7), Value(7, 7))
        # Entirely masked.
        array.mask = True
        self.assertEqual(prev_unmasked_value(array, 3), None)
        array.mask = False
        array.mask[2:8] = True
        self.assertEqual(prev_unmasked_value(array, 5), Value(1, 1))


class TestClosestUnmaskedValue(unittest.TestCase):
    def test_closest_unmasked_value(self):
        array = np.ma.arange(10)
        self.assertEqual(closest_unmasked_value(array, 5), Value(5, 5))
        # Floors index
        self.assertEqual(closest_unmasked_value(array, 5.5), Value(5, 5))
        self.assertEqual(closest_unmasked_value(array, -0.3), Value(0, 0))
        array[5:8] = np.ma.masked
        self.assertEqual(closest_unmasked_value(array, 5), Value(4, 4))
        array[5:8] = np.ma.masked
        self.assertEqual(closest_unmasked_value(array, 7), Value(8, 8))
        self.assertEqual(closest_unmasked_value(array, 3, start_index=2,
                                                stop_index=5),
                         Value(3, 3))
        # Handles index out of range
        self.assertEqual(closest_unmasked_value(array, 10), Value(9, 9))

    def test_closest_unmasked_value_negative_index(self):
        values = [
            0, 2503, 5012, 5003, 4968, 4925, 4844, 4893, 4476, 4385, 4395, 4332,
            4243, 4161, 4106, 4056, 3993, 3934, 3884, 3838, 3814, 3784, 3752,
            3720, 3688, 3656, 3625, 3597, 3561, 3530, 3502, 3479, 3452, 3425,
            3389, 3361, 3339, 3307, 3280, 3253, 3221, 3189, 3130, 3103, 3098,
            3098, 3093, 3084, 3066, 3053, 3039, 3030, 3021, 3003, 2989, 2984,
            2971, 2962, 2953, 2965, 2971, 2905, 2889, 2867, 2853, 2834, 2900,
            2767, 2753, 2708, 0,
        ]
        array = np.ma.array(values, mask=[True] * 2 + [False] * 68 + [True])
        self.assertEqual(closest_unmasked_value(array, -25), Value(46, 3093))
        self.assertEqual(closest_unmasked_value(array, -0.45), Value(2, 5012))

    def test_closest_unmasked_index_relative_to_start(self):
        array = np.ma.arange(10)
        self.assertEqual(closest_unmasked_value(array, 6, start_index=0),
                                                Value(6, 6))
        self.assertEqual(closest_unmasked_value(array, 6, start_index=3,
                                                stop_index=7), Value(6, 6))


class TestClumpMultistate(unittest.TestCase):
    # Reminder: clump_multistate(array, state, _slices, condition=True)
    def test_basic(self):
        # Includes test of passing single slice.
        values_mapping = {1: 'one', 2: 'two', 3: 'three'}
        array = np.ma.MaskedArray(data=[1, 2, 3, 2, 2, 1, 1],
                                  mask=[0, 0, 0, 0, 0, 0, 1])
        p = M('Test Node', array, values_mapping=values_mapping)
        result = clump_multistate(p.array, 'two', slice(0,7))
        expected = [slice(1, 2), slice(3, 5)]
        self.assertEqual(result, expected)
        result2 = clump_multistate(p.array, 'two')
        self.assertEqual(result2, expected)


    def test_complex(self):
        values_mapping = {1: 'one', 2: 'two', 3: 'three'}
        array = np.ma.MaskedArray(data=[1, 2, 3, 2, 2, 1, 1],
                                  mask=[0, 0, 0, 0, 0, 0, 1])
        p = M('Test Node', array, values_mapping=values_mapping)
        result = clump_multistate(p.array, 'three', [slice(0,7)], condition=False)
        expected = [slice(0, 2), slice(3, 6)] # Last value is masked
        self.assertEqual(result, expected)

    def test_null(self):
        values_mapping = {1: 'one', 2: 'two', 3: 'three'}
        array = np.ma.MaskedArray(data=[1, 2, 3, 2, 2, 1, 1],
                                  mask=[1, 0, 1, 0, 0, 0, 1])
        p = M('Test Node', array, values_mapping=values_mapping)
        result = clump_multistate(p.array, 'monty', [slice(0,7.5)], condition=True)
        expected = None
        self.assertEqual(result, expected)

    def test_multiple_slices(self):
        values_mapping = {1: 'one', 2: 'two', 3: 'three'}
        array = np.ma.MaskedArray(data=[1, 2, 3, 2, 2, 1, 1],
                                  mask=[0, 0, 0, 0, 0, 0, 0])
        p = M('Test Node', array, values_mapping=values_mapping)
        result = clump_multistate(p.array, 'two', [slice(0,2), slice(4,6)])
        expected = [slice(1, 2), slice(4, 5)]
        self.assertEqual(result, expected)

    def test_null_slice(self):
        values_mapping = {1: 'one', 2: 'two', 3: 'three'}
        array = np.ma.MaskedArray(data=[1, 2, 3, 2, 2, 1, 1],
                                  mask=[0, 0, 0, 0, 0, 0, 0])
        p = M('Test Node', array, values_mapping=values_mapping)
        result = clump_multistate(p.array, 'two', [])
        expected = []
        self.assertEqual(result, expected)

    def test_slice_not_at_start(self):
        values_mapping = {1: 'one', 2: 'two', 3: 'three'}
        array = np.ma.MaskedArray(data=[1, 2, 3, 2, 2, 1, 1],
                                  mask=[0, 0, 0, 0, 0, 0, 1])
        p = M('Test Node', array, values_mapping=values_mapping)
        result = clump_multistate(p.array, 'one', [slice(3,None)])
        expected = [slice(5, 6)]
        self.assertEqual(result, expected)


class TestCreatePhaseInside(unittest.TestCase):
    def test_phase_inside_basic(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)
        result = create_phase_inside(array, 1.0,0.0,2,5)
        answer = np.ma.array(data = [0,1,2,3,4,5,6,7],
                             mask = [1,1,0,0,0,0,1,1])
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_phase_inside_reversed(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)
        result = create_phase_inside(array, 1.0,0.1,5,2) # 2,5 > 5,2
        answer = np.ma.array(data = [0,1,2,3,4,5,6,7],
                             mask = [1,1,0,0,0,1,1,1])
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_phase_inside_positive_offset(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)
        result = create_phase_inside(array, 1.0,0.1,2,5)
        answer = np.ma.array(data = [0,1,2,3,4,5,6,7],
                             mask = [1,1,0,0,0,1,1,1])
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_phase_inside_negative_offset(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)
        result = create_phase_inside(array, 1.0,-0.1,2,5)
        answer = np.ma.array(data = [0,1,2,3,4,5,6,7],
                             mask = [1,1,1,0,0,0,1,1])
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_phase_inside_low_rate(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)*4
        result = create_phase_inside(array, 0.25,0.0,12,25)
        answer = np.ma.array(data = [0,4,8,12,16,20,24,28],
                             mask = [1,1,1,0,0,0,0,1])
        ma_test.assert_masked_array_approx_equal(result, answer)


class TestCreatePhaseOutside(unittest.TestCase):
    def test_phase_outside_low_rate(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)*4
        result = create_phase_outside(array, 0.25,0.0,7,21)
        answer = np.ma.array(data = [0,4,8,12,16,20,24,28],
                             mask = [0,0,1,1,1,1,0,0])
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_phase_inside_errors(self):
        # Reminder: create_phase_inside(reference, a, b)
        array = np.ma.arange(8)
        self.assertRaises(ValueError, create_phase_inside, array, 1, 0, -1, 5)
        self.assertRaises(ValueError, create_phase_inside, array, 1, 0, 10, 5)
        self.assertRaises(ValueError, create_phase_inside, array, 1, 0, 2, -1)
        self.assertRaises(ValueError, create_phase_inside, array, 1, 0, 2, 11)


class TestCycleCounter(unittest.TestCase):

    def setUp(self):
        self.array = \
            np.ma.sin(np.ma.arange(100) * 0.7 + 3) + \
            np.ma.sin(np.ma.arange(100) * 0.82)

    def test_cycle_counter(self):
        index, count = cycle_counter(self.array, 3.0, 10, 1.0, 0)
        self.assertEqual(index, 91)
        self.assertEqual(count, 3)

    def test_cycle_counter_with_offset(self):
        index, count = cycle_counter(self.array, 3.0, 10, 1.0, 1234)
        self.assertEqual(index, 1234 + 91)
        self.assertEqual(count, 3)

    def test_cycle_counter_too_slow(self):
        index, count = cycle_counter(self.array, 3.0, 1, 1.0, 0)
        self.assertEqual(index, None)
        self.assertEqual(count, None)

    def test_cycle_counter_empty(self):
        array = np.ma.array([])
        index, count = cycle_counter(array, 3.0, 10, 1.0, 0)
        self.assertEqual(index, None)
        self.assertEqual(count, None)


class TestCycleSelect(unittest.TestCase):

    def setUp(self):
        self.array = \
            np.ma.sin(np.ma.arange(100) * 0.7 + 3) + \
            np.ma.sin(np.ma.arange(100) * 0.82)

    def test_cycle_select(self):
        index, value = cycle_select(self.array, 3.0, 10, 1.0, 0)
        self.assertEqual(index, 25)
        self.assertAlmostEqual(value, 3.90451619)

    def test_incomplete_cycle_rejected(self):
        index, value = cycle_select(np.ma.array([0,0,5,5.0,2,2]),4.0, 3.0, 1.0)
        self.assertEqual(index, None)
        self.assertEqual(value, None)

    def test_full_cycle_identified(self):
        index, value = cycle_select(np.ma.array([1,0,5,5.0,0,1]),4.0, 4.0, 1.0)
        self.assertEqual(index, 2)
        self.assertEqual(value, 5)

    def test_cycle_select_with_offset(self):
        index, value = cycle_select(self.array, 3.0, 10, 1.0, 1234)
        self.assertEqual(index, 1234 + 25)
        self.assertAlmostEqual(value, 3.90451619)

    def test_cycle_select_too_slow(self):
        index, value = cycle_select(self.array, 3.0, 1, 1.0, 0)
        self.assertEqual(index, None)
        self.assertEqual(value, None)

    def test_cycle_select_empty(self):
        array = np.ma.array([])
        index, value = cycle_select(array, 3.0, 10, 1.0, 0)
        self.assertEqual(index, None)
        self.assertEqual(value, None)

    def test_cycle_select_negative_change(self):
        array = np.ma.array([0.0,3.0,-4.0,-2.0,0.0])
        index, value = cycle_select(array, 3.0, 10, 1.0, 0)
        self.assertEqual(index, 2)
        self.assertEqual(value, 4.0)


class TestCycleFinder(unittest.TestCase):

    def setUp(self):
        self.array = np.ma.array([0, 1, 3.8, 1, 0.3, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2])

    def test_cycle_finder_basic(self):
        idxs, vals = cycle_finder(self.array, min_step=2.1, include_ends=False)
        np.testing.assert_array_equal(idxs, [2, 4, 12])
        np.testing.assert_array_equal(vals, [3.8, 0.3, 4])

    def test_cycle_finder_default(self):
        idxs, vals = cycle_finder(self.array)
        np.testing.assert_array_equal(idxs, [0, 2, 4, 7, 9, 12, 14])
        np.testing.assert_array_equal(vals, [0., 3.8, 0.3, 3., 1., 4., 2.])

    def test_cycle_finder_single_peak(self):
        idxs, vals = cycle_finder(self.array, min_step=15)
        np.testing.assert_array_equal(idxs, 12)
        np.testing.assert_array_equal(vals, 4)

    def test_cycle_finder_single_trough(self):
        idxs, vals = cycle_finder(-self.array, min_step=15)
        np.testing.assert_array_equal(idxs, 12)
        np.testing.assert_array_equal(vals, -4)

    def test_cycle_finder_ramp(self):
        array = np.ma.array([0,1,2])
        idxs, vals = cycle_finder(array, include_ends=False)
        np.testing.assert_array_equal(idxs, None)
        np.testing.assert_array_equal(vals, None)

    def test_cycle_finder_removals(self):
        array = np.ma.array([0, 1, 2, 1, 2, 3, 2, 1, 2, 3, 4, 5, 4, 5, 6])
        idxs, vals = cycle_finder(array, min_step=1.5)
        np.testing.assert_array_equal(idxs, [0, 5, 7, 14])
        np.testing.assert_array_equal(vals, [0, 3, 1, 6])


class TestCycleMatch(unittest.TestCase):
    def test_find_a_match(self):
        cycles = np.array([10, 30, 100, 129, 144.0])
        # For a given value, find the closest and return the preceding and following values.
        self.assertEqual(cycle_match(30.0, cycles), (10, 100))
        self.assertEqual(cycle_match(32, cycles), (10, 100))
        # no matching index
        self.assertRaises(ValueError, cycle_match, 36, cycles, dist=1)
        # no previous
        self.assertEqual(cycle_match(8, cycles), (None, 30))
        # no next
        self.assertEqual(cycle_match(160, cycles, dist=50), (129, None))
        # finds closest
        self.assertEqual(cycle_match(120, cycles, dist=4000), (100, 144))
        # equal distance rounds down
        self.assertEqual(cycle_match(20, cycles, dist=12), (None, 30))

    def test_replicating_problem_case(self):
        # On one flight test, the index matched exactly, but the distance was non-zero
        cycles = np.array([10, 30, 100, 129, 144.0])
        self.assertEqual(cycle_match(100.0, cycles, dist=1), (30, 129))


class TestDatetimeOfIndex(unittest.TestCase):
    def test_index_of_datetime(self):
        start_datetime = datetime.now()
        dt = datetime_of_index(start_datetime, 160, frequency=4)
        self.assertEqual(dt, start_datetime + timedelta(seconds=40))


class TestFillMaskedEdges(unittest.TestCase):
    def test_fill_masked_edges(self):
        array = np.ma.arange(10)
        self.assertEqual(fill_masked_edges(array, 0).tolist(),
                         np.arange(10).tolist())
        array.mask = True
        self.assertEqual(fill_masked_edges(array, 0).tolist(),
                         [None] * 10)
        array.mask = False
        array.mask[:3] = True
        self.assertEqual(fill_masked_edges(array, 0).tolist(),
                         [0, 0, 0, 3, 4, 5, 6, 7, 8, 9])
        array.mask[-2:] = True
        self.assertEqual(fill_masked_edges(array, 0).tolist(),
                         [0, 0, 0, 3, 4, 5, 6, 7, 0, 0])
        array.mask[4:6] = True
        self.assertEqual(fill_masked_edges(array, 0).tolist(),
                         [0, 0, 0, 3, None, None, 6, 7, 0, 0])


class TestFilterSlicesLength(unittest.TestCase):
    def test_filter_slices_length(self):
        slices = [slice(1, 5), slice(4, 6), slice (5, 10)]
        result = filter_slices_length(slices, 0)
        self.assertEqual(result, slices)
        result = filter_slices_length(slices, 4)
        self.assertEqual(result, [slices[0], slices[2]])
        result = filter_slices_length(slices, 5)
        self.assertEqual(result, [slices[2]])


class TestFilterSlicesDuration(unittest.TestCase):
    def test_filter_slices_duration(self):
        slices = [slice(1, 5), slice(4, 6), slice (5, 10)]
        result = filter_slices_duration(slices, 0)
        self.assertEqual(result, slices)
        result = filter_slices_duration(slices, 4)
        self.assertEqual(result, [slices[0], slices[2]])
        result = filter_slices_duration(slices, 5)
        self.assertEqual(result, [slices[2]])
        result = filter_slices_duration(slices, 5, frequency=0.5)
        self.assertEqual(result, [slices[0], slices[2]])


class TestFilterVorIlsFrequencies(unittest.TestCase):
    def test_low_end_ils(self):
        array = np.ma.arange(107.9,109.0,.05)
        result = filter_vor_ils_frequencies(array, 'ILS')
        expected = [1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]
        np.testing.assert_array_equal(result.mask, expected)
        np.testing.assert_array_equal(result.data, array.data)

    def test_high_end_ils(self):
        array = np.ma.arange(111.50,112.35,.05)
        result = filter_vor_ils_frequencies(array, 'ILS')
        expected = [0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1]
        np.testing.assert_array_equal(result.mask, expected)
        np.testing.assert_array_equal(result.data, array.data)

    def test_low_end_vor(self):
        array = np.ma.arange(107.9,109.0,.05)
        result = filter_vor_ils_frequencies(array, 'VOR')
        expected = [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
        np.testing.assert_array_equal(result.mask, expected)
        np.testing.assert_array_equal(result.data, array.data)

    def test_high_end_vor(self):
        array = np.ma.arange(117.50,118.35,.05)
        result = filter_vor_ils_frequencies(array, 'VOR')
        expected = [1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1]
        np.testing.assert_array_equal(result.mask, expected)
        np.testing.assert_array_equal(result.data, array.data)

    def test_mask_propogation(self):
        array = np.ma.array([110.35]*5, mask=[0,1,1,0,0])
        result = filter_vor_ils_frequencies(array, 'ILS')
        expected = [0,1,1,0,0]
        np.testing.assert_array_equal(result.mask, expected)

    def test_unknown_navaid(self):
        array = np.ma.array([108.0])
        self.assertRaises(ValueError, filter_vor_ils_frequencies, array, 'DME')

    def test_real_value(self):
        array = np.ma.array([117.40000000000001]) # Taken from converted data file.
        result = filter_vor_ils_frequencies(array, 'VOR')
        expected = 117.40
        self.assertEqual(result.data[0], expected)


class TestFindEdges(unittest.TestCase):
    # Reminder: find_edges(array, _slice, direction='rising_edges')

    def test_find_edges_basic(self):
        array = np.ma.array([0,0,0,1,1,1])
        result = find_edges(array, slice(0,6))
        expected = [2.5]
        self.assertEqual(expected, result)

    def test_find_edges_slice(self):
        array = np.ma.array([0,0,0,1,1,1])
        result = find_edges(array, slice(0,2))
        expected = []
        self.assertEqual(expected, result)

    def test_find_edges_default_direction(self):
        array = np.ma.array([0,0,0,-1,-1,-1])
        result = find_edges(array, slice(0,6))
        expected = []
        self.assertEqual(expected, result)

    def test_find_edges_falling(self):
        array = np.ma.array([2,2,2,2,0,0])
        result = find_edges(array, slice(0,6), direction='falling_edges')
        expected = [3.5]
        self.assertEqual(expected, result)

    def test_find_edges_all(self):
        array = np.ma.array([1,1,0,0,2,2,0,0,-1-1])
        result = find_edges(array, slice(0,10), direction='all_edges')
        expected = [1.5,3.5,5.5,7.5]
        self.assertEqual(expected, result)

    def test_find_edges_masked_edge(self):
        edges = np.ma.array([1,1,0,0,0,1,1], mask=[0,0,0,0,1,1,0])
        self.assertEqual(find_edges(edges, direction='all_edges'),
                         [1.5, 3.5])

        no_edges = np.ma.array([1,1,0,0,0,1,1], mask=[0,0,1,1,1,0,0])
        self.assertFalse(find_edges(no_edges, direction='all_edges'))

    def test_find_edges_too_small_slice(self):
        # Too small slice should not raise exception.
        edges = np.ma.arange(10)
        self.assertEqual(find_edges(edges, _slice=slice(5, 5)), [])
        self.assertEqual(find_edges(edges, _slice=slice(5, 6)), [])
        self.assertEqual(find_edges(edges, _slice=slice(None, 1)), [])
        self.assertEqual(find_edges(edges, _slice=slice(9, None)), [])


class TestFindEdgesOnStateChange(unittest.TestCase):
    # Reminder...
    # find_edges_on_state_change(state, array, change='entering', phase=None, min_samples=3)
    class Switch(M):
        values_mapping = {0: 'off', 1: 'on'}

    def test_basic(self):
        multi = self.Switch(array=np.ma.array([0,0,1,1,0,0,1,1,0,0]))
        edges = find_edges_on_state_change('on', multi.array)
        expected = [1.5,5.5]
        self.assertEqual(edges, expected)

    def test_leaving(self):
        multi = self.Switch(array=np.ma.array([0,0,1,1,0,0,1,1,0,0]))
        edges = find_edges_on_state_change('off', multi.array, change='leaving')
        expected = [1.5,5.5]
        self.assertEqual(edges, expected)

    def test_entering_and_leaving(self):
        multi = self.Switch(array=np.ma.array([0,0,1,1,0,0,1,1,0,0]))
        edges = find_edges_on_state_change('on', multi.array, change='entering_and_leaving')
        expected = [1.5,3.5,5.5,7.5]
        self.assertEqual(edges, expected)

    def test_phases(self):
        multi = self.Switch(array=np.ma.array([0,0,1,1,0,0,1,1,0,0]))
        phase_list = buildsections('Test', [1,5],[0,5],[1,8],[4,9])
        edges = find_edges_on_state_change('on', multi.array, phase=phase_list)
        expected = [1.5,1.5,1.5,5.5,5.5]
        self.assertEqual(edges, expected)

    def test_misunderstood_edge(self):
        multi = self.Switch(array=np.ma.array([0,0,1,1,0,0,1,1,0,0]))
        self.assertRaises(ValueError, find_edges_on_state_change, 'on', multi.array, change='humbug')

    def test_misunderstood_state(self):
        multi = self.Switch(array=np.ma.array([0,1]))
        self.assertRaises(KeyError, find_edges_on_state_change, 'ha!', multi.array)

    def test_no_state_change(self):
        multi = self.Switch(array=np.ma.array([0,0,0,0,0,0]))
        edges = find_edges_on_state_change('on', multi.array)
        self.assertEqual(edges, [])

    def test_min_samples(self):
        array = np.ma.zeros(30)
        array[4] = 1  # 1 sample
        array[7:9] = 1  # 2 samples
        array[12:15] = 1  # 3 samples
        array[20:] = 1  # 10 samples
        array[25] = 0  # single sample while gear "Down" goes to "Up"
        touchdown = KeyTimeInstance(28, 'Touchdown')
        gear_down = MappedArray(array, values_mapping={0: 'Up', 1: 'Down'})

        gear_down_indexes = find_edges_on_state_change(
            'Down', gear_down, change='entering', phase=[slice(0, touchdown.index)], min_samples=3)
        self.assertEqual(len(gear_down_indexes), 2)
        self.assertEqual(gear_down_indexes, [11.5, 19.5])

        gear_down_indexes = find_edges_on_state_change(
                    'Down', gear_down, change='entering', phase=[slice(0, touchdown.index)], min_samples=2)
        self.assertEqual(gear_down_indexes, [6.5, 11.5, 19.5])

        gear_down_indexes = find_edges_on_state_change(
                    'Down', gear_down, change='entering', phase=[slice(0, touchdown.index)], min_samples=1)
        self.assertEqual(gear_down_indexes, [3.5, 6.5, 11.5, 19.5, 25.5])

    def test_slices_with_offset(self):
        multi = self.Switch(array=np.ma.array([0,0,1,1,0,0,1,1,0,0]))
        slices_ = [slice(2.9, 8.9)]
        edges = find_edges_on_state_change('on', multi.array, phase=slices_)
        expected = [5.5]
        self.assertEqual(edges, expected)

class TestFindTocTod(unittest.TestCase):
    def test_find_tod_with_smoothed_data(self):
        # sample data from Hercules during a low level circuit
        array = np.ma.array(
            [  30.83249375,   46.78510625,   48.2622    ,   48.2622    ,
               48.2622    ,   56.97705313,   57.7156    ,   75.14530625,
               76.6224    ,   85.33725313,   86.0758    ,   86.0758    ,
               86.0758    ,   86.0758    ,  103.50550625,  104.9826    ,
              113.69745313,  131.86570625,  142.05765313,  142.7962    ,
              160.22590625,  161.703     ,  161.703     ,  161.703     ,
              170.41785313,  188.58610625,  190.0632    ,  190.0632    ,
              216.20775938,  218.4234    ,  227.13825313,  245.30650625,
              255.49845313,  256.237     ,  256.237     ,  273.66670625,
              275.1438    ,  283.85865313,  302.02690625,  320.93370625,
              322.4108    ,  331.12565313,  349.29390625,  359.48585313,
              377.65410625,  379.1312    ,  387.84605313,  406.01430625,
              416.20625313,  416.9448    ,  443.08935938,  462.73470625,
              490.35635938,  501.28685313,  545.59966563,  549.2924    ,
              575.43695938,  586.36745313,  604.53570625,  632.15735938,
              651.80270625,  653.2798    ,  661.99465313,  688.87775938,
              708.52310625,  718.71505313,  736.88330625,  747.07525313,
              765.24350625,  775.43545313,  793.60370625,  795.0808    ,
              795.0808    ,  795.0808    ,  795.0808    ,  795.0808    ,
              795.0808    ,  795.0808    ,  803.79565313,  795.81934688,
              803.79565313,  804.5342    ,  804.5342    ,  821.96390625,
              823.441     ,  823.441     ,  823.441     ,  806.01129375,
              795.81934688,  795.0808    ,  777.65109375,  776.174     ,
              776.174     ,  776.174     ,  776.174     ,  776.174     ,
              776.174     ,  776.174     ,  776.174     ,  776.174     ,
              776.174     ,  776.174     ,  776.174     ,  776.174     ,
              776.174     ,  776.174     ,  776.174     ,  776.174     ,
              793.60370625,  795.0808    ,  795.0808    ,  795.0808    ,
              795.0808    ,  795.0808    ,  795.0808    ,  795.0808    ,
              795.0808    ,  795.0808    ,  777.65109375,  750.02944063,
              739.09894688,  720.93069375,  693.30904063,  682.37854688,
              664.21029375,  654.01834688,  635.85009375,  608.22844063,
              588.58309375,  578.39114688,  560.22289375,  558.7458    ,
              550.03094688,  531.86269375,  504.24104063,  493.31054688,
              492.572     ,  475.14229375,  464.95034688,  464.2118    ,
              446.78209375,  436.59014688,  418.42189375,  416.9448    ,
              416.9448    ,  408.22994688,  407.4914    ,  407.4914    ,
              390.06169375,  379.86974688,  379.1312    ,  379.1312    ,
              379.1312    ,  379.1312    ,  379.1312    ,  379.1312    ,
              387.84605313,  388.5846    ,  406.01430625,  407.4914    ,
              416.20625313,  408.22994688,  407.4914    ,  390.06169375,
              379.86974688,  379.1312    ,  379.1312    ,  361.70149375,
              351.50954688,  333.34129375,  331.8642    ,  331.8642    ,
              323.14934688])
        # data is already sliced for the required section
        res = find_toc_tod(array, slice(0, len(array)), 1, mode='tod')
        self.assertEqual(res, 117)
        # with some smoothing (as per Hercules Alt Std Smoothed
        smooth = moving_average(array, window=3, weightings=[0.25,0.5,0.25])
        res = find_toc_tod(smooth, slice(0, len(smooth)), 1, mode='tod')
        self.assertEqual(res, 116) # bit before previous

    def test_wrong_mode(self):
        alt=np.ma.array([0,0,0,5,10,15,20,20,20,16,11,6,1,1,1])
        alt *= 100
        ccd=slice(0,15,None)
        self.assertRaises(ValueError, find_toc_tod, alt, ccd, 1, 'oops')

    def test_basic(self):
        alt=np.ma.array([0,0,0,5,10,15,20,20,20,16,11,6,1,1,1])
        alt *= 100
        ccd=slice(0,15,None)
        self.assertEqual(find_toc_tod(alt, ccd, 0.1, 'toc'), 6)
        self.assertEqual(find_toc_tod(alt, ccd, 0.1, 'tod'), 8)

    def test_extended_toc(self):
        alt=np.ma.array([0]*100 + [5,10,15] + [20]*100 + [16,11,6] + [1]*100)
        alt *= 100
        ccd=slice(0,306,None)
        self.assertEqual(find_toc_tod(alt, ccd, 0.1, 'toc'), 103)

    def test_shortened_toc(self):
        alt=np.ma.array([0,0,0,5,10,15,20])
        alt *= 100
        ccd=slice(0,7,None)
        self.assertEqual(find_toc_tod(alt, ccd, 0.1, 'toc'), 6)

    def test_null_slice(self):
        alt=np.ma.array([0,0,0,5,10,15,20,20,20])
        alt *= 100
        ccd=slice(None,9,None)
        self.assertEqual(find_toc_tod(alt, ccd, 0.1, 'toc'), 6)
        ccd=slice(0, None, None)
        self.assertEqual(find_toc_tod(alt, ccd, 0.1, 'toc'), 6)
        ccd=slice(None, None, None)
        self.assertEqual(find_toc_tod(alt, ccd, 0.1, 'toc'), 6)

    def test_starts_non_zero(self):
        alt=np.ma.array([10,0,0,0,0,0,0])
        alt *= 100
        ccd=slice(0,7,None)
        self.assertEqual(find_toc_tod(alt, ccd, 0.1, 'toc'), 0)

    def test_masked_descent(self):
        alt = np.ma.concatenate((np.ma.arange(10), np.ma.arange(10, 0, -1)))
        alt = alt.astype(float)
        alt = np.tile(alt, 2)
        alt[20:] *= 1.5
        alt *= 100
        alt[11:20] = np.ma.masked
        ccd = slice(0, 20, None)
        self.assertEqual(find_toc_tod(alt, ccd, 0.1, 'tod'), 10)

class TestFirstOrderLag(unittest.TestCase):

    # first_order_lag (in_param, time_constant, hz, gain = 1.0, initial_value = 0.0)

    def test_firstorderlag_time_constant(self):
        # Note: Also tests initialisation.
        array = np.ma.zeros(10)
        # The result of processing this data is...
        result = first_order_lag (array, 2.0, 1.0, initial_value = 1.0)
        # The correct answer is...
        answer = np.ma.array(data=[0.8,0.48,0.288,0.1728,0.10368,0.062208,
                                   0.0373248,0.02239488,0.01343693,0.00806216],
                             mask = False)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_firstorderlag_sample_rate_chage(self):
        # Note: Also tests initialisation.
        array = np.ma.zeros(10)
        # The result of processing this data is...
        result = first_order_lag (array, 2.0, 2.0, initial_value = 1.0)
        # The correct answer is...
        answer = np.ma.array(data=[0.88888889,0.69135802,0.53772291,0.41822893,
                                 0.32528917,0.25300269,0.19677987,0.15305101,
                                 0.11903967,0.09258641], mask = False)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_firstorderlag_gain(self):
        array = np.ma.ones(20)
        result = first_order_lag (array, 1.0, 1.0, gain = 10.0)
        # With a short time constant and more samples, the end result will
        # reach the input level (1.0) multiplied by the gain.
        self.assertAlmostEqual(result.data[-1], 10.0)

    def test_firstorderlag_stability_check(self):
        array = np.ma.ones(4)
        # With a time constant of 1 and a frequency of 4, the simple algorithm
        # becomes too inaccurate to be useful.
        self.assertRaises(ValueError, first_order_lag, array, 0.2, 1.0)

    def test_firstorderlag_mask_retained(self):
        array = np.ma.zeros(5)
        array[3] = np.ma.masked
        result = first_order_lag (array, 1.0, 1.0, initial_value = 1.0)
        ma_test.assert_mask_equivalent(result.mask, [0,0,0,1,0],
                                      err_msg='Masks are not equal')


class TestFirstOrderWashout(unittest.TestCase):

    # first_order_washout (in_param, time_constant, hz, gain = 1.0, initial_value = 0.0)

    def test_firstorderwashout_time_constant(self):
        array = np.ma.ones(10)
        result = first_order_washout (array, 2.0, 1.0, initial_value = 0.0)
        # The correct answer is the same as for the first order lag test, but in
        # this case we are starting from zero and the input data is all 1.0.
        # The washout responds transiently then washes back out to zero,
        # providing the high pass filter that matches the low pass lag filter.
        answer = np.ma.array(data=[0.8,0.48,0.288,0.1728,0.10368,0.062208,
                                   0.0373248,0.02239488,0.01343693,0.00806216],
                             mask = False)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_firstorderwashout_sample_rate_chage(self):
        # Note: Also tests initialisation.
        array = np.ma.zeros(10)
        '''
        OK. Tricky test that needs some explanation.

        The initial value is the steady state input condition prior to the data
        we supply. This filter is a washout (high pass) filter, so the steady
        state output will always be zero.

        The initial condition is set to -1.0, then when the data arrives,
        array[0]=0.0 gives a +1.0 step change to the input and we get a positive
        kick on the output.
        '''
        result = first_order_washout (array, 2.0, 0.5, initial_value = -1.0)
        # The correct answer is...
        answer = np.ma.array(data=[6.66666667e-01,2.22222222e-01,7.40740741e-02,
                                   2.46913580e-02,8.23045267e-03,2.74348422e-03,
                                   9.14494742e-04,3.04831581e-04,1.01610527e-04,
                                   3.38701756e-05], mask = False)
        ma_test.assert_masked_array_approx_equal(result, answer)

    def test_firstorderwashout_gain(self):
        array = np.ma.ones(20)
        result = first_order_washout (array, 1.0, 1.0, gain = 10.0,
                                      initial_value = 0.0)
        # With a short time constant and more samples, the end result will
        # reach the input level (1.0) multiplied by the gain.
        self.assertAlmostEqual(result.data[0], 6.6666667)

    def test_firstorderwashout_stability_check(self):
        array = np.ma.ones(4)
        # With a time constant of 1 and a frequency of 4, the simple algorithm
        # becomes too inaccurate to be useful.
        self.assertRaises(ValueError, first_order_washout, array, 0.2, 1.0)

    def test_firstorderwashout_mask_retained(self):
        array = np.ma.zeros(5)
        array[3] = np.ma.masked
        result = first_order_washout (array, 1.0, 1.0, initial_value = 1.0)
        ma_test.assert_mask_equivalent(result.mask, [0,0,0,1,0],
                                      err_msg='Masks are not equal')


class TestFirstValidParameter(unittest.TestCase):

    def test_first_valid_parameter__none(self):
        p1 = P(name='A', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p2 = P(name='B', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p3 = P(name='C', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        self.assertEqual(first_valid_parameter(p1, p2, p3), None)

    def test_first_valid_parameter__fully_unmasked(self):
        p1 = P(name='A', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p2 = P(name='B', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p3 = P(name='C', array=np.ma.array(data=[0] * 5, mask=[False] * 5))
        self.assertEqual(first_valid_parameter(p1, p2, p3), p3)

    def test_first_valid_parameter__partially_unmasked(self):
        p1 = P(name='A', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p2 = P(name='B', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p3 = P(name='C', array=np.ma.array(data=[0] * 5, mask=[False] * 4 + [True]))
        self.assertEqual(first_valid_parameter(p1, p2, p3), p3)

    def test_first_valid_parameter__multiple(self):
        p1 = P(name='A', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p2 = P(name='B', array=np.ma.array(data=[0] * 5, mask=[True] * 4 + [False]))
        p3 = P(name='C', array=np.ma.array(data=[0] * 5, mask=[False] * 5))
        self.assertEqual(first_valid_parameter(p1, p2, p3), p2)

    def test_first_valid_parameter__valid_samples_within_single_phase(self):
        phases = [slice(1, 5)]
        p1 = P(name='A', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p2 = P(name='B', array=np.ma.array(data=[0] * 5, mask=[True] * 4 + [False]))
        p3 = P(name='C', array=np.ma.array(data=[0] * 5, mask=[False] * 5))
        self.assertEqual(first_valid_parameter(p1, p2, p3, phases=phases), p2)

    def test_first_valid_parameter__no_valid_samples_within_single_phase(self):
        phases = [slice(0, 4)]
        p1 = P(name='A', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p2 = P(name='B', array=np.ma.array(data=[0] * 5, mask=[True] * 4 + [False]))
        p3 = P(name='C', array=np.ma.array(data=[0] * 5, mask=[False] * 5))
        self.assertEqual(first_valid_parameter(p1, p2, p3, phases=phases), p3)

    def test_first_valid_parameter__no_valid_samples_within_single_phase_2(self):
        phases = [slice(1, 4)]
        p1 = P(name='A', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p2 = P(name='B', array=np.ma.array(data=[0] * 5, mask=[True] * 4 + [False]))
        p3 = P(name='C', array=np.ma.array(data=[0] * 5, mask=[False] + [True] * 4))
        self.assertEqual(first_valid_parameter(p1, p2, p3, phases=phases), None)

    def test_first_valid_parameter__valid_samples_within_multiple_phases(self):
        phases = [slice(0, 2), slice(3, 5)]
        p1 = P(name='A', array=np.ma.array(data=[0] * 5, mask=[True] * 5))
        p2 = P(name='B', array=np.ma.array(data=[0] * 5, mask=[True] * 4 + [False]))
        p3 = P(name='C', array=np.ma.array(data=[0] * 5, mask=[False] * 5))
        self.assertEqual(first_valid_parameter(p1, p2, p3, phases=phases), p2)


class TestFirstValidSample(unittest.TestCase):
    def test_first_valid_sample(self):
        result = first_valid_sample(np.ma.array(data=[11,12,13,14],mask=[1,0,1,0]))
        self.assertEqual(result, (1, 12))

    def test_first_valid_sample_all_masked(self):
        result = first_valid_sample(np.ma.array(data=[11,12,13,14],mask=True))
        self.assertEqual(result, (None, None))

    def test_first_valid_sample_offset(self):
        result = first_valid_sample(np.ma.array(data=[11,12,13,14],mask=[1,0,1,0]),2)
        self.assertEqual(result, (3,14))

    def test_first_valid_sample_at_offset(self):
        result = first_valid_sample(np.ma.array(data=[11,12,13,14],mask=[1,0,1,0]),1)
        self.assertEqual(result, (1,12))

    def test_first_valid_sample_overrun(self):
        result = first_valid_sample(np.ma.array(data=[11,12,13,14],mask=[1,0,1,0]),9)
        self.assertEqual(result, (None, None))

    def test_first_valid_sample_underrun(self):
        result = first_valid_sample(np.ma.array(data=[11,12,13,14],mask=[1,0,1,0]),-2)
        self.assertEqual(result, (None, None))


class TestLastValidSample(unittest.TestCase):
    def test_last_valid_sample(self):
        result = last_valid_sample(np.ma.array(data=[11,12,13,14],mask=[1,0,1,0]))
        self.assertEqual(result, (3,14))

    def test_last_valid_sample_all_masked(self):
        result = last_valid_sample(np.ma.array(data=[11,12,13,14],mask=True))
        self.assertEqual(result, (None, None))

    def test_last_valid_sample_offset(self):
        result = last_valid_sample(np.ma.array(data=[11,12,13,14],mask=[1,0,1,0]),-2)
        self.assertEqual(result, (1,12))

    def test_last_valid_sample_at_offset(self):
        result = last_valid_sample(np.ma.array(data=[11,12,13,14],mask=[1,0,1,0]),-3)
        self.assertEqual(result, (1,12))

    def test_last_valid_sample_overrun(self):
        result = last_valid_sample(np.ma.array(data=[11,12,13,14],mask=[1,0,1,0]),9)
        self.assertEqual(result, (None, None))

    def test_last_valid_sample_masked(self):
        result = last_valid_sample(np.ma.array(data=[11,12,13,14],mask=[0,0,0,1]))
        self.assertEqual(result, (2,13))

class TestGroundspeedFromPosition(unittest.TestCase):
    def test_north(self):
        # A degree of latitude is 60 nautical miles.
        # Travelling 1/60th of a degree in 60 seconds is one nautical mile a minute = 60 knots
        lat = np.ma.array(np.linspace(0.0, 1.0/60.0, num=60, endpoint=False))
        lon = np_ma_zeros_like(lat)
        result = groundspeed_from_position(lat, lon, 1.0)
        # Arbitrarily pick one point to check
        self.assertAlmostEqual(result[3], 60.0, places = 0)
        # and make sure the values are all practically the same
        self.assertAlmostEqual(np.ptp(result), 0.0)

    def test_east(self):
        # A degree of longitude at 60 North is 30 nautical miles.
        lon = np.ma.array(np.linspace(0.0, 1.0/60.0, num=60, endpoint=False))
        lat = np_ma_ones_like(lon)*60.0
        result = groundspeed_from_position(lat, lon, 1.0)
        # Arbitrarily pick one point to check
        self.assertAlmostEqual(result[3], 30.0, places = 0)
        # and make sure the values are all practically the same
        self.assertAlmostEqual(np.ptp(result), 0.0)

class TestGroundTrack(unittest.TestCase):
    def test_ground_track_basic(self):
        gspd = np.ma.array([60,60,60,60,60,60,60])
        hdg = np.ma.array([0,0.0,0.0,90,90,90,270])
        lat, lon = ground_track(0.0, 0.0, gspd, hdg, 1.0, 'landing')
        expected_lat = [0.0,0.00027759,0.00055518,0.00069398,0.00069398,0.00069398,0.00069398]
        expected_lon = [0.0,0.0,0.0,0.00013880,0.00041639,0.00069398,0.00069398]
        np.testing.assert_array_almost_equal(expected_lat, lat)
        np.testing.assert_array_almost_equal(expected_lon, lon)
    def test_ground_track_data_errors(self):
        # Mismatched array lengths
        gspd = np.ma.array([60])
        hdg = np.ma.array([0,0])
        self.assertRaises(ValueError, ground_track, 0.0, 0.0, gspd, hdg, 1.0, 'landing')
        # Direction not understood
        gspd = np.ma.array([60,60,60,60,60,60,60])
        hdg = np.ma.array([0,0.0,0.0,90,90,90,270])
        self.assertRaises(ValueError, ground_track, 0.0, 0.0, gspd, hdg, 1.0, 'touchdown')
    def test_ground_track_arrays_too_short(self):
        gspd = np.ma.array([60,60])
        hdg = np.ma.array([0,0])
        lat, lon = ground_track(0.0, 0.0, gspd, hdg, 1.0, 'landing')
        self.assertEqual(lat, None)
        self.assertEqual(lon, None)
    def test_ground_track_heading_continuous(self):
        # Heading continuous means headings can be large.
        gspd = np.ma.array([60,60,60,60,60,60,60])
        hdg = np.ma.array([-720,-360.0,720,90,-270,450,-90])
        lat, lon = ground_track(0.0, 0.0, gspd, hdg, 1.0, 'landing')
        expected_lat = [0.0,0.00027759,0.00055518,0.00069398,0.00069398,0.00069398,0.00069398]
        expected_lon = [0.0,0.0,0.0,0.00013880,0.00041639,0.00069398,0.00069398]
        np.testing.assert_array_almost_equal(expected_lat, lat)
        np.testing.assert_array_almost_equal(expected_lon, lon)
    def test_ground_track_masking(self):
        # Heading continuous means headings can be large.
        gspd = np.ma.array(data=[60,60,60,60,60,60,60],
                           mask=[0,0,1,0,0,0,0])
        hdg = np.ma.array(data=[0,0.0,0.0,90,90,90,270],
                          mask=[0,0,0,0,1,0,0])
        lat, lon = ground_track(0.0, 0.0, gspd, hdg, 1.0, 'landing')
        expected_lat = np.ma.array(data=[0.0,0.00027759,0.00055518,0.00069398,0.00069398,0.00069398,0.00069398],
                                   mask=[0,0,1,0,1,0,0])
        expected_lon = np.ma.array(data=[0.0,0.0,0.0,0.00013880,0.00041639,0.00069398,0.00069398],
                                   mask=[0,0,1,0,1,0,0])
        np.testing.assert_almost_equal(expected_lat.data, lat.data)
        np.testing.assert_almost_equal(expected_lon.data, lon.data)
        np.testing.assert_equal(expected_lat.mask, lat.mask)
        np.testing.assert_equal(expected_lon.mask, lon.mask)
    def test_ground_track_takeoff(self):
        gspd = np.ma.array([60,60,60,60,60,60,60])
        hdg = np.ma.array([0,0.0,0.0,90,90,90,270])
        lat, lon = ground_track(0.0, 0.0, gspd, hdg, 1.0, 'takeoff')
        expected_lat = [-0.00069398,-0.00041639,-0.00013880,0.0,0.0,0.0,0.0]
        expected_lon = [-0.00069398,-0.00069398,-0.00069398,-0.00055518,-0.00027759,0.0,0.0]
        np.testing.assert_array_almost_equal(expected_lat, lat)
        np.testing.assert_array_almost_equal(expected_lon, lon)


class TestGroundTrackPrecise(unittest.TestCase):
    # Precise Positioning version of Ground Track

    def test_ppgt_basic(self):
        lat=[]
        lon=[]
        hdg=[]
        gspd=[]
        duration_test_data_path = os.path.join(test_data_path,
                                               'precise_ground_track_test_data.csv')
        with open(duration_test_data_path, 'rt') as csvfile:
            self.reader = csv.DictReader(csvfile)
            for row in self.reader:
                # Sources to use from the recorded data file...
                # Latitude and Longitude should be prepared values
                # Heading should be Heading True Continuous
                # Groundspeed should be Groundspeed Signed
                lat.append(float(row['Latitude']))
                lon.append(float(row['Longitude']))
                hdg.append(float(row['Heading']))
                gspd.append(float(row['Groundspeed']))
            la, lo = ground_track_precise(np.ma.array(lat),
                                          np.ma.array(lon),
                                          np.ma.array(gspd),
                                          np.ma.array(hdg),
                                          1.0)
        self.assertLess(np.min(la), -.0002)
        self.assertGreater(np.max(la), +0.002)
        self.assertAlmostEqual(np.min(lo), 0.0)
        self.assertGreater(np.max(lo), -.001)

    def test_ppgt_dublin(self):
        lat_data=[]
        lon_data=[]
        hdg_data=[]
        gspd_data=[]
        duration_test_data_path = os.path.join(test_data_path,
                                               'precise_ground_track_test_data_Dublin.csv')
        with open(duration_test_data_path, 'rt') as csvfile:
            self.reader = csv.DictReader(csvfile)
            for row in self.reader:
                lat_data.append(float(row['Latitude']))
                lon_data.append(float(row['Longitude']))
                hdg_data.append(float(row['Heading']))
                gspd_data.append(float(row['Groundspeed']))
            self.lat = np.ma.masked_equal(np.ma.array(lat_data), 0.0)
            self.lon = np.ma.masked_equal(np.ma.array(lon_data), 0.0)
            self.hdg = np.ma.masked_equal(np.ma.array(hdg_data), 0.0)
            self.gspd = np.ma.array(gspd_data)

            la, lo = ground_track_precise(self.lat, self.lon, self.gspd,
                                           self.hdg, 1.0)
        # For this test data the worst case adjustment should be very small.
        # They are measured in knots and degrees, so less than 0.1 is fine.
        self.assertLess(lo[0], 0.1)

    def test_ppgt_svalbard(self):
        # Because Svalbard is a nicer word than Longyearbyen
        # This taxi out includes de-icing and a turn on the runway
        lat_data=[]
        lon_data=[]
        hdg_data=[]
        gspd_data=[]
        duration_test_data_path = os.path.join(test_data_path,
                                               'precise_ground_track_test_data_Svalbard.csv')
        with open(duration_test_data_path, 'rt') as csvfile:
            self.reader = csv.DictReader(csvfile)
            for row in self.reader:
                lat_data.append(float(row['Latitude']))
                lon_data.append(float(row['Longitude']))
                hdg_data.append(float(row['Heading']))
                gspd_data.append(float(row['Groundspeed']))
            self.lat = np.ma.masked_equal(np.ma.array(lat_data), 0.0)
            self.lon = np.ma.masked_equal(np.ma.array(lon_data), 0.0)
            self.hdg = np.ma.masked_equal(np.ma.array(hdg_data), 0.0)
            self.gspd = np.ma.array(gspd_data)

            la, lo = ground_track_precise(self.lat, self.lon, self.gspd,
                                           self.hdg, 1.0)
        self.assertTrue(True)


class TestGtpBlendCurve(unittest.TestCase):

    def test_gtp_blender(self):
        array = np.ma.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        expected = np.ma.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        result = gtp_blend_curve(array)
        ma_test.assert_masked_array_almost_equal(result, expected)


class TestHashArray(unittest.TestCase):
    def test_hash_array(self):
        section = [slice(0,10)]
        self.assertEqual(hash_array(np.ma.arange(10), section, 5),
                         hash_array(np.ma.arange(10), section, 5))
        self.assertNotEqual(hash_array(np.ma.arange(10), section, 5),
                            hash_array(np.ma.arange(1,11), section, 5))
        # Tests that mask contents affect the generated hash.

        ma1 = np.ma.arange(100, 200)
        ma2 = np.ma.array(np.arange(100, 200),
                          mask=np.concatenate((np.zeros(50), np.ones(1), np.zeros(49))))
        section = [slice(0,100)]
        self.assertNotEqual(hash_array(ma1, section, 5), hash_array(ma2, section, 5))
        self.assertEqual(hash_array(ma2, section, 5), hash_array(ma2, section, 5))
        self.assertEqual(hash_array(np.ma.arange(10, dtype=np.float_), [slice(0,10)], 5),
            'c29605eb4e50fbb653a19f1a28c4f0955721419f989f1ffd8cb2ed6f4914bbea')


class TestHeadingDiff(unittest.TestCase):
    def test_heading_diff(self):
        self.assertRaises(AssertionError, heading_diff, -10, 20)
        self.assertRaises(AssertionError, heading_diff, 375, 15)
        self.assertRaises(AssertionError, heading_diff, 20, -20)
        self.assertRaises(AssertionError, heading_diff, 30, 400)
        self.assertRaises(AssertionError, heading_diff, -86.2, 400)
        self.assertRaises(AssertionError, heading_diff, 500, 400)
        self.assertRaises(AssertionError, heading_diff, -86.2, -152)
        self.assertEqual(heading_diff(0, 0), 0)
        self.assertEqual(heading_diff(10, 0), -10)
        self.assertEqual(heading_diff(0, 10), 10)
        self.assertEqual(heading_diff(0, 180), 180)
        self.assertEqual(heading_diff(47.81925103290594,
                                      53.00950000000029), 5.1902489670943481)
        self.assertEqual(heading_diff(1.6407827938370874,
                                      359.8320000000002), -1.8087827938368832)
        self.assertEqual(heading_diff(10, 350), -20)
        self.assertEqual(heading_diff(350, 10), 20)
        self.assertEqual(heading_diff(320, 340), 20)
        self.assertEqual(heading_diff(340, 320), -20)

    def test_heading_diff_array(self):
        headings = np.ma.concatenate((np.arange(345, 360),np.arange(0, 25)))
        np.testing.assert_array_equal(heading_diff(headings, 250), np.ma.arange(-95, -135, -1))
        np.testing.assert_array_equal(heading_diff(headings, 50), np.ma.arange(65, 25, -1))
        np.testing.assert_array_equal(heading_diff(headings, 180),
                                      np.ma.concatenate((np.arange(-165, -180, -1), np.arange(180, 155, -1))))

        np.testing.assert_array_equal(heading_diff(250, headings), np.ma.arange(95, 135))
        np.testing.assert_array_equal(heading_diff(50, headings), np.ma.arange(-65, -25))
        np.testing.assert_array_equal(heading_diff(180, headings),
                                      np.ma.concatenate((np.arange(165, 180), np.arange(-180, -155))))

        np.testing.assert_array_equal(heading_diff(np.ma.arange(95, 135), np.ma.arange(95, 135)), np.ma.array([0]*40))
        # values taken from test_heading_diff test cases above
        array1 = np.ma.array([0,10,0,0,47.81925103290594,1.6407827938370874,10,350,320,340])
        array2 = np.ma.array([0,0,10,180,53.00950000000029,359.8320000000002,350,10,340,320])
        expected = np.ma.array([0,-10,10,180,5.1902489670943481,-1.8087827938368832,-20,20,20,-20])
        np.testing.assert_array_equal(heading_diff(array1, array2), expected)


class TestHysteresis(unittest.TestCase):
    def test_hysteresis(self):
        data = np.ma.array([0,1,2,1,0,-1,5,6,7,0],dtype=float)
        data[4] = np.ma.masked
        result = hysteresis(data,2)
        np.testing.assert_array_equal(result.filled(999),
                                      [0.5,1,1,1,999,0,5,6,6,0.5])

    def test_hysteresis_change_of_threshold(self):
        data = np.ma.array([0,1,2,1,0,-1,5,6,7,0],dtype=float)
        result = hysteresis(data,1)
        np.testing.assert_array_equal(result.data,[0.25,1.,1.5,1.,0.,
                                                   -0.5,5.,6.,6.5,0.25])

    def test_hysteresis_phase_stability(self):
        data = np.ma.array([0,1,2,3,4,5,5,4,3,2,1,0],dtype=float)
        result = hysteresis(data,2)
        # Hysteresis range of 2 = +/- 1.0, so top 1 gets "choppeed off".
        # Slopes remain unaltered, and endpoints are amended by half the
        # hysteresis, being affected only on one pass of this two-pass
        # process.
        np.testing.assert_array_equal(result.data,[0.5,1,2,3,4,4,4,4,3,2,1,0.5])

    def test_hysteresis_with_initial_data_masked(self):
        data = np.ma.array([0,1,2,1,-100000,-1,5,6,7,0],dtype=float)
        data[0] = np.ma.masked
        data[4] = np.ma.masked
        result = hysteresis(data,2)
        np.testing.assert_array_equal(result.filled(999),
                                      [999,1,1,1,999,0,5,6,6,0.5])

    def test_hysteresis_with_zero_or_negative_thresholod(self):
        data = np.ma.array([0,1])
        np.testing.assert_array_equal(data.data, hysteresis(data,0).data)
        self.assertRaises(ValueError, hysteresis, data, -3)

    """
    Hysteresis may need to be speeded up, in which case this test can be
    reinstated.

    def test_time_taken(self):
        from timeit import Timer
        timer = Timer(self.using_large_data)
        time = min(timer.repeat(1, 1))
        print("Time taken %s secs" % time)
        self.assertLess(time, 0.1, msg="Took too long")

    def using_large_data(self):
        data = np.ma.arange(100000)
        data[0] = np.ma.masked
        data[-1000:] = np.ma.masked
        res = hysteresis(data, 10)
        pass
    """


class TestIndexAtValue(unittest.TestCase):

    # Reminder: index_at_value (array, threshold, _slice=slice(None), endpoint='exact')

    def test_modes(self):
        self.assertRaises(AssertionError, index_at_value, np.ma.arange(4), 99, slice(1, None), endpoint='similar')

    def test_index_at_value_basic(self):
        array = np.ma.arange(4)
        self.assertEqual(index_at_value(array, 1.5, slice(0, 3)), 1.5)

    def test_index_at_value_no_slice(self):
        array = np.ma.arange(4)
        self.assertEqual(index_at_value(array, 1.5), 1.5)
        self.assertEqual(index_at_value(array, 1.5, slice(None, None, None)), 1.5)

    def test_index_at_value_backwards(self):
        array = np.ma.arange(8)
        self.assertEqual(index_at_value(array, 3.2, slice(6, 2, -1)), 3.2)

    def test_index_at_value_backwards_with_negative_values_a(self):
        array = np.ma.arange(8)*(-1.0)
        self.assertEqual(index_at_value(array, -3.2, slice(6, 2, -1)), 3.2)

    def test_index_at_value_backwards_with_negative_values_b(self):
        array = np.ma.arange(8)-10
        self.assertEqual(index_at_value(array, -5.2, slice(6, 2, -1)), 4.8)

    def test_index_at_value_right_at_start(self):
        array = np.ma.arange(4)
        self.assertEqual(index_at_value(array, 1.0, slice(1, 3)), 1.0)

    def test_index_at_value_right_at_end(self):
        array = np.ma.arange(4)
        self.assertEqual(index_at_value(array, 3.0, slice(1, 4)), 3.0)

    #==================================================================
    # Indexing from the end of the array results in an array length
    # mismatch. There is a failing test to cover this case which may work
    # with array[:end:-1] construct, but using slices appears insoluble.
    def test_index_at_value_backwards_from_end_minus_one(self):
        array = np.ma.arange(8)
        self.assertEqual(index_at_value(array, 7, slice(8, 3, -1)), 7)
    #==================================================================

    def test_index_at_value_backwards_to_start(self):
        array = np.ma.arange(8)
        self.assertEqual(index_at_value(array, 0, slice(5, 0, -1)), 0)

    def test_index_at_value_backwards_floating_point_end(self):
        array = np.ma.arange(4)
        self.assertEqual(index_at_value(array, 1.0, slice(3.4, 0.5, -1)), 1.0)

    def test_index_at_value_forwards_floating_point_end(self):
        array = np.ma.arange(4)
        self.assertEqual(index_at_value(array, 3.0, slice(0.6, 3.5)), 3.0)

    def test_index_at_value_threshold_not_crossed(self):
        array = np.ma.arange(4)
        self.assertEqual(index_at_value(array, 7.5, slice(0, 3)), None)

    def test_index_at_value_threshold_closing(self):
        array = np.ma.arange(4)
        self.assertEqual(index_at_value(array, 99, slice(1, None), endpoint='closing'), 3)

    def test_index_at_value_threshold_closing_backwards(self):
        array = 6-np.ma.arange(6)
        # array [6,5,4,3,2,1] with slice(None, 4, -1) = [1] which is index 5.
        self.assertEqual(index_at_value(array, 99, slice(None, 4, -1), endpoint='closing'), 5)

    def test_index_at_value_masked(self):
        array = np.ma.arange(4)
        array[1] = np.ma.masked
        self.assertEqual(index_at_value(array, 1.5, slice(0, 3)), None)

    def test_index_at_value_nan(self):
        array = np.ma.arange(4, dtype=np.float64)
        array[1] = np.NaN
        self.assertEqual(index_at_value(array, 1.5, slice(0, 3)), None)

    def test_index_at_value_slice_too_small(self):
        '''
        Returns None when there is only one value in the array since it cannot
        cross a threshold.
        '''
        array = np.ma.arange(50)
        self.assertEqual(index_at_value(array, 25, slice(25,26)), None)

    def test_index_at_value_slice_beyond_top_end_of_data(self):
        '''
        Returns None when there is only one value in the array since it cannot
        cross a threshold.
        '''
        array = np.ma.arange(50)
        self.assertEqual(index_at_value(array, 55, slice(40,60)), None)

    def test_index_at_value_slice_beyond_bottom_end_of_data(self):
        '''
        Returns None when there is only one value in the array since it cannot
        cross a threshold.
        '''
        array = np.ma.arange(50)
        self.assertEqual(index_at_value(array, 55, slice(-20,20)), None)

    def test_index_at_value_divide_by_zero_trap(self):
        '''
        Returns None when there is only one value in the array since it cannot
        cross a threshold.
        '''
        array = np.ma.arange(50)
        array[25:] -= 1
        array[23]=np.ma.masked
        array[26]=np.ma.masked
        self.assertEqual(index_at_value(array, 24, slice(20,30)), 24.5)

    def test_index_at_value_nearest(self):
        array = np.ma.array([0,1,2,1,2,3,2,1])
        self.assertEqual(index_at_value(array, 3.1, slice(1, 8), endpoint='nearest'), 5.0)
        # For comparison...
        self.assertEqual(index_at_value(array, 3.1, slice(1, 8), endpoint='exact'), None)
        self.assertEqual(index_at_value(array, 3.1, slice(1, 8), endpoint='closing'), 2.0)

    def test_index_at_value_closing(self):
        array = np.ma.array([0,1,2,2,2,2,2,2])
        self.assertEqual(index_at_value(array, 3.1, slice(0, 8), endpoint='first_closing'), 2)
        self.assertEqual(index_at_value(array, 3.1, slice(0, 8), endpoint='closing'), 7)

    def test_index_at_value_nearest_backwards(self):
        array = np.ma.array([0,1,2,3,2,1,2,1])
        self.assertEqual(index_at_value(array, 3.1, slice(7, 0, -1), endpoint='nearest'), 3.0)

    def test_index_at_value_all_masked(self):
        array = np.ma.array(data=[1.,2.,3.],mask=[1,1,1])
        self.assertEqual(index_at_value(array,2.5, slice(0,3), endpoint='closing'), None)

    def test_index_at_value_closing_to_start(self):
        array = np.ma.array(data=[6, 5, 4])
        self.assertEqual(index_at_value(array, 10, _slice=slice(3, 0, -1), endpoint='closing'), 0)


class TestIndexClosestValue(unittest.TestCase):
    def test_index_closest_value(self):
        array = np.ma.array([1, 2, 3, 4, 5, 4, 3])
        self.assertEqual(index_closest_value(array, 6, slice(0, 6)), 4)

    def test_index_closest_value_at_start(self):
        array = np.ma.array([6, 2, 3, 4, 5, 4, 3])
        self.assertEqual(index_closest_value(array, 7, slice(0, 6)), 0)

    def test_index_closest_value_at_end(self):
        array = np.ma.array([1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(index_closest_value(array, 99, slice(0, 6)), 5)

    def test_index_closest_value_negative(self):
        array = np.ma.array([3, 2, 1, 4, 5, 6, 7])
        self.assertEqual(index_closest_value(array, -9, slice(0, 6)), 2)

    def test_index_closest_value_sliced(self):
        array = np.ma.array([1, 2, 3, 4, 5, 4, 3])
        self.assertEqual(index_closest_value(array, 6, slice(2, 5)), 4)

    '''

    This test is for a function not currently avaialble, but included for future use.

    def test_index_closest_value_backwards(self):
        array = np.ma.array([3, 2, 1, 4, 5, 6, 7])
        self.assertEqual(index_closest_value(array, -9, slice(5, 1, -1)), 2)
    '''

class TestIndexAtDistance(unittest.TestCase):
    #@unittest.skip('FIXME')
    def test_basic(self):
        distance = 150.0
        index_ref = 0
        latitude_ref = 0.0
        longitude_ref = 10.0
        latitude = np.ma.array([0]*6000)
        longitude = np.ma.arange(10,20,10/6000.0)
        result = index_at_distance(distance, index_ref,
                                   latitude_ref, longitude_ref,
                                   latitude, longitude, 1.0)
        expected = 1499
        self.assertAlmostEqual(result, expected, places=1)

    def test_offset(self):
        distance = 150.0
        index_ref = 30
        latitude = np.ma.array([60]*6000)
        longitude = np.ma.arange(10,30,10/6000.0)
        result = index_at_distance(distance, index_ref,
                                   latitude[index_ref], longitude[index_ref],
                                   latitude, longitude, 1.0)
        # The answer is as basic test, but offset by 30 samples and due to the
        # nonlinearity of the haversin function the answer is slightly different
        # at this point on the earth, hence the additional 0.4.
        expected = 3028.7
        self.assertAlmostEqual(result, expected, places=1)

    def test_backwards(self):
        distance = -150.0
        index_ref = 2500
        latitude = np.ma.array([0]*3000)
        longitude = np.ma.arange(10,20,10/6000.0)
        result = index_at_distance(distance, index_ref,
                                   latitude[index_ref], longitude[index_ref],
                                   latitude, longitude, 1.0)
        # A different nonlinearity applies further north and east:
        expected = 1001
        self.assertAlmostEqual(result, expected, places=1)

    def test_too_far_forwards(self):
        distance = 1000.0
        index_ref = 0
        latitude_ref = 60.0
        longitude_ref = 0.0
        latitude = np.ma.arange(600, 700) / 10.0
        longitude = np.ma.arange(100) / 10.0
        result = index_at_distance(
            distance, index_ref, latitude_ref, longitude_ref, latitude,
            longitude, 1.0)
        self.assertIsNone(result)

    def test_too_far_backwards(self):
        distance = -1000.0
        index_ref = 100
        latitude_ref = 70.0
        longitude_ref = 10.0
        latitude = np.ma.arange(600, 700) / 10.0
        longitude = np.ma.arange(100) / 10.0
        result = index_at_distance(
            distance, index_ref, latitude_ref, longitude_ref, latitude,
            longitude, 1.0)
        self.assertIsNone(result)


class TestIndexOfFirstStart(unittest.TestCase):
    def test_index_start(self):
        b = np.array([0,0,1,1,1,0,0,1,1,1,1,0,0,0])
        pos = index_of_first_start(b, slice(2, -2))
        self.assertEqual(pos, 1.5)
        pos = index_of_first_start(b, min_dur=4)
        self.assertEqual(pos, 6.5)

    def test_index_of_last_start_backwards(self):
        # try scanning backwards
        b = np.array([0,0,1,1,1,0,0,1,1,1,1,0,0,0])
        self.assertRaises(ValueError, index_of_first_start,
                          b, slice(None, None, -1))


class TestIndexOfLastStop(unittest.TestCase):
    def test_index_stop(self):
        b = np.array([0,0,1,1,1,1,0,0,0,0,0,1,1,0])
        pos = index_of_last_stop(b, slice(2, -1))
        self.assertEqual(pos, 12.5)
        pos = index_of_last_stop(b, slice(None, None), min_dur=4)
        self.assertEqual(pos, 5.5)

    def test_index_of_last_stop_backwards(self):
        # try scanning backwards
        self.assertRaises(ValueError, index_of_last_stop,
                          np.array([0,1,0]), slice(None, None, -1))


class TestIndexAtValueOrLevelOff(unittest.TestCase):
    @unittest.skip('See Go Around And Climbout test cases')
    def test_reverse_level_off(self):
        self.assertFalse(True)


class TestIntegValue(unittest.TestCase):
    def test_integ_value(self):
        array = np.ma.arange(10, dtype=float)
        i, v = integ_value(array)
        self.assertEqual(i, 9) #
        self.assertEqual(v, 40.5)

        subslice = slice(3,8)
        res = integ_value(array, subslice)
        self.assertEqual(res.index, 7)
        self.assertEqual(res.value, 23.0)

    def test_integ_value_none(self):
        array = np.ma.array([])
        i, v = integ_value(array)
        self.assertEqual(i, None)
        self.assertEqual(v, None)


class TestInterpolate(unittest.TestCase):
    def test_interpolate_basic(self):
        array = np.ma.array(data=[0,0,2,0,0,3.5,0],
                            mask=[1,1,0,1,1,0,1],
                            dtype=float)
        expected = np.ma.array([2,2,2,2.5,3,3.5,3.5])
        result = interpolate(array)
        np.testing.assert_array_equal(result, expected)

    def test_interpolate_four_parts(self):
        array = np.ma.array(data=[2,0,2,0,2,0,2],
                            mask=[1,0,1,0,1,0,1])
        expected = np.ma.array([0]*7)
        result = interpolate(array)
        np.testing.assert_array_equal(result, expected)

    def test_interpolate_nothing_to_do_none_masked(self):
        array = np.ma.array(data=[0,0,2,0,0,3.5,0],
                            mask=[0,0,0,0,0,0,0],
                            dtype=float)
        result = interpolate(array)
        np.testing.assert_array_equal(result, array)

    def test_interpolate_nothing_to_do_all_masked(self):
        array = np.ma.array(data=[0,0,2,0,0,3.5,0],
                            mask=[1,1,1,1,1,1,1],
                            dtype=float)
        expected = np.ma.array(data=[0,0,0,0,0,0,0],
                            mask=False, dtype=float)
        result = interpolate(array)
        np.testing.assert_array_equal(result, expected)

    def test_interpolate_no_ends(self):
        array = np.ma.array(data=[5,0,0,20],
                            mask=[0,1,1,0],
                            dtype=float)
        expected = np.ma.array([5, 10, 15, 20])
        result = interpolate(array)
        np.testing.assert_array_equal(result, expected)

    def test_interpolate_masked_end(self):
        array = np.ma.array(data=[5,0,0,20],
                            mask=[1,0,1,0],
                            dtype=float)
        # array[0] = np.nan
        expected = np.ma.array([0, 0, 10, 20],
                          mask=[0, 0, 0, 0])
        result = interpolate(array)
        np.testing.assert_array_equal(result, expected)

    def test_interpolate_without_extrapolate(self):
        array = np.ma.array(data=[1, 0, 0, 0, 5, 0, 0, 0, 13, 0, 0, 0, 9, 0, 0, 1],
                            mask=[0, 1, 1, 1, 0, 1, 1, 1,  0, 1, 1, 1, 0, 1, 1, 1],
                            dtype=float)
        #                       1  -interp- 5 -interp-   13  -interp-    9 -untouched-
        expected = np.ma.array([1, 2, 3, 4, 5, 7, 9, 11, 13, 12, 11, 10, 9, 0, 0, 1],
                          mask=[0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0, 0, 1, 1, 1])
        result = interpolate(array, extrapolate=False)
        np.testing.assert_array_equal(result, expected)


class TestInterpolateCoarse(unittest.TestCase):
    def test_interpolate_coarse_1(self):
        lat_c_array = np.ma.array(
            [1000, 1000, 56, 56, 56, 57, 57, 58, 57, 57, 2000, 2000, 2000],
            mask=[True] * 2 + [False] * 8 + [True] * 3, dtype=np.float_)
        interpolated = interpolate_coarse(lat_c_array)
        self.assertEqual([('%.1f' % x) if x else None for x in interpolated.tolist()],
                         [None, None, '56.0', '56.3', '56.7', '57.0', '57.5', '58.0', '57.0', None, None, None, None])

    def test_interpolate_coarse_2(self):
        lon_c_array = np.ma.array(
            [1000, 1000, 60, 60, 60, 61, 61, 61, 62, 63, 2000, 2000, 2000],
            mask=[True] * 2 + [False] * 8 + [True] * 3, dtype=np.float_)
        interpolated = interpolate_coarse(lon_c_array)
        self.assertEqual([('%.1f' % x) if x else None for x in interpolated.tolist()],
                         [None, None, '60.0', '60.3', '60.7', '61.0', '61.3', '61.7', '62.0', '63.0', None, None, None])


class TestIndexOfDatetime(unittest.TestCase):
    def test_index_of_datetime(self):
        start_datetime = datetime.now()
        difference = timedelta(seconds=40)
        index_datetime = start_datetime + difference
        frequency = 4
        index = index_of_datetime(start_datetime, index_datetime, frequency)
        self.assertEqual(index, 160)


class TestInterleave(unittest.TestCase):
    def test_interleave(self):
        param1 = P('A1',np.ma.arange(4, dtype=float),1,0.2)
        param2 = P('A2',np.ma.arange(4, dtype=float)+10,1,0.7)
        result = interleave(param1, param2)
        np.testing.assert_array_equal(result.data,[0,10,1,11,2,12,3,13])
        np.testing.assert_array_equal(result.mask, False)

    def test_merge_alternage_sensors_mask(self):
        param1 = P('A1',np.ma.arange(4, dtype=float),1,0.2)
        param2 = P('A2',np.ma.arange(4, dtype=float)+10,1,0.7)
        param1.array[1] = np.ma.masked
        param2.array[2] = np.ma.masked
        result = interleave(param1, param2)
        np.testing.assert_array_equal(result.data[0:2], [0,10])
        np.testing.assert_array_equal(result.data[3:5], [11,2])
        np.testing.assert_array_equal(result.data[6:], [3,13])
        np.testing.assert_array_equal(result.mask, [False,False,True,
                                                    False,False,True,
                                                    False,False])


"""
class TestInterpolateParams(unittest.TestCase):
    def test_interpolate_params(self):
        param1 = P('A1',np.ma.arange(10),
                   frequency=1,offset=0.2)
        param2 = P('A2',np.ma.arange(0.2, 10, 0.5),
                   frequency=2,offset=0.7)
        param1.array[1] = np.ma.masked
        param2.array[2] = np.ma.masked
        array, freq, off = interpolate_params(param1, param2)
        np.testing.assert_array_equal(array.data[:5],
            [0.0, 0.27222222222222225, 0.49444444444444441, 0.71666666666666656,
             0.93888888888888866])
        np.testing.assert_array_equal(array[-5:],
            [8.9047619047619033, 9.1481481481481435, 9.3833333333333311,
             9.6055555555555525, np.nan])
        array.mask[-1] = True
        self.assertEqual(freq, 3)
        self.assertEqual(off, 3 * param1.offset)
"""


class TestIsIndexWithinSlice(unittest.TestCase):
    def test_is_index_within_slice(self):
        self.assertTrue(is_index_within_slice(1, slice(0,2)))
        self.assertTrue(is_index_within_slice(5, slice(5,7)))
        # Slice is not inclusive of last index.
        self.assertFalse(is_index_within_slice(7, slice(5,7)))
        self.assertTrue(is_index_within_slice(10, slice(8,None)))
        self.assertTrue(is_index_within_slice(10, slice(None, 12)))


class TestIsIndexWithinSlices(unittest.TestCase):
    def test_is_index_within_slices(self):
        self.assertTrue(is_index_within_slices(1, [slice(0,2), slice(5,10)]))
        self.assertTrue(is_index_within_slices(5, [slice(5,7), slice(9,10)]))
        # Slice is not inclusive of last index.
        self.assertFalse(is_index_within_slices(7, [slice(0,2), slice(5,7)]))
        self.assertTrue(is_index_within_slices(10, [slice(8,None)]))
        self.assertTrue(is_index_within_slices(10, [slice(None, 12)]))


class TestILSEstablished(unittest.TestCase):
    def test_basic(self):
        array = np.ma.array([0]*20)
        _slice = slice(1,19)
        hz = 1.0
        result = ils_established(array, _slice, hz)
        self.assertEqual(result, 1)

    def test_back(self):
        array = np.ma.array([0]*20)
        _slice = slice(19,1,-1)
        hz = 1.0
        result = ils_established(array, _slice, hz)
        self.assertEqual(result, 19)

    def test_unsteady(self):
        array = np.ma.array([1.0, 1.0, -1.0, -1.0]*5)
        _slice = slice(1,19)
        hz = 1.0
        result = ils_established(array, _slice, hz)
        self.assertEqual(result, None)

    def test_capture(self):
        array = np.ma.array([1.0, 1.0, -1.0, -1.0]*2 + [0.0]*15)
        _slice = slice(2,21)
        hz = 1.0
        result = ils_established(array, _slice, hz)
        self.assertEqual(result, 9)

    def test_too_short(self):
        array = np.ma.array([1.0, 1.0, -1.0, -1.0]*3 + [0.0]*10)
        _slice = slice(2,21)
        hz = 1.0
        result = ils_established(array, _slice, hz)
        self.assertEqual(result, None)

    def test_very_short(self):
        array = np.ma.array([1.0, 1.0, -1.0, -1.0]*3 + [0.0]*10)
        _slice = slice(20.2,20.8)
        hz = 1.0
        result = ils_established(array, _slice, hz)
        self.assertEqual(result, None)

    def test_immediate(self):
        array = np.ma.array([1.0, 1.0, -1.0, -1.0]*3 + [0.0]*10)
        _slice = slice(2,21)
        hz = 1.0
        result = ils_established(array, _slice, hz, point='immediate')
        self.assertEqual(result, 13)

class TestILSGlideslopeAlign(unittest.TestCase):
    def test_ils_glideslope_align(self):
        runway =  {'end': {'latitude': 60.280151,
                           'longitude': 5.222579},
                   'glideslope': {'latitude': 60.303809,
                                 'longitude': 5.216247},
                   'start': {'latitude': 60.30662494,
                             'longitude': 5.21370074}}
        result = ils_glideslope_align(runway)
        self.assertEqual(result['longitude'],5.214688131165883)
        self.assertEqual(result['latitude'],60.30368065424106)

    def test_ils_glideslope_missing(self):
        runway =  {'end': {'latitude': 60.280151,
                           'longitude': 5.222579},
                'start': {'latitude': 60.30662494,
                          'longitude': 5.21370074}}
        result = ils_glideslope_align(runway)
        self.assertEqual(result,None)


class TestILSLocalizerAlign(unittest.TestCase):
    def test_ils_localizer_align(self):
        runway =  {'end': {'latitude': 60.280151,
                              'longitude': 5.222579},
                      'localizer': {'latitude': 60.2788,
                                    'longitude': 5.22},
                      'start': {'latitude': 60.30662494,
                                'longitude': 5.21370074}}
        result = ils_localizer_align(runway)
        self.assertEqual(result['longitude'],5.2229505710057404)
        self.assertEqual(result['latitude'],60.27904301842346)

    def test_ils_localizer_missing(self):
        runway =  {'end': {'latitude': 60.280151,
                              'longitude': 5.222579},
                      'start': {'latitude': 60.30662494,
                                'longitude': 5.21370074}}
        result = ils_localizer_align(runway)
        self.assertEqual(result['longitude'],5.222579)
        self.assertEqual(result['latitude'],60.280151)


class TestIntegrate (unittest.TestCase):
    # Reminder: integrate(array, frequency, initial_value=0.0, scale=1.0,
    #                     direction="forwards", contiguous=False)):
    def test_integration_basic(self):
        result = integrate([10,0], 1.0)
        np.testing.assert_array_equal(result, [0.0,5.0])

    def test_integration_initial_value(self):
        result = integrate([0,0], 1.0, initial_value=3.0)
        np.testing.assert_array_equal(result, [3.0,3.0])

    def test_integration_frequency(self):
        result = integrate([0,10.0], 2.0)
        np.testing.assert_array_equal(result, [0.0,2.5])

    def test_integration_reverse(self):
        result = integrate([0,10,6], 1.0, initial_value=7, direction='reverse')
        np.testing.assert_array_equal(result, [20.0,15.0,7.0])

    def test_integration_backwards(self):
        result = integrate([0,10,6], 1.0, initial_value=7, direction='BaCKWardS')
        np.testing.assert_array_equal(result, [-6.0,-1.0,7.0])

    def test_integration_scale(self):
        result = integrate([1,1,1], 1.0, scale=10)
        np.testing.assert_array_equal(result, [0.0,10.0,20.0])

    def test_integration_sinewave(self):
        # Double integration of a sine wave reverses phase, so result is just error terms.
        testwave = np.sin(np.arange(0,20.0,0.01))
        step_1 = integrate(testwave, 1.0, scale=0.01, initial_value=-1.0)
        step_2 = integrate(step_1, 1.0, scale=0.01)
        self.assertLess(np.max(np.abs(step_2+testwave)), 0.001)

    def test_contiguous_option(self):
        data = np.ma.array(data = [1,1,1,1,0,2,2,2,3,3,3],
                           mask = [1,0,0,1,1,0,0,0,1,1,0])
        result = integrate(data,1.0, initial_value=5, contiguous=True)
        np.testing.assert_array_equal(result.data, [0,0,0,0,0,0,5,7,7,7,7])
        np.testing.assert_array_equal(result.mask, [1,1,1,1,1,1,0,0,1,1,1])

    def test_contiguous_reversed(self):
        data = np.ma.array(data = [1,1,1,1,0,2,2,2,3,3,3],
                           mask = [1,0,0,1,1,0,0,0,1,1,0])
        result = integrate(data,1.0, initial_value=4,
                           direction='reverse', contiguous=True)
        np.testing.assert_array_equal(result.data, [6,6,6,6,6,6,4,0,0,0,0])
        np.testing.assert_array_equal(result.mask, [1,1,1,1,1,0,0,1,1,1,1])

    def test_integration_masked_tail(self):
        # This test was added to assess the effect of masked values rolling back into the integrand.
        data = np.ma.array(data = [1,1,2,2],
                           mask = [0,0,0,1])
        result = integrate(data,1.0)
        assert_array_equal(np.ma.array(data=[0,1,2.5,2.5], mask=[0,0,0,1]),
                                   result)

    def test_integration_masked_tail_repaired(self):
        # This test was added to assess the effect of masked values rolling back into the integrand.
        data = np.ma.array(data = [1,1,2,99],
                           mask = [0,0,0,1])
        result = integrate(data,1.0, repair=True)
        assert_array_equal(np.ma.array([0,1,2.5,4.5]), result)

    def test_integration_extended(self):
        data = np.ma.array([1,3,5,7.0])
        result = integrate(data, 1.0, extend=True)
        np.testing.assert_array_equal(result.data, [1.0, 3.0, 7.0, 13.0])

    #TODO: test for mask repair


class TestIsSliceWithinSlice(unittest.TestCase):
    def test_is_slice_within_slice(self):
        self.assertTrue(is_slice_within_slice(slice(5,6), slice(4,7)))
        self.assertTrue(is_slice_within_slice(slice(4,6), slice(4,7)))
        self.assertTrue(is_slice_within_slice(slice(4,7), slice(4,7)))
        self.assertFalse(is_slice_within_slice(slice(4,8), slice(4,7)))
        self.assertFalse(is_slice_within_slice(slice(3,7), slice(4,7)))
        self.assertTrue(is_slice_within_slice(slice(None, None),
                                              slice(None, None)))
        self.assertFalse(is_slice_within_slice(slice(None, None),
                                               slice(None, 20)))
        self.assertFalse(is_slice_within_slice(slice(None, 15), slice(4, None)))
        self.assertTrue(is_slice_within_slice(slice(-1000, 15),
                                              slice(None, None)))
        self.assertTrue(is_slice_within_slice(slice(-1000, None),
                                              slice(None, None)))
        self.assertTrue(is_slice_within_slice(slice(None, 15),
                                              slice(None, None)))


class TestMaskInsideSlices(unittest.TestCase):
    # Note: This test used "assert_equal" but this does not test the
    # array correctly. Changed to "ma_test.assert_masked_array_equal" for
    # correct operation.
    def test_mask_inside_slices(self):
        slices = [slice(10, 20), slice(30, 40)]
        array = np.ma.arange(50)
        array.mask = np.array([True] * 5 + [False] * 20 + [True] * 10 + [False] * 10)
        expected_result = np.ma.arange(50)
        expected_result.mask = np.array([True] * 5 + [False] * 5 + [True] *  11 + [False] * 4 + [True] * 16 + [False] * 4 + [True] * 5)
        ma_test.assert_masked_array_equal(mask_inside_slices(array, slices),
                             expected_result)

    def test_mask_inside_slices_floating_ends(self):
        slices = [slice(10.5, 20.5), slice(30.1, 40.9)]
        array = np.ma.arange(50)
        array.mask = np.array([True] * 6 + [False] * 19 + [True] * 11 + [False] * 15)
        expected_result = np.ma.arange(50)
        expected_result.mask = np.array([True] * 6 + [False] * 4+ [True] *  12 + [False] * 3 + [True] * 17 + [False] * 8)
        ma_test.assert_masked_array_equal(mask_inside_slices(array, slices),
                             expected_result)

    def test_mask_inside_slices_null_ends(self):
        slices = [slice(None, 20.5), slice(30.1, None)]
        array = np.ma.arange(50)
        expected_result = np.ma.arange(50)
        expected_result.mask = np.array([True] * 22 + [False] * 8 + [True] * 20)
        ma_test.assert_masked_array_equal(mask_inside_slices(array, slices),
                             expected_result)


class TestMaxContinuousUnmasked(unittest.TestCase):
    def test_max_continuous_unmasked(self):
        data = np.ma.array(np.arange(20),
                           mask=[1,0,1,1,1,0,0,0,0,1,
                                 0,0,0,0,0,0,0,1,1,1])
        _max = max_continuous_unmasked(data)
        # test duration
        self.assertEqual(_max.stop-_max.start, 7)
        self.assertEqual(_max.start, 10)
        self.assertEqual(_max.stop, 17)
        self.assertFalse(np.any(data[_max].mask)) # none should be masked

    def test_max_continuous_unmasked_no_mask(self):
        # no mask
        data = np.ma.array(np.arange(20), mask=False)
        _max = max_continuous_unmasked(data)
        self.assertEqual(_max.stop-_max.start, 20)
        self.assertEqual(_max.start, 0)
        self.assertEqual(_max.stop, 20)

        # all masked
        data = np.ma.array(np.arange(5), mask=[1,1,1,1,1])
        _max = max_continuous_unmasked(data)
        self.assertEqual(_max, None)

        # no data
        data = np.ma.array([])
        _max = max_continuous_unmasked(data, slice(110,120))
        self.assertEqual(_max, None)

    def test_max_continuous_unmasked_with_slice(self):
        data = np.ma.array(np.arange(30), mask=[0,1,0,0,0,1,1,1,1,0,
                                                1,1,1,1,1,1,1,0,0,0,
                                                1,1,1,1,1,0,0,1,1,1,])
        _max = max_continuous_unmasked(data, slice(20,30))
        # test duration
        self.assertEqual(_max.stop-_max.start, 2)
        self.assertEqual(_max.start, 25)
        self.assertEqual(_max.stop, 27)


class TestMaskOutsideSlices(unittest.TestCase):
    # Note: This test used "assert_equal" but this does not test the
    # array correctly. Changed to "ma_test.assert_masked_array_equal" for
    # correct operation.
    def test_mask_outside_slices(self):
        slices = [slice(10, 20), slice(30, 40)]
        array = np.ma.arange(50)
        array.mask = np.array([False] * 10 + [True] * 5 + [False] * 35)
        expected_result = np.ma.arange(50)
        expected_result.mask = np.array([True] * 15 + [False] * 5 + [True] * 10 + [False] * 10 + [True] * 10)
        ma_test.assert_masked_array_equal(mask_outside_slices(array, slices),
                             expected_result)

    def test_mask_outside_slices_floating_ends(self):
        slices = [slice(10.5, 20.5), slice(30.1, 40.9)]
        array = np.ma.arange(50)
        array.mask = np.array([False] * 10 + [True] * 5 + [False] * 35)
        expected_result = np.ma.arange(50)
        expected_result.mask = np.array([True] * 15 + [False] * 5 + [True] * 11 + [False] * 9 + [True] * 10)
        ma_test.assert_masked_array_equal(mask_outside_slices(array, slices),
                                          expected_result)

    def test_mask_outside_slices_null_ends(self):
        slices = [slice(None, 20.5), slice(30.1, None)]
        array = np.ma.arange(50)
        expected_result = np.ma.arange(50)
        expected_result.mask = np.array([False] * 20 + [True] * 11 + [False] * 18 + [True])
        ma_test.assert_masked_array_equal(mask_outside_slices(array, slices),
                                          expected_result)

class TestMatchAltitudes(unittest.TestCase):
    def test_basic_operation(self):
        fine = np.ma.arange(20)+0.0
        coarse = np.ma.arange(20)+4500.0
        expected = fine + 5000.0
        result = match_altitudes(fine, coarse)
        ma_test.assert_masked_array_approx_equal(result, expected)

    def test_masked_fine(self):
        fine = np.ma.array(data=[1,2,3,4,-4995,-4994,-4993,-4992],
                           mask=[0,0,0,1,1,0,0,0])
        coarse = np.ma.array(data=[1,2,3,4,6,5,7,8])+5000.0
        expected = np.ma.array(data=[1,2,3,4,5,6,7,8], mask=fine.mask)+5000.0
        result = match_altitudes(fine, coarse)
        ma_test.assert_masked_array_approx_equal(result, expected)

    def test_masked_coarse(self):
        coarse = np.ma.array(data=[1,2,3,4,5,6,7,8],
                           mask=[0,0,0,1,1,0,0,0])
        fine = np.ma.array(data=[1,2,3,4,5,6,7,8])+5000.0
        expected = np.ma.array(data=fine.data, mask=coarse.mask) - 5000.0
        result = match_altitudes(fine, coarse)
        ma_test.assert_masked_array_approx_equal(result, expected)


class TestMaxValue(unittest.TestCase):
    def test_max_value(self):
        array = np.ma.concatenate([np.arange(50,100), np.arange(100,50,-1)])
        i, v = max_value(array)
        self.assertEqual(i, 50)
        self.assertEqual(v, 100)

        subslice = slice(80, 90)
        res = max_value(array, subslice)
        self.assertEqual(res.index, 80)
        self.assertEqual(res.value, 70)

        neg_step = slice(100,65,-10)
        self.assertRaises(ValueError, max_value, array, neg_step)
        ##self.assertEqual(res, (69, 81)) # you can get this if you use slice.stop!

    def test_max_value_non_integer_slices_no_limits(self):
        array = np.ma.arange(10, 15)
        i, v, = max_value(array)
        self.assertEqual(i, 4)
        self.assertEqual(v, 14)

    def test_max_value_integer_slices(self):
        array = np.ma.arange(10, 20)
        i, v, = max_value(array, slice(2,4))
        self.assertEqual(i, 3)
        self.assertEqual(v, 13)

    def test_max_value_non_integer_upper_edge(self):
        array = np.ma.arange(10, 15)
        i, v, = max_value(array, slice(2,3),None,3.7)
        self.assertEqual(i, 3.7)
        self.assertEqual(v, 13.7)

    def test_max_value_non_integer_lower_edge(self):
        array = np.ma.arange(20, 15, -1)
        i, v, = max_value(array, slice(2,3),1.3,None)
        self.assertEqual(i, 1.3)
        self.assertEqual(v, 18.7)

    def test_max_value_slice_mismatch(self):
        array = np.ma.arange(10, 15)
        i, v, = max_value(array, slice(100,101))
        self.assertEqual(i, None)
        self.assertEqual(v, None)

    def test_max_value_no_edge(self):
        array = np.ma.array(data=[2,3,4,8,9],mask=[0,0,0,1,1])
        i, v = max_value(array, slice(0,3),None,3.5)
        self.assertEqual(i, 2)
        self.assertEqual(v, 4) # Important that end case is ignored.

    def test_max_value_all_masked(self):
        array = np.ma.array(data=[0,1,2], mask=[1,1,1])
        i, v = max_value(array)
        self.assertEqual(i, None)
        self.assertEqual(v, None)

    def test_max_value_mapped_array_return_state(self):
        mapping = {0: '0', 1: 'SF1', 2: 'SF2', 3: 'SF3'}
        array = MappedArray([0, 1, 2, 3, 2, 1, 0], values_mapping=mapping)
        i, v = max_value(array)
        self.assertEqual(i, 3)
        self.assertEqual(v, 'SF3')


class TestAverageValue(unittest.TestCase):
    def test_average_value(self):
        array = np.ma.arange(10)
        self.assertEqual(average_value(array), Value(5, 4.5))

        array = np.ma.arange(30)
        self.assertEqual(average_value(array), Value(15, 14.5))


class TestMedianValue(unittest.TestCase):
    def test_median_value(self):
        array = np.ma.concatenate([np.arange(6), np.arange(4)])
        self.assertEqual(median_value(array), Value(5, 2))

        array = np.ma.arange(30)
        self.assertEqual(median_value(array), Value(15, 14.5))


class TestMaxAbsValue(unittest.TestCase):
    def test_max_abs_value(self):
        array = np.ma.concatenate([np.arange(-20,30), np.arange(10,-41, -1), np.arange(10)])
        self.assertEqual(max_abs_value(array), (100, -40))
        array = array*-1.0
        self.assertEqual(max_abs_value(array), (100, 40))

    def test_max_abs_slice_decimal(self):
        "where slice start is not a whole number, ensure maxima is returned"
        d = np.ma.array(data = [
            -0.000, -0.000, -0.352, -0.000, -0.000,  0.351,  0.351,
            -0.000, -0.000, -0.352, -0.352, -0.352, -0.000, -0.000,
            -0.352, -0.352, -0.352, -0.001, -0.001, -0.001, -0.001,
             0.351,  0.351,  0.351, -0.001, -0.001, -0.352, -0.352,
            -0.001, -0.352, -0.704, -1.056, -1.055, -0.704, -0.704])
        s = slice(3.482, 1241.611)
        res = max_abs_value(d, s)
        self.assertEqual(res.value, -1.056)
        self.assertEqual(res.index, 31)

    def test_max_abs_value_mapped_array_return_state(self):
        mapping = {0: '0', 1: 'SF1', 2: 'SF2', 3: 'SF3', -4: 'Special'}
        array = MappedArray([0, 1, 2, -4, 2, 1, 0], values_mapping=mapping)
        i, v = max_abs_value(array)
        self.assertEqual(i, 3)
        self.assertEqual(v, 'Special')


class TestMergeMasks(unittest.TestCase):
    def test_merge_masks_default(self):
        assert_equal(
            merge_masks([np.array([False, False, False, True, True]),
                         np.array([False, False, True, True, False])]),
            np.array([False, False, True, True, True]))

    def test_merge_masks_two_unmasked(self):
        assert_equal(
            merge_masks([np.array([False, False, False, True, True]),
                         np.array([False, False, True, True, False])],
                        min_unmasked=2),
            np.array([False, False, False, True, False]))


class TestMergeSources(unittest.TestCase):
    def test_merge_sources_basic(self):
        p1 = np.ma.zeros(4)
        p2 = np.ma.array([1,2,3,4])
        result = merge_sources(p1, p2)
        expected = np.ma.array([0,1,0,2,0,3,0,4])
        assert_array_equal(expected, result)


class TestMergeTwoParameters(unittest.TestCase):
    def test_merge_two_parameters_offset_ordered_forward(self):
        p1 = P(array=np.ma.zeros(4), frequency=1, offset=0.0)
        p2 = P(array=np.ma.arange(1, 5), frequency=1, offset=0.4)
        arr, freq, off = merge_two_parameters(p1, p2)
        self.assertEqual(arr[1], 1.0) # Differs from blend function here.
        self.assertEqual(freq, 2)
        self.assertAlmostEqual(off, -0.05)

    def test_merge_two_parameters_offset_ordered_backward(self):
        p1 = P(array=[5,10,7,8], frequency=2, offset=0.3)
        p2 = P(array=np.ma.arange(1, 5), frequency=2, offset=0.1)
        arr, freq, off = merge_two_parameters(p1, p2)
        self.assertEqual(arr[3], 10.0)
        self.assertEqual(freq, 4)
        self.assertAlmostEqual(off, 0.075)

    def test_merge_two_parameters_assertion_error(self):
        p1 = P(array=np.ma.zeros(4), frequency=1, offset=0.0)
        p2 = P(array=np.ma.ones(4), frequency=2, offset=0.2)
        self.assertRaises(AssertionError, merge_two_parameters, p1, p2)

    def test_merge_two_parameters_array_mismatch_error(self):
        p1 = P(array=np.ma.zeros(4), frequency=1, offset=0.0)
        p2 = P(array=np.ma.ones(3), frequency=1, offset=0.2)
        self.assertRaises(AssertionError, merge_two_parameters, p1, p2)

    def test_merge_two_parameters_arrays_biassed_error(self):
        p1 = P(array=np.ma.zeros(4), name='One', frequency=1, offset=0.9)
        p2 = P(array=np.ma.ones(4), name='Two', frequency=1, offset=0.9)
        self.assertRaises(ValueError, merge_two_parameters, p1, p2)


class TestMinValue(unittest.TestCase):
    def test_min_value(self):
        array = np.ma.concatenate([np.arange(50,100), np.arange(100,50,-1)])
        i, v = min_value(array)
        self.assertEqual(i, 0)
        self.assertEqual(v, 50)

        subslice = slice(80, 90)
        res = min_value(array, subslice)
        self.assertEqual(res.index, 89)
        self.assertEqual(res.value, 61)

        neg_step = slice(100,65,-10)
        self.assertRaises(ValueError, min_value, array, neg_step)

    def test_min_value_non_integer_slices_no_limits(self):
        array = 10-np.ma.arange(5)
        i, v, = min_value(array)
        self.assertEqual(i, 4)
        self.assertEqual(v, 6)

    def test_min_value_integer_slices(self):
        array = 10-np.ma.arange(10)
        i, v, = min_value(array, slice(2,4))
        self.assertEqual(i, 3)
        self.assertEqual(v, 7)

    def test_min_value_non_integer_upper_edge(self):
        array = 10-np.ma.arange(5)
        i, v, = min_value(array, slice(2,3),None,3.7)
        self.assertEqual(i, 3.7)
        self.assertEqual(v, 6.3)

    def test_min_value_non_integer_lower_edge(self):
        array = np.ma.arange(5)+5
        i, v, = min_value(array, slice(2,3),1.3,None)
        self.assertEqual(i, 1.3)
        self.assertEqual(v, 6.3)

    def test_min_value_slice_mismatch(self):
        array = 10-np.ma.arange(5)
        i, v, = min_value(array, slice(100,101))
        self.assertEqual(i, None)
        self.assertEqual(v, None)

    @unittest.skip("Min value on multi-states does not seem to be a valid test?")
    def test_min_value_mapped_array_return_state(self):
        mapping = {0: '0', 1: 'SF1', 2: 'SF2', 3: 'SF3'}
        array = MappedArray([3, 2, 1, 0, 1, 2, 3], values_mapping=mapping)
        i, v = min_value(array)
        self.assertEqual(i, 3)
        self.assertEqual(v, '0')


class TestMinimumUnmasked(unittest.TestCase):
    def test_min_unmasked_basic(self):
        a1= np.ma.array(data=[1.1,2.1,3.1,4.1],
                        mask=[True,True,False,False])
        a2= np.ma.array(data=[4.2,3.2,2.2,1.2],
                        mask=[True,False,True,False])
        expected= np.ma.array(data=[99,3.2,3.1,1.2],
                              mask=[True,False,False,False],
                              dtype=float)
        result = minimum_unmasked(a1,a2)
        np.testing.assert_array_equal(expected, result)


class TestBlendParameters(unittest.TestCase):

    # Reminder: blend_parameters(params, offset, frequency):

    # The setup and complex example provide the bulk of the evidence of
    # satisfactory operation here. The other test cases are derived from the
    # old BlendTwoParameters test to illustrate that we have replaced an
    # existing technique with an equivalent.
    def setUp(self):
        # Here is genuine sample data from a 737 NG. On this type the higher
        # sample rate parameter is delayed by about 900mS making the samples
        # almost coincident with the other sources.
        alt_a = [-4,-4,-4,-4,-4,-3,1,5,22,70,158,268,373,467,557,657,786,862,978,1099,
                 1195,1317,1456,1508,1554,1597,1639,1680,1715,1751,1759,1723,1734,1839,
                 1991,2090,2183,2269,2370,2473,2718,0]
        p_alt_a=P('Altitude Radio (A)', array=alt_a, frequency=0.5, offset=0.1)
        p_alt_a.array[-1]=np.ma.masked

        alt_b = [-4,-4,-3,4,65,260,461,652,851,1089,1304,1504,1594,1677,
                 1749,1719,1815,2080,2260,2470,2830]
        p_alt_b=P('Altitude Radio (B)', array=alt_b, frequency=0.25, offset=1.1)

        alt_c = [-5,-5,1,20,152,367,552,786,968,1199,1454,1552,1639,1714,
                 1767,1731,1995,2190,2390,2720,2880]
        p_alt_c=P('Altitude Radio (C)', array=alt_c, frequency=0.25, offset=3.1)
        p_alt_c.array.data[10:14]=0.0
        p_alt_c.array[10:14]=np.ma.masked

        self.params = (p_alt_a, p_alt_b, p_alt_c)

    def test_blend_parameters_errors(self):
        p1 = P(array=[0,0,0,1.0,2], frequency=1, offset=0.9)
        p2 = P(array=[1,2,3,4.0,5], frequency=1, offset=0.4)
        self.assertRaises(AssertionError, blend_parameters, (p1, p2), frequency=0.0)
        self.assertRaises(AssertionError, blend_parameters, (None, None))
        self.assertRaises(AssertionError, blend_parameters, (p1, p2), mode='silly')
        self.assertRaises(ValueError, blend_parameters, (p1, p2), mode='cubic', validity='silly')

    def test_cubic_validities(self):
        p1 = P('Altitude Radio (A)', array=np.ma.array([1.0]*10))
        p2 = P('Altitude Radio (B)', array=np.ma.array([2.0]*10, mask=True))
        p3 = P('Altitude Radio (C)', array=np.ma.array([4.0]*10, mask=True))
        result = blend_parameters((p1, p2, p3), offset=0.0, frequency=1.0, mode='cubic', validity='any_one')
        self.assertAlmostEqual(result[5], 1.0)
        result = blend_parameters((p1, p2, p3), offset=0.0, frequency=1.0, mode='cubic', validity='all_but_one')
        self.assertEqual(result[5].mask, True)
        result = blend_parameters((p1, p2, p3), offset=0.0, frequency=1.0, mode='cubic', validity='all')
        self.assertEqual(result[5].mask, True)

        p2 = P('Altitude Radio (B)', array=np.ma.array([2.0]*10))
        result = blend_parameters((p1, p2, p3), offset=0.0, frequency=1.0, mode='cubic', validity='any_one')
        self.assertAlmostEqual(result[5], 1.5)
        result = blend_parameters((p1, p2, p3), offset=0.0, frequency=1.0, mode='cubic', validity='all_but_one')
        self.assertAlmostEqual(result[5], 1.5)
        result = blend_parameters((p1, p2, p3), offset=0.0, frequency=1.0, mode='cubic', validity='all')
        self.assertEqual(result[5].mask, True)

        p3 = P('Altitude Radio (C)', array=np.ma.array([3.0]*10))
        result = blend_parameters((p1, p2, p3), offset=0.0, frequency=1.0, mode='cubic', validity='any_one')
        self.assertAlmostEqual(result[5], 2.0)
        result = blend_parameters((p1, p2, p3), offset=0.0, frequency=1.0, mode='cubic', validity='all_but_one')
        self.assertAlmostEqual(result[5], 2.0)
        result = blend_parameters((p1, p2, p3), offset=0.0, frequency=1.0, mode='cubic', validity='all')
        self.assertAlmostEqual(result[5], 2.0)

    def test_blend_linear_params_complex_example(self):
        result = blend_parameters(self.params, offset=0.0, frequency=2.0)
        self.assertAlmostEqual(result[30], 19.45, places=2)
        self.assertAlmostEqual(result[80], 1208, places=0)

    def test_blend_linear_two_parameters_p2_before_p1_equal_spacing(self):
        p1 = P(array=[0,0,0,1.0,2], frequency=1, offset=0.9)
        p2 = P(array=[1,2,3,4.0,5], frequency=1, offset=0.4)
        result = blend_parameters((p1, p2))
        self.assertGreater(result[1], 0.75)
        self.assertLess(result[1], 0.85)

    def test_blend_linear_two_parameters_offset_p2_before_p1_unequal_spacing(self):
        p1 = P(array=[5,10,7,8.0,8], frequency=1, offset=0.1)
        p2 = P(array=[1,2,3,4.0,5], frequency=1, offset=0.0)
        result = blend_parameters((p1, p2))
        self.assertGreater(result[2], 4.5)
        self.assertLess(result[2], 5.5)

    def test_blend_linear_two_parameters_offset_order_back_low_freq(self):
        p1 = P(array=[5,10,7,8.0,8], frequency=0.25, offset=0.1)
        p2 = P(array=[1,2,3,4.0,5], frequency=0.25, offset=0.0)
        result = blend_parameters((p1, p2))
        self.assertGreater(result[2], 4.4)
        self.assertLess(result[2], 5.5)

    def test_blend_linear_two_parameters_param_one_rubbish(self):
        p1 = P(array=[5,10,7,8,9], frequency=1, offset=0.1, name='First')
        p2 = P(array=[1,2,3,4,5], frequency=1, offset=0.0, name='Second')
        p1.array.mask = True
        result = blend_parameters((p1, None, p2))  # random None parameter too
        self.assertAlmostEqual(result[2], 3)

    def test_blend_linear_two_parameters_lower_op_freq(self):
        p1 = P(array=[5,10,7,8,5,7,4,2], frequency=2, offset=0.1, name='First')
        p2 = P(array=[1,2,3,4,5,4,3,2], frequency=2, offset=0.0, name='Second')
        p1.array[5:] = np.ma.masked
        result = blend_parameters((p1, p2))
        self.assertAlmostEqual(len(result), 4)

    def test_blend_linear_check_weighting(self):
        p1 = P(array=[1.0]*5, frequency = 1.0, offset = 0.2, name='First')
        p2 = P(array=[4.0]*10, frequency = 2.0, offset = 0.05, name='Second')
        result = blend_parameters((p1, p2))
        self.assertEqual(result.mask[0], True)
        self.assertEqual(result[4], 3)

    def test_blend_cubic_complex_example(self):
        result = blend_parameters(self.params, offset=0.0, frequency=2.0, mode='cubic')
        self.assertAlmostEqual(result[30], 14.225, places=2)
        self.assertAlmostEqual(result[80], 1215, places=0)

    def test_blend_cubic_two_parameters_p2_before_p1_equal_spacing(self):
        p1 = P(array=[0,0,0,1.0,2], frequency=1, offset=0.9)
        p2 = P(array=[1,2,3,4.0,5], frequency=1, offset=0.4)
        result = blend_parameters((p1, p2), mode='cubic')
        self.assertGreater(result[1], 0.75)
        self.assertLess(result[1], 0.85)

    def test_blend_cubic_two_parameters_offset_p2_before_p1_unequal_spacing(self):
        p1 = P(array=[5,10,7,8.0,8], frequency=1, offset=0.1)
        p2 = P(array=[1,2,3,4.0,5], frequency=1, offset=0.0)
        result = blend_parameters((p1, p2), mode='cubic')
        self.assertGreater(result[2], 4.5)
        self.assertLess(result[2], 5.5)

    def test_blend_cubic_two_parameters_offset_order_back_low_freq(self):
        p1 = P(array=[5,10,7,8.0,8], frequency=0.25, offset=0.1)
        p2 = P(array=[1,2,3,4.0,5], frequency=0.25, offset=0.0)
        result = blend_parameters((p1, p2), mode='cubic')
        self.assertGreater(result[2], 4.5)
        self.assertLess(result[2], 5.5)

    def test_blend_cubic_two_parameters_param_one_rubbish(self):
        p1 = P(array=[5,10,7,8,9], frequency=1, offset=0.1, name='First')
        p2 = P(array=[1,2,3,4,5], frequency=1, offset=0.0, name='Second')
        p1.array.mask = True
        result = blend_parameters((p1, None, p2), mode='cubic', validity='all_but_one')  # random None parameter too
        self.assertAlmostEqual(result[2], 3)

    def test_blend_cubic_two_parameters_lower_op_freq(self):
        p1 = P(array=[5,10,7,8,5,7,4,2], frequency=2, offset=0.1, name='First')
        p2 = P(array=[1,2,3,4,5,4,3,2], frequency=2, offset=0.0, name='Second')
        p1.array[5:] = np.ma.masked
        result = blend_parameters((p1, p2), mode='cubic')
        self.assertAlmostEqual(len(result), 4)

    def test_blend_parameters_outside_tolerance(self):
        p1 = P(array=np.ma.arange(20), frequency=2, name='First')
        p2 = P(array=np.ma.zeros(10), frequency=1, name='Second')
        result = blend_parameters((p1, p2), tolerance=9, frequency=1.0)
        ma_test.assert_masked_array_equal(result, np.ma.array(data=np.arange(10), mask=[0]*5+[1]*5) * 4/3.0)
        result = blend_parameters((p1, p2), tolerance=9, frequency=1.0, mode='cubic')
        ma_test.assert_masked_array_almost_equal(result, np.ma.array(data=np.arange(10), mask=[1]+[0]*4+[1]*5) * 4 / 3.0)

    def test_blend_cubic_short_valid_samples(self):
        # When the blended result has a lower frequency than the input parameters,
        # very short isolated valid data should be disregarded if not able to produce
        # one complete sample for the lower frequency.
        p1 = P(array=np.repeat(np.ma.array([0, 1, 2, 3, 4, 5]), 10),
               frequency=2, offset=0.1, name='First')
        p2 = P(array=np.repeat(np.ma.array([0, 1, 2, 3, 4, 5]), 10),
               frequency=2, offset=0.3, name='Second')
        # Make short valid samples for both param 1 and 2
        p1.array[:15] = np.ma.masked
        p1.array[32:47] = np.ma.masked
        p2.array[16:31] = np.ma.masked
        result = blend_parameters((p1, p2), mode='cubic', validity='all')
        self.assertTrue(np.all(result.mask[:25]))


class TestBlendParametersWeighting(unittest.TestCase):
    def test_weighting(self):
        array=np.ma.array(data=[0,0,0,0,0,0,0,0,0,0,0,0,0],
                          mask=[1,1,0,1,0,0,0,1,1,1,0,0,0])
        result = blend_parameters_weighting(array, 1.0)
        expected = [0.0,0.0,0.05,0.0,0.05,1.0,0.05,0.0,0.0,0.0,0.05,1.0,1.0]
        assert_equal(result.data, expected)

    def test_weighting_increased_freq(self):
        array=np.ma.array(data=[0,0,0,0,0,0],
                          mask=[1,1,1,0,0,0])
        result = blend_parameters_weighting(array, 2.0)
        expected = np.ma.masked_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.3, 0.5, 0.5, 0.5, 0.5])
        ma_test.assert_masked_array_almost_equal(result, expected)

    def test_weighting_decreased_freq(self):
        array=np.ma.array(data=[0,0,0,0,0,0],
                          mask=[1,1,1,0,0,0])
        result = blend_parameters_weighting(array, 0.5)
        expected = np.ma.array([0.0, 0.05, 2.0])
        ma_test.assert_masked_array_almost_equal(result, expected)

    def test_weighting_decreased_freq_odd_samples(self):
        array=np.ma.array(data=[0,0,0,0,0,0,0],
                          mask=[1,1,1,0,0,0,0])
        result = blend_parameters_weighting(array, 0.5)
        expected = np.ma.array([0.0, 0.05, 2.0])
        # When first run, the length of this array was 4, not 3 (!)
        ma_test.assert_masked_array_almost_equal(result, expected)


class TestBlendTwoParameters(unittest.TestCase):
    def test_blend_two_parameters_p2_before_p1_equal_spacing(self):
        p1 = P(array=[0,0,0,1.0], frequency=1, offset=0.9)
        p2 = P(array=[1,2,3,4.0], frequency=1, offset=0.4)
        arr, freq, off = blend_two_parameters(p1, p2)
        self.assertEqual(arr[1], 0.75)
        self.assertEqual(freq, 2)
        self.assertAlmostEqual(off, 0.4)

    def test_blend_two_parameters_offset_p2_before_p1_unequal_spacing(self):
        p1 = P(array=[5,10,7,8.0], frequency=2, offset=0.1)
        p2 = P(array=[1,2,3,4.0], frequency=2, offset=0.0)
        arr, freq, off = blend_two_parameters(p1, p2)
        self.assertEqual(arr[2], 6)
        self.assertEqual(freq, 4)
        self.assertEqual(off, 0.05)

    def test_blend_two_parameters_large_offset(self):
        p1 = P(array=[5,10,7,8.0], frequency=1, offset=0.7)
        p2 = P(array=[1,2,3,4.0], frequency=1, offset=0.9)
        arr, freq, off = blend_two_parameters(p1, p2)
        self.assertEqual(arr[2], 5.5)
        self.assertEqual(freq, 2)
        self.assertAlmostEqual(off, 0.3)

    def test_blend_two_parameters_offset_order_back_low_freq(self):
        p1 = P(array=[5,10,7,8.0], frequency=0.25, offset=0.1)
        p2 = P(array=[1,2,3,4.0], frequency=0.25, offset=0.0)
        arr, freq, off = blend_two_parameters(p1, p2)
        self.assertEqual(arr[2], 6)
        self.assertEqual(freq, 0.5)
        self.assertEqual(off, 0.05)

    def test_blend_two_parameters_assertion_error(self):
        p1 = P(array=np.zeros(4), frequency=1, offset=0.0)
        p2 = P(array=np.ones(4), frequency=2, offset=0.2)
        self.assertRaises(AssertionError, blend_two_parameters, p1, p2)

    def test_blend_two_parameters_array_mismatch_error(self):
        p1 = P(array=np.zeros(4), frequency=1, offset=0.0)
        p2 = P(array=np.ones(3), frequency=2, offset=0.2)
        self.assertRaises(AssertionError, blend_two_parameters, p1, p2)

    def test_blend_two_parameters_param_one_rubbish(self):
        p1 = P(array=[5,10,7,8], frequency=2, offset=0.1, name='First')
        p2 = P(array=[1,2,3,4], frequency=2, offset=0.0, name='Second')
        p1.array.mask = True
        arr, freq, off = blend_two_parameters(p1, p2)
        self.assertEqual(arr[4], 3)
        self.assertEqual(freq, 4)
        self.assertEqual(off, 0.0)

    def test_blend_two_parameters_param_two_rubbish(self):
        p1 = P(array=[5,10,7,8], frequency=2, offset=0.4, name='First')
        p2 = P(array=[1,2,3,4], frequency=2, offset=0.0, name='Second')
        p2.array.mask = True
        arr, freq, off = blend_two_parameters(p1, p2)
        self.assertEqual(arr[4], 8.5)
        self.assertEqual(freq, 4)
        self.assertAlmostEqual(off, 0.15)

    def test_blend_two_parameters_rejecting_no_change_data(self):
        p1 = P(array=[4.0]*4, frequency=1, offset=0.9)
        p2 = P(array=[1,2,3,4.0], frequency=1, offset=0.4)
        arr, freq, off = blend_two_parameters(p1, p2)
        self.assertEqual(arr[2], 2)
        self.assertEqual(freq, 2)
        self.assertAlmostEqual(off, 0.4)

    def test_blend_two_parameters_integer_values(self):
        # Aircraft with integer conversion of radio altimeter data failed to
        # correctly merge arrays due to Python integer division. This test
        # should be superfluous as we now force floating point conversion,
        # but is retained for completeness.
        p1 = P(array=[2,3,4], frequency=1, offset=0.9)
        p2 = P(array=[1,2,3], frequency=1, offset=0.4)
        arr, freq, off = blend_two_parameters(p1, p2)
        self.assertEqual(arr[0], 1.5)
        self.assertEqual(freq, 2.0)
        self.assertAlmostEqual(off, 0.4)

    def test_blend_two_parameters_for_ils_localizer(self):
        # Localizer feature allows rejection of one signal
        p1 = P(array=[1.5, 1.4, 1.3, 1.2, 1.1, 1.0], frequency=0.5, offset = 0.5)
        p2 = P(array=[0.2, 0.3, 0.4, 1.15, 1.05, 0.95], frequency=0.5, offset = 1.5)
        arr, freq, off = blend_two_parameters(p1, p2, mode='localizer')
        expected = np.ma.array(np.linspace(1.5, 0.95, num=12))
        # See blend_equispaced_sensors for endpoint handling...
        expected[-1] = 0.975
        ma_test.assert_masked_array_almost_equal(arr, expected)

    def test_blend_two_parameters_for_ils_localizer_reversed(self):
        # Localizer feature allows rejection of one signal
        p2 = P(array=[1.5, 1.4, 1.3, 1.2, 1.1, 1.0], frequency=0.5, offset = 0.5)
        p1 = P(array=[0.2, 0.3, 0.4, 1.15, 1.05, 0.95], frequency=0.5, offset = 1.5)
        arr, freq, off = blend_two_parameters(p1, p2, mode='localizer')
        expected = np.ma.array(np.linspace(1.5, 0.95, num=12))
        # See blend_equispaced_sensors for endpoint handling...
        expected[-1] = 0.975
        ma_test.assert_masked_array_almost_equal(arr, expected)

    def test_blend_two_parameters_for_ils_localizer_inverted(self):
        # Simply to make sure the ABS functions are right  :)
        p1 = P(array=[1.5, 1.4, 1.3, 1.2, 1.1, 1.0], frequency=0.5, offset = 0.5)
        p1.array *= -1
        p2 = P(array=[0.2, 0.3, 0.4, 1.15, 1.05, 0.95], frequency=0.5, offset = 1.5)
        p2.array *= -1
        arr, freq, off = blend_two_parameters(p1, p2, mode='localizer')
        expected = np.ma.array(np.linspace(-1.5, -0.95, num=12))
        # See blend_equispaced_sensors for endpoint handling...
        expected[-1] = -0.975
        ma_test.assert_masked_array_almost_equal(arr, expected)

    def test_blend_two_parameters_for_ils_glideslope(self):
        p1 = P(array=[1.5, 1.4, 1.3, 1.2, 1.1, 1.0], frequency=0.5, offset = 0.5)
        # The p2=1.3 value checks that we are using the right tolerance
        p2 = P(array=[0.2, 1.3, 0.4, 1.15, 1.05, 0.95], frequency=0.5, offset = 1.5)
        arr, freq, off = blend_two_parameters(p1, p2, mode='glideslope')
        expected = np.ma.array(np.linspace(1.5, 0.95, num=12))
        # See blend_equispaced_sensors for endpoint handling...
        expected[-1] = 0.975
        ma_test.assert_masked_array_almost_equal(arr, expected)

    def test_blend_two_parameters_ils_raises(self):
        p1 = P(array=[1.5, 1.4], frequency=0.5, offset = 0.5)
        p2 = P(array=[1.5, 1.4], frequency=0.5, offset = 1.5)
        self.assertRaises(ValueError, blend_two_parameters, p1, p2, mode='glodeslipe')

    def test_blend_two_parameters_interp_1hz(self):
        # run the _interp function on on first parameter and ensure the result is 2Hz but does not have an offset
        # of 0.5
        p1 = P(array=[-0.66028192, -0.66028192, -0.66028192, -0.66028192, -0.66028192,
                      -0.66028192, -0.66028192, -0.66028192, -0.66028192, -0.66028192], frequency=1, offset=0.5)
        p2 = P(array=[-0.693044, -0.693044, -0.693044, -0.693044, -0.693044,
                      -0.693044, -0.693044, -0.693044, -0.693044, -0.693044], frequency=1, offset=0.078125)
        array, frequency, offset = blend_two_parameters(p1, p2)
        self.assertEqual(len(array), 20)
        self.assertEqual(frequency, 2.0)
        self.assertEqual(offset, 0.0)


class TestModulo(unittest.TestCase):
    def test_modulo(self):
        self.assertEqual(modulo(10, 2), 0)
        self.assertEqual(modulo(11, 2), 1)

        # Fails with floats due to IEEE 754 imprecision.
        self.assertNotEqual(6 % 0.2, 0)
        self.assertEqual(modulo(6, 0.2), 0)


class TestMostPointsCost(unittest.TestCase):
    def test_mpc_assertion(self):
        coefs=[0.0,0.0]
        x = np.ma.array([0.0])
        y = np.ma.array([0.0])
        self.assertRaises(ValueError, most_points_cost, coefs, x, y)

    def test_mpc_colinear_raises(self):
        coefs=[-1.0,0.0]
        x = np.ma.array([0.0, 1.0, 2.0])
        y = np.ma.array([0.0, -1.0, -2.0])
        self.assertRaises(ValueError, most_points_cost, coefs, x, y)

    def test_mpc_imbalanced_raises(self):
        coefs=[0.0,0.0]
        x = np.ma.array([0.0, 2.0])
        y = np.ma.array([0.0])
        self.assertRaises(ValueError, most_points_cost, coefs, x, y)

    def test_mpc_basic(self):
        coefs=[0.0,1.0]
        x = np.ma.zeros(3, dtype=np.float64)
        y = np.ma.zeros(3, dtype=np.float64)
        result = most_points_cost(coefs, x, y)
        self.assertAlmostEqual(result, 0.003, places=3)


class TestMovingAverage(unittest.TestCase):
    def test_basic_average(self):
        res = moving_average(np.ma.array([1,2,3,3,3,2,1]), window=3)
        # note mask at the start
        ma_test.assert_masked_array_approx_equal(res, np.ma.array(data=[4/3.0, 2.0, 8/3.0, 3.0, 8/3.0, 2.0, 4/3.0],
                                                                  mask=[0,0,0,0,0,0,0]))
        # 7 went in, 7 come out
        self.assertEqual(len(res), 7)

    def test_custom_weightings(self):
        res = moving_average(np.ma.arange(10), window=5, weightings=[0.2]*5)
        expected = np.ma.array(data=[0.6, 1.2, 2, 3, 4, 5, 6, 7.0, 7.8, 8.4],
                               mask=[  0,   0, 0, 0, 0, 0, 0,   0,   0,   0])
        ma_test.assert_masked_array_approx_equal(res, expected)

    def test_masked_edges(self):
        array = np.ma.array(data=[1,2,3,4,5,5,5,5,5,5,5,5,4,5,5],
                            mask=[0,0,0,0,0,0,0,0,0,1,1,1,0,0,1])
        res = moving_average(array, window=5)
        self.assertEqual(len(res), 15)
        expected = np.ma.array(data=[1.6, 2.2, 3,3.8,4.4,4.8, 5, 4.8,4.6,4.4, 4.2, 4.2, 4.4, 4.6, 1],
                               mask=[  0,   0, 0,  0,  0,  0, 0,   0,  0,  1,   1,   1,   0,   0, 1])
        ma_test.assert_masked_array_approx_equal(res, expected)

    def test_short_data(self):
        array = np.ma.array([1.0, 2, 3, 4])
        res = moving_average(array)
        expected = np.ma.array([6, 7, 8, 9])/3.0
        ma_test.assert_masked_array_approx_equal(res, expected)


class TestNearestNeighbourMaskRepair(unittest.TestCase):
    def test_nn_mask_repair(self):
        array = np.ma.arange(30)
        array[20:22] = np.ma.masked
        res = nearest_neighbour_mask_repair(array)
        self.assertEqual(len(res), 30)
        self.assertEqual(list(res[19:23]), [19,19,22,22])

    def test_nn_mask_repair_mappedarray(self):
        ms = M('Test', array=np.ma.array([0]*21 + [1]*9),
               values_mapping={0:'off', 1:'on'})
        ms.array[18:22] = np.ma.masked
        res = nearest_neighbour_mask_repair(ms.array)
        self.assertEqual(len(res), 30)
        self.assertEqual(list(res[17:23]),
                         ['off', 'off', 'off', 'on', 'on', 'on'])

    def test_nn_mask_repair_with_masked_edges(self):
        array = np.ma.arange(30)
        array[:10] = np.ma.masked
        array[20:22] = np.ma.masked
        array[-2:] = np.ma.masked
        res = nearest_neighbour_mask_repair(array)
        self.assertEqual(len(res), 30)
        self.assertEqual(list(res[:10]), [10]*10)
        self.assertEqual(list(res[-3:]), [27,27,27])

    def test_nn_mask_repair_limited_rolls(self):
        array = np.ma.array([101.5]*10 + [0]*10 + [107.4]*10)
        array[10:20] = np.ma.masked
        self.assertEqual(np.ma.count(array), 20)
        res = nearest_neighbour_mask_repair(array, repair_gap_size=4)
        self.assertEqual(np.ma.count(res), 24)

    # the extrapolating is effective enough.
    ##def test_nn_mask_repair_doesnt_roll_edges(self):
        ##array = np.ma.array([101.5]*10 + [0]*10 + [107.4]*10 + [0]*3)
        ##array[10:20] = np.ma.masked
        ##array[30:34] = np.ma.masked
        ##self.assertEqual(np.ma.count(array), 20)
        ##res = nearest_neighbour_mask_repair(array, repair_gap_size=2)
        ##self.assertEqual(np.ma.count(res), 24)

    def test_nn_mask_repair_forward(self):
        array = np.ma.arange(30)
        array[20:22] = np.ma.masked
        res = nearest_neighbour_mask_repair(array, direction='forward')
        self.assertEqual(len(res), 30)
        self.assertEqual(list(res[19:23]), [19,19,19,22])

    def test_nn_mask_repair_backward(self):
        array = np.ma.arange(30)
        array[20:22] = np.ma.masked
        res = nearest_neighbour_mask_repair(array, direction='backward')
        self.assertEqual(len(res), 30)
        self.assertEqual(list(res[19:23]), [19,22,22,22])

    def test_nn_mask_repair_direction(self):
        args = (np.ma.array([]), )
        try:
            for direction in ('both', 'forward', 'backward'):
                nearest_neighbour_mask_repair(*args, direction=direction)
        except ValueError:
            self.fail('ValueError from nearest_neighbour_mask_repair() for valid direction.')
        self.assertRaises(ValueError, nearest_neighbour_mask_repair, *args, direction='invalid')


class TestNormalise(unittest.TestCase):
    def test_normalise_copy(self):
        md = np.ma.concatenate([np.arange(10, dtype=np.float64), np.arange(20, 30, dtype=np.float64)])
        res = normalise(md, copy=True)
        self.assertNotEqual(id(res), id(md))
        self.assertEqual(md.max(), 29)

        res = normalise(md, copy=False)
        self.assertEqual(id(res), id(md))
        self.assertEqual(md.max(), 1.0)

    def test_normalise_two_dims(self):
        # normalise over all axis
        md = np.ma.array([np.arange(10, dtype=np.float64), np.arange(20,30, dtype=np.float64)])
        res1 = normalise(md)
        # normalised to max val 30 means first 10 will be below 0.33 and second 10 above 0.66
        assert_array_less(res1[0,:], 0.33)
        assert_array_less(res1[1,:], 1.1)

        # normalise with max value
        md = np.ma.array([np.arange(10, dtype=np.float64), np.arange(20,30, dtype=np.float64)])
        res1 = normalise(md, scale_max=40)
        # normalised to max val 40 means first 10 will be below 0.33 and second 10 above 0.66
        assert_array_less(res1[0,:], 0.226)
        assert_array_less(res1[1,:], 0.776)

        # normalise per axis
        res2 = normalise(md, axis=1)
        # each axis should be between 0 and 1
        self.assertEqual(res2[0,:].max(), 1.0)
        self.assertEqual(res2[1,:].max(), 1.0)

        # normalise per on all values across 0 axis
        res3 = normalise(md, axis=0)
        # each axis should be between 0 and 1
        self.assertEqual(res3.shape, (2,10))

        # normalise with an invariant zero axis
        md[1] *= 0
        res4 = normalise(md, axis=1)
        # each axis should be between 0 and 1
        self.assertEqual(res4.shape, (2,10))
        self.assertEqual(np.ma.count(res4[1]), 0)

    def test_normalise_masked(self):
        arr = np.ma.arange(10, dtype=float)
        arr[0] = 1000
        arr[0] = np.ma.masked
        arr[9] = np.ma.masked
        # mask the max value
        # Q: This does not modify the array in place, yet res is not used?
        norm = normalise(arr)
        index, value = max_value(norm)
        self.assertEqual(index, 8)
        self.assertEqual(value, 1.0)

    def test_normalise_no_variation(self):
        arr = np.ma.array([0.0]*10)
        norm = normalise(arr)
        index, value = max_value(norm)
        self.assertEqual(index, None)
        self.assertEqual(value, None)

class TestNpMaZerosLike(unittest.TestCase):
    def test_zeros_like_basic(self):
        result = np_ma_zeros_like(np.ma.array([1,2,3]))
        expected = np.ma.array([0,0,0])
        assert_array_equal(expected, result)

    def test_zeros_like_from_mask(self):
        result = np_ma_zeros_like(np.ma.array([1,2,3]))
        expected = np.ma.array([0,0,0])
        assert_array_equal(expected, result)

    def test_zeros_like_from_masked(self):
        result = np_ma_zeros_like(np.ma.array(data=[1,2,3],mask=[1,0,1]))
        expected = np.ma.array([0,0,0])
        assert_array_equal(expected, result)

    def test_zeros_like_from_all_masked(self):
        # This was found to be a special case.
        result = np_ma_zeros_like(np.ma.array(data=[1,2,3],mask=[1,1,1]))
        expected = np.ma.array([0,0,0])
        assert_array_equal(expected, result)

    def test_zeros_like_seperate_mask(self):
        start_arr = np.ma.array([1, 2, 3], mask=[1, 0, 1])
        np.testing.assert_array_equal(start_arr.data, [1, 2, 3])
        np.testing.assert_array_equal(start_arr.mask, [1, 0, 1])
        new_arr = np_ma_zeros_like(start_arr, mask=start_arr.mask)
        np.testing.assert_array_equal(new_arr.data, [0, 0, 0,])
        np.testing.assert_array_equal(new_arr.mask, [1, 0, 1,])
        self.assertTrue(start_arr.mask is not new_arr.mask)
        new_arr[0] = 1
        self.assertTrue(start_arr.mask is not new_arr.mask)
        np.testing.assert_array_equal(new_arr.data, [1, 0, 0,])
        np.testing.assert_array_equal(new_arr.mask, [0, 0, 1,])
        np.testing.assert_array_equal(start_arr.data, [1, 2, 3,])
        np.testing.assert_array_equal(start_arr.mask, [1, 0, 1,])


class TestNpMaConcatenate(unittest.TestCase, M):
    def test_concatenation_of_numeric_arrays(self):
        a1 = np.ma.arange(0,4)
        a2 = np.ma.arange(4,8)
        answer = np.ma.arange(8)
        assert_array_equal(np_ma_concatenate([a1,a2]),answer)

    def test_rejection_of_differing_arrays(self):
        a1 = M(name = 'a1',
               array = np.ma.array(data=[1,0,1,0],
                                   mask=False),
               data_type = 'Derived Multi-state',
               values_mapping = {0: 'Zero', 1: 'One'}
               )
        a2 = M(name = 'a2',
               array = np.ma.array(data=[0,0,1,1],
                                   mask=False),
               data_type = 'Derived Multi-state',
               values_mapping = {0: 'No', 1: 'Yes'}
               )
        self.assertRaises(ValueError, np_ma_concatenate, [a1.array,a2.array])

    def test_concatenation_of_similar_arrays(self):
        a1 = M(name = 'a1',
               array = np.ma.array(data=[1,0,1,0],
                                   mask=False),
               data_type = 'Derived Multi-state',
               values_mapping = {0: b'No', 1: b'Yes'}
               )
        a2 = M(name = 'a2',
               array = np.ma.array(data=[b'No',b'No',b'Yes',b'Yes'],
                                   mask=False, dtype=np.bytes_),
               data_type = 'Derived Multi-state',
               values_mapping = {0: b'No', 1: b'Yes'}
               )
        result = np_ma_concatenate([a1.array, a2.array])
        self.assertEqual([x for x in result],
                         [b'Yes',b'No',b'Yes',b'No',b'No',b'No',b'Yes',b'Yes'])
        self.assertEqual(list(result.raw), [1,0,1,0,0,0,1,1])

    def test_single_file(self):
        a=np.ma.arange(5)
        result = np_ma_concatenate([a])
        self.assertEqual(len(result), 5)
        assert_array_equal(result,a)

    def test_empty_list(self):
        self.assertEqual(np_ma_concatenate([]),None)


class TestNpMaOnesLike(unittest.TestCase):
    def test_zeros_like_basic(self):
        result = np_ma_ones_like(np.ma.array([1,2,3]))
        expected = np.ma.array([1,1,1])
        assert_array_equal(expected, result)


class TestNpMaMaskedZerosLike(unittest.TestCase):
    def test_masked_zeros_like_basic(self):
        result = np_ma_masked_zeros_like(np.ma.array([1,2,3]))
        np.testing.assert_array_equal(result.data, [0, 0, 0])
        np.testing.assert_array_equal(result.mask, [1, 1, 1])


class TestOffsetSelect(unittest.TestCase):
    def test_simple_minimum(self):
        e1=P('e1', np.ma.array([1]), offset=0.2)
        off = offset_select('first', [e1])
        self.assertEqual(off, 0.2)

    def test_simple_maximum(self):
        e1=P('e1', np.ma.array([1]), offset=0.2)
        e2=P('e2', np.ma.array([2]), offset=0.5)
        off = offset_select('last', [e1, e2])
        self.assertEqual(off, 0.5)

    def test_simple_mean(self):
        e1=P('e1', np.ma.array([1]), offset=0.2)
        e2=P('e2', np.ma.array([2]), offset=0.6)
        off = offset_select('mean', [e1, e2])
        self.assertEqual(off, 0.4)

    def test_complex_mean(self):
        e1=P('e1', np.ma.array([1]), offset=0.2)
        e2=P('e2', np.ma.array([2]), offset=0.6)
        e4=P('e4', np.ma.array([4]), offset=0.1)
        off = offset_select('mean', [None, e1, e4, e2])
        self.assertEqual(off, 0.3)


class TestOverflowCorrection(unittest.TestCase):
    '''
    This is applied to rad alt signals only during cleansing
    '''
    def test_overflow_correction_basic(self):
        air_slices = [slice(30, 100)]
        alt = [0]*30 + \
            list(range(0, 2000, 100)) + \
            [2000]*30 + \
            list(range(2000, 0, -100))+ \
            [0]*30
        hide = [0]*10 + [1]*10 + [0]*31 + [1]*9 + [0]*10 + [1]*10 + \
            [0]*30 + [1]*10 + [0]*10
        rad_alt = np.ma.array(data=alt, mask=hide, dtype=np.float)
        rad_alt = np.ma.mod(rad_alt, 1024)
        save_rad_alt = np.ma.copy(rad_alt)

        # Check of core correction process
        alt_radio = P('Altitude Radio (A)', rad_alt, frequency=0.25)
        result = overflow_correction(alt_radio.array, alt_radio, air_slices)
        self.assertEqual(result[50], 2000.0)
        self.assertTrue(np.ma.is_masked(result[65]))

        # To ensure cruise peak to peak condition is satisfied from now onwards
        save_rad_alt[62] += 25

        # Shifted data at takeoff
        rad_alt = np.ma.copy(save_rad_alt)
        rad_alt[18:52] += 2048
        alt_radio = P('Altitude Radio (A)', rad_alt, frequency=0.25)
        result = overflow_correction(alt_radio.array, alt_radio, air_slices)
        self.assertEqual(result[31], 100.0)

        # Shifted data at landing
        rad_alt = np.ma.copy(save_rad_alt)
        rad_alt[78:112] += 4096
        alt_radio = P('Altitude Radio (A)', rad_alt, frequency=0.25)
        result = overflow_correction(alt_radio.array, alt_radio, air_slices)
        self.assertEqual(result[99], 100.0)

        # Shifted data in cruise
        rad_alt = np.ma.copy(save_rad_alt)
        rad_alt[54:78] -= 1024
        alt_radio = P('Altitude Radio (A)', rad_alt, frequency=0.25)
        result = overflow_correction(alt_radio.array, alt_radio, air_slices)
        self.assertEqual(result[65], 2000.0)


    def test_overflow_correction_a320(self):
        air_slices = [slice(336, 5397), slice(5859, 11520)]
        radioA = load(os.path.join(
            test_data_path, 'A320_Altitude_Radio_A_overflow.nod'))
        for n in [371, 5092, 5298, 5894, 11240, 11452]:
            radioA.array.mask[n] = False
        resA = overflow_correction(radioA.array, radioA, air_slices)
        sects = np.ma.clump_unmasked(resA)
        self.assertEqual(len(sects), 5)
        self.assertGreater(resA.max(), 5000)
        self.assertEqual(resA.min(), -2)

        radioB = load(os.path.join(
            test_data_path, 'A320_Altitude_Radio_B_overflow.nod'))
        for n in range(433, 447):
            radioB.array.mask[n] = False
        for n in [371, 435, 446, 5296, 5297, 5893, 11317, 11451]:
            radioB.array.mask[n] = False
        resB = overflow_correction(radioB.array, radioB, air_slices)
        sects = np.ma.clump_unmasked(resB)
        self.assertEqual(len(sects), 4)
        self.assertGreater(resB.max(), 5000)
        self.assertEqual(resB.min(), -2)

    def test_overflow_correction_a340(self):
        air_slices = [slice(1020, 34400)]
        radioA = load(os.path.join(
            test_data_path, 'A340_Altitude_Radio_A_overflow.nod'))
        for n in [2349, 2350, 6148, 6149]:
            radioA.array.mask[n] = False
        resA = overflow_correction(radioA.array, radioA, air_slices)
        sects = np.ma.clump_unmasked(resA)
        # 1 section for climb, one for descent
        self.assertEqual(len(sects), 2)
        self.assertGreater(resA.max(), 7500)
        self.assertEqual(resA.min(), 0)

        radioB = load(os.path.join(
            test_data_path, 'A340_Altitude_Radio_B_overflow.nod'))
        for n in [2349, 2350, 6147, 6148]:
            radioB.array.mask[n] = False
        resB = overflow_correction(radioB.array, radioB, air_slices)
        sects = np.ma.clump_unmasked(resB)
        # 1 section for climb, one for descent
        # - and a third covers the ARINC429 data which is no longer masked
        self.assertEqual(len(sects), 2)
        self.assertGreater(resB.max(), 7500)
        self.assertEqual(resB.min(), 0)

class TestOverflowCorrectionArray(unittest.TestCase):
    '''
    Most functions tested by TestOverflowCorrection above.
    '''
    def test_mask_retention(self):
        array = np.ma.array(data=[5]*10, mask=[0]*3 + [1]*4 + [0]*3)
        result = overflow_correction_array(array, 1024.0)
        self.assertEqual(result.mask[5], True)

    @unittest.skip('This is effectively a data spike which should be handled elsewhere')
    def test_mask_ncd(self):
        # A specific case to be masked, using real data
        array = np.ma.array(data = [2300.0, 2300.0, 2350.0, 2450.0, 2480.0,
                                    2500.0, 2520.0, 2570.0, 2630.0,
                                    2500.0, 2710.0, 2790.0, 2830.0,
                                    2500.0, 2840.0, 2910.0, 2870.0,
                                    2900.0, 2920.0, 2980.0])
        result = overflow_correction_array(array)
        expected_mask = [False, False, False, False, False,
                         True, True, True, True,
                         True, True, True, True,
                         True, False, False, False,
                         False, False, False]
        ma_test.assert_masked_array_equal(result,
                                          np.ma.array(data=array.data,
                                                      mask=expected_mask)
                                          )

    def test_level_changes(self):
        '''
        Data from actual flight case
        '''
        array = np.ma.zeros(20500)
        indexes = [1001, 1085, 9413, 15230, 20079, 20108, 20306]
        jumps = [-4086.0, -4032.0, 1029.0, -2955.0, 1895.0, 4010.0, 4063.0]
        for n, index in enumerate(indexes):
            array[index] = -jumps[n]
        array = np.ma.cumsum(array)
        result = overflow_correction_array(array, 1024.0)
        # Overflow correction errors result in a final value around -4000, so anything positive is fine.
        self.assertGreater(result[-1], 0.0)


class TestPeakCurvature(unittest.TestCase):
    # Also known as the "Truck and Trailer" algorithm, this detects the peak
    # curvature point in an array.

    # The simple way to find peak curvature would be to take the second
    # differential of the parameter and seek minima, however this is a poor
    # technique as differentiation is a noisy process. To cater for noise in
    # the data, this algorithm provides a more robust analysis using least
    # squares fits to two blocks of data (the "Truck" and the "Trailer")
    # separated by a small gap. The point where the two have greatest
    # difference in slope corresponds to the point of greatest curvature
    # between them.

    # Note: The results from the first two tests are in a range format as the
    # artificial data results in multiple maxima.

    def test_peak_curvature_basic(self):
        array = np.ma.concatenate([np.zeros(20), np.arange(20)])
        pc = peak_curvature(array)
        self.assertEqual(pc,18.5)
        #  Very artificial case returns first location of many seconds of
        #  high curvature.

    def test_peak_curvature(self):
        array = np.ma.concatenate([np.zeros(40), np.arange(40)])
        pc = peak_curvature(array)
        self.assertGreaterEqual(pc,35)
        self.assertLessEqual(pc,45)

    def test_peak_curvature_void(self):
        array = np.ma.array([])
        pc = peak_curvature(array)
        self.assertEqual(pc,None)

    def test_peak_curvature_convex(self):
        array = np.ma.concatenate([np.zeros(40), np.arange(0, -40, -1)])
        pc = peak_curvature(array, curve_sense='Convex')
        self.assertGreaterEqual(pc,35)
        self.assertLessEqual(pc,45)

    def test_peak_curvature_convex_big_concave(self):
        # Tests for identification of the convex section with a larger concave angle in the same data segment.
        array = np.ma.concatenate([np.zeros(40), np.arange(0, -40, -1), np.arange(-40, 60, 10)])
        pc = peak_curvature(array, curve_sense='Convex')
        self.assertGreaterEqual(pc,35)
        self.assertLessEqual(pc,45)

    def test_peak_curvature_concave_big_convex(self):
        # See above !
        array = np.ma.concatenate([np.zeros(40), np.arange(40), np.arange(40,-60,-10)])
        pc = peak_curvature(array, curve_sense='Concave')
        self.assertGreaterEqual(pc,35)
        self.assertLessEqual(pc,45)

    def test_peak_curvature_flat_data(self):
        array = np.ma.ones(40) * 34
        pc = peak_curvature(array)
        self.assertEqual(pc,None)

    def test_peak_curvature_short_flat_data(self):
        array = np.ma.ones(4) * 34
        pc = peak_curvature(array)
        self.assertEqual(pc,None)

    def test_peak_curvature_bipolar(self):
        array = np.ma.concatenate([np.zeros(40), np.arange(40)])
        pc = peak_curvature(array, curve_sense='Bipolar')
        self.assertGreaterEqual(pc,35)
        self.assertLessEqual(pc,45)

    def test_peak_curvature_real_data(self):
        array = np.ma.array([37.9,37.9,37.9,37.9,37.9,38.2,38.2,38.2,38.2,38.8,
                             38.2,38.8,39.1,39.7,40.6,41.5,42.7,43.6,44.5,46,
                             47.5,49.6,52,53.2,54.7,57.4,60.7,61.9,64.3,66.1,
                             69.4,70.6,74.2,74.8])
        pc = peak_curvature(array)
        self.assertGreaterEqual(pc,15.0)
        self.assertLessEqual(pc,15.1)

    def test_peak_curvature_with_slice(self):
        array = np.ma.concatenate([np.zeros(20), np.ones(20) * 10, np.zeros(20)])
        pc = peak_curvature(array, slice(10, 50), curve_sense='Bipolar')
        self.assertEqual(pc, 24.5)

    def test_peak_curvature_slice_backwards(self):
        array = np.ma.concatenate([np.zeros(40), np.arange(40)])
        pc = peak_curvature(array, slice(75, 10, -1))
        self.assertEqual(pc, 41.5)

    def test_peak_curvature_masked_data_no_curve(self):
        array = np.ma.concatenate([np.zeros(40), np.arange(40)])
        array[:4] = np.ma.masked
        array[16:] = np.ma.masked
        pc = peak_curvature(array, slice(0,40))
        self.assertEqual(pc, None)

    def test_peak_curvature_masked_data(self):
        array = np.ma.concatenate([np.zeros(40), np.arange(40)])
        array[:4] = np.ma.masked
        array[66:] = np.ma.masked
        pc = peak_curvature(array, slice(0,78))
        self.assertEqual(pc, 38.5)

    def test_high_speed_turnoff_case(self):
        hdg_data=[]
        data_path = os.path.join(test_data_path,
                                 'runway_high_speed_turnoff_test.csv')
        with open(data_path, 'rt') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                hdg_data.append(float(row['Heading']))
            array = np.ma.array(hdg_data)

        pc=peak_curvature(array, curve_sense='Bipolar')
        self.assertLess(pc, 85)
        self.assertGreater(pc, 75)

    def test_none_slice_provided(self):
        # Designed to capture TypeError:
        # TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
        array = np.ma.array(np.arange(10), mask=[0,0,0,0,0,0,0,1,1,1])
        # also checks lower case curve_sense is turned Title case
        res = peak_curvature(array, _slice=slice(None), curve_sense='bipolar')
        # Range(10) has no curvature shape to it
        self.assertEqual(res, None)

    def test_invalid_curve_sense(self):
        self.assertRaises(ValueError, peak_curvature, [], curve_sense='INVALID')

    def test_multiple_valid_slices(self):
        #                 ignore |  Concave  | ignore  |  Convex
        array = np.ma.array([2,4,6,7,8,19,36,6,4,2,1,-2,-7,-12,-19,-28],
                       mask=[0,0,1,0,0, 0, 0,1,0,0,0, 1, 0,  0,  0,  0])
        res = peak_curvature(array, curve_sense='Concave')
        self.assertEqual(res, 4)
        res = peak_curvature(array, curve_sense='Concave', _slice=slice(None, None, -1))
        self.assertEqual(res, 5)
        res = peak_curvature(array, curve_sense='Convex')
        self.assertEqual(res, 13)


class TestPeakIndex(unittest.TestCase):
    def test_peak_index_no_data(self):
        self.assertRaises(ValueError, peak_index, [])

    def test_peak_index_one_sample(self):
        self.assertEqual(peak_index([4]),0)

    def test_peak_index_two_samples_rising(self):
        self.assertEqual(peak_index([2,4]),1)

    def test_peak_index_two_samples_falling(self):
        self.assertEqual(peak_index([4,2]),0)

    def test_peak_index_three_samples_falling(self):
        self.assertEqual(peak_index([6,4,2]),0)

    def test_peak_index_three_samples_rising(self):
        self.assertEqual(peak_index([1,2,3]),2)

    def test_peak_index_three_samples_with_peak(self):
        self.assertEqual(peak_index([1,2,1]),1)

    def test_peak_index_three_samples_trap_linear(self):
        self.assertEqual(peak_index([0,0.0000001,0]),1)

    def test_peak_index_real_peak(self):
        peak=np.sin(np.arange(10)/3.)
        self.assertEqual(peak_index(peak),4.7141807866121832)


class TestRateOfChangeArray(unittest.TestCase):
    # 12/8/12 - introduced to allow array level access to rate of change.
    # Also, handling short arrays added.
    def test_case_basic(self):
        test_array = np.ma.array([0,1,2,3,4], dtype=float)
        sloped = rate_of_change_array(test_array, 2.0, width=1.0)
        answer = np.ma.array([2,2,2,2,2])
        assert_array_almost_equal(sloped, answer)

    def test_case_changing_widths(self):
        test_array = np.ma.arange(20, dtype=float)
        answer = np.ones_like(test_array)
        sloped = rate_of_change_array(test_array, 1.0, width=2)
        assert_array_almost_equal(sloped.data, answer)
        sloped = rate_of_change_array(test_array, 1.0, width=3)
        assert_array_almost_equal(sloped.data, answer)
        sloped = rate_of_change_array(test_array, 1.0, width=4)
        assert_array_almost_equal(sloped.data, answer)
        sloped = rate_of_change_array(test_array, 1.0, width=5)
        assert_array_almost_equal(sloped.data, answer)
        sloped = rate_of_change_array(test_array, 1.0, width=6)
        assert_array_almost_equal(sloped.data, answer)
        sloped = rate_of_change_array(test_array, 1.0, width=7)
        assert_array_almost_equal(sloped.data, answer)


    def test_case_short_data(self):
        test_array = np.ma.array([0,1,2,3,4], dtype=float)
        sloped = rate_of_change_array(test_array, 2.0, 10.0)
        answer = np.ma.array([0,0,0,0,0])
        assert_array_almost_equal(sloped, answer)

    def test_case_very_short_data(self):
        test_array = np.ma.array([99], dtype=float)
        sloped = rate_of_change_array(test_array, 2.0, 10.0)
        answer = np.ma.array([0])
        assert_array_almost_equal(sloped, answer)

    def test_masked(self):
        test_array = np.ma.array(data=[1,2,3,4,5,6,6,6,5,4,3,2,1],
                                 mask=[0,0,0,0,0,0,1,0,0,0,0,0,0])
        sloped = rate_of_change_array(test_array, 1.0, 4.0)
        answer = np.ma.array(data=[1,1,99,99,99,99,99,99,99,99,99,-1,-1],
                             mask=[0,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        assert_array_equal(sloped, answer)


class TestRateOfChange(unittest.TestCase):
    # 13/4/12 Changed timebase to be full width as this is more logical.
    # Reminder: was: rate_of_change(to_diff, half_width, hz) - half width in seconds.
    # Reminder: is: rate_of_change(to_diff, width) - width in seconds and freq from parameter.

    def test_rate_of_change_basic(self):
        sloped = rate_of_change(P('Test',
                                  np.ma.array([1, 0, -1, 2, 1, 3, 4, 6, 5, 7],
                                              dtype=float), 1), 4)
        answer = np.ma.array(data=[-1.0,-1.0,0.0,0.75,1.25,1.0,1.0,1.0,-1.0,2.0],
                             mask=False)
        ma_test.assert_mask_equivalent(sloped, answer)

    def test_rate_of_change_increased_frequency(self):
        sloped = rate_of_change(P('Test',
                                  np.ma.array([1, 0, -1, 2, 1, 3, 4, 6, 5, 7],
                                              dtype=float), 2), 4)
        answer = np.ma.array(data=[-2.0,-2.0,6.0,-2.0,1.0,1.75,2.0,4.0,-2.0,4.0],
                             mask=False)
        ma_test.assert_mask_equivalent(sloped, answer)

    def test_rate_of_change_reduced_frequency(self):
        sloped = rate_of_change(P('Test',
                                  np.ma.array([1, 0, -1, 2, 1, 3, 4, 6, 5, 7],
                                              dtype=float), 0.5), 4)
        answer = np.ma.array(data=[-0.5,-0.5,0.5,0.5,0.25,0.75,0.75,0.25,0.25,1.0],
                             mask=False)
        ma_test.assert_mask_equivalent(sloped, answer)

    def test_rate_of_change_reduced_frequency_extended(self):
        sloped = rate_of_change(P('Test',
                                  np.ma.arange(20, dtype=float), 0.5), 4)
        answer = np.ma.array(data=[0.5]*20, mask=False)
        ma_test.assert_mask_equivalent(sloped, answer)

    def test_rate_of_change_transfer_mask(self):
        sloped = rate_of_change(P('Test',
                                  np.ma.array(data = [1, 0, -1, 2, 1, 3, 4, 6, 5, 7],dtype=float,
                            mask = [0, 1,  0, 0, 0, 1, 0, 0, 0, 1]), 1), 2)
        answer = np.ma.array(data = [0,-1.0,0,1.0,0,1.5,0,0.5,0,0],
             mask = [True,False,True,False,True,False,True,False,True,True])
        ma_test.assert_mask_equivalent(sloped, answer)

    def test_rate_of_change_half_width_zero(self):
        self.assertRaises(ValueError,
                          rate_of_change,
                          P('Test',np.ma.array([0, 1, 0]), 1), 0)

    def test_rate_of_change_half_width_negative(self):
        self.assertRaises(ValueError,
                          rate_of_change,
                          P('Test',np.ma.array([0, 1, 0]), 1), -2)

    def test_rate_of_change_small_values(self):
        sloped = rate_of_change(P('Test',np.ma.arange(10)/100.0, 1), 4)
        answer = np.ma.array(data=[0.01]*10,mask=False)
        ma_test.assert_masked_array_approx_equal(sloped, answer)

    def test_rate_of_change_regression(self):
        sloped = rate_of_change(P('Test',
                                  np.ma.array([0,0,0,0,0,0,1,1,1,1,1,1],
                                              dtype=float), 1), 5, method='regression')
        answer = np.ma.array(data=[0.0,0.0,0.0,0.0,0.2,0.3,0.3,0.2,0.0,0.0,0.0,0.0],
                             mask=False)
        ma_test.assert_mask_equivalent(sloped, answer)


class TestRepairMask(unittest.TestCase):
    def setUp(self):
        self.basic_data = np.ma.array(
            [0, 0, 10, 0, 0, 20, 23, 26, 30, 0, 0],
            mask=[True] * 2 + [False] + [True] * 2 + [False] * 4 + [True] * 2)

    def test_repair_mask_basic_fill_start(self):
        self.assertEqual(repair_mask(self.basic_data,
                                     method='fill_start').tolist(),
                         [None, None, 10, 10, 10, 20, 23, 26, 30, 30, 30])
        self.assertEqual(repair_mask(self.basic_data, extrapolate=True,
                                     method='fill_start').tolist(),
                         [10, 10, 10, 10, 10, 20, 23, 26, 30, 30, 30])

    def test_repair_mask_basic_fill_stop(self):
        self.assertEqual(repair_mask(self.basic_data,
                                     method='fill_stop').tolist(),
                         [10, 10, 10, 20, 20, 20, 23, 26, 30, None, None])
        self.assertEqual(repair_mask(self.basic_data, extrapolate=True,
                                     method='fill_stop').tolist(),
                         [10, 10, 10, 20, 20, 20, 23, 26, 30, 30, 30])

    def test_repair_mask_basic_1(self):
        array = np.ma.arange(10)
        array[3] = np.ma.masked
        self.assertTrue(np.ma.is_masked(array[3]))
        array[6:8] = np.ma.masked
        res = repair_mask(array)
        np.testing.assert_array_equal(res.data,range(10))
        # test mask is now unmasked
        self.assertFalse(np.any(res.mask[3:9]))

    def test_repair_mask_too_much_invalid(self):
        array = np.ma.arange(20)
        array[4:15] = np.ma.masked
        res = repair_mask(array)
        ma_test.assert_masked_array_approx_equal(res, array)

    def test_repair_mask_not_at_start(self):
        array = np.ma.arange(10)
        array[0] = np.ma.masked
        res = repair_mask(array)
        ma_test.assert_masked_array_approx_equal(res, array)

    def test_repair_mask_not_at_end(self):
        array = np.ma.arange(10)
        array[9] = np.ma.masked
        res = repair_mask(array)
        ma_test.assert_masked_array_approx_equal(res, array)

    def test_repair_short_sample(self):
        # Very short samples were at one time returned as None, but simply
        # applying the normal "rules" seems more consistent, so this is a
        # test to show that an old function no longer applies.
        array = np.ma.arange(2)
        array[1] = np.ma.masked
        res = repair_mask(array)
        ma_test.assert_masked_array_approx_equal(res, array)

    def test_extrapolate(self):
        array = np.ma.array([2,4,6,7,5,3,1],mask=[1,1,0,0,1,1,1])
        res = repair_mask(array, extrapolate=True)
        expected = np.ma.array([6,6,6,7,7,7,7],mask=[0,0,0,0,0,0,0])
        assert_array_equal(res, expected)

    def test_fully_masked_array(self):
        array = np.ma.array(np.arange(10), mask=[1]*10)
        # fully masked raises ValueError
        self.assertRaises(ValueError, repair_mask, array)
        # fully masked returns a masked zero array
        res = repair_mask(array, raise_entirely_masked=False)
        assert_array_equal(res.data, array.data)
        assert_array_equal(res.mask, True)

    def test_repair_mask_basic_2(self):
        array = np.ma.arange(10)
        array[5] = np.ma.masked
        array[7:9] = np.ma.masked
        res = repair_mask(array, repair_above=5)
        np.testing.assert_array_equal(res.data,range(10))
        # test only array[5] is still masked as is the first
        self.assertFalse(np.ma.is_masked(res[4]))
        self.assertTrue(np.ma.is_masked(res[5]))
        self.assertFalse(np.ma.is_masked(res[6]))
        self.assertFalse(np.ma.is_masked(res[7]))
        self.assertFalse(np.ma.is_masked(res[8]))
        self.assertFalse(np.ma.is_masked(res[9]))


class TestResample(unittest.TestCase):
    def test_resample_upsample(self):
        assert_equal(
            resample(np.array([True, False, True, True]), 0.5, 1),
            np.array([True, True, False, False, True, True, True, True]))

    def test_resample_downsample(self):
        assert_equal(
            resample(np.array([True, False, True, False, True]), 4, 2),
            np.array([True, True]))


class TestRoundToNearest(unittest.TestCase):
    def test_round_to_nearest(self):
        array = np.ma.arange(50)
        res = round_to_nearest(array, 5)

        self.assertEqual(list(res[:15]),
                         [0,0,0,5,5,5,5,5,10,10,10,10,10,15,15])
        self.assertEqual(list(res[-7:]),
                         [45]*5 + [50]*2)

    def test_round_to_nearest_with_mask(self):
        array = np.ma.arange(20)
        array.mask = [True]*10 + [False]*10
        res = round_to_nearest(array, 5)
        self.assertEqual(list(np.ma.filled(res, fill_value=-1)),
                         [-1]*10 + [10,10,10,15,15,15,15,15,20,20])


class TestRMSNoise(unittest.TestCase):
    def test_rms_noise_basic(self):
        array = np.ma.array([0,0,1,0,0])
        result = rms_noise(array)
        expected = sqrt(1.5/3.0)
        self.assertAlmostEqual(result, expected)

    def test_rms_noise_patent_example(self):
        array = np.ma.array([8,7,6,9,4,8,2,7,5])
        result = rms_noise(array)
        expected = sqrt(107.75/7.0)
        self.assertAlmostEqual(result, expected)

    def test_rms_noise_ignores_slope(self):
        array = np.ma.arange(20)
        result = rms_noise(array)
        expected = 0.0
        self.assertAlmostEqual(result, expected)

    def test_rms_noise_masked(self):
        array = np.ma.array([0,0,0,1,0,0,0])
        array[3]=np.ma.masked
        result = rms_noise(array)
        expected = 0.0
        self.assertAlmostEqual(result, expected)

    def test_rms_noise_no_valid_data(self):
        array = np.ma.array([0,0,1,0,0])
        array[2]=np.ma.masked
        result = rms_noise(array)
        expected = None
        self.assertAlmostEqual(result, expected)

    def test_rms_noise_with_ignore_three(self):
        # This ignores three values, the 45 and the adjacent "minima"
        array = np.ma.array([0,0,1,0,0,45,0,0,1,0,0])
        result = rms_noise(array, ignore_pc=30)
        expected = sqrt(1.5/3.0)
        self.assertAlmostEqual(result, expected)

    def test_rms_noise_with_ignore_one(self):
        # This ignores only the 45 value
        array = np.ma.array([0,0,1,0,0,45,0,0,1,0,0])
        result = rms_noise(array, ignore_pc=10)
        expected = 11.2666
        self.assertAlmostEqual(result, expected, places=3)

    def test_rms_noise_gave_nan(self):
        # This case caused a significant failure when Altitude STD failed its
        # noise test! Code changed to only remove extreme values when the
        # array is long enough.
        array = np.ma.array([-77.0, -61.0, -47.0])
        result = rms_noise(array, ignore_pc=10)
        expected = 1.0
        self.assertEqual(result, expected)


class TestRunwayDistanceFromEnd(unittest.TestCase):
    def test_null(self):
        runway =  {'end': {'latitude': 60.280151,
                      'longitude': 5.222579},
              'localizer': {'latitude': 60.2789,
                            'longitude': 5.223},
              'glideslope': {'latitude': 60.300981,
                             'longitude': 5.214092,
                             'threshold_distance': 1161},
              'start': {'latitude': 60.30662494,
                        'longitude': 5.21370074}}
        result = runway_distance_from_end(runway, 60.280151, 5.222579)
        expected = 0.0
        self.assertEqual(result, expected)

    def test_runway_dist_by_coordinates(self):
        runway =  {'end': {'latitude': 60.280151,
                      'longitude': 5.222579},
              'localizer': {'latitude': 60.2789,
                            'longitude': 5.223},
              'glideslope': {'latitude': 60.300981,
                             'longitude': 5.214092,
                             'threshold_distance': 1161},
              'start': {'latitude': 60.30662494,
                        'longitude': 5.21370074}}
        result = runway_distance_from_end(runway, 60.30662494, 5.21370074)
        expected = 2984.0
        self.assertAlmostEqual(result, expected, places=0)

    def test_runway_dist_by_identifier(self):
        runway =  {'end': {'latitude': 60.280151,
                      'longitude': 5.222579},
              'localizer': {'latitude': 60.2789,
                            'longitude': 5.223},
              'glideslope': {'latitude': 60.300981,
                             'longitude': 5.214092,
                             'threshold_distance': 1161},
              'start': {'latitude': 60.30662494,
                        'longitude': 5.21370074}}
        result = runway_distance_from_end(runway, point='start')
        expected = 2984.0
        self.assertAlmostEqual(result, expected, places=0)

    def test_runway_dist_not_recognised(self):
        runway =  {'end': {'latitude': 60.280151,
                           'longitude': 5.222579},
                   'start': {'latitude': 60.30662494,
                             'longitude': 5.21370074},
                   'id':'Test Case'}
        result = runway_distance_from_end(runway, point='threshold')
        self.assertEqual(result, None)


class TestRunwayDistances(unittest.TestCase):
    # This single test case used data for Bergen and had reasonable accuracy.
    # However, since setting up this test the runway has been extended so
    # DON'T use this for navigation !!!
    def setUp(self):
        self.result = []
        self.runway =  {'end': {'latitude': 60.280151,
                                'longitude': 5.222579},
                        'localizer': {'latitude': 60.2789,
                                      'frequency': u'109900M',
                                      'longitude': 5.223,
                                      'heading': 173,
                                      'beam_width': 4.5},
                        'glideslope': {'latitude': 60.300981,
                                       'frequency': u'333800M',
                                       'angle': 3.1,
                                       'longitude': 5.214092,
                                       'threshold_distance': 1161},
                        'start': {'latitude': 60.30662494,
                                  'longitude': 5.21370074},
                        'strip': {'width': 147, 'length': 9810,
                                  'id': 4097, 'surface': u'ASP'},
                        'identifier': u'17', 'id': 8193}

    def test_runway_distances(self):
        result = runway_distances(self.runway)

        self.assertAlmostEqual(result[0],3125, places=0)
        # correct:self.assertAlmostEqual(result[0],3125, places=0)
        self.assertAlmostEqual(result[1],2503, places=0)
        self.assertAlmostEqual(result[2],141.0, places=1)
        # Optional glideslope antenna projected position...
        self.assertAlmostEqual(result[3],60.3, places=1)
        self.assertAlmostEqual(result[4],5.22, places=2)

    def test_runway_distances_with_localizer_database_error(self):
        self.runway['localizer']['latitude'] = 5.223
        self.runway['localizer']['longitude'] = 60.2789
        result = runway_distances(self.runway)

        # Check the localizer position is unchanged.
        self.assertAlmostEqual(result[0],3125, places=0)

    def test_runway_distances_with_glildeslope_database_error(self):
        self.runway['glideslope']['latitude'] = 5.214092
        self.runway['glideslope']['longitude'] = 60.300981
        result = runway_distances(self.runway)

        # Check the glildeslope position is unchanged.
        self.assertAlmostEqual(result[1],2503, places=0)


class TestRunwayDeviation(unittest.TestCase):
    # Reminder: def runway_deviation(array, runway):
    def test_runway_deviation(self):
        runway =  {'end': {'latitude': 60.280151,
                           'longitude': 5.222579},
                   'start': {'latitude': 60.30662494,
                             'longitude': 5.21370074},
                   'magnetic_heading': 160}
        head=np.ma.array([170.568, 180.568, 160.568, 3050.568, -1269.432])
        expected=np.ma.array([0.0, 10.0, -10.0, 0.0, 0.0])
        result = runway_deviation(head, runway)
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_runway_deviation_preset_heading(self):
        # pass
        head = np.ma.array([118, 120, 122, 123, 124])
        res = runway_deviation(head, heading=120)
        self.assertEqual(list(res), [-2, 0, 2, 3, 4])


class TestRunwayHeading(unittest.TestCase):
    # This test case uses data for Bergen and has been checked against
    # Google Earth measurements for reasonable accuracy.
    def test_runway_heading(self):
        runway =  {'end': {'latitude': 60.280151,
                              'longitude': 5.222579},
                      'localizer': {'latitude': 60.2789,
                                    'longitude': 5.223,
                                    'heading': 999},
                      'start': {'latitude': 60.30662494,
                                'longitude': 5.21370074},
                      'magnetic_heading':170}
        rwy_hdg = runway_heading(runway)
        self.assertLess(abs(rwy_hdg - 170.6), 0.3)

    # This case illustrates use of an Attribute.
    def test_dubai(self):
        rwy = A(name='test',
                value=[{'runway': {'end': {'latitude': 25.262131,
                                           'longitude': 55.347572},
                                   'start': {'latitude': 25.243322,
                                             'longitude': 55.381519},
                                   'magnetic_heading':300}}])
        result = runway_heading(rwy.value[0]['runway'])
        self.assertGreater(result, 298)
        self.assertLess(result, 302)

    # This case tests handling of a rare database error in open source airport databases.
    def test_database_reversed(self):
        rwy = A(name='Nadi',
                value=[{'runway': {'end': {'latitude': -17.7727,
                                           'longitude': 177.429},
                                   'start': {'latitude': -17.7499,
                                             'longitude': 177.44756},
                                   'magnetic_heading': 2.4,
                                   'identifier': '20'}}])
        result = runway_heading(rwy.value[0]['runway'])
        # Runway 20 at Nadi is 37 deg by Google Earth
        self.assertGreater(result, 35)
        self.assertLess(result, 40)


class TestRunwayLength(unittest.TestCase):
    @mock.patch('analysis_engine.library.great_circle_distance__haversine')
    def test_runway_length(self, _dist):
        _dist.return_value = 100
        length = runway_length({'start': {'latitude': 10, 'longitude': 20},
                                'end': {'latitude': 30, 'longitude': 40}})
        _dist.assert_called_with(10, 20, 30, 40)
        self.assertEqual(length, 100)


class TestRunwaySnap(unittest.TestCase):
    def test_runway_snap(self):
        runway =  {'end': {'latitude': 60.280151,
                              'longitude': 5.222579},
                      'start': {'latitude': 60.30662494,
                                'longitude': 5.21370074}}
        lat, lon = runway_snap(runway, 60.29, 5.23)
        self.assertEqual(lat,60.289141034411045)
        self.assertEqual(lon,5.219564115819171)

    def test_runway_snap_a(self):
        runway =  {'end': {'latitude': 60.2,
                              'longitude': 5.2},
                      'start': {'latitude': 60.3,
                                'longitude': 5.25}}
        lat, lon = runway_snap(runway, 60.2, 5.2)
        self.assertEqual(lat,60.2)
        self.assertEqual(lon,5.2)

    def test_runway_snap_b(self):
        runway =  {'end': {'latitude': 60.2,
                              'longitude': 5.2},
                      'start': {'latitude': 60.3,
                                'longitude': 5.25}}
        lat, lon = runway_snap(runway, 60.3, 5.25)
        self.assertEqual(lat,60.3)
        self.assertEqual(lon,5.25)

    def test_runway_snap_d(self):
        runway =  {'end': {'latitude': 60.2,
                              'longitude': 5.2},
                      'start': {'latitude': 60.2,
                                'longitude': 5.2}}
        lat, lon = runway_snap(runway, 60.3, 5.25)
        self.assertEqual(lat,None)
        self.assertEqual(lon,None)

    def test_snap_no_runway(self):
        runway =  {}
        lat, lon = runway_snap(runway, 60.3, 5.25)
        self.assertEqual(lat,None)
        self.assertEqual(lon,None)


class TestRunwaySnapDict(unittest.TestCase):
    @mock.patch('analysis_engine.library.runway_snap')
    def test_runway_snap_dict(self, runway_snap):
        runway_snap.return_value = (70, 80)
        runway = {'start': {'latitude': 10, 'longitude': 20},
                  'end': {'latitude': 30, 'longitude': 40}}
        lat = 50
        lon = 60
        coords = runway_snap_dict(runway, lat, lon)
        runway_snap.assert_called_with(runway, lat, lon)
        self.assertEqual(coords, {'latitude': 70, 'longitude': 80})


class TestRunwayTouchdown(unittest.TestCase):
    def test_runway_touchdown_quite_short(self):
        # Based on Stellar Air Park in Phoenix - the touchdown markings look strangely deep, and are
        # more like 350m down the runway. Still, the end locations are fine.
        runway = {'start': {'latitude':  33.29360, 'longitude': -111.915817},
                  'end': {'latitude':  33.303773, 'longitude': -111.915792}}
        tdn_dist, coords = runway_touchdown(runway)
        self.assertEqual(tdn_dist, 265)
        self.assertAlmostEqual(coords['latitude'], 33.295982, places=4)
        self.assertAlmostEqual(coords['longitude'], -111.915803, places=4)

    def test_runway_touchdown_zero(self):
        # Helipads have zero length, for example Battersea  :o)
        runway = {'start': {'latitude': 51.469955, 'longitude': -0.179543},
                      'end': {'latitude': 51.469955, 'longitude': -0.179543}}
        tdn_dist, coords = runway_touchdown(runway)
        self.assertEqual(tdn_dist, 0)
        self.assertAlmostEqual(coords['latitude'], 51.469955)
        self.assertAlmostEqual(coords['longitude'], -0.179543, places=4)

"""
class TestSectionContainsKti(unittest.TestCase):
    def test_valid(self):
        section =  S(items=[Section('first_section', slice(4,6))])
        kti = KTI(items=[KeyTimeInstance(name='More Test', index=5)])
        self.assertTrue(section_contains_kti(section.get_first(), kti))

    def test_invalid_for_two_ktis(self):
        section =  S(items=[Section('first_section', slice(4,8))])
        kti = KTI(items=[KeyTimeInstance(name='More Test', index=5),
                         KeyTimeInstance(name='More Test', index=6)])
        self.assertFalse(section_contains_kti(section.get_first(), kti))

    def test_invalid_for_no_ktis(self):
        section =  S(items=[Section('first_section', slice(4,8))])
        kti = []
        self.assertFalse(section_contains_kti(section.get_first(), kti))

    def test_invalid_for_two_slices(self):
        section =  S(items=[Section('first_section', slice(4,8)),
                            Section('second_section', slice(14,18))])
        kti = KTI(items=[KeyTimeInstance(name='More Test', index=5)])
        self.assertFalse(section_contains_kti(section, kti))
"""


class TestRunsOfOnes(unittest.TestCase):

    def setUp(self):
        self.test_array = np.ma.array(
            [0,0,1,0,1,1,1,1,1,0,0,1,1,1,0,1,1,1],
            mask=14 * [False] + 4 * [True])

    def test_runs_of_ones(self):
        result = runs_of_ones(self.test_array)
        self.assertEqual(result, [slice(2, 3), slice(4, 9), slice(11, 14)])

    def test_runs_of_ones_min_samples(self):
        result = runs_of_ones(self.test_array, min_samples=2)
        self.assertEqual(result, [slice(4, 9), slice(11, 14)])

    def test_runs_of_ones_skip_mask(self):
        this_array = np.ma.array(data=[0,0,1,1,0,1,1,0,1,1,0,0],
                                 mask=[0,0,0,0,1,0,0,0,0,0,0,0])
        result = runs_of_ones(this_array, skip_mask=True)
        self.assertEqual(result, [slice(2, 7), slice(8, 10)])

    def test_runs_of_ones_empty_array(self):
        self.assertEqual(runs_of_ones(np.empty(0)), [])

    def test_runs_of_ones_all_masked(self):
        result = runs_of_ones(np.ma.array(data=[1]*5,
                                          mask=[True]*5))
        self.assertEqual(np.ma.count(result), 0)


class TestSlicesOfRuns(unittest.TestCase):

    def test__slices_of_runs(self):
        array = np.ma.repeat(range(0, 3), 3)
        result = slices_of_runs(array)
        expected = [(0, [slice(0, 3)]), (1, [slice(3, 6)]), (2, [slice(6, 9)])]
        self.assertIsInstance(result, types.GeneratorType)
        self.assertEqual(list(result), expected)

    def test__slices_of_runs__multiple_slices(self):
        array = np.ma.repeat((0, 1, 0), 3)
        result = slices_of_runs(array)
        expected = [(0, [slice(0, 3), slice(6, 9)]), (1, [slice(3, 6)])]
        self.assertIsInstance(result, types.GeneratorType)
        self.assertEqual(list(result), expected)

    def test__slices_of_runs__exclude_masked(self):
        array = np.ma.repeat(range(0, 2), 5)
        array[3:7] = np.ma.masked
        result = slices_of_runs(array)
        expected = [(0, [slice(0, 3)]), (1, [slice(7, 10)])]
        self.assertIsInstance(result, types.GeneratorType)
        self.assertEqual(list(result), expected)

    def test__slices_of_runs__mapped_array(self):
        array = np.ma.repeat(range(0, 3), 3)
        array.mask = np.ma.getmaskarray(array)  # irritating...
        array = MappedArray(array, values_mapping={0: 'A', 1: 'B', 2: 'C'})
        result = slices_of_runs(array)
        expected = [('A', [slice(0, 3)]), ('B', [slice(3, 6)]), ('C', [slice(6, 9)])]
        self.assertIsInstance(result, types.GeneratorType)
        self.assertEqual(list(result), expected)

    def test__slices_of_runs__flat(self):
        array = np.ma.repeat(range(0, 2), 5)
        array[3:7] = np.ma.masked
        result = slices_of_runs(array, flat=True)
        expected = [slice(0, 3, None), slice(7, 10, None)]
        self.assertIsInstance(result, types.GeneratorType)
        self.assertEqual(list(result), expected)


class TestShiftSlice(unittest.TestCase):
    def test_shift_slice(self):
        a = slice(1, 3, None)
        b = 10
        self.assertEqual(shift_slice(a, b), slice(11, 13, None))

    def test_shift_slice_too_short(self):
        a = slice(3.3, 3.4)
        b = 6
        self.assertEqual(shift_slice(a, b), None)

    def test_shift_slice_transfer_none(self):
        a = slice(30.3, None)
        b = 3
        self.assertEqual(shift_slice(a, b), slice(33.3, None))

    def test_shift_slice_transfer_none_reversed(self):
        a = slice(None, 23.8)
        b = 4.2
        self.assertEqual(shift_slice(a, b), slice(None, 28.0))

    def test_shift_slice_no_shift(self):
        a = slice(2, 5)
        self.assertEqual(shift_slice(a, 0), a)
        self.assertEqual(shift_slice(a, None), a)

    def test_shift_slice_len_1(self):
        self.assertEqual(shift_slice(slice(82, 83), 413),
                         slice(495, 496))


class TestShiftSlices(unittest.TestCase):
    def test_shift_slices(self):
        a = [slice(1, 3, None)]
        b = 10
        self.assertEqual(shift_slices(a, b), [slice(11, 13, None)])

    def test_shift_slices_incl_none(self):
        a = [slice(1, 3, None), None, slice(2, 4, 2)]
        b = 10
        self.assertEqual(shift_slices(a, b), [slice(11, 13, None),
                                              slice(12, 14, 2)])

    def test_shift_slices_real_data(self):
        a = [slice(0, 1, None), slice(599, 933, None),
             slice(1988, 1992, None), slice(2018, 2073, None)]
        b = 548.65
        self.assertEqual(len(shift_slices(a,b)),4)
        self.assertEqual(shift_slices(a,b)[0].stop,549.65)
        self.assertEqual(shift_slices(a,b)[-1].start,2566.65)

    def test_shift_slices_no_shift(self):
        a = [slice(4, 7, None), slice(17, 12, -1)]
        self.assertEqual(shift_slices(a, 0), a)
        self.assertEqual(shift_slices(a, None), a)


class TestSliceDuration(unittest.TestCase):
    def test_slice_duration(self):
        duration = slice_duration(slice(10, 20), 2)
        self.assertEqual(duration, 5)
        duration = slice_duration(slice(None, 20), 0.5)
        self.assertEqual(duration, 40)
        self.assertRaises(ValueError, slice_duration, slice(20, None), 1)


class TestSlicesAnd(unittest.TestCase):
    def test_slices_and(self):
        self.assertEqual(slices_and([slice(2,5)],[slice(3,7)]),
                         [slice(3,5)])
        self.assertEqual(slices_and([slice(2,5),slice(7,None)],[slice(3,9)]),
                         [slice(3,5), slice(7,9)])
        self.assertEqual(slices_and([slice(2,5),slice(7,None)],[slice(9,3,-1)]),
                         [slice(4,5), slice(7,10)])
        self.assertEqual(slices_and([slice(5,2,-1),slice(7,None)],[slice(9,3,-1)]),
                         [slice(4,6), slice(7,10)])


class TestSlicesAbove(unittest.TestCase):
    def test_slices_above(self):
        array = np.ma.concatenate([np.arange(10), np.arange(10)])
        array.mask = np.repeat([0, 1, 0, 1], [12, 2, 4, 2])
        # Let's make sure we don't alter the original array mask
        original = array.copy()
        repaired_array, slices = slices_above(array, 5)
        self.assertEqual(slices, [slice(5, 10, None), slice(15, 18, None)])
        np.testing.assert_array_equal(original.mask, array.mask)


class TestSlicesAfter(unittest.TestCase):
    def test_slices_after(self):
        self.assertEqual(slices_after([], 5), [])
        self.assertEqual(slices_after([slice(10, 15)], 5), [slice(10, 15)])
        self.assertEqual(slices_after([slice(10, 15)], 12), [slice(12, 15)])
        self.assertEqual(slices_after([slice(0, 5), slice(10, 15),
                                       slice(10, 20), slice(20, 30)], 12),
                         [slice(12, 15), slice(12, 20), slice(20, 30)])


class TestSlicesBefore(unittest.TestCase):
    def test_slices_before(self):
        self.assertEqual(slices_before([], 20), [])
        self.assertEqual(slices_before([slice(10, 15)], 20), [slice(10, 15)])
        self.assertEqual(slices_before([slice(10, 15)], 12), [slice(10, 12)])
        self.assertEqual(slices_before([slice(0, 5), slice(8, 15),
                                       slice(10, 20), slice(20, 30)], 12),
                         [slice(0, 5), slice(8, 12), slice(10, 12)])


class TestSlicesBelow(unittest.TestCase):
    def test_slices_below(self):
        array = np.ma.concatenate([np.arange(10), np.arange(10)])
        array.mask = [True] * 2 + [False] * 18
        repaired_array, slices = slices_below(array, 5)
        self.assertEqual(slices, [slice(2, 6, None), slice(10, 16, None)])


class TestSlicesBetween(unittest.TestCase):
    def test_slices_between(self):
        array = np.ma.arange(20)
        array.mask = [True] * 10 + [False] * 10
        repaired_array, slices = slices_between(array, 5, 15)
        self.assertEqual(slices, [slice(10, 15)])


class TestSlicesDuration(unittest.TestCase):
    def test_slices_duration(self):
        self.assertEqual(slices_duration([], 1), 0)
        self.assertEqual(slices_duration([slice(0, 10)], 1), 10)
        self.assertEqual(slices_duration([slice(5, 10), slice(12,15),
                                          slice(30, 60)], 2), 19)


class TestSlicesContract(unittest.TestCase):
    def test_slices_contract(self):
        slices = [slice(None, 10), slice(30, 40), slice(70, None)]
        self.assertEqual(slices_contract(slices, 0), slices)
        expected = [slice(2, 8), slice(32, 38), slice(72, None)]
        self.assertEqual(slices_contract(slices, 2), expected)
        expected = [slice(2, 8), slice(32, 38), slice(72, 78)]
        self.assertEqual(slices_contract(slices, 2, max_index=80), expected)
        # Smaller slices may be contracted 'out of existence'.
        slices = [slice(None, 20, 2), slice(30, 40), slice(70, None)]
        expected = [slice(6, 14, 2), slice(76, None)]
        self.assertEqual(slices_contract(slices, 6), expected)
        expected = [slice(6, 14, 2)]
        self.assertEqual(slices_contract(slices, 6, max_index=80), expected)
        expected = [slice(5, 15, 2)]
        self.assertEqual(slices_contract(slices, 5, max_index=80), expected)
        expected = [slice(90, None)]
        self.assertEqual(slices_contract(slices, 20), expected)
        expected = []
        self.assertEqual(slices_contract(slices, 20, max_index=80), expected)

        # Negative step is not supported.
        slices = [slice(5, 25), slice(40, 60, -2)]
        self.assertRaises(NotImplementedError, slices_contract, slices, 2)
        # Negative or 0 max_index is invalid.
        slices = [slice(5, 25)]
        self.assertRaises(ValueError, slices_contract, slices, 2, 0)
        self.assertRaises(ValueError, slices_contract, slices, 5, -4)


class TestSlicesContractDuration(unittest.TestCase):
    def test_slices_contract_duration(self):
        slices = [slice(None, 10), slice(30, 40), slice(70, None)]
        expected = [slice(2, 8), slice(32, 38), slice(72, None)]
        self.assertEqual(slices_contract_duration(slices, 1/2.0, 1), expected)
        expected = [slice(0.5, 9.5), slice(30.5, 39.5), slice(70.5, 79.5)]
        self.assertEqual(slices_contract_duration(slices, 2, 1, max_index=80), expected)


class TestSlicesExtend(unittest.TestCase):
    def test_slices_extend(self):
        slices = [slice(None, 10), slice(30, 40), slice(70, None)]
        self.assertEqual(slices_extend(slices, 0), slices)
        expected = [slice(0, 15), slice(25, 45), slice(65, None)]
        self.assertEqual(slices_extend(slices, 5), expected)
        expected = [slice(0, 52), slice(58, None)]
        self.assertEqual(slices_extend(slices, 12), expected)
        # Step is retained.
        slices = [slice(10, 20, 3)]
        expected = [slice(5, 25, 3)]
        self.assertEqual(slices_extend(slices, 5), expected)
        # Negative step is not supported.
        slices = [slice(10, 20, -1)]
        self.assertRaises(NotImplementedError, slices_extend, slices, 5)
        # Does not extend slices beyond 0 otherwise slicing behaviour changes.
        slices = [slice(0, 10)]
        expected = [slice(0, 15)]
        self.assertEqual(slices_extend(slices, 5), expected)


class TestSlicesExtendDuration(unittest.TestCase):
    def test_slices_extend_duration(self):
        slices = [slice(None, 10), slice(30, 40), slice(70, None)]
        expected = [slice(0, 16), slice(24, 46), slice(64, None)]
        self.assertEqual(slices_extend_duration(slices, 1/2.0, 3), expected)


class TestSliceSamples(unittest.TestCase):
    def test_slice_samples(self):
        test_slice=slice(45,47,1)
        self.assertEqual(slice_samples(test_slice), 2)

    def test_slice_samples_backwards(self):
        test_slice=slice(48,45,-1)
        self.assertEqual(slice_samples(test_slice), 3)

    def test_slice_samples_stepping(self):
        test_slice=slice(10,20,4)
        self.assertEqual(slice_samples(test_slice), 3)

    def test_slice_samples_step_none(self):
        test_slice=slice(10,20)
        self.assertEqual(slice_samples(test_slice), 10)

    def test_slice_samples_start_none(self):
        test_slice=slice(None,20)
        self.assertEqual(slice_samples(test_slice), 0)

    def test_slice_samples_stop_none(self):
        test_slice=slice(5,None,20)
        self.assertEqual(slice_samples(test_slice), 0)


class TestSlicesFromTo(unittest.TestCase):
    def test_slices_from_to_invalid_threshold(self):
        array = mock.Mock()
        self.assertRaises(ValueError, slices_from_to, array, 0, 10, threshold=-0.1)
        self.assertRaises(ValueError, slices_from_to, array, 0, 10, threshold=1.1)
        self.assertRaises(ValueError, slices_from_to, array, 0, 10, threshold=100)

    def test_slices_from_to(self):
        array = np.ma.arange(20)
        array.mask = [True] * 10 + [False] * 10
        # Ascending.
        repaired_array, slices = slices_from_to(array, 5, 15)
        self.assertEqual(slices, [slice(10, 15)])
        # Descending.
        repaired_array, slices = slices_from_to(array, 18, 3)
        self.assertEqual(slices, [])
        array = np.ma.arange(20, 0, -1)
        array.mask = [True] * 10 + [False] * 10
        repaired_array, slices = slices_from_to(array, 18, 3)
        self.assertEqual(slices, [slice(10, 17)])

    def test_slices_from_to_landing(self):
        '''
        Common usage it to end in 0 for ...to landing. This test covers this
        specific case.
        '''
        array = np.ma.array([25, 20, 15, 10, 5, 0, 0, 0, 0])
        _, slices = slices_from_to(array, 17, 0)
        self.assertEqual(slices, [slice(2, 5, None)])

    def test_slices_from_to_short_up(self):
        '''
        A problem was found with very short resulting arrays, which this test
        covers. In fact, very short cases (one or two valid samples) still
        cause a problem as the from_to process calls a lower routine which
        masks data outside the range leaving you uncertain which direction
        the band was passing through.
        '''
        array = np.ma.array([1,3,5,7])
        _, slices = slices_from_to(array, 2, 6)
        self.assertEqual(slices[0], slice(1,3,None))
        _, slices = slices_from_to(array, 2, 4)
        self.assertEqual(slices[0], slice(1,2,None))
        array = np.ma.array([7,5,3,1])
        _, slices = slices_from_to(array, 2, 6)
        self.assertEqual(slices, [])
        _, slices = slices_from_to(array, 2, 4)
        self.assertEqual(slices, [])

    def test_slices_from_to_short_down(self):
        '''
        See above.
        '''
        array = np.ma.array([7,5,3,1])
        _, slices = slices_from_to(array, 6, 2)
        self.assertEqual(slices[0], slice(1,3,None))
        _, slices = slices_from_to(array, 6, 4)
        self.assertEqual(slices[0], slice(1,2,None))
        array = np.ma.array([1,3,5,7])
        _, slices = slices_from_to(array, 6, 2)
        self.assertEqual(slices, [])
        _, slices = slices_from_to(array, 6, 4)
        self.assertEqual(slices, [])

    def test_slices_from_to_single_sample(self):
        '''
        Single samples will only be detected if there is an unmasked sample
        before and an unmasked sample after.
        '''
        array = np.ma.array([7,5,3,1], mask=[False] * 4)
        # Start of array.
        _, slices = slices_from_to(array, 8, 6)
        self.assertEqual(slices, [])
        # End of array.
        _, slices = slices_from_to(array, 2, 0)
        self.assertEqual(slices, [])
        # Within array.
        _, slices = slices_from_to(array, 4, 2)
        self.assertEqual(slices, [slice(2,3)])
        # Masked before.
        array.mask[0:2] = True
        _, slices = slices_from_to(array, 4, 2)
        self.assertEqual(slices, [])
        # Masked after.
        array.mask[0:2] = False
        array.mask[3] = True
        _, slices = slices_from_to(array, 4, 2)
        self.assertEqual(slices, [])
        # Masked before and after.
        array.mask[0:2] = True
        array.mask[3] = True
        _, slices = slices_from_to(array, 4, 2)
        self.assertEqual(slices, [])

    def test_slices_from_to_two_samples(self):
        '''
        #Single samples will only be detected if there is an unmasked sample
        #before and an unmasked sample after.
        '''
        array = np.ma.array([9, 7, 5, 3, 1], mask=[False] * 5)
        # Start of array.
        _, slices = slices_from_to(array, 10, 6)
        self.assertEqual(slices, [slice(0, 2)])
        # End of array.
        _, slices = slices_from_to(array, 4, 0)
        self.assertEqual(slices, [slice(3, 5)])
        # Within array.
        _, slices = slices_from_to(array, 6, 2)
        self.assertEqual(slices, [slice(2, 4)])
        # Masked before.
        array.mask[0:2] = True
        _, slices = slices_from_to(array, 6, 2)
        self.assertEqual(slices, [slice(2, 4)])
        # Masked after.
        array.mask[0:2] = False
        array.mask[4] = True
        _, slices = slices_from_to(array, 6, 2)
        self.assertEqual(slices, [slice(2, 4)])
        # Masked before and after.
        array.mask[0:2] = True
        array.mask[4] = True
        _, slices = slices_from_to(array, 6, 2)
        self.assertEqual(slices, [slice(2, 4)])

    def test_slices_from_to_dip(self):
        array = np.ma.array([10, 8, 6, 4, 2, 4, 6, 8, 10])
        _, slices = slices_from_to(array, 9, 3)
        self.assertEqual(slices, [slice(1, 4)])
        _, slices = slices_from_to(array, 9, 0)
        self.assertEqual(slices, [slice(1, 4)])
        _, slices = slices_from_to(array, 3, 9)
        self.assertEqual(slices, [slice(5, 8)])
        _, slices = slices_from_to(array, 0, 9)
        self.assertEqual(slices, [slice(4, 8)])
        # Multiple curves.
        array = np.ma.array([10, 8, 6, 4, 5, 6, 7, 8, 6, 4, 2, 4, 6, 8, 10])
        _, slices = slices_from_to(array, 9, 1)
        self.assertEqual(slices, [slice(1, 10)])
        _, slices = slices_from_to(array, 1, 9)
        self.assertEqual(slices, [slice(10, 14)])
        array = np.ma.array([10, 8, 6, 4, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 6, 8,
                             10])
        _, slices = slices_from_to(array, 9, 1)
        self.assertEqual(slices, [slice(1, 4)])
        _, slices = slices_from_to(array, 1, 9)
        self.assertEqual(slices, [slice(4, 17)])
        # threshold
        array = np.ma.array([10, 8, 6, 4, 2, 4, 6, 8, 10, 8, 7, 8, 10])
        _, slices = slices_from_to(array, 8, 0)
        self.assertEqual(slices, [slice(2, 4), ])
        _, slices = slices_from_to(array, 8, 0, threshold=0.2)
        self.assertEqual(slices, [slice(2, 4)])
        _, slices = slices_from_to(array, 8, 0, threshold=0.8)
        self.assertEqual(slices, [])

    def test_slices_from_to_peak(self):
        array = np.ma.array([2, 4, 6, 8, 10, 8, 6, 4, 2])
        #_, slices = slices_from_to(array, 9, 3)
        #self.assertEqual(slices, [slice(5, 8)])
        _, slices = slices_from_to(array, 9, 0)
        self.assertEqual(slices, [slice(5, 9)])
        _, slices = slices_from_to(array, 3, 9)
        self.assertEqual(slices, [slice(1, 4)])
        _, slices = slices_from_to(array, 0, 9)
        self.assertEqual(slices, [slice(0, 4)])
        # threshold
        array = np.ma.array([10, 12, 14, 16, 18, 16, 14, 12, 10, 12, 13, 12, 10, 10, 12, 14, 16, 18, 20, 22])
        _, slices = slices_from_to(array, 12, 20)
        self.assertEqual(slices, [slice(2, 4), slice(15, 18)])
        _, slices = slices_from_to(array, 12, 20, threshold=0.2)
        self.assertEqual(slices, [slice(2, 4), slice(15, 18)])
        _, slices = slices_from_to(array, 12, 20, threshold=0.8)
        self.assertEqual(slices, [slice(15, 18)])
        _, slices = slices_from_to(array, 12, 20, threshold=1)
        self.assertEqual(slices, [slice(15, 18)])

    def test_slices_from_to_only_within_range(self):
        array = np.ma.array([5, 5, 5, 5, 5, 5, 5, 5, 5])
        # level data should not be considered as changing from one value
        # to another
        _, slices = slices_from_to(array, 9, 3)
        self.assertEqual(slices, [])
        _, slices = slices_from_to(array, 3, 9)
        self.assertEqual(slices, [])
        array[:2] = np.ma.masked
        _, slices = slices_from_to(array, 3, 9)
        self.assertEqual(slices, [])
        array[-3:] = np.ma.masked
        _, slices = slices_from_to(array, 9, 0)
        self.assertEqual(slices, [])
        # data does not cross through midpoint
        array = np.ma.array([5, 6, 5, 6, 5, 6, 5, 6, 5])
        _, slices = slices_from_to(array, 3, 9)
        self.assertEqual(slices, [])
        # data crosses through midpoint, but has indeterminate direction
        # XXX: could dissect data, but this would complicate the algorithm
        # further.
        array = np.ma.array([5, 6, 7, 8, 7, 6, 5, 4, 5])
        _, slices = slices_from_to(array, 3, 9)
        self.assertEqual(slices, [])
        # data crosses through midpoint and appears to be linear.
        # linear climbing
        array = np.ma.array([2, 3, 4, 5, 6, 7, 8])
        _, slices = slices_from_to(array, 1, 9)
        self.assertEqual(slices, [slice(0, 7)])
        array[:2] = np.ma.masked
        _, slices = slices_from_to(array, 1, 9)
        self.assertEqual(slices, [slice(2, 7)])
        array[-1:] = np.ma.masked
        _, slices = slices_from_to(array, 1, 9)
        self.assertEqual(slices, [slice(2, 6)])
        # linear descending
        array = np.ma.array([8, 7, 6, 5, 4, 3, 2])
        _, slices = slices_from_to(array, 9, 1)
        self.assertEqual(slices, [slice(0, 7)])
        array[:2] = np.ma.masked
        _, slices = slices_from_to(array, 9, 1)
        self.assertEqual(slices, [slice(2, 7)])
        array[-1:] = np.ma.masked
        _, slices = slices_from_to(array, 9, 1)
        self.assertEqual(slices, [slice(2, 6)])

    def test_slices_from_to_starts_within_range(self):
        # stops outside of range
        # linear climbing
        array = np.ma.array([5, 6, 7, 8, 9, 10])
        # wrong direction
        _, slices = slices_from_to(array, 9, 2)
        self.assertEqual(slices, [])
        # right direction
        _, slices = slices_from_to(array, 2, 9)
        self.assertEqual(slices, [slice(0, 4)])
        array[:2] = np.ma.masked
        _, slices = slices_from_to(array, 2, 9)
        self.assertEqual(slices, [slice(2, 4)])
        # linear descending
        array = np.ma.array([10, 9, 8, 7, 6, 5])
        # wrong direction
        _, slices = slices_from_to(array, 6, 12)
        self.assertEqual(slices, [])
        # right direction
        _, slices = slices_from_to(array, 12, 6)
        self.assertEqual(slices, [slice(0, 4)])
        array[:2] = np.ma.masked
        _, slices = slices_from_to(array, 12, 6)
        self.assertEqual(slices, [slice(2, 4)])
        # dip curve
        array = np.ma.array([10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11, 12])
        _, slices = slices_from_to(array, 11, 2)
        self.assertEqual(slices, [slice(0, 5)])
        _, slices = slices_from_to(array, 2, 11)
        self.assertEqual(slices, [slice(5, 11)])
        # peak curve
        array = np.ma.array([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8])
        _, slices = slices_from_to(array, 17, 9)
        self.assertEqual(slices, [slice(5, 11)])
        _, slices = slices_from_to(array, 9, 17)
        self.assertEqual(slices, [slice(0, 5)])

    def test_slices_from_to_stops_within_range(self):
        # stops outside of range
        # linear climbing
        array = np.ma.array([5, 6, 7, 8, 9, 10])
        # wrong direction
        _, slices = slices_from_to(array, 12, 6)
        self.assertEqual(slices, [])
        # right direction
        _, slices = slices_from_to(array, 6, 12)
        self.assertEqual(slices, [slice(2, 6)])
        array[:2] = np.ma.masked
        _, slices = slices_from_to(array, 6, 12)
        self.assertEqual(slices, [slice(2, 6)])
        # linear descending
        array = np.ma.array([10, 9, 8, 7, 6, 5])
        # wrong direction
        _, slices = slices_from_to(array, 2, 9)
        self.assertEqual(slices, [])
        # right direction
        _, slices = slices_from_to(array, 9, 2)
        self.assertEqual(slices, [slice(2, 6)])
        array[:3] = np.ma.masked
        _, slices = slices_from_to(array, 9, 2)
        self.assertEqual(slices, [slice(3, 6)])
        # dip curve
        array = np.ma.array([12, 11, 10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10])
        _, slices = slices_from_to(array, 11, 2)
        self.assertEqual(slices, [slice(2, 7)])
        _, slices = slices_from_to(array, 2, 11)
        self.assertEqual(slices, [slice(7, 13)])
        # peak curve
        array = np.ma.array([8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10])
        _, slices = slices_from_to(array, 17, 9)
        self.assertEqual(slices, [slice(7, 13)])
        _, slices = slices_from_to(array, 9, 17)
        self.assertEqual(slices, [slice(2, 7)])

    def test_slices_from_to_only_outside_of_range(self):
        array = np.ma.array([5, 5, 5, 5, 5, 5, 5, 5, 5])
        _, slices = slices_from_to(array, 2, 4)
        self.assertEqual(slices, [])
        _, slices = slices_from_to(array, 4, 0)
        self.assertEqual(slices, [])
        _, slices = slices_from_to(array, 6, 20)
        self.assertEqual(slices, [])
        _, slices = slices_from_to(array, 15, 7)
        self.assertEqual(slices, [])

    def test_slices_from_to_invalid_range(self):
        array = np.ma.arange(10)
        self.assertRaises(ValueError, slices_from_to, array, 2, 2)
        self.assertRaises(ValueError, slices_from_to, array, -2, -2)

    def test_slices_from_to_alt_aal(self):
        '''
        Altitude AAL features a Go Around which provides an example of a dip
        in real data.
        '''
        alt_aal = load(os.path.join(test_data_path,
                                    'slices_from_to_alt_aal.nod'))
        _, slices = slices_from_to(alt_aal.array, 1000, 20)
        self.assertEqual(slices, [slice(9340, 9448),
                                  slice(10625, 10774)])


class TestSlicesFromKtis(unittest.TestCase):
    def test_basic_structure(self):
        kti_1 = KTI(items=[KeyTimeInstance(1, 'KTI_1')])
        kti_2 = KTI(items=[KeyTimeInstance(3, 'KTI_2')])
        slices = slices_from_ktis(kti_1, kti_2)
        self.assertEqual(slices, [slice(1,4)])

    def test_reverse_order(self):
        kti_1 = KTI(items=[KeyTimeInstance(3, 'KTI_1')])
        kti_2 = KTI(items=[KeyTimeInstance(2, 'KTI_2')])
        slices = slices_from_ktis(kti_1, kti_2)
        self.assertEqual(slices, [])

    def test_multiple(self):
        kti_1 = KTI(items=[KeyTimeInstance(10, 'KTI_1'),
                           KeyTimeInstance(5, 'KTI_1'),
                           KeyTimeInstance(2, 'KTI_1'),
                           KeyTimeInstance(13, 'KTI_1')])
        kti_2 = KTI(items=[KeyTimeInstance(27, 'KTI_2'),
                           KeyTimeInstance(18, 'KTI_2'),
                           KeyTimeInstance(8, 'KTI_2'),
                           KeyTimeInstance(-3, 'KTI_2'),
                           KeyTimeInstance(20, 'KTI_2')])
        slices = slices_from_ktis(kti_1, kti_2)
        self.assertEqual(slices, [slice(5,9), slice(13,19)])

    def test_nones(self):
        kti_1 = KTI(items=[])
        kti_2 = KTI(items=[])
        slices = slices_from_ktis(kti_1, kti_2)
        self.assertEqual(slices, [])
        slices = slices_from_ktis(kti_1, None)
        self.assertEqual(slices, [])
        slices = slices_from_ktis(None, kti_2)
        self.assertEqual(slices, [])
        slices = slices_from_ktis(None, None)
        self.assertEqual(slices, [])

    def test_not_lists(self):
        kti_1 = KTI(items=[KeyTimeInstance(5, 'KTI_1')])[0]
        kti_2 = KeyTimeInstance(8, 'KTI_2')
        slices = slices_from_ktis(kti_1, kti_2)
        self.assertEqual(slices, [slice(5,9)])


class TestSliceMultiply(unittest.TestCase):
    def test_slice_multiply(self):
        self.assertEqual(slice_multiply(slice(1,2,3),2),
                         slice(2,4,6))
        self.assertEqual(slice_multiply(slice(None,None,None),1),
                         slice(None,None,None))
        self.assertEqual(slice_multiply(slice(1,6,None),0.5),
                         slice(1,3,None))
        self.assertEqual(slice_multiply(slice(1,2,0.5),-2),
                         slice(-2,-4,-1))


class TestSliceRound(unittest.TestCase):
    def test_slice_round(self):
        self.assertEqual(slice_round(slice(None)), slice(None))
        self.assertEqual(slice_round(slice(5, None)), slice(5, None))
        self.assertEqual(slice_round(slice(None, 5)), slice(None, 5))
        self.assertEqual(slice_round(slice(1, 5)), slice(1, 5))
        self.assertEqual(slice_round(slice(1.2, 5)), slice(1, 5))
        self.assertEqual(slice_round(slice(1.7, 5)), slice(2, 5))
        self.assertEqual(slice_round(slice(1.2, 5.2)), slice(1, 5))
        self.assertEqual(slice_round(slice(1.2, 5.7)), slice(1, 6))


class TestSlicesRound(unittest.TestCase):
    def test_slices_round(self):
        self.assertEqual(slices_round([]), [])
        self.assertEqual(slices_round([slice(None)]), [slice(None)])
        self.assertEqual(slices_round([slice(None), slice(1.2, 2.7)]), [slice(None), slice(1, 3)])


class TestSlicesMultiply(unittest.TestCase):
    def test_slices_multiply(self):
        slices = [slice(1,2,3),slice(None,None,None),slice(1,2,None)]
        result = [slice(3,6,9),slice(None,None,None),slice(3,6,None)]
        self.assertEqual(slices_multiply(slices,3),result)

    def test_slices_multiply_with_zero_start(self):
        slices = [slice(0,2,None)]
        result = [slice(0,8,None)]
        self.assertEqual(slices_multiply(slices,4),result)


class TestSlicesOverlap(unittest.TestCase):
    def test_slices_overlap(self):
        # overlap
        first = slice(10,20)
        second = slice(15,25)
        self.assertTrue(slices_overlap(first, second))
        self.assertTrue(slices_overlap(second, first))

        # None in slices
        start_none = slice(None, 12)
        self.assertTrue(slices_overlap(first, start_none))
        self.assertFalse(slices_overlap(second, start_none))
        self.assertTrue(slices_overlap(start_none, first))
        self.assertFalse(slices_overlap(start_none, second))

        end_none = slice(22,None)
        self.assertFalse(slices_overlap(first, end_none))
        self.assertTrue(slices_overlap(second, end_none))
        self.assertFalse(slices_overlap(end_none, first))
        self.assertTrue(slices_overlap(end_none, second))

        both_none = slice(None, None)
        self.assertTrue(slices_overlap(first, both_none))
        self.assertTrue(slices_overlap(second, both_none))
        self.assertTrue(slices_overlap(both_none, first))
        self.assertTrue(slices_overlap(both_none, second))

        # no overlap
        no_overlap = slice(25,40)
        self.assertFalse(slices_overlap(second, no_overlap))
        self.assertFalse(slices_overlap(no_overlap, first))

        # step negative
        self.assertRaises(ValueError, slices_overlap, first, slice(1,2,-1))

class TestSlicesOverlapMerge(unittest.TestCase):

    def test_slices_overlap_merge_basic(self):
        first = [slice(10,20)]
        second = [slice(15,25)]
        self.assertEqual(slices_overlap_merge(first, second),
                         [slice(10, 25)])

    def test_slices_overlap_merge_no_overlap(self):
        first = [slice(10,20)]
        second = [slice(25,35)]
        self.assertEqual(slices_overlap_merge(first, second),
                         [slice(10, 20)])

    def test_slices_overlap_merge_no_first(self):
        first = []
        second = [slice(25,35)]
        self.assertEqual(slices_overlap_merge(first, second), [])

    def test_slices_overlap_merge_no_second(self):
        first = [slice(10,20)]
        second = []
        self.assertEqual(slices_overlap_merge(first, second),
                         [slice(10, 20)])

    def test_slices_overlap_extend_start(self):
        first = [slice(20,35)]
        second = [slice(10,25)]
        self.assertEqual(slices_overlap_merge(first, second, extend_start=2),
                             [slice(8, 35)])

    def test_slices_overlap_extend_stop(self):
        first = [slice(10,25)]
        second = [slice(20,35)]
        self.assertEqual(slices_overlap_merge(first, second, extend_stop=2),
                             [slice(10, 37)])

    def test_slices_overlap_merge_limited(self):
        first = [slice(10,20)]
        second = [slice(1,35)]
        self.assertEqual(slices_overlap_merge(first, second, limit_start=2, limit_end=3),
                         [slice(8, 23)])

class TestSlicesOverlay(unittest.TestCase):
    def test_slices_and(self):
        # overlay
        first = [slice(10, 20)]
        second = [slice(15, 25)]
        self.assertEqual(slices_and(first, second), [slice(15, 20)])

        # no overlap
        no_overlap = slice(25,40)
        self.assertEqual(slices_and(second, [no_overlap]), [])

        ## This test now redundant as slices_and handles reverse order.
        ### step negative
        ##self.assertRaises(ValueError, slices_and, first, [slice(1, 2, -1)])

        # complex with all four permutations
        first = [slice(5, 15), slice(20, 25), slice(30, 40)]
        second = [slice(10, 35), slice(45, 50)]
        result = [slice(10, 15), slice(20, 25), slice(30, 35)]
        self.assertEqual(slices_and(first,second),result)


class TestSlicesRemoveOverlaps(unittest.TestCase):
    def test_basic__duplicate_slices(self):
        slices = [slice(3, 8), slice(3, 8), slice(20, 22)]
        newlist = slices_remove_overlaps(slices)
        expected = [slice(3, 8), slice(20, 22)]
        self.assertEqual(expected, newlist)

    def test_basic__first_slice_longer(self):
        slices = [slice(1, 5), slice(3, 8), slice(20, 22)]
        newlist = slices_remove_overlaps(slices)
        expected = [slice(3, 8), slice(20, 22)]
        self.assertEqual(expected, newlist)

    def test_basic__second_slice_longer(self):
        slices = [slice(1, 5), slice(3, 7), slice(20, 22)]
        newlist = slices_remove_overlaps(slices)
        expected = [slice(1, 5), slice(20, 22)]
        self.assertEqual(expected, newlist)

    def test_basic__longest_slice_overlapping_two(self):
        slices = [slice(1, 5), slice(3, 7), slice(2, 22)]
        newlist = slices_remove_overlaps(slices)
        expected = [slice(2, 22)]
        self.assertEqual(expected, newlist)

    def test_complex(self):
        slices = [slice(*s) for s in ((538, 570), (571, 582), (605, 606), (539, 540), (541, 544), (545, 546), (547, 558), (559, 561), (562, 563), (564, 582))]
        newlist = slices_remove_overlaps(slices)
        expected = [slice(538, 570), slice(571, 582), slice(605, 606)]
        self.assertEqual(expected, newlist)


class TestSlicesRemoveSmallGaps(unittest.TestCase):
    def test_slice_removal(self):
        slicelist = [slice(1, 3), slice(5, 7), slice(20, 22)]
        newlist = slices_remove_small_gaps(slicelist)
        expected = [slice(1, 7), slice(20, 22)]
        self.assertEqual(expected, newlist)

    def test_slice_removal_count_based(self):
        slicelist = [slice(1, 3), slice(5, 7), slice(20, 22)]
        newlist = slices_remove_small_gaps(slicelist, count=10)
        expected = [slice(1, 7), slice(20, 22)]
        self.assertEqual(expected, newlist)

    def test_slice_removal_big_time(self):
        slicelist = [slice(1, 3), slice(5, 7), slice(20, 22)]
        newlist = slices_remove_small_gaps(slicelist,time_limit=15)
        expected = [slice(1, 22)]
        self.assertEqual(expected, newlist)

    def test_slice_removal_big_freq(self):
        slicelist = [slice(1, 3), slice(5, 7), slice(20, 22)]
        newlist = slices_remove_small_gaps(slicelist, hz=2)
        expected = [slice(1, 22)]
        self.assertEqual(expected, newlist)


    def test_slice_return_single_slice(self):
        slicelist = [slice(5, 7)]
        newlist = slices_remove_small_gaps(slicelist, hz=2)
        expected = [slice(5, 7)]
        self.assertEqual(expected, newlist)

    def test_slice_return_none(self):
        slicelist = [None]
        newlist = slices_remove_small_gaps(slicelist, hz=2)
        expected = [None]
        self.assertEqual(expected, newlist)

    def test_slice_return_empty(self):
        slicelist = []
        newlist=slices_remove_small_gaps(slicelist, hz=2)
        expected = []
        self.assertEqual(expected, newlist)

    def test_slice_none_within_slices(self):
        slicelist = [slice(1, None), slice(4,6), slice(None, 8)]
        newlist = slices_remove_small_gaps(slicelist)
        self.assertEqual(newlist, [slice(None, None, None)])

    def test_slice_start_none(self):
        slicelist = [slice(None, 5), slice(6, 7)]
        newlist = slices_remove_small_gaps(slicelist)
        self.assertEqual(newlist, [slice(None, 7)])

    def test_unsorted_slices(self):
        slicelist = [slice(*s) for s in (
            (2565, 4378), (1911, 2277), (1642, 1817), (631, 796), (820, 892), (2420, 2489),
            (1846, 1910), (2317, 2363), (2364, 2401), (2490, 2508), (2402, 2419), (798, 815),
            (1818, 1832), (2303, 2316), (2290, 2302), (1149, 1160), (1021, 1032), (1037, 1047),
            (1836, 1841), (962, 966), (609, 612), (1161, 1164), (2286, 2289), (1842, 1845))]
        newlist = slices_remove_small_gaps(slicelist, 4)
        self.assertEqual(newlist, [slice(*s) for s in (
            (609, 612), (631, 815), (820, 892), (962, 966), (1021, 1032), (1037, 1047),
            (1149, 1164), (1642, 1832), (1836, 2277), (2286, 2508), (2565, 4378))])


class TestSlicesRemoveSmallSlices(unittest.TestCase):
    def test_slice_removal(self):
        slicelist = [slice(1, 13), slice(25, 27), slice(30, 43)]
        newlist = slices_remove_small_slices(slicelist)
        expected = [slice(1, 13), slice(30, 43)]
        self.assertEqual(expected, newlist)

    def test_slice_removal_time_set(self):
        slicelist = [slice(1, 13), slice(25, 27), slice(30, 33)]
        newlist = slices_remove_small_slices(slicelist, time_limit=5)
        expected = [slice(1, 13)]
        self.assertEqual(expected, newlist)

    def test_slice_removal_time_set_different_freq(self):
        slicelist = [slice(1, 13), slice(25, 27), slice(30, 33)]
        newlist = slices_remove_small_slices(slicelist, time_limit=10, hz=0.5)
        expected = [slice(1, 13)]
        self.assertEqual(expected, newlist)

    def test_slice_removal_count_set(self):
        slicelist = [slice(1, 13), slice(25, 27), slice(30, 33)]
        newlist = slices_remove_small_slices(slicelist, count=5)
        expected = [slice(1, 13)]
        self.assertEqual(expected, newlist)

    def test_slice_removal_null_cases(self):
        slicelist = []
        newlist = slices_remove_small_slices(slicelist, count=5)
        expected = []
        self.assertEqual(expected, newlist)

        slicelist = None
        newlist = slices_remove_small_slices(slicelist, count=5)
        expected = None
        self.assertEqual(expected, newlist)

class TestSlicesNot(unittest.TestCase):
    def test_slices_not_internal(self):
        slice_list = [slice(10,13),slice(16,25)]
        self.assertEqual(slices_not(slice_list), [slice(13,16)])

    def test_slices_not_extended(self):
        slice_list = [slice(10,13)]
        self.assertEqual(slices_not(slice_list, begin_at=2, end_at=18),
                         [slice(2,10),slice(13,18)])
        self.assertEqual(slices_not(slice_list, begin_at=0, end_at=18),
                         [slice(0,10),slice(13,18)])
        self.assertEqual(slices_not(slice_list, begin_at=None, end_at=18),
                         [slice(13,18),])

    def test_slices_not_extended_2(self):
        slice_list = [slice(None, None, None)]
        self.assertEqual(slices_not(slice_list, begin_at=0, end_at=10), [])
        self.assertEqual(slices_not(slice_list, begin_at=None, end_at=10), [])

    def test_slices_not_to_none(self):
        slice_list = [slice(10,None),slice(2,3)]
        self.assertEqual(slices_not(slice_list),[slice(3,10)])

    def test_slices_not_to_none_empty(self):
        slice_list = [slice(10,None)]
        self.assertEqual(slices_not(slice_list),[])

    def test_slices_not_from_none_empty(self):
        slice_list = [slice(None,13)]
        self.assertEqual(slices_not(slice_list),[])

    def test_slices_not_from_none(self):
        slice_list = [slice(None,13),slice(15,20)]
        self.assertEqual(slices_not(slice_list),[slice(13,15)])

    def test_slices_not_null(self):
        self.assertEqual(slices_not(None), [slice(None,None,None)])
        self.assertEqual(slices_not([]), [slice(None,None,None)])
        self.assertEqual(slices_not([slice(4,6)]),[])

    def test_slices_misordered(self):
        slice_list = [slice(25,16,-1),slice(10,13)]
        self.assertEqual(slices_not(slice_list), [slice(13,17)])

    def test_slices_not_error(self):
        slice_list = [slice(1,5,2)]
        self.assertRaises(ValueError, slices_not, slice_list)

    def test_slices_not_return_float(self):
        slice_list = [slice(2060.2265625, 39992.2265625, None)]
        begin_at = 956.078125
        end_at = 40800.078125
        result = slices_not(slice_list, begin_at, end_at)
        expected_result = [
            slice(956.078125, 2060.078125, None),
            slice(39992.078125, 40800.078125, None)
        ]
        self.assertEqual(result, expected_result)


class TestSlicesAndNot(unittest.TestCase):

    def test_slices_and_not_return_float(self):
        mobiles_slices = [slice(956.078125, 40800.078125, None)]
        fasts_slices = [slice(2060.2265625, 39992.2265625, None)]
        result = slices_and_not(mobiles_slices, fasts_slices)
        expected_result = [
            slice(956.078125, 2060.078125, None),
            slice(39992.078125, 40800.078125, None)
        ]
        self.assertEqual(result, expected_result)


class TestSlicesOr(unittest.TestCase):

    def test_slices_or_single_list(self):
        '''
        The contents of a single list should be OR'd.
        '''
        slices = [slice(None, 15), slice(25, 45), slice(65, None)]
        self.assertEqual(slices_or(slices), slices)

    def test_slices_or_with_overlap(self):
        slice_list_a = [slice(10,13)]
        slice_list_b = [slice(16,25)]
        slice_list_c = [slice(20,31)]
        self.assertEqual(slices_or(slice_list_a,
                                   slice_list_b,
                                   slice_list_c),
                         [slice(10,13), slice(16,31)])

    def test_slices_or_lists(self):
        slice_list_a = [slice(10,13), slice(16,25)]
        slice_list_b = [slice(20,31)]
        self.assertEqual(slices_or(slice_list_a,
                                   slice_list_b),
                         [slice(10,13), slice(16,31)])

    def test_slices_or_empty_first_list(self):
        slice_list_a = []
        slice_list_b = [slice(1,3)]
        self.assertEqual(slices_or(slice_list_a, slice_list_b),
                         [slice(1, 3)])

    def test_slices_or_one_list(self):
        self.assertEqual(slices_or([slice(1,2)]), [slice(1,2)])

    def test_slices_or_with_none(self):
        result = slices_or([None], [slice(1, 3)])
        self.assertEqual(result, [slice(1, 3)])
        result = slices_or([slice(1, 3)], [None])
        self.assertEqual(result, [slice(1, 3)])
        result = slices_or([None])
        self.assertEqual(result, [])

    def test_slices_or_open_range(self):
        slice_list_a = [slice(2, 10)]
        slice_list_b = [slice(None, 4), slice(7, 9)]
        slice_list_c = [slice(6, None)]
        self.assertEqual(slices_or(slice_list_a, slice_list_b, slice_list_c),
                         [slice(None, None)])

    def test_slices_or_real_example(self):
        # This comes from a flight where two radio altimeters were starting
        # to operate during the descent. The original code produced an answer
        # [slice(0.0, 394, None), slice(979.0, 1198, None), slice(972.0, 1198, None)]
        # inerror.
        self.assertEqual(slices_or([slice(0.0, 394, None),
                                     slice(975.0, 980, None),
                                     slice(984.0, 1198, None)],
                                    [slice(0.0, 394, None),
                                     slice(972.0, 977, None),
                                     slice(979.0, 1198, None)]),
                                   [slice(0,394), slice(972,1198)])

    def test_infinite_loop(self):
        # This slice list caused the program to stick in a loop for one version.
        slice_list = [slice(10.0, 13, None), slice(14.0, 17, None), slice(18.0, 21, None), slice(22.0, 25, None), slice(40.0, 43, None), slice(44.0, 47, None), slice(48.0, 51, None), slice(52.0, 55, None), slice(56.0, 59, None), slice(60.0, 63, None), slice(64.0, 67, None), slice(68.0, 71, None), slice(72.0, 75, None), slice(76.0, 79, None), slice(80.0, 83, None), slice(84.0, 87, None), slice(88.0, 91, None), slice(92.0, 95, None), slice(96.0, 99, None), slice(100.0, 103, None), slice(104.0, 107, None), slice(108.0, 111, None), slice(112.0, 115, None), slice(116.0, 119, None), slice(120.0, 123, None), slice(124.0, 127, None), slice(128.0, 131, None), slice(132.0, 135, None), slice(136.0, 139, None), slice(140.0, 143, None), slice(144.0, 147, None), slice(148.0, 151, None), slice(152.0, 155, None), slice(156.0, 159, None), slice(160.0, 163, None), slice(164.0, 167, None), slice(168.0, 171, None), slice(172.0, 175, None), slice(176.0, 179, None), slice(180.0, 183, None), slice(184.0, 187, None), slice(188.0, 191, None), slice(192.0, 195, None), slice(196.0, 199, None), slice(200.0, 203, None), slice(204.0, 207, None), slice(208.0, 211, None), slice(212.0, 215, None), slice(216.0, 219, None), slice(220.0, 223, None), slice(224.0, 227, None), slice(228.0, 231, None), slice(232.0, 235, None), slice(236.0, 239, None), slice(240.0, 243, None), slice(244.0, 247, None), slice(248.0, 251, None), slice(252.0, 255, None), slice(256.0, 259, None), slice(260.0, 263, None), slice(264.0, 267, None), slice(268.0, 271, None), slice(272.0, 275, None), slice(276.0, 279, None), slice(280.0, 283, None), slice(284.0, 287, None), slice(288.0, 291, None), slice(292.0, 295, None), slice(296.0, 299, None), slice(300.0, 303, None), slice(304.0, 307, None), slice(308.0, 311, None), slice(312.0, 315, None), slice(316.0, 319, None), slice(320.0, 323, None), slice(324.0, 327, None), slice(328.0, 331, None), slice(332.0, 335, None), slice(336.0, 339, None), slice(340.0, 343, None), slice(344.0, 347, None), slice(348.0, 351, None), slice(352.0, 355, None), slice(356.0, 359, None), slice(360.0, 363, None), slice(364.0, 367, None), slice(368.0, 371, None), slice(372.0, 375, None), slice(376.0, 379, None), slice(380.0, 383, None), slice(384.0, 387, None), slice(388.0, 391, None), slice(392.0, 395, None), slice(396.0, 399, None), slice(400.0, 403, None), slice(404.0, 407, None), slice(408.0, 411, None), slice(412.0, 415, None), slice(416.0, 419, None), slice(420.0, 423, None), slice(424.0, 427, None), slice(428.0, 431, None), slice(432.0, 435, None), slice(436.0, 439, None), slice(440.0, 443, None), slice(444.0, 447, None), slice(448.0, 451, None), slice(452.0, 455, None), slice(456.0, 459, None), slice(460.0, 463, None), slice(464.0, 467, None), slice(468.0, 471, None), slice(472.0, 475, None), slice(476.0, 479, None), slice(480.0, 483, None), slice(484.0, 487, None), slice(488.0, 491, None), slice(492.0, 495, None), slice(496.0, 499, None), slice(500.0, 503, None), slice(504.0, 507, None), slice(508.0, 511, None), slice(512.0, 515, None), slice(516.0, 519, None), slice(520.0, 523, None), slice(524.0, 527, None), slice(528.0, 531, None), slice(532.0, 535, None), slice(536.0, 539, None), slice(540.0, 543, None), slice(544.0, 547, None), slice(548.0, 551, None), slice(552.0, 555, None), slice(556.0, 559, None), slice(560.0, 563, None), slice(564.0, 567, None), slice(568.0, 571, None), slice(572.0, 575, None), slice(576.0, 579, None), slice(580.0, 583, None), slice(584.0, 587, None), slice(588.0, 591, None), slice(592.0, 595, None), slice(596.0, 599, None), slice(600.0, 603, None), slice(604.0, 607, None), slice(608.0, 611, None), slice(612.0, 615, None), slice(616.0, 619, None), slice(620.0, 623, None), slice(624.0, 627, None), slice(628.0, 631, None), slice(632.0, 635, None), slice(636.0, 639, None), slice(640.0, 643, None), slice(644.0, 647, None), slice(648.0, 651, None), slice(652.0, 655, None), slice(656.0, 659, None), slice(660.0, 663, None), slice(664.0, 667, None), slice(668.0, 671, None), slice(672.0, 675, None), slice(676.0, 679, None), slice(680.0, 683, None), slice(684.0, 687, None), slice(688.0, 691, None), slice(692.0, 695, None), slice(696.0, 699, None), slice(700.0, 703, None), slice(704.0, 707, None), slice(708.0, 711, None), slice(712.0, 715, None), slice(716.0, 719, None), slice(720.0, 723, None), slice(724.0, 727, None), slice(728.0, 731, None), slice(732.0, 735, None), slice(736.0, 739, None), slice(740.0, 743, None), slice(744.0, 747, None), slice(748.0, 751, None), slice(752.0, 755, None), slice(756.0, 759, None), slice(760.0, 763, None), slice(764.0, 767, None), slice(768.0, 771, None), slice(772.0, 775, None), slice(776.0, 779, None), slice(780.0, 783, None), slice(784.0, 787, None), slice(788.0, 791, None), slice(792.0, 795, None), slice(796.0, 799, None), slice(7620.0, 7623, None), slice(7624.0, 7627, None), slice(7628.0, 7631, None), slice(7632.0, 7635, None), slice(7636.0, 7639, None), slice(7640.0, 7643, None), slice(7644.0, 7647, None), slice(7648.0, 7651, None), slice(7652.0, 7655, None), slice(7656.0, 7659, None), slice(7660.0, 7663, None), slice(7664.0, 7667, None), slice(7668.0, 7671, None), slice(7672.0, 7675, None), slice(7676.0, 7679, None), slice(7680.0, 7683, None), slice(7684.0, 7687, None), slice(7688.0, 7691, None), slice(7692.0, 7695, None), slice(7696.0, 7699, None), slice(7700.0, 7703, None), slice(7704.0, 7707, None), slice(7708.0, 7711, None), slice(7712.0, 7715, None), slice(7716.0, 7719, None), slice(7720.0, 7723, None), slice(7724.0, 7727, None), slice(7728.0, 7731, None), slice(7732.0, 7735, None), slice(7736.0, 7739, None), slice(7740.0, 7743, None), slice(7744.0, 7747, None), slice(7748.0, 7751, None), slice(7752.0, 7755, None), slice(7756.0, 7759, None), slice(7760.0, 7763, None), slice(7764.0, 7767, None), slice(7768.0, 7771, None), slice(7772.0, 7775, None), slice(7776.0, 7779, None), slice(7780.0, 7783, None), slice(7784.0, 7787, None), slice(7788.0, 7791, None), slice(7792.0, 7795, None), slice(7796.0, 7799, None), slice(7800.0, 7803, None), slice(7804.0, 7807, None), slice(7808.0, 7811, None), slice(7812.0, 7815, None), slice(7816.0, 7819, None), slice(7820.0, 7823, None), slice(7824.0, 7827, None), slice(7828.0, 7831, None), slice(7832.0, 7835, None), slice(7836.0, 7839, None), slice(7840.0, 7843, None), slice(7844.0, 7847, None), slice(7848.0, 7851, None), slice(7852.0, 7855, None), slice(7856.0, 7859, None), slice(7860.0, 7863, None), slice(7864.0, 7867, None), slice(7868.0, 7871, None), slice(7872.0, 7875, None), slice(7876.0, 7879, None), slice(7880.0, 7883, None), slice(7884.0, 7887, None), slice(7888.0, 7891, None), slice(7892.0, 7895, None), slice(7896.0, 7899, None), slice(7900.0, 7903, None), slice(7904.0, 7907, None), slice(7908.0, 7911, None), slice(7912.0, 7915, None), slice(7916.0, 7919, None), slice(7920.0, 7923, None), slice(7924.0, 7927, None), slice(7928.0, 7931, None), slice(7932.0, 7935, None), slice(7936.0, 7939, None), slice(7940.0, 7943, None), slice(7944.0, 7946, None), slice(7958.0, 7966, None), slice(7967.0, 7970, None), slice(7971.0, 7974, None), slice(7975.0, 7978, None), slice(7986.0, 8000, None), slice(8001.0, 8004, None), slice(8005.0, 8008, None), slice(8009.0, 8014, None), slice(8015.0, 8029, None), slice(8030.0, 8033, None), slice(8034.0, 8037, None), slice(8038.0, 8041, None), slice(8042.0, 8045, None), slice(8046.0, 8049, None), slice(8050.0, 8053, None), slice(8054.0, 8057, None), slice(8058.0, 8061, None), slice(8062.0, 8065, None), slice(8066.0, 8069, None), slice(8070.0, 8073, None), slice(8074.0, 8112, None), slice(8114.0, 8116, None), slice(8146.0, 8212, None), slice(8213.0, 8219, None), slice(8220.0, 8223, None), slice(8224.0, 8227, None), slice(8228.0, 8230, None), slice(8241.0, 8243, None), slice(8244.0, 8247, None), slice(8248.0, 8251, None), slice(8252.0, 8254, None), slice(8261.0, 8263, None), slice(8264.0, 8267, None), slice(8268.0, 8271, None), slice(8272.0, 8275, None), slice(8276.0, 8279, None), slice(8280.0, 8283, None), slice(8284.0, 8287, None), slice(8288.0, 8291, None), slice(8292.0, 8295, None), slice(8296.0, 8298, None), slice(10.0, 13, None), slice(14.0, 17, None), slice(18.0, 21, None), slice(22.0, 25, None), slice(28.0, 31, None), slice(32.0, 35, None), slice(36.0, 39, None), slice(40.0, 43, None), slice(44.0, 47, None), slice(48.0, 51, None), slice(52.0, 55, None), slice(56.0, 59, None), slice(60.0, 63, None), slice(64.0, 67, None), slice(68.0, 71, None), slice(72.0, 75, None), slice(76.0, 79, None), slice(80.0, 83, None), slice(84.0, 87, None), slice(88.0, 91, None), slice(92.0, 95, None), slice(96.0, 99, None), slice(100.0, 103, None), slice(104.0, 107, None), slice(108.0, 111, None), slice(112.0, 115, None), slice(116.0, 119, None), slice(120.0, 123, None), slice(124.0, 127, None), slice(128.0, 131, None), slice(132.0, 135, None), slice(136.0, 139, None), slice(140.0, 143, None), slice(144.0, 147, None), slice(148.0, 151, None), slice(152.0, 155, None), slice(156.0, 159, None), slice(160.0, 163, None), slice(164.0, 167, None), slice(168.0, 171, None), slice(172.0, 175, None), slice(176.0, 179, None), slice(180.0, 183, None), slice(184.0, 187, None), slice(188.0, 191, None), slice(192.0, 195, None), slice(196.0, 199, None), slice(200.0, 203, None), slice(204.0, 207, None), slice(208.0, 211, None), slice(212.0, 215, None), slice(216.0, 219, None), slice(220.0, 223, None), slice(224.0, 227, None), slice(228.0, 231, None), slice(232.0, 235, None), slice(236.0, 239, None), slice(240.0, 243, None), slice(244.0, 247, None), slice(248.0, 251, None), slice(252.0, 255, None), slice(256.0, 259, None), slice(260.0, 263, None), slice(264.0, 267, None), slice(268.0, 271, None), slice(272.0, 275, None), slice(276.0, 279, None), slice(280.0, 283, None), slice(284.0, 287, None), slice(288.0, 291, None), slice(292.0, 295, None), slice(296.0, 299, None), slice(300.0, 303, None), slice(304.0, 307, None), slice(308.0, 311, None), slice(312.0, 315, None), slice(316.0, 319, None), slice(320.0, 323, None), slice(324.0, 327, None), slice(328.0, 331, None), slice(332.0, 335, None), slice(336.0, 339, None), slice(340.0, 343, None), slice(344.0, 347, None), slice(348.0, 351, None), slice(352.0, 355, None), slice(356.0, 359, None), slice(360.0, 363, None), slice(364.0, 367, None), slice(368.0, 371, None), slice(372.0, 375, None), slice(376.0, 379, None), slice(380.0, 383, None), slice(384.0, 387, None), slice(388.0, 391, None), slice(392.0, 395, None), slice(396.0, 399, None), slice(400.0, 403, None), slice(404.0, 407, None), slice(408.0, 411, None), slice(412.0, 415, None), slice(416.0, 419, None), slice(420.0, 423, None), slice(424.0, 427, None), slice(428.0, 431, None), slice(432.0, 435, None), slice(436.0, 439, None), slice(440.0, 443, None), slice(444.0, 447, None), slice(448.0, 451, None), slice(452.0, 455, None), slice(456.0, 459, None), slice(460.0, 463, None), slice(464.0, 467, None), slice(468.0, 471, None), slice(472.0, 475, None), slice(476.0, 479, None), slice(480.0, 483, None), slice(484.0, 487, None), slice(488.0, 491, None), slice(492.0, 495, None), slice(496.0, 499, None), slice(500.0, 503, None), slice(504.0, 507, None), slice(508.0, 511, None), slice(512.0, 515, None), slice(516.0, 519, None), slice(520.0, 523, None), slice(524.0, 527, None), slice(528.0, 531, None), slice(532.0, 535, None), slice(536.0, 539, None), slice(540.0, 543, None), slice(544.0, 547, None), slice(548.0, 551, None), slice(552.0, 555, None), slice(556.0, 559, None), slice(560.0, 563, None), slice(564.0, 567, None), slice(568.0, 570, None), slice(581.0, 583, None), slice(584.0, 587, None), slice(588.0, 591, None), slice(592.0, 595, None), slice(596.0, 598, None), slice(600.0, 603, None), slice(604.0, 606, None), slice(609.0, 611, None), slice(612.0, 614, None), slice(617.0, 619, None), slice(621.0, 623, None), slice(624.0, 627, None), slice(628.0, 630, None), slice(632.0, 635, None), slice(636.0, 639, None), slice(640.0, 643, None), slice(644.0, 647, None), slice(648.0, 651, None), slice(652.0, 655, None), slice(656.0, 659, None), slice(660.0, 663, None), slice(664.0, 667, None), slice(668.0, 671, None), slice(672.0, 675, None), slice(676.0, 679, None), slice(680.0, 683, None), slice(684.0, 687, None), slice(688.0, 691, None), slice(692.0, 695, None), slice(696.0, 699, None), slice(700.0, 703, None), slice(704.0, 707, None), slice(708.0, 711, None), slice(712.0, 715, None), slice(716.0, 719, None), slice(720.0, 723, None), slice(724.0, 727, None), slice(728.0, 731, None), slice(732.0, 735, None), slice(736.0, 739, None), slice(740.0, 743, None), slice(744.0, 747, None), slice(748.0, 751, None), slice(752.0, 755, None), slice(756.0, 759, None), slice(760.0, 763, None), slice(764.0, 767, None), slice(768.0, 771, None), slice(772.0, 775, None), slice(776.0, 779, None), slice(780.0, 783, None), slice(784.0, 787, None), slice(788.0, 791, None), slice(792.0, 795, None), slice(796.0, 799, None), slice(7620.0, 7623, None), slice(7624.0, 7627, None), slice(7628.0, 7631, None), slice(7632.0, 7635, None), slice(7636.0, 7639, None), slice(7640.0, 7643, None), slice(7644.0, 7647, None), slice(7648.0, 7651, None), slice(7652.0, 7655, None), slice(7656.0, 7659, None), slice(7660.0, 7663, None), slice(7664.0, 7667, None), slice(7668.0, 7671, None), slice(7672.0, 7675, None), slice(7676.0, 7679, None), slice(7680.0, 7683, None), slice(7684.0, 7687, None), slice(7688.0, 7691, None), slice(7692.0, 7695, None), slice(7696.0, 7699, None), slice(7700.0, 7703, None), slice(7704.0, 7707, None), slice(7708.0, 7711, None), slice(7712.0, 7715, None), slice(7716.0, 7719, None), slice(7720.0, 7723, None), slice(7724.0, 7727, None), slice(7728.0, 7731, None), slice(7732.0, 7735, None), slice(7736.0, 7739, None), slice(7740.0, 7743, None), slice(7744.0, 7747, None), slice(7748.0, 7751, None), slice(7752.0, 7755, None), slice(7756.0, 7759, None), slice(7760.0, 7763, None), slice(7764.0, 7767, None), slice(7768.0, 7771, None), slice(7772.0, 7775, None), slice(7776.0, 7779, None), slice(7780.0, 7783, None), slice(7784.0, 7787, None), slice(7788.0, 7791, None), slice(7792.0, 7795, None), slice(7796.0, 7799, None), slice(7800.0, 7803, None), slice(7804.0, 7807, None), slice(7808.0, 7811, None), slice(7812.0, 7815, None), slice(7816.0, 7819, None), slice(7820.0, 7823, None), slice(7824.0, 7827, None), slice(7828.0, 7831, None), slice(7832.0, 7835, None), slice(7836.0, 7839, None), slice(7840.0, 7843, None), slice(7844.0, 7847, None), slice(7848.0, 7851, None), slice(7852.0, 7855, None), slice(7856.0, 7859, None), slice(7860.0, 7863, None), slice(7864.0, 7867, None), slice(7868.0, 7871, None), slice(7872.0, 7875, None), slice(7876.0, 7879, None), slice(7880.0, 7883, None), slice(7884.0, 7887, None), slice(7888.0, 7891, None), slice(7892.0, 7895, None), slice(7896.0, 7899, None), slice(7900.0, 7903, None), slice(7904.0, 7907, None), slice(7908.0, 7911, None), slice(7912.0, 7915, None), slice(7916.0, 7919, None), slice(7920.0, 7923, None), slice(7924.0, 7927, None), slice(7928.0, 7931, None), slice(7932.0, 7935, None), slice(7936.0, 7939, None), slice(7940.0, 7943, None), slice(7944.0, 7947, None), slice(7953.0, 7955, None), slice(7956.0, 7959, None), slice(7960.0, 7963, None), slice(7964.0, 7967, None), slice(7968.0, 7971, None), slice(7972.0, 7975, None), slice(7976.0, 7979, None), slice(7985.0, 7998, None), slice(7999.0, 8002, None), slice(8003.0, 8006, None), slice(8007.0, 8010, None), slice(8011.0, 8026, None), slice(8027.0, 8030, None), slice(8031.0, 8034, None), slice(8035.0, 8038, None), slice(8039.0, 8042, None), slice(8043.0, 8046, None), slice(8047.0, 8050, None), slice(8051.0, 8054, None), slice(8055.0, 8058, None), slice(8059.0, 8062, None), slice(8063.0, 8066, None), slice(8067.0, 8070, None), slice(8071.0, 8073, None), slice(8074.0, 8094, None), slice(8095.0, 8098, None), slice(8099.0, 8102, None), slice(8103.0, 8113, None), slice(8146.0, 8211, None), slice(8213.0, 8220, None), slice(8221.0, 8224, None), slice(8225.0, 8228, None), slice(8229.0, 8232, None), slice(8233.0, 8236, None), slice(8237.0, 8240, None), slice(8241.0, 8244, None), slice(8245.0, 8248, None), slice(8249.0, 8252, None), slice(8253.0, 8256, None), slice(8257.0, 8260, None), slice(8261.0, 8264, None), slice(8265.0, 8268, None), slice(8269.0, 8272, None), slice(8273.0, 8276, None), slice(8277.0, 8280, None), slice(8281.0, 8284, None), slice(8285.0, 8288, None), slice(8289.0, 8292, None), slice(8293.0, 8296, None), slice(10.0, 13, None), slice(14.0, 17, None), slice(18.0, 21, None), slice(22.0, 25, None), slice(28.0, 31, None), slice(32.0, 35, None), slice(36.0, 39, None), slice(40.0, 43, None), slice(44.0, 47, None), slice(48.0, 51, None), slice(52.0, 55, None), slice(56.0, 59, None), slice(60.0, 63, None), slice(64.0, 67, None), slice(68.0, 71, None), slice(72.0, 75, None), slice(76.0, 79, None), slice(80.0, 83, None), slice(84.0, 87, None), slice(88.0, 91, None), slice(92.0, 95, None), slice(96.0, 99, None), slice(100.0, 103, None), slice(104.0, 107, None), slice(108.0, 111, None), slice(112.0, 115, None), slice(116.0, 119, None), slice(120.0, 123, None), slice(124.0, 127, None), slice(128.0, 131, None), slice(132.0, 135, None), slice(136.0, 139, None), slice(140.0, 143, None), slice(144.0, 147, None), slice(148.0, 151, None), slice(152.0, 155, None), slice(156.0, 159, None), slice(160.0, 163, None), slice(164.0, 167, None), slice(168.0, 171, None), slice(172.0, 175, None), slice(176.0, 179, None), slice(180.0, 183, None), slice(184.0, 187, None), slice(188.0, 191, None), slice(192.0, 195, None), slice(196.0, 199, None), slice(200.0, 203, None), slice(204.0, 207, None), slice(208.0, 211, None), slice(212.0, 215, None), slice(216.0, 219, None), slice(220.0, 223, None), slice(224.0, 227, None), slice(228.0, 231, None), slice(232.0, 235, None), slice(236.0, 239, None), slice(240.0, 243, None), slice(244.0, 247, None), slice(248.0, 251, None), slice(252.0, 255, None), slice(256.0, 259, None), slice(260.0, 263, None), slice(264.0, 267, None), slice(268.0, 271, None), slice(272.0, 275, None), slice(276.0, 279, None), slice(280.0, 283, None), slice(284.0, 287, None), slice(288.0, 291, None), slice(292.0, 295, None), slice(296.0, 299, None), slice(300.0, 303, None), slice(304.0, 307, None), slice(308.0, 311, None), slice(312.0, 315, None), slice(316.0, 319, None), slice(320.0, 323, None), slice(324.0, 327, None), slice(328.0, 331, None), slice(332.0, 335, None), slice(336.0, 339, None), slice(340.0, 343, None), slice(344.0, 347, None), slice(348.0, 351, None), slice(352.0, 355, None), slice(356.0, 359, None), slice(360.0, 363, None), slice(364.0, 367, None), slice(368.0, 371, None), slice(372.0, 375, None), slice(376.0, 379, None), slice(380.0, 383, None), slice(384.0, 387, None), slice(388.0, 391, None), slice(392.0, 395, None), slice(396.0, 399, None), slice(400.0, 403, None), slice(404.0, 407, None), slice(408.0, 411, None), slice(412.0, 415, None), slice(416.0, 419, None), slice(420.0, 423, None), slice(424.0, 427, None), slice(428.0, 431, None), slice(432.0, 435, None), slice(436.0, 439, None), slice(440.0, 443, None), slice(444.0, 447, None), slice(448.0, 451, None), slice(452.0, 455, None), slice(456.0, 459, None), slice(460.0, 463, None), slice(464.0, 467, None), slice(468.0, 471, None), slice(472.0, 475, None), slice(476.0, 479, None), slice(480.0, 483, None), slice(484.0, 487, None), slice(488.0, 491, None), slice(492.0, 495, None), slice(496.0, 499, None), slice(500.0, 503, None), slice(504.0, 507, None), slice(508.0, 511, None), slice(512.0, 515, None), slice(516.0, 519, None), slice(520.0, 523, None), slice(524.0, 527, None), slice(528.0, 531, None), slice(532.0, 535, None), slice(536.0, 539, None), slice(540.0, 543, None), slice(544.0, 547, None), slice(548.0, 551, None), slice(552.0, 555, None), slice(556.0, 559, None), slice(560.0, 563, None), slice(564.0, 567, None), slice(568.0, 570, None), slice(581.0, 583, None), slice(584.0, 587, None), slice(588.0, 591, None), slice(592.0, 595, None), slice(600.0, 603, None), slice(604.0, 606, None), slice(609.0, 611, None), slice(612.0, 614, None), slice(616.0, 619, None), slice(621.0, 623, None), slice(624.0, 627, None), slice(628.0, 630, None), slice(632.0, 635, None), slice(636.0, 639, None), slice(640.0, 643, None), slice(644.0, 647, None), slice(648.0, 651, None), slice(652.0, 655, None), slice(656.0, 659, None), slice(660.0, 663, None), slice(664.0, 667, None), slice(668.0, 671, None), slice(672.0, 675, None), slice(676.0, 679, None), slice(680.0, 683, None), slice(684.0, 687, None), slice(688.0, 691, None), slice(692.0, 695, None), slice(696.0, 699, None), slice(700.0, 703, None), slice(704.0, 707, None), slice(708.0, 711, None), slice(712.0, 715, None), slice(716.0, 719, None), slice(720.0, 723, None), slice(724.0, 727, None), slice(728.0, 731, None), slice(732.0, 735, None), slice(736.0, 739, None), slice(740.0, 743, None), slice(744.0, 747, None), slice(748.0, 751, None), slice(752.0, 755, None), slice(756.0, 759, None), slice(760.0, 763, None), slice(764.0, 767, None), slice(768.0, 771, None), slice(772.0, 775, None), slice(776.0, 779, None), slice(780.0, 783, None), slice(784.0, 787, None), slice(788.0, 791, None), slice(792.0, 795, None), slice(796.0, 799, None), slice(7620.0, 7623, None), slice(7624.0, 7627, None), slice(7628.0, 7631, None), slice(7632.0, 7635, None), slice(7636.0, 7639, None), slice(7640.0, 7643, None), slice(7644.0, 7647, None), slice(7648.0, 7651, None), slice(7652.0, 7655, None), slice(7656.0, 7659, None), slice(7660.0, 7663, None), slice(7664.0, 7667, None), slice(7668.0, 7671, None), slice(7672.0, 7675, None), slice(7676.0, 7679, None), slice(7680.0, 7683, None), slice(7684.0, 7687, None), slice(7688.0, 7691, None), slice(7692.0, 7695, None), slice(7696.0, 7699, None), slice(7700.0, 7703, None), slice(7704.0, 7707, None), slice(7708.0, 7711, None), slice(7712.0, 7715, None), slice(7716.0, 7719, None), slice(7720.0, 7723, None), slice(7724.0, 7727, None), slice(7728.0, 7731, None), slice(7732.0, 7735, None), slice(7736.0, 7739, None), slice(7740.0, 7743, None), slice(7744.0, 7747, None), slice(7748.0, 7751, None), slice(7752.0, 7755, None), slice(7756.0, 7759, None), slice(7760.0, 7763, None), slice(7764.0, 7767, None), slice(7768.0, 7771, None), slice(7772.0, 7775, None), slice(7776.0, 7779, None), slice(7780.0, 7783, None), slice(7784.0, 7787, None), slice(7788.0, 7791, None), slice(7792.0, 7795, None), slice(7796.0, 7799, None), slice(7800.0, 7803, None), slice(7804.0, 7807, None), slice(7808.0, 7811, None), slice(7812.0, 7815, None), slice(7816.0, 7819, None), slice(7820.0, 7823, None), slice(7824.0, 7827, None), slice(7828.0, 7831, None), slice(7832.0, 7835, None), slice(7836.0, 7839, None), slice(7840.0, 7843, None), slice(7844.0, 7847, None), slice(7848.0, 7851, None), slice(7852.0, 7855, None), slice(7856.0, 7859, None), slice(7860.0, 7863, None), slice(7864.0, 7867, None), slice(7868.0, 7871, None), slice(7872.0, 7875, None), slice(7876.0, 7879, None), slice(7880.0, 7883, None), slice(7884.0, 7887, None), slice(7888.0, 7891, None), slice(7892.0, 7895, None), slice(7896.0, 7899, None), slice(7900.0, 7903, None), slice(7904.0, 7907, None), slice(7908.0, 7911, None), slice(7912.0, 7915, None), slice(7916.0, 7919, None), slice(7920.0, 7923, None), slice(7924.0, 7927, None), slice(7928.0, 7931, None), slice(7932.0, 7935, None), slice(7936.0, 7939, None), slice(7940.0, 7943, None), slice(7944.0, 7947, None), slice(7948.0, 7951, None), slice(7952.0, 7955, None), slice(7956.0, 7959, None), slice(7960.0, 7963, None), slice(7964.0, 7967, None), slice(7968.0, 7971, None), slice(7972.0, 7975, None), slice(7976.0, 7979, None), slice(7980.0, 7983, None), slice(7985.0, 7993, None), slice(7994.0, 7997, None), slice(7998.0, 8001, None), slice(8002.0, 8005, None), slice(8006.0, 8009, None), slice(8010.0, 8018, None), slice(8019.0, 8022, None), slice(8023.0, 8026, None), slice(8027.0, 8030, None), slice(8031.0, 8034, None), slice(8035.0, 8038, None), slice(8039.0, 8042, None), slice(8043.0, 8046, None), slice(8047.0, 8050, None), slice(8051.0, 8054, None), slice(8055.0, 8058, None), slice(8059.0, 8062, None), slice(8063.0, 8066, None), slice(8067.0, 8070, None), slice(8071.0, 8074, None), slice(8075.0, 8088, None), slice(8089.0, 8092, None), slice(8093.0, 8096, None), slice(8097.0, 8100, None), slice(8101.0, 8104, None), slice(8105.0, 8108, None), slice(8109.0, 8112, None), slice(8146.0, 8211, None), slice(8212.0, 8216, None), slice(8217.0, 8220, None), slice(8221.0, 8224, None), slice(8225.0, 8228, None), slice(8229.0, 8232, None), slice(8233.0, 8236, None), slice(8237.0, 8240, None), slice(8241.0, 8244, None), slice(8245.0, 8248, None), slice(8249.0, 8252, None), slice(8253.0, 8256, None), slice(8257.0, 8260, None), slice(8261.0, 8264, None), slice(8265.0, 8268, None), slice(8269.0, 8272, None), slice(8273.0, 8276, None), slice(8277.0, 8280, None), slice(8281.0, 8284, None), slice(8285.0, 8288, None), slice(8289.0, 8292, None), slice(8293.0, 8296, None), slice(28.0, 32, None), slice(33.0, 36, None), slice(7118.0, 7121, None), slice(7958.0, 7965, None), slice(7966.0, 7969, None), slice(7970.0, 7973, None), slice(7974.0, 7977, None), slice(7978.0, 7981, None), slice(7986.0, 8000, None), slice(8001.0, 8004, None), slice(8005.0, 8008, None), slice(8009.0, 8015, None), slice(8016.0, 8030, None), slice(8031.0, 8034, None), slice(8035.0, 8038, None), slice(8039.0, 8042, None), slice(8043.0, 8046, None), slice(8047.0, 8050, None), slice(8051.0, 8054, None), slice(8055.0, 8058, None), slice(8059.0, 8062, None), slice(8063.0, 8066, None), slice(8067.0, 8070, None), slice(8071.0, 8073, None), slice(8074.0, 8112, None), slice(8146.0, 8220, None), slice(8221.0, 8224, None), slice(8225.0, 8228, None), slice(8229.0, 8232, None), slice(8233.0, 8236, None), slice(8237.0, 8240, None), slice(8241.0, 8244, None), slice(8245.0, 8248, None), slice(8249.0, 8252, None), slice(8253.0, 8256, None), slice(8257.0, 8260, None), slice(8261.0, 8264, None), slice(8265.0, 8268, None), slice(8269.0, 8272, None), slice(8273.0, 8276, None), slice(8277.0, 8280, None), slice(8281.0, 8284, None), slice(8285.0, 8288, None), slice(8289.0, 8292, None), slice(8293.0, 8296, None)]
        result = slices_or(slice_list)
        expected = [slice(572.0, 575, None), slice(576.0, 579, None), slice(8114.0, 8116, None), slice(8296.0, 8298, None), slice(596.0, 599, None), slice(10.0, 13, None), slice(14.0, 17, None), slice(18.0, 21, None), slice(22.0, 25, None), slice(36.0, 39, None), slice(40.0, 43, None), slice(44.0, 47, None), slice(48.0, 51, None), slice(52.0, 55, None), slice(56.0, 59, None), slice(60.0, 63, None), slice(64.0, 67, None), slice(68.0, 71, None), slice(72.0, 75, None), slice(76.0, 79, None), slice(80.0, 83, None), slice(84.0, 87, None), slice(88.0, 91, None), slice(92.0, 95, None), slice(96.0, 99, None), slice(100.0, 103, None), slice(104.0, 107, None), slice(108.0, 111, None), slice(112.0, 115, None), slice(116.0, 119, None), slice(120.0, 123, None), slice(124.0, 127, None), slice(128.0, 131, None), slice(132.0, 135, None), slice(136.0, 139, None), slice(140.0, 143, None), slice(144.0, 147, None), slice(148.0, 151, None), slice(152.0, 155, None), slice(156.0, 159, None), slice(160.0, 163, None), slice(164.0, 167, None), slice(168.0, 171, None), slice(172.0, 175, None), slice(176.0, 179, None), slice(180.0, 183, None), slice(184.0, 187, None), slice(188.0, 191, None), slice(192.0, 195, None), slice(196.0, 199, None), slice(200.0, 203, None), slice(204.0, 207, None), slice(208.0, 211, None), slice(212.0, 215, None), slice(216.0, 219, None), slice(220.0, 223, None), slice(224.0, 227, None), slice(228.0, 231, None), slice(232.0, 235, None), slice(236.0, 239, None), slice(240.0, 243, None), slice(244.0, 247, None), slice(248.0, 251, None), slice(252.0, 255, None), slice(256.0, 259, None), slice(260.0, 263, None), slice(264.0, 267, None), slice(268.0, 271, None), slice(272.0, 275, None), slice(276.0, 279, None), slice(280.0, 283, None), slice(284.0, 287, None), slice(288.0, 291, None), slice(292.0, 295, None), slice(296.0, 299, None), slice(300.0, 303, None), slice(304.0, 307, None), slice(308.0, 311, None), slice(312.0, 315, None), slice(316.0, 319, None), slice(320.0, 323, None), slice(324.0, 327, None), slice(328.0, 331, None), slice(332.0, 335, None), slice(336.0, 339, None), slice(340.0, 343, None), slice(344.0, 347, None), slice(348.0, 351, None), slice(352.0, 355, None), slice(356.0, 359, None), slice(360.0, 363, None), slice(364.0, 367, None), slice(368.0, 371, None), slice(372.0, 375, None), slice(376.0, 379, None), slice(380.0, 383, None), slice(384.0, 387, None), slice(388.0, 391, None), slice(392.0, 395, None), slice(396.0, 399, None), slice(400.0, 403, None), slice(404.0, 407, None), slice(408.0, 411, None), slice(412.0, 415, None), slice(416.0, 419, None), slice(420.0, 423, None), slice(424.0, 427, None), slice(428.0, 431, None), slice(432.0, 435, None), slice(436.0, 439, None), slice(440.0, 443, None), slice(444.0, 447, None), slice(448.0, 451, None), slice(452.0, 455, None), slice(456.0, 459, None), slice(460.0, 463, None), slice(464.0, 467, None), slice(468.0, 471, None), slice(472.0, 475, None), slice(476.0, 479, None), slice(480.0, 483, None), slice(484.0, 487, None), slice(488.0, 491, None), slice(492.0, 495, None), slice(496.0, 499, None), slice(500.0, 503, None), slice(504.0, 507, None), slice(508.0, 511, None), slice(512.0, 515, None), slice(516.0, 519, None), slice(520.0, 523, None), slice(524.0, 527, None), slice(528.0, 531, None), slice(532.0, 535, None), slice(536.0, 539, None), slice(540.0, 543, None), slice(544.0, 547, None), slice(548.0, 551, None), slice(552.0, 555, None), slice(556.0, 559, None), slice(560.0, 563, None), slice(564.0, 567, None), slice(568.0, 571, None), slice(580.0, 583, None), slice(584.0, 587, None), slice(588.0, 591, None), slice(592.0, 595, None), slice(600.0, 603, None), slice(604.0, 607, None), slice(608.0, 611, None), slice(612.0, 615, None), slice(616.0, 619, None), slice(620.0, 623, None), slice(624.0, 627, None), slice(628.0, 631, None), slice(632.0, 635, None), slice(636.0, 639, None), slice(640.0, 643, None), slice(644.0, 647, None), slice(648.0, 651, None), slice(652.0, 655, None), slice(656.0, 659, None), slice(660.0, 663, None), slice(664.0, 667, None), slice(668.0, 671, None), slice(672.0, 675, None), slice(676.0, 679, None), slice(680.0, 683, None), slice(684.0, 687, None), slice(688.0, 691, None), slice(692.0, 695, None), slice(696.0, 699, None), slice(700.0, 703, None), slice(704.0, 707, None), slice(708.0, 711, None), slice(712.0, 715, None), slice(716.0, 719, None), slice(720.0, 723, None), slice(724.0, 727, None), slice(728.0, 731, None), slice(732.0, 735, None), slice(736.0, 739, None), slice(740.0, 743, None), slice(744.0, 747, None), slice(748.0, 751, None), slice(752.0, 755, None), slice(756.0, 759, None), slice(760.0, 763, None), slice(764.0, 767, None), slice(768.0, 771, None), slice(772.0, 775, None), slice(776.0, 779, None), slice(780.0, 783, None), slice(784.0, 787, None), slice(788.0, 791, None), slice(792.0, 795, None), slice(796.0, 799, None), slice(7620.0, 7623, None), slice(7624.0, 7627, None), slice(7628.0, 7631, None), slice(7632.0, 7635, None), slice(7636.0, 7639, None), slice(7640.0, 7643, None), slice(7644.0, 7647, None), slice(7648.0, 7651, None), slice(7652.0, 7655, None), slice(7656.0, 7659, None), slice(7660.0, 7663, None), slice(7664.0, 7667, None), slice(7668.0, 7671, None), slice(7672.0, 7675, None), slice(7676.0, 7679, None), slice(7680.0, 7683, None), slice(7684.0, 7687, None), slice(7688.0, 7691, None), slice(7692.0, 7695, None), slice(7696.0, 7699, None), slice(7700.0, 7703, None), slice(7704.0, 7707, None), slice(7708.0, 7711, None), slice(7712.0, 7715, None), slice(7716.0, 7719, None), slice(7720.0, 7723, None), slice(7724.0, 7727, None), slice(7728.0, 7731, None), slice(7732.0, 7735, None), slice(7736.0, 7739, None), slice(7740.0, 7743, None), slice(7744.0, 7747, None), slice(7748.0, 7751, None), slice(7752.0, 7755, None), slice(7756.0, 7759, None), slice(7760.0, 7763, None), slice(7764.0, 7767, None), slice(7768.0, 7771, None), slice(7772.0, 7775, None), slice(7776.0, 7779, None), slice(7780.0, 7783, None), slice(7784.0, 7787, None), slice(7788.0, 7791, None), slice(7792.0, 7795, None), slice(7796.0, 7799, None), slice(7800.0, 7803, None), slice(7804.0, 7807, None), slice(7808.0, 7811, None), slice(7812.0, 7815, None), slice(7816.0, 7819, None), slice(7820.0, 7823, None), slice(7824.0, 7827, None), slice(7828.0, 7831, None), slice(7832.0, 7835, None), slice(7836.0, 7839, None), slice(7840.0, 7843, None), slice(7844.0, 7847, None), slice(7848.0, 7851, None), slice(7852.0, 7855, None), slice(7856.0, 7859, None), slice(7860.0, 7863, None), slice(7864.0, 7867, None), slice(7868.0, 7871, None), slice(7872.0, 7875, None), slice(7876.0, 7879, None), slice(7880.0, 7883, None), slice(7884.0, 7887, None), slice(7888.0, 7891, None), slice(7892.0, 7895, None), slice(7896.0, 7899, None), slice(7900.0, 7903, None), slice(7904.0, 7907, None), slice(7908.0, 7911, None), slice(7912.0, 7915, None), slice(7916.0, 7919, None), slice(7920.0, 7923, None), slice(7924.0, 7927, None), slice(7928.0, 7931, None), slice(7932.0, 7935, None), slice(7936.0, 7939, None), slice(7940.0, 7943, None), slice(7944.0, 7947, None), slice(7948.0, 7951, None), slice(7952.0, 7955, None), slice(28.0, 32, None), slice(32.0, 36, None), slice(7118.0, 7121, None), slice(7956.0, 7983, None), slice(7985.0, 8030, None), slice(8030.0, 8034, None), slice(8034.0, 8038, None), slice(8038.0, 8042, None), slice(8042.0, 8046, None), slice(8046.0, 8050, None), slice(8050.0, 8054, None), slice(8054.0, 8058, None), slice(8058.0, 8062, None), slice(8062.0, 8066, None), slice(8066.0, 8070, None), slice(8070.0, 8074, None), slice(8074.0, 8113, None), slice(8146.0, 8220, None), slice(8220.0, 8224, None), slice(8224.0, 8228, None), slice(8228.0, 8232, None), slice(8233.0, 8236, None), slice(8237.0, 8240, None), slice(8241.0, 8244, None), slice(8244.0, 8248, None), slice(8248.0, 8252, None), slice(8252.0, 8256, None), slice(8257.0, 8260, None), slice(8261.0, 8264, None), slice(8264.0, 8268, None), slice(8268.0, 8272, None), slice(8272.0, 8276, None), slice(8276.0, 8280, None), slice(8280.0, 8284, None), slice(8284.0, 8288, None), slice(8288.0, 8292, None), slice(8292.0, 8296, None)]
        self.assertEqual(result,
                         expected)

class TestStepLocalCusp(unittest.TestCase):
    def test_step_cusp_basic(self):
        array = np.ma.array([3,7,9,9])
        test_slice=slice(0, 4, 1)
        cusp = step_local_cusp(array, test_slice)
        self.assertEqual(cusp, 2)

    def test_step_cusp_downward(self):
        array = np.ma.array([9,8,9,9,8,5,1,1,1,1])
        test_slice=slice(3, None, None)
        cusp = step_local_cusp(array, test_slice)
        self.assertEqual(cusp, 3)

    def test_step_cusp_negative_with_step(self):
        array = np.ma.array([1,1,1,1,1,5,8,9,9,9,9])
        test_slice=slice(7, 0, -1)
        cusp = step_local_cusp(array, test_slice)
        self.assertEqual(cusp, 4)

    def test_step_cusp_level_start(self):
        array = np.ma.array([1,1,1,1,1,5,8,9,9,9,9])
        test_slice=slice(None, None, None)
        cusp = step_local_cusp(array, test_slice)
        self.assertEqual(cusp, 0)

    def test_step_cusp_level_start_reverse(self):
        array = np.ma.array([1,1,1,1,1,5,8,9,9,9,9])
        test_slice=slice(9, 0, -1)
        cusp = step_local_cusp(array, test_slice)
        self.assertEqual(cusp, 0)

    def test_step_cusp_static(self):
        array = np.ma.array([33]*8)
        test_slice=slice(None, None, None)
        cusp = step_local_cusp(array, test_slice)
        self.assertEqual(cusp, 0)

    def test_step_cusp_short(self):
        array = np.ma.array([33])
        test_slice=slice(None, None, None)
        cusp = step_local_cusp(array, test_slice)
        self.assertEqual(cusp, 0)


class TestStepValues(unittest.TestCase):
    '''
    Step values is used to reduce continuously variable arrays to defined
    settings, e.g. flap settings. As the nature of the data is irregular,
    specimen data has been used as the primary source, and manually checked
    graph outputs were used to derive the test result arrays.

    # useful for plotting most of the test cases:
    from analysis_engine.plot_flight import plot_parameter; plot_parameter(array); plot_parameter(stepped)

    # useful for compressing results:
    print(compress_iter_repr(stepped, int))
    '''

    def test_step_values(self):
        # borrowed from TestSlat
        array = np.ma.concatenate([np.arange(25), np.arange(-5,0)])
        array[1] = np.ma.masked
        array = step_values(array, ( 0, 16, 20, 23))
        self.assertEqual(len(array), 30)
        self.assertEqual(
            list(np.ma.filled(array, fill_value=-999)),
            [0, -999, 0, 0, 0, 0, 0, 0, 0,
             16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
             20, 20, 20,
             23, 23, 23,
             0, 0, 0, 0, 0])

    def test_step_inital_level(self):
        array = np.ma.arange(9,14,0.6)
        stepped = step_values(array, (10, 11, 15))
        self.assertEqual(list(stepped),
                         [10, 10, 10, 11, 11, 11, 11, 15, 15])

    def test_step_leading_edge(self):
        array = np.ma.array([0,0.1,0.0,0.8,1.6,3,6,6,6,6,8,13,14,14,9,9,9,6,3,2,
                             2,2,2,0,0,0])
        stepped = step_values(array, (0, 1, 5, 10, 15), step_at='move_start')
        # used to be this
        ##self.assertEqual(list(stepped),
                         ##[0]*3+[1]+[5]*6+[10]+[15]*3+[10]*3+[5]+[1]*5+[0]*3)
        # now this:
        self.assertEqual(stepped.tolist(),
                         [0]*3+[1]+[5]*5+[10]*2+[15]*2+[10]*3+[5]*2+[1]*5+[0]*3)

    def test_step_move_start(self):
        array = np.ma.array(data=[0]*5+[1,2,3,4]+[5]*5)
        stepped = step_values(array, (0, 4), step_at='move_start')
        self.assertEqual(list(stepped), [0]*5+[4]*9)

    def test_step_leading_edge_real_data(self):
        array = np.ma.array([0, 0, 0, 0, 0, 0, 0.12, 0.37, 0.5, 0.49, 0.49,
                             0.67, 0.98, 1.15, 1.28, 1.5, 1.71, 1.92, 2.12,
                             2.32, 2.53, 2.75, 2.96, 3.18, 3.39, 3.6, 3.83,
                             4.06, 4.3, 4.57, 4.82, 5.1, 5.41, 5.85, 7.12,
                             9.92, 13.24, 15.03, 15.36, 15.36, 15.36, 15.37,
                             15.38, 15.39, 15.37, 15.37, 15.41, 15.44])
        array = np.ma.concatenate((array, array[::-1]))
        stepped = step_values(array, (0, 1, 5, 15), step_at='move_start')
        self.assertEqual(stepped.tolist(),
                         [0]*11+[1]*3+[5]*19+[15]*26+[5]*7+[1]*19+[0]*11)

    @unittest.skip('skip was used when creating Flap Lever from Flap Surface '
                   'but this is no longer required.')
    def test_step_leading_edge_skip_real_data(self):
        # from docstring:
        #:param skip: Selects whether steps that are passed straight through should be mapped or not. Only relates to 'move_start' or 'move_end' options.
        #:type skip: logical, default = False
        array = np.ma.array([0, 0, 0, 0, 0, 0, 0.12, 0.37, 0.5, 0.49, 0.49,
                             0.67, 0.98, 1.15, 1.28, 1.5, 1.71, 1.92, 2.12,
                             2.32, 2.53, 2.75, 2.96, 3.18, 3.39, 3.6, 3.83,
                             4.06, 4.3, 4.57, 4.82, 5.1, 5.41, 5.85, 7.12,
                             9.92, 13.24, 15.03, 15.36, 15.36, 15.36, 15.37,
                             15.38, 15.39, 15.37, 15.37, 15.41, 15.44])
        array = np.ma.concatenate((array, array[::-1]))
        stepped = step_values(array, (0, 1, 5, 15), step_at='move_start',
                              skip=True, rate_threshold=0.1)
        self.assertEqual(stepped.tolist(), [0]*10+[15]*47+[0]*39)

    def test_step_midpoint_real_data(self):
        array = np.ma.array([0, 0, 0, 0, 0, 0, 0.12, 0.37, 0.5, 0.49, 0.49,
                             0.67, 0.98, 1.15, 1.28, 1.5, 1.71, 1.92, 2.12,
                             2.32, 2.53, 2.75, 2.96, 3.18, 3.39, 3.6, 3.83,
                             4.06, 4.3, 4.57, 4.82, 5.1, 5.41, 5.85, 7.12,
                             9.92, 13.24, 15.03, 15.36, 15.36, 15.36, 15.37,
                             15.38, 15.39, 15.37, 15.37, 15.41, 15.44])
        array = np.ma.concatenate((array, array[::-1]))
        stepped = step_values(array, (0, 1, 5, 15), step_at='midpoint')
        self.assertEqual(stepped.tolist(),
                         [0]*11+[1]*12+[5]*13+[15]*24+[5]*13+[1]*12+[0]*11)

    def test_step_excluding_transition_real_data(self):
        array = np.ma.array([0, 0, 0, 0, 0, 0, 0.12, 0.37, 0.5, 0.49, 0.49,
                             0.67, 0.98, 1.15, 1.28, 1.5, 1.71, 1.92, 2.12,
                             2.32, 2.53, 2.75, 2.96, 3.18, 3.39, 3.6, 3.83,
                             4.06, 4.3, 4.57, 4.82, 5.1, 5.41, 5.85, 7.12,
                             9.92, 13.24, 15.03, 15.36, 15.36, 15.36, 15.37,
                             15.38, 15.39, 15.37, 15.37, 15.41, 15.44])
        array = np.ma.concatenate((array, array[::-1]))
        stepped = step_values(array, (0, 1, 5, 15), step_at='excluding_transition')
        self.assertEqual(stepped.tolist(), [0]*11+[1]*19+[5]*7+[15]*22+[5]*7+[1]*19+[0]*11)


    def test_step_including_transition_real_data(self):
        array = np.ma.array([0, 0, 0, 0, 0, 0, 0.12, 0.37, 0.5, 0.49, 0.49,
                             0.67, 0.98, 1.15, 1.28, 1.5, 1.71, 1.92, 2.12,
                             2.32, 2.53, 2.75, 2.96, 3.18, 3.39, 3.6, 3.83,
                             4.06, 4.3, 4.57, 4.82, 5.1, 5.41, 5.85, 7.12,
                             9.92, 13.24, 15.03, 15.36, 15.36, 15.36, 15.37,
                             15.38, 15.39, 15.37, 15.37, 15.41, 15.44])
        array = np.ma.concatenate((array, array[::-1]))
        stepped = step_values(array, (0, 1, 5, 15), step_at='including_transition')
        self.assertEqual(stepped.tolist(),[0]*11+[1]*3+[5]*19+[15]*30+[5]*19+[1]*3+[0]*11)

    def test_step_including_transition_edge_case(self):
        # This set of values caused problems with a low sample rate flap (767-4 frame)
        array = np.ma.array([0, 1, 1, 1, 1, 0.878, 0.5270, 0.0, 0.0, 0.0, 0.45, 1])
        stepped = step_values(array, (0, 1), hz=0.25, step_at='including_transition')
        self.assertEqual(stepped.tolist(),[0]+[1]*6+[0]*3+[1]*2)

    @unittest.skip("move_stop no longer in use")
    def test_step_trailing_edge_real_data(self):
        array = np.ma.array([0, 0, 0, 0, 0, 0, 0.12, 0.37, 0.5, 0.49, 0.49,
                             0.67, 0.98, 1.15, 1.28, 1.5, 1.71, 1.92, 2.12,
                             2.32, 2.53, 2.75, 2.96, 3.18, 3.39, 3.6, 3.83,
                             4.06, 4.3, 4.57, 4.82, 5.1, 5.41, 5.85, 7.12,
                             9.92, 13.24, 15.03, 15.36, 15.36, 15.36, 15.37,
                             15.38, 15.39, 15.37, 15.37, 15.41, 15.44])
        array = np.ma.concatenate((array, array[::-1]))
        # incorrect step_at arguent
        self.assertRaises(ValueError, step_values, array, (0, 1, 5, 15),
                          step_at='move_end')
        stepped = step_values(array, (0, 1, 5, 15), step_at='move_stop')
        self.assertEqual(stepped.tolist(),
                         [0]*12+[1]*18+[5]*7+[15]*26+[5]*19+[1]*6+[0]*8)

    def test_step_trailing_edge_masked_data(self):
        array = np.ma.array(
            [0, 0, 0, 0, 4.92184, 4.92184, 4.92184, 4.92184,
             4.92184, 4.92184, 4.92184, 4.92184, 4.92184, 4.92184])
        array = np.ma.concatenate((array, array[::-1]))
        array[:4] = np.ma.masked
        stepped = step_values(array, (0, 1, 5, 15), step_at='move_stop')
        expected = np.ma.array([5.0]*24+[0.0]*4)
        expected[:4] = np.ma.masked
        self.assertEqual(stepped.tolist(), expected.tolist())

    def test_flap_transition_real_data(self):
        flap = load(os.path.join(test_data_path,
                                 'flap_transition_test_data.nod'))
        # including transition
        res = step_values(flap.array, (0, 1, 5, 15, 20, 25, 30), step_at='including_transition')
        inc_edges = find_edges(res)
        self.assertEqual(inc_edges,
            [364.5, 387.5, 410.5, 5727.5, 5816.5, 5926.5, 5939.5, 5989.5, 6017.5])
        # excluding transition
        res = step_values(flap.array, (0, 1, 5, 15, 20, 25, 30), step_at='excluding_transition')
        exc_edges = find_edges(res)
        self.assertEqual(exc_edges,
            [369.5, 407.5, 421.5, 5732.5, 5839.5, 5938.5, 5945.5, 5995.5, 6024.5])

    def test_flap_transitions_in_masked_data_1(self):
        # Ensure flap angle is corrected when masked data hides transition
        flap_angle = np.ma.array([0.004763]*30 + [0.0]*100 + [0.99]*200 + list(np.arange(0.99, 0.001, -0.1)) + [0.0]*100)
        flap_angle[30:130] = np.ma.masked
        #flap_angle = np.ma.array([0.004763]*3 + [0.0]*10 + [0.99]*4)
        #flap_angle[3:13] = np.ma.masked
        res = step_values(flap_angle, (0, 1, 5, 15, 20, 25, 30), hz=1, step_at='move_start')
        self.assertEqual(find_edges(res, direction='all_edges'), [129.5, 330.5])

    def test_flap_transitions_in_masked_data_2(self):
        # Ensure flap angle is corrected when masked data hides transition
        flap_angle = np.ma.concatenate([np.arange(0, 30, 0.1),
                                        np.arange(30, 0, -0.1)])
        #flap_angle = np.ma.array([0.004763]*30 + [0.0]*100 + [0.99]*200 + list(np.arange(0.99, 0.001, -0.1)) + [0.0]*100)
        flap_angle[20:40] = np.ma.masked
        flap_angle[60:80] = np.ma.masked
        flap_angle[100:120] = np.ma.masked
        flap_angle[140:160] = np.ma.masked
        flap_angle[180:200] = np.ma.masked
        flap_angle[220:240] = np.ma.masked
        flap_angle[260:280] = np.ma.masked

        flap_angle[320:340] = np.ma.masked
        flap_angle[360:380] = np.ma.masked
        flap_angle[400:420] = np.ma.masked
        flap_angle[440:460] = np.ma.masked
        flap_angle[480:500] = np.ma.masked
        flap_angle[520:540] = np.ma.masked
        flap_angle[560:580] = np.ma.masked
        #flap_angle = np.ma.array([0.004763]*3 + [0.0]*10 + [0.99]*4)
        #flap_angle[3:13] = np.ma.masked
        res = step_values(flap_angle, (0, 1, 5, 15, 20, 25, 30), hz=1,
                          step_at='move_start')
        self.assertEqual(find_edges(res), [4.5, 38.5, 119.5, 173.5, 238.5, 278.5])

    def test_step_values_masked_end(self):
        array = load_compressed(
            os.path.join(test_data_path, 'step_values_flap_masked_end.npz'))
        # unsorted steps matches original input during process flight.
        steps = [0, 1, 2, 5, 40, 10, 15, 25, 30]
        res = step_values(array, steps, step_at='move_start')
        self.assertEqual(res[11000:11087].tolist(), [30] * 7 + [0] * 80)
        self.assertTrue(res.mask[11087:].all())


class TestCompressIterRepr(unittest.TestCase):
    def test_compress_iter_repr(self):
        self.assertEqual(compress_iter_repr([0,0,1,0,2,2,2], join=' + '),
                         "[0]*2 + [1] + [0] + [2]*3")
        self.assertEqual(compress_iter_repr(['a', 'a', 'a']),
                         "['a']*3")
        # interesting side effect - int(5.4) == int('5')
        self.assertEqual(compress_iter_repr([4.0, 5.4, '5'], int),
                         "[4]+[5]*2")
        test_array = np.ma.array(data=[1,1,1,2,2,2,3,3,3],
                                 mask=[1,1,0,0,0,1,1,1,0])
        self.assertEqual(compress_iter_repr(test_array),
                         "[masked]*2+[1]+[2]*2+[masked]*3+[3]")


class TestStraightenAltitudes(unittest.TestCase):
    def test_alt_basic(self):
        data=np.ma.array([5.0,100.0,300.0,450.0,46.0,380.0,230.0,110.0,0.0])
        result=straighten_altitudes(data, None, 500.0, False)
        self.assertEqual(np.ma.max(result),546.0)

    def test_alt_offset_from_coarse(self):
        data=np.ma.array([5.0,100.0,300.0,450.0,46.0,380.0,230.0,110.0,0.0])
        fine = data + 1500.0
        coarse = data * 1.01
        result=straighten_altitudes(fine, coarse, 500.0, False)
        self.assertEqual(np.ma.max(result),546.0)

    def test_alt_offset_from_coarse_just_first_sample(self):
        fine = np.ma.array([495.0,100.0,300.0,450.0,46.0,380.0,230.0,110.0,0.0])
        coarse = np.ma.array([5.0,100.0,300.0,450.0,46.0,380.0,230.0,110.0,0.0])
        result=straighten_altitudes(fine, coarse, 500.0, False)
        self.assertEqual(np.ma.max(result),546.0)


class TestStraightenLongitude(unittest.TestCase):
    def test_straighten_longitude_without_overflow(self):
        data = np.ma.array([35.5,29.5,11.3,0.0,2.5,8.1,14.4])
        np.testing.assert_array_almost_equal(straighten_longitude(data), data)

    def test_straighten_longitude_single_overflow(self):
        data = np.ma.array([175.5,179.5,179.1,179.9,-179.9,-178.2,-176.5,-175.2,-174.1,-172.4])
        expected = np.ma.array(
            [175.5,179.5,179.1,179.9,180.1,181.8,183.5,184.8,185.9,187.6])
        np.testing.assert_array_almost_equal(straighten_longitude(data), expected)

    def test_straighten_longitude_single_overflow_and_return(self):
        data = np.ma.array([175.5,179.5,179.1,179.9, -179.9,-178.2,176.5,175.2,174.1,172.4])
        expected = np.ma.array(
            [175.5, 179.5, 179.1, 179.9, 180.1, 181.8, 176.5, 175.2, 174.1, 172.4])
        np.testing.assert_array_almost_equal(straighten_longitude(data), expected)

    def test_straighten_longitude_single_overflow_masked(self):
        mask = [True,False,False,True,True,False,False,False,False,True]
        data = np.ma.array([175.5,179.5,179.1,179.9, -179.9,-178.2,-176.5,-175.2,-174.1,-172.4],
                           mask=mask)
        expected = np.ma.array(
            [175.5, 179.5, 179.1, 179.9, -179.9, 181.8, 183.5, 184.8, 185.9, -172.4],
            mask=mask)
        np.testing.assert_array_almost_equal(straighten_longitude(data), expected)

    def test_straighten_longitude_single_overflow_and_return_masked(self):
        mask = [True,False,False,True,True,False,True,False,False,True]
        data = np.ma.array([175.5,179.5,179.1,179.9, -179.9,-178.2,176.5,175.2,174.1,172.4],
                           mask=mask)
        expected = np.ma.array(
            [175.5, 179.5, 179.1, 179.9, 180.1, 181.8, 176.5, 175.2, 174.1, 172.4],
            mask=mask)
        np.testing.assert_array_almost_equal(straighten_longitude(data), expected)

    def test_straighten_longitude_1_spike(self):
        '''
        Spike does not trigger overflow as with straighten().
        '''
        data = load_compressed(os.path.join(test_data_path, 'straighten_longitude_1.npz'))
        np.testing.assert_array_almost_equal(straighten_longitude(data), data)

    def test_straighten_longitude_2_drop(self):
        '''
        Overflow is not triggered as drop is too small.
        '''
        data = load_compressed(os.path.join(test_data_path, 'straighten_longitude_2.npz'))
        np.testing.assert_array_almost_equal(straighten_longitude(data), data)

    def test_straighten_longitude_3(self):
        '''
        Ignore spikes.
        '''
        data = load_compressed(os.path.join(test_data_path, 'straighten_longitude_3.npz'))
        np.testing.assert_array_almost_equal(straighten_longitude(data), data)

    def test_straighten_longitude_4(self):
        data = load_compressed(os.path.join(test_data_path, 'straighten_longitude_4.npz'))
        np.testing.assert_array_almost_equal(straighten_longitude(data), data)

    def test_straighten_longitude_single_overflow_followed_by_masked_data(self):
        data = np.ma.array([175.5,179.5,179.1,179.9,-179.9,-178.2,-176.5,-175.2,-174.1,-172.4])
        data[6] = np.ma.masked
        expected = np.ma.array(
            [175.5,179.5,179.1,179.9,180.1,181.8,183.5,184.8,185.9,187.6])
        expected[6] = np.ma.masked
        np.testing.assert_array_almost_equal(straighten_longitude(data), expected)


class TestStraightenHeadings(unittest.TestCase):
    def test_straighten_headings(self):
        data = np.ma.array([35.5,29.5,11.3,0.0,348.4,336.8,358.9,2.5,8.1,14.4])
        expected = np.ma.array(
            [35.5,29.5,11.3,0.0,-11.6,-23.2,-1.1,2.5,8.1,14.4])
        np.testing.assert_array_almost_equal(straighten_headings(data),expected)

    def test_straighten_headings_starting_masked(self):
        data = np.ma.array(
            data=[0]*10+[6]*8+[1]*4+[10,5,0,355,350]+[0]*4,
            mask=[True]*10+[False]*8+[True]*4+[False]*5+[True]*4, dtype=float)
        expected = np.ma.array(
            data=[0]*10+[6]*8+[0]*4+[10,5,0,-5,-10]+[0]*4,
            mask=[True]*10+[False]*8+[True]*4+[False]*5+[True]*4, dtype=float)
        ma_test.assert_masked_array_approx_equal(straighten_headings(data),
                                                 expected)

    def test_straighten_headings_masked_rollover(self):
        mask = [False, False, False, True, True, False, False, False, True,
                True, False, False]
        data = np.ma.array(
            [340, 345, 350, 8539, 2920, 10, 15, 20, 8580, 6581, 35, 40],
            mask=mask, dtype=float)
        expected = np.ma.array(
            [340, 345, 350, 8539, 2920, 370, 375, 380, 8580, 6581, 395, 400],
            mask=mask, dtype=float)
        ma_test.assert_masked_array_approx_equal(straighten_headings(data),
                                                 expected)

    def test_straighten_headings_2(self):
        array = load_compressed(os.path.join(test_data_path, 'straighten_headings_1.npz'))
        result = straighten_headings(array)
        # result does not jump between overflows
        self.assertTrue(np.ma.max(np.ma.ediff1d(result)) < 10)


class TestStraighten(unittest.TestCase):
    def test_offsets(self):
        result = straighten(np.ma.array([0.0]), np.ma.array([9.0]), 20.0, False)
        self.assertEqual(result[0], 0.0)
        result = straighten(np.ma.array([0.0]), np.ma.array([100.0]), 43.0, False)
        self.assertEqual(result[0], 86.0)
        result = straighten(np.ma.array([0.0]), np.ma.array([-200.0]), 45.0, False)
        self.assertEqual(result[0], -180.0)


class TestSmoothSignal(unittest.TestCase):
    def setUp(self):
        signal = load(os.path.join(test_data_path, 'ae-1165-flap_angle.nod'))
        _slices = runs_of_ones(np.logical_and(signal.array>=0.7,
                                              signal.array<=2.3))
        self.signal_to_smooth = signal.array[_slices[2]]

    def test_smooth_signal_flat(self):
        expected = [
            0.9668, 0.9844, 1.0371, 1.0547, 1.0723, 1.125, 1.1953, 1.2656,
            1.3359, 1.3887, 1.4238, 1.4766, 1.5469, 1.6172, 1.6875, 1.7402,
            1.7754, 1.8281, 1.8984, 1.9687, 2.039, 2.0918, 2.1094, 2.1269,
        ]
        smoothed_signal = smooth_signal(self.signal_to_smooth,
                                        window_len=5, window='flat')
        sig_rounded = [round(l,4) for l in smoothed_signal]
        uniques = [i for i in sorted(Counter(sig_rounded))]
        self.assertEqual(len(uniques), len(expected))
        for u, e in zip(uniques, expected):
            self.assertAlmostEqual(u, e, places=2)

    def test_smooth_signal_hanning(self):
        expected = [
            0.8789, 0.9448, 1.0327, 1.0547, 1.0767, 1.1645, 1.2964, 1.3843,
            1.4062, 1.4282, 1.5161, 1.6479, 1.7358, 1.7578, 1.7798, 1.8677,
            1.9995, 2.0874, 2.1094, 2.1313,
        ]
        smoothed_signal = smooth_signal(self.signal_to_smooth,
                                        window_len=5, window='hanning')
        sig_rounded = [round(l,4) for l in smoothed_signal]
        uniques = [i for i in sorted(Counter(sig_rounded))]
        self.assertEqual(len(uniques), len(expected))
        for u, e in zip(uniques, expected):
            self.assertAlmostEqual(u, e, places=2)

    def test_smooth_signal_hamming(self):
        expected = [
            0.8946, 0.9487, 1.0241, 1.0515, 1.0547, 1.0578, 1.0853, 1.17,
            1.2909, 1.3756, 1.4031, 1.4094, 1.4368, 1.5216, 1.6424, 1.7272,
            1.7547, 1.7609, 1.7884, 1.8732, 1.994, 2.0788, 2.1062, 2.1094,
            2.1125, 2.1305,
        ]
        smoothed_signal = smooth_signal(self.signal_to_smooth,
                                        window_len=5, window='hamming')
        sig_rounded = [round(l,4) for l in smoothed_signal]
        uniques = [i for i in sorted(Counter(sig_rounded))]
        self.assertEqual(len(uniques), len(expected))
        for u, e in zip(uniques, expected):
            self.assertAlmostEqual(u, e, places=2)

    def test_smooth_signal_bartlett(self):
        expected = [
            0.8789, 0.9448, 1.0327, 1.0547, 1.0767, 1.1645, 1.2964, 1.3843,
            1.4062, 1.4282, 1.5161, 1.6479, 1.7358, 1.7578, 1.7798, 1.8677,
            1.9995, 2.0874, 2.1094, 2.1313,
        ]
        smoothed_signal = smooth_signal(self.signal_to_smooth,
                                        window_len=5, window='bartlett')
        sig_rounded = [round(l,4) for l in smoothed_signal]
        uniques = [i for i in sorted(Counter(sig_rounded))]
        self.assertEqual(len(uniques), len(expected))
        for u, e in zip(uniques, expected):
            self.assertAlmostEqual(u, e, places=2)

    def test_smooth_signal_blackman(self):
        expected = [
            0.8622, 0.949, 1.0369, 1.0547, 1.0725, 1.1604, 1.3006, 1.3885,
            1.4062, 1.424, 1.5119, 1.6521, 1.74, 1.7578, 1.7756, 1.8635,
            2.0037, 2.0916, 2.1094, 2.1271,
        ]
        smoothed_signal = smooth_signal(self.signal_to_smooth,
                                        window_len=5, window='blackman')
        sig_rounded = [round(l,4) for l in smoothed_signal]
        uniques = [i for i in sorted(Counter(sig_rounded))]
        self.assertEqual(len(uniques), len(expected))
        for u, e in zip(uniques, expected):
            self.assertAlmostEqual(u, e, places=2)


class TestSmoothTrack(unittest.TestCase):
    def test_smooth_track_latitude(self):
        lat = np.ma.array([0,0,0,1,1,1], dtype=float)
        lon = np.ma.zeros(6, dtype=float)
        lat_s, lon_s, cost = smooth_track(lat, lon, None, 0.25)
        self.assertLess (cost,26)

    def test_smooth_track_longitude(self):
        lon = np.ma.array([0,0,0,1,1,1], dtype=float)
        lat = np.ma.zeros(6, dtype=float)
        lat_s, lon_s, cost = smooth_track(lat, lon, None, 0.25)
        self.assertLess (cost,26)

    def test_smooth_track_sample_rate_change(self):
        lon = np.ma.array([0,0,0,1,1,1], dtype=float)
        lat = np.ma.zeros(6, dtype=float)
        lat_s, lon_s, cost = smooth_track(lat, lon, None, 1.0)
        self.assertLess (cost,251)
        self.assertGreater (cost,250)

    def test_smooth_track_speed(self):
        lon = np.ma.arange(10000, dtype=float)
        lon = lon%27
        lat = np.ma.zeros(10000, dtype=float)
        start = process_time()
        lat_s, lon_s, cost = smooth_track(lat, lon, None, 0.25)
        end = process_time()
        self.assertLess(end-start, 1.0)


class TestSubslice(unittest.TestCase):
    def test_subslice(self):
        # test basic
        orig = slice(2,10)
        new = slice(2, 4)
        res = subslice(orig, new)
        self.assertEqual(res, slice(4, 6))
        fifty = list(range(50))
        self.assertEqual(fifty[orig][new], fifty[res])

        # test basic starting from zero
        orig = slice(2,10)
        new = slice(0, 4)
        res = subslice(orig, new)
        self.assertEqual(res, slice(2, 6))
        fifty = list(range(50))
        self.assertEqual(fifty[orig][new], fifty[res])

        orig = slice(10,20,2)
        new = slice(2, 4, 1)
        res = subslice(orig, new)
        thirty = list(range(30))
        self.assertEqual(thirty[orig][new], thirty[res])
        self.assertEqual(res, slice(14, 18, 2))

        # test step
        orig = slice(100,200,10)
        new = slice(1, 5, 2)
        sub = subslice(orig, new)
        two_hundred = list(range(0,200))
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(sub, slice(110, 150, 20))

        # test negative step
        orig = slice(200,100,-10)
        new = slice(1, 5, 2)
        sub = subslice(orig, new)
        two_hundred = list(range(201))
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(sub, slice(190, 150, -20))

        orig = slice(100,200,10)
        new = slice(5, 1, -2)
        sub = subslice(orig, new)
        two_hundred = list(range(201))
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(sub, slice(150, 110, -20))
        self.assertEqual(two_hundred[sub], [150, 130]) #fix

        # test invalid back step
        orig = slice(0,200,10)
        new = slice(1, 5, -2)
        sub = subslice(orig, new)
        two_hundred = list(range(201))
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], []) # invalid returns no data
        self.assertEqual(sub, slice(10, 50, -20))

        # test no start
        orig = slice(None,100,10)
        new = slice(5, 1, -2)
        sub = subslice(orig, new)
        two_hundred = list(range(200))
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], [50,30])
        self.assertEqual(sub, slice(50, 10, -20))

        orig = slice(0,10,2)
        new = slice(None, 4)
        sub = subslice(orig, new)
        two_hundred = list(range(5))
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], [0,2,4]) # also tests outside of range
        self.assertEqual(sub, slice(0, 8, 2))

        # test None start and invalid back step
        orig = slice(None,200,10)
        new = slice(1, 5, -2)
        sub = subslice(orig, new)
        two_hundred = list(range(201))
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], [])
        self.assertEqual(sub, slice(10, 50, -20))

        # test None at end of second slice
        orig = slice(0,10,2)
        new = slice(1, None)
        sub = subslice(orig, new)
        two_hundred = list(range(5))
        self.assertEqual(two_hundred[orig][new], two_hundred[sub])
        self.assertEqual(two_hundred[sub], [2,4])
        self.assertEqual(sub, slice(2, 10, 2))

        # Actual case from test 6_737_1_RD0001851371
        orig = slice(419, 423, None)
        new = slice(0, None, 1)
        sub = subslice(orig, new)
        self.assertEqual(sub,slice(419,423,None))

        orig = slice(419, 423, None)
        new = slice(0, None, None)
        sub = subslice(orig, new)
        self.assertEqual(sub,slice(419,423,None))

        #TODO: test negative start, stop and step


class TestTouchdownInertial(unittest.TestCase):
    @unittest.skip("Not Implemented")
    def test_touchdown_inertial(self):
        self.assertTrue(False)


class TestTrackLinking(unittest.TestCase):
    def test_track_linking_basic(self):
        pos = np.ma.array(data=[0]*16,mask=False)
        local = np.ma.arange(16, dtype=float)
        local[0:3]=np.ma.masked
        local[6:10]=np.ma.masked
        local[13:]=np.ma.masked
        local[8:] -= 2.5
        # plot_parameter(local)
        result = track_linking(pos,local)
        expected = np.ma.array(data = [3.0,3.0,3.0,3.0,4.0,5.0,5.5,6.0,
                                       6.5,7.0,7.5,8.5,9.5,9.5,9.5,9.5],
                               mask = False)
        np.testing.assert_array_equal(expected,result)
        # plot_parameter(expected)


class TestTrimSlices(unittest.TestCase):
    def test_trim_slices(self):
        self.assertEqual(trim_slices([], 1, 1, 10), [])
        self.assertEqual(trim_slices([slice(0, 5)], 1, 1, 10), [slice(1, 4)])
        self.assertEqual(trim_slices([slice(0, 8), slice(12, 20)], 2, 1, 20),
                         [slice(2, 6), slice(14, 18)])
        self.assertEqual(trim_slices([slice(None, 5)], 1, 1, 10), [slice(1, 4)])
        self.assertEqual(trim_slices([slice(None, None)], 1, 1, 10),
                         [slice(1, 9)])
        self.assertEqual(trim_slices([slice(4, 26)], 2, 4, 30),
                         [slice(12, 18)])


"""
For the Truck and Trailer algorithm, see TestPeakCurvature above.
"""

class TestUniqueValue(unittest.TestCase):
    def test_unique_values(self):
        res = unique_values(np.ma.array([0,1,1,1,4,4,7,1,1],
                                   mask=[1,0,0,0,0,0,0,0,1]))
        self.assertEqual(res, {1: 4,
                               4: 2,
                               7: 1})

    def test_unique_with_multistate(self):
        array = MappedArray(np.ma.array([0, 1, 0, 0, 0, 1, 1],
                                   mask=[1, 0, 0, 0, 0, 0, 1]),
                            values_mapping={0:'A', 1:'B'})
        res = unique_values(array)
        self.assertEqual(res, {'A': 3, 'B': 2})

    def test_no_valid_data(self):
        # returns None when nothing is valid
        res = unique_values(np.ma.array([10, 20, 10], mask=True))
        self.assertEqual(res, {})


class TestMostCommonValue(unittest.TestCase):
    def test_most_common_value(self):
        res = most_common_value(
            np.ma.array([0, 1, 1, 1, 4, 4, 7, 1, 1],
                        mask=[1, 0, 0, 0, 0, 0, 0, 0, 1])
        )
        self.assertEqual(res, 1)

    #def test_most_common_value_negative_values_excluded(self):
    #    res = most_common_value(
    #        np.ma.array([0, -1, -1, -1, 4, 4, 7, 1, 1],
    #                    mask=[1, 0, 0, 0, 0, 0, 0, 0, 1])
    #    )
    #    self.assertEqual(res, 4)

    def test_unique_with_multistate(self):
        array = MappedArray(
            np.ma.array([0, 3, 0, 0, 0, 3, 3],
                        mask=[1, 0, 0, 0, 0, 0, 1]),
            values_mapping={0: 'A', 3: 'B'}
        )
        res = most_common_value(array)
        self.assertEqual(res, 'A')

    def test_no_valid_data(self):
        # returns None when nothing is valid
        res = most_common_value(np.ma.array([10, 20, 10], mask=True))
        self.assertEqual(res, None)


class TestValueAtTime(unittest.TestCase):
    # Reminder: value_at_time (array, hz, offset, time_index)

    def test_value_at_time_basic(self):
        array = np.ma.arange(4)
        self.assertEqual (value_at_time(array, 1, 0.0, 2.5), 2.5)

    def test_value_at_time_right_at_start_of_data(self):
        array = np.ma.arange(4) + 22.3
        self.assertEqual (value_at_time(array, 1, 0.0, 0.0), 22.3)

    def test_value_at_time_right_at_end_of_data(self):
        array = np.ma.arange(4) + 22.3
        self.assertEqual (value_at_time(array, 1.0, 0.0, 3.0), 25.3)

    def test_value_at_time_assertion_just_below_range(self):
        array = np.ma.arange(4)+7.0
        # Note: Frequency and offset selected to go more than one sample period below bottom of range.
        self.assertEqual (value_at_time(array, 1, 0.1, 0.0), 7.0)

    def test_value_at_time_with_lower_value_masked(self):
        array = np.ma.arange(4) + 7.4
        array[1] = np.ma.masked
        self.assertEqual (value_at_time(array, 2.0, 0.2, 1.0), 9.4)

    def test_value_at_time_with_higher_value_masked(self):
        array = np.ma.arange(4) + 7.4
        array[2] = np.ma.masked
        self.assertEqual (value_at_time(array, 2.0, 0.2, 1.0), 8.4)

    def test_value_at_time_with_neither_value_masked(self):
        array = np.ma.arange(4) + 7.4
        array[3] = np.ma.masked
        self.assertEqual (value_at_time(array, 2.0, 0.2, 1.0), 9.0)

    def test_value_at_time_with_both_values_masked(self):
        array = np.ma.arange(4) + 7.4
        array[1] = np.ma.masked
        array[2] = np.ma.masked
        self.assertEqual (value_at_time(array, 2.0, 0.2, 1.0), None)


class TestValueAtDatetime(unittest.TestCase):
    @mock.patch('analysis_engine.library.value_at_time')
    def test_value_at_datetime(self, value_at_time):
        array = mock.Mock()
        hz = mock.Mock()
        offset = mock.Mock()
        start_datetime = datetime.now()
        seconds = 20
        value_datetime = start_datetime + timedelta(seconds=seconds)
        value = value_at_datetime(start_datetime, array, hz, offset,
                                  value_datetime)
        value_at_time.assert_called_once_with(array, hz, offset, seconds)
        self.assertEqual(value, value_at_time.return_value)


class TestValueAtIndex(unittest.TestCase):

    # Reminder: value_at_time (array, index) This function is thoroughly
    # tested by the higher level value_at_time function, so this single test
    # just establishes confidence in the ability to access the lower level
    # function directly.

    def test_value_at_index_basic(self):
        array = np.ma.arange(4)
        self.assertEqual(value_at_index(array, 1.5), 1.5)

    def test_value_at_index_basic_list(self):
        array = list(range(4))
        self.assertEqual(value_at_index(array, 1.5), 1.5)

    def test_value_at_index_just_above_range(self):
        array = np.ma.arange(4)
        self.assertEqual(value_at_index(array, 3.7), 3.0)

    def test_value_at_index_just_below_range(self):
        array = np.ma.arange(4)
        self.assertEqual(value_at_index(array, -0.5), 0.0)

    def test_value_at_index_masked(self):
        array = np.ma.arange(4)
        array[2] = np.ma.masked
        self.assertEqual(value_at_index(array, 2), None)

    def test_value_at_index_non_interpolated(self):
        array = np.ma.arange(4)
        for x in (2.00, 2.25):
            self.assertEqual(value_at_index(array, x, interpolate=False), 2)
        for x in (2.50, 2.75, 3.00):
            self.assertEqual(value_at_index(array, x, interpolate=False), 3)

    def test_value_at_index_masked_non_interpolated(self):
        array = np.ma.arange(4)
        array[2] = np.ma.masked
        for x in (2.00, 2.25, 2.50, 2.75, 3.00):
            expected = None if x == 2.00 else 3
            self.assertEqual(value_at_index(array, x, interpolate=False), expected)
        array = np.ma.arange(4)
        array[3] = np.ma.masked
        for x in (2.00, 2.25, 2.50, 2.75, 3.00):
            expected = None if x == 3.00 else 2
            self.assertEqual(value_at_index(array, x, interpolate=False), expected)


class TestVstackParams(unittest.TestCase):
    def test_vstack_params(self):
        a = P('a', array=np.ma.arange(10))
        b = np.ma.arange(10, 20)
        a.array[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        c = None
        assert_array_equal(
            np.ma.filled(vstack_params(a), 99),
            np.array([[99, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        )
        # test mixed types (Parameter, Masked Array, None)
        assert_array_equal(
            np.ma.filled(vstack_params(None, a, b, c), 99),
            np.array([[99,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                      [99, 11, 12, 13, 14, 15, 16, 17, 18, 99]])
        )
        self.assertRaises(ValueError, vstack_params, None, None, None)


class TestVstackParamsWhereState(unittest.TestCase):
    def test_vstack_only_one_param(self):
        # typical test
        m1 = M(array=MappedArray(
            np.ma.array([2]*8 + [3]*2), values_mapping={1:'one', 2:'two'}))
        m2 = M(array=MappedArray(
            np.ma.array([0]*5 + [1]*5), values_mapping={1:'one', 2:'two'}))
        m3 = None
        res = vstack_params_where_state((m1, 'two'),  # first 8
                                        (m2, 'one'),  # last 5
                                        (m3, 'one'))  # ignored as None
        self.assertTrue(np.all(
            res == [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]))
        self.assertTrue(np.all(
            res.all(axis=0) == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]))

        # only one array with data
        res = vstack_params_where_state((m1, 'two'),  # has data
                                        (m3, 'one'))  # has no data
        self.assertTrue(np.all(
            res == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]))

        # no arrays
        self.assertRaises(ValueError, vstack_params_where_state,
                          (None, 'blah'), (None, 'blah'))

#-----------------------------------------------------------------------------
#Tests for Atmospheric and air speed calculations derived from AeroCalc test
#suite. Changes relate to simplification of units and translation to Numpy.
#-----------------------------------------------------------------------------
class TestAlt2Press(unittest.TestCase):
    def test_01(self):
        # Truth values from NASA RP 1046
        Value = alt2press(np.ma.array([5000]))
        Truth = np.ma.array([843.0725884]) # mBar
        #self.assertAlmostEqual(Value, Truth)
        ma_test.assert_masked_array_almost_equal(Value, Truth)

    def test_02(self):
        # Truth values from aerospaceweb
        Value = alt2press(25000)
        Truth = 376.0087326
        self.assertAlmostEqual(Value, Truth, delta = 1e-5)

    def test_03(self):
        # Truth values from aerospaceweb
        Value = alt2press(45000)
        Truth = 147.4755452
        self.assertAlmostEqual(Value, Truth, delta = 1e-5)

    def test_04(self):
        # Truth values from NASA RP 1046
        Value = alt2press(ut.convert(25000, ut.METER, ut.FT))
        Truth = 25.492
        # Wide tolerance as we're not going to be using this for
        # commercial air transport !
        self.assertAlmostEqual(Value, Truth, delta = 1e-0)


class TestAlt2PressRatio(unittest.TestCase):
    def test_01(self):
        Value = alt2press_ratio(0)
        self.assertEqual(Value, 1.0)

    def test_02(self):
        # Truth values from NASA RP 1046
        Value = alt2press_ratio(-1000)
        Truth = 2193.82 / 2116.22
        self.assertAlmostEqual(Value, Truth, delta = 1e-5)

    def test_03(self):
        # Truth values from NASA RP 1046
        Value = alt2press_ratio(ut.convert(20000, ut.METER, ut.FT))
        Truth = 5474.87 / 101325
        self.assertAlmostEqual(Value, Truth, delta = 1e-5)

    def test_04(self):
        # Typical value at 25,000 ft
        # From Aerospace.web
        # ratio = (1 - h/145442)^(5.255876)
        Value = alt2press_ratio(25000)
        Truth = np.power(1.0-25000./145442.,5.255876)
        self.assertAlmostEqual(Value, Truth)


class TestAirDensity(unittest.TestCase):
    def test_01(self):
        Value = air_density(0.0, 15.0)
        self.assertEqual(Value, 1.225)

    def test_02(self):
        Value = air_density(20000.0, -24.6240)
        self.assertAlmostEqual(Value, 0.652694, places=5)

    def test_03(self):
        Value = air_density(10000.0, 20.1880)
        self.assertAlmostEqual(Value, 0.827539, places=5)


class TestCas2Dp(unittest.TestCase):

    # Many AeroCalc tests using different units removed as we are
    # standardising the units for use inside the algorithms.
    # Test 08a added to include alternative Truth value
    # Test 17 added to confirm operation in typical takeoff/landing speeds

    def test_08(self):

        # 244 kt to pa
        # truth value from NASA RP 1046

        Value = cas2dp(244)
        Truth = 99.837
        self.assertAlmostEqual(Value, Truth, delta = 1e-2)

    def test_08a(self):

        # 244 kt to pa
        # truth value from aerospaceweb

        Value = cas2dp(244)
        Truth = 99.8355
        self.assertAlmostEqual(Value, Truth, delta = 1e-3)

    def test_16(self):

        # 1000 kt to pa
        # truth value from NASA RP 1046

        self.assertRaises(ValueError, cas2dp, 1000)

    def test_17(self):

        # 120 kt
        # truth value from aerospaceweb

        Value = cas2dp(120)
        Truth = 23.5351
        self.assertAlmostEqual(Value, Truth, delta = 1e-2)


class TestCasAlt2Mach(unittest.TestCase):
    def test_cas_alt2mach(self):
        cas = np.ma.array(data=[300.0, 300, 300, 300], mask=[0,1,0,0])
        alt_ft = np.ma.array(data=[0, 0, 0, 30000.0], mask=[0,0,1,0])
        result = cas_alt2mach(cas, alt_ft)
        expected = np.ma.array(data=[0.4535, 99, 99, 0.79],mask=[0,1,1,0])
        ma_test.assert_masked_array_approx_equal(result[:3], expected[:3], decimal=4)
        # this test function requires an array, not a single value, to work. Hence the odd index.
        ma_test.assert_masked_array_approx_equal(result[3:], expected[3:], decimal=3)


class TestDelay(unittest.TestCase):

    def test_basic(self):
        array=np.ma.arange(5)
        result = delay(array,2.0)
        expected = np.ma.array(data=[0,0,0,1,2],
                               mask=[1,1,0,0,0])
        np.testing.assert_array_equal(result, expected)

    def test_no_delay(self):
        array=np.ma.arange(5)
        result = delay(array,0.0)
        ma_test.assert_masked_array_approx_equal(result, array)

    def test_excessive(self):
        array=np.ma.arange(5)
        result = delay(array,20)
        expected = np.ma.array(np.zeros(5), mask=True)
        ma_test.assert_masked_array_approx_equal(result, expected)

class TestDp2Cas(unittest.TestCase):

    def test_dp2cas(self):

        # Tests low speed and masking supersonic cases in one go.

        # 244 kt in pa
        # truth value from NASA RP 1046

        # 1000 kt in pa
        # truth value from NASA RP 1046

        Value = dp2cas(np.ma.array([99.837, 2490.53]))
        Truth = np.ma.array(data=[244.0, 1000.0], mask=[False,True])
        ma_test.assert_masked_array_almost_equal (Value,Truth,decimal=2)


class TestDp2Tas(unittest.TestCase):

    def test_dp2tas_1(self):
        # "Null" case to start
        Value = dp2tas(0.0, 0.0, 15.0)
        Truth = 0.0
        self.assertEqual(Value, Truth)

    def test_dp2tas_2(self):
        # Trivial case = 200 KIAS
        Value = dp2tas(66.3355, 20000.0, -24.586)
        Truth = 270.4489
        self.assertAlmostEqual(Value, Truth, delta = 1)

    def test_dp2tas_3(self):
        # 200 KIAS at ISA + 20C
        Value = dp2tas(66.3355, 20000.0, -13.4749)
        Truth = 276.4275
        self.assertAlmostEqual(Value, Truth, delta = 1)

    def test_dp2tas_4(self):
        # Speed up to 300 KIAS and higher
        Value = dp2tas(153.5471, 30000.0, -44.35)
        Truth = 465.6309
        self.assertAlmostEqual(Value, Truth, delta = 1)

    def test_dp2tas_5(self):
        # Still 300 KIAS but Stratospheric
        Value = dp2tas(153.5469, 45000.0, -56.5)
        Truth = 608.8925
        self.assertAlmostEqual(Value, Truth, delta = 1)


class TestMachTat2Sat(unittest.TestCase):

    def test_machtat2sat(self):
        # Mach 0.5, 15 deg C, K = 0.5
        mach = np.ma.array(data=[0.5, 0.5], mask=[False, True])
        tat = np.ma.array([15.0, 15.0])
        sat = machtat2sat(mach, tat, recovery_factor = 0.5)
        truth = 7.97195121951
        self.assertAlmostEqual(sat[0], truth, delta = 1e-5)
        # Test did check masked value result. Reason unknown, so dropped.
        #self.assertAlmostEqual(sat.data[1], truth, delta = 1e-5)


class TestMachSat2Tat(unittest.TestCase):

    def test_machsat2tat(self):
        mach = np.ma.array(data=[0.5, 0.5])
        sat = np.ma.array(data=[7.97195121951, 7.97195121951])
        value = machsat2tat(mach, sat, recovery_factor = 0.5)
        truth = 15.0
        self.assertAlmostEqual(value[0], truth, delta = 1e-5)


class TestAlt2Sat(unittest.TestCase):

    def test_alt2sat(self):

        value = alt2sat(np.ma.array([0.0, 15000.0, 45000.0]))
        truth = np.ma.array(data=[15.0, -14.718, -56.5])
        ma_test.assert_masked_array_almost_equal(value, truth)

class TestAltDev2Alt(unittest.TestCase):

    def test_on_ISA(self):
        alt = np.ma.array([0.0, 3000.0, 30000.0])
        dev = np.ma.array([0.0, 0.0, 0.0])
        # On ISA conditions, no change in altitude.
        ma_test.assert_masked_array_almost_equal(alt_dev2alt(alt, dev), alt)

    def test_off_ISA(self):
        alt = np.ma.array([1400.0, 7000, 30000])
        # We just add the temperature offset to the ISA conditions.
        dev = np.ma.array([-65.0,-20.0,-5.0])
        result = np.ma.array([1084, 6514, 29479])
        ma_test.assert_masked_array_almost_equal(alt_dev2alt(alt, dev), result,
                                                 decimal=0)


class TestFromIsa(unittest.TestCase):

    def test_from_ISA(self):
        self.assertEqual(from_isa(0.0, 15.0), 0.0)
        self.assertEqual(from_isa(0.0, 0.0), -15.0)
        self.assertAlmostEqual(from_isa(5000.0, 0.0), -5.094)
        self.assertAlmostEqual(from_isa(5000.0, -30), -35.094)
        self.assertAlmostEqual(from_isa(40000.0, 0.0), 56.5, places=3)


class TestLevelOffIndex(unittest.TestCase):
    def test_level_off_index_index_array_too_small(self):
        self.assertEqual(level_off_index(np.ma.arange(5), 1, 10, 1), None)
        self.assertEqual(
            level_off_index(np.ma.arange(60), 1, 10, 1, _slice=slice(2,11)),
            None)

    def test_level_off_index_index_all_unstable(self):
        self.assertEqual(level_off_index(np.ma.arange(10), 1, 2, 1), None)

    def test_level_off_index_index_all_stable(self):
        self.assertEqual(level_off_index(np.ma.arange(10), 1, 2, 5), 2)
        self.assertEqual(level_off_index(np.ma.arange(10), 1, 2, 5,
                                          _slice=slice(5,9)), 7)

    def test_level_off_index_basic(self):
        array = np.ma.array([0,2,4,6,8,10,8,6,5,4.5,5,4.5,5])
        self.assertEqual(level_off_index(array, 1, 3, 2), 10)
        self.assertEqual(level_off_index(array, 1, 3, 2, include_window=False),
                         7)

    def test_level_off_index_basic_masked(self):
        array = np.ma.array([0,0,0,0,8,10,8,6,5,4.5,5,4.5,5, 7, 9, 11, 13, 15],
                            mask=[True] * 4 + [False] * 9 + [True] * 5)
        self.assertEqual(level_off_index(np.ma.repeat(array, 2), 1, 6, 2), 20)

    def test_level_off_index_eng_n3(self):
        array = load_compressed(
            os.path.join(test_data_path, 'eng_n3.npz'))
        index = level_off_index(array, 1, 10, 1, _slice=slice(60, 750))
        self.assertEqual(index, 100)
        index = level_off_index(array, 1, 10, 1, _slice=slice(3625, None))
        self.assertEqual(index, 3674)

    def test_level_off_masked(self):
        array = load_compressed(
            os.path.join(test_data_path, 'level_off_index_eng_n3.npz'))
        index = level_off_index(array, 1, 10, 1)
        self.assertEqual(index, 28)


class TestDpOverP2mach(unittest.TestCase):

    def test_dp_over_p2mach(self):

        value = dp_over_p2mach(np.ma.array([.52434, .89072, 1.1]))

        # truth values from NASA RP 1046

        truth = np.ma.array(data=[0.8, 0.999, 1.0], mask=[False, False, True])
        ma_test.assert_masked_array_almost_equal (value, truth, decimal=3)

    def test_dp_over_p2mach_masked_dp_over_p(self):
        # In cases of data corruption, large or negative dp/p values may arise.
        value = dp_over_p2mach(np.ma.array(data=[-0.52434, .89072, 51.1],
                                           mask=[1,0,1]))
        truth = np.ma.array(data=[0.0, 0.999, 1.0], mask=[True, False, True])
        ma_test.assert_masked_array_almost_equal(value,truth, decimal=3)


class TestPress2Alt(unittest.TestCase):
    def test_press2alt(self):
        value = press2alt(np.ma.array([14.696, 10.108, 4.3727, 2.1490]))
        truth = np.ma.array(data=[0.0, 10000, 30000, 45000])
        ma_test.assert_masked_array_almost_equal (value,truth, decimal=-3)
##Tests used to prove low level functions for press2alt
##class TestPress2AltGradient(unittest.TestCase):
    ##def test_01(self):
        ##Value = press2alt_gradient(np.ma.array([1013.25,  696.817, 300.896]))
        ##Truth = np.ma.array(data=[0.0, 10000, 30000])
        ##ma_test.assert_masked_array_almost_equal (Value,Truth, decimal=1)
##class TestPress2AltIsothermal(unittest.TestCase):
    ##def test_01(self):
        ##Value = press2alt_isothermal(np.ma.array([188.23, 148.17]))
        ##Truth = np.ma.array(data=[40000, 45000])
        ##ma_test.assert_masked_array_almost_equal (Value,Truth, decimal=-3)


class TestSecondWindow(unittest.TestCase):

    def test_second_window_too_large_for_array(self):
        ma_test.assert_masked_array_almost_equal(
            second_window(np.ma.arange(0, 20), 2, 45),
            np_ma_masked_zeros(20))

    def test_three_second_window_incrementing(self):
        ma_test.assert_masked_array_almost_equal(
            second_window(np.ma.arange(0, 20, 1), 2, 3),
            np.ma.masked_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] + [0] * 6,
                               mask=[False] * 14 + [True] * 6))

    def test_three_second_window_incrementing_large(self):
        mask = [True] * 6 + [False] * 10 + [True] * 4 + [False] * 5 + [True] * 5
        array = np.ma.array(np.arange(0, 30, 1), mask=mask)
        expected_mask = [True] * 6 + [False] * 4 + [True] * 20
        ma_test.assert_masked_array_almost_equal(
            second_window(array, 2, 3),
            np.ma.masked_array([0] * 6 + [6, 7, 8, 9] + [0] * 20,
                               mask=expected_mask))

    def test_three_second_window_decrementing(self):
        ma_test.assert_masked_array_almost_equal(
            second_window(np.ma.arange(20, 0, -1), 2, 3),
            np.ma.masked_array([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7] + [0] * 6,
                               mask=[False] * 14 + [True] * 6))

    def test_three_second_window_peak(self):
        data = np.ma.concatenate([np.arange(0, 5, 0.5),
                                  np.arange(5, 0, -0.5)])
        expected = np.ma.masked_array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.5, 3.5, 3.5,
                                       3.5, 3.5, 3.5] + [0] * 6,
                                      mask=np.concatenate((np.zeros(14), np.ones(6))))
        # Test multiple frequencies abiding to 2/frequency.
        ma_test.assert_masked_array_almost_equal(
            second_window(data, 2, 3),
            expected)
        ma_test.assert_masked_array_almost_equal(
            second_window(data, 1, 6),
            expected)
        ma_test.assert_masked_array_almost_equal(
            second_window(data, 1, 5, extend_window=True),
            expected)

    def test_second_window_invalid_args(self):
        self.assertRaises(ValueError, second_window, np.ma.arange(10), 0.25, 1)
        self.assertRaises(ValueError, second_window, np.ma.arange(10), 0.5, 3)
        self.assertRaises(ValueError, second_window, np.ma.arange(10), 1, 3)
        self.assertRaises(ValueError, second_window, np.ma.arange(10), 2, 3.5)
        self.assertRaises(ValueError, second_window, np.ma.arange(10), 10, 3.1292577)

    def test_three_second_window_basic_trough(self):
        ma_test.assert_masked_array_almost_equal(
            second_window(np.ma.concatenate([np.arange(10, 0, -0.5),
                                             np.arange(0, 10, 0.5)]), 2, 3),
            np.ma.masked_array([10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5,
                                4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.5, 1.5, 1.5,
                                1.5, 1.5, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5] +
                               [0] * 6, mask=np.concatenate((np.zeros(34), np.ones(6)))))

    def test_three_second_window_trough(self):
        ma_test.assert_masked_array_almost_equal(
            second_window(np.ma.concatenate([np.arange(5, 0, -0.5),
                                             np.arange(0, 10, 1)]), 2, 3),
            np.ma.array([5, 4.5, 4, 3.5, 3, 2.5, 2, 2, 2, 2, 2, 2, 2, 3] + [0] * 6, mask=np.concatenate((np.zeros(14), np.ones(6)))))

    def test_three_second_window_masked(self):
        masked_data = np.ma.arange(10, 0, -0.5)
        masked_data[8:10] = np.ma.masked
        ma_test.assert_masked_array_almost_equal(
            second_window(masked_data, 2, 3),
            np.ma.array([10, 9.5] + [0] * 8 + [5, 4.5, 4, 3.5] + [0] * 6,
                        mask=[False] * 2 + [True] * 8 + [False] * 4 + [True] * 6),
        )

    def test_five_second_window_basic_trough(self):
        ma_test.assert_masked_array_almost_equal(
            second_window(np.ma.concatenate([np.arange(10, 0, -0.5),
                                             np.arange(0, 10, 0.5)]), 2, 5),
            np.ma.masked_array([10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0,
                                5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.5, 2.5,
                                2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0,
                                3.5, 4.0, 4.5] + [0] * 10,
                               mask=np.concatenate((np.zeros(30), np.ones(10)))))

    def test_five_second_window_trough(self):
        ma_test.assert_masked_array_almost_equal(
            second_window(np.ma.concatenate([np.arange(10, 0, -0.5),
                                             np.arange(0, 20, 1)]), 2, 5),
            np.ma.array([10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0,
                         4.5, 4.0, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5,
                         3.5, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] + [0] * 10,
                        mask=np.concatenate((np.zeros(30), np.ones(10)))))

    @unittest.skip('Not Implemented')
    def test_three_second_window(self):
        self.assertTrue(False)
        #amv2 = load_compressed('...airspeed_minus_v2.npz')
        #ma_test.assert_masked_array_almost_equal (
        #    sample_window(amv2, 3),
        #    [10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 4, 6, 8, 10, 12, 14, 16])

    def test_three_second_window_with_real_data(self):
        sw = load(os.path.join(test_data_path, 'second_window.nod'))
        res = second_window(sw.array, sw.frequency, 3)
        self.assertEqual(np.ma.count(res), 40972)


class TestLookupTable(unittest.TestCase):

    class Expected(object):
        tables = {'v2': {}}
        fallback = {'vref': {}}

    def setUp(self):
        self.attrs = (
            A('Model', 'B737-333'),
            A('Series', 'B737-300'),
            A('Family', 'B737 Classic'),
            A('Engine Type', 'CFM56-3B1'),
            A('Engine Series', 'CRM56-3'),
        )
        self.values = [a.value for a in self.attrs]

    @patch('analysis_engine.library.at')
    @patch.object(P, 'warning')
    def test_lookup_table__not_found(self, log, at):
        at.get_vspeed_map.side_effect = KeyError
        table = lookup_table(P, 'v2', *self.attrs)
        at.get_vspeed_map.assert_called_once_with(*self.values)
        self.assertEqual(log.call_count, 1)
        self.assertEqual(table, None)

    @patch('analysis_engine.library.at')
    @patch.object(P, 'warning')
    def test_lookup_table__found_standard_table(self, log, at):
        at.get_vspeed_map.return_value = self.Expected
        table = lookup_table(P, 'v2', *self.attrs)
        at.get_vspeed_map.assert_called_once_with(*self.values)
        self.assertEqual(log.call_count, 0)
        self.assertIsInstance(table, self.Expected)

    @patch('analysis_engine.library.at')
    @patch.object(P, 'warning')
    def test_lookup_table__found_fallback_table(self, log, at):
        at.get_vspeed_map.return_value = self.Expected
        table = lookup_table(P, 'vref', *self.attrs)
        at.get_vspeed_map.assert_called_once_with(*self.values)
        self.assertEqual(log.call_count, 0)
        self.assertIsInstance(table, self.Expected)

    @patch('analysis_engine.library.at')
    @patch.object(P, 'warning')
    def test_lookup_table__found_class_without_table(self, log, at):
        at.get_vspeed_map.return_value = self.Expected
        table = lookup_table(P, 'vmo', *self.attrs)
        at.get_vspeed_map.assert_called_once_with(*self.values)
        self.assertEqual(log.call_count, 1)
        self.assertEqual(table, None)

class TestNearestRunway(unittest.TestCase):

    '''
    Tests for Nearest Runway
    '''
    airports = yaml.load(open(os.path.join(test_data_path, 'airports.yaml'), 'rb'),
                         Loader=yaml.Loader)
    _airports = airports['airports']
    _expected = {
        '001': {
            'id': 10816,
            'identifier': u'27*',
            'magnetic_heading': 272.0,
            'start': {
                'elevation': 78,
                'latitude': 51.47767519999998,
                'longitude': -0.43326099999999224
            },
            'end': {
                'elevation': 78,
                'latitude': 51.47748872593225,
                'longitude': -0.48499105282286614
            },
            'glideslope': {
                'angle': 3.0,
                'elevation': 78,
                'latitude': 51.47883099999999,
                'longitude': -0.4382060000000034,
                'threshold_distance': 1124
            },
            'localizer': {
                'beam_width': 4.5,
                'elevation': 78,
                'frequency': 110300.00,
                'heading': 271,
                'is_offset': False,
                'latitude': 51.477467,
                'longitude': -0.49373899999999565,
            },
            'strip': {'id': 5408, 'length': 12799, 'surface': u'ASP', 'width': 164},
        },
        '002': {
            'id': 10814,
            'identifier': u'27L',
            'magnetic_heading': Decimal('272.0'),
            'start': {
                'elevation': 77,
                'latitude': 51.46495199999997,
                'longitude': -0.4340779999999912
            },
            'end': {
                'elevation': 75,
                'latitude': 51.464765624976685,
                'longitude': -0.4822469198065629
            },
            'glideslope': {
                'angle': 3.0,
                'elevation': 77,
                'latitude': 51.46376501247372,
                'longitude': -0.43896011904906,
                'threshold_distance': 1072
            },
            'localizer': {
                'beam_width': 4.5,
                'elevation': 75,
                'frequency': Decimal('109500.00'),
                'heading': 271,
                'is_offset': False,
                'latitude': 51.46476099999998,
                'longitude': -0.49111899999999203
            },
            'strip': {'id': 5407, 'length': 12001, 'surface': u'ASP', 'width': 164}
        },
        '003': {
            'id': 5,
            'identifier': u'18',
            'magnetic_heading': 182.0,
            'start': {
                'latitude': 41.997416999999999,
                'longitude': -87.900406000000004,
            },
            'end': {
                'latitude': 41.982785999999997,
                'longitude': -87.900400000000005,
            },
            'strip': {
                'id': 3,
                'length': 5332,
                'surface': u'ASP',
                'width': 150,
            },
        },
        '004': {
            'id': 13411,
            'identifier': u'13*',
            'magnetic_heading': Decimal('137.2'),
            'start': {
                'latitude': 41.788069,
                'elevation': 620,
                'longitude': -87.758803
            },
            'end': {
                'latitude': 41.780772,
                'elevation': 620,
                'longitude': -87.748553
            },
            'strip': {
                'width': 60,
                'length': 3859,
                'id': 6706,
                'surface': u'CON'
            },
            'localizer': {
                'elevation': 620,
                'is_offset': False
            },
            'glideslope': {'elevation': 620}
        },
        '005': {
            'id': 6,
            'identifier': u'36',
            'magnetic_heading': 2.0,
            'start': {
                'latitude': 41.982785999999997,
                'longitude': -87.900400000000005,
            },
            'end': {
                'latitude': 41.997416999999999,
                'longitude': -87.900406000000004,
            },
            'strip': {
                'id': 3,
                'length': 5332,
                'surface': u'ASP',
                'width': 150,
            },
        },
        '006': u'Magnetic heading is required to determine runway.',
        '007': u'Heading must be numeric.',
        '008': u'Heading must be between 0° and 360°: \'%s\'.',
        '009': u'Expected comma-separated latitude and longitude in decimal degrees.',
        '010': u'Latitude must be specified in decimal degrees.',
        '011': u'Longitude must be specified in decimal degrees.',
        '012': u'Latitude must be between -90° and 90°: \'%s\'.',
        '013': u'Longitude must be between -180° and 180°: \'%s\'.',
        '014': u'No runway found at airport for the magnetic heading [%s]',
        '015': u'ILS frequency must be numeric.',
        '016': u'Localizer ILS frequency is out-of-range.',
        '017': {
            'end': {
                'elevation': 119,
                'latitude': 43.30633764082481,
                'longitude': -2.9249050434919237
            },
            'glideslope': {
                'angle': 3.4,
                'elevation': 120,
                'latitude': 43.29787499999998,
                'longitude': -2.9050420000000097,
                'threshold_distance': 902
            },
            'id': 9916,
            'identifier': u'30',
            'localizer': {
                'beam_width': 4.5,
                'elevation': 119,
                'frequency': Decimal('110300.00'),
                'heading': 300,
                'is_offset': False,
                'latitude': 43.30729199999997,
                'longitude': -2.927516999999999
            },
            'magnetic_heading': Decimal('298.0'),
            'start': {
                'elevation': 120,
                'latitude': 43.29770864269298,
                'longitude': -2.901346028469935
            },
            'strip': {'id': 4958, 'length': 8530, 'surface': u'ASP', 'width': 148},
        },
        '018': {
            'id': 11172,
            'identifier': u'20',
            'magnetic_heading': Decimal('201.0'),
            'start': {'elevation': 44,
                      'latitude': 50.95712713152393,
                      'longitude': -1.352988127649349
                      },
            'end': {'elevation': 31,
                    'latitude': 50.94360437892113,
                    'longitude': -1.3605209484101468
                    },
            'glideslope': {'angle': 3.1,
                           'elevation': 44,
                           'frequency': Decimal('110750.00'),
                           'latitude': 50.955138726435244,
                           'longitude': -1.355741592590264,
                           'threshold_distance': 918
                           },
            'localizer': {'beam_width': 4.5,
                          'elevation': 31,
                          'frequency': Decimal('110750.00'),
                          'heading': 201,
                          'is_offset': False,
                          'latitude': 50.942072718495965,
                          'longitude': -1.3613830932541067
                          },
            'strip': {'id': 5586, 'length': 5653, 'surface': u'ASP', 'width': 120}},
    }

    def test_find_nearest_with_valid_heading(self):
        '''
        Test finding nearest runway with valid magnetic heading.
        '''
        # TODO: switch for airport with single runway
        runway = nearest_runway(self._airports['001'], 270.5)
        self.assertEqual(runway, self._expected['001'])

    def test_find_nearest_with_invalid_heading(self):
        '''
        Test finding nearest runway with invalid magnetic heading.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['008'],
                                callableObj=nearest_runway, args=[self._airports['001'], 'ABC'])

    def test_find_nearest_with_positive_overflow_heading(self):
        '''
        Test finding nearest runway with positive overflowed magnetic heading.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['008'],
                                callableObj=nearest_runway, args=[self._airports['001'], 361])

    def test_find_nearest_with_negative_overflow_heading(self):
        '''
        Test finding nearest runway with negative overflowed magnetic heading.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['008'],
                                callableObj=nearest_runway, args=[self._airports['001'], -1])

    def test_find_nearest_without_heading(self):
        '''
        Test finding nearest runway without a magnetic heading.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['006'],
                                callableObj=nearest_runway, args=[self._airports['001'], None])

    def test_find_nearest_with_valid_coordinates(self):
        '''
        Test finding nearest runway with valid coordinates.
        '''
        runway = nearest_runway(self._airports['001'], 270.5, latitude=51.464927, longitude=-0.440458)
        self.assertEqual(runway, self._expected['002'])

    def test_find_nearest_with_invalid_latitude(self):
        '''
        Test finding nearest runway with invalid latitude.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['010'],
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 270.5],
                                kwargs={'latitude':'A', 'longitude':0})

    def test_find_nearest_with_positive_overflow_latitude(self):
        '''
        Test finding nearest runway with positive overflowed latitude.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['012'],
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 270.5],
                                kwargs={'latitude':100, 'longitude':0})

    def test_find_nearest_with_negative_overflow_latitude(self):
        '''
        Test finding nearest runway with negative overflowed latitude.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['012'],
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 270.5],
                                kwargs={'latitude':-100, 'longitude':0})

    def test_find_nearest_with_invalid_longitude(self):
        '''
        Test finding nearest runway with invalid longitude.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['011'],
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 270.5],
                                kwargs={'latitude':0, 'longitude':'A'})

    def test_find_nearest_with_positive_overflow_longitude(self):
        '''
        Test finding nearest runway with positive overflowed longitude.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['013'],
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 270.5],
                                kwargs={'latitude':0, 'longitude':200})

    def test_find_nearest_with_negative_overflow_longitude(self):
        '''
        Test finding nearest runway with negative overflowed longitude.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['013'],
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 270.5],
                                kwargs={'latitude':0, 'longitude':-200})

    def test_find_nearest_with_one_coordinate(self):
        '''
        Test finding nearest runway with one coordinate.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['009'],
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 270.5],
                                kwargs={'latitude':0})

    def test_find_nearest_with_unknown_heading(self):
        '''
        Test finding nearest runway with unknown magnetic heading.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['014'] % 180.0,
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 180.0],
                                kwargs={'latitude':0})

    def test_find_nearest_with_valid_ilsfreq_1(self):
        '''
        Test finding nearest runway with valid localizer frequency.
        '''
        runway = nearest_runway(self._airports['001'], 270.5, ilsfreq=109.5)
        self.assertEqual(runway, self._expected['002'])

    def test_find_nearest_with_valid_ilsfreq_2(self):
        '''
        Test finding nearest runway with valid localizer frequency.
        '''
        runway = nearest_runway(self._airports['001'], 270.5, ilsfreq=109.51)
        self.assertEqual(runway, self._expected['002'])

    def test_find_nearest_with_invalid_ilsfreq(self):
        '''
        Test finding nearest runway with invalid localizer frequency.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['015'],
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 270.5],
                                kwargs={'ilsfrq':'ABC'})

    def test_find_nearest_with_out_of_range_ilsfreq_1(self):
        '''
        Test finding nearest runway with out-of-range localizer frequency.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['016'],
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 270.5],
                                kwargs={'ilsfrq':105000})

    def test_find_nearest_with_out_of_range_ilsfreq_2(self):
        '''
        Test finding nearest runway with out-of-range localizer frequency.
        '''
        self.assertRaisesRegex(Exception,
                                self._expected['016'],
                                callableObj=nearest_runway,
                                args=[self._airports['001'], 270.5],
                                kwargs={'ilsfrq':115000})

    def test_find_nearest_with_unknown_ilsfreq(self):
        '''
        Test finding nearest runway with unknown localizer frequency.
        '''
        runway = nearest_runway(self._airports['001'], 270.5, ilsfreq=108.95)
        self.assertEqual(runway, self._expected['001'])

    def test_find_nearest_with_unknown_ilsfreq_fallback_coordinates(self):
        '''
        Test finding nearest runway with unknown localizer frequency.
        '''
        runway = nearest_runway(self._airports['001'], 270.5, latitude=51.464927, longitude=-0.440458, ilsfreq=108.95)
        self.assertEqual(runway, self._expected['002'])

    def test_find_nearest_with_single_runway(self):
        '''
        Test finding nearest runway when only a single runway on heading.
        '''
        runway = nearest_runway(self._airports['005'], 200.5)
        self.assertEqual(runway, self._expected['018'])

    def test_find_nearest_with_triple_runway(self):
        '''
        Test finding nearest runway when only a single runway on heading.
        '''
        runway = nearest_runway(self._airports['003'], 130.5, hint='landing')
        self.assertEqual(runway, self._expected['004'])

    def test_find_nearest_with_heading_wrap_around(self):
        '''
        Test finding nearest runway when heading must wrap from 360° --> 0°.
        '''
        runway = nearest_runway(self._airports['002'], 359.5)
        self.assertEqual(runway, self._expected['005'])

    def test_find_nearest_with_runway_conflict_error(self):
        '''
        Test finding nearest runway when idents conflict because too similar.

        Note: This used to raise an error. Now we just attempt to choose the
              runway that has the closest heading to the one provided. If we
              guess wrongly, the achieved flight record will just have to be
              updated and we will have to reprocess the flight.
        '''
        #  TODO: test case to fit description
        runway = nearest_runway(self._airports['004'], 301.640625, hint='landing')
        self.assertEqual(runway, self._expected['017'])


class TestWrapArray(unittest.TestCase):
    def test_wrap_array_heading(self):
        array = np.ma.arange(720)
        wrapped = wrap_array('Heading', array)
        self.assertLessEqual(wrapped.max(), 360)
        self.assertGreaterEqual(wrapped.min(), 0)

        array = -array
        wrapped = wrap_array('Heading', array)
        self.assertLessEqual(wrapped.max(), 360)
        self.assertGreaterEqual(wrapped.min(), 0)

    def test_wrap_array_longitude(self):
        # flying from -180deg to the East
        array = np.ma.arange(-180, 720)
        wrapped = wrap_array('Longitude', array)
        self.assertLessEqual(wrapped.max(), 180)
        self.assertGreaterEqual(wrapped.min(), -180)

        # flying from 180deg to the West
        array = -array
        wrapped = wrap_array('Longitude', array)
        self.assertLessEqual(wrapped.max(), 180)
        self.assertGreaterEqual(wrapped.min(), -180)


class TestMb2Ft(unittest.TestCase):
    def test_mb2ft(self):
        self.assertAlmostEqual(mb2ft(1013), 6.83, places=2)
        self.assertAlmostEqual(mb2ft(1020), -183.77, places=2)
        self.assertAlmostEqual(mb2ft(1024), -292.21, places=2)


class TestMaxMaintainedValue(unittest.TestCase):
    def test_example_max_maintained_value(self):
        arrays = np.ma.array([1,2,3,4,3,4,3,4,3,2,5,2])
        index, value = max_maintained_value(arrays, 5, 1)
        self.assertEqual(index, 2)
        self.assertEqual(value, 3)

    def test_short_down_spike(self):
        arrays = np.ma.array([1, 8, 8, 8, 1, 8, 1, 7, 7, 7, 7, 7, 9])
        index, value = max_maintained_value(arrays, 5, 1)
        self.assertEqual(index, 7)
        self.assertEqual(value, 7)

    def test_all_masked(self):
        arrays = np.ma.array([1, 8, 8, 8, 1, 8, 1, 7, 7, 7, 7, 7, 9], mask=True)
        index, value = max_maintained_value(arrays, 5, 1)
        self.assertEqual(index, None)
        self.assertEqual(value, None)

    def test_partly_masked(self):
        arrays = np.ma.array([1, 7, 7, 7, 7, 7, 9, 1, 8, 8, 8, 1, 8])
        arrays[1:3] = np.ma.masked
        index, value = max_maintained_value(arrays, 5, 1)
        self.assertEqual(index, 3)
        self.assertEqual(value, 1)

    def test_slice_outside_array(self):
        arrays = np.ma.array([1,2,3,4,3,4,3,4,3,2,5,2])
        arrays[2] = np.ma.masked
        index, value = max_maintained_value(arrays, 5, 1, slice(100, 200))
        self.assertEqual(index, None)
        self.assertEqual(value, None)

    def test_max_maintained_value(self):
        eng_torq_max=load(os.path.join(test_data_path,
                                       'ebe456663820_eng_torq_max.nod'))
        go_arounds=load(os.path.join(test_data_path,
                                     'ebe456663820_go_arounds.nod'))
        phase = go_arounds.get_slices()[0]
        expected = (
            (10, 2922, 148.697),
            (20, 2920, 146.5),
            (300, 2733.5, 4.248)
        )
        hz = eng_torq_max.frequency
        for sec, idx, val in expected:
            index, value = max_maintained_value(eng_torq_max.array, sec, hz, phase)
            self.assertAlmostEqual(index, idx, places=0)
            self.assertAlmostEqual(value, val, places=3)


class TestFindClimbCruiseDescent(unittest.TestCase):
    def test_climb_cruise_descent_start_midflight(self):
        # This test will find out if we can separate the two humps on this camel
        testwave = np.ma.concatenate((np.ones(5) * 15000, np.arange(15000, 1000, -1000)))
        camel = find_climb_cruise_descent(testwave)
        self.assertEqual(len(camel), 1)
        self.assertEqual(camel[0], slice(None, 18))

    def test_climb_cruise_descent_end_midflight(self):
        # This test will find out if we can separate the two humps on this camel
        testwave = np.ma.concatenate((np.arange(1000,15000,1000), np.ones(5) * 15000))
        camel = find_climb_cruise_descent(testwave)
        self.assertEqual(len(camel), 1)
        self.assertEqual(camel[0], slice(0, None))

    def test_climb_cruise_descent_all_high(self):
        # This test will find out if we can separate the two humps on this camel
        testwave = np.ma.ones(5) * 15000
        camel = find_climb_cruise_descent(testwave)
        self.assertEqual(len(camel), 0)

    def test_climb_cruise_descent_one_humps(self):
        # This test will find out if we can separate the two humps on this camel
        testwave = np.ma.cos(np.arange(0, 3.14 * 2, 0.1)) * -3000 + 12500
        camel = find_climb_cruise_descent(testwave)
        self.assertEqual(len(camel), 1)
        self.assertEqual(camel[0], slice(0, len(testwave)-1))

    def test_climb_cruise_descent_two_humps(self):
        # This test will find out if we can separate the two humps on this camel
        testwave = np.ma.cos(np.arange(0, 3.14 * 4, 0.1)) * -3000 + 12500
        camel = find_climb_cruise_descent(testwave)
        self.assertEqual(len(camel), 2)
        self.assertEqual(camel[0], slice(0, 63))
        self.assertEqual(camel[1], slice(63, 125))

    def test_climb_cruise_descent_three_humps(self):
        # This test will find out if we can separate the two humps on this camel
        testwave = np.ma.cos(np.arange(0, 3.14 * 6, 0.1)) * (-3000) + 12500
        camel = find_climb_cruise_descent(testwave)
        self.assertEqual(len(camel), 3)
        self.assertEqual(camel[0], slice(0, 63))
        self.assertEqual(camel[1], slice(63, 126))
        self.assertEqual(camel[2], slice(126, 188))

    def test_climb_cruise_descent_masked(self):
        # This test will find out if we can separate the two humps on this camel
        testwave = np.ma.cos(np.arange(0, 3.14 * 6, 0.1)) * (-3000) + 12500
        testwave[63:125] = np.ma.masked
        camel = find_climb_cruise_descent(testwave)
        self.assertEqual(len(camel), 2)
        self.assertEqual(camel[0], slice(0, 126))
        self.assertEqual(camel[1], slice(126, 188))

    def test_climb_cruise_descent_repair_mask(self):
        # If the Altitude STD mask isn't repaired, a spurious cycle is reported
        # by cycle_finder which results in an infinite loop.
        camel = find_climb_cruise_descent(
            load_compressed(
                os.path.join(test_data_path, 'climb_cruise_descent_alt_std.npz')
            )
        )
        self.assertEqual(len(camel), 1)


class TestMaintainAltitude(unittest.TestCase):
    def test_stay_within_50_ft(self):
        altitude = np.ma.concatenate((
            np.ma.arange(1000, 2000, 100),
            np.ones(30) * 2000,
            np.ma.arange(2000, 3000, 100),
        ))
        target = np.ma.ones(50) * 2000
        distance = altitude - target
        self.assertTrue(maintain_altitude(distance))

    def test_fly_through_target(self):
        altitude = np.ma.arange(1000, 4000, 100)
        target = np.ma.ones(30) * 2000
        distance = altitude - target
        self.assertFalse(maintain_altitude(distance))
