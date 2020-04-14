from __future__ import print_function

import mock
import numpy as np
import os.path
import pytz
import unittest

from datetime import datetime

from analysis_engine.split_hdf_to_segments import (
    _calculate_start_datetime,
    _get_normalised_split_params,
    _mask_invalid_years,
    _segment_type_and_slice,
    append_segment_info,
    calculate_fallback_dt,
    get_dt_arrays,
    get_valid_dt_slices,
    has_constant_time,
    split_segments,
    PRECISE,
)
from analysis_engine.node import M, P, Parameter

from hdfaccess.file import hdf_file

from flightdatautilities.array_operations import load_compressed
from flightdatautilities.filesystem_tools import copy_file

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')
this_year = datetime.now().year


def load_array(filename):
    return load_compressed(os.path.join(test_data_path, filename))


class MockHDF(dict):
    def __init__(self, data={}, duration=1):
        self.update(data)
        self.duration = duration


class TestInvalidYears(unittest.TestCase):
    def test_mask_invalid_years(self):
        array = np.ma.array([0, 2, 9, 10, 13, 14, 15, 88, 99,
                             101, 199, 600,
                             1950, 1990, 2000, 2001, 2010, 2013, 2014, 2015, 2999,
                             55555, 99999])
        # expecting:
        #[0 2 9 10 13 -- -- -- -- -- -- -- 1990 2000 2001 2010 2013 -- -- -- -- --]
        exp_mask = [0, 0, 0, 0, 0, 1, 1, 1, 1,
                    1, 1, 1,
                    1, 0, 0, 0, 0, 0, 1, 1, 1,
                    1, 1]
        res = _mask_invalid_years(array, latest_year=2013)
        self.assertTrue(np.all(res.mask == exp_mask))


class TestDateTimeFunctions(unittest.TestCase):
    def test_calculate_fallback_dt(self):
        hdf = mocked_hdf()('slow')
        dt = datetime(2012, 12, 12, 12, 13, 2, tzinfo=pytz.utc)
        # test no change
        new_dt = calculate_fallback_dt(hdf, datetime(2012, 12, 12, 12, 13, 2, tzinfo=pytz.utc),
                                       datetime(2012, 12, 12, 12, 13, 2, tzinfo=pytz.utc), True)
        self.assertEqual(new_dt, dt)
        # test 50 seconds (duration) earlier as relative to end
        new_dt = calculate_fallback_dt(hdf, datetime(2012, 12, 12, 12, 13, 2, tzinfo=pytz.utc),
                                       datetime(2012, 12, 12, 12, 13, 2, tzinfo=pytz.utc), False)
        expected_dt = datetime(2012, 12, 12, 12, 12, 12, tzinfo=pytz.utc)
        self.assertEqual(new_dt, expected_dt)

    def test_constant_time(self):
        hdf = mocked_hdf()('slow')
        # mocked hdf seconds increment
        self.assertFalse(has_constant_time(hdf))

        #TODO: Test case for calculate_fallback_dt where has_constant_time is
        # True

    def test_get_dt_arrays__hour_after_midnight(self):
        '''
        test to check hour following midnight does not get replaced if 0 and less than 1 hour in length
        '''
        duration = 1740 # 29 minutes
        year = P('Year', array=np.ma.array([2016]*duration))
        month = P('Month', array=np.ma.array([8]*duration))
        day = P('Day', array=np.ma.array([5]*duration))
        hour = P('Hour', array=np.ma.array([0]*duration))
        minute = P('Minute', array=np.ma.repeat(np.ma.arange(duration//60), 60))
        second = P('Second', array=np.ma.array(np.tile(np.arange(60), duration//60)))
        values = {'Year': year, 'Month': month, 'Day': day, 'Hour': hour, 'Minute': minute, 'Second': second}
        def hdf_get(arg):
            return values[arg]
        hdf = mock.Mock()
        hdf.duration = duration
        hdf.get.side_effect = hdf_get
        dt_arrays, precise_timestamp, dt_param_state = get_dt_arrays(hdf, datetime(2016, 8, 4, 23, 59, 59), datetime(2016, 8, 5, 1, 59, 59))
        self.assertEqual(dt_arrays, [year.array, month.array, day.array, hour.array, minute.array, second.array])
        self.assertTrue(precise_timestamp)
        self.assertTrue([p for p in dt_param_state.values() if p != PRECISE] == [])

    def test_get_dt_arrays__valid_slices(self):
        duration = 3599
        year = P('Year', array=np.ma.array([2019]*duration))
        month = P('Month', array=np.ma.array([2]*duration))
        day = P('Day', array=np.ma.array([1]*duration))
        hour = P('Hour', array=np.ma.array([0]*duration))
        minute = P('Minute', array=np.ma.repeat(np.ma.arange(duration//60), 61)[:duration])
        second = P('Second', array=np.ma.array(np.tile(np.arange(60), duration//59))[:duration])
        values = {'Year': year, 'Month': month, 'Day': day, 'Hour': hour, 'Minute': minute, 'Second': second}

        def hdf_get(arg):
            return values[arg]

        hdf = mock.Mock()
        hdf.duration = duration
        hdf.get.side_effect = hdf_get
        dt_arrays, precise_timestamp, dt_param_state = get_dt_arrays(hdf, datetime(2019, 1, 31, 13, 50, 40),
                                                                     datetime(2019, 2, 1, 1, 59, 59),
                                                                     valid_slices=[slice(150, 1000), slice(1500, 3000)])

        self.assertTrue(len(dt_arrays[0]) == len(dt_arrays[1]) == len(dt_arrays[2]) == \
                        len(dt_arrays[3]) == len(dt_arrays[4]) == len(dt_arrays[5]) == duration)

    def test_get_valid_dt_slices_all_unmasked(self):

        def hdf_get(key):
            return Parameter(key, array=arrays[key], frequency=1)

        hdf = mock.Mock()
        hdf.get = mock.Mock()
        hdf.get.side_effect = hdf_get

        arrays = {
            'Airspeed': load_array('airspeed_sample.npz'),
        }

        valid_dt_slices = get_valid_dt_slices(hdf)

        self.assertEqual(valid_dt_slices[0].start, 0)
        self.assertEqual(valid_dt_slices[0].stop, 41396)

    def test_get_valid_dt_slices_some_masked(self):

        def hdf_get(key):
            return Parameter(key, array=arrays[key], frequency=1)

        hdf = mock.Mock()
        hdf.get = mock.Mock()
        hdf.get.side_effect = hdf_get

        arrays = {
            'Airspeed': load_array('airspeed_sample_masked_under_80.npz'),
        }

        valid_dt_slices = get_valid_dt_slices(hdf)

        self.assertEqual(valid_dt_slices[0].start, 412)
        self.assertEqual(valid_dt_slices[0].stop, 1911)
        self.assertEqual(valid_dt_slices[1].start, 3204)
        self.assertEqual(valid_dt_slices[1].stop, 8551)
        self.assertEqual(valid_dt_slices[2].start, 9355)
        self.assertEqual(valid_dt_slices[2].stop, 14771)
        self.assertEqual(valid_dt_slices[3].start, 15969)
        self.assertEqual(valid_dt_slices[3].stop, 21434)
        self.assertEqual(valid_dt_slices[4].start, 22135)
        self.assertEqual(valid_dt_slices[4].stop, 27047)
        self.assertEqual(valid_dt_slices[5].start, 28413)
        self.assertEqual(valid_dt_slices[5].stop, 33427)
        self.assertEqual(valid_dt_slices[6].start, 34775)
        self.assertEqual(valid_dt_slices[6].stop, 40832)

    def test_get_valid_dt_slices_all_masked(self):

        def hdf_get(key):
            return Parameter(key, array=arrays[key], frequency=1)

        hdf = mock.Mock()
        hdf.get = mock.Mock()
        hdf.get.side_effect = hdf_get

        arrays = {
            'Airspeed': load_array('airspeed_sample_all_masked.npz'),
        }

        valid_dt_slices = get_valid_dt_slices(hdf)

        self.assertEqual(valid_dt_slices, [])

    def test_calculate_start_datetime_multiple_valid_slices(self):
        aspd = load_array('airspeed_sample_masked_under_80.npz')
        duration = len(aspd)

        airspeed = P('Airspeed', array=aspd)
        year = P('Year', array=np.ma.array([2019]*duration))
        month = P('Month', array=np.ma.array([2]*duration))
        day = P('Day', array=np.ma.array([1]*duration))
        hour = P('Hour', array=np.ma.array([0]*duration))
        minute = P('Minute', array=np.ma.repeat(np.ma.arange(duration//60), 61)[:duration])
        second = P('Second', array=np.ma.array(np.tile(np.arange(60), duration//59))[:duration])
        values = {'Year': year, 'Month': month, 'Day': day, 'Hour': hour, 'Minute': minute, 'Second': second, 'Airspeed': airspeed}

        def hdf_get(arg):
            return values[arg]

        hdf = mock.Mock()
        hdf.duration = duration
        hdf.get = mock.Mock()
        hdf.get.side_effect = hdf_get
        start_dt = datetime(2019, 2, 1, 0, 0, tzinfo=pytz.utc)
        new_dt, _precise_timestamp, _conf = _calculate_start_datetime(hdf, datetime(2019, 2, 2, 0, 1, 1, tzinfo=pytz.utc),
                                           datetime(2012, 2, 2, 1, 2, 2, tzinfo=pytz.utc))

        self.assertEqual(new_dt, start_dt)


class TestSplitSegments(unittest.TestCase):
    def test_split_segments(self):
        # TODO: Test engine param splitting.
        # Mock hdf
        airspeed_array = np.ma.concatenate([np.arange(300, dtype=float),
                                            np.arange(300, 0, -1, dtype=float)])

        airspeed_frequency = 2
        airspeed_secs = len(airspeed_array) / airspeed_frequency

        heading_array = np.ma.arange(len(airspeed_array) / 2, dtype=float) % 360
        heading_frequency = 1
        heading_array.mask = False

        eng_array = None
        eng_frequency = 1

        dfc_array = np.ma.arange(0, 300, 2)

        hdf = mock.Mock()
        hdf.get = mock.Mock()
        hdf.get.return_value = None
        hdf.__contains__ = mock.Mock()
        hdf.__contains__.return_value = False
        hdf.reliable_frame_counter = False
        hdf.duration = 50

        def hdf_getitem(self, key, **kwargs):
            if key == 'Airspeed':
                return Parameter('Airspeed', array=airspeed_array,
                                 frequency=airspeed_frequency)
            elif key == 'Frame Counter':
                return Parameter('Frame Counter', array=dfc_array,
                                 frequency=0.25)
            elif key == 'Heading':
                # TODO: Give heading specific data.
                return Parameter('Heading', array=heading_array,
                                 frequency=heading_frequency)
            elif key == 'Eng (1) N1' and eng_array is not None:
                return Parameter('Eng (1) N1', array=eng_array,
                                 frequency=eng_frequency)
            elif key == 'Segment Split':
                seg_split = M('Segment Split', array=np.ma.zeros(len(heading_array), dtype=int),
                                 frequency=heading_frequency, values_mapping={0: "-", 1: "Split"})
                seg_split.array[390//heading_frequency] = "Split"
                return seg_split
            else:
                raise KeyError
        hdf.__getitem__ = hdf_getitem

        def hdf_get_param(key, valid_only=False):
            # Pretend that we recorded Heading True only on this aircraft
            if key == 'Heading':
                raise KeyError
            elif key == 'Heading True':
                return Parameter('Heading True', array=heading_array,
                                 frequency=heading_frequency)
        hdf.get_param = hdf_get_param

        # Unmasked single flight.
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # DFC should not affect result.
        hdf.reliable_frame_counter = True
        # Mask within slow data should not affect result.
        airspeed_array[:50] = np.ma.masked
        airspeed_array[-50:] = np.ma.masked
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # Masked beginning of speedy data will affect result.
        airspeed_array[:100] = np.ma.masked
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'STOP_ONLY')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # Masked end of speedy data will affect result.
        airspeed_array = np.ma.concatenate([np.arange(300),
                                            np.arange(300, 0, -1)])
        airspeed_array[-100:] = np.ma.masked
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'START_ONLY')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # Masked beginning and end of speedy data will affect result.
        airspeed_array[:100] = np.ma.masked
        airspeed_array[-100:] = np.ma.masked
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'MID_FLIGHT')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)

        # Two flights, split will be made using DFC.
        airspeed_array = np.ma.concatenate([np.arange(0, 200, 0.5),
                                            np.arange(200, 0, -0.5),
                                            np.arange(0, 200, 0.5),
                                            np.arange(200, 0, -0.5)])
        airspeed_secs = len(airspeed_array) / airspeed_frequency
        heading_array = np.ma.concatenate((np.arange(len(airspeed_array) // 4, dtype=float) % 360,
                                           np.zeros(len(airspeed_array) // 4)))

        # DFC jumps exactly half way.
        dfc_array = np.ma.concatenate([np.arange(0, 100),
                                       np.arange(200, 300)])

        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 2)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, 398)
        # Heading diff not exceed HEADING_CHANGE_TAXI_THRESHOLD
        segment_type, segment_slice, start_padding = segment_tuples[1]
        self.assertEqual(segment_type, 'NO_MOVEMENT')
        self.assertEqual(segment_slice.start, 398)
        self.assertEqual(segment_slice.stop, airspeed_secs)

        # Split using engine params where DFC does not jump.
        eng_array = np.ma.concatenate([np.arange(0, 100, 0.5),
                                       np.arange(100, 0, -0.5),
                                       np.arange(0, 100, 0.5),
                                       np.arange(100, 0, -0.5)])
        segment_tuples = split_segments(hdf, {})
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, 398.0)
        # Heading diff not exceed HEADING_CHANGE_TAXI_THRESHOLD
        segment_type, segment_slice, start_padding = segment_tuples[1]
        self.assertEqual(segment_type, 'NO_MOVEMENT')
        self.assertEqual(segment_slice.start, 398.0)
        self.assertEqual(segment_slice.stop, airspeed_secs)

        # Split using Turning where DFC does not jump.
        dfc_array = np.ma.concatenate([np.arange(4000, 4096),
                                       np.arange(0, 105)])
        heading_array = np.ma.concatenate([np.arange(390, 0, -1),
                                           np.zeros(10),
                                           np.arange(400, 800)])
        eng_array = None
        segment_tuples = split_segments(hdf, {})
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, 395)
        segment_type, segment_slice, start_padding = segment_tuples[1]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 395)
        self.assertEqual(segment_slice.stop, airspeed_secs)

        # Split Segment Split
        hdf.__contains__.return_value = True
        segment_tuples = split_segments(hdf, {})
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, 390)
        segment_type, segment_slice, start_padding = segment_tuples[1]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 390)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        hdf.__contains__.return_value = False

        # Same split conditions, but does not split on jumping DFC because
        # reliable_frame_counter is False.
        hdf.reliable_frame_counter = False
        dfc_array = np.ma.masked_array(np.random.randint(1000, size=((len(dfc_array),))))
        segment_tuples = split_segments(hdf, {})
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, 395)
        segment_type, segment_slice, start_padding = segment_tuples[1]
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment_slice.start, 395)
        self.assertEqual(segment_slice.stop, airspeed_secs)

        # Test that no heading change returns "NO_MOVEMENT"
        heading_array = np.ones_like(airspeed_array) * 20
        # add a bit of erroneous movement
        heading_array[5] += 2
        heading_array[5] += 12
        heading_array[5] += 9
        heading_array[5] += 4
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 2)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'NO_MOVEMENT')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, 400)
        segment_type, segment_slice, start_padding = segment_tuples[1]
        self.assertEqual(segment_type, 'NO_MOVEMENT')
        self.assertEqual(segment_slice.start, 400)
        self.assertEqual(segment_slice.stop, airspeed_secs)

        # Airspeed always slow.
        eng_array = np.ma.concatenate([np.arange(0, 100, 0.5),
                                       np.arange(100, 0, -0.5)])
        eng_frequency = 2
        airspeed_array = np.ma.concatenate([np.arange(50),
                                            np.arange(50, 0, -1)])
        airspeed_secs = len(airspeed_array) / airspeed_frequency
        heading_array = np.ma.arange(0, 100, 2) % 360

        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'GROUND_ONLY')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)

        # Airspeed Fully Masked slow.
        airspeed_array = np.ma.masked_all(100)
        heading_array = np.ma.arange(len(airspeed_array) / 2) % 360
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'NO_MOVEMENT')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # Unmasked single flight less than 3 minutes.
        airspeed_array = np.ma.concatenate([np.arange(200),
                                            np.arange(200, 0, -1)])
        heading_array = np.ma.arange(len(airspeed_array) / 2) % 360
        airspeed_secs = len(airspeed_array) / airspeed_frequency
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'START_ONLY', msg="Fast should not be long enough to be a START_AND_STOP")
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # Unmasked single flight less than 3 minutes.
        airspeed_array = np.ma.concatenate([np.arange(200),
                                            np.arange(200, 0, -1)])
        airspeed_frequency = 0.5
        heading_array = np.ma.arange(len(airspeed_array) / 2) % 360
        airspeed_secs = len(airspeed_array) / airspeed_frequency
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'START_AND_STOP', msg="Fast should be long enough to be a START_AND_STOP")
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)

        # Airspeed always slow. No Eng so heading changes should be ignored. (eg Herc)
        eng_array = None
        eng_frequency = None
        airspeed_array = np.ma.concatenate([np.arange(50),
                                            np.arange(50, 0, -1)])
        airspeed_secs = len(airspeed_array) / airspeed_frequency
        heading_array = np.ma.arange(0, 100, 2) % 360

        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 1)
        segment_type, segment_slice, start_padding = segment_tuples[0]
        self.assertEqual(segment_type, 'NO_MOVEMENT')
        self.assertEqual(segment_slice.start, 0)
        self.assertEqual(segment_slice.stop, airspeed_secs)
        # TODO: Test engine parameters.

    @unittest.skipIf(
        not os.path.isfile(os.path.join(test_data_path, "1_7295949_737-3C.hdf5")), "Test file not present")
    def test_split_segments_737_3C(self):
        '''Splits on both DFC Jump and Engine parameters.'''
        hdf = hdf_file(os.path.join(test_data_path, "1_7295949_737-3C.hdf5"))
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(segment_tuples,
                         [('START_AND_STOP', slice(0, 3168.0, None)),
                          ('START_AND_STOP', slice(3168.0, 6260.0, None)),
                          ('START_AND_STOP', slice(6260.0, 9504.0, None)),
                          ('START_AND_STOP', slice(9504.0, 12680.0, None)),
                          ('START_AND_STOP', slice(12680.0, 15571.0, None)),
                          ('START_AND_STOP', slice(15571.0, 18752.0, None))])

    def test_split_segments_data_1(self):
        '''Splits on both DFC Jump and Engine parameters.'''
        hdf_path = os.path.join(test_data_path, "split_segments_1.hdf5")
        temp_path = copy_file(hdf_path)
        hdf = hdf_file(temp_path)
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(segment_tuples,
                         [('START_AND_STOP', slice(0, 9952.0, None), 0),
                          ('START_AND_STOP', slice(9952.0, 21799.0, None), 0),
                          ('START_AND_STOP', slice(21799.0, 24665.0, None), 3),
                          ('START_AND_STOP', slice(24665.0, 27898.0, None), 1),
                          ('START_AND_STOP', slice(27898.0, 31424.0, None), 2)])

    def test_split_segments_data_2(self):
        '''Splits on both DFC Jump and Engine parameters.'''
        hdf_path = os.path.join(test_data_path, "split_segments_2.hdf5")
        temp_path = copy_file(hdf_path)
        hdf = hdf_file(temp_path)

        segment_tuples = split_segments(hdf, {})
        self.assertEqual(segment_tuples,
                         [('START_AND_STOP', slice(0, 3407.0, None), 0),
                          ('START_AND_STOP', slice(3407.0, 6362.0, None), 15),
                          ('START_AND_STOP', slice(6362.0, 9912.0, None), 26),
                          ('START_AND_STOP', slice(9912.0, 13064.0, None), 56),
                          ('START_AND_STOP', slice(13064.0, 16467.0, None), 8),
                          ('START_AND_STOP', slice(16467.0, 19065.0, None), 19),
                          ('GROUND_ONLY', slice(19065.0, 19200.0, None), 57)])

    def test_split_segments_data_3(self):
        '''Splits on both Engine and Heading parameters.'''
        hdf_path = os.path.join(test_data_path, "split_segments_3.hdf5")
        temp_path = copy_file(hdf_path)
        hdf = hdf_file(temp_path)

        segment_tuples = split_segments(hdf, {})

        #for a, e in zip(segment_tuples, expected):
            #print(a,e,a==e)

        self.assertEqual(segment_tuples,
                         [('START_AND_STOP', slice(0, 3989.0, None), 0),
                          ('START_AND_STOP', slice(3989.0, 7049.0, None), 1),
                          ('START_AND_STOP', slice(7049.0, 9569.0, None), 1),
                          ('START_AND_STOP', slice(9569.0, 12889.0, None), 1),
                          ('START_AND_STOP', slice(12889.0, 15867.0, None), 1),
                          ('START_AND_STOP', slice(15867.0, 18526.0, None), 3),
                          ('START_AND_STOP', slice(18526.0, 21726.0, None), 2),
                          ('START_AND_STOP', slice(21726.0, 24209.0, None), 2),
                          ('START_AND_STOP', slice(24209.0, 26607.0, None), 1),
                          ('START_AND_STOP', slice(26607.0, 28534.0, None), 3),
                          ('START_AND_STOP', slice(28534.0, 30875.0, None), 2),
                          ('START_AND_STOP', slice(30875.0, 33488.0, None), 3),
                          ('NO_MOVEMENT', slice(33488.0, 33680.0, None), 0),])

    @unittest.skipIf(
        not os.path.isfile(os.path.join(test_data_path, "4_3377853_146-301.hdf5")), "Test file not present")
    def test_split_segments_146_300(self):
        hdf = hdf_file(os.path.join(test_data_path, "4_3377853_146-301.hdf5"))
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(segment_tuples,
                         [('START_AND_STOP', slice(0, 24801.0, None)),
                          ('START_AND_STOP', slice(24801.0, 30000.0, None)),
                          ('START_AND_STOP', slice(30000.0, 49999.0, None)),
                          ('START_AND_STOP', slice(49999.0, 69999.0, None)),
                          ('START_AND_STOP', slice(69999.0, 73552.0, None))])

    @mock.patch('analysis_engine.split_hdf_to_segments.settings')
    def test_split_segments_multiple_types(self, settings):
        '''
        Test data has multiple segments of differing segment types.
        Test data has already been validated
        '''
        # Overriding MINIMUM_FAST_DURATION.
        settings.AIRSPEED_THRESHOLD = 80
        settings.AIRSPEED_THRESHOLD_TIME = 3 * 60
        settings.HEADING_CHANGE_TAXI_THRESHOLD = 60
        settings.MINIMUM_SPLIT_DURATION = 300
        settings.MINIMUM_FAST_DURATION = 0
        settings.MINIMUM_SPLIT_PARAM_VALUE = 0.175
        settings.HEADING_RATE_SPLITTING_THRESHOLD = 0.1
        settings.MAX_TIMEBASE_AGE = 365 * 10
        settings.MIN_FAN_RUNNING = 10
        settings.VERTICAL_SPEED_FOR_CLIMB_PHASE = 800.0
        settings.VERTICAL_SPEED_FOR_DESCENT_PHASE = -500.0

        hdf_path = os.path.join(test_data_path, "split_segments_multiple_types.hdf5")
        temp_path = copy_file(hdf_path)
        hdf = hdf_file(temp_path)

        # Set the airspeed to zero for one segment
        a = hdf.get('Airspeed')
        a.array[84756:113640] = 0.0
        hdf.set_param(a)

        self.maxDiff = None
        segment_tuples = split_segments(hdf, {})
        self.assertEqual(len(segment_tuples), 15, msg="Unexpected number of segments detected")
        segment_types = tuple(x[0] for x in segment_tuples)
        self.assertEqual(segment_types,
                         ('STOP_ONLY',
                          'START_ONLY',
                          'START_AND_STOP',
                          'INVALID',
                          'START_AND_STOP',
                          'START_AND_STOP',
                          'START_AND_STOP',
                          'STOP_ONLY',
                          'START_ONLY',
                          'START_ONLY',
                          'START_AND_STOP',
                          'START_ONLY',
                          'START_AND_STOP',
                          'START_AND_STOP',
                          'START_ONLY'))


    @unittest.skipIf(
        not os.path.isfile(os.path.join(test_data_path, "rto_split_segment.hdf5")), "Test file not present")
    def test_rto_correct_side_of_split(self):
        '''
        Test to ensure that RTO's are on the correct side of splitting, i.e. at
        the beginning of a flight. This example HDF5 file appears to have two
        stationary engine activities and an RTO between the two flights.
        This creates 6 sizeable slices (potential splitting points) where the
        engine parameters normalised to 0.
        We're interested in making the segment split within the first of these
        eng_min_slices slices (Between indices 11959.5 to 12336.5).
        Ideally the segment split should be halfway between this, at 12148.0.
        '''
        hdf = hdf_file(os.path.join(test_data_path, "rto_split_segment.hdf5"))
        segment_tuples = split_segments(hdf, {})
        split_idx = 12148.0
        self.assertEqual(
            segment_tuples,
            [('START_AND_STOP', slice(0, split_idx, None), 0),
             ('START_AND_STOP', slice(split_idx, 21997.0, None), 52),
             ('GROUND_ONLY', slice(21997.0, 22784.0, None), 45),]
        )

    def test__get_normalised_split_params(self):
        hdf = mock.Mock()
        hdf.get = mock.Mock()
        hdf.get.return_value = None
        hdf.reliable_frame_counter = False

        arrays = {
            'Eng (1) N1': load_array('split_segments_eng_1_n1_slowslice.npz'),
            'Eng (2) N1': load_array('split_segments_eng_2_n1_slowslice.npz'),
            'Groundspeed': load_array('split_segments_groundspeed_slowslice.npz'),
        }

        def hdf_getitem(self, key, **kwargs):
            return Parameter(key, array=arrays[key], frequency=1)

        hdf.__getitem__ = hdf_getitem

        norm_array, freq = _get_normalised_split_params(hdf)
        # find split which without Groundspeed and averaging of the
        # normalised parameters would not work properly due to a single
        # engine taxi. Before using groundspeed and averaging, the minimum of
        # the engine only parameters would split too early, during the taxi_in
        self.assertEqual(np.ma.argmin(norm_array), 715)


class mocked_hdf(object):
    def __init__(self, path=None):
        pass

    def __call__(self, path):
        self.path = path
        if path == 'slow':
            self.airspeed = np.ma.arange(10, 20).repeat(5)
        else:
            self.airspeed = np.ma.array(load_array('airspeed_sample.npz'))
        self.duration = len(self.airspeed)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, key, default=None):
        return self.__getitem__(key)

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        if key == 'Airspeed':
            data = self.airspeed
        else:
            if self.path == 'invalid timestamps':
                if key == 'Year':
                    data = np.ma.array([0] * self.duration)
                elif key == 'Month':
                    data = np.ma.array([13] * self.duration)
                elif key == 'Day':
                    data = np.ma.array([31] * self.duration)
                else:
                    data = np.ma.arange(0, self.duration)
            else:
                if key == 'Year':
                    if self.path == 'future timestamps':
                        data = np.ma.array([datetime.now().year + 1] * self.duration)
                    elif self.path == 'old timestamps':
                        data = np.ma.array([1999] * self.duration)
                    else:
                        data = np.ma.array([2012] * self.duration)
                elif key == 'Month':
                    data = np.ma.array([12] * self.duration)
                elif key == 'Day':
                    data = np.ma.array([25] * self.duration)
                else:
                    data = np.ma.arange(0, self.duration)
        return P(key, array=data)


class TestSegmentInfo(unittest.TestCase):
    @mock.patch('analysis_engine.split_hdf_to_segments.logger')
    @mock.patch('analysis_engine.split_hdf_to_segments.sha_hash_file')
    @mock.patch('analysis_engine.split_hdf_to_segments.hdf_file', new_callable=mocked_hdf)
    def test_timestamps_in_past(self, hdf_file_patch, sha_hash_file_patch, logger_patch):
        # No longer raising exception, using epoch instead with exception logging,
        # allows segment to be created.
        # example where it goes fast
        #seg = append_segment_info('old timestamps', 'START_AND_STOP',
        #                          slice(10,1000), 4,
        #                          fallback_dt=datetime(2012,12,12,0,0,0))
        #self.assertEqual(seg.start_dt, datetime(2012,12,12,0,0,0))
        #self.assertEqual(seg.go_fast_dt, datetime(2012,12,12,0,6,52))
        #self.assertEqual(seg.stop_dt, datetime(2012,12,12,11,29,56))
        append_segment_info(
            'invalid timestamps', 'START_AND_STOP', slice(10, 1000), 4,
            fallback_dt=datetime(2009, 12, 12, 0, 0, 0, tzinfo=pytz.utc))
        self.assertTrue(logger_patch.exception.called)
        self.assertEqual(logger_patch.exception.call_args[0], ('Unable to calculate timebase, using 1970-01-01 00:00:00+0000!',))

    @mock.patch('analysis_engine.split_hdf_to_segments.sha_hash_file')
    @mock.patch('analysis_engine.split_hdf_to_segments.hdf_file', new_callable=mocked_hdf)
    def test_timestamps_in_future_use_fallback_year(self, hdf_file_patch, sha_hash_file_patch):
        # Using fallback time is no longer recommended example where it goes fast
        seg = append_segment_info('future timestamps', 'START_AND_STOP', slice(10, 1000), 4,
                                  fallback_dt=datetime(2012, 12, 12, 0, 0, 0, tzinfo=pytz.utc))
        self.assertEqual(seg.start_dt, datetime(2012, 12, 25, 0, 0, 0, tzinfo=pytz.utc))
        self.assertEqual(seg.go_fast_dt, datetime(2012, 12, 25, 0, 6, 52, tzinfo=pytz.utc))
        self.assertEqual(seg.stop_dt, datetime(2012, 12, 25, 11, 29, 56, tzinfo=pytz.utc))
        ## This was true, but now we invalidate the Year component! Raising exception is more pythonic
        #self.assertRaises(TimebaseError, append_segment_info,
                         #'future timestamps', 'START_AND_STOP', slice(10,1000),
                         #4, fallback_dt=datetime(2012, 12, 12, 0, 0, 0, tzinfo=pytz.utc))

    @mock.patch('analysis_engine.split_hdf_to_segments.sha_hash_file')
    @mock.patch('analysis_engine.split_hdf_to_segments.hdf_file', new_callable=mocked_hdf)
    def test_append_segment_info(self, hdf_file_patch, sha_hash_file_patch):
        # example where it goes fast
        # TODO: Increase slice to be realitic for duration of data
        seg = append_segment_info('fast', 'START_AND_STOP', slice(10, 1000), 4)
        self.assertEqual(seg.path, 'fast')
        self.assertEqual(seg.part, 4)
        self.assertEqual(seg.type, 'START_AND_STOP')
        self.assertEqual(seg.start_dt, datetime(2012, 12, 25, 0, 0, 0, tzinfo=pytz.utc))
        self.assertEqual(seg.go_fast_dt, datetime(2012, 12, 25, 0, 6, 52, tzinfo=pytz.utc))
        self.assertEqual(seg.stop_dt, datetime(2012, 12, 25, 11, 29, 56, tzinfo=pytz.utc))

    @mock.patch('analysis_engine.split_hdf_to_segments.sha_hash_file')
    @mock.patch('analysis_engine.split_hdf_to_segments.hdf_file', new_callable=mocked_hdf)
    def test_append_segment_info_no_gofast(self, hdf_file_patch, sha_hash_file_patch):
        sha_hash_file_patch.return_value = 'ABCDEFG'
        # example where it does not go fast
        seg = append_segment_info('slow', 'GROUND_ONLY', slice(10, 110), 1)
        self.assertEqual(seg.path, 'slow')
        self.assertEqual(seg.go_fast_dt, None)  # didn't go fast
        self.assertEqual(seg.start_dt, datetime(2012, 12, 25, 0, 0, 0, tzinfo=pytz.utc))  # still has a start
        self.assertEqual(seg.part, 1)
        self.assertEqual(seg.type, 'GROUND_ONLY')
        self.assertEqual(seg.hash, 'ABCDEFG')  # taken from the "file"
        self.assertEqual(seg.stop_dt, datetime(2012, 12, 25, 0, 0, 50, tzinfo=pytz.utc))  # +50 seconds of airspeed

    @mock.patch('analysis_engine.split_hdf_to_segments.logger')
    @mock.patch('analysis_engine.split_hdf_to_segments.sha_hash_file')
    @mock.patch('analysis_engine.split_hdf_to_segments.hdf_file', new_callable=mocked_hdf)
    def test_invalid_datetimes(self, hdf_file_patch, sha_hash_file_patch, logger_patch):
        # No longer raising exception, using epoch instead
        #seg = append_segment_info('invalid timestamps', 'START_AND_STOP', slice(10,110), 2)

        seg = append_segment_info('invalid timestamps', 'START_AND_STOP', slice(28000, 34000), 6)
        self.assertEqual(seg.start_dt, datetime(1970, 1, 1, 0, 0, tzinfo=pytz.utc))  # start of time!
        self.assertEqual(seg.go_fast_dt, datetime(1970, 1, 1, 0, 6, 52, tzinfo=pytz.utc))  # went fast
        self.assertTrue(logger_patch.exception.called)
        self.assertEqual(logger_patch.exception.call_args[0], ('Unable to calculate timebase, using 1970-01-01 00:00:00+0000!',))

    def test_calculate_start_datetime(self):
        hdf = MockHDF({
            'Year': P('Year', np.ma.array([2011])),
            'Month': P('Month', np.ma.array([11])),
            'Day': P('Day', np.ma.array([11])),
            'Hour': P('Hour', np.ma.array([11])),
            'Minute': P('Minute', np.ma.array([11])),
            'Second': P('Second', np.ma.array([11]))
        })
        dt = datetime(2012, 12, 12, 12, 12, 12, tzinfo=pytz.utc)
        # test with all params
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        self.assertEqual(res, datetime(2011, 11, 11, 11, 11, 11, tzinfo=pytz.utc))
        self.assertTrue(precise_timestamp)
        # test without Year
        del hdf['Year']
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        self.assertEqual(res, datetime(2012, 11, 11, 11, 11, 11, tzinfo=pytz.utc))
        self.assertFalse(precise_timestamp)
        # test without Month
        del hdf['Month']
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        self.assertEqual(res, datetime(2012, 12, 11, 11, 11, 11, tzinfo=pytz.utc))
        self.assertFalse(precise_timestamp)
        # test without Day
        del hdf['Day']
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        self.assertEqual(res, datetime(2012, 12, 12, 11, 11, 11, tzinfo=pytz.utc))
        self.assertFalse(precise_timestamp)
        # test without Hour
        del hdf['Hour']
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        self.assertEqual(res, datetime(2012, 12, 12, 12, 11, 11, tzinfo=pytz.utc))
        self.assertFalse(precise_timestamp)
        # test without Minute
        del hdf['Minute']
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        self.assertEqual(res, datetime(2012, 12, 12, 12, 12, 11, tzinfo=pytz.utc))
        self.assertFalse(precise_timestamp)
        # test without Second
        del hdf['Second']
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        self.assertEqual(res, datetime(2012, 12, 12, 12, 12, 12, tzinfo=pytz.utc))
        self.assertFalse(precise_timestamp)

    def test_empty_year_no_seconds(self):
        # NB: 12's are the fallback_dt, 11's are the recorded time parameters
        dt = datetime(2012, 12, 12, 12, 12, 10, tzinfo=pytz.utc)
        # Test only without second and empty year
        hdf = MockHDF({
            'Month': P('Month', np.ma.array([11, 11, 11, 11])),
            'Day': P('Day', np.ma.array([])),
            'Hour': P('Hour', np.ma.array([11, 11, 11, 11], mask=[True, False, False, False])),
            'Minute': P('Minute', np.ma.array([11, 11]), frequency=0.5),
        }, duration=4)
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        # 9th second as the first sample (10th second) was masked
        self.assertEqual(res, datetime(2012, 11, 12, 11, 11, 10, tzinfo=pytz.utc))
        self.assertFalse(precise_timestamp)

    def test_year_00_uses_fallback_year(self):
        # Ensure that a timebase error is not raised due to old date!
        #Other than the year 2000 or possibly 2100, no date values
        # can be all 0's
        dt = datetime(2012, 12, 12, 12, 12, 10, tzinfo=pytz.utc)
        # Test only without second and empty year
        hdf = MockHDF({
            'Year': P('Year', np.ma.array([0, 0, 0, 0])),
            'Month': P('Month', np.ma.array([11, 11, 11, 11])),
            'Day': P('Day', np.ma.array([11, 11, 11, 11])),
            'Hour': P('Hour', np.ma.array([11, 11, 11, 11], mask=[True, False, False, False])),
            'Minute': P('Minute', np.ma.array([11, 11]), frequency=0.5),
        }, duration=4)
        # add a masked invalid value
        hdf['Year'].array[2] = 50
        hdf['Year'].array[2] = np.ma.masked
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        self.assertEqual(res, datetime(2012, 11, 11, 11, 11, 10, tzinfo=pytz.utc))
        self.assertFalse(precise_timestamp)

    def test_no_year_with_a_very_recent_fallback(self):
        """When fallback is a year after the flight and Year is not recorded,
        the result would be in the future. Ensure that a year is taken off if
        required.

        Generally Day and Month are recorded together and Year is optional.

        Should only Time be recorded, using the fallback date part is as good
        as you get.
        """
        # ensure current datetime is very recent
        dt = datetime.utcnow().replace(tzinfo=pytz.utc)
        # Year is not recorded, and the data is for the very end of the
        # previous year. NB: This test could fail if ran on the very last day
        # of the month!
        hdf = MockHDF({
            'Month': P('Month', np.ma.array([12, 12])),  # last month of year
            'Day': P('Day', np.ma.array([31, 31])),  # last day
            'Hour': P('Hour', np.ma.array([23, 23])),  # last hour
            'Minute': P('Minute', np.ma.array([59, 59])),  # last minute
            'Second': P('Minute', np.ma.array([58, 59])),  # last two seconds
        }, duration=2)
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        # result is a year behind the fallback datetime, even though the
        # fallback Year was used.
        self.assertEqual(res.year, datetime.now().year - 1)
        self.assertFalse(precise_timestamp)
        #self.assertEqual(res, datetime(2012,6,1,11,11,1))

    def test_midnight_rollover(self):
        """
        When a flight starts just before midnight, the majority of the
        flight will be in the next day so the fallback_dt needs to adjust
        throughout the data otherwise it will try to force the next day's
        flight to appear to that of the previous.
        """
        # fallback is at start of the recording
        dt = datetime(2012, 12, 12, 23, 59, 58, tzinfo=pytz.utc)
        hdf = MockHDF({
            'Hour': P('Hour', np.ma.array([23, 23] + [0] * 18)),
            'Minute': P('Minute', np.ma.array([59, 59] + [0] * 18)),
            'Second': P('Minute', np.ma.array([58, 59] + list(range(18)))),  # last two seconds and start of next flight
        }, duration=20)
        res, precise_timestamp, _ = _calculate_start_datetime(hdf, dt, dt)
        # result is the exact start of the data for the timestamp (not a day before!)
        self.assertEqual(res, datetime(2012, 12, 12, 23, 59, 58, tzinfo=pytz.utc))
        self.assertFalse(precise_timestamp)


class TestSegmentTypeAndSlice(unittest.TestCase):

    def test_segment_type_and_slice_1(self):
        # Unmasked fast Airspeed at the beginning of the data which is difficult
        # to validate should be ignored in segment type identification.
        speed_array = load_array('segment_type_and_slice_1_speed.npz')
        heading_array = load_array('segment_type_and_slice_1_heading.npz')
        eng_arrays = load_array('segment_type_and_slice_1_eng_arrays.npz')
        aircraft_info = {'Aircraft Type': 'aeroplane'}
        thresholds = {'hash_min_samples': 64, 'speed_threshold': 80, 'min_split_duration': 100, 'min_duration': 180}
        hdf = mock.Mock()
        hdf.superframe_present = False
        segment_type, segment, array_start_secs = _segment_type_and_slice(
            speed_array, 1, heading_array, 1, 0, 11824, eng_arrays,
            aircraft_info, thresholds, hdf, None)
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment, slice(0, 11824))
        self.assertEqual(array_start_secs, 0)

    def test_segment_type_and_slice_2(self):
        # Gear on Ground is used instead of Eng (1) Nr
        speed_array = load_array('segment_type_and_slice_2_speed.npz')
        heading_array = load_array('segment_type_and_slice_2_heading.npz')
        eng_arrays = load_array('segment_type_and_slice_2_eng_arrays.npz')
        aircraft_info = {'Aircraft Type': 'helicopter'}
        thresholds = {'hash_min_samples': 64, 'speed_threshold': 90, 'min_split_duration': 100, 'min_duration': 180}
        def get(key):
            if key == 'Gear On Ground':
                return Parameter('Gear on Ground', array=load_array('segment_type_and_slice_2_gog.npz'))
        hdf = mock.MagicMock()
        hdf.get.side_effect = get
        hdf.superframe_present = False
        segment_type, segment, array_start_secs = _segment_type_and_slice(
            speed_array, 1, heading_array, 1, 0, 5736, eng_arrays,
            aircraft_info, thresholds, hdf, None)
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment, slice(0, 5736))
        self.assertEqual(array_start_secs, 0)

    def test_segment_type_and_slice_3(self):
        # Gear on Ground is used instead of Eng (1) Nr
        speed_array = load_array('segment_type_and_slice_3_speed.npz')
        heading_array = load_array('segment_type_and_slice_3_heading.npz')
        eng_arrays = load_array('segment_type_and_slice_3_eng_arrays.npz')
        aircraft_info = {'Aircraft Type': 'helicopter'}
        thresholds = {'hash_min_samples': 64, 'speed_threshold': 90, 'min_split_duration': 100, 'min_duration': 180}
        def get(key):
            if key == 'Collective':
                return Parameter(key, array=load_array('segment_type_and_slice_3_collective.npz'), frequency=8.)
        hdf = mock.MagicMock()
        hdf.get.side_effect = get
        hdf.superframe_present = False
        segment_type, segment, array_start_secs = _segment_type_and_slice(
            speed_array, 1, heading_array, 2, 0, 5560, eng_arrays,
            aircraft_info, thresholds, hdf, None)
        self.assertEqual(segment_type, 'START_AND_STOP')
        self.assertEqual(segment, slice(0, 5560))
        self.assertEqual(array_start_secs, 0)


