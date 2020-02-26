import numpy as np
import os
import unittest
import yaml

from datetime import datetime, timedelta
from mock import Mock, call, patch

from analysis_engine import __version__, settings
from analysis_engine.library import align
from analysis_engine.node import (
    A, App, KPV, KTI, P, S,
    load,
    KeyPointValue,
    KeyTimeInstance,
    Section,
)

from analysis_engine.flight_attribute import (
    DeterminePilot,
    DestinationAirport,
    Duration,
    FlightID,
    FlightNumber,
    FlightType,
    InvalidFlightType,
    LandingAirport,
    LandingDatetime,
    LandingFuel,
    LandingGrossWeight,
    LandingPilot,
    LandingRunway,
    OffBlocksDatetime,
    OnBlocksDatetime,
    TakeoffAirport,
    TakeoffDatetime,
    TakeoffFuel,
    TakeoffGrossWeight,
    TakeoffPilot,
    TakeoffRunway,
    Version,
)

from flightdatautilities import api

from analysis_engine.test_utils import buildsection

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

airports = yaml.load(open(os.path.join(test_data_path, 'airports.yaml'), 'rb'), Loader=yaml.FullLoader)


def setUpModule():
    settings.API_HANDLER = 'analysis_engine.api_handler.FileHandler'


class NodeTest(object):
    def test_can_operate(self):
        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations()),
                self.operational_combination_length,
            )
        else:
            self.assertEqual(
                self.node_class.get_operational_combinations(),
                self.operational_combinations,
            )


class TestDeterminePilot(unittest.TestCase):

    def test__autopilot_engaged(self):
        determine_pilot = DeterminePilot()
        # No autopilots engaged:
        pilot = determine_pilot._autopilot_engaged(0, 0)
        self.assertEqual(pilot, None)
        # Autopilot 1 engaged:
        pilot = determine_pilot._autopilot_engaged(1, 0)
        self.assertEqual(pilot, 'Captain')
        # Autopilot 2 engaged:
        pilot = determine_pilot._autopilot_engaged(0, 1)
        self.assertEqual(pilot, 'First Officer')
        # Autopilots 1 & 2 engaged:
        pilot = determine_pilot._autopilot_engaged(1, 1)
        self.assertEqual(pilot, None)

    def test__controls_changed(self):
        determine_pilot = DeterminePilot()
        slice_ = slice(0, 3)
        below_tolerance = np.ma.array([
            settings.CONTROLS_IN_USE_TOLERANCE / 4.0, 0, 0,
            settings.CONTROLS_IN_USE_TOLERANCE / 2.0, 0, 0,
        ])
        above_tolerance = np.ma.array([
            settings.CONTROLS_IN_USE_TOLERANCE * 4.0, 0, 0,
            settings.CONTROLS_IN_USE_TOLERANCE * 2.0, 0, 0,
        ])
        # Both pitch and roll below tolerance:
        pitch, roll = below_tolerance, below_tolerance
        change = determine_pilot._controls_changed(slice_, pitch, roll)
        self.assertFalse(change)
        # Pitch above tolerance:
        pitch, roll = above_tolerance, below_tolerance
        change = determine_pilot._controls_changed(slice_, pitch, roll)
        self.assertTrue(change)
        # Roll above tolerance:
        pitch, roll = below_tolerance, above_tolerance
        change = determine_pilot._controls_changed(slice_, pitch, roll)
        self.assertTrue(change)
        # Both pitch and roll above tolerance:
        pitch, roll = above_tolerance, above_tolerance
        change = determine_pilot._controls_changed(slice_, pitch, roll)
        self.assertTrue(change)
        # Both pitch and roll above tolerance outside of slice:
        slice_ = slice(1, 3)
        pitch, roll = above_tolerance, above_tolerance
        change = determine_pilot._controls_changed(slice_, pitch, roll)
        self.assertFalse(change)

    def test__controls_in_use(self):
        pitch_capt = Mock()
        pitch_fo = Mock()
        roll_capt = Mock()
        roll_fo = Mock()

        section = Section('Takeoff', slice(0, 3), 0, 3)

        # Note: We instantiate one of the subclasses of DeterminePilot as we
        #       use logging methods not defined in this abstract superclass.
        determine_pilot = LandingPilot()
        determine_pilot._controls_changed = Mock()

        # Neither pilot's controls changed:
        determine_pilot._controls_changed.reset_mock()
        determine_pilot._controls_changed.side_effect = [False, False]
        pilot = determine_pilot._controls_in_use(pitch_capt, pitch_fo, roll_capt, roll_fo, section)
        determine_pilot._controls_changed.assert_has_calls([
            call(section.slice, pitch_capt, roll_capt),
            call(section.slice, pitch_fo, roll_fo),
        ])
        self.assertEqual(pilot, None)
         # Only captain's controls changed:
        determine_pilot._controls_changed.reset_mock()
        determine_pilot._controls_changed.side_effect = [True, False]
        pilot = determine_pilot._controls_in_use(pitch_capt, pitch_fo, roll_capt, roll_fo, section)
        determine_pilot._controls_changed.assert_has_calls([
            call(section.slice, pitch_capt, roll_capt),
            call(section.slice, pitch_fo, roll_fo),
        ])
        self.assertEqual(pilot, 'Captain')
        # Only first Officer's controls changed:
        determine_pilot._controls_changed.reset_mock()
        determine_pilot._controls_changed.side_effect = [False, True]
        pilot = determine_pilot._controls_in_use(pitch_capt, pitch_fo, roll_capt, roll_fo, section)
        determine_pilot._controls_changed.assert_has_calls([
            call(section.slice, pitch_capt, roll_capt),
            call(section.slice, pitch_fo, roll_fo),
        ])
        self.assertEqual(pilot, 'First Officer')
        # Both pilot's controls changed:
        determine_pilot._controls_changed.reset_mock()
        determine_pilot._controls_changed.side_effect = [True, True]
        pilot = determine_pilot._controls_in_use(pitch_capt, pitch_fo, roll_capt, roll_fo, section)
        determine_pilot._controls_changed.assert_has_calls([
            call(section.slice, pitch_capt, roll_capt),
            call(section.slice, pitch_fo, roll_fo),
        ])
        self.assertEqual(pilot, None)

    def test__determine_pilot(self):
        determine_pilot = DeterminePilot()

        pitch_capt = Mock()
        pitch_fo = Mock()
        roll_capt = Mock()
        roll_fo = Mock()
        cc_capt = Mock()
        cc_fo = Mock()
        pitch_capt.array = Mock()
        pitch_fo.array = Mock()
        roll_capt.array = Mock()
        roll_fo.array = Mock()
        cc_capt.array = Mock()
        cc_fo.array = Mock()
        ap1 = Mock()
        ap2 = Mock()
        phase = Mock()

        determine_pilot._autopilot_engaged = Mock()
        determine_pilot._controls_in_use = Mock()
        determine_pilot._control_column_in_use = Mock()
        determine_pilot.set_flight_attr = Mock()

        def reset_all_mocks():
            determine_pilot._autopilot_engaged.reset_mock()
            determine_pilot._controls_in_use.reset_mock()
            determine_pilot._control_column_in_use.reset_mock()
            determine_pilot.set_flight_attr.reset_mock()

        # Controls in use, no phase.
        reset_all_mocks()
        pilot = determine_pilot._determine_pilot(
            None, pitch_capt, pitch_fo, roll_capt, roll_fo, cc_capt, cc_fo,
            None, None, None)
        self.assertFalse(determine_pilot._autopilot_engaged.called)
        self.assertFalse(determine_pilot._controls_in_use.called)
        self.assertEqual(pilot, None)
        # Controls in use with phase. Pilot cannot be discerned.
        reset_all_mocks()
        determine_pilot._controls_in_use.return_value = None
        determine_pilot._control_column_in_use.return_value = None
        pilot = determine_pilot._determine_pilot(
            None, pitch_capt, pitch_fo, roll_capt, roll_fo, cc_capt, cc_fo,
            phase, None, None)
        self.assertFalse(determine_pilot._autopilot_engaged.called)
        determine_pilot._controls_in_use.assert_called_once_with(
            pitch_capt.array, pitch_fo.array, roll_capt.array, roll_fo.array,
            phase)
        determine_pilot._control_column_in_use.assert_called_once_with(
            cc_capt.array, cc_fo.array, phase)
        self.assertEqual(pilot, determine_pilot._controls_in_use.return_value)
        # Controls in use with phase. Pilot returned
        reset_all_mocks()
        determine_pilot._controls_in_use.return_value = 'Captain'
        pilot = determine_pilot._determine_pilot(
            None, pitch_capt, pitch_fo, roll_capt, roll_fo, cc_capt, cc_fo, phase,
            None, None)
        self.assertFalse(determine_pilot._autopilot_engaged.called)
        determine_pilot._controls_in_use.assert_called_once_with(pitch_capt.array, pitch_fo.array, roll_capt.array, roll_fo.array, phase)
        self.assertEqual(pilot, determine_pilot._controls_in_use.return_value)
        # Only Autopilot.
        reset_all_mocks()
        determine_pilot._autopilot_engaged.return_value = 'Captain'
        pilot = determine_pilot._determine_pilot(None, None, None, None, None,
                                                 None, None, None, ap1, ap2)
        determine_pilot._autopilot_engaged.assert_called_once_with(ap1, ap2)
        self.assertFalse(determine_pilot._controls_in_use.called)
        self.assertEqual(pilot, determine_pilot._autopilot_engaged.return_value)
        # Controls in Use overrides Autopilot.
        reset_all_mocks()
        determine_pilot._controls_in_use.return_value = 'Captain'
        determine_pilot._autopilot_engaged.return_value = 'First Officer'
        pilot = determine_pilot._determine_pilot(
            None, pitch_capt, pitch_fo, roll_capt, roll_fo, cc_capt, cc_fo,
            phase, ap1, ap2)
        self.assertFalse(determine_pilot._autopilot_engaged.called)
        determine_pilot._controls_in_use.assert_called_once_with(
            pitch_capt.array, pitch_fo.array, roll_capt.array, roll_fo.array, phase)
        self.assertEqual(pilot, determine_pilot._controls_in_use.return_value)
        # Autopilot is used when Controls in Use does not provide an answer.
        reset_all_mocks()
        determine_pilot._autopilot_engaged.return_value = 'First Officer'
        determine_pilot._controls_in_use.return_value = None
        determine_pilot._control_column_in_use.return_value = None
        pilot = determine_pilot._determine_pilot(
            None, pitch_capt, pitch_fo, roll_capt, roll_fo, cc_capt, cc_fo, phase,
            ap1, ap2)
        determine_pilot._autopilot_engaged.assert_called_once_with(ap1, ap2)
        determine_pilot._controls_in_use.assert_called_once_with(
            pitch_capt.array, pitch_fo.array, roll_capt.array, roll_fo.array,
            phase)
        self.assertEqual(pilot,
                         determine_pilot._autopilot_engaged.return_value)


    def get_params(self, hdf_path, _slice, phase_name):
        import shutil
        import tempfile
        from hdfaccess.file import hdf_file

        with tempfile.NamedTemporaryFile() as temp_file:
            shutil.copy(hdf_path, temp_file.name)

            with hdf_file(hdf_path) as hdf:
                pitch_capt = hdf.get('Pitch (Capt)')
                pitch_fo = hdf.get('Pitch (FO)')
                roll_capt = hdf.get('Roll (Capt)')
                roll_fo = hdf.get('Roll (FO)')
                cc_capt = hdf.get('Control Column Force (Capt)')
                cc_fo = hdf.get('Control Column Force (FO)')

                for par in pitch_capt, pitch_fo, roll_capt, roll_fo, cc_capt, \
                        cc_fo:
                    if par is not None:
                        ref_par = par
                        break

        phase = S(name=phase_name, frequency=1)
        phase.create_section(_slice)
        phase = phase.get_aligned(ref_par)[0]

        # Align the arrays, usually done in the Nodes
        for par in pitch_fo, roll_capt, roll_fo, cc_capt, cc_fo:
            if par is None:
                continue
            par.array = align(par, ref_par)
            par.hz = ref_par.hz

        return pitch_capt, pitch_fo, roll_capt, roll_fo, cc_capt, cc_fo, phase

    def test_determine_pilot_from_hdf_control_column(self):
        '''
        Use Control Column Force (*) to determine the flying pilot
        '''
        import logging

        determine_pilot = DeterminePilot()
        # warning method is normally initialised with Node superclass in one of
        # the DeterminePilot's ancestors
        determine_pilot.warning = logging.warning
        determine_pilot.info = logging.info

        def test_from_file(hdf_path, _slice, phase_name, expected_pilot):
            (pitch_capt, pitch_fo, roll_capt, roll_fo, cc_capt, cc_fo,
             phase) = self.get_params(hdf_path, _slice, phase_name)

            pilot = determine_pilot._determine_pilot(
                None, pitch_capt, pitch_fo, roll_capt, roll_fo, None, None,
                phase, None, None)

            # pitch and roll are not enough to determine the pilot
            self.assertIsNone(pilot)

            pilot = determine_pilot._determine_pilot(
                None, pitch_capt, pitch_fo, roll_capt, roll_fo, cc_capt,
                cc_fo, phase, None, None)

            # control column force should allow to determine the pilot
            self.assertEqual(pilot, expected_pilot)

        items = [
            [
                os.path.join(test_data_path, 'identify_pilot_01.hdf5'),
                'Captain',
                'Captain',
                slice(686, 751),
                slice(40773, 40872)
            ],
            [
                os.path.join(test_data_path, 'identify_pilot_02.hdf5'),
                'First Officer',
                'First Officer',
                slice(1015, 1162),
                slice(41451, 41511)
            ],
        ]

        for fn, to_pilot, ld_pilot, to, ld in items:
            test_from_file(fn, to, 'Takeoff', to_pilot)
            test_from_file(fn, ld, 'Landing', ld_pilot)


class TestDestinationAirport(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(DestinationAirport.get_operational_combinations(),
                         [('Destination',),
                          ('AFR Destination Airport',),
                          ('Destination', 'AFR Destination Airport')])

    def setUp(self):
        dest_array = np.ma.array(
            [b'FDSL', b'FDSL', b'FDSL', b'FDSL', b'ABCD', b'ABCD'],
            mask=[True, False, False, False, False, True])
        self.dest = P('Destination', array=dest_array)
        self.afr_dest = A('AFR Destination Airport', value={'id': 2000})
        self.node = DestinationAirport()

    @patch('analysis_engine.api_handler.FileHandler.get_airport')
    def test_derive_dest(self, get_airport):
        self.node.derive(self.dest, None)
        self.assertEqual(self.node.value, get_airport.return_value)
        get_airport.assert_called_once_with('FDSL')

    def test_derive_afr_dest(self):
        self.node.derive(None, self.afr_dest)
        self.assertEqual(self.node.value, self.afr_dest.value)

    @patch('analysis_engine.api_handler.FileHandler.get_airport')
    def test_derive_both(self, get_airport):
        self.node.derive(self.dest, self.afr_dest)
        self.assertEqual(self.node.value, get_airport.return_value)
        get_airport.assert_called_once_with('FDSL')

    def test_derive_invalid(self):
        dest_array = np.ma.array(
            ['000', '0000', '0000', '0000', '00', '0000'],
            mask=[True, False, False, False, False, True])
        dest = P('Destination', array=dest_array)
        self.node.derive(dest, None)
        self.assertEqual(self.node.value, None)


class TestDuration(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Duration.get_operational_combinations(),
                         [('FDR Takeoff Datetime', 'FDR Landing Datetime')])

    def test_derive(self):
        duration = Duration()
        duration.set_flight_attr = Mock()
        takeoff_dt = A('FDR Takeoff Datetime',
                       value=datetime(1970, 1, 1, 0, 1, 0))
        landing_dt = A('FDR Landing Datetime',
                       value=datetime(1970, 1, 1, 0, 2, 30))
        duration.derive(takeoff_dt, landing_dt)
        duration.set_flight_attr.assert_called_once_with(90)


class TestFlightID(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(FlightID.get_operational_combinations(),
                         [('AFR Flight ID',)])

    def test_derive(self):
        afr_flight_id = A('AFR Flight ID', value=10245)
        flight_id = FlightID()
        flight_id.set_flight_attr = Mock()
        flight_id.derive(afr_flight_id)
        flight_id.set_flight_attr.assert_called_once_with(10245)


class TestFlightNumber(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(FlightNumber.get_operational_combinations(),
                         [('Flight Number', 'Mobile')])

    def test_derive_basic(self):
        flight_number_param = P('Flight Number',
                                array=np.ma.masked_array([103, 102,102]))
        mobiles = buildsection('Mobile', 0, 3)
        flight_number = FlightNumber()
        flight_number.set_flight_attr = Mock()
        flight_number.derive(flight_number_param, mobiles)
        flight_number.set_flight_attr.assert_called_with('102')

    def test_derive(self):
        flight_number = load(os.path.join(test_data_path,
                                          'FDRFlightNumber_FlightNumber.nod'))
        mobiles = buildsection('Mobile', 10, 130)
        node = FlightNumber()
        node.derive(flight_number, mobiles)
        self.assertEqual(node.value, '805')

    def test_derive_ascii(self):
        flight_number_param = P('Flight Number',
                                array=np.ma.masked_array(['ABC', 'DEF', 'DEF'],
                                                         dtype=np.string_))
        mobiles = buildsection('Mobile', 0, 3)
        flight_number = FlightNumber()
        flight_number.set_flight_attr = Mock()
        flight_number.derive(flight_number_param, mobiles)
        flight_number.set_flight_attr.assert_called_with('DEF')
        flight_number.set_flight_attr.reset_mock()
        # Entirely masked.
        flight_number_param.array[:] = np.ma.masked
        flight_number.derive(flight_number_param, mobiles)
        flight_number.set_flight_attr.called = False

    def test_derive_masked(self):
        '''

        '''
        flight_number = FlightNumber()
        number_param = P(
            'Flight Number',
            array=np.ma.array([0, 0, 0, 0, 36, 36, 0, 0, 0, 0],
                              mask=[True] * 4 + [False] * 2 + [True] * 4))
        mobiles = buildsection('Mobile', 0, 10)
        flight_number.derive(number_param, mobiles)
        self.assertEqual(flight_number.value, '36')

    def test_derive_most_common_positive_float(self):
        flight_number = FlightNumber()

        neg_number_param = P(
            'Flight Number',
            array=np.ma.array([-1,2,-4,10,20,40,11]))
        mobiles = buildsection('Mobile', 0, 7)
        flight_number.derive(neg_number_param, mobiles)
        self.assertEqual(flight_number.value, None)

        # TODO: Implement variance checks as below
        ##high_variance_number_param = P(
            ##'Flight Number',
            ##array=np.ma.array([2,2,4,4,4,7,7,7,4,5,4,7,910]))
        ##self.assertRaises(ValueError, flight_number.derive, high_variance_number_param)

        flight_number_param= P(
            'Flight Number',
            array=np.ma.array([2,555.6,444,444,444,444,444,444,888,444,444,
                               444,444,444,444,444,444,7777,9100]))
        mobiles = buildsection('Mobile', 0, 19)
        flight_number.set_flight_attr = Mock()
        flight_number.derive(flight_number_param, mobiles)
        flight_number.set_flight_attr.assert_called_with('444')

    def test_derive_restrict_to_mobile(self):
        flight_number_param = P('Flight Number',
                                array=np.ma.repeat([103, 102], [30, 20]))
        mobiles = buildsection('Mobile', 20, 50)
        flight_number = FlightNumber()
        flight_number.derive(flight_number_param, mobiles)
        self.assertEqual(flight_number.value, '102')


class TestLandingAirport(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = LandingAirport
        self.operational_combinations = [
            ('Approach Information',),
            ('AFR Landing Airport',),
            ('Approach Information', 'AFR Landing Airport'),
            ('Approach Information', 'Precise Positioning'),
            ('AFR Landing Airport', 'Precise Positioning'),
            ('Approach Information', 'AFR Landing Airport', 'Precise Positioning'),
        ]

    def test_derive_afr_fallback(self):
        afr_apt = A(name='AFR Landing Airport', value={'id': 25})
        apt = self.node_class()
        apt.set_flight_attr = Mock()
        approach = App()
        approach.create_approach('LANDING', slice(None))
        apt.derive(approach, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()


class TestLandingDatetime(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LandingDatetime.get_operational_combinations(),
                         [('Start Datetime', 'Touchdown')])

    def test_derive(self):
        landing_datetime = LandingDatetime()
        landing_datetime.set_flight_attr = Mock()
        start_datetime = A('Start Datetime', datetime(1970, 1, 1))
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(12, 'a'),
                                            KeyTimeInstance(30, 'b')])
        touchdown.frequency = 0.5
        landing_datetime.derive(start_datetime, touchdown)
        expected_datetime = datetime(1970, 1, 1, 0, 1)
        landing_datetime.set_flight_attr.assert_called_once_with(
            expected_datetime)
        touchdown = KTI('Touchdown')
        landing_datetime.set_flight_attr = Mock()
        landing_datetime.derive(start_datetime, touchdown)
        landing_datetime.set_flight_attr.assert_called_once_with(None)


class TestLandingFuel(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LandingFuel.get_operational_combinations(),
                         [('AFR Landing Fuel',), ('Fuel Qty At Touchdown',),
                          ('AFR Landing Fuel', 'Fuel Qty At Touchdown')])

    def test_derive(self):
        landing_fuel = LandingFuel()
        landing_fuel.set_flight_attr = Mock()
        # Only 'AFR Takeoff Fuel' dependency.
        afr_landing_fuel = A('AFR Landing Fuel', value=100)
        landing_fuel.derive(afr_landing_fuel, None)
        landing_fuel.set_flight_attr.assert_called_once_with(100)
        # Only 'Fuel Qty At Liftoff' dependency.
        fuel_qty_at_touchdown = KPV('Fuel Qty At Touchdown',
                                    items=[KeyPointValue(87, 160),
                                           KeyPointValue(132, 200)])
        landing_fuel.set_flight_attr = Mock()
        landing_fuel.derive(None, fuel_qty_at_touchdown)
        landing_fuel.set_flight_attr.assert_called_once_with(200)
        # Both, 'AFR Takeoff Fuel' used.
        landing_fuel.set_flight_attr = Mock()
        landing_fuel.derive(afr_landing_fuel, fuel_qty_at_touchdown)
        landing_fuel.set_flight_attr.assert_called_once_with(100)


class TestLandingGrossWeight(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LandingGrossWeight.get_operational_combinations(),
                         [('Gross Weight At Touchdown',)])

    def test_derive(self):
        landing_gross_weight = LandingGrossWeight()
        landing_gross_weight.set_flight_attr = Mock()
        touchdown_gross_weight = KPV('Gross Weight At Touchdown',
                                     items=[KeyPointValue(5, 15, 'a'),
                                            KeyPointValue(12, 120, 'b')])
        landing_gross_weight.derive(touchdown_gross_weight)
        landing_gross_weight.set_flight_attr.assert_called_once_with(120)


class TestLandingPilot(unittest.TestCase):

    def test_can_operate(self):
        opts = LandingPilot.get_operational_combinations()
        combinations = [
            # Only AFR
            ('AFR Landing Pilot',),
            # Only Pilot Flying
            ('Pilot Flying', 'Landing'),
            # Only Controls:
            ('Pitch (Capt)', 'Pitch (FO)', 'Roll (Capt)', 'Roll (FO)', 'Landing'),
            # Only Control Column Forces:
            ('Control Column Force (Capt)', 'Control Column Force (FO)', 'Landing'),
            # Only Autopilot:
            ('AP (1) Engaged', 'AP (2) Engaged', 'Touchdown'),
            # Everything:
            ('Pilot Flying',
             'Pitch (Capt)', 'Pitch (FO)', 'Roll (Capt)', 'Roll (FO)',
             'Control Column Force (Capt)', 'Control Column Force (FO)',
             'AP (1) Engaged', 'AP (2) Engaged', 'Landing', 'Touchdown'),
        ]
        for combination in combinations:
            self.assertIn(combination, opts, msg=combination)

    @patch('analysis_engine.library.value_at_index')
    def test_derive(self, value_at_index):
        ap1 = Mock()
        ap2 = Mock()
        phase = Mock()

        pitch_capt = Mock()
        pitch_fo = Mock()
        roll_capt = Mock()
        roll_fo = Mock()

        ap1_eng = Mock()
        ap2_eng = Mock()
        value_at_index.side_effect = [ap1, ap2]

        landings = Mock()
        landings.get_last = Mock()
        landings.get_last.return_value = phase

        touchdowns = Mock()
        touchdowns.get_last = Mock()
        touchdowns.get_last.return_value = Mock()

        pilot = LandingPilot()
        pilot._determine_pilot = Mock()
        pilot._determine_pilot.return_value = Mock()

        pilot.derive(
            pitch_capt, pitch_fo, roll_capt, roll_fo, ap1_eng, ap2_eng, None, None, landings, touchdowns, None)

        self.assertEqual(pilot.value, pilot._determine_pilot.return_value)

        pilot.derive(pitch_capt, pitch_fo, roll_capt, roll_fo, ap1_eng, ap2_eng, None, None, None, landings,
                     touchdowns, A('AFR Landing Pilot', 'FIRST_OFFICER'))
        self.assertEqual(pilot.value, 'First Officer')


class TestLandingRunway(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = LandingRunway
        self.operational_combinations = [
            ('Approach Information',),
            ('AFR Landing Runway',),
            ('Approach Information', 'AFR Landing Runway'),
            ('Approach Information', 'Precise Positioning'),
            ('AFR Landing Runway', 'Precise Positioning'),
            ('Approach Information', 'AFR Landing Runway', 'Precise Positioning'),
        ]

    def test_derive_afr_fallback(self):
        afr_rwy = A(name='AFR Landing Runway', value={'identifier': '27R'})
        approach = App()
        approach.create_approach('LANDING', slice(None))
        rwy = self.node_class()
        rwy.set_flight_attr = Mock()
        # Check that the AFR airport was used and the API wasn't called:
        rwy.derive(None, None)
        rwy.set_flight_attr.assert_called_once_with(None)
        rwy.set_flight_attr.reset_mock()
        rwy.derive(approach, afr_rwy)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)


class TestOffBlocksDatetime(unittest.TestCase):
    def test_derive(self):
        # Empty.
        off_blocks = KTI('Off Blocks', items=[])
        start_datetime = A(name='Start Datetime', value=datetime.now())
        off_blocks_datetime = OffBlocksDatetime()
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(off_blocks, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(None)

        # One instance.
        off_blocks = KTI('Off Blocks', items=[KeyTimeInstance(name='Off Blocks',
                                                    index=300)])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(off_blocks, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(
            start_datetime.value + timedelta(seconds=300))

        # More than one instance.
        off_blocks = KTI('Off Blocks', items=[
            KeyTimeInstance(name='Off Blocks', index=400),
            KeyTimeInstance(name='Off Blocks', index=600),
        ])
        off_blocks_datetime.set_flight_attr = Mock()
        off_blocks_datetime.derive(off_blocks, start_datetime)
        off_blocks_datetime.set_flight_attr.assert_called_once_with(
            start_datetime.value + timedelta(seconds=400))


class TestOnBlocksDatetime(unittest.TestCase):
    def test_derive(self):
        # Empty.
        on_blocks = KTI('On Blocks', items=[])
        start_datetime = A(name='Start Datetime', value=datetime.now())
        on_blocks_datetime = OnBlocksDatetime()
        on_blocks_datetime.set_flight_attr = Mock()
        on_blocks_datetime.derive(on_blocks, start_datetime)
        on_blocks_datetime.set_flight_attr.assert_called_once_with(None)

        # One instance.
        on_blocks = KTI('On Blocks', items=[KeyTimeInstance(name='On Blocks',
                                                    index=300)])
        on_blocks_datetime.set_flight_attr = Mock()
        on_blocks_datetime.derive(on_blocks, start_datetime)
        on_blocks_datetime.set_flight_attr.assert_called_once_with(
            start_datetime.value + timedelta(seconds=300))

        # More than one instance.
        on_blocks = KTI('On Blocks', items=[
            KeyTimeInstance(name='On Blocks', index=400),
            KeyTimeInstance(name='On Blocks', index=600),
        ])
        on_blocks_datetime.set_flight_attr = Mock()
        on_blocks_datetime.derive(on_blocks, start_datetime)
        on_blocks_datetime.set_flight_attr.assert_called_once_with(
            start_datetime.value + timedelta(seconds=600))


class TestTakeoffAirport(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TakeoffAirport
        self.check_operational_combination_length_only = True
        self.operational_combination_length = 23

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Latitude At Liftoff', 'Longitude At Liftoff')))
        self.assertTrue(self.node_class.can_operate(('AFR Takeoff Airport',)))
        self.assertTrue(self.node_class.can_operate(('Latitude Off Blocks', 'Longitude Off Blocks')))
        self.assertFalse(self.node_class.can_operate(('Latitude At Liftoff', 'Longitude Off Blocks')))

    @patch('analysis_engine.api_handler.FileHandler.get_nearest_airport')
    def test_derive_airport_not_found(self, get_nearest_airport):
        '''
        Attribute is not set when airport is not found.
        '''
        get_nearest_airport.side_effect = api.NotFoundError('Not Found.')
        lat = KPV(name='Latitude At Liftoff', items=[
            KeyPointValue(index=12, value=4.0),
            KeyPointValue(index=32, value=6.0),
        ])
        lon = KPV(name='Longitude At Liftoff', items=[
            KeyPointValue(index=12, value=3.0),
            KeyPointValue(index=32, value=9.0),
        ])
        afr_apt = A(name='AFR Takeoff Airport', value={'id': 25})
        precise = A(name='Precise Positioning')
        precise.value = True
        apt = self.node_class()
        apt.set_flight_attr = Mock()
        # Check that no attribute is created if not found via API:
        apt.derive(lat, lon, None)
        apt.set_flight_attr.assert_called_once_with(None)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        get_nearest_airport.reset_mock()
        # Check that the AFR airport was used if not found via API:
        apt.derive(lat, lon, afr_apt, precise)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        get_nearest_airport.reset_mock()

    @patch('analysis_engine.api_handler.FileHandler.get_nearest_airport')
    def test_derive_airport_found(self, get_nearest_airport):
        '''
        Attribute is set when airport is found.
        '''
        info = {'id': 123, 'distance': 2}
        get_nearest_airport.return_value = [info]
        lat = KPV(name='Latitude At Liftoff', items=[
            KeyPointValue(index=12, value=4.0),
            KeyPointValue(index=32, value=6.0),
        ])
        lon = KPV(name='Longitude At Liftoff', items=[
            KeyPointValue(index=12, value=3.0),
            KeyPointValue(index=32, value=9.0),
        ])
        afr_apt = A(name='AFR Takeoff Airport', value={'id': 25, 'distance': 2})
        precise = A(name='Precise Positioning')
        precise.value = True
        apt = self.node_class()
        apt.set_flight_attr = Mock()
        # Check that the airport returned via API is used for the attribute:
        apt.derive(lat, lon, afr_apt, precise)
        apt.set_flight_attr.assert_called_once_with(info)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        get_nearest_airport.reset_mock()
        # Check that the airport returned via API is used for the attribute:
        apt.derive(lat, lon)
        apt.set_flight_attr.assert_called_once_with(info)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        get_nearest_airport.reset_mock()
        # Check that the airport returned via API is used for the attribute:
        apt.derive(None, None, None, None, lat, lon)
        apt.set_flight_attr.assert_called_once_with(info)
        apt.set_flight_attr.reset_mock()
        get_nearest_airport.assert_called_once_with(4.0, 3.0)
        get_nearest_airport.reset_mock()

    @patch('analysis_engine.api_handler.FileHandler.get_nearest_airport')
    def test_derive_afr_fallback(self, get_nearest_airport):
        info = {'id': '50'}
        get_nearest_airport.return_value = info
        lat = KPV(name='Latitude At Liftoff', items=[
            KeyPointValue(index=12, value=4.0),
            KeyPointValue(index=32, value=6.0),
        ])
        lon = KPV(name='Longitude At Liftoff', items=[
            KeyPointValue(index=12, value=3.0),
            KeyPointValue(index=32, value=9.0),
        ])
        afr_apt = A(name='AFR Takeoff Airport', value={'id': 25})
        apt = self.node_class()
        apt.set_flight_attr = Mock()
        # Check that the AFR airport was used and the API wasn't called:
        apt.derive(None, None, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        assert not get_nearest_airport.called, 'method should not have been called'
        apt.derive(lat, None, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        assert not get_nearest_airport.called, 'method should not have been called'
        apt.derive(None, lon, afr_apt)
        apt.set_flight_attr.assert_called_once_with(afr_apt.value)
        apt.set_flight_attr.reset_mock()
        assert not get_nearest_airport.called, 'method should not have been called'


class TestTakeoffDatetime(unittest.TestCase):

    def setUp(self):
        self.node_class = TakeoffDatetime
        self.takeoff_datetime = self.node_class()
        self.takeoff_datetime.set_flight_attr = Mock()
        self.start_dt = A('Start Datetime', value=datetime(1970, 1, 1))
        self.liftoff = KTI(name='Liftoff', frequency=2, items=[
            KeyTimeInstance(index=64),
        ])
        self.rto = buildsection('Rejected Takeoff', 15, 20)
        self.off_blocks = KTI(name='Off Blocks', frequency=0.25, items=[
            KeyTimeInstance(index=3),
        ])

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Start Datetime',)))
        self.assertFalse(self.node_class.can_operate(('Liftoff', 'Off Blocks')))

    def test_derive(self):
        self.takeoff_datetime.derive(self.liftoff, self.rto, self.off_blocks, self.start_dt)
        self.takeoff_datetime.set_flight_attr.assert_called_once_with(datetime(1970, 1, 1, 0, 0, 32))

    def test_derive__rto(self):
        self.takeoff_datetime.derive(None, self.rto, self.off_blocks, self.start_dt)
        self.takeoff_datetime.set_flight_attr.assert_called_once_with(datetime(1970, 1, 1, 0, 0, 15))

    def test_derive__ground_run(self):
        self.takeoff_datetime.derive(None, None, self.off_blocks, self.start_dt)
        self.takeoff_datetime.set_flight_attr.assert_called_once_with(datetime(1970, 1, 1, 0, 0, 12))

    def test_derive__incomplete(self):
        self.takeoff_datetime.derive(None, None, None, self.start_dt)
        self.takeoff_datetime.set_flight_attr.assert_called_once_with(self.start_dt.value)


class TestTakeoffFuel(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffFuel.get_operational_combinations(),
                         [('AFR Takeoff Fuel',), ('Fuel Qty At Liftoff',),
                          ('AFR Takeoff Fuel', 'Fuel Qty At Liftoff')])

    def test_derive(self):
        takeoff_fuel = TakeoffFuel()
        takeoff_fuel.set_flight_attr = Mock()
        # Only 'AFR Takeoff Fuel' dependency.
        afr_takeoff_fuel = A('AFR Takeoff Fuel', value=100)
        takeoff_fuel.derive(afr_takeoff_fuel, None)
        takeoff_fuel.set_flight_attr.assert_called_once_with(100)
        # Only 'Fuel Qty At Liftoff' dependency.
        fuel_qty_at_liftoff = KPV('Fuel Qty At Liftoff',
                                  items=[KeyPointValue(132, 200)])
        takeoff_fuel.set_flight_attr = Mock()
        takeoff_fuel.derive(None, fuel_qty_at_liftoff)
        takeoff_fuel.set_flight_attr.assert_called_once_with(200)
        # Both, 'AFR Takeoff Fuel' used.
        takeoff_fuel.set_flight_attr = Mock()
        takeoff_fuel.derive(afr_takeoff_fuel, fuel_qty_at_liftoff)
        takeoff_fuel.set_flight_attr.assert_called_once_with(100)


class TestTakeoffGrossWeight(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffGrossWeight.get_operational_combinations(),
                         [('Gross Weight At Liftoff',)])

    def test_derive(self):
        takeoff_gross_weight = TakeoffGrossWeight()
        takeoff_gross_weight.set_flight_attr = Mock()
        liftoff_gross_weight = KPV('Gross Weight At Liftoff',
                                   items=[KeyPointValue(5, 135, 'a'),
                                          KeyPointValue(12, 120, 'b')])
        takeoff_gross_weight.derive(liftoff_gross_weight)
        takeoff_gross_weight.set_flight_attr.assert_called_once_with(135)


class TestTakeoffPilot(unittest.TestCase):

    def test_can_operate(self):
        opts = TakeoffPilot.get_operational_combinations()
        combinations = [
            # Only AFR
            ('AFR Takeoff Pilot',),
            # Only Pilot Flying
            ('Pilot Flying', 'Takeoff'),
            # Only Controls:
            ('Pitch (Capt)', 'Pitch (FO)', 'Roll (Capt)', 'Roll (FO)', 'Takeoff'),
            # Only Control Column Forces:
            ('Control Column Force (Capt)', 'Control Column Force (FO)', 'Takeoff'),
            # Only Autopilot:
            ('AP (1) Engaged', 'AP (2) Engaged', 'Liftoff'),
            # Everything:
            ('Pilot Flying',
             'Pitch (Capt)', 'Pitch (FO)', 'Roll (Capt)', 'Roll (FO)',
             'Control Column Force (Capt)', 'Control Column Force (FO)',
             'AP (1) Engaged', 'AP (2) Engaged', 'Takeoff', 'Liftoff'),
        ]
        for combination in combinations:
            self.assertIn(combination, opts, msg=combination)

    @patch('analysis_engine.library.value_at_index')
    def test_derive(self, value_at_index):
        ap1 = Mock()
        ap2 = Mock()
        ap3 = Mock()
        phase = Mock()

        pitch_capt = Mock()
        pitch_fo = Mock()
        roll_capt = Mock()
        roll_fo = Mock()

        ap1_eng = Mock()
        ap2_eng = Mock()
        ap3_eng = Mock()
        value_at_index.side_effect = [ap1, ap2, ap3]

        takeoffs = Mock()
        takeoffs.get_first = Mock()
        takeoffs.get_first.return_value = phase

        liftoffs = Mock()
        liftoffs.get_first = Mock()
        liftoffs.get_first.return_value = Mock()

        pilot = TakeoffPilot()
        pilot._determine_pilot = Mock()
        pilot._determine_pilot.return_value = Mock()

        pilot.derive(pitch_capt, pitch_fo, roll_capt, roll_fo, ap1_eng, ap2_eng, ap3_eng, None, None, takeoffs,
                     liftoffs, None, None)

        self.assertEqual(pilot.value, pilot._determine_pilot.return_value)

        pilot.derive(pitch_capt, pitch_fo, roll_capt, roll_fo, ap1_eng, ap2_eng, ap3_eng, None, None, takeoffs,
                     liftoffs, None, A('AFR Takeoff Pilot', 'FIRST_OFFICER'))
        self.assertEqual(pilot.value, 'First Officer')


class TestTakeoffRunway(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = TakeoffRunway
        self.operational_combination_length = 160
        self.check_operational_combination_length_only = True

    @patch('analysis_engine.flight_attribute.nearest_runway')
    def test_derive(self, nearest_runway):
        info = {'identifier': '27L', 'length': 20}
        nearest_runway.return_value = info
        fdr_apt = A(name='FDR Takeoff Airport', value={'id': 25, 'runways':[info]})
        afr_apt = None
        lat = KPV(name='Latitude At Liftoff', items=[
            KeyPointValue(index=1, value=4.0),
            KeyPointValue(index=2, value=6.0),
        ])
        lon = KPV(name='Longitude At Liftoff', items=[
            KeyPointValue(index=1, value=3.0),
            KeyPointValue(index=2, value=9.0),
        ])
        hdg = KPV(name='Heading During Takeoff', items=[
            KeyPointValue(index=1, value=20.0),
            KeyPointValue(index=2, value=60.0),
        ])
        precise = A(name='Precise Positioning')
        rwy = self.node_class()
        rwy.set_flight_attr = Mock()
        # Test with bare minimum information:
        rwy.derive(fdr_apt, afr_apt, hdg)
        rwy.set_flight_attr.assert_called_once_with(info)
        rwy.set_flight_attr.reset_mock()
        nearest_runway.assert_called_once_with(fdr_apt.value, 20.0, hint='takeoff')
        nearest_runway.reset_mock()
        # Test for aircraft where positioning is not precise:
        precise.value = True
        rwy.derive(fdr_apt, afr_apt, hdg, lat, lon, None, None, precise)
        rwy.set_flight_attr.assert_called_once_with(info)
        rwy.set_flight_attr.reset_mock()
        nearest_runway.assert_called_once_with(fdr_apt.value, 20.0, latitude=np.array([4.0]), longitude=np.array([3.0]))
        nearest_runway.reset_mock()
        # Test for aircraft where positioning is not precise:
        # NOTE: Latitude and longitude are still used for determining the
        #       takeoff runway, even when positioning is not precise!
        precise.value = False
        rwy.derive(fdr_apt, afr_apt, hdg, lat, lon, None, None, precise)
        rwy.set_flight_attr.assert_called_once_with(info)
        rwy.set_flight_attr.reset_mock()
        nearest_runway.assert_called_once_with(fdr_apt.value, 20.0, latitude=np.array([4.0]), longitude=np.array([3.0]),  hint='takeoff')
        nearest_runway.reset_mock()

    @patch('analysis_engine.flight_attribute.nearest_runway')
    def test_derive_afr_fallback(self, nearest_runway):
        info = {'identifier': '09L'}
        def runway_side_effect(apt, hdg, *args, **kwargs):
            if hdg == 90.0:
                return info
        nearest_runway.side_effect = runway_side_effect
        fdr_apt = A(name='FDR Takeoff Airport', value={'id': 50, 'runways':[info]})
        afr_rwy = A(name='AFR Takeoff Runway', value={'identifier': '27R'})
        hdg_a = KPV(name='Heading During Takeoff', items=[
            KeyPointValue(index=1, value=270.0),
            KeyPointValue(index=2, value=360.0),
        ])
        hdg_b = KPV(name='Heading During Takeoff', items=[
            KeyPointValue(index=1, value=90.0),
            KeyPointValue(index=2, value=180.0),
        ])
        precise = A(name='Precise Positioning')
        rwy = self.node_class()
        rwy.set_flight_attr = Mock()
        # Check that the AFR airport was used and the API wasn't called:
        rwy.derive(None, None, None)
        rwy.set_flight_attr.assert_called_once_with(None)
        rwy.set_flight_attr.reset_mock()
        assert not nearest_runway.called, 'method should not have been called'
        rwy.derive(fdr_apt, afr_rwy, None)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        rwy.set_flight_attr.reset_mock()
        assert not nearest_runway.called, 'method should not have been called'
        rwy.derive(None, afr_rwy, hdg_a)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        rwy.set_flight_attr.reset_mock()
        assert not nearest_runway.called, 'method should not have been called'
        rwy.derive(None, afr_rwy, None)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        rwy.set_flight_attr.reset_mock()
        assert not nearest_runway.called, 'method should not have been called'
        # Check wrong heading triggers AFR:
        precise.value = True
        rwy.derive(fdr_apt, afr_rwy, hdg_a, None, None, None, None, precise)
        rwy.set_flight_attr.assert_called_once_with(afr_rwy.value)
        nearest_runway.assert_called_once_with(fdr_apt.value, hdg_a.get_first().value)
        rwy.set_flight_attr.reset_mock()
        nearest_runway.reset_mock()
        rwy.derive(fdr_apt, afr_rwy, hdg_b, None, None, None, None, precise)
        rwy.set_flight_attr.assert_called_once_with(info)
        nearest_runway.assert_called_once_with(fdr_apt.value, hdg_b.get_first().value)

    def test_derive__bilbao(self):
        fdr_apt = A(name='FDR Takeoff Airport', value=airports['airports']['004'])
        toff_hdg = KPV(name='Heading During Takeoff', items=[
            KeyPointValue(index=710.4453125, value=299.53125),
        ])
        toff_lat = KPV(name='Latitude At Liftoff', items=[
            KeyPointValue(index=724.50390625, value=43.3009835333),
        ])
        toff_lon = KPV(name='Longitude At Liftoff', items=[
            KeyPointValue(index=724.50390625, value=-2.91060447693),
        ])
        accel_start_lat = KPV(name='Latitude Takeoff Acceleration Start', items=[
            KeyPointValue(index=695.76852498, value=43.2960891724),
        ])
        accel_start_lon = KPV(name='Longitude Takeoff Acceleration Start', items=[
            KeyPointValue(index=695.76852498, value=-2.89747238159),
        ])
        precise = A(name='Precise Positioning', value=True)

        node = self.node_class()
        node.derive(fdr_apt, None, toff_hdg, toff_lat, toff_lon, accel_start_lat,
                   accel_start_lon, precise)

        self.assertEqual(node.value['identifier'], '30')


class TestFlightType(unittest.TestCase):
    def test_can_operate(self):
        poss_combs = FlightType.get_operational_combinations()
        self.assertEqual(len(poss_combs), 2**9-1)

    def test_derive(self):
        '''
        Tests every flow, but does not test every conceivable set of arguments.
        '''
        type_node = FlightType()
        type_node.set_flight_attr = Mock()
        # Liftoff and Touchdown.
        fast = S('Fast', items=[slice(5,10)])
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(5, 'a')])
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(10, 'x')])
        type_node.derive(None, fast, None, liftoffs, touchdowns, None, None, None)
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.COMPLETE)
        # Would be 'COMPLETE', but 'AFR Type' overrides it.
        afr_type = A('AFR Type', value=FlightType.Type.FERRY)
        type_node.set_flight_attr = Mock()
        type_node.derive(afr_type, fast, None, liftoffs, touchdowns, None, None, None)
        type_node.set_flight_attr.assert_called_once_with(FlightType.Type.FERRY)
        # Liftoff missing.
        empty_liftoffs = KTI('Liftoff')
        type_node.set_flight_attr = Mock()
        try:
            type_node.derive(None, fast, None, empty_liftoffs, touchdowns, None, None, None)
        except InvalidFlightType as err:
            self.assertEqual(err.flight_type, 'TOUCHDOWN_ONLY')
        # Touchdown missing.
        empty_touchdowns = KTI('Touchdown')
        type_node.set_flight_attr = Mock()
        try:
            type_node.derive(None, fast, None, liftoffs, empty_touchdowns, None, None, None)
        except InvalidFlightType as err:
            self.assertEqual(err.flight_type, 'LIFTOFF_ONLY')

        # Liftoff and Touchdown missing, only Fast.
        type_node.set_flight_attr = Mock()
        type_node.derive(None, fast, None, empty_liftoffs, empty_touchdowns, None,
                         None, None)
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.INCOMPLETE)
        # Liftoff, Touchdown and Fast missing.
        empty_fast = fast = S('Fast')
        type_node.set_flight_attr = Mock()
        type_node.derive(None, empty_fast, None, empty_liftoffs, empty_touchdowns,
                         None, None, None)
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.INCOMPLETE)
        # Liftoff, Touchdown and Fast missing, RTO.
        rto = buildsection('Rejected Takeoff', 5, 10)
        type_node.set_flight_attr = Mock()
        type_node.derive(None, empty_fast, None, empty_liftoffs, empty_touchdowns,
                         None, rto, None)
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.REJECTED_TAKEOFF)
        # Liftoff after Touchdown.
        late_liftoffs = KTI('Liftoff', items=[KeyTimeInstance(20, 'a')])
        type_node.set_flight_attr = Mock()
        try:
            type_node.derive(None, fast, None, late_liftoffs, touchdowns, None, None, None)
        except InvalidFlightType as err:
            self.assertEqual(err.flight_type, 'TOUCHDOWN_BEFORE_LIFTOFF')
        # Touch and Go before Touchdown.
        afr_type = A('AFR Type', value=FlightType.Type.TRAINING)
        touch_and_gos = KTI('Touch And Go', items=[KeyTimeInstance(7, 'a')])
        type_node.set_flight_attr = Mock()
        type_node.derive(afr_type, fast, None, liftoffs, touchdowns, touch_and_gos,
                         None, None)
        type_node.set_flight_attr.assert_called_once_with(
            FlightType.Type.TRAINING)
        # Touch and Go after Touchdown.
        afr_type = A('AFR Type', value=FlightType.Type.TRAINING)
        touch_and_gos = KTI('Touch And Go', items=[KeyTimeInstance(15, 'a')])
        type_node.set_flight_attr = Mock()
        try:
            type_node.derive(afr_type, fast, None, liftoffs, touchdowns,
                             touch_and_gos, None, None)
        except InvalidFlightType as err:
            self.assertEqual(err.flight_type, 'LIFTOFF_ONLY')


class TestAnalysisDatetime(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestVersion(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Version.get_operational_combinations(),
                         [('Start Datetime',)])

    def test_derive(self):
        version = Version()
        version.set_flight_attr = Mock()
        version.derive()
        version.set_flight_attr.assert_called_once_with(__version__)


##############################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
