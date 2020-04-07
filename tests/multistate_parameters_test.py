import datetime
import numpy as np
import os
import unittest
import inspect

from mock import patch
from numpy.ma.testutils import assert_array_equal

from hdfaccess.parameter import MappedArray
from flightdatautilities import aircrafttables as at, units as ut
from flightdatautilities import masked_array_testutils as ma_test
from analysis_engine.test_utils import buildsection, buildsections

from analysis_engine.node import (
    A, M, P, S, App, Attribute, load, Section
)

from analysis_engine.library import (
    unique_values,
    runs_of_ones
)

from analysis_engine.multistate_parameters import (
    AOAAbnormalOperation,
    APEngaged,
    APChannelsEngaged,
    APLateralMode,
    APVerticalMode,
    APUOn,
    APURunning,
    CargoSmokeOrFire,
    Configuration,
    Daylight,
    DualInput,
    ThrustModeSelected,
    EngBleedOpen,
    EngRunning,
    Eng1Running,
    Eng2Running,
    Eng3Running,
    Eng4Running,
    Eng_AllRunning,
    Eng_AnyRunning,
    Eng_1_Fire,
    Eng_2_Fire,
    Eng_3_Fire,
    Eng_4_Fire,
    Eng_Fire,
    Eng_Oil_Press_Warning,
    EventMarker,
    Flap,
    FlapExcludingTransition,
    FlapIncludingTransition,
    FlapLever,
    FlapLeverSynthetic,
    Flaperon,
    FuelQty_Low,
    GearDown,
    GearDownInTransit,
    GearDownSelected,
    GearInTransit,
    GearOnGround,
    GearUp,
    GearUpInTransit,
    GearUpSelected,
    Gear_RedWarning,
    KeyVHFCapt,
    KeyVHFFO,
    MasterCaution,
    MasterWarning,
    PackValvesOpen,
    PilotFlying,
    PitchAlternateLaw,
    PitchDisconnect,
    Slat,
    SlatExcludingTransition,
    SlatFullyExtended,
    SlatIncludingTransition,
    SlatInTransit,
    SlatPartExtended,
    SlatRetracted,
    SmokeWarning,
    SpeedControl,
    SpeedbrakeDeployed,
    SpeedbrakeSelected,
    StableApproach,
    StableApproachExcludingEngThrust,
    StallWarning,
    StickPusher,
    StickShaker,
    TakeoffConfigurationWarning,
    TAWSAlert,
    TAWSDontSink,
    TAWSGlideslopeCancel,
    TAWSTerrainCaution,
    TAWSTerrainWarning,
    TAWSTooLowGear,
    TCASFailure,
    TCASRA,
    ThrustReversers,
    Transmitting
)


##############################################################################
# Test Configuration


def setUpModule():
    at.configure(package='flightdatautilities.aircrafttables')


##############################################################################


test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

class NodeTest(object):
    def test_can_operate(self):
        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations()),
                self.operational_combination_length,
            )
        else:
            combinations = list(map(set, self.node_class.get_operational_combinations()))
            for combination in map(set, self.operational_combinations):
                self.assertIn(combination, combinations)

    def get_params_from_hdf(self, hdf_path, param_names, _slice=None,
                            phase_name='Phase'):
        import shutil
        import tempfile
        from hdfaccess.file import hdf_file

        params = []
        phase = None

        with tempfile.NamedTemporaryFile() as temp_file:
            shutil.copy(hdf_path, temp_file.name)

            with hdf_file(hdf_path) as hdf:
                for param_name in param_names:
                    p = hdf.get(param_name)
                    if param_name == 'Pilot Flying':
                        p.array.values_mapping = {0: '-', 1: 'Captain', 2: 'First Officer'}
                    params.append(p)

        if _slice:
            phase = S(name=phase_name, frequency=1)
            phase.create_section(_slice)
            phase = phase.get_aligned(params[0])

        return params, phase


class TestAOAAbnormalOperation(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AOAAbnormalOperation

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(['AOA (R) Failure']))
        self.assertTrue(self.node_class.can_operate(['AOA (R) Failure', 'AOA Secondary Heater']))
        self.assertTrue(self.node_class.can_operate([
            'AOA (L) Failure',
            'AOA (L) Signal Failure',
            'AOA (L) Primary Heater',
            'AOA (R) Failure',
            'AOA (R) Signal Failure',
            'AOA (R) Primary Heater',
            'AOA Signal Failure',
            'AOA Secondary Heater',
            'AOA Correction Program',
        ]))

    def test_derive_single(self):
        val_mapping = {
            0: '-',
            1: 'Failed',
        }
        l_failure = M('AOA (L) Failure', array=np.ma.array(([0]*7 + [1] + [0]*2)), values_mapping=val_mapping)
        node = self.node_class()
        node.derive(l_failure, None, None, None, None, None, None, None, None)
        self.assertTrue(all(node.array == ['-', '-', '-', '-', '-', '-', '-', 'AOA (L) Failure', '-', '-']))

    def test_derive_all(self):
        failed_values_mapping = {0: '-', 1: 'Failed'}
        on_values_mapping = {0: 'On', 1: '-'}
        aoa_l = M(
            'AOA (L) Failure',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=failed_values_mapping,
        )
        aoa_l_signal = M(
            'AOA (L) Signal Failure',
            array=np.ma.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=failed_values_mapping,
        )
        aoa_l_heater = M(
            'AOA (L) Primary Heater',
            array=np.ma.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=on_values_mapping,
        )
        aoa_r = M(
            'AOA (R) Failure',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=failed_values_mapping,
        )
        aoa_r_signal = M(
            'AOA (R) Signal Failure',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=failed_values_mapping,
        )
        aoa_r_heater = M(
            'AOA (R) Primary Heater',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=on_values_mapping,
        )
        aoa_signal_fail = M(
            'AOA Signal Failure',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=failed_values_mapping,
        )
        aoa_secondary_heater = M(
            'AOA Secondary Heater',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=on_values_mapping,
        )
        aoa_correction = M(
            'AOA Correction Program',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping={0: '-', 1:'Yes'},
        )

        node=self.node_class()
        node.derive(aoa_l, aoa_l_signal, aoa_l_heater, aoa_r, aoa_r_signal,
                    aoa_r_heater, aoa_signal_fail, aoa_secondary_heater, aoa_correction)
        self.assertTrue(all(node.array == ['-', '-', 'AOA (L) Signal Failure', 'AOA (L) Signal Failure',
                                           'AOA (L) Primary Heater', 'AOA (L) Primary Heater', 'AOA (R) Failure',
                                           'AOA (R) Failure', 'AOA (R) Signal Failure', 'AOA (R) Signal Failure',
                                           'AOA (L) Failure', 'AOA (L) Failure', 'AOA (R) Primary Heater',
                                           'AOA (R) Primary Heater', 'AOA Signal Failure', 'AOA Signal Failure',
                                           'AOA Secondary Heater', 'AOA Secondary Heater', '-', '-',
                                           'AOA Correction Program', 'AOA Correction Program', '-', '-', '-', '-', '-', '-',
                                           '-', '-', '-', '-', '-', '-']))


class TestAPLateralMode(unittest.TestCase):

    def test_can_operate(self):
        # Avoid exploding long list of combinations.
        self.assertTrue(APLateralMode.can_operate(['Lateral Mode Selected']))
        self.assertTrue(APLateralMode.can_operate(['Runway Mode Active']))
        self.assertTrue(APLateralMode.can_operate([
            'Lateral Mode Selected',
            'Runway Mode Active',
            'NAV Mode Active',
            'ILS Localizer Capture Active',
            'ILS Localizer Track Active',
            'Roll Go Around Mode Active',
            'Land Track Active',
            'Heading Mode Active']))

    def test_derive_lateral_mode_selected(self):
        lateral_mode_selected_values_mapping = {
            0: '-',
            1: 'Runway Mode Active',
            2: 'NAV Mode Active',
            3: 'ILS Localizer Capture Active',
            4: 'Unused Mode',
        }
        lateral_mode_selected_array = np.ma.array(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        lateral_mode_selected = M(
            'Lateral Mode Selected',
            array=lateral_mode_selected_array,
            values_mapping=lateral_mode_selected_values_mapping)
        node = APLateralMode()
        node.derive(
            lateral_mode_selected, None, None, None, None, None, None, None)
        self.assertTrue(
            all(node.array ==
                ['-', '-', 'RWY', 'RWY', 'NAV', 'NAV', 'LOC CAPT', 'LOC CAPT', '-', '-']))

    def test_derive_all(self):
        activated_values_mapping = {0: '-', 1: 'Activated'}
        runway_mode_active = M(
            'Runway Mode Active',
            array=np.ma.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        nav_mode_active = M(
            'NAV Mode Active',
            array=np.ma.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        ils_localizer_capture_active = M(
            'ILS Localizer Capture Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        ils_localizer_track_active = M(
            'ILS Localizer Track Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        roll_go_around_mode_active = M(
            'Roll Go Around Mode Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        land_track_active = M(
            'Land Track Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        heading_mode_active = M(
            'Heading Mode Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        node = APLateralMode()
        node.derive(
            None, runway_mode_active, nav_mode_active,
            ils_localizer_capture_active, ils_localizer_track_active,
            roll_go_around_mode_active, land_track_active, heading_mode_active)
        self.assertTrue(
            all(node.array == ['-', '-',
                               'RWY', 'RWY',
                               'NAV', 'NAV',
                               'LOC CAPT', 'LOC CAPT',
                               'LOC', 'LOC',
                               'ROLL OUT', 'ROLL OUT',
                               'LAND', 'LAND',
                               'HDG', 'HDG',
                               '-', '-']))


class TestAPVerticalMode(unittest.TestCase):

    def setUp(self):
        self._longitudinal_mode_selected_values_mapping = {
            0: '-',
            1: 'Altitude',
            2: 'Final Descent Mode',
            3: 'Flare Mode',
            4: 'Land Track Active',
            5: 'Vertical Speed Engaged',
            6: 'Unused Mode',
        }
        self._pitch_mode_values_mapping = {
            0: '-',
            1: 'FLARE',
            3: 'FINAL DES',
            8: 'FPA',
            9: 'V/S',
            10: 'GO AROUND',
            11: 'TAKE OFF',
            14: 'DES',
            15: 'CLB',
            16: 'ALT',
            32: 'ALT*',
            64: 'G/S',
            128: 'G/S*'
        }

    def test_can_operate(self):
        # Avoid exploding long list of combinations.
        self.assertTrue(APVerticalMode.can_operate(['Climb Mode Active']))
        self.assertTrue(APVerticalMode.can_operate(['Longitudinal Mode Selected']))
        self.assertTrue(APVerticalMode.can_operate([
            'Climb Active',
            'Longitudinal Mode Selected',
            'ILS Glideslope Capture Active',
            'ILS Glideslope Active',
            'Flare Mode',
            'AT Active',
            'Open Climb Mode',
            'Open Descent Mode',
            'Altitude Capture Mode',
            'Altitude Mode',
            'Expedite Climb Mode',
            'Expedite Descent Mode']))

    def test_derive_longitudinal_mode_selected(self):
        longitudinal_mode_selected_array = np.ma.array(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        longitudinal_mode_selected = M(
            'Longitudinal Mode Selected',
            array=longitudinal_mode_selected_array,
            values_mapping=self._longitudinal_mode_selected_values_mapping,
        )
        node = APVerticalMode()
        node.derive(
            None, None, longitudinal_mode_selected, None, None, None, None, None,
            None, None, None, None, None)
        self.assertTrue(
            all(node.array ==
                ['-', '-',
                 'ALT CSTR', 'ALT CSTR',
                 'FINAL', 'FINAL',
                 'FLARE', 'FLARE',
                 'LAND', 'LAND',
                 'V/S', 'V/S',
                 '-', '-']))

    def test_derive_all(self):
        activated_values_mapping = {0: '-', 1: 'Activated'}
        at_active = M(
            'AT Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        climb_active = M(
            'Climb Mode Active',
            array=np.ma.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        ils_glideslope_capture_active = M(
            'ILS Glideslope Capture Active',
            array=np.ma.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        ils_glideslope_active = M(
            'ILS Glideslope Active',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        flare_mode = M(
            'Flare Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping={0: '-', 1: 'Engaged'},
        )
        open_climb_mode = M(
            'Open Climb Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        open_descent_mode = M(
            'Open Descent Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        altitude_capture_mode = M(
            'Altitude Capture Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        altitude_mode = M(
            'Altitude Capture Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        expedite_climb_mode = M(
            'Expedite Climb Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        expedite_descent_mode = M(
            'Expedite Descent Mode',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            values_mapping=activated_values_mapping,
        )
        longitudinal_mode_selected = M(
            'Longitudinal Mode Selected',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 4, 4, 5, 5, 0, 0]),
            values_mapping=self._longitudinal_mode_selected_values_mapping,
        )
        pitch_mode = M(
            'Pitch Mode',
            array=np.ma.array([15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 4, 4, 5, 5, 32, 32]),
            values_mapping=self._pitch_mode_values_mapping,
        )
        node = APVerticalMode()
        node.derive(at_active, climb_active, longitudinal_mode_selected,
                    ils_glideslope_capture_active, ils_glideslope_active,
                    flare_mode, open_climb_mode, open_descent_mode,
                    altitude_capture_mode, altitude_mode,
                    expedite_climb_mode, expedite_descent_mode, None, pitch_mode)
        self.assertTrue(
            all(node.array ==
                ['CLB', 'CLB',
                 'CLB', 'CLB',
                 'GS CAPT', 'GS CAPT',
                 'GS', 'GS',
                 'FLARE', 'FLARE',
                 'DES', 'DES',
                 'OP CLB', 'OP CLB',
                 'OP DES', 'OP DES',
                 'ALT CAPT', 'ALT CAPT',
                 'ALT', 'ALT',
                 'EXPED CLB', 'EXPED CLB',
                 'EXPED DES', 'EXPED DES',
                 'FLARE', 'FLARE',
                 'FINAL', 'FINAL',
                 'LAND', 'LAND',
                 'V/S', 'V/S',
                 'ALT CAPT', 'ALT CAPT']))


class TestAPEngaged(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = APEngaged
        self.operational_combinations = [
            ('AP (1) Engaged',),
            ('AP (2) Engaged',),
            ('AP (3) Engaged',),
            ('AP (1) Engaged', 'AP (2) Engaged'),
            ('AP (1) Engaged', 'AP (3) Engaged'),
            ('AP (2) Engaged', 'AP (3) Engaged'),
            ('AP (1) Engaged', 'AP (2) Engaged', 'AP (3) Engaged'),
        ]
    def test_single_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')
        eng = APEngaged()
        eng.derive(ap1, None, None)
        expected = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={0: '-', 1: 'Engaged'},
                   name='AP Engaged',
                   frequency=1,
                   offset=0.1)
        assert_array_equal(expected.array, eng.array)

    def test_dual_ap(self):
        # Two result in just "Engaged" state still
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')
        ap2 = M(array=np.ma.array(data=[0,0,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged')
        ap3 = None
        eng = APEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0,0,1,1,1,0]),
                   values_mapping={0: '-', 1: 'Engaged'},
                   name='AP Engaged',
                   frequency=1,
                   offset=0.1)

        assert_array_equal(expected.array, eng.array)

    def test_triple_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged',
                   frequency=1,
                   offset=0.1)
        ap2 = M(array=np.ma.array(data=[0,1,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged',
                   frequency=1,
                   offset=0.2)
        ap3 = M(array=np.ma.array(data=[0,0,1,1,1,1]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (3) Engaged',
                   frequency=1,
                   offset=0.4)
        eng = APEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0,1,1,1,1,1]),
                   values_mapping={0: '-', 1: 'Engaged'},
                   name='AP Engaged',
                   frequency=1,
                   offset=0.25)

        assert_array_equal(expected.array, eng.array)


class TestAPUOn(unittest.TestCase):
    def test_can_operate(self):
        opts = APUOn.get_operational_combinations()
        self.assertTrue(('APU (1) On',) in opts)
        self.assertTrue(('APU (2) On',) in opts)
        self.assertTrue(('APU (1) On', 'APU (2) On') in opts)

    def test_derive(self):
        values_mapping = {0: '-', 1: 'On'}
        apu_1 = M(name='APU (1) On',
                  array=np.ma.array(data=[0, 1, 1, 1, 0, 0, 0]),
                  values_mapping=values_mapping)
        apu_2 = M(name='APU (2) On',
                  array=np.ma.array(data=[0, 0, 1, 1, 1, 0, 0]),
                  values_mapping=values_mapping)
        node = APUOn()
        node.derive(apu_1, None)
        expected = ['-'] + ['On'] * 3 + ['-'] * 3
        np.testing.assert_array_equal(node.array, expected)
        node = APUOn()
        node.derive(None, apu_2)
        expected = ['-'] * 2 + ['On'] * 3 + ['-'] * 2
        np.testing.assert_array_equal(node.array, expected)
        node = APUOn()
        node.derive(apu_1, apu_2)
        expected = ['-'] + ['On'] * 4 + ['-'] * 2
        np.testing.assert_array_equal(node.array, expected)


class TestAPURunning(unittest.TestCase):
    def test_can_operate(self):
        opts = APURunning.get_operational_combinations()
        self.assertTrue(('APU N1',) in opts)
        self.assertTrue(('APU Generator AC Voltage',) in opts)
        self.assertTrue(('APU Bleed Valve Open',) in opts)
        self.assertTrue(('APU Fuel Flow',) in opts)
        self.assertTrue(('APU On',) in opts)

    def test_derive_apu_n1(self):
        apu_n1 = P('APU N1', array=np.ma.array([0, 40, 80, 100, 70, 30, 0.0]))
        run = APURunning()
        run.derive(apu_n1, None, None, None, None)
        expected = ['-'] * 2 + ['Running'] * 3 + ['-'] * 2
        np.testing.assert_array_equal(run.array, expected)

    def test_derive_apu_voltage(self):
        apu_voltage = P('APU Generator AC Voltage',
                        array=np.ma.array([0, 115, 115, 115, 114, 0, 0]))
        run = APURunning()
        run.derive(None, apu_voltage, None, None, None)
        expected = ['-'] + ['Running'] * 4 + ['-'] * 2
        np.testing.assert_array_equal(run.array, expected)

    def test_derive_apu_bleed_valve_open(self):
        apu_bleed_valve_open = M('APU Bleed Valve Open',
                                 array=np.ma.array([0,1,1,0,1],
                                                   mask=[False] * 4 + [True]),
                                 values_mapping={0: '-', 1: 'Open'})
        run = APURunning()
        run.derive(None, None, apu_bleed_valve_open, None, None)
        expected = ['-'] + ['Running'] * 2 + ['-'] * 2
        np.testing.assert_array_equal(run.array, expected)

    def test_derive_apu_fuel_flow(self):
        apu_fuel_flow = P('APU Fuel Flow',
                        array=np.ma.array([0, 120, 120, 120, 116, 112, 112, 112, 0, 0, 0, 120, 120, 120, 0, 0]))
        run = APURunning()
        run.derive(None, None, None, apu_fuel_flow, None)
        expected = ['-'] + ['Running'] * 7 + ['-'] * 3 + ['Running'] * 3 + ['-'] * 2
        np.testing.assert_array_equal(run.array, expected)

    def test_derive_apu_on(self):
        apu_on = M('APU On',
                   array=np.ma.array([0,0,1,1,1,0,0,0,0,1,1,1,0]),
                   mask=False,
                   values_mapping={0: '-', 1:'Open'})
        run = APURunning()
        run.derive(None, None, None, None, apu_on)
        expected = ['-']*2 + ['Running']*3 + ['-']*4 + ['Running']*3 + ['-']
        np.testing.assert_array_equal(run.array, expected)


class TestAPChannelsEngaged(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = APChannelsEngaged
        self.operational_combinations = [
            ('AP (1) Engaged', 'AP (2) Engaged'),
            ('AP (1) Engaged', 'AP (3) Engaged'),
            ('AP (2) Engaged', 'AP (3) Engaged'),
            ('AP (1) Engaged', 'AP (2) Engaged', 'AP (3) Engaged'),
        ]

    def test_single_ap(self):
        # Cannot auto_land on one AP
        ap1 = M(array=np.ma.array(data=[0,0,0,0,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')
        values_mapping = {0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'}
        eng = APChannelsEngaged()
        eng.derive(ap1, None, None)
        expected = M(array=np.ma.array(data=[0,0,0,0,0,0]),
                   values_mapping=values_mapping,
                   name='AP Channels Engaged',
                   frequency=1,
                   offset=0.1)
        assert_array_equal(expected.array, eng.array)

    def test_dual_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged')
        ap2 = M(array=np.ma.array(data=[0,0,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged')
        ap3 = None
        values_mapping = {0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'}
        eng = APChannelsEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0, 0, 1, 2, 1, 0]),
                   values_mapping=values_mapping,
                   name='AP Channels Engaged',
                   frequency=1,
                   offset=0.1)

        assert_array_equal(expected.array, eng.array)

    def test_triple_ap(self):
        ap1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (1) Engaged',
                   frequency=1,
                   offset=0.1)
        ap2 = M(array=np.ma.array(data=[0,1,0,1,1,0]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (2) Engaged',
                   frequency=1,
                   offset=0.2)
        ap3 = M(array=np.ma.array(data=[0,0,1,1,1,1]),
                   values_mapping={1:'Engaged',0:'-'},
                   name='AP (3) Engaged',
                   frequency=1,
                   offset=0.4)
        values_mapping = {0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'}
        eng = APChannelsEngaged()
        eng.derive(ap1, ap2, ap3)
        expected = M(array=np.ma.array(data=[0, 1, 2, 3, 2, 1]),
                   values_mapping=values_mapping,
                   name='AP Channels Engaged',
                   frequency=1,
                   offset=0.25)

        assert_array_equal(expected.array, eng.array)


class TestCargoSmokeOrFire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = CargoSmokeOrFire
        self.all_opts = CargoSmokeOrFire.get_dependency_names()
        self.param_array = np.ma.array(data=[0, 1, 0, 0, 0])

    def test_can_operate(self):
        for opt in self.all_opts:
            self.assertTrue(CargoSmokeOrFire.can_operate([opt]))

        for i in range(len(self.all_opts)):
            self.assertTrue(CargoSmokeOrFire.can_operate(self.all_opts[:i+1]))

    def test_derive(self):
        node = self.node_class()

        for opt in self.all_opts:
            derive_args = [P(name=opt, array=self.param_array, frequency=1)] + [None] * (len(self.all_opts) - 1)
            node.derive(*derive_args)
            self.assertEqual(node.state['Warning'], 1)
            self.assertEqual(node.offset, 0.0)

    def test_derive_multiple(self):
        node = self.node_class()

        for i in range(len(self.all_opts)):
            params = [P(name=opt, array=self.param_array, frequency=1) for opt in self.all_opts[:i+1]]
            derive_args = params + [None] * (len(self.all_opts) - (i+1))
            node.derive(*derive_args)
            self.assertEqual(node.state['Warning'], 1)
            self.assertEqual(node.offset, 0.0)


class TestConfiguration(unittest.TestCase, NodeTest):
    AVAILABLE_CONF_STATES = {
        'Airbus': {
            0: '0',
            10: '1',
            12: '1(T/O)',
            13: '1+F',
            16: '1*',
            20: '2',
            26: '2*',
            30: '3',
            33: '3+S',
            40: '4',
            50: '5',
            90: 'Full',
        },
        'Embraer': {
            1:  '0',
            2:  '1',
            4:  '2',
            8:  '3',
            16: '4',
            32: '5',
            64: 'Full',
        }
    }

    def setUp(self):
        self.node_class = Configuration

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_conf_angles.side_effect = ({}, {}, KeyError, {}, KeyError, {}, {})
        self.assertTrue(self.node_class.can_operate(
            ('Flap Including Transition', 'Slat Including Transition', 'Manufacturer', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', 'A320-201'),
            series=A('Series', 'A320-200'),
            family=A('Family', 'A320'),
        ))
        self.assertTrue(self.node_class.can_operate(
            ('Flap Including Transition', 'Slat Including Transition', 'Manufacturer', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', 'A330-301'),
            series=A('Series', 'A330-300'),
            family=A('Family', 'A330'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Including Transition', 'Slat Including Transition', 'Manufacturer', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', None),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Including Transition', 'Slat Including Transition', 'Manufacturer', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', 'A310-101'),
            series=A('Series', 'A310-100'),
            family=A('Family', 'A310'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Including Transition', 'Slat Including Transition', 'Manufacturer', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Boeing'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', None),
        ))
        self.assertTrue(self.node_class.can_operate(
            ('Flap Including Transition', 'Slat Including Transition', 'Manufacturer', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', 'A330'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Including Transition', 'Slat Including Transition', 'Manufacturer', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', None),
            series=A('Series', 'A340-500'),
            family=A('Family', None),
        ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive(self, at):
        at.get_conf_angles.return_value = {
            '0':    (0, 0, 0),
            '1':    (16, 0, 0),
            '1+F':  (16, 8, 0),
            '1*':   (20, 8, 0),
            '2':    (20, 14, 5),
            '2*':   (23, 14, 10),
            '3':    (23, 22, 15),
            'Full': (23, 32, 10),
        }
        at.constants.AVAILABLE_CONF_STATES = self.AVAILABLE_CONF_STATES
        _am = A('Model', 'A330-301')
        _as = A('Series', 'A330-300')
        _af = A('Family', 'A330')
        manufacturer = A('Manufacturer', 'Airbus')
        attributes = (_am, _as, _af, manufacturer)
        # Note: The last state is invalid...
        s = [0] * 4 + [16] * 4 + [16] * 4 + [20] * 1 + [20] * 4 + [23] * 1 + [23] * 4 + [23] * 4
        f = [0] * 4 + [0]  * 4 + [8]  * 4 + [8]  * 1 + [14] * 4 + [14] * 1 + [22] * 4 + [32] * 4
        z = lambda i: {x: str(x) for x in np.ma.unique(i)}
        slat = M('Slat', np.ma.array(s), values_mapping=z(s))
        flap = M('Flap', np.ma.array(f), values_mapping=z(f))
        node = self.node_class()
        node.derive(flap, slat, *attributes)
        attributes = (a.value for a in attributes)
        at.get_conf_angles.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, self.AVAILABLE_CONF_STATES['Airbus'])
        self.assertEqual(node.units, None)
        self.assertIsInstance(node.array, MappedArray)
        values = unique_values(node.array)
        self.assertEqual(values, {'Full': 4, '1+F': 4, '1': 4, '0': 4, '3': 4, '2': 4, '1*': 1, '2*': 1})

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive_embraer(self, at):
        at.get_conf_angles.return_value = {
            '0':    (0, 0, None),
            '1':    (15, 5, None),
            '2':    (15, 10, None),
            '3':    (15, 20, None),
            '4':    (25, 20, None),
            '5':    (25, 20, None),
            'Full': (25, 35, None),
        }
        at.constants.AVAILABLE_CONF_STATES = self.AVAILABLE_CONF_STATES
        _am = A('Model', None)
        _as = A('Series', None)
        _af = A('Family', 'ERJ-170/175')
        approach_landing = buildsections('Approach And Landing', (13, 27))
        taxi_in = buildsections('Taxi In', (27, 32))
        manufacturer = A('Manufacturer', 'Embraer')
        attributes = (_am, _as, _af, manufacturer)
        s = [0] * 4 + [25] * 4 + [0] * 4 + [15] * 4 + [15] * 4 + [15] * 1 + [25] * 1 + [25] * 4 + [25] * 4
        f = [0] * 4 + [20] * 4 + [0] * 4 + [5]  * 4 + [10] * 4 + [20] * 1 + [20] * 1 + [35] * 4 + [20] * 4
        z = lambda i: {x: str(x) for x in np.ma.unique(i)}
        slat = M('Slat', np.ma.array(s), values_mapping=z(s))
        flap = M('Flap', np.ma.array(f), values_mapping=z(f))
        node = self.node_class()
        node.derive(flap, slat, *attributes, approach_landing, taxi_in)
        attributes = (a.value for a in attributes)
        at.get_conf_angles.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, self.AVAILABLE_CONF_STATES['Embraer'])
        self.assertEqual(node.units, None)
        self.assertIsInstance(node.array, MappedArray)
        values = unique_values(node.array)
        self.assertEqual(values, {'Full': 4, '0': 8, '1': 4, '2': 4, '3': 1, '4': 4, '5': 5})


class TestDaylight(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Latitude Smoothed', 'Longitude Smoothed',
                     'Start Datetime', 'HDF Duration')]
        opts = Daylight.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_daylight_aligning(self):
        lat = P('Latitude', np.ma.array([51.1789]*128), offset=0.1)
        lon = P('Longitude', np.ma.array([-1.8264]*128))
        start_dt = A('Start Datetime', datetime.datetime(2012,6,20, 20,25))
        dur = A('HDF Duration', 128)

        don = Daylight()
        don.get_derived((lat, lon, start_dt, dur))
        self.assertEqual(list(don.array), [np.ma.masked] + ['Day']*31)
        self.assertEqual(don.frequency, 0.25)
        self.assertEqual(don.offset, 0)

    def test_father_christmas(self):
        # Starting on the far side of the world, he flies all round
        # delivering parcels mostly by night (in the northern lands).
        lat=P('Latitude', np.ma.arange(60,64,1/64.0))
        lon=P('Longitude', np.ma.arange(-180,180,90/64.0))
        start_dt = A('Start Datetime', datetime.datetime(2012,12,25,1,00))
        dur = A('HDF Duration', 256)

        don = Daylight()
        don.align_frequency = 1/64.0  # Force frequency to simplify test
        don.get_derived((lat, lon, start_dt, dur))
        expected = ['Day', 'Night', 'Night', 'Night']
        np.testing.assert_array_equal(don.array, expected)  # FIX required to test as no longer superframe samples


class TestDualInput(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = DualInput
        self.pilot_map = {0: '-', 1: 'Captain', 2: 'First Officer'}
        self.operational_combinations = [
            ('Pilot Flying', 'Sidestick Angle (Capt)', 'Sidestick Angle (FO)', 'Family')
        ]

    def test_derive(self):
        pilot_array = MappedArray([1] * 20 + [0] * 10 + [2] * 20,
                                  values_mapping=self.pilot_map)
        capt_array = np.ma.concatenate((15 + np.arange(20), np.zeros(30)))
        fo_array = np.ma.concatenate((np.zeros(30), 15 + np.arange(20)))
        # Dual input
        fo_array[5:10] = 15
        capt_array[30:35] = 1.2
        pilot = M('Pilot Flying', pilot_array, values_mapping=self.pilot_map)
        capt = P('Sidestick Angle (Capt)', capt_array)
        fo = P('Sidestick Angle (FO)', fo_array)
        family = A('Family', 'A330')
        node = self.node_class()
        node.derive(pilot, capt, fo, family)

        expected_array = MappedArray(
            np.ma.zeros(capt_array.size),
            values_mapping=self.node_class.values_mapping)

        expected_array[5:10] = 'Dual'
        np.testing.assert_array_equal(node.array, expected_array)
        np.testing.assert_array_equal(node.array[30:35], ['-']*5)

    def test_derive_from_hdf(self):
        (capt, fo), phase = self.get_params_from_hdf(
            os.path.join(test_data_path, 'dual_input.hdf5'),
            ['Sidestick Angle (Capt)', 'Sidestick Angle (FO)'])

        pilot_array = MappedArray(np.ones(840),
                                  values_mapping=self.pilot_map)
        pilot = M('Pilot Flying', pilot_array, values_mapping=self.pilot_map)
        family = A('Family', 'A330')

        node = self.node_class()
        node.derive(pilot, capt, fo, family)

        expected_array = MappedArray(
            np.ma.zeros(pilot.array.size),
            values_mapping=self.node_class.values_mapping)

        expected_array[178:188] = 'Dual'
        expected_array[421:464] = 'Dual'
        expected_array[487:506] = 'Dual'
        np.testing.assert_array_equal(node.array, expected_array)

    def test_not_triggered_at_minimum_resolution(self):
        capt_array = np.ma.array([10.0]*20)
        fo_array = np.ma.array([0.0]*20)
        # Dual input

        # The resolution of the signals on some Airbus types is given by:
        resolution = 0.703129
        # So the sidesticks can sit with one bit offset in pitch and roll at
        # this angle (given 10% increase for safety):
        min_res = ((resolution*resolution)*2*1.1)**0.5

        fo_array[5:10] = min_res
        pilot = M('Pilot Flying', np.ma.array([1]*20), values_mapping=self.pilot_map)
        capt = P('Sidestick Angle (Capt)', capt_array)
        fo = P('Sidestick Angle (FO)', fo_array)

        family = A('Family', 'A330')
        node = self.node_class()
        node.derive(pilot, capt, fo, family)

        expected_array = MappedArray(
            np.ma.zeros(capt_array.size),
            values_mapping=self.node_class.values_mapping)
        expected_array[5:10] = '-'
        np.testing.assert_array_equal(node.array, expected_array)

class TestEng_1_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_1_Fire
        self.operational_combinations = [('Eng (1) Fire On Ground', 'Eng (1) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (1) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (1) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_2_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_2_Fire
        self.operational_combinations = [('Eng (2) Fire On Ground', 'Eng (2) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (2) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (2) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_3_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_3_Fire
        self.operational_combinations = [('Eng (3) Fire On Ground', 'Eng (3) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (3) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (3) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEng_4_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_4_Fire
        self.operational_combinations = [('Eng (4) Fire On Ground', 'Eng (4) Fire In Air')]

    def test_derive(self):
        fire_gnd = M(
            name='Eng (4) Fire On Ground',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        fire_air = M(
            name='Eng (4) Fire On Ground',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Fire'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(fire_gnd, fire_air)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestEventMarker(unittest.TestCase):

    def test_can_operate(self):
        self.assertTrue(EventMarker.can_operate(('Event Marker (1)',)))
        self.assertTrue(EventMarker.can_operate(('Event Marker (2)',)))
        self.assertTrue(EventMarker.can_operate(('Event Marker (3)',)))
        self.assertTrue(EventMarker.can_operate(('Event Marker (Capt)',)))
        self.assertTrue(EventMarker.can_operate(('Event Marker (FO)',)))
        self.assertTrue(EventMarker.can_operate(('Event Marker (1)',
                                                 'Event Marker (2)',
                                                 'Event Marker (3)')))
        self.assertTrue(EventMarker.can_operate(('Event Marker (Capt)',
                                                 'Event Marker (FO)')))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_Fire(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_Fire
        self.operational_combinations = [
            ('Eng (1) Fire',), ('Eng (2) Fire',), ('Eng (3) Fire',), ('Eng (4) Fire',),
            ('Eng (1) Fire', 'Eng (2) Fire'), ('Eng (1) Fire', 'Eng (3) Fire'),
            ('Eng (1) Fire', 'Eng (4) Fire'), ('Eng (2) Fire', 'Eng (3) Fire'),
            ('Eng (2) Fire', 'Eng (4) Fire'), ('Eng (3) Fire', 'Eng (4) Fire'),
            ('Eng (1) Fire', 'Eng (2) Fire', 'Eng (3) Fire'),
            ('Eng (1) Fire', 'Eng (2) Fire', 'Eng (4) Fire'),
            ('Eng (1) Fire', 'Eng (3) Fire', 'Eng (4) Fire'),
            ('Eng (2) Fire', 'Eng (3) Fire', 'Eng (4) Fire'),
            ('Eng (1) Fire', 'Eng (2) Fire', 'Eng (3) Fire', 'Eng (4) Fire'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngRunning(unittest.TestCase):

    def test_class_name(self):
        self.assertEqual(Eng1Running.get_name(), 'Eng (1) Running')
        self.assertEqual(Eng2Running.get_name(), 'Eng (2) Running')
        self.assertEqual(Eng3Running.get_name(), 'Eng (3) Running')
        self.assertEqual(Eng4Running.get_name(), 'Eng (4) Running')

    def test_can_operate(self):
        self.assertTrue(EngRunning.can_operate(['Eng (0) N1']))
        self.assertTrue(EngRunning.can_operate(['Eng (0) N2']))
        self.assertTrue(EngRunning.can_operate(['Eng (0) Np']))
        self.assertTrue(EngRunning.can_operate(['Eng (0) Fuel Flow']))
        self.assertTrue(EngRunning.can_operate(['Eng (0) N1', 'Eng (0) N2']))
        # Check engnums are replaced correctly
        self.assertTrue(Eng1Running.can_operate(['Eng (1) N1']))
        self.assertTrue(Eng2Running.can_operate(['Eng (2) N1']))
        self.assertTrue(Eng3Running.can_operate(['Eng (3) N1']))
        self.assertTrue(Eng4Running.can_operate(['Eng (4) N1']))

    def test_determine_running(self):
        # This is tested by the TestEng_AllRunning below.
        pass


class TestEng_AllRunning(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = Eng_AllRunning
        self.operational_combinations = [
            ('Eng (*) N1 Min',),
            ('Eng (*) Np Min',),
            ('Eng (*) N2 Min',), ('Eng (*) Fuel Flow Min',),
            ('Eng (*) N2 Min', 'Eng (*) Fuel Flow Min'),
        ]

    def test_derive_n2_only(self):
        n2_array = np.ma.array([0, 5, 10, 15, 11, 5, 0])
        n2 = P('Eng (*) N2 Min', array=n2_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, n2, None, None)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_ff_only(self):
        ff_array = np.ma.array([10, 20, 50, 55, 51, 15, 10])
        ff = P('Eng (*) Fuel Flow Min', array=ff_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, None, None, ff)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_n2_ff(self):
        n2_array = np.ma.array([0, 5, 11, 15, 11, 5, 0])
        n2 = P('Eng (*) N2 Min', array=n2_array)
        ff_array = np.ma.array([10, 20, 50, 55, 51, 51, 10])
        ff = P('Eng (*) Fuel Flow Min', array=ff_array)
        expected = [False, False, True, True, True, True, False]
        node = self.node_class()
        node.derive(None, n2, None, ff)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_n1_only(self):
        # fallback to N1 if no N2 and FF
        n1_array = np.ma.array([0, 5, 10, 15, 11, 5, 0])
        n1 = P('Eng (*) N2 Min', array=n1_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, n1, None, None)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_np_only(self):
        # Propellors
        np_array = np.ma.array([0, 5, 10, 15, 11, 5, 0])
        eng_np = P('Eng (*) Np Min', array=np_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, None, eng_np , None)
        self.assertEqual(node.array.raw.tolist(), expected)



class TestEng_AnyRunning(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = Eng_AnyRunning
        self.operational_combinations = [
            ('Eng (*) N1 Max',),
            ('Eng (*) Np Max',),
            ('Eng (*) N2 Max',), ('Eng (*) Fuel Flow Max',),
            ('Eng (*) N2 Max', 'Eng (*) Fuel Flow Max'),
        ]

    def test_derive_n2_only(self):
        n2_array = np.ma.array([0, 5, 10, 15, 11, 5, 0])
        n2 = P('Eng (*) N2 Max', array=n2_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, n2, None, None)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_ff_only(self):
        ff_array = np.ma.array([10, 20, 50, 55, 51, 15, 10])
        ff = P('Eng (*) Fuel Flow Max', array=ff_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, None, None, ff)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_n2_ff(self):
        n2_array = np.ma.array([0, 5, 11, 15, 11, 5, 0])
        n2 = P('Eng (*) N2 Max', array=n2_array)
        ff_array = np.ma.array([10, 20, 50, 55, 51, 51, 10])
        ff = P('Eng (*) Fuel Flow Max', array=ff_array)
        expected = [False, False, True, True, True, True, False]
        node = self.node_class()
        node.derive(None, n2, None, ff)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_n1_only(self):
        # fallback to N1
        n1_array = np.ma.array([0, 5, 10, 15, 11, 5, 0])
        n1 = P('Eng (*) N1 Max', array=n1_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(n1, None, None, None)
        self.assertEqual(node.array.raw.tolist(), expected)

    def test_derive_np_only(self):
        # Propellors
        np_array = np.ma.array([0, 5, 10, 15, 11, 5, 0])
        eng_np = P('Eng (*) Np Min', array=np_array)
        expected = [False, False, False, True, True, False, False]
        node = self.node_class()
        node.derive(None, None, eng_np , None)
        self.assertEqual(node.array.raw.tolist(), expected)


class TestEng_Oil_Press_Warning(unittest.TestCase):
    def test_can_operate(self):
        combinations = Eng_Oil_Press_Warning.get_operational_combinations()
        self.assertTrue(('Eng (1) Oil Press Low',) in combinations)
        self.assertTrue(('Eng (2) Oil Press Low',) in combinations)
        self.assertTrue(('Eng (3) Oil Press Low',) in combinations)
        self.assertTrue(('Eng (4) Oil Press Low',) in combinations)
        self.assertTrue(('Eng (1) Oil Press Low',
                         'Eng (2) Oil Press Low') in combinations)
        self.assertTrue(('Eng (1) Oil Press Low',
                         'Eng (2) Oil Press Low',
                         'Eng (3) Oil Press Low',
                         'Eng (4) Oil Press Low',) in combinations)

    def test_derive(self):
        eng_values_mapping = {0: '-', 1: 'Low Press'}
        eng_1_array = np.ma.array([1,1,0,0,1,0,1,0])
        eng_1 = M('Eng (1) Oil Press Low', array=eng_1_array,
                  values_mapping=eng_values_mapping)
        eng_2_array = np.ma.array([0,1,1,0,0,0,0,1])
        eng_2 = M('Eng (2) Oil Press Low', array=eng_2_array,
                  values_mapping=eng_values_mapping)
        node = Eng_Oil_Press_Warning()
        node.derive(eng_1, eng_2, None, None)
        node

    def test_derive_extra(self):
        # Not sure what "node" does in the last line, so clunky tests added.
        eng_values_mapping = {0: 'Closed', 1: 'Open'}
        eng_1_array = np.ma.array([1,1,0,0,1,0,1,0])
        eng_1 = M('Eng (1) Bleed', array=eng_1_array,
                  values_mapping=eng_values_mapping)
        eng_2_array = np.ma.array([0,1,1,0,0,0,0,1])
        eng_2 = M('Eng (2) Bleed', array=eng_2_array,
                  values_mapping=eng_values_mapping)
        node = EngBleedOpen()
        node.derive(eng_1, eng_2, None, None)
        self.assertEqual(node.array.raw.tolist(), [True, True, True, False, True, False, True, True])
        self.assertEqual(node.values_mapping, eng_values_mapping)


class TestEngBleedOpen(unittest.TestCase):
    def test_can_operate(self):
        combinations = EngBleedOpen.get_operational_combinations()
        self.assertFalse(('Eng (1) Bleed',) in combinations)
        self.assertFalse(('Eng (2) Bleed',) in combinations)
        self.assertFalse(('Eng (3) Bleed',) in combinations)
        self.assertFalse(('Eng (4) Bleed',) in combinations)
        self.assertTrue(('Eng (1) Bleed',
                         'Eng (2) Bleed') in combinations)
        self.assertTrue(('Eng (1) Bleed',
                         'Eng (2) Bleed',
                         'Eng (3) Bleed',
                         'Eng (4) Bleed',) in combinations)

    def test_derive(self):
        # Copy from TestEng_Oil_Press_Warning
        eng_values_mapping = {0: 'Closed', 1: 'Open'}
        eng_1_array = np.ma.array([1,1,0,0,1,0,1,0])
        eng_1 = M('Eng (1) Bleed', array=eng_1_array,
                  values_mapping=eng_values_mapping)
        eng_2_array = np.ma.array([0,1,1,0,0,0,0,1])
        eng_2 = M('Eng (2) Bleed', array=eng_2_array,
                  values_mapping=eng_values_mapping)
        node = EngBleedOpen()
        node.derive(eng_1, eng_2, None, None)
        node

    def test_derive_extra(self):
        # Not sure what
        eng_values_mapping = {0: 'Closed', 1: 'Open'}
        eng_1_array = np.ma.array([1,1,0,0,1,0,1,0])
        eng_1 = M('Eng (1) Bleed', array=eng_1_array,
                  values_mapping=eng_values_mapping)
        eng_2_array = np.ma.array([0,1,1,0,0,0,0,1])
        eng_2 = M('Eng (2) Bleed', array=eng_2_array,
                  values_mapping=eng_values_mapping)
        node = EngBleedOpen()
        node.derive(eng_1, eng_2, None, None)
        self.assertEqual(node.array.raw.tolist(), [True, True, True, False, True, False, True, True])
        self.assertEqual(node.values_mapping, eng_values_mapping)


class TestEngThrustModeRequired(unittest.TestCase):
    def test_can_operate(self):
        opts = ThrustModeSelected.get_operational_combinations()
        self.assertTrue(('Thrust Mode Selected (L)',) in opts)
        self.assertTrue(('Thrust Mode Selected (R)',) in opts)
        self.assertTrue(('Thrust Mode Selected (L)', 'Thrust Mode Selected (R)') in opts)

    def test_derive_one_param(self):
        thrust_array = np.ma.array([0, 0, 1, 0])
        thrust = M('Thrust Mode Selected (R)', array=thrust_array,
                   values_mapping=ThrustModeSelected.values_mapping)
        node = ThrustModeSelected()
        node.derive(None, thrust)
        self.assertEqual(thrust.array.raw.tolist(), thrust_array.tolist())

    def test_derive_four_params(self):
        thrust_array1 = np.ma.array([0, 0, 1, 0],
                                    mask=[False, False, True, False])
        thrust_array2 = np.ma.array([1, 0, 0, 0],
                                    mask=[True, False, False, False])
        thrust1 = M('Thrust Mode Selected (L)', array=thrust_array1,
                    values_mapping=ThrustModeSelected.values_mapping)
        thrust2 = M('Thrust Mode Selected (R)', array=thrust_array2,
                    values_mapping=ThrustModeSelected.values_mapping)
        node = ThrustModeSelected()
        node.derive(thrust1, thrust2)

        self.assertEqual(
            node.array.tolist(),
            MappedArray([1, 0, 1, 0],
                        mask=[True, False, True, False],
                        values_mapping=ThrustModeSelected.values_mapping).tolist())


class TestFlap(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Flap

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_flap_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
            ac_type=A('Aircraft Type', 'aeroplane')
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', None),
            ac_type=A('Aircraft Type', 'aeroplane')
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'AW139'),
            series=A('Series', 'AW139'),
            family=A('Family', 'AW139'),
            ac_type=A('Aircraft Type', 'helicopter')
        ))
        self.assertFalse(self.node_class.can_operate(
            tuple(), frame=A('Frame', 'L382-Hercules')))
        self.assertTrue(self.node_class.can_operate(
            ('Altitude AAL',), frame=A('Frame', 'L382-Hercules')))
        self.assertFalse(self.node_class.can_operate(
            tuple(), family=A('Family', 'C208')))
        self.assertTrue(self.node_class.can_operate(
            ('Altitude AAL',), family=A('Family', 'C208')))
        self.assertFalse(self.node_class.can_operate(
            tuple(), family=A('Family', 'Citation VLJ')))
        self.assertTrue(self.node_class.can_operate(
            ('HDF Duration', 'Landing', 'Takeoff'), family=A('Family', 'Citation VLJ')))

    @patch('analysis_engine.library.at')
    def test_derive(self, at):
        at.get_flap_map.return_value = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        _am = A('Model', 'B737-333')
        _as = A('Series', 'B737-300')
        _af = A('Family', 'B737 Classic')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + list(range(42)) + [42] * 5)
        flap = P(name='Flap Angle', array=array, frequency=2)
        node = self.node_class()
        node.derive(flap, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        self.assertEqual(node.array.raw.tolist(), [0] * 11 + [40] * 92 + [None])

    @patch('analysis_engine.library.at')
    def test_derive__md82(self, at):
        at.get_flap_map.return_value = {f: str(f) for f in (0, 13, 20, 25, 30, 40)}
        _am = A('Model', None)
        _as = A('Series', None)
        _af = A('Family', 'DC-9')
        attributes = (_am, _as, _af)
        array = np.ma.array(list(range(50)) + list(range(-5, 0)) + [13.1, 1.3, 10, 10])
        flap = P(name='Flap Angle', array=array, frequency=1)
        for index in (1, 57, 58):
            flap.array[index] = np.ma.masked
        node = self.node_class()
        node.derive(flap, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        # Note multipliers introduced as output frequency now raised to 4Hz.
        self.assertEqual(node.array.size, 59*4)
        self.assertEqual(node.array.raw.tolist(),[0.0]+7*[None]+189*[40]+6*[0]+19*[13]+3*[0]+11*[None])
        self.assertEqual(node.array.mask.sum(), 18)
        self.assertTrue(node.array.mask[1*4])
        self.assertTrue(node.array.mask[57*4])
        self.assertTrue(node.array.mask[58*4])

    @patch('analysis_engine.library.at')
    #
    # Note: This test is somewhat academic as the Beechcraft does not record
    # Flap Angle, rather has discrete switches for Flap position.
    #
    def test_derive__beechcraft(self, at):
        at.get_flap_map.return_value = {f: str(f) for f in (0, 17.5, 35)}
        _am = A('Model', '1900D')
        _as = A('Series', '1900D')
        _af = A('Family', '1900')
        attributes = (_am, _as, _af)
        array = np.ma.array((0, 5, 7.2, 17, 17.4, 17.4, 17.4, 17.4, 17.4, 17.9, 20, 30, 30))
        flap = P(name='Flap Angle', array=array, frequency=1) # Frame sample rate is 1Hz.
        node = self.node_class()
        node.derive(flap, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        ma_test.assert_masked_array_equal(
            node.array.raw[::4],
            np.ma.array([0.0, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 35.0, 35.0, 35.0, 35.0]),
        )

    @patch('analysis_engine.library.at')
    def test_derive__hercules(self, at):
        # No flap recorded; ensure it converts exactly the same...
        at.get_flap_map.return_value = {0: '0', 50: '50', 100: '100'}
        _am = A('Model', None)
        _as = A('Series', None)
        _af = A('Family', 'C-130')
        _fr = A('Frame', 'L382-Hercules')
        array = np.ma.array((0, 0, 50, 1500, 1500, 1500, 2500, 2500, 1500, 1500, 50, 50))
        alt_aal = P(name='Altitude AAL', array=array)
        node = self.node_class()
        node.derive(None, _am, _as, _af, _fr, alt_aal)
        self.assertEqual(at.get_flap_map.call_count, 0)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.PERCENT)
        self.assertIsInstance(node.array, MappedArray)
        ma_test.assert_masked_array_equal(node.array, np.ma.array((50, 50, 50, 0, 0, 0, 0, 0, 50, 50, 100, 100)))

    def test_derive__c208(self):
        at.get_flap_map.return_value = {0: '0', 10: '10', 20: '20', 40: '40'}
        _am = A('Model', None)
        _as = A('Series', None)
        _af = A('Family', 'C208')
        _fr = A('Frame', None)
        array = np.ma.array((0, 0, 50, 450, 1000, 1500, 2500, 1500, 1000, 450, 50, 50))
        alt_aal = P(name='Altitude AAL', array=array)
        node = self.node_class()
        node.derive(None, _am, _as, _af, _fr, alt_aal)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        ma_test.assert_masked_array_equal(node.array, np.ma.array((40, 40, 40, 40, 0, 0, 0, 0, 0, 40, 40, 40)))

    def test_derive__citation(self):
        _am = A('Model', None)
        _as = A('Series', None)
        _fr = A('Frame', None)
        _af = A('Family', 'Citation VLJ')
        _hd = A('HDF Duration', 18)
        _to = buildsections('Takeoff', (2, 4), (10, 12))
        _ld = buildsections('Landing', (6, 8), (14, 16))
        node = self.node_class()
        node.derive(None, _am, _as, _af, _fr, None, _hd, _to, _ld)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        self.assertEqual(node.frequency, 1)
        self.assertEqual(node.offset, 0)
        ma_test.assert_masked_array_equal(node.array, np.ma.array((0, 0, 15, 15, 15, 0, 30, 30, 30, 0, 15, 15, 15, 0, 30, 30, 30, 0)))

    def test_derive__flap_1_2(self):
        _am = A('Model', 'B737-448(F)')
        _as = A('Series', 'B737-400')
        _fr = A('Frame', '737-i')
        _af = A('Family', 'B737 Classic')

        flap_angle = load(os.path.join(test_data_path, 'ae-1165-flap_angle.nod'))
        node = self.node_class()
        node.derive(flap_angle, _am, _as, _af, _fr, None, None, None, None)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        # Should be 4 slices @ flap 1, 4 slices @ flap 2 and 3 slices @ flap 5
        flap1slices = runs_of_ones(node.array == 1)
        flap2slices = runs_of_ones(node.array == 2)
        flap5slices = runs_of_ones(node.array == 5)
        self.assertEqual(len(flap1slices),4)
        self.assertEqual(len(flap2slices),4)
        self.assertEqual(len(flap5slices),3)
        # Interested in flap transition around index 7500 & 7700
        # it should go 1 - 2 - 5 (before fix, flap 2 was missed).
        self.assertTrue(flap1slices[2].stop == flap2slices[2].start)
        self.assertTrue(flap2slices[2].stop == flap5slices[1].start)

class TestFlapExcludingTransition(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapExcludingTransition

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_flap_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', None),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'AW139'),
            series=A('Series', 'AW139'),
            family=A('Family', 'AW139'),
            ac_type=A('Aircraft Type', 'helicopter')
        ))

    @patch('analysis_engine.library.at')
    def test_derive(self, at):
        at.get_flap_map.return_value = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        _am = A('Model', 'B737-333')
        _as = A('Series', 'B737-300')
        _af = A('Family', 'B737 Classic')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + list(range(42)) + [42] * 5)
        flap = P(name='Flap Angle', array=array, frequency=2)
        node = self.node_class()
        node.derive(flap, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        self.assertEqual(node.frequency, 4)
        self.assertEqual(node.array.raw.tolist(), [0] * 95 + [40] * 8 + [None])

    def test_derive__flap_1_2(self):
        _am = A('Model', 'B737-448(F)')
        _as = A('Series', 'B737-400')
        _af = A('Family', 'B737 Classic')

        flap_angle = load(os.path.join(test_data_path, 'ae-1165-flap_angle.nod'))
        node = self.node_class()
        node.derive(flap_angle, _am, _as, _af)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        # Should be 4 slices @ flap 1, 4 slices @ flap 2 and 3 slices @ flap 5
        flap1slices = runs_of_ones(node.array == 1)
        flap2slices = runs_of_ones(node.array == 2)
        flap5slices = runs_of_ones(node.array == 5)
        self.assertEqual(len(flap1slices),4)
        self.assertEqual(len(flap2slices),4)
        self.assertEqual(len(flap5slices),3)
        # Interested in flap transition around index 7500 & 7700
        # it should go 1 - 2 - 5 (before fix, flap 2 was missed).
        self.assertTrue(flap1slices[2].stop == flap2slices[2].start)
        self.assertTrue(flap2slices[2].stop == flap5slices[1].start)


class TestFlapIncludingTransition(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapIncludingTransition

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_flap_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', None),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'AW139'),
            series=A('Series', 'AW139'),
            family=A('Family', 'AW139'),
            ac_type=A('Aircraft Type', 'helicopter')
        ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive(self, at):
        at.get_flap_map.return_value = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        _am = A('Model', 'B737-333')
        _as = A('Series', 'B737-300')
        _af = A('Family', 'B737 Classic')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + list(range(42)) + [42] * 5)
        flap = P(name='Flap Angle', array=array, frequency=2)
        node = self.node_class()
        node.derive(flap, None, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 5.0,
                    5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 15.0,
                    15.0, 15.0, 15.0, 15.0, 25.0, 25.0, 25.0, 25.0,
                    25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 30.0, 30.0,
                    30.0, 30.0, 30.0, 40.0, 40.0, 40.0, 40.0, 40.0,
                    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
                    40.0, 40.0, 40.0]
        self.assertEqual(node.array.raw.tolist(), expected)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__recorded_flap(self, at):
        at.get_flap_map.return_value = {0: '0', 10: '10', 20: '20', 39: '39'}
        _am = A('Model', 'B737-333')
        _as = A('Series', 'B737-300')
        _af = A('Family', 'B737 Classic')
        attributes = (_am, _as, _af)

        flap_mapping = {8: '39', 1: '0', 2: '10', 4: '20'}
        array = np.ma.repeat((1, 2, 4, 8), 10)
        array.mask = np.ma.getmaskarray(array)
        flap_array = MappedArray(array, values_mapping=flap_mapping)
        flap = M(name='Flap', array=flap_array, frequency=2)
        node = self.node_class()
        node.derive(None, flap, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        expected = np.repeat((0, 10, 20, 39), 10)
        self.assertEqual(node.array.raw.tolist(), expected.tolist())

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__single_point(self, at):
        at.get_flap_map.return_value = {0: '0', 10: '10', 20: '20', 30: '30', 40: '40'}
        _am = A('Model', 'B737-333')
        _as = A('Series', 'B737-300')
        _af = A('Family', 'B737 Classic')
        attributes = (_am, _as, _af)

        flap_mapping = {39: '39', 0: '0', 10: '10', 20: '20'}
        array = np.ma.array([0]*4 + [10, 10, 10] + [0]*4 + [10, 10, 10, 40, 40, 40] + \
                            [20] * 6 + [40, 10] + [0] * 6)
        array = (1.1 * array + 0.9 * np.roll(array, 1))/ 2.0
        array.mask = np.ma.getmaskarray(array)
        #flap_array = MappedArray(array, values_mapping=flap_mapping)
        flap = P(name='Flap Angle', array=array, frequency=1)
        node = self.node_class()
        node.derive(flap, None, *attributes)
        attributes = (a.value for a in attributes)
        at.get_flap_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_flap_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        expected = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0,
                    10.0, 10.0, 30.0, 30.0, 40.0, 40.0, 30.0, 20.0, 20.0, 20.0,
                    20.0, 20.0, 30.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(node.array.raw.tolist(), expected)


class TestFlapLever(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapLever

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(
            ('Flap Lever Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic')),
        )
        self.assertFalse(self.node_class.can_operate(
            ('Flap Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'AW139'),
            series=A('Series', 'AW139'),
            family=A('Family', 'AW139'),
            ac_type=A('Aircraft Type', 'helicopter')
        ))

    @unittest.skip('Test not implemented.')
    def test_derive(self):
        pass


'''
Replacing Flap Lever (Synthetic) with new algorithm. Does not consider Slat.

class TestFlapLeverSynthetic(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapLeverSynthetic

    def test_can_operate(self):
        with patch('analysis_engine.multistate_parameters.at') as at:
            at.get_flap_map.side_effect = [{}, {}, KeyError]
            self.assertTrue(self.node_class.can_operate(
                ('Flap Angle', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertTrue(self.node_class.can_operate(
                ('Flap Angle', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            # Required parameter Flap Angle missing.
            self.assertFalse(self.node_class.can_operate(
                ('Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertFalse(self.node_class.can_operate(
                ('Flap Angle', 'Model', 'Series', 'Family'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__crj900(self, at):
        at.get_conf_angles.side_effect = KeyError
        at.get_lever_map.return_value = {f: str(f) for f in (0, 1, 8, 20, 30, 45)}
        at.get_lever_angles.return_value = {
            '0':  (0, 0, None),
            '1':  (20, 0, None),
            '8':  (20, 8, None),
            '20': (20, 20, None),
            '30': (25, 30, None),
            '45': (25, 45, None),
        }
        # Prepare our generated flap and slat arrays:
        flap_array = [0.0, 0, 8, 8, 0, 0, 0, 8, 20, 30, 45, 8, 0, 0, 0]
        slat_array = [0.0, 0, 20, 20, 20, 0, 0, 20, 20, 25, 25, 20, 20, 0, 0]
        flap_array = MappedArray(np.repeat(flap_array, 10),
                values_mapping={f: str(f) for f in (0, 8, 20, 30, 45)})
        slat_array = MappedArray(np.repeat(slat_array, 10),
                values_mapping={s: str(s) for s in (0, 20, 25)})

        ### Add some noise to make our flap and slat angles more realistic:
        ##flap_array += np.ma.sin(range(len(flap_array))) * 0.1
        ##slat_array -= np.ma.sin(range(len(slat_array))) * 0.1

        # Derive the synthetic flap lever:
        flap = M('Flap', flap_array)
        slat = M('Slat', slat_array)
        flaperon = None
        model = A('Model', 'CRJ900 (CL-600-2D24)')
        series = A('Series', 'CRJ900')
        family = A('Family', 'CL-600')
        node = self.node_class()
        node.derive(flap, slat, flaperon, model, series, family)

        # Check against an expected array of lever detents:
        expected = [0, 0, 8, 8, 1, 0, 0, 8, 20, 30, 45, 8, 1, 0, 0]
        mapping = {x: str(x) for x in sorted(set(expected))}
        array = MappedArray(np.repeat(expected, 10), values_mapping=mapping)
        np.testing.assert_array_equal(node.array, array)

    def test_derive__b737ng(self):
        # Prepare our generated flap array:
        flap_array = [0.0, 0, 5, 2, 1, 0, 0, 10, 15, 25, 30, 40, 0, 0, 0]
        flap_array = MappedArray(np.repeat(flap_array, 10),
                values_mapping={f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)})

        ### Add some noise to make our flap angles more realistic:
        ##flap_array += np.ma.sin(range(len(flap_array))) * 0.05

        # Derive the synthetic flap lever:
        flap = M('Flap', flap_array)
        slat = None
        flaperon = None
        model = A('Model', 'B737-333')
        series = A('Series', 'B737-300')
        family = A('Family', 'B737 Classic')
        node = self.node_class()
        node.derive(flap, slat, flaperon, model, series, family)

        # Check against an expected array of lever detents:
        expected = [0, 0, 5, 2, 1, 0, 0, 10, 15, 25, 30, 40, 0, 0, 0]
        mapping = {x: str(x) for x in sorted(set(expected))}
        array = MappedArray(np.repeat(expected, 10), values_mapping=mapping)
        np.testing.assert_array_equal(node.array, array)

    def test_derive__a330(self):
        # A330 uses Configuration conditions.
        # Prepare our generated flap and slat arrays:
        #                 -  - 1+F   -  -  1   1*  2   2*  3  Full -
        slat_array =     [0, 0, 16, 16, 0, 16, 20, 20, 23, 23, 23, 0]
        flap_array =     [0, 0,  8,  8, 0,  0,  8, 14, 14, 22, 32, 0]
        flaperon_array = [0, 0,  5,  0, 0,  0, 10, 10, 10, 10, 10, 0]
        expected = ['Lever 0', 'Lever 0', 'Lever 1', 'Lever 0', 'Lever 0',
                    'Lever 1', 'Lever 2', 'Lever 2', 'Lever 3', 'Lever 3',
                    'Lever Full', 'Lever 0']
        repeat = 1
        flap_array = MappedArray(np.repeat(flap_array, repeat),
                values_mapping={f: str(f) for f in (0, 8, 14, 22, 32)})
        slat_array = MappedArray(np.repeat(slat_array, repeat),
                values_mapping={s: str(s) for s in (0, 16, 20, 23)})
        flaperon_array = MappedArray(np.repeat(flaperon_array, repeat),
                values_mapping={a: str(a) for a in (0, 5, 10)})

        # Derive the synthetic flap lever:
        flap = M('Flap', flap_array)
        slat = M('Slat', slat_array)
        flaperon = M('Flaperon', flaperon_array)
        model = A('Model', None)
        series = A('Series', None)
        family = A('Family', 'A330')
        node = self.node_class()
        node.derive(flap, slat, flaperon, model, series, family)

        mapping = {x: str(x) for x in sorted(set(expected))}
        self.assertEqual(list(node.array), list(np.repeat(expected, repeat)))
'''




class TestFlapLeverSynthetic(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapLeverSynthetic

    def test_can_operate(self):
        with patch('analysis_engine.multistate_parameters.at') as at:
            at.get_conf_angles.side_effect = KeyError
            at.get_lever_angles.return_value = {}
            # Test we can operate for something in the lever mapping.
            self.assertTrue(self.node_class.can_operate(
                ('Flap', 'Slat', 'Manufacturer', 'Model', 'Series', 'Family'),
                manufacturer=A('Manufacturer', 'Bombardier'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            # Test we can operate for the above even though slat missing.
            self.assertTrue(self.node_class.can_operate(
                ('Flap', 'Manufacturer', 'Model', 'Series', 'Family'),
                manufacturer=A('Manufacturer', 'Bombardier'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
        # Test we can operate for something not in the lever mapping.
        self.assertTrue(self.node_class.can_operate(
            ('Flap', 'Manufacturer',  'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Boeing'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))
        # Test we can operate even though the above *has* a slat.
        self.assertTrue(self.node_class.can_operate(
            ('Flap', 'Slat', 'Manufacturer', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Boeing'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))
        # Test we can operate with Flaperons
        self.assertTrue(self.node_class.can_operate(
            ('Flap', 'Slat', 'Flaperon', 'Manufacturer', 'Model', 'Series', 'Family'),
            manufacturer=A('Manufacturer', 'Airbus'),
            model=A('Model', None),
            series=A('Series', None),
            family=A('Family', 'A330'),
        ))

        with patch('analysis_engine.multistate_parameters.at') as at:
            at.get_conf_angles.side_effect = KeyError
            # Requires Slat.
            at.get_lever_angles.return_value = {
                'Lever 0': (0, 0, None),
                'Lever 1': (20, 9, None),
                'Lever 2': (20, 20, None),
                'Lever 3': (20, 40, None),
            }
            self.assertFalse(self.node_class.can_operate(
                ('Flap', 'Manufacturer', 'Model', 'Series', 'Family'),
                manufacturer=A('Manufacturer', 'Bombardier'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertTrue(self.node_class.can_operate(
                ('Flap', 'Slat', 'Manufacturer', 'Model', 'Series', 'Family'),
                manufacturer=A('Manufacturer', 'Bombardier'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            # Requires Flaperon.
            at.get_lever_angles.return_value = {
                'Lever 0': (None, 0, 1),
                'Lever 1': (None, 9, 2),
                'Lever 2': (None, 20, 3),
                'Lever 3': (None, 40, 4),
            }
            self.assertFalse(self.node_class.can_operate(
                ('Flap', 'Manufacturer', 'Model', 'Series', 'Family'),
                manufacturer=A('Manufacturer', 'Bombardier'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertTrue(self.node_class.can_operate(
                ('Flap', 'Flaperon', 'Manufacturer', 'Model', 'Series', 'Family'),
                manufacturer=A('Manufacturer', 'Bombardier'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            # Requires Slat and Flaperon.
            at.get_lever_angles.return_value = {
                'Lever 0': (0, 0, 1),
                'Lever 1': (20, 9, 2),
                'Lever 2': (20, 20, 3),
                'Lever 3': (20, 40, 4),
            }
            self.assertFalse(self.node_class.can_operate(
                ('Flap', 'Manufacturer', 'Model', 'Series', 'Family'),
                manufacturer=A('Manufacturer', 'Bombardier'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertFalse(self.node_class.can_operate(
                ('Flap', 'Slat', 'Manufacturer', 'Model', 'Series', 'Family'),
                manufacturer=A('Manufacturer', 'Bombardier'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertFalse(self.node_class.can_operate(
                ('Flap', 'Flaperon', 'Manufacturer', 'Model', 'Series', 'Family'),
                manufacturer=A('Manufacturer', 'Bombardier'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))
            self.assertTrue(self.node_class.can_operate(
                ('Flap', 'Slat', 'Flaperon', 'Manufacturer', 'Model', 'Series', 'Family'),
                manufacturer=A('Manufacturer', 'Bombardier'),
                model=A('Model', 'CRJ900 (CL-600-2D24)'),
                series=A('Series', 'CRJ900'),
                family=A('Family', 'CL-600'),
            ))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__crj900(self, at):
        at.get_conf_angles.side_effect = KeyError
        at.get_lever_map.return_value = {f: str(f) for f in (0, 1, 8, 20, 30, 45)}
        at.get_lever_angles.return_value = {
            '0':  (0, 0, None),
            '1':  (20, 0, None),
            '8':  (20, 8, None),
            '20': (20, 20, None),
            '30': (25, 30, None),
            '45': (25, 45, None),
        }
        # Prepare our generated flap and slat arrays:
        flap_array = [0.0, 0, 8, 8, 0, 0, 0, 8, 20, 30, 45, 8, 0, 0, 0]
        slat_array = [0.0, 0, 20, 20, 20, 0, 0, 20, 20, 25, 25, 20, 20, 0, 0]
        flap_array = MappedArray(np.repeat(flap_array, 10),
                values_mapping={f: str(f) for f in (0, 8, 20, 30, 45)})
        slat_array = MappedArray(np.repeat(slat_array, 10),
                values_mapping={s: str(s) for s in (0, 20, 25)})

        ### Add some noise to make our flap and slat angles more realistic:
        ##flap_array += np.ma.sin(range(len(flap_array))) * 0.1
        ##slat_array -= np.ma.sin(range(len(slat_array))) * 0.1

        # Derive the synthetic flap lever:
        flap = M('Flap', flap_array)
        slat = M('Slat', slat_array)
        flaperon = None
        model = A('Model', 'CRJ900 (CL-600-2D24)')
        series = A('Series', 'CRJ900')
        family = A('Family', 'CL-600')
        manufacturer = A('Manufacturer', 'Bombardier')
        node = self.node_class()
        node.derive(flap, slat, flaperon, manufacturer, model, series, family)

        # Check against an expected array of lever detents:
        expected = [0, 0, 8, 8, 1, 0, 0, 8, 20, 30, 45, 8, 1, 0, 0]
        mapping = {x: str(x) for x in sorted(set(expected))}
        array = MappedArray(np.repeat(expected, 10), values_mapping=mapping)
        np.testing.assert_array_equal(node.array, array)

    def test_derive__b737ng(self):
        # Prepare our generated flap array:
        flap_array = [0.0, 0, 5, 2, 1, 0, 0, 10, 15, 25, 30, 40, 0, 0, 0]
        flap_array = MappedArray(np.repeat(flap_array, 10),
                values_mapping={f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)})

        ### Add some noise to make our flap angles more realistic:
        ##flap_array += np.ma.sin(range(len(flap_array))) * 0.05

        # Derive the synthetic flap lever:
        flap = M('Flap', flap_array)
        slat = None
        flaperon = None
        model = A('Model', 'B737-333')
        series = A('Series', 'B737-300')
        family = A('Family', 'B737 Classic')
        manufacturer = A('Manufacturer', 'Boeing')
        node = self.node_class()
        node.derive(flap, slat, flaperon, manufacturer, model, series, family)

        # Check against an expected array of lever detents:
        expected = [0, 0, 5, 2, 1, 0, 0, 10, 15, 25, 30, 40, 0, 0, 0]
        mapping = {x: str(x) for x in sorted(set(expected))}
        array = MappedArray(np.repeat(expected, 10), values_mapping=mapping)
        np.testing.assert_array_equal(node.array, array)

    def test_derive__a330(self):
        # A330 uses Configuration conditions.
        # Prepare our generated flap and slat arrays:
        #                 -  - 1+F   -  -  1   1*  2   2*  3  Full -
        slat_array =     [0, 0, 16,  0, 0, 16, 20, 20, 23, 23, 23, 0]
        flap_array =     [0, 0,  8,  0, 0,  0,  8, 14, 14, 22, 32, 0]
        flaperon_array = [0, 0,  5,  0, 0,  0, 10, 10, 10, 10, 10, 0]
        expected = ['Lever 0', 'Lever 0', 'Lever 1', 'Lever 0', 'Lever 0',
                    'Lever 1', 'Lever 2', 'Lever 2', 'Lever 3', 'Lever 3',
                    'Lever Full', 'Lever 0']
        repeat = 1
        flap_array = MappedArray(np.repeat(flap_array, repeat),
                values_mapping={f: str(f) for f in (0, 8, 14, 22, 32)})
        slat_array = MappedArray(np.repeat(slat_array, repeat),
                values_mapping={s: str(s) for s in (0, 16, 20, 23)})
        flaperon_array = MappedArray(np.repeat(flaperon_array, repeat),
                values_mapping={a: str(a) for a in (0, 5, 10)})

        # Derive the synthetic flap lever:
        flap = M('Flap', flap_array)
        slat = M('Slat', slat_array)
        flaperon = M('Flaperon', flaperon_array)
        model = A('Model', None)
        series = A('Series', None)
        family = A('Family', 'A330')
        manufacturer = A('Manufacturer', 'Airbus')
        node = self.node_class()
        node.derive(flap, slat, flaperon, manufacturer, model, series, family)

        self.assertEqual(list(node.array), list(np.repeat(expected, repeat)))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__e170(self, at):
        # Test applies to E170_EBD_047 only
        at.get_conf_angles.side_effect = KeyError
        at.get_lever_angles.return_value = {
            'Lever 0':    (0, 0, None),
            'Lever 1':    (15, 5, None),
            'Lever 2':    (15, 10, None),
            'Lever 3':    (15, 20, None),
            'Lever 4':    (25, 20, None),
            'Lever 5':    (25, 20, None),
            'Lever Full': (25, 35, None),
        }
        at.get_lever_map.return_value = {
            1:'Lever 0',
            2:'Lever 1',
            4:'Lever 2',
            8:'Lever 3',
            16:'Lever 4',
            32:'Lever 5',
            64:'Lever Full',
        }

        slat_array = [0, 15, 15, 15, 25, 25, 25, 25, 25, 25, 25, 25]
        flap_array = [0, 5, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        expected = ['Lever 0', 'Lever 1', 'Lever 2', 'Lever 3', 'Lever 4',
                    'Lever 4', 'Lever 4', 'Lever 5', 'Lever 5', 'Lever 5',
                    'Lever 4', 'Lever 4']
        repeat = 1
        flap_array = MappedArray(np.repeat(flap_array, repeat),
                values_mapping={f: str(f) for f in (0, 5, 10, 20, 35)})
        slat_array = MappedArray(np.repeat(slat_array, repeat),
                values_mapping={s: str(s) for s in (0, 15, 25)})

        # Derive the synthetic flap lever:
        flap = M('Flap', flap_array)
        slat = M('Slat', slat_array)

        model = A('Model', None)
        series = A('Series', None)
        family = A('Family', 'ERJ-170/175')
        manufacturer = A('Manufacturer', 'Embraer')
        frame = A('Frame', 'E170_EBD_047')

        approach = buildsections('Approach And Landing', (7,10))
        node = self.node_class()
        node.derive(flap, slat, None, manufacturer, model, series, family, approach, frame)
        self.assertEqual(list(node.array), list(np.repeat(expected, repeat)))


class TestFlaperon(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(Flaperon.can_operate(
            ('Aileron (L)', 'Aileron (R)', 'Model', 'Series', 'Family'),
            model=Attribute('Model', 'A330-222'),
            series=Attribute('Series', 'A330-200'),
            family=Attribute('Family', 'A330')))

    def test_derive(self):
        al = load(os.path.join(test_data_path, 'aileron_left.nod'))
        ar = load(os.path.join(test_data_path, 'aileron_right.nod'))
        al.frequency = 2
        ar.frequency = 2
        model = A('Model', 'A330-222')
        series = A('Series', 'A330-200')
        family = A('Family', 'A330')
        flaperon = Flaperon()
        flaperon.derive(al, ar, model, series, family)
        self.assertTrue(
            flaperon.array.raw.tolist()[53:-4] ==
            [10.0] * 11 +
            [5.0] * 244 +
            [10.0] * 584 +
            [5.0] * 21 +
            [0.0] * 21775 +
            [5.0] * 11 +
            [10.0] * 310 +
            [0.0] * 278 +
            [5.0] +
            [10.0] * 260
        )

    def test_derive_with_rapid_aileron_cycling(self):
        al = load(os.path.join(test_data_path, 'A330_Rapid_Aileron_L_Cycle.nod'))
        ar = load(os.path.join(test_data_path, 'A330_Rapid_Aileron_R_Cycle.nod'))
        al.frequency = 1
        ar.frequency = 1
        model = A('Model', 'A330-222')
        series = A('Series', 'A330-200')
        family = A('Family', 'A330')
        flaperon = Flaperon()
        flaperon.derive(al, ar, model, series, family)
        self.assertTrue(
            flaperon.array.raw.tolist()[9:-12] == [10]*2+[0.0]*501+[5.0]*687+\
            [0.0]*18340+[5.0]*10+[10.0]*364+[0.0]*378+[10]*241
        )


class TestFuelQtyLow(unittest.TestCase):
    def test_can_operate(self):
        opts = FuelQty_Low.get_operational_combinations()
        self.assertIn(('Fuel Qty Low',), opts)
        self.assertIn(('Fuel Qty (L) Low',), opts)
        self.assertIn(('Fuel Qty (R) Low',), opts)
        self.assertIn(('Fuel Qty (L) Low', 'Fuel Qty (R) Low'), opts)

    def test_derive_fuel_qty_low_warning(self):
        low = M(array=np.ma.array([0,0,0,1,1,0]), values_mapping={1: 'Warning'})
        warn = FuelQty_Low()
        warn.derive(low, None, None)
        self.assertEqual(warn.array.sum(), 2)

    def test_derive_fuel_qty_low_warning_two_params(self):
        one = M(array=np.ma.array([0,0,0,1,1,0]), values_mapping={1: 'Warning'})
        two = M(array=np.ma.array([0,0,1,1,0,0]), values_mapping={1: 'Warning'})
        warn = FuelQty_Low()
        warn.derive(None, one, two)
        self.assertEqual(warn.array.sum(), 3)


class TestGearOnGround(unittest.TestCase):

    def setUp(self):
        self.node_class = GearOnGround

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        expected =[
            ('Gear (L) On Ground',),
            ('Gear (R) On Ground',),
            ('Gear (L) On Ground', 'Gear (R) On Ground'),
            ]
        for combination in expected:
            self.assertTrue(combination in opts)

    def test_gear_on_ground_basic(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground',
                   frequency=1,
                   offset=0.1)
        p_right = M(array=np.ma.array(data=[0,1,1,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground',
                    frequency=1,
                    offset=0.6)
        wow=GearOnGround()
        wow.derive(p_left, p_right)
        np.testing.assert_array_equal(wow.array, [0,0,0,1,1,1,1,1])
        self.assertEqual(wow.frequency, 2.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_common_word(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground',
                   frequency=1,
                   offset=0.1)
        p_right = M(array=np.ma.array(data=[0,1,1,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground',
                    frequency=1,
                    offset=0.1)
        wow=GearOnGround()
        wow.derive(p_left, p_right)
        np.testing.assert_array_equal(wow.array, [0,1,1,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_left_only(self):
        p_left = M(array=np.ma.array(data=[0,0,1,1]),
                   values_mapping={0:'Air',1:'Ground'},
                   name='Gear (L) On Ground',
                   frequency=1,
                   offset=0.1)
        wow=GearOnGround()
        wow.derive(p_left, None)
        np.testing.assert_array_equal(wow.array, [0,0,1,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.1)

    def test_gear_on_ground_right_only(self):
        p_right = M(array=np.ma.array(data=[0,0,0,1]),
                    values_mapping={0:'Air',1:'Ground'},
                    name='Gear (R) On Ground',
                    frequency=1,
                    offset=0.7)
        wow=GearOnGround()
        wow.derive(None, p_right)
        np.testing.assert_array_equal(wow.array, [0,0,0,1])
        self.assertEqual(wow.frequency, 1.0)
        self.assertAlmostEqual(wow.offset, 0.7)


class TestGear_RedWarning(unittest.TestCase):

    def test_can_operate(self):
        opts = Gear_RedWarning.get_operational_combinations()
        self.assertEqual(len(opts), 7)
        self.assertIn(('Gear (L) Red Warning', 'Airborne'), opts)
        self.assertIn(('Gear (L) Red Warning',
                       'Gear (N) Red Warning',
                       'Gear (R) Red Warning',
                       'Airborne'), opts)

    def test_derive(self):
        gear_warn_l = M('Gear (L) Red Warning',
                        np.ma.array([0,0,0,1,0,0,0,0,0,1,0,0]),
                        values_mapping={1:'Warning', 0:'-'})
        gear_warn_l.array[0] = np.ma.masked
        gear_warn_n = M('Gear (N) Red Warning',
                        np.ma.array([0,1,0,0,1,0,0,0,1,0,0,0]),
                        values_mapping={1:'Warning', 0:'-'})
        gear_warn_r = M('Gear (R) Red Warning',
                        np.ma.array([0,0,0,0,0,1,0,1,0,0,0,0]),
                        values_mapping={1:'Warning', 0:'-'})
        airs = S(items=[Section('Airborne', slice(2, 11), 2, 10)])
        gear_warn = Gear_RedWarning()
        gear_warn.derive(gear_warn_l, gear_warn_n, gear_warn_r, airs)
        self.assertEqual(list(gear_warn.array),
                         ['-', '-', '-', 'Warning', 'Warning', 'Warning',
                          '-', 'Warning', 'Warning', 'Warning', '-', '-'])

class TestKeyVHFCapt(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(KeyVHFCapt.get_operational_combinations(),
                         [('Key VHF (1) (Capt)',),
                          ('Key VHF (2) (Capt)',),
                          ('Key VHF (3) (Capt)',),
                          ('Key VHF (1) (Capt)', 'Key VHF (2) (Capt)'),
                          ('Key VHF (1) (Capt)', 'Key VHF (3) (Capt)'),
                          ('Key VHF (2) (Capt)', 'Key VHF (3) (Capt)'),
                          ('Key VHF (1) (Capt)', 'Key VHF (2) (Capt)', 'Key VHF (3) (Capt)')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestKeyVHFFO(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(KeyVHFFO.get_operational_combinations(),
                         [('Key VHF (1) (FO)',),
                          ('Key VHF (2) (FO)',),
                          ('Key VHF (3) (FO)',),
                          ('Key VHF (1) (FO)', 'Key VHF (2) (FO)'),
                          ('Key VHF (1) (FO)', 'Key VHF (3) (FO)'),
                          ('Key VHF (2) (FO)', 'Key VHF (3) (FO)'),
                          ('Key VHF (1) (FO)', 'Key VHF (2) (FO)', 'Key VHF (3) (FO)')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestMasterCaution(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterCaution
        self.operational_combinations = [
            ('Master Caution (Capt)',),
            ('Master Caution (FO)',),
            ('Master Caution (Capt)', 'Master Caution (FO)'),
            ('Master Caution (Capt) (2)',),
            ('Master Caution (FO) (2)',),
            ('Master Caution (Capt) (2)', 'Master Caution (FO) (2)'),
        ]

    def test_derive(self):
        warn_capt = M(
            name='Master Caution (Capt)',
            array=np.ma.array(data=[0, 1, 0, 0, 0, 0]),
            values_mapping={0: '-', 1: 'Caution'},
            frequency=1,
            offset=0.1,
        )
        warn_fo = M(
            name='Master Caution (FO)',
            array=np.ma.array(data=[0, 0, 1, 0, 0, 0]),
            values_mapping={0: '-', 1: 'Caution'},
            frequency=1,
            offset=0.1,
        )
        warn_capt_2 = M(
            name='Master Caution (Capt) (2)',
            array=np.ma.array(data=[0, 0, 0, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Caution'},
            frequency=1,
            offset=0.1,
        )
        warn_fo_2 = M(
            name='Master Caution (FO) (2)',
            array=np.ma.array(data=[0, 0, 0, 0, 1, 0]),
            values_mapping={0: '-', 1: 'Caution'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(warn_capt, warn_fo, warn_capt_2, warn_fo_2)
        np.testing.assert_array_equal(node.array, [0, 1, 1, 1, 1, 0])



class TestMasterWarning(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MasterWarning
        self.operational_combinations = [
            ('Master Warning (Capt)',),
            ('Master Warning (FO)',),
            ('Master Warning (Capt)', 'Master Warning (FO)'),
        ]

    def test_derive(self):
        warn_capt = M(
            name='Master Warning (Capt)',
            array=np.ma.array(data=[0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Warning'},
            frequency=1,
            offset=0.1,
        )
        warn_fo = M(
            name='Master Warning (FO)',
            array=np.ma.array(data=[0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Warning'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(warn_capt, warn_fo)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


    def test_derive_single(self):
        warn_capt = M(
            name='Master Warning (Capt)',
            array=np.ma.array(data=[0, 0, 0, 0, 0, 0]),
            values_mapping={0: '-', 1: 'Warning'},
            frequency=1,
            offset=0.1,
        )
        warn_fo = M(
            name='Master Warning (FO)',
            array=np.ma.array(data=[0, 0, 1, 1, 1, 0]),
            values_mapping={0: '-', 1: 'Warning'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(warn_capt, warn_fo)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 0])


class TestPackValvesOpen(unittest.TestCase):
    def test_can_operate(self):
        opts = PackValvesOpen.get_operational_combinations()
        self.assertEqual(opts, [
            ('ECS Pack (1) On', 'ECS Pack (2) On'),
            ('ECS Pack (1) On', 'ECS Pack (1) High Flow', 'ECS Pack (2) On'),
            ('ECS Pack (1) On', 'ECS Pack (2) On', 'ECS Pack (2) High Flow'),
            ('ECS Pack (1) On', 'ECS Pack (1) High Flow', 'ECS Pack (2) On', 'ECS Pack (2) High Flow')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPilotFlying(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = PilotFlying
        self.operational_combinations = [
            ('Sidestick Angle (Capt)', 'Sidestick Angle (FO)'),
        ]

    def test_derive(self):
        stick_capt_array = np.ma.concatenate((np.zeros(100),
                                              np.zeros(100) + 20))
        stick_fo_array = np.ma.concatenate((np.zeros(100) + 20,
                                            np.zeros(100)))
        stick_capt = P('Sidestick Angle (Capt)', array=stick_capt_array)
        stick_fo = P('Sidestick Angle (FO)', array=stick_fo_array)
        node = self.node_class()
        node.derive(stick_capt, stick_fo)
        expected_array = MappedArray(np.concatenate((np.ones(100) * 2, np.ones(100))))
        expected = M('Pilot Flying', expected_array,
                     values_mapping=PilotFlying.values_mapping)
        np.testing.assert_array_equal(node.array, expected.array)


class TestPitchAlternateLaw(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchAlternateLaw
        self.operational_combinations = [
            ('Pitch Alternate Law (1)',),
            ('Pitch Alternate Law (2)',),
            ('Pitch Alternate Law (1)', 'Pitch Alternate Law (2)'),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_derive_basic(self):
        alt_law1 = M(
            name='Pitch Alternate Law (1)',
            array=np.ma.array([0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Engaged'},
            frequency=1,
            offset=0.1,
        )
        alt_law2 = M(
            name='Pitch Alternate Law (2)',
            array=np.ma.array([0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Engaged'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(alt_law1, alt_law2)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestPitchDisconnect(unittest.TestCase):

    def setUp(self):
        self.node_class = PitchDisconnect

    def test_derive_basic(self):
        pitch_disc_1 = M(
            name='Pitch Disconnect (1)',
            array=np.ma.array([0, 0, 0, 1, 1, 1]),
            values_mapping={0: '-', 1: 'Disconnect'},
            frequency=1,
            offset=0.1,
        )
        pitch_disc_2 = M(
            name='Pitch Disconnect (2)',
            array=np.ma.array([0, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Disconnect'},
            frequency=1,
            offset=0.1,
        )
        node = self.node_class()
        node.derive(pitch_disc_1, pitch_disc_2)
        np.testing.assert_array_equal(node.array, [0, 0, 1, 1, 1, 1])


class TestSlat(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Slat

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_slat_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'CRJ900 (CL-600-2D24)'),
            series=A('Series', 'CRJ900'),
            family=A('Family', 'CL-600'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))

    @patch('analysis_engine.library.at')
    def test_derive(self, at):
        at.get_slat_map.return_value = {s: str(s) for s in (0, 16, 25)}
        _am = A('Model', None)
        _as = A('Series', 'A300B4')
        _af = A('Family', 'A300')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + list(range(27)) + [27] * 5)
        slat = P(name='Slat Angle', array=array, frequency=2)
        node = self.node_class()
        node.derive(slat, *attributes)
        attributes = (a.value for a in attributes)
        at.get_slat_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_slat_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        self.assertEqual(node.frequency, 4)
        self.assertEqual(node.array.raw.tolist(), [0] * 11 + [25] * 62 + [None])


class TestSlatExcludingTransition(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SlatExcludingTransition

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_slat_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'CRJ900 (CL-600-2D24)'),
            series=A('Series', 'CRJ900'),
            family=A('Family', 'CL-600'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))

    @patch('analysis_engine.library.at')
    def test_derive(self, at):
        at.get_slat_map.return_value = {s: str(s) for s in (0, 16, 25)}
        _am = A('Model', None)
        _as = A('Series', 'A300B4')
        _af = A('Family', 'A300')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + list(range(27)) + [27] * 5)
        slat = P(name='Slat Angle', array=array, frequency=2)
        node = self.node_class()
        node.derive(slat, *attributes)
        attributes = (a.value for a in attributes)
        at.get_slat_map.assert_called_once_with(*attributes)
        self.assertEqual(node.values_mapping, at.get_slat_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        self.assertEqual(node.array.raw.tolist(), [0] * 65 + [25] * 8 + [None])


class TestSlatIncludingTransition(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SlatIncludingTransition

    @patch('analysis_engine.multistate_parameters.at')
    def test_can_operate(self, at):
        at.get_slat_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'CRJ900 (CL-600-2D24)'),
            series=A('Series', 'CRJ900'),
            family=A('Family', 'CL-600'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Slat Angle', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))

    @patch('analysis_engine.library.at')
    def test_derive(self, at):
        at.get_slat_map.return_value = {s: str(s) for s in (0, 16, 20, 23)}
        _am = A('Model', 'A330-343E')
        _as = A('Series', 'A330-300')
        _af = A('Family', 'A330')
        attributes = (_am, _as, _af)
        array = np.ma.array([0] * 5 + list(range(27)) + [27] * 5)
        slat = P(name='Slat Angle', array=array, frequency=2)
        node = self.node_class()
        node.derive(slat, *attributes)
        attributes = (a.value for a in attributes)
        self.assertEqual(node.values_mapping, at.get_slat_map.return_value)
        self.assertEqual(node.units, ut.DEGREE)
        self.assertIsInstance(node.array, MappedArray)
        self.assertEqual(node.frequency, 1)
        self.assertEqual(node.array.raw.tolist(), [0] * 6 + [16] * 16 + [20] * 4 + [23] * 11)


class TestSlatFullyExtended(unittest.TestCase):

    def test_can_operate(self):
        node = self.node_class()
        operational_combinations = node.get_operational_combinations()
        self.assertEqual(len(operational_combinations), 65535) # 2**16-1

    def setUp(self):
        extended_l_array = [ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0]
        extended_r_array = [ 0,  0,  0,  0,  1,  1,  1,  0,  0,  0]
        extended_7_array = [ 0,  0,  0,  0,  1,  1,  1,  0,  0,  0]

        self.extended_l = M(name='Slat (L1) Fully Extended', array=np.ma.array(extended_l_array), values_mapping={1:'Extended'})
        self.extended_r = M(name='Slat (R3) Fully Extended', array=np.ma.array(extended_r_array), values_mapping={1:'Extended'})
        self.extended_7 = M(name='Slat (7) Fully Extended', array=np.ma.array(extended_7_array), values_mapping={1:'Extended'})
        self.node_class = SlatFullyExtended

    def test_derive(self):
        result = [ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0]
        node = self.node_class()
        node.derive(self.extended_l,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.extended_r,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.extended_7,
                    None,)
        np.testing.assert_equal(node.array.data, result)

    def test_derive_masked_value(self):
        self.extended_l.array.mask = [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0]
        self.extended_r.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]
        self.extended_7.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        result_array = [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0]
        result_mask =  [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        node = self.node_class()
        node.derive(self.extended_l,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.extended_r,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.extended_7,
                    None,)
        np.testing.assert_equal(node.array.data, result_array)
        np.testing.assert_equal(node.array.mask, result_mask)


class TestSlatPartExtended(unittest.TestCase):

    def test_can_operate(self):
        node = self.node_class()
        operational_combinations = node.get_operational_combinations()
        self.assertEqual(len(operational_combinations), 65535) # 2**16-1

    def setUp(self):
        extended_l_array = [ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0]
        extended_r_array = [ 0,  0,  0,  0,  1,  1,  1,  0,  0,  0]
        extended_7_array = [ 0,  0,  0,  0,  1,  1,  1,  0,  0,  0]

        self.extended_l = M(name='Slat (L1) Part Extended', array=np.ma.array(extended_l_array), values_mapping={1:'Part Extended'})
        self.extended_r = M(name='Slat (R3) Part Extended', array=np.ma.array(extended_r_array), values_mapping={1:'Part Extended'})
        self.extended_7 = M(name='Slat (7) Part Extended', array=np.ma.array(extended_7_array), values_mapping={1:'Part Extended'})
        self.node_class = SlatPartExtended

    def test_derive(self):
        result = [ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0]
        node = self.node_class()
        node.derive(self.extended_l,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.extended_r,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.extended_7,
                    None,)
        np.testing.assert_equal(node.array.data, result)

    def test_derive_masked_value(self):
        self.extended_l.array.mask = [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0]
        self.extended_r.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]
        self.extended_7.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        result_array = [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0]
        result_mask =  [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        node = self.node_class()
        node.derive(self.extended_l,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.extended_r,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.extended_7,
                    None,)
        np.testing.assert_equal(node.array.data, result_array)
        np.testing.assert_equal(node.array.mask, result_mask)


class TestSlatInTransit(unittest.TestCase):

    def test_can_operate(self):
        node = self.node_class()
        operational_combinations = node.get_operational_combinations()
        self.assertEqual(len(operational_combinations), 65535) # 2**16-1

    def setUp(self):
        transit_l_array = [ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0]
        transit_r_array = [ 0,  0,  0,  0,  1,  1,  1,  0,  0,  0]
        transit_7_array = [ 0,  0,  0,  0,  1,  1,  1,  0,  0,  0]

        self.transit_l = M(name='Slat (L1) In Transit', array=np.ma.array(transit_l_array), values_mapping={1:'In Transit'})
        self.transit_r = M(name='Slat (R3) In Transit', array=np.ma.array(transit_r_array), values_mapping={1:'In Transit'})
        self.transit_7 = M(name='Slat (7) In Transit', array=np.ma.array(transit_r_array), values_mapping={1:'In Transit'})
        self.node_class = SlatInTransit

    def test_derive(self):
        result = [ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0]
        node = self.node_class()
        node.derive(self.transit_l,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.transit_r,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.transit_7,
                    None,)
        np.testing.assert_equal(node.array.data, result)

    def test_derive_masked_value(self):
        self.transit_l.array.mask = [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0]
        self.transit_r.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]
        self.transit_7.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        result_array = [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0]
        result_mask =  [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        node = self.node_class()
        node.derive(self.transit_l,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.transit_r,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.transit_7,
                    None,)
        np.testing.assert_equal(node.array.data, result_array)
        np.testing.assert_equal(node.array.mask, result_mask)


class TestSlatRetracted(unittest.TestCase):

    def test_can_operate(self):
        node = self.node_class()
        operational_combinations = node.get_operational_combinations()
        self.assertEqual(len(operational_combinations), 255) # 2**8-1

    def setUp(self):
        retracted_l_array = [ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0]
        retracted_r_array = [ 0,  0,  0,  0,  1,  1,  1,  0,  0,  0]

        self.retracted_l = M(name='Slat (L1) Retracted', array=np.ma.array(retracted_l_array), values_mapping={1:'Retracted'})
        self.retracted_r = M(name='Slat (R3) Retracted', array=np.ma.array(retracted_r_array), values_mapping={1:'Retracted'})
        self.node_class = SlatRetracted

    def test_derive(self):
        result = [ 0,  0,  0,  0,  0,  1,  1,  0,  0,  0]
        node = self.node_class()
        node.derive(self.retracted_l,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.retracted_r,
                    None,)
        np.testing.assert_equal(node.array.data, result)

    def test_derive_masked_value(self):
        self.retracted_l.array.mask = [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0]
        self.retracted_r.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        result_array = [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0]
        result_mask =  [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        node = self.node_class()
        node.derive(self.retracted_l,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.retracted_r,
                    None,)
        np.testing.assert_equal(node.array.data, result_array)
        np.testing.assert_equal(node.array.mask, result_mask)


class TestSmokeWarning(unittest.TestCase):

    def setUp(self):
        self.node_class = SmokeWarning

    @unittest.skip("Taking too long generating all 2,097,151 combinations")
    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(len(opts), 2**21-1)

    def test_can_operate_simple(self):
        expected_params = [
            'Smoke Avionics Warning',
            'Smoke Avionics (1) Warning',
            'Smoke Avionics (2) Warning',
            'Smoke Lavatory Warning',
            'Smoke Lavatory (1) Warning',
            'Smoke Lavatory (2) Warning',
            'Smoke Cabin Warning',
            'Smoke Cabin Rest (1) Warning',
            'Smoke Cabin Rest (2) Warning',
            'Smoke Cargo Warning',
            'Smoke Cargo Fwd (1) Warning',
            'Smoke Cargo Fwd (2) Warning',
            'Smoke Cargo Aft (1) Warning',
            'Smoke Cargo Aft (2) Warning',
            'Smoke Cargo Rest (1) Warning',
            'Smoke Cargo Rest (2) Warning',
            'Smoke Lower Deck Stowage',
            'Smoke Avionic Bulk',
            'Smoke IFEC',
            'Smoke BCRC',
            'Smoke Autonomous VCC',
        ]
        derived_args = inspect.getfullargspec(self.node_class.derive).defaults
        derived_params = [n.name for n in derived_args]
        self.assertEqual(len(derived_params), len(expected_params))
        self.assertEqual(sorted(derived_params), sorted(expected_params))
        for warning in expected_params:
            self.assertTrue(self.node_class.can_operate([warning]))


    def test_derive(self):
        one = M('Smoke Avionics (1) Warning', np.ma.array([0, 1, 0, 0, 0, 0]),
                offset=0.7, frequency=2.0,
                values_mapping={0: '-', 1: 'Smoke'})
        two = M('Smoke Avionics (2) Warning', np.ma.array([0, 0, 0, 0, 1, 0]),
                offset=0.2, frequency=2.0,
                values_mapping={0: '-', 1: 'Smoke'})
        node = self.node_class()
        node.derive(None, one, two, *[None]*18)
        expected = np.ma.array([0, 1, 0, 0, 1, 0])
        np.testing.assert_equal(node.array.raw, expected)


class TestSpeedbrakeDeployed(unittest.TestCase):

    def test_can_operate(self):
        # get_operational_combinations is too slow for lots of dependencies
        self.assertTrue(self.node_class.can_operate(('Ground Spoiler Deployed',)))
        self.assertTrue(self.node_class.can_operate(('Speedbrake (Tail) Deployed',)))
        self.assertTrue(self.node_class.can_operate(('Ground Spoiler (L) Deployed', 'Ground Spoiler (R) Deployed')))
        self.assertTrue(self.node_class.can_operate(('Spoiler (L) (1) Deployed', 'Spoiler (R) (1) Deployed')))
        self.assertTrue(self.node_class.can_operate(('Ground Spoiler (L) (1) Deployed', 'Ground Spoiler (R) (1) Deployed')))
        self.assertTrue(self.node_class.can_operate(('Ground Spoiler (L) (2) Deployed', 'Ground Spoiler (R) (2) Deployed')))
        self.assertTrue(self.node_class.can_operate(('Spoiler (L) Outboard Deployed', 'Spoiler (R) Outboard Deployed')))
        self.assertTrue(self.node_class.can_operate(('Spoiler',)))
        self.assertTrue(self.node_class.can_operate(('Spoiler (L)', 'Spoiler (R)')))
        self.assertTrue(self.node_class.can_operate(('Spoiler (L) (1)', 'Spoiler (R) (1)')))
        self.assertTrue(self.node_class.can_operate(('Spoiler (L) Outboard', 'Spoiler (R) Outboard')))
        self.assertTrue(self.node_class.can_operate((
            'Spoiler (L) (1) Deployed', 'Spoiler (L) (2) Deployed', 'Spoiler (L) (3) Deployed', 'Spoiler (L) (4) Deployed',
            'Spoiler (L) (5) Deployed', 'Spoiler (L) (6) Deployed', 'Spoiler (L) (7) Deployed',
            'Spoiler (R) (1) Deployed', 'Spoiler (R) (2) Deployed', 'Spoiler (R) (3) Deployed', 'Spoiler (R) (4) Deployed',
            'Spoiler (R) (5) Deployed', 'Spoiler (R) (6) Deployed', 'Spoiler (R) (7) Deployed',
            'Spoiler (L) (1)', 'Spoiler (L) (2)', 'Spoiler (L) (3)', 'Spoiler (L) (4)',
            'Spoiler (L) (5)', 'Spoiler (L) (6)', 'Spoiler (L) (7)',
            'Spoiler (R) (1)', 'Spoiler (R) (2)', 'Spoiler (R) (3)', 'Spoiler (R) (4)',
            'Spoiler (R) (5)', 'Spoiler (R) (6)', 'Spoiler (R) (7)',
        )))

    def setUp(self):
        deployed_l_array = [ 0,  0,  0,  1,  0,  1,  1,  1,  0,  0]
        deployed_r_array = [ 0,  0,  0,  0,  1,  1,  1,  1,  0,  0]

        self.deployed_l = M(name='Spoiler (L) Deployed', array=np.ma.array(deployed_l_array), values_mapping={1:'Deployed'})
        self.deployed_r = M(name='Spoiler (R) Deployed', array=np.ma.array(deployed_r_array), values_mapping={1:'Deployed'})
        self.node_class = SpeedbrakeDeployed

    def test_derive(self):
        result = [ 0,  0,  0,  0,  0,  1,  1,  1,  0,  0]
        node = self.node_class()
        node.derive(None, None, self.deployed_l, self.deployed_r, *[None] * 43)
        np.testing.assert_equal(node.array.data, result)

    def test_derive_masked_value(self):
        self.deployed_l.array.mask = [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0]
        self.deployed_r.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        result_array = [ 0,  0,  0,  0,  0,  0,  1,  1,  0,  0]
        result_mask =  [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        node = self.node_class()
        node.derive(None, None, self.deployed_l, self.deployed_r, *[None] * 43)
        np.testing.assert_equal(node.array.data, result_array)
        np.testing.assert_equal(node.array.mask, result_mask)

    def test_derive_ignore_single_value(self):
        self.deployed_l.array = np.ma.array([ 0,  0,  0,  1,  0,  1,  0,  1,  0,  0])
        self.deployed_r.array = np.ma.array([ 0,  0,  0,  0,  1,  0,  0,  1,  1,  0])

        result = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        node = self.node_class()
        node.derive(None, None, self.deployed_l, self.deployed_r, *[None] * 43)
        np.testing.assert_equal(node.array.data, result)

    def test_derive_spoiler_angle(self):
        spoiler = P('Spoiler', array=[0, 5, 10, 50, 10, 5, 0])
        node = self.node_class()
        node.derive(*[None] * 26 + [spoiler] + [None] * 18)
        self.assertEqual(
            node.array.tolist(),
            ['-', '-', 'Deployed', 'Deployed', 'Deployed', '-', '-'],
        )

    def test_derive_spoiler_pair_angle_1(self):
        spoiler_l = P('Spoiler (L)', array=[0, 5, 10, 50, 10, 5, 0])
        spoiler_r = P('Spoiler (R)', array=[0, 0, 5, 10, 50, 10, 5])
        node = self.node_class()
        node.derive(*[None] * 27 + [spoiler_l, spoiler_r] + [None] * 15)
        self.assertEqual(
            node.array.tolist(),
            ['-', '-', '-', 'Deployed', 'Deployed', '-', '-'],
        )

    def test_derive_spoiler_pair_angle_2(self):
        spoiler_l_1 = P('Spoiler (L) (1)', array=[0, 5, 10, 50, 10, 5, 0])
        spoiler_r_1 = P('Spoiler (R) (1)', array=[0, 0, 5, 10, 50, 10, 5])
        node = self.node_class()
        node.derive(*[None] * 29 + [spoiler_l_1] + [None] * 6 + [spoiler_r_1] + [None] * 8)
        self.assertEqual(
            node.array.tolist(),
            ['-', '-', '-', 'Deployed', 'Deployed', '-', '-'],
        )


class TestSpeedbrakeSelected(unittest.TestCase):

    def test_can_operate(self):
        opts = SpeedbrakeSelected.get_operational_combinations()
        self.assertTrue(('Speedbrake Deployed',) in opts)
        self.assertTrue(('Speedbrake', 'Family') in opts)
        self.assertTrue(('Speedbrake Handle', 'Family') in opts)
        self.assertTrue(('Speedbrake Handle', 'Speedbrake', 'Family') in opts)

    def test_derive(self):
        # test with deployed
        spd_sel = SpeedbrakeSelected()
        spd_sel.derive(
            deployed=M(array=np.ma.array(
                [0, 0, 0, 1, 1, 0]), values_mapping={1:'Deployed'}),
            armed=M(array=np.ma.array(
                [0, 0, 1, 1, 0, 0]), values_mapping={1:'Armed'})
        )
        self.assertEqual(list(spd_sel.array),
            ['Stowed', 'Stowed', 'Armed/Cmd Dn', 'Deployed/Cmd Up', 'Deployed/Cmd Up', 'Stowed'])

    def test_derive_from_handle(self):
        handle_array = np.ma.concatenate([np.arange(0, 2, 0.1),
                                          np.arange(2, 0, -0.1)])
        spd_sel = SpeedbrakeSelected()
        array = spd_sel.derive_from_handle(handle_array)
        self.assertEqual(list(array), # MappedArray .tolist() does not output states.
                         ['Stowed']*10+['Deployed/Cmd Up']*20+['Stowed']*10)

        handle_array = np.ma.concatenate([np.arange(0, 2, 0.1),
                                          np.ones(10) * 13,
                                          np.arange(0.99, 0, -0.1)])
        spd_sel = SpeedbrakeSelected()
        array = spd_sel.derive_from_handle(handle_array, deployed=5, armed=1)
        self.assertEqual(list(array), # MappedArray .tolist() does not output states.
                         ['Stowed']*10+['Armed/Cmd Dn']*10+['Deployed/Cmd Up']*10+['Stowed']*10)

        handle_array = np.ma.concatenate([np.arange(0, 2, 0.1),
                                          np.ones(10) * 13,
                                          np.arange(0.99, 0, -0.1)])
        spd_sel = SpeedbrakeSelected()
        array = spd_sel.derive_from_handle(handle_array, deployed=5)
        self.assertEqual(list(array), # MappedArray .tolist() does not output states.
                         ['Stowed']*20+['Deployed/Cmd Up']*10+['Stowed']*10)

    def test_derive_from_handle_mask_below_armed(self):
        array = np.ma.arange(-5, 15)
        result = SpeedbrakeSelected.derive_from_handle(array, deployed=10,
                                                       armed=0,
                                                       mask_below_armed=True)
        self.assertEqual(list(result),
                         [np.ma.masked] * 5 + ['Stowed'] + 9 * ['Armed/Cmd Dn'] +
                         5 * ['Deployed/Cmd Up'])

    def test_bd100_speedbrake(self):
        handle = load(os.path.join(
            test_data_path, 'SpeedbrakeSelected_SpeedbrakeHandle_2.nod'))
        spoiler_gnd_armed = load(os.path.join(
            test_data_path, 'SpeedbrakeSelected_SpoilerGroundArmed_2.nod'))
        array = SpeedbrakeSelected.bd100_speedbrake(handle.array,
                                                    spoiler_gnd_armed.array)
        self.assertTrue(all(x == 'Armed/Cmd Dn' for x in
                            array[spoiler_gnd_armed.array == 'Armed']))
        self.assertEqual(np.ma.concatenate([np.arange(8802, 8824),
                                            np.arange(11463, 11523),
                                            np.arange(11545, 11575),
                                            np.arange(11840, 12013)]).tolist(),
                         np.ma.where(array == 'Deployed/Cmd Up')[0].tolist())

    def test_b737_speedbrake(self):
        self.maxDiff = None
        spd_sel = SpeedbrakeSelected()
        spdbrk = P(array=np.ma.array([0]*10 + [1.3]*20 + [0.2]*10))
        handle = P(array=np.ma.arange(40))
        # Follow the spdbrk only
        res = spd_sel.b737_speedbrake(spdbrk, None)
        self.assertEqual(list(res),
                        ['Stowed']*10 + ['Deployed/Cmd Up']*20 + ['Stowed']*10)
        # Follow the handle only
        res = spd_sel.b737_speedbrake(None, handle)
        self.assertEqual(list(res),
                        ['Stowed']*3 + ['Armed/Cmd Dn']*12 + ['Deployed/Cmd Up']*25)
        # Follow the combination
        res = spd_sel.b737_speedbrake(spdbrk, handle)
        self.assertEqual(list(res),
                        ['Stowed']*3 + ['Armed/Cmd Dn']*7 + ['Deployed/Cmd Up']*30)

    def test_derive_from_armed_and_speedbrake(self):
        self.maxDiff = None
        spd_sel = SpeedbrakeSelected()
        spdbrk = P(array=np.ma.array([0]*10 + [1.3]*20 + [0.2]*10))
        armed = M(array=np.ma.array(
                [0]*5 + [1]*5 + [0]*30), values_mapping={1:'Armed'})
        # A320 test
        res = spd_sel.derive_from_armed_and_speedbrake(armed, spdbrk)
        self.assertEqual(list(res),
                        ['Stowed']*5 + ['Armed/Cmd Dn']*5 + ['Deployed/Cmd Up']*20 + ['Stowed']*10)

        # MD-11 test
        spdbrk.array = spdbrk.array * 11
        res = spd_sel.derive_from_armed_and_speedbrake(armed, spdbrk, threshold=10)
        self.assertEqual(list(res),
                        ['Stowed']*5 + ['Armed/Cmd Dn']*5 + ['Deployed/Cmd Up']*20 + ['Stowed']*10)


    def test_b787_speedbrake(self):
        handle = load(os.path.join(
            test_data_path, 'SpeedBrakeSelected_SpeedbrakeHandle.nod'))

        result = SpeedbrakeSelected.b787_speedbrake(handle)
        self.assertEqual(len(np.ma.where(result == 0)[0]), 9445)
        self.assertEqual(np.ma.where(result == 1)[0].tolist(),
                         [8189, 8190, 8451, 8524, 8525] + list(range(9098, 9223)))
        self.assertEqual(np.ma.where(result == 2)[0].tolist(),
                         list(range(8191, 8329)) + list(range(8452, 8524)) + list(range(9223, 9262)))


class TestStableApproach(unittest.TestCase):
    def test_can_operate(self):
        opts = StableApproach.get_operational_combinations()
        combinations = [
            # all
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Avg For 10 Sec', 'Altitude AAL', 'Vapp'),
            # exc. Vapp
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Avg For 10 Sec', 'Altitude AAL'),
            # exc. Airspeed Relative
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Avg For 10 Sec', 'Altitude AAL', 'Vapp'),
            # exc. Vapp and Airspeed Relative
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Avg For 10 Sec', 'Altitude AAL'),
            # exc. ILS Glideslope and Vapp
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Localizer', 'Eng (*) N1 Avg For 10 Sec', 'Altitude AAL'),
            # exc. ILS Glideslope and ILS Localizer and Vapp
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'Eng (*) N1 Avg For 10 Sec', 'Altitude AAL'),
            # using EPR and exc. Airspeed Relative
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) EPR Avg For 10 Sec', 'Altitude AAL', 'Vapp'),
            # including Family attribute
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) EPR Avg For 10 Sec', 'Altitude AAL', 'Vapp', 'Family'),
            # exc. Track Deviation
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Eng (*) N1 Avg For 10 Sec', 'Altitude AAL', 'Vapp'),
        ]
        for combo in combinations:
            self.assertIn(combo, opts)

    def test_stable_approach(self):
        stable = StableApproach()
        apps = App()
        apps.create_approach('LANDING', slice(15, 20),
                             gs_est=True, loc_est=True,
                             landing_runway={'localizer':{'is_offset':False}})

        # Arrays will be 20 seconds long, index 4, 13,14,15 are stable
        #0. first and last values are not in approach slice
        phases = S()
        phases.create_section(slice(1,20))
        #1. gear up for index 0-2
        g = [ 0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
        gear = M(array=np.ma.array(g), values_mapping={1:'Down'})
        #2. landing flap invalid index 0, 5
        # Setting 30 is ignored as it's above 1,000ft
        f = [ 5, 15, 30, 15, 15,  0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        flap = P(array=np.ma.array(f))
        #3. trackk deviation stays within limits except for index 11-12, although we weren't on the track sample 15 (masked out)
        h = [20, 20,  2,  3,  4,  8,  0,  0,  0,  0,  2, 20, 20,  8,  2,  0,  1,  1,  1,  1,  1]
        hm= [ 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0]
        track = P(array=np.ma.array(h, mask=hm))
        #4. airspeed relative within limits for periods except 0-3
        a = [50, 50, 50, 45,  9,  8,  3, 7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        aspd = P(array=np.ma.array(a))
        #5. glideslope deviation is out for index 8, index 10-11 ignored as under 200ft, last 4 values ignored due to alt cutoff
        g = [ 6,  6,  6,  6,  0, .5, .5,-.5,1.2,0.9,1.4,1.3,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        gm= [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        glide = P(array=np.ma.array(g, mask=gm))
        #6. localizer deviation is out for index 7, index 10 ignored as just under 200ft, last 4 values ignored due to alt cutoff
        l = [ 0,  0,  0,  0,  0,  0,  0,  2,  0.8, 0.1, -3,  0,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        loc = P(array=np.ma.array(l))
        #7. Vertical Speed too great at index 8, but change is smoothed out and at 17 (59ft)
        v = [-500] * 20
        v[6] = -2000
        v[18:19] = [-2000]*1
        vert_spd = P(array=np.ma.array(v))

        #TODO: engine cycling at index 12?

        #8. Engine power too low at index 5-12
        e = [80, 80, 80, 80, 80, 30, 20, 30, 20, 30, 20, 30, 44, 40, 80, 80, 80, 50, 50, 50, 50]
        eng = P(array=np.ma.array(e))

        # Altitude for cutoff heights, 9th element is 200 below, last 4 values are below 100ft last 2 below 50ft
        al = list(range(2000,219,-200)) + list(range(219,18, -20)) + [0]
        # == [2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 219, 199, 179, 159, 139, 119, 99, 79, 59, 39, 19]
        alt = P(array=np.ma.array(al))
        # DERIVE without using Vapp (using Vref limits)
        stable.derive(apps, phases, gear, flap, track, aspd, None, vert_spd, glide, loc, eng, None, alt, None)
        self.assertEqual(len(stable.array), len(alt.array))
        self.assertEqual(len(stable.array), len(track.array))

        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 6, 5, 8, 8, 3, 3, 9, 9, 9, 9, 9, 9, 9, 0])
        self.assertEqual(list(stable.array.mask),
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        #========== NO GLIDESLOPE ==========
        # Test without the use of Glideslope (not established according to
        # Approach) therefore instability for index 7-10 is now due to low
        # Engine Power
        glide2 = P(array=np.ma.array([3.5]*20))
        apps[0].gs_est = False
        stable.derive(apps, phases, gear, flap, track, aspd, None, vert_spd, glide2, loc, eng, None, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 8, 8, 8, 8, 3, 3, 9, 9, 9, 9, 9, 9, 9, 0])

        #========== VERTICAL SPEED ==========
        # Test with a lot of vertical speed (rather than just gusts above)
        # Note: Glideslope still not established.
        v2 = [-1800] * 20
        vert_spd2 = P(array=np.ma.array(v2))
        stable.derive(apps, phases, gear, flap, track, aspd, None, vert_spd2, glide2, loc, eng, None, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 7, 2, 7, 7, 7, 7, 7, 3, 3, 7, 7, 7, 7, 7, 7, 7, 0])

        #========== UNSTABLE GLIDESLOPE JUST ABOVE 200ft ==========
        # Test that with unstable glideslope just before 200ft, this stability
        # reason is continued to touchdown. Higher level checks (Track Dev at 3)
        # still take priority at indexes 11-12
        #                                        219ft == 1.5 dots
        apps[0].gs_est = True
        g3 = [ 6,  6,  6,  6,  0, .5, .5,-.5,1.2,1.5,1.4,1.3,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        gm = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        glide3 = P(array=np.ma.array(g3, mask=gm))
        stable.derive(apps, phases, gear, flap, track, aspd, None, vert_spd, glide3, loc, eng, None, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 8, 6, 5, 5, 5, 3, 3, 5, 5, 5, 5, 5, 5, 5, 0])


    def test_with_real_data(self):
        apps = App()
        apps.create_approach('LANDING', slice(2800, 3000),
                             gs_est=True, loc_est=True,
                             landing_runway={'localizer':{'is_offset':False}})
        phases = S(items=[Section(name='Approach And Landing',
                                slice=slice(2702, 2993, None),
                                start_edge=2702.0, stop_edge=2993.0)])


        def test_node(name):
            return load(os.path.join(test_data_path, 'Stable Approach - '+name+'.nod'))
        stable = StableApproach()

        gear = test_node('Gear Down')
        flap = test_node('Flap')
        tdev = test_node('Track Deviation From Runway')
        vspd = test_node('Vertical Speed')
        gdev = test_node('ILS Glideslope')
        ldev = test_node('ILS Localizer')
        eng = test_node('Eng (star) N1 Min For 5 Sec')
        alt = test_node('Altitude AAL')

        stable.derive(
            apps=apps,
            phases=phases,
            gear=gear,
            flap=flap,
            tdev=tdev,
            aspd_rel=None,
            aspd_minus_sel=None,
            vspd=vspd,
            gdev=gdev,
            ldev=ldev,
            eng_n1=eng,
            eng_epr=None,
            alt=alt,
            vapp=None)

        self.assertEqual(len(stable.array), len(alt.array))
        analysed = np.ma.clump_unmasked(stable.array)
        self.assertEqual(len(analysed), 1)
        # valid for the approach slice
        self.assertEqual(analysed[0].start, apps[0].slice.start)
        # stop is 10 secs after touchdown
        self.assertEqual(analysed[0].stop, 2946)

        sect = stable.array[analysed[0]]
        # assert that last few values are correct (masked in gear and flap params should not influence end result)
        self.assertEqual(list(sect[-4:]), ['Stable']*4)
        self.assertEqual(list(sect[0:73]), ['Gear Not Down']*73)
        self.assertEqual(list(sect[74:117]), ['Not Landing Flap']*43)
        # 10 samples above 1000ft where Eng N1 was not yet stable
        self.assertEqual(list(sect[117:127]), ['Eng Thrust Not Stable']*10)
        self.assertTrue(np.all(sect[127:] == ['Stable']))


class TestStableApproachExcludingEngThrust(unittest.TestCase):
    def test_can_operate(self):
        opts = StableApproachExcludingEngThrust.get_operational_combinations()
        combinations = [
            # all
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Altitude AAL', 'Vapp'),
            # exc. Vapp
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Altitude AAL'),
            # exc. Airspeed Relative
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Altitude AAL', 'Vapp'),
            # exc. Vapp and Airspeed Relative
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Altitude AAL'),
            # exc. ILS Glideslope and Vapp
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Localizer', 'Altitude AAL'),
            # exc. ILS Glideslope and ILS Localizer and Vapp
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'Altitude AAL'),
            # using EPR and exc. Airspeed Relative
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Track Deviation From Runway', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Altitude AAL', 'Vapp'),
            # exc. Track Deviation From Runway
            ('Approach Information', 'Descent', 'Gear Down', 'Flap', 'Airspeed Relative For 3 Sec', 'Vertical Speed', 'ILS Glideslope', 'ILS Localizer', 'Altitude AAL', 'Vapp'),
        ]
        for combo in combinations:
            self.assertIn(combo, opts)

    def test_stable_approach(self):
        stable = StableApproachExcludingEngThrust()
        apps = App()
        apps.create_approach('LANDING', slice(15, 20),
                             gs_est=True, loc_est=True,
                             landing_runway={'localizer':{'is_offset':False}})

        # Arrays will be 20 seconds long, index 4, 13,14,15 are stable
        #0. first and last values are not in approach slice
        phases = S()
        phases.create_section(slice(1,20))
        #1. gear up for index 0-2
        g = [ 0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
        gear = M(array=np.ma.array(g), values_mapping={1:'Down'})
        #2. landing flap invalid index 0, 5
        # Setting 30 is ignored as it's above 1,000ft
        f = [ 5, 15, 30, 15, 15,  0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        flap = P(array=np.ma.array(f))
        #3. trackk deviation stays within limits except for index 11-12, although we weren't on the track sample 15 (masked out)
        h = [20, 20,  2,  3,  4,  8,  0,  0,  0,  0,  2, 20, 20,  8,  2,  0,  1,  1,  1,  1,  1]
        hm= [ 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0]
        track = P(array=np.ma.array(h, mask=hm))
        #4. airspeed relative within limits for periods except 0-3
        a = [50, 50, 50, 45,  9,  8,  3, 7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        aspd = P(array=np.ma.array(a))
        #5. glideslope deviation is out for index 8, index 10-11 ignored as under 200ft, last 4 values ignored due to alt cutoff
        g = [ 6,  6,  6,  6,  0, .5, .5,-.5,1.2,0.9,1.4,1.3,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        gm= [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        glide = P(array=np.ma.array(g, mask=gm))
        #6. localizer deviation is out for index 7, index 10 ignored as just under 200ft, last 4 values ignored due to alt cutoff
        l = [ 0,  0,  0,  0,  0,  0,  0,  2,  0.8, 0.1, -3,  0,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        loc = P(array=np.ma.array(l))
        #7. Vertical Speed too great at index 8, but change is smoothed out and at 17 (59ft)
        v = [-500] * 20
        v[6] = -2000
        v[18:19] = [-2000]*1
        vert_spd = P(array=np.ma.array(v))

        # Altitude for cutoff heights, 9th element is 200 below, last 4 values are below 100ft last 2 below 50ft
        al = list(range(2000,219,-200)) + list(range(219,18, -20)) + [0]
        # == [2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 219, 199, 179, 159, 139, 119, 99, 79, 59, 39, 19]
        alt = P(array=np.ma.array(al))
        # DERIVE without using Vapp (using Vref limits)
        stable.derive(apps, phases, gear, flap, track, aspd, None, vert_spd, glide, loc, alt, None)
        self.assertEqual(len(stable.array), len(alt.array))
        self.assertEqual(len(stable.array), len(track.array))

        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 9, 6, 5, 9, 9, 3, 3, 9, 9, 9, 9, 9, 9, 9, 0])
        self.assertEqual(list(stable.array.mask),
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        #========== NO GLIDESLOPE ==========
        # Test without the use of Glideslope (not established according to
        # Approach) therefore instability for index 7-10 is now due to low
        # Engine Power
        glide2 = P(array=np.ma.array([3.5]*20))
        apps[0].gs_est = False
        stable.derive(apps, phases, gear, flap, track, aspd, None, vert_spd, glide2, loc, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 9, 9, 9, 9, 9, 3, 3, 9, 9, 9, 9, 9, 9, 9, 0])

        #========== VERTICAL SPEED ==========
        # Test with a lot of vertical speed (rather than just gusts above)
        # Note: Glideslope still not established.
        v2 = [-1800] * 20
        vert_spd2 = P(array=np.ma.array(v2))
        stable.derive(apps, phases, gear, flap, track, aspd, None, vert_spd2, glide2, loc, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 7, 2, 7, 7, 7, 7, 7, 3, 3, 7, 7, 7, 7, 7, 7, 7, 0])

        #========== UNSTABLE GLIDESLOPE JUST ABOVE 200ft ==========
        # Test that with unstable glideslope just before 200ft, this stability
        # reason is continued to touchdown. Higher level checks (Track Dev at 3)
        # still take priority at indexes 11-12
        #                                        219ft == 1.5 dots
        apps[0].gs_est = True
        g3 = [ 6,  6,  6,  6,  0, .5, .5,-.5,1.2,1.5,1.4,1.3,  0,  0,  0,  0,  0, -2, -2, -2, -2]
        gm = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        glide3 = P(array=np.ma.array(g3, mask=gm))
        stable.derive(apps, phases, gear, flap, track, aspd, None, vert_spd, glide3, loc, alt, None)
        self.assertEqual(list(stable.array.data),
        #index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20
               [0, 1, 1, 4, 9, 2, 9, 6, 5, 5, 5, 3, 3, 5, 5, 5, 5, 5, 5, 5, 0])


    def test_with_real_data(self):
        apps = App()
        apps.create_approach('LANDING', slice(2800, 3000),
                             gs_est=True, loc_est=True,
                             landing_runway={'localizer':{'is_offset':False}})
        phases = S(items=[Section(name='Approach And Landing',
                                slice=slice(2702, 2993, None),
                                start_edge=2702.0, stop_edge=2993.0)])


        def test_node(name):
            return load(os.path.join(test_data_path, 'Stable Approach - '+name+'.nod'))
        stable = StableApproachExcludingEngThrust()

        gear = test_node('Gear Down')
        flap = test_node('Flap')
        tdev = test_node('Track Deviation From Runway')
        vspd = test_node('Vertical Speed')
        gdev = test_node('ILS Glideslope')
        ldev = test_node('ILS Localizer')
        alt = test_node('Altitude AAL')

        stable.derive(
            apps=apps,
            phases=phases,
            gear=gear,
            flap=flap,
            tdev=tdev,
            aspd_rel=None,
            aspd_minus_sel=None,
            vspd=vspd,
            gdev=gdev,
            ldev=ldev,
            alt=alt,
            vapp=None)

        self.assertEqual(len(stable.array), len(alt.array))
        analysed = np.ma.clump_unmasked(stable.array)
        self.assertEqual(len(analysed), 1)
        # valid for the approach slice
        self.assertEqual(analysed[0].start, apps[0].slice.start)
        # stop is 10 secs after touchdown
        self.assertEqual(analysed[0].stop, 2946)

        sect = stable.array[analysed[0]]
        # assert that last few values are correct (masked in gear and flap params should not influence end result)
        self.assertEqual(list(sect[-4:]), ['Stable']*4)
        self.assertEqual(list(sect[0:73]), ['Gear Not Down']*73)
        self.assertEqual(list(sect[74:117]), ['Not Landing Flap']*43)
        self.assertTrue(np.all(sect[117:] == ['Stable']))


class TestStallWarning(unittest.TestCase):

    def test_can_operate(self):
        opts = StallWarning.get_operational_combinations()
        self.assertEqual(len(opts), 6)

    def test_derive(self):
        one = M('Stall Warning (1)', np.ma.array([0, 1, 0, 0, 0, 0]),
                offset=0.7, frequency=2.0,
                values_mapping={0: '-', 1: 'Warning'})
        two = M('Stall Warning (2)', np.ma.array([0, 0, 0, 0, 1, 0]),
                offset=0.2, frequency=2.0,
                values_mapping={0: '-', 1: 'Warning'})
        ss = StallWarning()
        ss.derive(one, two)
        expected = np.ma.array([0, 1, 0, 0, 1, 0])
        np.testing.assert_equal(ss.array.raw, expected)


class TestStickShaker(unittest.TestCase):

    def test_can_operate(self):
        opts = StickShaker.get_operational_combinations()
        self.assertEqual(len(opts), 126)

    def test_derive(self):
        left = M('Stick Shaker (L)', np.ma.array([0, 1, 0, 0, 0, 0]),
                 offset=0.7, frequency=2.0,
                 values_mapping={0: '-', 1: 'Shake'})
        right = M('Stick Shaker (R)', np.ma.array([0, 0, 0, 0, 1, 0]),
                  offset=0.2, frequency=2.0,
                  values_mapping={0: '-', 1: 'Shake'})
        ss = StickShaker()
        ss.derive(left, right, None, None, None, None)
        expected = np.ma.array([0, 1, 0, 0, 1, 0])
        np.testing.assert_equal(ss.array.raw, expected)

    def test_single_source(self):
        left=M('Stick Shaker (L)',np.ma.array([0,1,0,0,0,0]),
               offset=0.7, frequency=2.0,
               values_mapping = {0: '-',1: 'Shake',})
        ss=StickShaker()
        ss.derive(left, None, None, None, None, None)
        expected = np.ma.array([0,1,0,0,0,0])
        np.testing.assert_equal(ss.array, expected)

    def test_not_777(self):
        left=M('Stick Shaker (L)',np.ma.array([0,1,0,0,0,0]),
                       offset=0.7, frequency=2.0,
                       values_mapping = {0: '-',1: 'Shake',})
        ss=StickShaker()
        self.assertRaises(ValueError, ss.derive,
                          left, None, None, None, None, None,
                          A('Frame', 'B777'))

class TestStickPusher(unittest.TestCase):

    def test_can_operate(self):
        opts = StickPusher.get_operational_combinations()
        self.assertEqual(len(opts), 3)

    def test_derive(self):
        left = M('Stick Pusher (L)', np.ma.array([0, 1, 0, 0, 0, 0]),
                 offset=0.7, frequency=2.0,
                 values_mapping={0: '-', 1: 'Push'})
        right = M('Stick Pusher (R)', np.ma.array([0, 0, 0, 0, 1, 0]),
                  offset=0.2, frequency=2.0,
                  values_mapping={0: '-', 1: 'Push'})
        sp = StickPusher()
        sp.derive(left, right)
        expected = np.ma.array([0, 1, 0, 0, 1, 0,])
        np.testing.assert_equal(sp.array, expected)

    def test_single_source(self):
        left = M('Stick Pusher (L)',np.ma.array([0,1,0,0,0,0]),
                 offset=0.7, frequency=2.0,
                 values_mapping = {0: '-',1: 'Push',})
        sp = StickPusher()
        sp.derive(None, left) # Just for variety
        expected = np.ma.array([0,1,0,0,0,0])
        np.testing.assert_equal(sp.array, expected)


class TestThrustReversers(unittest.TestCase):

    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    def setUp(self):
        eng_1_unlocked_array = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0]
        eng_1_deployed_array = [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0]
        eng_2_unlocked_array = [ 1,  1,  1,  1,  0,  1,  0,  0,  0,  0]
        eng_2_deployed_array = [ 1,  1,  1,  1,  1,  0,  0,  0,  0,  0]

        self.eng_1_unlocked = M(name='Eng (1) Thrust Reverser Unlocked', array=np.ma.array(eng_1_unlocked_array), values_mapping={1:'Unlocked'})
        self.eng_1_deployed = M(name='Eng (1) Thrust Reverser Deployed', array=np.ma.array(eng_1_deployed_array), values_mapping={1:'Deployed'})
        self.eng_2_unlocked = M(name='Eng (2) Thrust Reverser Unlocked', array=np.ma.array(eng_2_unlocked_array), values_mapping={1:'Unlocked'})
        self.eng_2_deployed = M(name='Eng (2) Thrust Reverser Deployed', array=np.ma.array(eng_2_deployed_array), values_mapping={1:'Deployed'})
        self.thrust_reversers = ThrustReversers()

    def test_derive(self):
        result = [ 2,  2,  2,  2,  1,  1,  0,  0,  0,  0]
        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                None,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 21)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)

    def test_derive_masked_value(self):
        self.eng_1_unlocked.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]
        self.eng_1_deployed.array.mask = [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0]
        self.eng_2_unlocked.array.mask = [ 0,  0,  0,  1,  0,  0,  0,  1,  1,  0]
        self.eng_2_deployed.array.mask = [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        result_array = [ 2,  2,  2,  2,  1,  1,  0,  0,  0,  0]
        result_mask =  [ 0,  0,  0,  0,  0,  1,  0,  0,  1,  0]

        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                None,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 21)
        np.testing.assert_equal(self.thrust_reversers.array.data, result_array)
        np.testing.assert_equal(self.thrust_reversers.array.mask, result_mask)

    def test_derive_in_transit_avaliable(self):
        result = [ 2,  2,  1,  1,  1,  1,  0,  0,  0,  0]
        transit_array = [ 0,  0,  1,  1,  1,  1,  0,  0,  0,  0]
        eng_1_in_transit = M(name='Eng (1) Thrust Reverser In Transit', array=np.ma.array(transit_array), values_mapping={1:'In Transit'})
        self.thrust_reversers.get_derived([self.eng_1_deployed,
                                None,
                                None,
                                self.eng_1_unlocked,
                                None,
                                None,
                                eng_1_in_transit,
                                self.eng_2_deployed,
                                None,
                                None,
                                self.eng_2_unlocked] + [None] * 21)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)

    def test_derive_unlock_at_edges(self):
        '''
        test for aircraft which only record Thrust Reverser Unlocked during
        transition, not whilst deployed
        '''
        result =               [ 0, 1, 1, 1, 2, 2, 1, 1, 0, 0]

        eng_1_unlocked_array = [ 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
        eng_1_deployed_array = [ 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        eng_2_unlocked_array = [ 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        eng_2_deployed_array = [ 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]

        eng_1_unlocked = M(name='Eng (1) Thrust Reverser Unlocked', array=np.ma.array(eng_1_unlocked_array), values_mapping={1:'Unlocked'})
        eng_1_deployed = M(name='Eng (1) Thrust Reverser Deployed', array=np.ma.array(eng_1_deployed_array), values_mapping={1:'Deployed'})
        eng_2_unlocked = M(name='Eng (2) Thrust Reverser Unlocked', array=np.ma.array(eng_2_unlocked_array), values_mapping={1:'Unlocked'})
        eng_2_deployed = M(name='Eng (2) Thrust Reverser Deployed', array=np.ma.array(eng_2_deployed_array), values_mapping={1:'Deployed'})

        self.thrust_reversers.get_derived([eng_1_deployed,
                                None,
                                None,
                                eng_1_unlocked,
                                None,
                                None,
                                None,
                                eng_2_deployed,
                                None,
                                None,
                                eng_2_unlocked] + [None] * 21)
        np.testing.assert_equal(self.thrust_reversers.array.data, result)


class TestTakeoffConfigurationWarning(unittest.TestCase):

    def test_can_operate(self):
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Stabilizer Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Parking Brake Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Flap Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Gear Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration AP Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Aileron Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Rudder Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Spoiler Warning',)))
        self.assertTrue(TakeoffConfigurationWarning.can_operate(
            ('Takeoff Configuration Stabilizer Warning',
             'Takeoff Configuration Parking Brake Warning',
             'Takeoff Configuration Flap Warning',
             'Takeoff Configuration Gear Warning',
             'Takeoff Configuration Rudder Warning',
             'Takeoff Configuration Spoiler Warning',)))

    @unittest.skip('Test Not Implemented')
    def test_derive_basic(self):
        pass


class TestTAWSAlert(unittest.TestCase):
    def test_can_operate(self):
        parameters = ['TAWS Caution Terrain',
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
                       ]
        for p in parameters:
            self.assertTrue(TAWSAlert.can_operate([p]))

    def setUp(self):
        terrain_array = [1,1,0,1,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0]
        pull_up_array = [0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0]

        self.airs = S(name='Airborne')
        self.airs.create_section(slice(5,15))
        self.terrain = M(name='TAWS Terrain', array=np.ma.array(terrain_array), values_mapping={1:'Warning'})
        self.pull_up = M(name='TAWS Pull Up', array=np.ma.array(pull_up_array), values_mapping={1:'Warning'})
        self.taws_alert = TAWSAlert()

    def test_derive(self):
        result = [0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0]

        self.taws_alert.get_derived((self.airs,
                                None,
                                None,
                                None,
                                None,
                                None,
                                self.pull_up,
                                None,
                                None,
                                None,
                                None,
                                self.terrain,
                                None,
                                None,
                                None,
                                None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)

    def test_derive_masked_values(self):
        result = [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0]
        self.terrain.array[8] = np.ma.masked
        self.terrain.array[10] = np.ma.masked

        self.taws_alert.get_derived((self.airs,
                                None,
                                None,
                                None,
                                None,
                                None,
                                self.pull_up,
                                None,
                                None,
                                None,
                                None,
                                self.terrain,
                                None,
                                None,
                                None,
                                None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)

    def test_derive_zeros(self):
        result = [0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0]

        terrain_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        caution = M(name='TAWS Caution Terrain', array=np.ma.array(terrain_array), values_mapping={1:'Warning'})
        caution.array.mask = True

        self.taws_alert.get_derived((self.airs,
                                     caution,
                                     None,
                                     None,
                                     None,
                                     None,
                                     self.pull_up,
                                     None,
                                     None,
                                     None,
                                     None,
                                     self.terrain,
                                     None,
                                     None,
                                     None,
                                     None,))
        np.testing.assert_equal(self.taws_alert.array.data, result)


class TestTAWSDontSink(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(TAWSDontSink.get_operational_combinations(),
                         [('TAWS (L) Dont Sink',),
                          ('TAWS (R) Dont Sink',),
                          ('TAWS (L) Dont Sink', 'TAWS (R) Dont Sink')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestTAWSGlideslopeCancel(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(TAWSGlideslopeCancel.get_operational_combinations(),
                         [('TAWS (L) Glideslope Cancel',),
                          ('TAWS (R) Glideslope Cancel',),
                          ('TAWS (L) Glideslope Cancel', 'TAWS (R) Glideslope Cancel')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestTAWSTooLowGear(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(TAWSTooLowGear.get_operational_combinations(),
                         [('TAWS (L) Too Low Gear',),
                          ('TAWS (R) Too Low Gear',),
                          ('TAWS (L) Too Low Gear', 'TAWS (R) Too Low Gear')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestTAWSTerrainCaution(unittest.TestCase):
    def setUp(self):
        self.node_class = TAWSTerrainCaution

    def test_name(self):
        self.assertEqual(self.node_class.name, 'TAWS Terrain Caution')


    def test_derive(self):
        terrain_cpt = M(
            name='TAWS Terrain Caution (Capt)',
            array=np.ma.array([0, 0, 1, 1, 0, 0, 0]),
            values_mapping={0: '-', 1: 'Caution'},
        )

        terrain_fo = M(
            name='TAWS Terrain Caution (FO)',
            array=np.ma.array([0, 1, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Caution'},
        )
        expected = np.ma.array([0, 1, 1, 1, 1, 0, 0])
        node = self.node_class()
        node.derive(terrain_cpt, terrain_fo)

        assert_array_equal(expected, node.array)


class TestTAWSTerrainWarning(unittest.TestCase):
    def setUp(self):
        self.node_class = TAWSTerrainWarning

    def test_name(self):
        self.assertEqual(self.node_class.name, 'TAWS Terrain Warning')

    def test_derive(self):
        terrain_cpt = M(
            name='TAWS Terrain Warning (Capt)',
            array=np.ma.array([0, 0, 1, 1, 0, 0, 0]),
            values_mapping={0: '-', 1: 'Warning'},
        )

        terrain_fo = M(
            name='TAWS Terrain Warning (FO)',
            array=np.ma.array([0, 1, 0, 1, 1, 0, 0]),
            values_mapping={0: '-', 1: 'Warning'},
        )
        expected = np.ma.array([0, 1, 1, 1, 1, 0, 0])
        node = self.node_class()
        node.derive(terrain_cpt, terrain_fo)

        assert_array_equal(expected, node.array)


class TestTCASFailure(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(TCASFailure.get_operational_combinations(),
                         [('TCAS (L) Failure',),
                          ('TCAS (R) Failure',),
                          ('TCAS (L) Failure', 'TCAS (R) Failure')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestTCASRA(unittest.TestCase):
    def setUp(self):
        self.node_class = TCASRA

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(),
                         [('TCAS RA (1)',),
                          ('TCAS RA (2)',),
                          ('TCAS RA (1)', 'TCAS RA (2)')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        ra_1 = M(
            name='TCAS RA (1)',
            array=np.ma.array((0, 0, 1, 1, 0, 0)),
            values_mapping={0: '-', 1: 'RA'},
        )
        ra_2 = M(
            name='TCAS RA (2)',
            array=np.ma.array((0, 1, 1, 0, 1, 0)),
            values_mapping={0: '-', 1: 'RA'},
        )
        node = self.node_class()
        node.derive(ra_1, ra_2)
        expected = M(
            name='TCAS RA',
            array=np.ma.array((0, 1, 1, 1, 1, 0)),
            values_mapping={0: '-', 1: 'RA'},
        )
        assert_array_equal(node.array, expected.array)


class TestSpeedControl(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SpeedControl
        self.operational_combinations = [
            ('Speed Control Auto',),
            ('Speed Control Manual',),
            ('Speed Control Auto', 'Speed Control Manual'),
            ('Speed Control (1) Auto',),
            ('Speed Control (1) Manual',),
            ('Speed Control (1) Auto', 'Speed Control (1) Manual'),
            ('Speed Control (2) Auto',),
            ('Speed Control (2) Manual',),
            ('Speed Control (2) Auto', 'Speed Control (2) Manual'),
        ]

    def test_derive(self):
        sc0a = M(
            name='Speed Control Auto',
            array=np.ma.array((0, 0, 1, 1, 0, 0)),
            values_mapping={0: 'Manual', 1: 'Auto'},
        )
        sc0m = M(
            name='Speed Control Manual',
            array=np.ma.array((0, 1, 1, 1, 1, 0)),
            values_mapping={0: 'Auto', 1: 'Manual'},
        )
        node = self.node_class()
        node.derive(sc0a, sc0m, None, None, None, None)
        expected = M(
            name='Speed Control',
            array=np.ma.array((1, 0, 1, 1, 0, 1)),
            values_mapping={0: 'Manual', 1: 'Auto'},
        )
        assert_array_equal(node.array, expected.array)



class TestGearUpInTransit(unittest.TestCase):

    #Gear Up In Transit
    # combine individal (L/R/C/N) params
    # Gear Up Selected changed to Up to Gear Down
    # Gear Down nolonger Down to Gear Up changing to Up
    # Gear Up Selected changed to Up + following Gear In Transit
    # Gear Up Selected changed to Up + transition time
    # Gear Down changed to Up + transition time
    # Gear Up changed to Up - transition time
    # Gear Down changed to Up + following Gear In Transit
    # Gear Up changed to Up - following Gear In Transit

    def setUp(self):
        self.node_class = GearUpInTransit

        self.values_mapping = {
            0: '-',
            1: 'Retracting' # 'Extending' for Down In Transit
        }
        self.expected = M('Gear Up In Transit', array=np.ma.array([0]*5 + [1]*10 + [0]*45),
                          values_mapping=self.values_mapping)
        # Add tests for short transitions where neither Gear Down nor Gear Up
        # show a change. Can only be picked up if we record one of the following
        # parameters: Gear Position, Gear In Transit, Gear Red or
        # Gear Up Selected. See GearDownInTransit test for more info.
        # Gear Red Warnings unfortunately triggers spuriously on some dataframes.
        # So we cannot detect this edge case with Red Warnings.
        self.expected_short = M('Gear Up In Transit', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*10 + [0]*35),
                                values_mapping=self.values_mapping)
        self.family = A('Aircraft Family', value='Generic Family')
        self.series = A('Aircraft Series', value='Generic Series')
        self.model = A('Aircraft Model', value='Generic Model')
        self.airborne=buildsection('Airborne', 1, 59)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()

        possible_combinations = (
            ('Gear Up', 'Gear Up Selected'),
            ('Gear Down', 'Gear Up'),
            ('Gear Up Selected', 'Gear In Transit'),
            ('Gear Down', 'Gear In Transit'),
            ('Gear Up', 'Gear In Transit'),
            ('Gear Position',),
        )

        for params in possible_combinations:
            self.assertTrue(params in combinations)

        fallback_params = ('Model', 'Series', 'Family', 'Gear Up Selected', 'Airborne')
        self.assertFalse(self.node_class.can_operate(fallback_params,
                                                     model=self.model, series=self.series, family=self.family))
        self.assertTrue(self.node_class.can_operate(fallback_params,
                                                    model=self.model,
                                                    series=self.series,
                                                    family=A('Family',value='B737 Classic')))

    #@patch('analysis_engine.multistate_parameters.at')
    #def test_derive__combine(self, at):
        #at.get_gear_transition_times.return_value = (15, 15)
        ## combine individal (L/R/C/N) params
        #left = M('Gear (L) Up In Transit', array=np.ma.array([0]*5 + [1]*9 + [0]*46),
                 #values_mapping=self.values_mapping)
        #right = M('Gear (R) Up In Transit', array=np.ma.array([0]*6 + [1]*9 + [0]*45),
                  #values_mapping=self.values_mapping)
        #node = self.node_class()
        #node.derive(left, None, right, None, None, None, None, None, None, None, self.airborne, self.model, self.series, self.family)

        #np.testing.assert_array_equal(node.array, self.expected.array)
        #self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__up_sel_gear_up(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Up Selected to Gear Up
        # M('Gear Up In Transit', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*10 + [0]*35))
        up_sel = M('Gear Up Selected', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        gear_up = M('Gear Up', array=np.ma.array([0]*25 + [1]*20 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        node = self.node_class()
        node.derive(None, gear_up, up_sel, None, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_gear_up(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Down no longer Down to Gear Up changing to Up
        gear_up = M('Gear Up', array=np.ma.array([0]*14 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                      values_mapping={0: 'Up', 1: 'Down'})
        node = self.node_class()
        node.derive(gear_down, gear_up, None, None, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_gear_up_offset(self, at):
        # Gear Up and Gear Down are offset by 0.5 sec. We need to ensure Gear Up In Transit
        # won't leave a gap before Gear Up is Up.
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Down no longer Down to Gear Up changing to Up
        gear_up = M('Gear Up', array=np.ma.array([0]*14 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'}, offset=0.7)
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                      values_mapping={0: 'Up', 1: 'Down'}, offset=0.2)
        node = self.node_class()
        node.get_derived(
            (gear_down, gear_up, None, None, None, None, self.airborne,
             self.model, self.series, self.family)
        )

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

        # Align to Gear Up now
        gear_up_in_transit = node.get_aligned(gear_up)
        # Make sure there are no gaps
        self.assertEqual(gear_up_in_transit.array[13], 'Retracting')


    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__up_sel_gear_in_transit(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Up Selected Up + Gear In Transit
        up_sel = M('Gear Up Selected', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        in_trans = M('Gear In Transit', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*10 + [0]*20 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'In Transit'})
        node = self.node_class()
        node.derive(None, None, up_sel, in_trans, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__up_sel_spurious_gear_in_transit(self, at):
        '''
        Edge case: spurious Gear In Transit producing a short Gear Up Selected
        End of Gear In Transit coincides with Gear Up Selected going Up again
        '''
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Up Selected Up + Gear In Transit
        up_sel = M(
            'Gear Up Selected',
            array=np.ma.array([0]*5 + [1]*10 + [0]*2 + [1]*13 + [0]*15),
            values_mapping={0: 'Down', 1: 'Up'}
        )
        in_trans = M(
            'Gear In Transit',
            array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*2 + [0]*13 + [1]*10 + [0]*5),
            values_mapping={0: '-', 1: 'In Transit'}
        )
        node = self.node_class()
        node.derive(None, None, up_sel, in_trans, None, None, self.airborne, self.model, self.series, self.family)

        expected = M(
            'Gear Up In Transit',
            array=np.ma.array([0]*5 + [1]*3 + [0]*22 + [0]*15),
            values_mapping=self.values_mapping
        )
        np.testing.assert_array_equal(node.array, expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__up_sel_gear_in_transit_prior_airborne(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Up Selected Up + Gear In Transit
        up_sel = M('Gear Up Selected', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        in_trans = M('Gear In Transit', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*10 + [0]*20 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'In Transit'})
        airborne = buildsection('Airborne', 5, 59)
        node = self.node_class()

        node.derive(None, None, up_sel, in_trans, None, None, airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__up_sel_transition_time(self, at):
        at.get_gear_transition_times.return_value = (10, 10)  # patch transition time to be 10 seconds
        # Gear Up Selected changed to Up + transition time
        up_sel = M('Gear Up Selected', array=np.ma.array([0]*5 + [1]*40 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        node = self.node_class()
        node.derive(None, None, up_sel, None, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_transition_time(self, at):
        at.get_gear_transition_times.return_value = (10, 10)  # patch transition time to be 10 seconds
        # Gear Down changed to Up + transition time
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                   values_mapping={0: 'Up', 1: 'Down'})
        node = self.node_class()
        node.derive(gear_down, None, None, None, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_up_transition_time(self, at):
        at.get_gear_transition_times.return_value = (10, 10)  # patch transition time to be 10 seconds
        # Gear Up changed to Up - transition time
        gear_up = M('Gear Up', array=np.ma.array([0]*15 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        node = self.node_class()
        node.derive(None, gear_up, None, None, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_gear_in_transit(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Down changed to Up + following Gear In Transit
        gear_down = M('Gear Down', array=np.ma.array([1]*15 + [0]*40 + [1]*5),
                   values_mapping={0: 'Up', 1: 'Down'})
        in_trans = M('Gear In Transition', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*10 + [0]*20 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'In Transit'})
        node = self.node_class()
        node.derive(gear_down, None, None, in_trans, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_gear_in_transit_overlap(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Down changed to Up + following Gear In Transit
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                      values_mapping={0: 'Up', 1: 'Down'})
        in_trans = M('Gear In Transition', array=np.ma.array([0]*4 + [1]*10 + [0]*31 + [1]*10 + [0]*5),
                     values_mapping={0: '-', 1: 'In Transit'})
        expected = M('Gear Up In Transit', array=np.ma.array([0]*4 + [1]*10 + [0]*46),
                     values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(gear_down, None, None, in_trans, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_up_gear_in_transit(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Up changed to Up - following Gear In Transit
        gear_up = M('Gear Up', array=np.ma.array([0]*25 + [1]*20 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        in_trans = M('Gear In Transition', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*10 + [0]*20 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'In Transit'})
        node = self.node_class()
        node.derive(None, gear_up, None, in_trans, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_up_gear_in_transit_low_sample_rate(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Up changed to Up - following Gear In Transit
        gear_up = M('Gear Up', array=np.ma.array([0]*25 + [1]*20 + [0]*15),
                    values_mapping={0: 'Down', 1: 'Up'},
                    frequency=0.25)
        in_trans = M('Gear In Transition', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*10 + [0]*20 + [1]*10 + [0]*5),
                     values_mapping={0: '-', 1: 'In Transit'},
                     frequency=0.25)
        node = self.node_class()
        airborne = buildsection('Airborne', 17, 239)
        node.get_derived((None, gear_up, None, in_trans, None, None, airborne, self.model, self.series, self.family))

        expected_short = M('Gear Up In Transit',
                           array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*10 + [0]*35),
                           values_mapping=self.values_mapping)
        np.testing.assert_array_equal(node.array, expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_position(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Up changed to Up - following Gear In Transit
        gear_pos = M('Gear Position', array=np.ma.array([1]*5 + [3]*3 + [1]*7 + [3]*10 + [2]*20 + [3]*10 + [1]*5),
                      values_mapping={0: '-', 1: 'Down', 2: 'Up', 3: 'In Transit'})
        node = self.node_class()
        node.derive(None, None, None, None, None, gear_pos, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__up_sel_red_warn(self, at):
        at.get_gear_transition_times.return_value = (10, 10)  # patch transition time to be 10 seconds
        # Gear Up Selected changed to Up + transition time
        up_sel = M('Gear Up Selected', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        red_warn = M('Gear (*) Red Warning', array=np.ma.array([0]*5 + [1]*3 + [0]*7 + [1]*10 + [0]*20 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'Warning'})
        node = self.node_class()
        node.derive(None, None, up_sel, None, red_warn, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_red_warn(self, at):
        at.get_gear_transition_times.return_value = (10, 10)  # patch transition time to be 10 seconds
        # Gear Down changed to Up + transition time
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                   values_mapping={0: 'Up', 1: 'Down'})
        red_warn = M('Gear (*) Red Warning', array=np.ma.array([0]*5 + [1]*10 + [0]*10 + [1]*3 + [0]*17 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'Warning'})
        node = self.node_class()
        node.derive(gear_down, None, None, None, red_warn, None, self.airborne, self.model, self.series, self.family)
        # Gear Red Warning triggers spuriously. Cannot detect short periods not linked with Gear Down.
        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_up_red_warn(self, at):
        at.get_gear_transition_times.return_value = (5, 5)  # patch transition time to be 5 seconds
        # Gear Up changed to Up - transition time
        gear_up = M('Gear Up', array=np.ma.array([0]*15 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        red_warn = M('Gear (*) Red Warning', array=np.ma.array([0]*5 + [1]*10 + [0]*10 + [1]*3 + [0]*17 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'Warning'})
        node = self.node_class()
        node.derive(None, gear_up, None, None, red_warn, None, self.airborne, self.model, self.series, self.family)
        # Gear Red Warning triggers spuriously. Cannot detect short periods not linked with Gear Up.
        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_red_warn_B737_classic(self, at):
        at.get_gear_transition_times.return_value = (10, 10)  # patch transition time to be 10 seconds
        # Gear Down changed to Up + transition time
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                   values_mapping={0: 'Up', 1: 'Down'})
        red_warn = M('Gear (*) Red Warning', array=np.ma.array([0]*5 + [1]*10 + [0]*10 + [1]*3 + [0]*17 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'Warning'})
        node = self.node_class()
        node.derive(gear_down, None, None, None, red_warn, None, self.airborne, self.model, self.series, A('Family', value='B737 Classic'))
        # Gear Red Warning triggers spuriously. Cannot detect short periods not linked with Gear Down.
        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_up_AW169(self, at):
        at.get_gear_transition_times.return_value = (10, 10)
        gear_down = M('Gear Down', array=np.ma.array([1]*25 + [0]*20 + [1]*15),
                    values_mapping={1: 'Down', 0: 'Up'})

        node = self.node_class()
        node.derive(gear_down, None, None, None, None, None, self.airborne, self.model, self.series, A('Family', value='AW169'))

        expected = M('Gear Up In Transit', array=np.ma.array([0]*15 + [1]*10 + [0]*35),
                     values_mapping=self.values_mapping)
        np.testing.assert_array_equal(node.array, expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


class TestGearDownInTransit(unittest.TestCase):
    # Gear Down In Transit
    # Gear Down Selected changed to Down to Gear Up
    # Gear Up no longer Up to Gear Down changing to Down
    # Gear Down Selected changed to Down + following Gear In Transit
    # Gear Down Selected changed to Down + transition time
    # Gear Up changed to Down + transition time
    # Gear Down changed to Down - transition time
    # Gear Down changed to Down - following Gear In Transit

    def setUp(self):
        self.node_class = GearDownInTransit

        self.values_mapping = {
            0: '-',
            1: 'Extending' # 'Retracting' for Up In Transit
        }
        self.expected = M('Gear Down In Transit', array=np.ma.array([0]*45 + [1]*10 + [0]*5),
                          values_mapping=self.values_mapping)
        # Flight hash b2e7c6b6a28f shows gear in transit for a short amount of
        # time before gear is raised immediately.
        # Gear Down parameter did not show a Down state when the
        # landing gear was lowered, neither did Gear Up show a Down position at
        # that moment. We ought to detect that period too when recording the
        # necessary parameters: Gear Position, Gear In Transit, Gear Red or
        # Gear Down Selected
        # Gear Red Warnings unfortunately triggers spuriously on some dataframes.
        # So we cannot detect this edge case with Red Warnings.
        self.expected_short = M('Gear Down In Transit', array=np.ma.array([0]*35 + [1]*3 + [0]*7 + [1]*10 + [0]*5),
                                values_mapping=self.values_mapping)
        self.family = A('Aircraft Family', value='Generic Family')
        self.series = A('Aircraft Series', value='Generic Series')
        self.model = A('Aircraft Model', value='Generic Model')
        self.airborne=buildsection('Airborne', 1, 59)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()

        possible_combinations = (
            ('Gear Down', 'Gear Down Selected'),
            ('Gear Down', 'Gear Up'),
            ('Gear Down Selected', 'Gear In Transit'),
            ('Gear Up', 'Gear In Transit'),
            ('Gear Down', 'Gear In Transit'),
            ('Gear Position',),
        )

        for params in possible_combinations:
            self.assertTrue(params in combinations)

        fallback_params = ('Model', 'Series', 'Family', 'Gear Down Selected', 'Airborne')
        self.assertFalse(self.node_class.can_operate(fallback_params,
                                        model=self.model, series=self.series, family=self.family))
        self.assertTrue(self.node_class.can_operate(fallback_params,
                                                    model=self.model,
                                                    series=self.series,
                                                    family=A('Family',value='B737 Classic')))

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__down_sel_gear_down(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Down Selected changed to Up to short Gear Down to Up and to Gear Down
        down_sel = M('Gear Down Selected', array=np.ma.array([1]*5 + [0]*30 + [1]*3 + [0]*7 + [1]*15),
                   values_mapping={0: 'Up', 1: 'Down'})
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                      values_mapping={0: 'Up', 1: 'Down'})
        node = self.node_class()
        node.derive(gear_down, None, down_sel, None, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_gear_up(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Down no longer Down to Gear Up changing to Up
        gear_up = M('Gear Up', array=np.ma.array([0]*15 + [1]*31 + [0]*14),
                   values_mapping={0: 'Down', 1: 'Up'})
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                      values_mapping={0: 'Up', 1: 'Down'})
        node = self.node_class()
        node.derive(gear_down, gear_up, None, None, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__down_sel_gear_in_transit(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Down Selected Down + Gear In Transit
        down_sel = M('Gear Down Selected', array=np.ma.array([1]*5 + [0]*30 + [1]*3 + [0]*7 + [1]*15),
                   values_mapping={0: 'Up', 1: 'Down'})
        in_trans = M('Gear In Transition', array=np.ma.array([0]*5 + [1]*10 + [0]*20 + [1]*3 + [0]*7 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'In Transit'})
        node = self.node_class()
        node.derive(None, None, down_sel, in_trans, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__down_sel_transition_time(self, at):
        at.get_gear_transition_times.return_value = (10, 10)
        # Gear Up Selected changed to Up + transition time
        down_sel = M('Gear Down Selected', array=np.ma.array([1]*5 + [0]*40 + [1]*15),
                     values_mapping={0: 'Up', 1: 'Down'})
        node = self.node_class()
        node.derive(None, None, down_sel, None, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_transition_time(self, at):
        at.get_gear_transition_times.return_value = (10, 10)
        # Gear Down changed to Up + transition time
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                   values_mapping={0: 'Up', 1: 'Down'})
        node = self.node_class()
        node.derive(gear_down, None, None, None, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_up_transition_time(self, at):
        at.get_gear_transition_times.return_value = (10, 10)
        # Gear Up changed to Up - transition time
        gear_up = M('Gear Up', array=np.ma.array([0]*15 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        node = self.node_class()
        node.derive(None, gear_up, None, None, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_gear_in_transit(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Down changed to Up + following Gear In Transit
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                   values_mapping={0: 'Up', 1: 'Down'})
        in_trans = M('Gear In Transition', array=np.ma.array([0]*5 + [1]*10 + [0]*20 + [1]*3 + [0]*7 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'In Transit'})
        node = self.node_class()
        node.derive(gear_down, None, None, in_trans, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_gear_in_transit_low_sample_rate(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Down changed to Up + following Gear In Transit
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                      values_mapping={0: 'Up', 1: 'Down'},
                      frequency=0.25)
        in_trans = M('Gear In Transition', array=np.ma.array([0]*5 + [1]*10 + [0]*20 + [1]*3 + [0]*7 + [1]*10 + [0]*5),
                     values_mapping={0: '-', 1: 'In Transit'},
                     frequency=0.25)
        airborne=buildsection('Airborne', 17, 230)  # Frequency 1 Hz
        node = self.node_class()
        node.get_derived((gear_down, None, None, in_trans, None, None, airborne, self.model, self.series, self.family))

        expected_short = M('Gear Down In Transit',
                           array=np.ma.array([0]*35 + [1]*3 + [0]*7 + [1]*10 + [0]*5),
                           values_mapping=self.values_mapping)
        np.testing.assert_array_equal(node.array, expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_up_gear_in_transit(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Up changed to Up - following Gear In Transit
        gear_up = M('Gear Up', array=np.ma.array([0]*15 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        in_trans = M('Gear In Transition', array=np.ma.array([0]*5 + [1]*10 + [0]*20 + [1]*3 + [0]*7 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'In Transit'})
        node = self.node_class()
        node.derive(None, gear_up, None, in_trans, None, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_position(self, at):
        at.get_gear_transition_times.return_value = (15, 15)
        # Gear Position changing from Down to Up then In Transit for a short time then Up and Down
        gear_pos = M('Gear Position', array=np.ma.array([1]*5 + [3]*10 + [2]*20 + [3]*3 + [2]*7 + [3]*10 + [1]*5),
                      values_mapping={0: '-', 1: 'Down', 2: 'Up', 3: 'In Transit'})
        node = self.node_class()
        node.derive(None, None, None, None, None, gear_pos, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__down_sel_red_warning(self, at):
        at.get_gear_transition_times.return_value = (10, 10)
        # Gear Down Selected changed to Up + red warning
        ac_series = A('Generic') # patch transition time to be 10 seconds
        down_sel = M('Gear Down Selected', array=np.ma.array([1]*5 + [0]*30 + [1]*3 + [0]*7 + [1]*15),
                     values_mapping={0: 'Up', 1: 'Down'})
        gear_red_warn = M('Gear (*) Red Warning', array=np.ma.array([0]*5 + [1]*10 + [0]*20 + [1]*3 + [0]*7 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'Warning'})
        node = self.node_class()
        node.derive(None, None, down_sel, None, gear_red_warn, None, self.airborne, self.model, self.series, self.family)

        np.testing.assert_array_equal(node.array, self.expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_red_warning(self, at):
        at.get_gear_transition_times.return_value = (10, 10)
        # Gear Down changed to Up + Red Warning
        ac_series = A('Generic') # patch transition time to be 10 seconds
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                   values_mapping={0: 'Up', 1: 'Down'})
        red_warn = M('Gear (*) Red Warning', array=np.ma.array([0]*5 + [1]*10 + [0]*20 + [1]*3 + [0]*7 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'Warning'})
        node = self.node_class()
        node.derive(gear_down, None, None, None, red_warn, None, self.airborne, self.model, self.series, self.family)
        # Gear Red Warning triggers spuriously. Cannot detect short periods not linked with Gear Down.
        expected_short = M('Gear Down In Transit', array=np.ma.array([0]*45 + [1]*10 + [0]*5),
                                        values_mapping=self.values_mapping)
        np.testing.assert_array_equal(node.array, expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_up_red_warning(self, at):
        at.get_gear_transition_times.return_value = (5, 5)
        # Gear Up changed to Up + Red Warning
        ac_series = A('Generic') # patch transition time to be 5 seconds
        gear_up = M('Gear Up', array=np.ma.array([0]*15 + [1]*30 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        red_warn = M('Gear (*) Red Warning', array=np.ma.array([0]*5 + [1]*10 + [0]*20 + [1]*3 + [0]*7 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'Warning'})
        node = self.node_class()
        node.derive(None, gear_up, None, None, red_warn, None, self.airborne, self.model, self.series, self.family)
        # Gear Red Warning triggers spuriously. Cannot detect short periods not linked with Gear Down.
        expected_short = M('Gear Down In Transit', array=np.ma.array([0]*45 + [1]*10 + [0]*5),
                                        values_mapping=self.values_mapping)
        np.testing.assert_array_equal(node.array, expected_short.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    @patch('analysis_engine.multistate_parameters.at')
    def test_derive__gear_down_red_warning_B737_classic(self, at):
        at.get_gear_transition_times.return_value = (10, 10)
        # Gear Down changed to Up + Red Warning and transition time
        ac_series = A('Generic') # patch transition time to be 10 seconds
        gear_down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                   values_mapping={0: 'Up', 1: 'Down'})
        red_warn = M('Gear (*) Red Warning', array=np.ma.array([0]*5 + [1]*10 + [0]*25 + [1]*15 + [0]*5),
                      values_mapping={0: '-', 1: 'Warning'})
        node = self.node_class()
        node.derive(gear_down, None, None, None, red_warn, None, self.airborne, self.model, self.series, A('Family', value='B737 Classic'))

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


class TestGearDown(unittest.TestCase):

    # Gear Down
    # Gear Down Selected - Gear Up In Transit

    def setUp(self):
        self.node_class = GearDown

        self.values_mapping = {
            0: 'Up',
            1: 'Down',
        }
        self.expected = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                          values_mapping=self.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()

        possible_combinations = (
            ('Gear Down In Transit', 'Gear Down Selected'),
            ('Gear Position',),
        )

        for params in possible_combinations:
            self.assertTrue(params in combinations)

    def test_derive__down_sel_down_in_transit(self):
        down_sel = M('Gear Down Selected', array=np.ma.array([1]*5 + [0]*40 + [1]*15),
                   values_mapping={0: 'Up', 1: 'Down'})
        down_transit = M('Gear Down In Transit', array=np.ma.array([0]*45 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'Extending'})
        node = self.node_class()
        node.derive(down_transit, down_sel, None)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    def test_derive__gear_position(self):
        gear_pos = M('Gear Position', array=np.ma.array([1]*5 + [3]*10 + [2]*30 + [3]*10 + [1]*5),
                      values_mapping={0: '-', 1: 'Down', 2: 'Up', 3: 'In Transit'})
        node = self.node_class()
        node.derive(None, None, gear_pos)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)



class TestGearUp(unittest.TestCase):

    #Gear Up
    # Gear Up Selected - Gear Up In Transit

    def setUp(self):
        self.node_class = GearUp

        self.values_mapping = {
            0: 'Down',
            1: 'Up',
        }
        self.expected = M('Gear Up', array=np.ma.array([0]*15 + [1]*30 + [0]*15),
                          values_mapping=self.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()

        possible_combinations = (
            ('Gear Up In Transit', 'Gear Up Selected'),
            ('Gear Position',),
        )

        for params in possible_combinations:
            self.assertTrue(params in combinations)

    def test_derive__up_sel_up_in_transit(self):
        up_sel = M('Gear Up Selected', array=np.ma.array([0]*5 + [1]*40 + [0]*15),
           values_mapping={0: 'Down', 1: 'Up'})
        up_trans = M('Gear Up In Transit', array=np.ma.array([0]*5 + [1]*10 + [0]*45),
                      values_mapping={0: '-', 1: 'Retracting'})
        node = self.node_class()
        node.derive(up_trans, up_sel, None)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    def test_derive__gear_position(self):
        gear_pos = M('Gear Position', array=np.ma.array([1]*5 + [3]*10 + [2]*30 + [3]*10 + [1]*5),
                      values_mapping={0: '-', 1: 'Down', 2: 'Up', 3: 'In Transit'})
        node = self.node_class()
        node.derive(None, None, gear_pos)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


class TestGearDownSelected(unittest.TestCase):

    #Gear Down Selection
    # Invert Gear Up Selection
    # Gear Down + Gear Down In Transit

    def setUp(self):
        self.node_class = GearDownSelected

        self.values_mapping = {
            0: 'Up',
            1: 'Down',
        }
        self.expected = M('Gear Up', array=np.ma.array([1]*5 + [0]*40 + [1]*15),
                          values_mapping=self.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()

        possible_combinations = (('Gear Up Selected',),)
        for params in possible_combinations:
            self.assertTrue(params in combinations)

    def test_derive__up_sel(self):
        # Invert Gear Up Selection
        up_sel = M('Gear Up Selected', array=np.ma.array([0]*5 + [1]*40 + [0]*15),
                   values_mapping={0: 'Down', 1: 'Up'})
        node = self.node_class()
        node.derive(up_sel)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


class TestGearUpSelected(unittest.TestCase):

    #Gear Up Selection
    # Invert Gear Down Selection
    # Gear Up + Gear Up In Transit

    def setUp(self):
        self.node_class = GearUpSelected
        self.values_mapping = {
            0: 'Down',
            1: 'Up',
        }
        self.expected = M('Gear Up', array=np.ma.array([0]*5 + [1]*40 + [0]*15),
                          values_mapping=self.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()

        possible_combinations = (
            ('Gear Up', 'Gear Up In Transit'),
            ('Gear Down', 'Gear Down In Transit'),
            ('Gear Down Selected', ),
        )

        for params in possible_combinations:
            self.assertTrue(params in combinations)

    def test_derive__up_up_in_transit(self):
        gear_up = M('Gear Up', array=np.ma.array([0]*15 + [1]*30 + [0]*15),
           values_mapping={0: 'Down', 1: 'Up'})
        up_trans = M('Gear Up In Transit', array=np.ma.array([0]*5 + [1]*10 + [0]*45),
                      values_mapping={0: '-', 1: 'Retracting'})
        node = self.node_class()
        node.derive(gear_up, up_trans, None, None, None)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    def test_derive__down_down_in_transit(self):
        down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                   values_mapping={0: 'Up', 1: 'Down'})
        down_transit = M('Gear Down In Transit', array=np.ma.array([0]*45 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'Extending'})
        node = self.node_class()
        node.derive(None, None, down, down_transit, None)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    def test_derive__down_selected(self):
        gear_down_sel = M(
            'Gear Down Selected',
            array=np.ma.array([1]*5 + [0]*40 + [1]*15),
            values_mapping={0: 'Up', 1: 'Down'}
        )
        node = self.node_class()
        node.derive(None, None, None, None, gear_down_sel)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)

    def test_derive__down_selected_low_sample_rate(self):
        gear_down_sel = M(
            'Gear Down Selected',
            array=np.ma.array([1]*2 + [0]*20 + [1]*8),
            values_mapping={0: 'Up', 1: 'Down'},
            frequency=0.5
        )
        down = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                   values_mapping={0: 'Up', 1: 'Down'})
        down_transit = M('Gear Down In Transit', array=np.ma.array([0]*45 + [1]*10 + [0]*5),
                      values_mapping={0: '-', 1: 'Extending'})
        node = self.node_class()
        node.derive(None, None, down, down_transit, gear_down_sel)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


class TestGearInTransit(unittest.TestCase):

    #Gear In Transit
    # combine individal (L/R/C/N) params
    # Gear Up In Transit + Gear Down In Transit

    def setUp(self):
        self.node_class = GearInTransit
        self.values_mapping = {
            0: '-',
            1: 'In Transit',
        }
        self.expected = M(
            'Gear In Transition',
            array=np.ma.array([0]*5 + [1]*10 + [0]*30 + [1]*10 + [0]*5),
            values_mapping=self.values_mapping
        )

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()

        possible_combinations = (
            ('Gear Down In Transit', 'Gear Up In Transit'),
        )
        for params in possible_combinations:
            self.assertTrue(params in combinations)

    def test_derive(self):
        gear_down_transit = M('Gear Down In Transit',
                              array=np.ma.array([0]*45 + [1]*10 + [0]*5),
                              values_mapping={0: '-', 1: 'Extending'})
        gear_up_transit = M('Gear Up In Transit',
                            array=np.ma.array([0]*5 + [1]*10 + [0]*45),
                            values_mapping={0: '-', 1: 'Retracting'})
        node = self.node_class()
        node.derive(gear_down_transit, gear_up_transit)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


class TestTransmitting(unittest.TestCase):

    def setUp(self):
        self.node_class = Transmitting

    def test_can_operate(self):
        # Due to the huge number of optional dependencies, the powerset
        # generated by get_operational_combinations contains 8388608
        # combinations and raises a MemoryError.
        self.assertTrue(self.node_class.can_operate(('Key HF',)))
        self.assertTrue(self.node_class.can_operate(('Key HF (1)',)))
        self.assertTrue(self.node_class.can_operate(('Key HF (2)',)))
        self.assertTrue(self.node_class.can_operate(('Key HF (3)',)))
        self.assertTrue(self.node_class.can_operate(('Key HF (1) (Capt)',)))
        self.assertTrue(self.node_class.can_operate(('Key HF (2) (Capt)',)))
        self.assertTrue(self.node_class.can_operate(('Key HF (3) (Capt)',)))
        self.assertTrue(self.node_class.can_operate(('Key HF (1) (FO)',)))
        self.assertTrue(self.node_class.can_operate(('Key HF (2) (FO)',)))
        self.assertTrue(self.node_class.can_operate(('Key HF (3) (FO)',)))
        self.assertTrue(self.node_class.can_operate(('Key Satcom',)))
        self.assertTrue(self.node_class.can_operate(('Key Satcom (1)',)))
        self.assertTrue(self.node_class.can_operate(('Key Satcom (2)',)))
        self.assertTrue(self.node_class.can_operate(('Key VHF',)))
        self.assertTrue(self.node_class.can_operate(('Key VHF (1)',)))
        self.assertTrue(self.node_class.can_operate(('Key VHF (2)',)))
        self.assertTrue(self.node_class.can_operate(('Key VHF (3)',)))
        self.assertTrue(self.node_class.can_operate(('Key VHF (1) (Capt)',)))
        self.assertTrue(self.node_class.can_operate(('Key VHF (2) (Capt)',)))
        self.assertTrue(self.node_class.can_operate(('Key VHF (3) (Capt)',)))
        self.assertTrue(self.node_class.can_operate(('Key VHF (1) (FO)',)))
        self.assertTrue(self.node_class.can_operate(('Key VHF (2) (FO)',)))
        self.assertTrue(self.node_class.can_operate(('Key VHF (3) (FO)',)))

    def test_derive_single(self):
        array = MappedArray([0, 1, 1, 0, 0, 1, 0],
                            values_mapping={0: '-', 1: 'Keyed'})
        dependency_names = self.node_class.get_dependency_names()
        for idx, name in enumerate(dependency_names):
            node = self.node_class()
            node.derive(*([None] * idx) + [M(name, array)] + ([None] * (len(dependency_names) - idx - 1)))
            self.assertEqual(node.array.raw.tolist(), array.raw.tolist())

    def test_derive_multiple(self):
        node = self.node_class()
        node.derive(
            None,
            M('Key HF (1)', MappedArray([0, 1, 1, 0, 0, 1, 0],
                                        values_mapping={0: '-', 1: 'Keyed'})),
            M('Key HF (2)', MappedArray([0, 0, 1, 1, 0, 0, 0],
                                        values_mapping={0: '-', 1: 'Keyed'})),
            *[None] * (len(self.node_class.get_dependency_names()) - 3))
        self.assertEqual(node.array.tolist(),
                         ['-', 'Transmit', 'Transmit', 'Transmit', '-', 'Transmit', '-'])



