import numpy as np

import datetime
import os
import shutil
import sys
import tempfile
import unittest
import csv

from unittest.mock import Mock, patch

from numpy.ma.testutils import assert_array_almost_equal, assert_array_equal, assert_almost_equal, assert_equal

from hdfaccess.file import hdf_file
from flightdatautilities import aircrafttables as at, units as ut
from flightdatautilities.array_operations import load_compressed
from flightdatautilities.aircrafttables.interfaces import VelocitySpeed
from flightdatautilities import masked_array_testutils as ma_test
from flightdatautilities.filesystem_tools import copy_file

from analysis_engine.flight_phase import Fast, Mobile
from analysis_engine.library import (align,
                                     max_value,
                                     np_ma_masked_zeros_like,
                                     np_ma_ones_like,
                                     overflow_correction,
                                     unique_values)

from analysis_engine.node import (
    A, M, P, S, KPV, KTI, aeroplane, App, ApproachItem, Attribute,
    helicopter, KeyPointValue, KeyTimeInstance, load, Parameter, Section
)

from analysis_engine.process_flight import process_flight

from analysis_engine.test_utils import buildsection, buildsections

# Use pre-processed version
from analysis_engine.pre_processing.merge_parameters import (
    LatitudePrepared as LatitudePreparedLatLon,
    LongitudePrepared as LongitudePreparedLatLon
)

from analysis_engine.derived_parameters import (
    AOA,
    AccelerationAcrossTrack,
    AccelerationAlongTrack,
    AccelerationForwards,
    AccelerationLateralOffsetRemoved,
    AccelerationLateralSmoothed,
    AccelerationLongitudinalOffsetRemoved,
    AccelerationNormalLowLimitForLandingWeight,
    AccelerationNormalHighLimitForLandingWeight,
    AccelerationNormalHighLimitWithFlapsDown,
    AccelerationSideways,
    AccelerationVertical,
    AccelerationNormalOffsetRemoved,
    Aileron,
    AileronLeft,
    AileronRight,
    AimingPointRange,
    AircraftEnergy,
    AirspeedMinusAirspeedSelectedFor3Sec,
    AirspeedMinusAirspeedSelectedFMS,
    AirspeedMinusAirspeedSelectedFMSFor3Sec,
    AirspeedMinusFlapManoeuvreSpeed,
    AirspeedMinusFlapManoeuvreSpeedFor3Sec,
    AirspeedMinusMinimumAirspeed,
    AirspeedMinusMinimumAirspeedFor3Sec,
    AirspeedMinusV2,
    AirspeedMinusV2For3Sec,
    AirspeedMinusVapp,
    AirspeedMinusVappFor3Sec,
    AirspeedMinusVref,
    AirspeedMinusVrefFor3Sec,
    AirspeedMinusVLS,
    AirspeedMinusVLSFor3Sec,
    AirspeedRelative,
    AirspeedRelativeFor3Sec,
    AirspeedSelected,
    AirspeedSelectedForApproaches,
    AirspeedTrue,
    AltitudeAAL,
    AltitudeAALForFlightPhases,
    AltitudeQNH,
    AltitudeQNHCapt,
    AltitudeQNHFO,
    AltitudeVisualizationWithGroundOffset,
    AltitudeVisualizationWithoutGroundOffset,
    AltitudeRadio,
    AltitudeRadioOffsetRemoved,
    AltitudeSTDSmoothed,
    AltitudeTail,
    ApproachRange,
    BaroCorrection,
    BrakePressure,
    Brake_C_Temp,
    Brake_L_Temp,
    Brake_R_Temp,
    Brake_TempAvg,
    Brake_TempMax,
    Brake_TempMin,
    CabinAltitude,
    ClimbForFlightPhases,
    ControlColumn,
    ControlColumnForce,
    ControlWheel,
    ControlWheelForce,
    ControlColumnForceAtControlWheelCapt,
    ControlColumnForceAtControlWheelFO,
    CoordinatesSmoothed,
    DescendForFlightPhases,
    DistanceFlown,
    DistanceToLanding,
    DistanceTravelled,
    Drift,
    Elevator,
    ElevatorLeft,
    ElevatorRight,
    EngTPRLimitDifference,
    Eng_1_FuelBurn,
    Eng_2_FuelBurn,
    Eng_3_FuelBurn,
    Eng_4_FuelBurn,
    Eng_EPRAvg,
    Eng_EPRMax,
    Eng_EPRMin,
    Eng_EPRMinFor5Sec,
    Eng_FuelFlow,
    Eng_FuelFlowMax,
    Eng_FuelFlowMin,
    Eng_N1Avg,
    Eng_N1Max,
    Eng_N1Min,
    Eng_N1MinFor5Sec,
    Eng_N2Avg,
    Eng_N2Max,
    Eng_N2Min,
    Eng_N3Avg,
    Eng_N3Max,
    Eng_N3Min,
    Eng_NpAvg,
    Eng_NpMax,
    Eng_NpMin,
    Eng_TorqueAvg,
    Eng_TorqueMax,
    Eng_TorqueMin,
    Eng_VibAMax,
    Eng_VibBMax,
    Eng_VibBroadbandMax,
    Eng_VibCMax,
    Eng_VibN1Max,
    Eng_VibN2Max,
    Eng_VibN3Max,
    Eng_VibNpMax,
    FlapAngle,
    FlapManoeuvreSpeed,
    FlapSynchroAsymmetry,
    ApproachFlightPathAngle,
    FuelQty,
    FuelQtyC,
    FuelQtyL,
    FuelQtyR,
    FuelQtyAux,
    GrossWeight,
    GrossWeightSmoothed,
    Groundspeed,
    GroundspeedSigned,
    Heading,
    HeadingContinuous,
    HeadingRate,
    HeadingTrue,
    HeadingTrueContinuous,
    Headwind,
    ILSFrequency,
    KineticEnergy,
    LatitudePrepared,
    LatitudeSmoothed,
    LongitudePrepared,
    LongitudeSmoothed,
    MMOLookup,
    Mach,
    MagneticVariation,
    MagneticVariationFromRunway,
    MinimumAirspeed,
    MinimumCleanLookup,
    Pitch,
    PitchFor3Sec,
    PotentialEnergy,
    Roll,
    RollFor3Sec,
    RollRate,
    RollRateForTouchdown,
    RollRateAtTouchdownLimit,
    Rudder,
    RudderPedal,
    RudderPedalCapt,
    RudderPedalFO,
    SAT_ISA,
    SidestickAngleCapt,
    SidestickAngleFO,
    SlatAngle,
    SlopeAngleToAimingPoint,
    SlopeAngleToLanding,
    SlopeToAimingPoint,
    SlopeToLanding,
    Speedbrake,
    Stabilizer,
    TAT,
    ThrottleLevers,
    ThrustAsymmetry,
    TorqueAsymmetry,
    Track,
    TrackContinuous,
    TrackDeviationFromRunway,
    TrackTrue,
    TrackTrueContinuous,
    Turbulence,
    VMOLookup,
    Vapp,
    VappLookup,
    VerticalSpeed,
    VerticalSpeedForFlightPhases,
    VerticalSpeedFor3Sec,
    VerticalSpeedInertial,
    Vref,
    VrefLookup,
    VLSLookup,
    WheelSpeed,
    WheelSpeedLeft,
    WheelSpeedRight,
    WindAcrossLandingRunway,
    WindDirection,
    ZeroFuelWeight
)

from analysis_engine.settings import GRAVITY_IMPERIAL

##############################################################################
# Test Configuration


def setUpModule():
    at.configure(package='flightdatautilities.aircrafttables')


##############################################################################


debug = sys.gettrace() is not None

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

def assert_array_within_tolerance(actual, desired, tolerance=1, similarity=100):
    '''
    Check that the actual array within tolerance of the desired array is
    at least similarity percent.

    :param tolerance: relative difference between the two array values
    :param similarity: percentage that must pass the tolerance test
    '''
    within_tolerance = abs(actual -  desired) <= tolerance
    percent_similar = np.ma.sum(within_tolerance) / float(len(within_tolerance)) * 100
    if percent_similar <= similarity:
        raise AssertionError(
            'actual array tolerance only is %.2f%% similar to desired array.'
            'tolerance %.2f minimum similarity required %.2f%%' % (
                percent_similar, tolerance, similarity))


class TemporaryFileTest(object):
    '''
    Test using a temporary copy of a predefined file.
    '''
    def setUp(self):
        if getattr(self, 'source_file_path', None):
            self.make_test_copy()

    def tearDown(self):
        if self.test_file_path:
            os.unlink(self.test_file_path)
            self.test_file_path = None

    def make_test_copy(self):
        '''
        Copy the test file to temporary location, used by setUp().
        '''
        # Create the temporary file in the most secure way
        f = tempfile.NamedTemporaryFile(delete=False)
        self.test_file_path = f.name
        f.close()
        shutil.copy2(self.source_file_path, self.test_file_path)


class NodeTest(object):

    def generate_attributes(self, manufacturer):
        if manufacturer == 'boeing':
            _am = A('Model', 'B737-333')
            _as = A('Series', 'B737-300')
            _af = A('Family', 'B737 Classic')
            _et = A('Engine Type', 'CFM56-3B1')
            _es = A('Engine Series', 'CFM56-3')
            return (_am, _as, _af, _et, _es)
        if manufacturer == 'airbus':
            _am = A('Model', 'A330-333')
            _as = A('Series', 'A330-300')
            _af = A('Family', 'A330')
            _et = A('Engine Type', 'Trent 772B-60')
            _es = A('Engine Series', 'Trent 772B')
            return (_am, _as, _af, _et, _es)
        if manufacturer == 'beechcraft':
            _am = A('Model', '1900D')
            _as = A('Series', '1900D')
            _af = A('Family', '1900')
            _et = A('Engine Type', 'PT6A-67D')
            _es = A('Engine Series', 'PT6A')
            return (_am, _as, _af, _et, _es)
        raise ValueError('Unexpected lookup for attributes.')

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

        params = []
        phase = None

        with tempfile.NamedTemporaryFile() as temp_file:
            shutil.copy(hdf_path, temp_file.name)

            with hdf_file(hdf_path) as hdf:
                for param_name in param_names:
                    params.append(hdf.get(param_name))

        if _slice:
            phase = S(name=phase_name, frequency=1)
            phase.create_section(_slice)
            phase = phase.get_aligned(params[0])

        return params, phase


##### FIXME: Re-enable when 'AT Engaged' has been implemented.
####class TestATEngaged(unittest.TestCase, NodeTest):
####
####    def setUp(self):
####        self.node_class = ATEngaged
####        self.operational_combinations = [
####            ('AT (1) Engaged',),
####            ('AT (2) Engaged',),
####            ('AT (3) Engaged',),
####            ('AT (1) Engaged', 'AT (2) Engaged'),
####            ('AT (1) Engaged', 'AT (3) Engaged'),
####            ('AT (2) Engaged', 'AT (3) Engaged'),
####            ('AT (1) Engaged', 'AT (2) Engaged', 'AT (3) Engaged'),
####        ]
####
####    @unittest.skip('Test Not Implemented')
####    def test_derive(self):
####        self.assertTrue(False, msg='Test not implemented.')


##############################################################################


class TestAccelerationLateralSmoothed(unittest.TestCase):
    def test_can_operate(self):
        opts = AccelerationLateralSmoothed.get_operational_combinations()
        self.assertEqual(opts, [('Acceleration Lateral Offset Removed',)])

    def test_smoothing(self):
        acc = AccelerationLateralSmoothed()
        acc.derive(P(array=np.ma.concatenate((np.zeros(26), [5, 10, 10, 5, 25, 10, 5, 5, 0])),
                     frequency=4))
        self.assertEqual(acc.window, 9)  # 2secs * 4hz +1
        self.assertEqual(np.ma.min(acc.array), 0)
        self.assertAlmostEqual(np.ma.max(acc.array), 8.333, 2)


class TestAccelerationVertical(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal Offset Removed',
                     'Acceleration Lateral Offset Removed',
                     'Acceleration Longitudinal', 'Pitch', 'Roll')]
        opts = AccelerationVertical.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_acceleration_vertical_level_on_gound(self):
        # Invoke the class object
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            Parameter('Acceleration Normal Offset Removed', np.ma.ones(8), 8),
            Parameter('Acceleration Lateral Offset Removed', np.ma.zeros(4), 4),
            Parameter('Acceleration Longitudinal', np.ma.zeros(4), 4),
            Parameter('Pitch', np.ma.zeros(2), 2),
            Parameter('Roll', np.ma.zeros(2), 2),
        ])

        #                                     x   interp  x  pitch/roll masked
        ma_test.assert_masked_array_approx_equal(
            acc_vert.array,
            np.ma.array(np.ones(8), mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )

    def test_acceleration_vertical_pitch_up(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal Offset Removed', np.ma.ones(8) * 0.8660254,8),
            P('Acceleration Lateral Offset Removed', np.ma.zeros(4), 4),
            P('Acceleration Longitudinal', np.ma.ones(4) * 0.5,4),
            P('Pitch', np.ma.ones(2) * 30.0,2),
            P('Roll', np.ma.zeros(2), 2)
        ])

        #                                     x   interp  x  pitch/roll masked
        ma_test.assert_masked_array_approx_equal(
            acc_vert.array,
            np.ma.array(np.ones(8), mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )

    def test_acceleration_vertical_pitch_up_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal Offset Removed', np.ma.ones(8) * 0.8, 8),
            P('Acceleration Lateral Offset Removed', np.ma.ones(4) * (-0.2), 4),
            P('Acceleration Longitudinal', np.ma.ones(4) * 0.3, 4),
            P('Pitch',np.ma.ones(2) * 30.0, 2),
            P('Roll',np.ma.ones(2) * 20, 2)])

        expected = np.ma.array(np.ones(8) * 0.86027777, mask=np.concatenate((np.zeros(5), np.ones(3))))
        ma_test.assert_masked_array_approx_equal(acc_vert.array, expected)

    def test_acceleration_vertical_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.get_derived([
            P('Acceleration Normal Offset Removed', np.ma.ones(8) * 0.7071068, 8),
            P('Acceleration Lateral Offset Removed', np.ma.ones(4) * -0.7071068, 4),
            P('Acceleration Longitudinal', np.ma.zeros(4), 4),
            P('Pitch', np.ma.zeros(2), 2),
            P('Roll', np.ma.ones(2) * 45, 2),
        ])
        #                                     x   interp  x  pitch/roll masked
        ma_test.assert_masked_array_approx_equal(
            acc_vert.array,
            np.ma.array(np.ones(8), mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )


class TestAccelerationForwards(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal Offset Removed',
                    'Acceleration Longitudinal', 'Pitch')]
        opts = AccelerationForwards.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_acceleration_forward_level_on_gound(self):
        # Invoke the class object
        acc_fwd = AccelerationForwards(frequency=4)

        acc_fwd.get_derived([
            Parameter('Acceleration Normal Offset Removed', np.ma.ones(8), 8),
            Parameter('Acceleration Longitudinal', np.ma.ones(4) * 0.1,4),
            Parameter('Pitch', np.ma.zeros(2), 2)
        ])
        ma_test.assert_masked_array_approx_equal(
            acc_fwd.array,
            np.ma.array(np.ones(8) * 0.1, mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )

    def test_acceleration_forward_pitch_up(self):
        acc_fwd = AccelerationForwards(frequency=4)

        acc_fwd.get_derived([
            P('Acceleration Normal Offset Removed', np.ma.ones(8) * 0.8660254, 8),
            P('Acceleration Longitudinal', np.ma.ones(4) * 0.5, 4),
            P('Pitch', np.ma.ones(2) * 30.0, 2)
        ])

        ma_test.assert_masked_array_approx_equal(
            acc_fwd.array,
            np.ma.array(np.zeros(8), mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )


class TestAccelerationSideways(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal Offset Removed',
                    'Acceleration Lateral Offset Removed',
                    'Acceleration Longitudinal', 'Pitch', 'Roll')]
        opts = AccelerationSideways.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_acceleration_sideways_level_on_gound(self):
        # Invoke the class object
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.get_derived([
            Parameter('Acceleration Normal Offset Removed', np.ma.ones(8),8),
            Parameter('Acceleration Lateral Offset Removed', np.ma.ones(4)*0.05,4),
            Parameter('Acceleration Longitudinal', np.ma.zeros(4),4),
            Parameter('Pitch', np.ma.zeros(2),2),
            Parameter('Roll', np.ma.zeros(2),2)
        ])
        #                                     x   interp  x  pitch/roll masked
        ma_test.assert_masked_array_approx_equal(
            acc_lat.array,
            np.ma.array(np.ones(8) * 0.05, mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )

    def test_acceleration_sideways_pitch_up(self):
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.get_derived([
            P('Acceleration Normal Offset Removed',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Lateral Offset Removed',np.ma.zeros(4),4),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2),
            P('Roll',np.ma.zeros(2),2)
        ])
        #                                     x   interp  x  pitch/roll masked
        ma_test.assert_masked_array_approx_equal(
            acc_lat.array,
            np.ma.array(np.zeros(8), mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )

    def test_acceleration_sideways_roll_right(self):
        acc_lat = AccelerationSideways(frequency=8)

        acc_lat.get_derived([
            P('Acceleration Normal Offset Removed',np.ma.ones(8)*0.7071068,8),
            P('Acceleration Lateral Offset Removed',np.ma.ones(4)*(-0.7071068),4),
            P('Acceleration Longitudinal',np.ma.zeros(4),4),
            P('Pitch',np.ma.zeros(2),2),
            P('Roll',np.ma.ones(2)*45,2)
        ])
        #                                     x   interp  x  pitch/roll masked
        ma_test.assert_masked_array_approx_equal(
            acc_lat.array,
            np.ma.array(np.zeros(8), mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )


class TestAccelerationAcrossTrack(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Forwards',
                    'Acceleration Sideways', 'Drift')]
        opts = AccelerationAcrossTrack.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_acceleration_across_side_only(self):
        acc_across = AccelerationAcrossTrack()
        acc_across.get_derived([
            Parameter('Acceleration Forwards', np.ma.ones(8), 8),
            Parameter('Acceleration Sideways', np.ma.ones(4)*0.1, 4),
            Parameter('Drift', np.ma.zeros(2), 2)])
        ma_test.assert_masked_array_approx_equal(
            acc_across.array,
            np.ma.array(np.ones(8) * 0.1, mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )

    def test_acceleration_across_resolved(self):
        acc_across = AccelerationAcrossTrack()
        acc_across.get_derived([
            P('Acceleration Forwards',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Sideways',np.ma.ones(4)*0.5,4),
            P('Drift',np.ma.ones(2)*30.0,2)])

        ma_test.assert_masked_array_approx_equal(
            acc_across.array,
            np.ma.array(np.zeros(8), mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )


class TestAccelerationAlongTrack(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Forwards',
                    'Acceleration Sideways', 'Drift')]
        opts = AccelerationAlongTrack.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_acceleration_along_forward_only(self):
        acc_along = AccelerationAlongTrack()
        acc_along.get_derived([
            Parameter('Acceleration Forwards', np.ma.ones(8)*0.2,8),
            Parameter('Acceleration Sideways', np.ma.ones(4)*0.1,4),
            Parameter('Drift', np.ma.zeros(2),2)])

        ma_test.assert_masked_array_approx_equal(
            acc_along.array,
            np.ma.array(np.ones(8) * 0.2, mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )

    def test_acceleration_along_resolved(self):
        acc_along = AccelerationAlongTrack()
        acc_along.get_derived([
            P('Acceleration Forwards',np.ma.ones(8)*0.1,8),
            P('Acceleration Sideways',np.ma.ones(4)*0.2,4),
            P('Drift',np.ma.ones(2)*10.0,2)])
        ma_test.assert_masked_array_approx_equal(
            acc_along.array,
            np.ma.array(np.ones(8) * 0.13321041, mask=np.concatenate((np.zeros(5), np.ones(3)))),
        )


class TestAccelerationNormalLowLimitForLandingWeight(unittest.TestCase):
    def test_derive(self):
        gross_weight = P('Gross Weight Smoothed',
                         np.ma.array([99999, 35000, 35000, 34000 ,33400,
                                      33300, 31000, 30000, 27000, 26000,
                                      25000, 24500, 23500, 23000, 0.000,]))
        acc_n_lim = AccelerationNormalLowLimitForLandingWeight()
        acc_n_lim.derive(gross_weight)
        expected = np.ma.concatenate((np.ones(3) * 1.75, np.ones(7) * 2.0, np.ones(1) * 2.0332777592530844, np.ones(1) * 2.0666222074024674, np.ones(1) * 2.133311103701234, np.ones(1) * 2.166655551850617, np.ones(1) * 2.2))
        ma_test.assert_array_equal(acc_n_lim.array, expected)

class TestAccelerationNormalHighLimitForLandingWeight(unittest.TestCase):
    def test_derive(self):
        gross_weight = P('Gross Weight Smoothed',
                         np.ma.array([99999, 35000, 35000, 34000 ,33400,
                                      33300, 31000, 30000, 27000, 26000,
                                      25000, 24500, 23500, 23000, 0.000,]))
        acc_n_lim = AccelerationNormalHighLimitForLandingWeight()
        acc_n_lim.derive(gross_weight)
        expected = np.ma.concatenate((np.ones(3) * 1.93, np.ones(7) * 2.2, np.ones(1) * 2.236605535178393, np.ones(1) * 2.2732844281427145, np.ones(1) * 2.346642214071357, np.ones(1) * 2.3833211070356786, np.ones(1) * 2.42))
        ma_test.assert_array_equal(acc_n_lim.array, expected)


class TestAccelerationNormalHighLimitWithFlapsDown(unittest.TestCase):

    node_class = AccelerationNormalHighLimitWithFlapsDown

    def setUp(self):
        self.mtow = A('Maximum Takeoff Weight', value=82190.0)
        self.mlw = A('Maximum Landing Weight', value=69308.0)
        self.mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}

    def test_can_operate(self):
        node = self.node_class()
        expected =('Flap Lever', 'Gross Weight Smoothed',
                   'Maximum Takeoff Weight', 'Maximum Landing Weight')
        b737_max = A('Family', value='B737 MAX')
        self.assertTrue(node.can_operate(expected, b737_max))

        expected =('Flap Lever (Synthetic)', 'Gross Weight Smoothed',
                   'Maximum Takeoff Weight', 'Maximum Landing Weight')
        self.assertTrue(node.can_operate(expected, b737_max))

        self.assertFalse(node.can_operate(expected))
        self.assertFalse(node.can_operate(('Flap Lever', 'Gross Weight Smoothed'), b737_max))

    def test_derive(self):

        flap = M('Flap Lever', array=np.ma.repeat(5, 20),
                 values_mapping=self.mapping, frequency=4)
        gw = P('Gross Weight Smoothed', array=np.ma.repeat(55000, 20/2.),
               frequency=2)

        node = self.node_class()
        node.get_derived((flap, None, gw, self.mtow, self.mlw))

        self.assertEqual(len(node.array), 20)
        ma_test.assert_array_equal(node.array, np.ma.repeat(2.0, 20))

    def test_derive_over_MLW_flaps_less_than_30(self):
        flap = M('Flap Lever', array=np.ma.repeat(5, 20),
                 values_mapping=self.mapping, frequency=4)
        gw = P('Gross Weight Smoothed', array=np.ma.repeat(75000, 20/2),
               frequency=2)

        node = self.node_class()
        node.get_derived((flap, None, gw, self.mtow, self.mlw))

        self.assertEqual(len(node.array), 20)
        ma_test.assert_array_equal(node.array, np.ma.repeat(2.0, 20))

    def test_derive_below_MLW_flaps_30(self):
        flap = M('Flap Lever', array=np.ma.repeat(30, 20),
                 values_mapping=self.mapping, frequency=4)
        gw = P('Gross Weight Smoothed', array=np.ma.repeat(65000, 20/2.),
               frequency=2)

        node = self.node_class()
        node.get_derived((flap, None, gw, self.mtow, self.mlw))

        self.assertEqual(len(node.array), 20)
        ma_test.assert_array_equal(node.array, np.ma.repeat(2.0, 20))

    def test_derive_over_MLW_flaps_30(self):
        flap = M('Flap Lever', array=np.ma.repeat(30, 20),
                 values_mapping=self.mapping, frequency=4)
        gw = P('Gross Weight Smoothed', array=np.ma.repeat(75000, 20/2.),
               frequency=2)
        mtow = self.mtow.value
        mlw = self.mlw.value
        # Find expected g by linear interpolation
        expected_g = (75000 - mlw) / (mtow - mlw) * (1.5 - 2.0) + 2.0
        # When Gross Weight Smoothed is aligned to Flap, its last value
        # will be masked, because we do not extrapolate values.
        # This means that the last expected value will have the default
        # value of 2.0g.
        expected = np.ma.concatenate((np.ma.repeat(expected_g, 19), [2.0]))

        node = self.node_class()
        node.get_derived((flap, None, gw, self.mtow, self.mlw))

        self.assertEqual(len(node.array), 20)
        assert_array_almost_equal(node.array, expected)

    def test_derive_over_MTOW_flaps_40(self):
        flap = M('Flap Lever', array=np.ma.repeat(40, 20),
                 values_mapping=self.mapping, frequency=4)
        gw = P('Gross Weight Smoothed', array=np.ma.repeat(82200, 20/2.),
               frequency=2)
        # Above MTOW, with flaps 30 or above, the max acceleration is 1.5g
        expected_g = 1.5
        # When Gross Weight Smoothed is aligned to Flap, its last value
        # will be masked, because we do not extrapolate values.
        # This means that the last expected value will have the default
        # value of 2.0g.
        expected = np.ma.concatenate((np.ma.repeat(expected_g, 19), [2.0]))

        node = self.node_class()
        node.get_derived((flap, None, gw, self.mtow, self.mlw))

        self.assertEqual(len(node.array), 20)
        assert_array_almost_equal(node.array, expected)

    def test_derive_flaps_up(self):
        flap = M('Flap Lever', array=np.ma.repeat(0, 20),
                 values_mapping=self.mapping, frequency=4)
        gw = P('Gross Weight Smoothed', array=np.ma.repeat(62200, 20/2.),
               frequency=2)

        node = self.node_class()
        node.get_derived((flap, None, gw, self.mtow, self.mlw))

        self.assertTrue(node.array.mask.all())


class TestAirspeedSelectedForApproaches(unittest.TestCase):

    node_class = AirspeedSelectedForApproaches

    def test_derive(self):
        airspd = P('Airspeed Selected', array=np.ma.arange(10), frequency=1 / 64.)
        fast = buildsection('Fast', 0, 9)
        p = self.node_class(frequency=1 / 64.)
        p.derive(airspd, fast)
        self.assertEqual(len(p.array), 640)
        self.assertTrue(p.array[1], 1)

    def test_derive__superframe_change_after_fast(self):
        '''
        test to check that Airspeed Selected being reset to 100 after runway
        turnoff when recorded 1/64 does not affect the values used in approach
        '''
        speeds = np.ma.array([158, 250, 300, 278, 320, 240, 175, 135, 100])
        airspd = P('Airspeed Selected', array=speeds.repeat(10), frequency=1 / 64.)
        fast = buildsection('Fast', 5, 79.5)
        fast.frequency = 1 / 64.
        node = self.node_class(frequency=1 / 64.)
        node.derive(airspd, fast)
        self.assertEqual(len(node.array), 5760)
        final_approach_values = node.array[4992:5184]
        self.assertEqual(len(np.unique(final_approach_values)), 1)
        self.assertEqual(np.unique(final_approach_values)[0], 135)


class TestAirspeedSelected(unittest.TestCase):
    def test_can_operate(self):
        opts = AirspeedSelected.get_operational_combinations()
        self.assertIn(('Airspeed Selected (1)', 'Airspeed Selected (2)', 'Airspeed Selected (3)', 'Airspeed Selected (4)'), opts)
        self.assertIn(('Airspeed Selected (L)', 'Airspeed Selected (R)'), opts)
        self.assertIn(('Airspeed Selected (L)', 'Airspeed Selected (R)', 'Airspeed Selected (MCP)'), opts)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestAirspeedTrue(unittest.TestCase):
    def test_can_operate(self):
        self.assertIn(('Airspeed', 'Altitude STD Smoothed'), AirspeedTrue.get_operational_combinations())
        self.assertIn(('Airspeed', 'Altitude STD Smoothed', 'SAT',
                       'Takeoff', 'Landing', 'Rejected Takeoff',
                       'Groundspeed', 'Acceleration Forwards'),
                      AirspeedTrue.get_operational_combinations())

    def test_tas_basic(self):
        cas = P('Airspeed', np.ma.array([100, 200, 300]))
        alt = P('Altitude STD Smoothed', np.ma.array([0, 20000, 40000]))
        sat = P('SAT', np.ma.array([20, -10, -55]))
        tas = AirspeedTrue()
        tas.derive(cas, alt, sat)
        result = [100.864, 278.375, 555.595]
        self.assertLess(abs(tas.array.data[0] - result[0]), 0.01)
        self.assertLess(abs(tas.array.data[1] - result[1]), 0.01)
        self.assertLess(abs(tas.array.data[2] - result[2]), 0.01)

    def test_tas_masks(self):
        cas = P('Airspeed', np.ma.array([100, 200, 300]))
        alt = P('Altitude STD Smoothed', np.ma.array([0, 20000, 40000]))
        tat = P('TAT', np.ma.array([20, -10, -40]))
        tas = AirspeedTrue()
        cas.array[0] = np.ma.masked
        alt.array[1] = np.ma.masked
        tat.array[2] = np.ma.masked
        tas.derive(cas, alt, tat)
        self.assertTrue(tas.array.mask.all())

    def test_tas_no_tat(self):
        cas = P('Airspeed', np.ma.array([100, 200, 300]))
        alt = P('Altitude STD Smoothed', np.ma.array([0, 10000, 20000]))
        tas = AirspeedTrue()
        tas.derive(cas, alt, None)
        result = [100.000, 231.575, 400.097]
        self.assertLess(abs(tas.array.data[0] - result[0]), 0.01)
        self.assertLess(abs(tas.array.data[1] - result[1]), 0.01)
        self.assertLess(abs(tas.array.data[2] - result[2]), 0.01)

    def test_tas_with_gs_extensions(self):
        # With zero wind, the tas and gs will be identical at the ends of the data.
        accel = np.array([0.0, 4.7, 9.5, 14.3, 19.0, 23.8, 28.6, 33.3, 38.1, 42.9,
                          47.6, 52.4, 57.1, 61.9, 66.7, 71.4, 76.2, 81.0, 85.7, 90.5])
        cas = P('Airspeed', np.ma.concatenate((np.zeros(9), accel[9:], accel[-1:8:-1], np.zeros(9))))
        gspd = P('Groundspeed', np.ma.concatenate((accel, accel[::-1])))
        gspd.array[0:3] = [12.0, 12.0, 12.0]
        gspd.array[-4:] = [14.0, 14.0, 14.0, 14.0]
        alt = P('Altitude STD Smoothed', np.ma.zeros(40))
        acc = P('Acceleration Forwards', np.ma.concatenate((np.ones(20) * 0.25, np.ones(20) * -0.25)))
        toffs = buildsection('Takeoff', 0, 18)
        lands = buildsection('Landing', 21, None)
        tas = AirspeedTrue()
        tas.derive(cas, alt, None, toffs, lands, None, gspd, acc)
        expected = np.ma.concatenate((accel, accel[::-1]))
        assert_array_almost_equal(tas.array[3:-4], expected[3:-4], decimal=1)
        assert_array_almost_equal(tas.array[:3], np.ones(3) * 12, decimal=1)
        assert_array_almost_equal(tas.array[-4:], np.ones(4) * 14, decimal=1)
        # Curiously, the test above only checks the valid samples, so no
        # extrapolation is needed to pass, hence a check on validity is
        # essential !
        self.assertEqual(np.ma.count(tas.array), 40)

    def test_tas_no_gs_extensions(self):
        # With no groundspeed available, the true airspeed is an integration
        # of acceleration from the ends of the available data. The array
        # "speed" corresponds to a 0.25g acceleration and deceleration.
        speed = np.array([0.0, 4.7, 9.5, 14.3, 19.0, 23.8, 28.6, 33.3, 38.1, 42.9,
                          47.6, 52.4, 57.1, 61.9, 66.7, 71.4, 76.2, 81.0, 85.7, 90.5])
        cas = P('Airspeed', np.ma.concatenate((np.zeros(9), speed[9:], speed[-1:8:-1], np.zeros(9))))
        alt = P('Altitude STD Smoothed', np.ma.zeros(40))
        acc = P('Acceleration Forwards', np.ma.concatenate((np.ones(20) * 0.25, np.ones(20) * -0.25)))
        toffs = buildsection('Takeoff', 0, 18)
        lands = buildsection('Landing', 21, None)
        tas = AirspeedTrue()
        tas.derive(cas, alt, None, toffs, lands, None, None, acc)
        expected = np.ma.concatenate((speed, speed[::-1]))
        assert_array_almost_equal(tas.array, expected, decimal=1)
        # Curiously, the test above only checks the valid samples, so no
        # extrapolation is needed to pass, hence a check on validity is
        # essential !
        self.assertEqual(np.ma.count(tas.array), 40)

    def test_tas_rto(self):
        speed = np.array([0.0, 4.7, 9.5, 14.3, 19.0, 23.8, 28.6, 33.3, 38.1, 42.9,
                          47.6, 52.4, 57.1, 61.9, 66.7, 71.4, 76.2, 81.0, 85.7, 90.5])
        cas = P('Airspeed', np.ma.concatenate((np.zeros(9), speed[9:], speed[-1:8:-1], np.zeros(9))))
        alt = P('Altitude STD Smoothed', np.ma.zeros(40))
        acc = P('Acceleration Forwards', np.ma.concatenate((np.ones(20) * 0.25, np.ones(20) * -0.25)))
        rtos = buildsection('Rejected Takeoff', 1, 38)
        tas = AirspeedTrue()
        tas.derive(cas, alt, None, None, None, rtos, None, acc)
        expected = np.ma.concatenate((speed, speed[::-1]))
        assert_array_almost_equal(tas.array, expected, decimal=1)


class TestAltitudeSTDSmoothed(unittest.TestCase):
    def test_can_operate(self):
        opts = AltitudeSTDSmoothed.get_operational_combinations()
        self.assertTrue(('Altitude STD', 'Frame',) in opts)
        self.assertTrue(('Altitude STD (Fine)', 'Altitude STD', 'Frame') in opts)
        self.assertTrue(('Altitude STD (Capt)', 'Altitude STD (FO)', 'Frame') in opts)

    def test_derive_atr_42(self):
        frame = Attribute('Frame', 'ATR42_V2_Quad')
        alt = load(os.path.join(test_data_path, 'AltitudeSTDSmoothed_alt.nod'))
        node = AltitudeSTDSmoothed()
        node.derive(None, alt, None, None, frame)
        # np.ma.sum(np.ma.abs(np.ma.diff(alt.array[4800:5000]))) == 2000
        linear_diff = np.ma.sum(np.ma.abs(np.ma.diff(node.array[4800:5000])))
        self.assertTrue(linear_diff < 200)
        max_diff = np.ma.max(np.ma.abs(alt.array - node.array))
        self.assertTrue(max_diff < 100)

    def test_derive_747_200(self):
        frame = Attribute('Frame', '747-200-xxx')
        alt_f = P('Altitude STD (Fine)', array=[4000,4999,1000,2000])
        alt_c = P('Altitude STD', array=[-900,-800,1000,2000])
        node = AltitudeSTDSmoothed()
        node.derive(alt_f, alt_c, None, None, frame)
        self.assertTrue(int(node.array[0]) == 333)
        self.assertTrue(int(node.array[-1]) == 666)


class TestAltitudeAAL(unittest.TestCase):
    def test_can_operate(self):
        opts = AltitudeAAL.get_operational_combinations()
        self.assertTrue(('Altitude STD Smoothed', 'Fast') in opts)
        self.assertTrue(('Altitude Radio Offset Removed', 'Altitude STD Smoothed', 'Fast') in opts)

    def test_alt_aal_basic(self):
        data = np.ma.array([-3, 0, 30, 80, 250,] + [560]*10 + [560, 220, 70, 20, -5])
        alt_std = P(array=data + 300)
        alt_rad = P(array=data)
        fast_data = np.ma.ones(20) * 100
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad,alt_std, phase_fast)
        expected = np.ma.array([0, 0, 30, 80, 250,] + [560]*10 + [560, 220, 70, 20, 0])
        assert_array_equal(expected, alt_aal.array.data)

    def test_alt_aal_bounce_rejection(self):
        data = np.ma.array([-3, 0, 30, 80, 250, 560, 220, 70, 20, -5, 2, 2, 2,
                            -3, -3])
        alt_std = P(array=data + 300)
        alt_rad = P(array=data)
        fast_data = np.ma.ones(15) * 100
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, phase_fast)
        expected = np.ma.array([0, 0, 30, 80, 250, 560, 220, 70, 20, 0, 0, 0, 0,
                                0, 0])
        assert_array_equal(expected, alt_aal.array.data)

    def test_alt_aal_no_ralt(self):
        data = np.ma.array([-3, 0, 30, 80, 250,] + [580]*10 + [580, 220, 70, 20, 25])
        alt_std = P(array=data + 300)
        slow_and_fast_data = np.ma.array([70] + [85] * 17 + [70]*2)
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', slow_and_fast_data))
        pitch = P('Pitch', np_ma_ones_like(slow_and_fast_data))
        alt_aal = AltitudeAAL()
        alt_aal.derive(None, alt_std, phase_fast, pitch)
        expected = np.ma.array([0, 0, 30, 80, 250,] + [560]*10 + [560, 200, 50, 0, 0])
        assert_array_equal(expected, alt_aal.array.data)

    def test_alt_aal_complex(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -3000 + \
            np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 7996
        rad_wave = np.copy(testwave)
        rad_wave[110:140] -= 8765 # The ground is 8,765 ft high at this point.
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        phase_fast = buildsection('Fast', 0, len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', rad_data),
                       P('Altitude STD Smoothed', testwave),
                       phase_fast)
        '''
        import matplotlib.pyplot as plt
        plt.plot(testwave)
        plt.plot(rad_data)
        plt.plot(alt_aal.array)
        plt.show()
        '''
        # Check that the waveform reaches the right points.
        np.testing.assert_equal(alt_aal.array[0], 0.0)
        np.testing.assert_almost_equal(alt_aal.array[34], 7013, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[60], 3308, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[124], 217, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[191], 8965, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[254], 3288, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[313], 17, decimal=0)

    def test_alt_aal_complex_no_ralt_flying_below_takeoff_airfield(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -2000 + \
            np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * 5000 + 0
        phase_fast = buildsection('Fast', 0, len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(None,
                       P('Altitude STD Smoothed', testwave),
                       phase_fast,
                       P('Pitch', np_ma_ones_like(testwave))
                       )
        '''
        import matplotlib.pyplot as plt
        plt.plot(testwave, '-b')
        plt.plot(alt_aal.array, '-r')
        plt.show()
        '''

    def test_alt_aal_complex_with_mask(self):
        #testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -3000 + \
            #np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 7996

        # Slope of np.ma.arange(0,5000, 50) reduced to ensure at least one
        # sample point fell in the range 0-100ft for the alt_rad logic to
        # work. DJ.
        std_wave = np.ma.concatenate([np.zeros(50),
                                      np.arange(0, 5000, 50),
                                      np.zeros(50) + 5000,
                                      np.arange(5000,5500, 200),
                                      np.zeros(50) + 5500,
                                      np.arange(5500, 0, -500),
                                      np.zeros(50)])
        rad_wave = np.copy(std_wave) - 8
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        phase_fast = buildsection('Fast', 35, len(std_wave))
        std_wave += 1000
        rad_data[42:48] = np.ma.masked
        alt_aal = AltitudeAAL()
        alt_aal.derive(
            P('Altitude Radio', np.ma.copy(rad_data)),
            P('Altitude STD Smoothed', np.ma.copy(std_wave)),
            phase_fast,
            P('Pitch', np_ma_ones_like(rad_data)))
        '''
        import matplotlib.pyplot as plt
        plt.plot(std_wave, '-b')
        plt.plot(rad_data, 'o-r')
        plt.plot(alt_aal.array, '-k')
        plt.show()
        '''
        #  Check alt aal does not try to jump to alt std in masked period of
        #  alt rad
        self.assertEqual(alt_aal.array[45], 0)  # NOT 1000!

    def test_alt_aal_complex_doubled(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 5500
        rad_wave = np.copy(testwave)-500
        #rad_wave[110:140] -= 8765 # The ground is 8,765 ft high at this point.
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        double_test = np.ma.concatenate((testwave, testwave))
        double_rad = np.ma.concatenate((rad_data, rad_data))
        phase_fast = buildsection('Fast', 0, 2*len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', double_rad),
                       P('Altitude STD Smoothed', double_test),
                       phase_fast)
        '''
        import matplotlib.pyplot as plt
        plt.plot(double_test, '-b')
        plt.plot(double_rad, 'o-r')
        plt.plot(alt_aal.array, '-k')
        plt.show()
        '''
        self.assertNotEqual(alt_aal.array[200], 0.0)
        np.testing.assert_equal(alt_aal.array[0], 0.0)

    def test_alt_aal_complex_doubled_with_touch_and_go(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 5000
        rad_wave = np.copy(testwave)-500
        #rad_wave[110:140] -= 8765 # The ground is 8,765 ft high at this point.
        rad_data = np.ma.masked_greater(rad_wave, 2600)
        double_test = np.ma.concatenate((testwave, testwave))
        double_rad = np.ma.concatenate((rad_data, rad_data))
        phase_fast = buildsection('Fast', 0, 2*len(testwave))
        alt_aal = AltitudeAAL()
        alt_aal.derive(P('Altitude Radio', double_rad),
                       P('Altitude STD Smoothed', double_test),
                       phase_fast)
        '''
        import matplotlib.pyplot as plt
        plt.plot(double_test, '-b')
        plt.plot(double_rad, 'o-r')
        plt.plot(alt_aal.array, '-k')
        plt.show()
        '''
        np.testing.assert_equal(alt_aal.array[300:310], [0.0]*10)


    def test_alt_aal_complex_no_rad_alt(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2 * 5, 0.1)) * -3000 + \
            np.ma.cos(np.arange(0, 3.14 * 2, 0.02)) * -5000 + 7996
        testwave[255:]=testwave[254]
        testwave[:5]=500.0
        phase_fast = buildsection('Fast', 0, 254)
        alt_aal = AltitudeAAL()
        alt_aal.derive(None,
                       P('Altitude STD Smoothed', testwave),
                       phase_fast,
                       P('Pitch', np_ma_ones_like(testwave))
                       )
        '''
        import matplotlib.pyplot as plt
        plt.plot(testwave)
        plt.plot(alt_aal.array)
        plt.show()
        '''
        np.testing.assert_equal(alt_aal.array[0], 0.0)
        np.testing.assert_almost_equal(alt_aal.array[34], 6620, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[60], 2915, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[124], 8594, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[191], 8594, decimal=0)
        np.testing.assert_almost_equal(alt_aal.array[254], 0, decimal=0)

    def test_alt_aal_no_rad_alt_pitch_inclusion(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 2.7, 0.3)) * -280 + 380
        join = 20
        x = testwave[join]
        n = len(testwave) - join
        testwave[join:] = np.linspace(x, x-80, n)
        phase_fast = buildsection('Fast', 0, 28)
        pch = np_ma_ones_like(testwave)
        pch[23:27]=2.0
        alt_aal = AltitudeAAL()
        alt_aal.derive(None,
                       P('Altitude STD Smoothed', testwave),
                       phase_fast,
                       P('Pitch', pch)
                       )

        '''
        import matplotlib.pyplot as plt
        plt.plot(testwave)
        plt.plot(alt_aal.array)
        plt.show()
        '''
        # Check that AAL goes to zero at the last pitch high point and stays there.
        self.assertGreater(alt_aal.array[25], 0.0)
        self.assertEqual(alt_aal.array[26], 0.0)
        self.assertEqual(alt_aal.array[27], 0.0)

    def test_alt_aal_misleading_rad_alt(self):
        # Spurious Altitude Radio data when Altitude STD was above 30000 ft was
        # causing Altitude AAL to be shifted down to -30000 ft.
        alt_std = load(os.path.join(
            test_data_path, 'AltitudeAAL_AltitudeSTDSmoothed.nod'))
        alt_rad = load(os.path.join(
            test_data_path, 'AltitudeAAL_AltitudeRadio.nod'))
        fast = load(os.path.join(
            test_data_path, 'AltitudeAAL_Fast.nod'))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, fast)
        self.assertEqual(np.ma.min(alt_aal.array), 0.0)

    def test_alt_aal__heli__missing_rad_alt(self):
        '''
        Helicopter missing Altitude Radio, ensure 0 ft on ground even if alt
        std records positive.
        '''

        first_climb = np.ma.arange(0, 3000, 10)
        second_climb = np.ma.arange(20, 3000, 10)

        alt_array = np.ma.zeros(1350)
        alt_array[50:350] = first_climb
        alt_array[350:648] = second_climb[::-1]
        alt_array[648:702] = np.ma.ones(54) * 20 # Alt std indicating 20ft on ground
        alt_array[702:1000] = second_climb
        alt_array[1000:1300] = first_climb[::-1]
        alt_std = P('Alttitude STD Smoothed', array=alt_array)
        fast = buildsection('Fast', 50, 1300)
        gog_array = np.ma.ones(1350)
        gog_array[50:648] = 0
        gog_array[702:1300] = 0
        gog = M(name='Gear On Ground', array=gog_array, values_mapping={0: 'Air', 1: 'Ground'})

        alt_aal = AltitudeAAL()
        alt_aal.derive(None, alt_std, fast, None, gog, helicopter)
        self.assertEqual(alt_aal.array[670], 0)

    @unittest.skip('Test Not Implemented')
    def test_alt_aal_faulty_alt_rad(self):
        '''
        When 'Altitude Radio' does not reach 0 after touchdown due to an arinc
        signal being recorded, 'Altitude AAL' did not fill the second half of
        its array. Since the array is initialised as zeroes
        '''
        hdf_copy = copy_file(os.path.join(test_data_path,
                                          'alt_aal_faulty_alt_rad.hdf5'),
                             postfix='_test_copy')
        process_flight(hdf_copy, 'G-DEMA', {
            'Engine Count': 2,
            'Frame': '737-3C', # TODO: Change.
            'Manufacturer': 'Boeing',
            'Model': 'B737-86N',
            'Precise Positioning': True,
            'Series': 'B767-300',
        })
        with hdf_file(hdf_copy) as hdf:
            hdf['Altitude AAL']
            self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_alt_aal_without_alt_rad(self):
        '''
        When 'Altitude Radio' is not available, 'Altitude AAL' is created from
        'Altitude STD' using the cycle_finder and peak_curvature algorithms.
        Currently, cycle_finder is accurately locating the index where the
        aircraft begins to climb. This section of data is passed into
        peak_curvature, which is designed to find the first curve in a piece of
        data. The problem is that data from before the first curve, where the
        aircraft starts climbing, is not included, and peak_curvature detects
        the second curve at approximately 120 feet.
        '''
        hdf_copy = copy_file(os.path.join(test_data_path,
                                          'alt_aal_without_alt_rad.hdf5'),
                             postfix='_test_copy')
        process_flight(hdf_copy, 'G-DEMA', {
            'Engine Count': 2,
            'Frame': '737-3C', # TODO: Change.
            'Manufacturer': 'Boeing',
            'Model': 'B737-86N',
            'Precise Positioning': True,
            'Series': 'B767-300',
        })
        with hdf_file(hdf_copy) as hdf:
            hdf['Altitude AAL']
            self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_alt_aal_training_flight(self):
        alt_std = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-training-alt_std.nod'))
        alt_rad = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-training-alt_rad.nod'))
        fasts = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-training-fast.nod'))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, fasts)
        peak_detect = np.ma.masked_where(alt_aal.array < 500, alt_aal.array)
        peaks = np.ma.clump_unmasked(peak_detect)
        # Check to test that all 6 altitude sections are inculded in alt aal
        self.assertEqual(len(peaks), 6)

    @unittest.skip('Test Not Implemented')
    def test_alt_aal_goaround_flight(self):
        alt_std = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-goaround-alt_std.nod'))
        alt_rad = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-goaround-alt_rad.nod'))
        fasts = load(os.path.join(test_data_path,
                                    'TestAltitudeAAL-goaround-fast.nod'))
        alt_aal = AltitudeAAL()
        alt_aal.derive(alt_rad, alt_std, fasts)
        difs = np.diff(alt_aal.array)
        index, value = max_value(np.abs(difs))
        # Check to test that the step occurs during cruse and not the go-around
        self.assertTrue(1290 <= index < 1850)

    def test_find_liftoff_start_on_herc(self):
        # Herc (L100) climbs in a straight line without noticable concave
        # curvature at liftoff; ensure index is kept close
        aal = AltitudeAAL(frequency=2)
        herc_alt_std = np.ma.array([
       -143.20034375,    0.        , -142.46179687,    0.        ,
       -140.98470312,    0.        , -140.24615625,    0.        ,
       -138.7690625 ,    0.        , -138.03051562,    0.        ,
       -135.814875  ,    0.        , -134.33778125,    0.        ,
       -132.8606875 ,    0.        , -132.8606875 ,    0.        ,
       -130.64504687,    0.        , -129.9065    ,    0.        ,
       -128.42940625,    0.        , -127.69085937,    0.        ,
       -125.47521875,    0.        , -123.25957812,    0.        ,
       -121.0439375 ,    0.        , -118.82829687,    0.        ,
       -116.61265625,    0.        , -114.39701562,    0.        ,
       -110.70428125,    0.        , -108.48864062,    0.        ,
       -106.273     ,    0.        , -102.58026562,    0.        ,
        -99.62607812,    0.        ,  -97.4104375 ,    0.        ,
        -92.97915625,    0.        ,  -89.28642187,    0.        ,
        -84.85514062,    0.        ,  -81.16240625,    0.        ,
        -78.20821875,    0.        , -901.50908125, -881.86373437,
       -862.2183875 , -838.95416094, -815.68993437, -792.05643437,
       -768.42293437, -749.14686094, -729.8707875 , -706.60656094,
       -683.34233437, -664.06626094, -644.7901875 , -617.16853437,
       -589.54688125, -565.54410781, -541.54133437, -526.6226875 ,
       -511.70404062, -488.8090875 , -465.91413437, -450.9954875 ,
       -436.07684062, -421.89674062, -407.71664062, -389.17911406,
       -370.6415875 , -347.37736094, -324.11313437, -304.83706094,
       -285.5609875 , -262.29676094, -239.03253437, -224.1138875 ,
       -209.19524062, -186.3002875 , -163.40533437, -144.12926094,
       -124.8531875 , -110.30381406,  -95.75444062,  -81.57434062,
        -67.39424062,  -53.21414062,  -39.03404062,  -24.85394062,
        -10.67384062,    3.50625938,   17.68635938,   27.50903281,
         37.33170625,   51.14253281,   64.95335938,   79.13345938,
         93.31355938,  107.49365938,  121.67375938,  127.13900625,
        132.60425313,  150.40323281,  168.2022125 ,  182.75158594,
        197.30095938,  211.48105938,  225.66115938,  239.84125938,
        254.02135938,  268.20145938,  282.38155938,  296.56165938,
        310.74175938,  324.92185938,  339.10195938,  353.28205938,
        367.46215938,  381.64225938,  395.82235938,  414.35988594,
        432.8974125 ,  451.8042125 ,  470.7110125 ,  485.26038594,
        499.80975938,  513.98985938,  528.16995938,  546.70748594,
        565.2450125 ,  579.79438594,  594.34375938,  608.52385938,
        622.70395938,  645.5989125 ,  668.49386563,  687.76993906,
        707.0460125 ,  721.59538594,  736.14475938,  759.0397125 ])
        herc_alt_std[:62] = np.ma.masked
        idx = aal.find_liftoff_start(herc_alt_std)
        self.assertEqual(idx, 63)

    def test_alt_aal_helicopter_no_rad_alt(self):
        '''
        real heli example with no rad alt
        '''
        alt_std = load(os.path.join(
            test_data_path, 'AltitudeAAL__S76__AltitudeSTDSmoothed.nod'))
        pitch = load(os.path.join(
            test_data_path, 'AltitudeAAL__S76__Pitch.nod'))
        gog = load(os.path.join(
            test_data_path, 'AltitudeAAL__S76__GearOnGround.nod'))
        fast = buildsection('Fast', 832, 10750)
        alt_aal = AltitudeAAL()
        alt_aal.derive(None, alt_std, fast, pitch, gog, helicopter)


class TestAimingPointRange(unittest.TestCase):
    def test_basic_scaling(self):
        approaches = App(items=[ApproachItem(
            'Landing', slice(3, 8),
            landing_runway={'end':
                    {'elevation': 3294,
                     'latitude': 31.497511,
                     'longitude': 65.833933},
                    'start':
                    {'elevation': 3320,
                     'latitude': 31.513997,
                     'longitude': 65.861714}})])
        app_rng=P('Approach Range',
                  array=np.ma.arange(10000.0, -2000.0, -1000.0))
        apr = AimingPointRange()
        apr.derive(app_rng, approaches)
        # convoluted way to check masked outside slice !
        self.assertEqual(apr.array[0].mask, np.ma.masked.mask)
        self.assertAlmostEqual(apr.array[4], 1.67, places=2)


class TestAltitudeAALForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL',)]
        opts = AltitudeAALForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_altitude_AAL_for_flight_phases_basic(self):
        alt_4_ph = AltitudeAALForFlightPhases()
        alt_4_ph.derive(Parameter('Altitude AAL',
                                  np.ma.array(data=[-1,100,200,100,0],
                                              mask=[0,0,1,1,0])))
        expected = np.ma.array(data=[0,100,66,33,0],mask=False)
        # ...because data interpolates across the masked values and integer
        # values are rounded.
        assert_array_equal(alt_4_ph.array, expected)


'''
class TestAltitudeForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)]
        opts = AltitudeForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_altitude_for_phases_repair(self):
        alt_4_ph = AltitudeForFlightPhases()
        raw_data = np.ma.array([0,1,2])
        raw_data[1] = np.ma.masked
        alt_4_ph.derive(Parameter('Altitude AAL', raw_data, 1,0.0))
        expected = np.ma.array([0,1,2], mask=False)
        assert_array_equal(alt_4_ph.array, expected)

    def test_altitude_AAL_for_flight_phases_reject_ground(self):
        alt_4_ph = AltitudeAALForFlightPhases()
        testwave = np.abs(np.sin(np.arange(0,6.35,0.1))*200)
        alt_AAL = P('Altitude AAL', np.ma.array(testwave), 1,0.0)
        gog = M('Gear On Ground', array=(['Air']*32 + ['Ground']*32),
                values_mapping = {0: 'Air', 1: 'Ground'})
        alt_4_ph.derive(alt_AAL, gog)
        answer = np.ma.array(list(testwave[:32])+[0.0]*32)
        np.testing.assert_array_almost_equal(alt_4_ph.array, answer)
'''

class TestBaroCorrection(unittest.TestCase):
    def test_can_operate(self):
        opts = BaroCorrection.get_operational_combinations()
        self.assertTrue(
            BaroCorrection.can_operate(('Altitude Baro (1)', 'Altitude STD',))
        )
        self.assertTrue(
            BaroCorrection.can_operate(('Altitude Baro', 'Altitude STD',))
        )
        self.assertTrue(
            BaroCorrection.can_operate(('Baro Correction (Capt)', 'Baro Correction (FO)',))
        )

    def test_derive_from_baro(self):
        baro_cpt = P('Baro Correction (Capt)', [998, 998, 1013, 1013, 1013],
                     frequency=1/64.0, offset=10.0)

        baro_fo = P('Baro Correction (FO)',       [998, 998, 1013, 1013, 1013],
                    frequency=1/64.0, offset=42.0)
        baro = BaroCorrection()
        res = baro.get_derived((baro_cpt, baro_fo, None, None, None))
        self.assertEqual(baro.hz, 2 / 64.0)
        self.assertEqual(baro.offset, 10.0)
        expected = np.ma.array([998.0, 998.0, 998.0, 1005.5, 1009.25, 1009.25,
                                1013.0, 1013.0, 1013.0, 0.0])
        expected[-1] = np.ma.masked
        assert_array_equal(baro.array, expected)

    def test_derive_from_baro_with_alt_baro_available(self):
        baro_cpt = P('Baro Correction (Capt)', [998, 998, 1013, 1013, 1013],
                     frequency=1/4.0, offset=1.0)

        baro_fo = P('Baro Correction (FO)',       [998, 998, 1013, 1013, 1013],
                    frequency=1/4.0, offset=3.0)
        alt_baro = P('Altitude Baro (1)', [1000, 1000],
                     frequency=1/64.0, offset=1.0)
        alt_std = P('Altitude STD', [1000] * 5 * 4,
                    frequency=1.0, offset=0.0)
        baro = BaroCorrection()
        res = baro.get_derived((baro_cpt, baro_fo, alt_baro, None, alt_std))
        self.assertEqual(baro.hz, 2 / 4.0)
        self.assertEqual(baro.offset, 1.0)
        expected = np.ma.array([998.0, 998.0, 998.0, 1005.5, 1009.25, 1009.25,
                                1013.0, 1013.0, 1013.0, 0.0])
        expected[-1] = np.ma.masked
        assert_array_equal(baro.array, expected)

    def test_derive_from_alt_baro(self):
        alt_diff = 28 * 15
        alt_baro = P('Altitude Baro (1)', [1000 - alt_diff] * 2 + [1000] * 3,
                     frequency=1/64.0, offset=10.0)

        alt_std = P('Altitude STD', [1000] * 5 * 64,
                    frequency=1.0, offset=0.0)
        baro = BaroCorrection()
        res = baro.get_derived((None, None, alt_baro, None, alt_std))
        self.assertEqual(baro.hz, 1 / 64.0)
        self.assertEqual(baro.offset, 10.0)
        assert_array_equal(
            baro.array,
            [998, 998, 1013, 1013, 1013]
        )

    def test_derive_from_alt_baro_solo(self):
        alt_diff = 28 * 15
        alt_baro = P('Altitude Baro', [1000 - alt_diff] * 2 + [1000] * 3,
                     frequency=1/64.0, offset=10.0)

        alt_std = P('Altitude STD', [1000] * 5 * 64,
                    frequency=1.0, offset=0.0)
        baro = BaroCorrection()
        res = baro.get_derived((None, None, None, alt_baro, alt_std))
        self.assertEqual(baro.hz, 1 / 64.0)
        self.assertEqual(baro.offset, 10.0)
        assert_array_equal(
            baro.array,
            [998, 998, 1013, 1013, 1013]
        )


class TestAltitudeQNH(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AltitudeQNH
        self.operational_combinations = [('Altitude STD', 'Baro Correction')]

    def test_attribute(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Altitude QNH')
        self.assertEqual(node.units, ut.FT)

    def test_derive(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction', np.ma.arange(1000,1025, dtype=np.float))

        node = self.node_class()
        node.derive(alt_std, baro, None, None, None, None)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            9911, 9938, 9965, 9993, 10020, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))

    def test_baro_setting(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction', np.ma.arange(1000,1025, dtype=np.float))
        values_mapping = {0: 'ALT QFE', 1: 'ALT QNH', 2: 'ALT STD'}
        baro_sel = M(
            'Baro Setting Selection',
            array = np.ma.array([1] * 10 + [2] * 5 + [1] * 10),
            values_mapping=values_mapping
        )

        node = self.node_class()
        node.derive(alt_std, baro, baro_sel, None, None, None)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            10000, 10000, 10000, 10000, 10000, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))

    def test_baro_cpt_fo_setting(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction', np.ma.arange(1000,1025, dtype=np.float))
        values_mapping = {0: 'QFE', 1: 'QNH', 2: 'STD', 3: 'Not Used'}
        baro_sel_cpt = M(
            'Baro Setting Selection (Capt)',
            array = np.ma.array([1] * 10 + [2] * 5 + [1] * 10),
            values_mapping=values_mapping
        )
        baro_sel_fo = M(
            'Baro Setting Selection (FO)',
            array = np.ma.array([1] * 10 + [2] * 5 + [1] * 10),
            values_mapping=values_mapping
        )

        node = self.node_class()
        node.derive(alt_std, baro, None, baro_sel_cpt, baro_sel_fo, None)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            10000, 10000, 10000, 10000, 10000, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))

    def test_baro_correction_isis(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction', np.ma.arange(1000,1025, dtype=np.float))
        baro_cor_isis = P(
            'Baro Correction (ISIS)',
            array = np.ma.concatenate((np.ones(10) * 1001, np.ones(5) * 1013, np.ones(10) * 1024)),
        )

        node = self.node_class()
        node.derive(alt_std, baro, None, None, None, baro_cor_isis)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            10000, 10000, 10000, 10000, 10000, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))


class TestAltitudeQNHCapt(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AltitudeQNHCapt
        self.operational_combinations = [('Altitude STD', 'Baro Correction (Capt)')]

    def test_attribute(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Altitude QNH (Capt)')
        self.assertEqual(node.units, ut.FT)

    def test_derive(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction (Capt)', np.ma.arange(1000, 1025, dtype=np.float))

        node = self.node_class()
        node.derive(alt_std, baro, None, None, None)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            9911, 9938, 9965, 9993, 10020, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))

    def test_baro_setting_capt(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction (Capt)',
                 np.ma.arange(1000, 1025, dtype=np.float))
        values_mapping = {0: 'QFE', 1: 'QNH', 2: 'STD', 3: 'Not Used'}
        baro_sel = M(
            'Baro Setting Selection (Capt)',
            array = np.ma.array([1] * 10 + [2] * 5 + [1] * 10),
            values_mapping=values_mapping
        )

        node = self.node_class()
        node.derive(alt_std, baro, baro_sel, None, None)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            10000, 10000, 10000, 10000, 10000, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))

    def test_baro_setting(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction (Capt)', np.ma.arange(1000, 1025, dtype=np.float))
        values_mapping = {0: 'QFE', 1: 'QNH', 2: 'STD'}
        baro_sel = M(
            'Baro Setting Selection',
            array = np.ma.array([1] * 10 + [2] * 5 + [1] * 10),
            values_mapping=values_mapping
        )

        node = self.node_class()
        node.derive(alt_std, baro, baro_sel, None, None)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            10000, 10000, 10000, 10000, 10000, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))

    def test_baro_correction_isis(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction (Capt)',
                 np.ma.arange(1000, 1025, dtype=np.float))
        baro_cor_isis = P(
            'Baro Correction (ISIS)',
            array = np.ma.concatenate((
                np.ones(10) * 1001,
                np.ones(5) * 1013,
                np.ones(10) * 1024
            )),
        )

        node = self.node_class()
        node.derive(alt_std, baro, None, None, baro_cor_isis)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            10000, 10000, 10000, 10000, 10000, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))


class TestAltitudeQNHFO(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AltitudeQNHFO
        self.operational_combinations = [('Altitude STD', 'Baro Correction (FO)')]

    def test_attribute(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Altitude QNH (FO)')
        self.assertEqual(node.units, ut.FT)

    def test_derive(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction (FO)',
                 np.ma.arange(1000, 1025, dtype=np.float))

        node = self.node_class()
        node.derive(alt_std, baro, None, None, None)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            9911, 9938, 9965, 9993, 10020, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))

    def test_baro_setting_fo(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction (FO)',
                 np.ma.arange(1000, 1025, dtype=np.float))
        values_mapping = {0: 'QFE', 1: 'QNH', 2: 'STD', 3: 'Not Used'}
        baro_sel = M(
            'Baro Setting Selection (FO)',
            array = np.ma.array([1] * 10 + [2] * 5 + [1] * 10),
            values_mapping=values_mapping
        )

        node = self.node_class()
        node.derive(alt_std, baro, baro_sel, None, None)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            10000, 10000, 10000, 10000, 10000, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))


    def test_baro_cpt_fo_setting(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction', np.ma.arange(1000,1025, dtype=np.float))
        values_mapping = {0: 'QFE', 1: 'QNH', 2: 'STD', 3: 'Not Used'}
        baro_sel_fo = M(
            'Baro Setting Selection (FO)',
            array = np.ma.array([1] * 10 + [2] * 5 + [1] * 10),
            values_mapping=values_mapping
        )
        baro_sel = M(
            'Baro Setting Selection',
            array = np.ma.array([1] * 10 + [2] * 5 + [1] * 10),
            values_mapping=values_mapping
        )

        node = self.node_class()
        node.derive(alt_std, baro, baro_sel_fo, baro_sel, None)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            10000, 10000, 10000, 10000, 10000, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))

    def test_baro_correction_isis(self):
        alt_std = P('Altitude STD', np.ma.ones(25, dtype=np.float) * 10000)
        baro = P('Baro Correction (FO)', np.ma.arange(1000, 1025, dtype=np.float))
        baro = P('Baro Correction', np.ma.arange(1000,1025, dtype=np.float))
        baro_cor_isis = P(
            'Baro Correction (ISIS)',
            array = np.ma.concatenate((np.ones(10) * 1001, np.ones(5) * 1013, np.ones(10) * 1024)),
        )

        node = self.node_class()
        node.derive(alt_std, baro, None, None,  baro_cor_isis)

        expected_alt_qnh = [
            9636, 9663, 9691, 9719, 9746, 9774, 9801, 9828, 9856, 9883,
            10000, 10000, 10000, 10000, 10000, 10047, 10074, 10102, 10129, 10156,
            10183, 10210, 10238, 10265, 10292
        ]
        for expected, got in zip(expected_alt_qnh, node.array):
            self.assertEqual(expected, int(got))


class TestAltitudeVisualizationWithGroundOffset(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = AltitudeVisualizationWithGroundOffset
        self.operational_combinations = [
            ('Altitude AAL', 'Altitude STD Smoothed'),
            ('Altitude AAL', 'Altitude STD Smoothed', 'FDR Landing Airport'),
            ('Altitude AAL', 'Altitude STD Smoothed', 'FDR Takeoff Airport'),
            ('Altitude AAL', 'Altitude STD Smoothed', 'FDR Landing Airport',
             'FDR Takeoff Airport'),
            ('Altitude AAL', 'Altitude STD Smoothed', 'FDR Landing Airport',
             'FDR Takeoff Airport', 'Climb', 'Descent'),
            ('Altitude AAL', 'Altitude STD Smoothed', 'FDR Landing Airport',
             'FDR Takeoff Airport', 'Climb', 'Descent', 'Approach Information')
        ]

        array = np.ma.concatenate((
            np.ones(5) * 1000,
            np.arange(1000, 16000, 1000),
            np.ones(4) * 15000,
            np.arange(15000, 4000, -1000),
            np.ones(5) * 4000
        ))
        self.alt_std = P(name='Altitude STD Smoothed', array=array)

        array = np.ma.concatenate((
            np.zeros(5),
            np.arange(0, 15000, 1000),
            np.ones(4) * 10000,
            np.arange(10000, -1000, -1000),
            np.zeros(5)
        ))
        self.alt_aal = P(name='Altitude AAL', array=array)
        self.t_apt = A(name='FDR Takeoff Airport', value={'elevation': 1050})
        self.l_apt = A(name='FDR Landing Airport', value={'elevation': 4100})

        self.climbs = buildsection('Climb', 7, 19)
        self.descents = buildsection('Descent', 24, 35)

        self.apps = App(items=[
            ApproachItem(
                'LANDING', slice(27, 33),
                airport={'elevation': 4100.0}
            )
        ])
        self.logger = 'analysis_engine.derived_parameters.AltitudeVisualizationWithGroundOffset'

    def test_no_rwy_info(self):
        # Check no runway information results in a copy of Altitude STD Smoothed
        alt_qnh = self.node_class()
        with self.assertLogs(self.logger, level='WARNING') as cm:
            alt_qnh.derive(self.alt_aal, self.alt_std)
            self.assertEqual(
                cm.output,
                [f'WARNING:{self.logger}:No takeoff and landing elevation, using Altitude STD Smoothed.']
            )

        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.alt_std.array)
        self.assertEqual(alt_qnh.offset, self.alt_std.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_std.frequency)

    def test_alt_std_adjustment(self):
        alt_qnh = self.node_class()
        alt_qnh.derive(self.alt_aal, self.alt_std, self.l_apt,
                       self.t_apt, self.climbs, self.descents, None, self.apps)
        self.assertEqual(alt_qnh.array[2], 1050.0)  # Takeoff elevation
        self.assertEqual(alt_qnh.array[36], 4100.0)  # Landing elevation
        self.assertEqual(alt_qnh.array[22], 15000.0)  # Cruise at STD

    def test_no_landing_apt_info(self):
        apps = App(items=[
                    ApproachItem(
                        'LANDING', slice(27, 33),
                        airport={},
                        approach_runway=None)
                ])
        l_apt = A(name='FDR Landing Airport', value=None)
        alt_qnh = self.node_class()

        with self.assertLogs(self.logger, level='WARNING') as cm:
            alt_qnh.derive(self.alt_aal, self.alt_std, l_apt, self.t_apt,
                           self.climbs, self.descents, None, apps)
            self.assertEqual(
                cm.output,
                [f'WARNING:{self.logger}:No landing elevation, using Altitude STD Smoothed for descent.']
            )

        self.assertEqual(alt_qnh.array[2], 1050.0)  # Takeoff elevation
        self.assertEqual(alt_qnh.array[35], 4000.0)  # Landing elevation
        self.assertEqual(alt_qnh.array[22], 15000.0)  # Cruise at STD

    def test_no_takeoff_apt_info(self):
        t_apt = A(name='FDR Takeoff Airport', value=None)
        alt_qnh = self.node_class()

        with self.assertLogs(self.logger, level='WARNING') as cm:
            alt_qnh.derive(self.alt_aal, self.alt_std, self.l_apt, t_apt,
                           self.climbs, self.descents, None, self.apps)
            self.assertEqual(
                cm.output,
                [f'WARNING:{self.logger}:No takeoff elevation, using Altitude STD Smoothed for climb.']
            )

        self.assertEqual(alt_qnh.array[2], 1000.0)  # Takeoff elevation
        self.assertEqual(alt_qnh.array[35], 4100.0)  # Landing elevation
        self.assertEqual(alt_qnh.array[22], 15000.0)  # Cruise at STD

    def test_no_takeoff_apt_elevation(self):
        t_apt = A(name='FDR Takeoff Airport', value={})
        alt_qnh = self.node_class()

        with self.assertLogs(self.logger, level='WARNING') as cm:
            alt_qnh.derive(self.alt_aal, self.alt_std, self.l_apt, t_apt,
                           self.climbs, self.descents, None, self.apps)
            self.assertEqual(
                cm.output,
                [f'WARNING:{self.logger}:No takeoff elevation, using Altitude STD Smoothed for climb.']
            )

        self.assertEqual(alt_qnh.array[2], 1000.0)  # Takeoff elevation
        self.assertEqual(alt_qnh.array[35], 4100.0)  # Landing elevation
        self.assertEqual(alt_qnh.array[22], 15000.0)  # Cruise at STD

    def test_trap_alt_difference(self):
        apps = App(items=[
            ApproachItem(
                    'LANDING', slice(27, 33),
                    airport={'elevation': 100.0},
            )
        ])
        climbs = buildsection('Climb', 7, 19)
        descents = buildsection('Descent', 24, 32)
        alt_qnh = self.node_class()
        self.assertRaises(ValueError, alt_qnh.derive,
            self.alt_aal, self.alt_std, self.l_apt,
            self.t_apt, climbs, descents, None, apps
        )

    def test_2_approaches(self):
        array = np.ma.concatenate((
            np.ones(5) * 1000,
            np.arange(1000, 16000, 1000),
            np.ones(4) * 15000,
            np.arange(15000, 4000, -1000),
            np.ones(2) * 4000,  # Touch and go
            np.arange(4000, 8000, 1000),
            np.ones(4) * 7000,
            np.arange(7000, -1000, -1000),
            np.zeros(4)  # Landing
        ))
        alt_std = P(name='Altitude STD Smoothed', array=array)
        array = np.ma.concatenate((
            np.zeros(5),
            np.arange(0, 15000, 1000),
            np.ones(4) * 10000,
            np.arange(10000, -1000, -1000),
            np.zeros(2),  # Touch and go
            np.arange(0, 4000, 1000),
            np.ones(4) * 7000,
            np.arange(7000, -1000, -1000),
            np.zeros(4)  # Landing
        ))
        alt_aal = P(name='Altitude AAL', array=array)
        t_apt = A(name='FDR Takeoff Airport', value={'elevation': 1050})
        l_apt = A(name='FDR Landing Airport', value={'elevation': 4100})
        climbs = buildsection('Climb', 7, 19)
        descents = buildsections('Descent', [24, 35], [45, 53])
        tocs = KTI('Top Of Climb', items=[
            KeyTimeInstance(19), KeyTimeInstance(40)
        ])
        apps = App(items=[
            ApproachItem(
                    'TOUCH_AND_GO', slice(27, 35),
                    airport={'elevation': 4100.0},
            ),
            ApproachItem(
                    'LANDING', slice(47, 52),
                    airport={'elevation': 0.0},
            ),
        ])
        alt_qnh = self.node_class()
        alt_qnh.derive(alt_aal, alt_std, l_apt, t_apt,
                       climbs, descents, tocs, apps)
        self.assertEqual(alt_qnh.array[2], 1050.0)  # Takeoff elevation
        self.assertEqual(alt_qnh.array[36], 4100.0)  # Touch and go elevation
        self.assertEqual(alt_qnh.array[52], 0.0)  # Landing elevation
        self.assertEqual(alt_qnh.array[20], 15000.0)  # Alt STD in cruise

    def test_missing_app_rwy_info(self):
        apps = App(items=[
            ApproachItem(
                'LANDING', slice(27, 33),
                airport={},
                approach_runway=None)
        ])
        alt_qnh = self.node_class()
        alt_qnh.derive(self.alt_aal, self.alt_std, self.l_apt,
                       self.t_apt, self.climbs, self.descents, None, apps)
        self.assertEqual(alt_qnh.array[2], 1050.0)  # Takeoff elevation
        self.assertEqual(alt_qnh.array[36], 4100.0)  # Landing elevation
        self.assertEqual(alt_qnh.array[22], 15000.0)  # Cruise at STD


class TestAltitudeVisualizationWithoutGroundOffset(unittest.TestCase, NodeTest):

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Altitude STD Smoothed')))
        self.assertFalse(self.node_class.can_operate(('Altitude STD Smoothed', 'Cruise')))
        self.assertFalse(self.node_class.can_operate(('Altitude AAL', 'Cruise')))
        self.assertTrue(self.node_class.can_operate(('Altitude AGL', 'Altitude STD Smoothed'),
                                                    ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Altitude AAL', 'Altitude STD Smoothed'),
                                                     ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Altitude AAL', 'Altitude STD Smoothed'),
                                                    ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate(('Altitude AAL', 'Altitude STD Smoothed', 'Cruise')))

    def setUp(self):
        self.node_class = AltitudeVisualizationWithoutGroundOffset
        self.operational_combinations = [
            ('Altitude AAL', 'Altitude STD Smoothed'),
            ('Altitude AGL', 'Altitude STD Smoothed'),
            ('Altitude AGL', 'Altitude AAL', 'Altitude STD Smoothed'),
        ]
        data = [np.ma.arange(0, 1000, step=30)]
        data.append(data[0][::-1] + 50)
        self.alt_aal_1 = P(name='Altitude AAL', array=np.ma.concatenate(data))
        self.alt_aal_2 = P(name='Altitude AAL', array=np.ma.concatenate((np.zeros(5), np.arange(0, 15000, 1000), np.ones(4) * 10000, np.arange(10000, -1000, -1000), np.zeros(5))))
        self.alt_aal_3 = P(name='Altitude AAL', array=np.ma.append(self.alt_aal_2.array, self.alt_aal_2.array))
        self.alt_std_1 = P(name='Altitude STD Smoothed', array=np.ma.concatenate((np.ones(5) * 1000, np.arange(1000, 16000, 1000), np.ones(4) * 15000, np.arange(15000, 4000, -1000), np.ones(5) * 4000)))
        self.alt_std_2 = P(name='Altitude STD Smoothed', array=np.ma.append(self.alt_std_1.array, self.alt_std_1.array))
        self.alt_std_3 = P(name='Altitude STD Smoothed', array=np.ma.concatenate((np.ones(5) * 150, np.arange(150, 2400, 150), np.ones(4) * 2250, np.arange(2250, 600, -150), np.ones(5) * 600)))
        self.alt_std_4 = P(name='Altitude STD Smoothed', array=np.ma.append(self.alt_std_3.array, self.alt_std_3.array))
        self.alt_agl_1 = P(name='Altitude AGL', array=np.ma.concatenate((np.zeros(5), np.arange(0, 2250, 150), np.ones(4) * 2000, np.arange(2000, -200, -200), np.zeros(5))))
        self.alt_agl_2 = P(name='Altitude AGL', array=np.ma.append(self.alt_agl_1.array, self.alt_agl_1.array))
        self.expected = []


        # 1. Data same as Altitude AAL, no mask applied:
        data = np.ma.copy(self.alt_aal_1.array)
        self.expected.append(data)
        # 2. None masked, data Altitude AAL, +50 ft t/o, +100 ft ldg:
        data = np.ma.array([50, 80, 110, 140, 170, 200, 230, 260, 290, 320,
            350, 351, 352, 354, 355, 357, 358, 360, 361, 363, 364, 366, 367,
            368, 370, 371, 373, 374, 376, 377, 379, 380, 382, 383, 385, 386,
            387, 389, 390, 392, 393, 395, 396, 398, 399, 401, 402, 403, 405,
            406, 408, 409, 411, 412, 414, 415, 417, 418, 420, 390, 360, 330,
            300, 270, 240, 210, 180, 150])
        data.mask = False
        self.expected.append(data)

    @unittest.skip('New implementation broke these tests!')
    def test_derive__output(self):
        alt_qnh = self.node_class()
        # Check no airport/runway information results in a fully masked copy of Altitude AAL:
        alt_qnh.derive(self.alt_aal_1)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[0])
        self.assertEqual(alt_qnh.offset, self.alt_aal_1.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal_1.frequency)
        # Check everything works calling with airport details:
        alt_qnh.derive(self.alt_aal_1, self.alt_std_1)
        ma_test.assert_masked_array_approx_equal(alt_qnh.array, self.expected[1])
        self.assertEqual(alt_qnh.offset, self.alt_aal_1.offset)
        self.assertEqual(alt_qnh.frequency, self.alt_aal_1.frequency)

    def test_alt_std_adjustment(self):
        cruise = buildsection('Cruise', 19, 25)
        alt_qnh = self.node_class()
        alt_qnh.derive(None, self.alt_aal_2, self.alt_std_1, cruise)
        self.assertEqual(alt_qnh.array[2], 0.0)
        self.assertEqual(alt_qnh.array[36], 0.0)
        self.assertEqual(alt_qnh.array[22], 11600.0)  # Cruise at STD

    def test_no_cruise(self):
        cruise = S('Cruise', items=[])
        alt_vis = self.node_class()
        alt_vis.derive(None, self.alt_aal_2, self.alt_std_1, cruise)
        self.assertEqual(alt_vis.array[2], 0.0)
        self.assertEqual(alt_vis.array[36], 0.0)
        self.assertEqual(alt_vis.array[22], 10000.0)  # at Max STD

    def test_multiple_cruises(self):
        cruises = buildsections('Cruise', [19, 25], [59, 65])
        alt_qnh = self.node_class()
        alt_qnh.derive(None, self.alt_aal_3, self.alt_std_2, cruises)
        self.assertEqual(alt_qnh.array[2], 0.0)
        self.assertEqual(alt_qnh.array[36], 0.0)
        self.assertEqual(alt_qnh.array[22], 11600.0)  # Cruise at STD
        self.assertEqual(alt_qnh.array[62], 11600.0)  # Cruise at STD

    def test_agl_single_cruise(self):
        cruise = buildsection('Cruise', 19, 25)
        alt_qnh = self.node_class()
        alt_qnh.derive(self.alt_agl_1, None, self.alt_std_3, cruise)
        self.assertEqual(alt_qnh.array[2], 0.0)
        self.assertEqual(alt_qnh.array[36], 0.0)
        self.assertEqual(alt_qnh.array[22], 2010.0)  # Cruise at STD

    def test_agl_multiple_cruises(self):
        cruises = buildsections('Cruise', [19, 25], [59, 65])
        alt_qnh = self.node_class()
        alt_qnh.derive(self.alt_agl_2, None, self.alt_std_4, cruises)
        self.assertEqual(alt_qnh.array[2], 0.0)
        self.assertEqual(alt_qnh.array[36], 0.0)
        self.assertEqual(alt_qnh.array[22], 2010.0)  # Cruise at STD
        self.assertEqual(alt_qnh.array[62], 2010.0)  # Cruise at STD

class TestAltitudeRadio(unittest.TestCase):
    """
    def test_can_operate(self):
        expected = [('Altitude Radio Sensor', 'Pitch',
                     'Main Gear To Altitude Radio')]
        opts = AltitudeRadio.get_operational_combinations()
        self.assertEqual(opts, expected)
    """

    # This test retained as it checks the particularly awkward 737NG configuration.
    def test_altitude_radio_737_3C(self):
        alt_rad = AltitudeRadio()
        fast = S(items=[Section('Fast', slice(0, 20), 0, 20)])
        alt_rad.derive(Parameter('Altitude Radio (A)',
                                 np.ma.array(data=[10.0]*9 + [50.1], mask=[0]*10), 0.5,  0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array(data=[20.0,20.0,20.0,20.0,60.2], mask=[0]*5), 0.25, 1.0),
                       Parameter('Altitude Radio (C)',
                                 np.ma.array(data=[30.0,30.0,30.0,30.0,70.3], mask=[0]*5), 0.25, 3.0),
                       None, None, None, None, None, None, None)
        self.assertGreater(alt_rad.array[4], 16)
        self.assertLess(alt_rad.array[4], 30)
        self.assertEqual(alt_rad.array.mask[0], True)
        self.assertEqual(alt_rad.array.mask[4], False)
        self.assertEqual(alt_rad.offset, 0.0)
        self.assertEqual(alt_rad.frequency, 4.0)

    @unittest.skip('Code now requires signal change; this test defunct')
    def test_altitude_radio_737_5_EFIS(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(Parameter('Altitude Radio (A)',
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5, 0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.5, 1.0),
                       None, None, None, None, None, None)
        answer = np.ma.array(data=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.1, 15.1, 15.1, 15.1, 15.2, 15.2, 15.3, 15.3, 15.4],
                             mask=[True] + ([False] * 38) + [True])
        assert_array_almost_equal(alt_rad.array, answer, decimal=1)
        self.assertEqual(alt_rad.offset, 0.0)
        self.assertEqual(alt_rad.frequency, 4.0)

    @unittest.skip('Code now requires signal change; this test defunct')
    def test_altitude_radio_737_5_Analogue(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(Parameter('Altitude Radio (A)',
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5, 0.0),
                       Parameter('Altitude Radio (B)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.5, 1.0),
                       None, None, None, None, None, None)
        answer = np.ma.array(data=[
            15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
            15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
            15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.1, 15.1,
            15.1, 15.1, 15.2, 15.2, 15.3, 15.3, 15.4], mask=[True] + (38 * [False]) + [False])
        assert_array_almost_equal(alt_rad.array, answer, decimal=1)
        self.assertEqual(alt_rad.offset, 0.0)
        self.assertEqual(alt_rad.frequency, 4.0)

    @unittest.skip('Code now requires signal change; this test defunct')
    def test_altitude_radio_787(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(None, None, None,
                       Parameter('Altitude Radio (L)',
                                 np.ma.array([10.0,10.0,10.0,10.0,10.1]), 0.5, 0.0),
                       Parameter('Altitude Radio (R)',
                                 np.ma.array([20.0,20.0,20.0,20.0,20.2]), 0.5, 1.0),
                       None, None, None)
        answer = np.ma.array(data=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.1, 15.1, 15.1, 15.1, 15.2, 15.2, 15.3, 15.3, 15.4],
                             mask=[True] + (38 * [False]) + [False])
        assert_array_almost_equal(alt_rad.array, answer, decimal=1)
        self.assertEqual(alt_rad.offset, 0.0)
        self.assertEqual(alt_rad.frequency, 4.0)

    def test_altitude_radio_A320(self):
        # strictly these are two flights, but that should not matter
        fast = S(frequency=0.5,
                 items=[Section('Fast', slice(336, 5397), 336, 5397),
                        Section('Fast', slice(5859, 11520), 5859, 11520)])
        radioA = load(os.path.join(
            test_data_path, 'A320_Altitude_Radio_A_overflow.nod'))
        for n in [371, 5092, 5298, 5894, 11240, 11452]:
            radioA.array.mask[n] = False
        radioB = load(os.path.join(
            test_data_path, 'A320_Altitude_Radio_B_overflow.nod'))
        for n in range(433, 447):
            radioB.array.mask[n] = False
        for n in [371, 435, 446, 5296, 5297, 5893, 11317, 11451]:
            radioB.array.mask[n] = False
        radioA.array = overflow_correction(radioA.array, radioA, fast.get_slices())
        radioB.array = overflow_correction(radioB.array, radioB, fast.get_slices())
        radioA.array = np.ma.masked_greater(radioA.array, 5000.0)
        radioB.array = np.ma.masked_greater(radioA.array, 5000.0)
        rad = AltitudeRadio()
        rad.derive(radioA, radioB, None, None, None, None, None, None,
                   None, family=A('Family', 'A320'))

        sects = np.ma.clump_unmasked(rad.array)
        self.assertEqual(len(sects), 5)
        for sect in sects[0], sects[2]:
            # takeoffs
            self.assertAlmostEqual(rad.array[sect.start] / 10., 0, 0)
        for sect in sects[1], sects[4]:
            # landings
            self.assertAlmostEqual(rad.array[sect.stop - 1] / 10., 0, 0)

    @unittest.skip('Code now requires signal change; this test defunct')
    def test_altitude_radio_A320_zero_dropouts(self):
        # Four cases of data dropout at around 700ft in the descent. If the
        # dropout is not handled correctly, the preceding data is much lower,
        # hence fails the simple > 500ft test.
        radioA = load_compressed(os.path.join(
            test_data_path, 'A320_F_Altitude_Radio_(A)_dropout.npz'))
        radioA[3500:24900] = np.ma.masked
        radioA = overflow_correction(radioA)
        self.assertGreater(radioA[28527], 500.0)

        radioB = load_compressed(os.path.join(
            test_data_path, 'A320_F_Altitude_Radio_(B)_dropout.npz'))
        radioB = overflow_correction(radioB)
        self.assertGreater(radioB[28527], 500.0)

        radioA = load_compressed(os.path.join(
            test_data_path, 'A320_K_Altitude_Radio_(A)_dropout.npz'))
        radioA = overflow_correction(radioA)
        self.assertGreater(radioA[62710], 500.0)

        radioB = load_compressed(os.path.join(
            test_data_path, 'A320_K_Altitude_Radio_(B)_dropout.npz'))
        radioB = overflow_correction(radioB)
        self.assertGreater(radioB[62710], 500.0)

    def test_altitude_radio_A320_small_jumps(self):
        # Some small jumps on a few aircraft caused problems.
        fast = buildsection('Fast', 248.8, 3674.3)
        radioA = load(os.path.join(
            test_data_path, 'Altitude_Radio_A_A320_eec5df85279d.nod'))
        radioA.array = overflow_correction(radioA.array, radioA, fast.get_slices())
        rad = AltitudeRadio()
        rad.derive(radioA, None, None, None, None, None, None, None,
                   None, family=A('Family', 'A320'))
        self.assertGreater(rad.array[14150], 1300)
        self.assertLess(rad.array[14150], 1500)

    def test_altitude_radio_A330(self):
        fast = buildsection('Fast', 480, 31032)
        radioA = load(os.path.join(
            test_data_path, 'A330_AltitudeRadio_A_overflow_8191.nod'))
        for n in [648, 649, 30710, 30711]:
            radioA.array.mask[n] = False
        radioB = load(os.path.join(
            test_data_path, 'A330_AltitudeRadio_B_overflow_8191.nod'))
        for n in range(649, 30711):
            radioB.array.mask[n] = False
        radioA.array = overflow_correction(radioA.array, radioA, fast.get_slices())
        radioB.array = overflow_correction(radioB.array, radioB, fast.get_slices())
        rad = AltitudeRadio()
        rad.derive(radioA, radioB, None, None, None, None, None, None,
                   None, family=A('Family', 'A330'))

        sects = np.ma.clump_unmasked(rad.array)
        self.assertEqual(len(sects), 3)
        self.assertLess(sects[1].start, 2000)
        self.assertGreater(sects[1].stop, 2500)
        self.assertAlmostEqual(rad.array[2708], 4997, places=0)
        self.assertEqual(sects[2].start, 122508)
        self.assertAlmostEqual(rad.array[122508], 4994, places=0)

    def test_altitude_radio_CL_600(self):
        alt_rad = AltitudeRadio()
        array = np.ma.concatenate((np.arange(5,-5,-1), np.arange(-5,15))).astype(float)
        fast = buildsection('Fast', 0, 6)
        alt_rad.derive(None, None, None,
                       Parameter('Altitude Radio (L)', array, 1.0, 0.0),
                       None, None, None, None,
                       Parameter('Pitch',
                                 np.ma.concatenate((np.zeros(30), np.ones(30) * 5, np.ones(30) * 10, np.ones(30) * 20)).astype(float), 4.0, 0.0),
                       family=A('Family', 'CL-600'))
        self.assertAlmostEqual(alt_rad.array.data[4], 2.5) # -1.5ft offset
        self.assertEqual(alt_rad.array.data[36], -3.675) # -1.5ft & 5deg
        self.assertEqual(alt_rad.array.data[76], 6.15) # -1.5ft & 10 deg

    def test_altitude_radio_no_valid_data(self):
        fast = buildsection('Fast', 0, 10)
        array = np.ma.array(data=[1]*10, mask=[1]*10)
        radio_L = P('Altitude Radio (L)', array, 1.0, 0.0)
        radio_R = P('Altitude Radio (R)', array, 1.0, 0.0)
        rad = AltitudeRadio()
        rad.derive(radio_L, radio_R, None, None, None, None, None, None,
                   None, family=A('Family', 'A330'))
        self.assertEqual(len(rad.array), 40)

    def test_rad_alt_sources_contain_nan(self):
        array = np.ma.array(data=[np.nan] + [1]*8 + [np.nan], mask=[0]*10)
        radio_L = P('Altitude Radio (L)', array, 1.0, 0.0)
        radio_R = P('Altitude Radio (R)', array, 1.0, 0.0)
        rad = AltitudeRadio()
        rad.derive(radio_L, radio_R, None, None, None, None, None, None,
                   None, family=A('Family', 'A330'))
        self.assertEqual(len(rad.array), 40)

    @unittest.skip('Code for this test no longer used')
    def test_altitude_radio_with_diversion(self):
        alt_rad = AltitudeRadio()
        fast = buildsection('Fast', 0, 200)
        alt_rad.derive(
            Parameter('Altitude Radio (A)', np.ma.array(
                data=[0.0]*40 + [28000.0]*40 + [50.0]*20 + [100.0]*20 + [25000.0]*40 + [0.0]*40,
                mask=[0.0]*40 + [1.0]*40 + [0.0]*40 + [1.0]*40 + [0.0]*40),
            ),
            Parameter('Altitude Radio (b)', np.ma.array(
                data=[0.0]*20 + [28000.0]*20 + [50.0]*10 + [100.0]*10 + [25000.0]*20 + [0.0]*20,
                mask=[0.0]*20 + [1.0]*20 + [0.0]*20 + [1.0]*20 + [0.0]*20),
            ),
            None, None, None, None, None, None, None, None,
        )
        self.assertListEqual(list(alt_rad.array[1:159]), [0.0]*158)
        self.assertTrue(all(alt_rad.array.mask[160:320]))
        self.assertTrue(all(alt_rad.array.mask[480:640]))
        self.assertLess(alt_rad.array[321], 1075)  # end of first ccd should have difference between alt std and rad.
        self.assertAlmostEqual(alt_rad.array.data[478], 100.0, places=1)
        self.assertTrue(all(x == 0.0 for x in alt_rad.array.mask[641:799]))
        self.assertEqual(alt_rad.offset, 0.0)
        self.assertEqual(alt_rad.frequency, 4.0)

    @unittest.skip('Test data not available')
    def test_altitude_radio_b737_no_overflow(self):
        source_A = load(os.path.join(test_data_path, 'radio_737_test_source_A.nod'))
        source_B = load(os.path.join(test_data_path, 'radio_737_test_source_B.nod'))
        source_C = load(os.path.join(test_data_path, 'radio_737_test_source_C.nod'))
        alt_std = load(os.path.join(test_data_path, 'radio_737_test_alt_std.nod'))
        pitch = load(os.path.join(test_data_path, 'radio_737_test_pitch.nod'))
        fast = load(os.path.join(test_data_path, 'radio_737_test_fast.nod'))
        rad = AltitudeRadio()
        rad.derive(source_A, source_B, source_C, None, None, None, None, None,
                   alt_std, pitch,
                   fast, family=A('Family', 'B737'))

        sects = np.ma.clump_unmasked(rad.array)
        self.assertEqual(len(sects), 2)
        self.assertGreater(np.ma.min(rad.array), -4.92)


class TestAltitudeRadioOffsetRemoved(unittest.TestCase):
    def setUp(self):
        self.test_array = np.ma.array([1,1,1,1,1,1,1,1,3,8,13,28,67])

    def test_can_operate(self):
        expected = [('Altitude Radio', 'Fast')]
        opts = AltitudeRadioOffsetRemoved.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_basic_operation(self):
        ralt = P('Altitude Radio', self.test_array)
        fast = buildsection('Fast', 2, None)
        aor = AltitudeRadioOffsetRemoved()
        aor.derive(ralt, fast)
        expected = ralt.array - 1
        ma_test.assert_masked_array_equal(aor.array, expected)

    def test_no_change_for_wierd_values(self):
        ralt = P('Altitude Radio', self.test_array + 312)
        aor = AltitudeRadioOffsetRemoved()
        aor.derive(ralt)
        expected = ralt.array
        ma_test.assert_masked_array_equal(aor.array, expected)

    def test_no_change_for_negative_values(self):
        ralt = P('Altitude Radio', self.test_array - 4)
        aor = AltitudeRadioOffsetRemoved()
        aor.derive(ralt)
        expected = ralt.array
        ma_test.assert_masked_array_equal(aor.array, expected)

    def test_no_change_for_excessive_adjustment(self):
        ralt = P('Altitude Radio', self.test_array + 11)
        aor = AltitudeRadioOffsetRemoved()
        aor.derive(ralt)
        expected = ralt.array
        ma_test.assert_masked_array_equal(aor.array, expected)

    def test_with_realistic_values(self):
        testwave = np.ma.hstack([[-1.6]*10, 1500.0*(1.0 - np.cos(np.arange(0,6.1,0.1))), [1.2]*20])
        testwave = np.ma.masked_greater(testwave, 2500.0)
        ralt = P('Altitude Radio', testwave)
        fast = buildsection('Fast', 2, None)
        aor = AltitudeRadioOffsetRemoved()
        aor.derive(ralt, fast)
        expected = np_ma_masked_zeros_like(testwave)
        expected = testwave - 1.2

        ## Plot to check shape of test waveform
        #import matplotlib.pyplot as plt
        #plt.plot(testwave)
        #plt.plot(expected)
        #plt.show()
        ma_test.assert_masked_array_equal(aor.array, expected)

    def test_with_go_around(self):
        testwave = np.ma.cos(np.arange(0, 3.14 * 4, 0.2)) * -1500 + 1500 + \
            np.ma.cos(np.arange(0, 3.14 * 2, 0.1)) * -100 + 100
        testwave = np.ma.masked_greater(testwave, 2500.0)
        fast = buildsection('Fast', 2, None)
        ralt = P('Altitude Radio', testwave)
        aor = AltitudeRadioOffsetRemoved()
        aor.derive(ralt, fast)

        ## Plot to check shape of test waveform
        #import matplotlib.pyplot as plt
        #plt.plot(testwave+100) # Offset to make it easier to see.
        #plt.plot(aor.array)
        #plt.show()

        ma_test.assert_masked_array_equal(aor.array, testwave.copy())


    def test_no_change_for_mask_near_liftoff(self):
        ralt = P('Altitude Radio', np.ma.array(data=[1,1,1,1,3,8,13,28,67],
                                               mask=[0,0,1,0,0,0, 0, 0, 0]))
        aor = AltitudeRadioOffsetRemoved()
        aor.derive(ralt)
        expected = ralt.array
        ma_test.assert_masked_array_equal(aor.array, expected)


"""
class TestAltitudeRadioForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio Offset Removed',)]
        opts = AltitudeRadioForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_altitude_for_radio_phases_repair(self):
        alt_4_ph = AltitudeRadioForFlightPhases()
        raw_data = np.ma.array([0,1,2])
        raw_data[1] = np.ma.masked
        alt_4_ph.derive(Parameter('Altitude Radio', raw_data, 1,0.0))
        expected = np.ma.array([0,0,0],mask=False)
        assert_array_equal(alt_4_ph.array, expected)
"""


"""
class TestAltitudeQNH(unittest.TestCase):
    # Needs airport database entries simulated. TODO.

"""

'''
class TestAltitudeSTD(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeSTD.get_operational_combinations(),
          [('Altitude STD (Coarse)', 'Altitude STD (Fine)'),
           ('Altitude STD (Coarse)', 'Vertical Speed')])

    def test__high_and_low(self):
        high_values = np.ma.array([15000, 16000, 17000, 18000, 19000, 20000,
                                   19000, 18000, 17000, 16000],
                                  mask=[False] * 9 + [True])
        low_values = np.ma.array([15500, 16500, 17500, 17800, 17800, 17800,
                                  17800, 17800, 17500, 16500],
                                 mask=[False] * 8 + [True] + [False])
        alt_std_high = Parameter('Altitude STD High', high_values)
        alt_std_low = Parameter('Altitude STD Low', low_values)
        alt_std = AltitudeSTD()
        result = alt_std._high_and_low(alt_std_high, alt_std_low)
        assert_equal(result,
                             np.ma.masked_array([15500, 16500, 17375, 17980, 19000,
                                                 20000, 19000, 17980, 17375, 16500],
                                                mask=[False] * 8 + 2 * [True]))

    @patch('analysis_engine.derived_parameters.first_order_lag')
    def test__rough_and_ivv(self, first_order_lag):
        alt_std = AltitudeSTD()
        alt_std_rough = Parameter('Altitude STD Rough',
                                  np.ma.array([60, 61, 62, 63, 64, 65],
                                              mask=[False] * 5 + [True]))
        first_order_lag.side_effect = lambda arg1, arg2, arg3: arg1
        ivv = Parameter('Inertial Vertical Speed',
                        np.ma.array([60, 120, 180, 240, 300, 360],
                                    mask=[False] * 4 + [True] + [False]))
        result = alt_std._rough_and_ivv(alt_std_rough, ivv)
        assert_equal(result,
                             np.ma.masked_array([61, 63, 65, 67, 0, 0],
                                                mask=[False] * 4 + [True] * 2))

    def test_derive(self):
        alt_std = AltitudeSTD()
        # alt_std_high and alt_std_low passed in.
        alt_std._high_and_low = Mock()
        high_and_low_array = 3
        alt_std._high_and_low.return_value = high_and_low_array
        alt_std_high = 1
        alt_std_low = 2
        alt_std.derive(alt_std_high, alt_std_low, None, None)
        alt_std._high_and_low.assert_called_once_with(alt_std_high, alt_std_low)
        self.assertEqual(alt_std.array, high_and_low_array)
        # alt_std_rough and ivv passed in.
        rough_and_ivv_array = 6
        alt_std._rough_and_ivv = Mock()
        alt_std._rough_and_ivv.return_value = rough_and_ivv_array
        alt_std_rough = 4
        ivv = 5
        alt_std.derive(None, None, alt_std_rough, ivv)
        alt_std._rough_and_ivv.assert_called_once_with(alt_std_rough, ivv)
        self.assertEqual(alt_std.array, rough_and_ivv_array)
        # All parameters passed in (improbable).
        alt_std.derive(alt_std_high, alt_std_low, alt_std_rough, ivv)
        self.assertEqual(alt_std.array, high_and_low_array)
        '''


class TestAltitudeTail(unittest.TestCase):
    def test_can_operate(self):
        expected = [(
            'Altitude Radio', 'Pitch', 'Takeoff', 'Go Around And Climbout', 'Landing',
            'Ground To Lowest Point Of Tail', 'Main Gear To Lowest Point Of Tail',
        )]
        opts = AltitudeTail.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_altitude_tail(self):
        talt = AltitudeTail()

        talt.derive(Parameter('Altitude Radio', np.ma.zeros(10), 1,0.0),
                    Parameter('Pitch', np.ma.arange(10) * 2, 1, 0.0),
                    buildsections('Takeoff', [0, 3]),
                    buildsections('Go Around And Climbout', [5, 6]),
                    buildsections('Landing', [8, 10]),
                    Attribute('Ground To Lowest Point Of Tail', ut.convert(10.0, ut.FT, ut.METER)),
                    Attribute('Main Gear To Lowest Point Of Tail', ut.convert(35.0, ut.FT, ut.METER)))
        result = talt.array
        # At 35ft tail arm and 16deg nose up, the tail just scrapes the runway with 10ft
        # clearance at the mainwheels...
        answer = np.ma.array(data=[10.0,
                                    8.7777730,
                                    7.5525615,
                                    6.3213517,
                                    5.0810707,
                                    3.8285556,
                                    2.5605203,
                                    1.2735199,
                                    -0.0360885,
                                    -1.372189],
                             dtype=np.float, mask=[0]*4+[1]+[0]*2+[1]+[0]*2)
        ma_test.assert_masked_array_almost_equal(result, answer)

    def test_altitude_tail_after_lift(self):
        talt = AltitudeTail()
        talt.derive(Parameter('Altitude Radio', np.ma.array([0, 1.372189])),
                    Parameter('Pitch', np.ma.array([0, 18])),
                    buildsections('Takeoff', [0, 2]),
                    buildsections('Go Around And Climbout', [2, 2]),
                    buildsections('Landing', [2, 2]),
                    Attribute('Ground To Lowest Point Of Tail', ut.convert(10.0, ut.FT, ut.METER)),
                    Attribute('Main Gear To Lowest Point Of Tail', ut.convert(35.0, ut.FT, ut.METER)))
        result = talt.array
        # Lift 5ft
        answer = np.ma.array(data=[10, 0.0],
                             dtype=np.float, mask=False)
        np.testing.assert_array_almost_equal(result.data, answer.data)


###############################################################################
# Brakes

class TestBrake_C_Temp(unittest.TestCase):

    def setUp(self):
        self.node_class = Brake_C_Temp

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertTrue(len(poss_combs[0]), 8)
        self.assertTrue('Brake (C) (1) Temp' in poss_combs[0])
        self.assertFalse('Brake (C) (9) Temp' in poss_combs[0])
        self.assertFalse('Brake (C) Temp' in poss_combs[0])

    def test_derive(self):
        brake_1_array =   [0, 30, 50, 80,  100, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]
        brake_3_array =   [0,  0, 30, 60,   85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        expected_array =  [0, 15, 40, 70, 92.5, 100, 70, 70, 70, 50, 50, 20,  5,  5, 0]

        brake_1 = P(name='Brake (C) (1) Temp', array=brake_1_array, frequency=1, offset=0.1)
        brake_3 = P(name='Brake (C) (3) Temp', array=brake_3_array, frequency=1, offset=0.5)
        node = self.node_class()
        node.derive(brake_1, None, brake_3, None, None, None, None, None)
        assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.3)

    def test_median(self):
        brake_1_array =   [30, 30]
        brake_2_array =   [40, 40]
        brake_3_array =   [100, 100]
        expected_array =  [40, 40]
        brake_1 = P(name='Brake (C) (1) Temp', array=brake_1_array)
        brake_2 = P(name='Brake (C) (2) Temp', array=brake_2_array)
        brake_3 = P(name='Brake (C) (3) Temp', array=brake_3_array)
        node = self.node_class()
        node.derive(brake_1, brake_2, brake_3, None, None, None, None, None)
        assert_array_equal(node.array, expected_array)

class TestBrake_L_Temp(unittest.TestCase):

    def setUp(self):
        self.node_class = Brake_L_Temp

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertTrue(len(poss_combs[0]), 6)
        self.assertTrue('Brake (L) (1) Temp' in poss_combs[0])

    def test_derive(self):
        brake_1_array =   [30, 30]
        brake_2_array =   [40, 40]
        brake_3_array =   [100, 100]
        expected_array =  [40, 40]
        brake_1 = P(name='Brake (L) (1) Temp', array=brake_1_array)
        brake_2 = P(name='Brake (L) (2) Temp', array=brake_2_array)
        brake_3 = P(name='Brake (L) (3) Temp', array=brake_3_array)
        node = self.node_class()
        node.derive(brake_1, brake_2, brake_3, None, None, None)
        assert_array_equal(node.array, expected_array)

class TestBrake_R_Temp(unittest.TestCase):

    def setUp(self):
        self.node_class = Brake_R_Temp

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertTrue(len(poss_combs[0]), 6)
        self.assertTrue('Brake (R) (1) Temp' in poss_combs[0])

    def test_derive(self):
        brake_1_array =   [30, 30]
        brake_2_array =   [40, 40]
        brake_3_array =   [100, 100]
        expected_array =  [40, 40]
        brake_1 = P(name='Brake (R) (1) Temp', array=brake_1_array)
        brake_2 = P(name='Brake (R) (2) Temp', array=brake_2_array)
        brake_3 = P(name='Brake (R) (3) Temp', array=brake_3_array)
        node = self.node_class()
        node.derive(brake_1, brake_2, brake_3, None, None, None)
        assert_array_equal(node.array, expected_array)

class TestBrake_TempAvg(unittest.TestCase):

    def setUp(self):
        self.node_class = Brake_TempAvg

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertEqual(len(poss_combs), 2**15-1)

    def test_derive(self):
        brake_1_array =   [0, 30, 50, 80,  100, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]
        brake_5_array =   [0,  0, 30, 60,   85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        expected_array =  [0, 15, 40, 70, 92.5, 100, 70, 70, 70, 50, 50, 20,  5,  5, 0]

        brake_1 = P(name='Brake (L) (1) Temp', array=brake_1_array, frequency=1,
                    offset=0.1)

        brake_5 = P(name='Brake (R) (1) Temp', array=brake_5_array, frequency=1,
                    offset=0.5)

        node = self.node_class()
        node.derive(brake_1, None, brake_5, None, None, None, None, None, None, None, None, None, None, None, None)

        assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.3)

    def test_r(self):
        brake_R_array =   [0,  0, 30, 60,   85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        brake_R = P(name='Brake (R) Temp', array=brake_R_array, frequency=1,
                         offset=0.3)
        node = self.node_class()
        node.derive(None, None, None, None, None, None, None, None, None, None, brake_R, None, None, None, None)
        assert_array_equal(node.array, brake_R_array)


class TestBrake_TempMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Brake_TempMax

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertEqual(len(poss_combs), 2**15-1)

    def test_derive(self):
        brake_1_array =  [0, 30, 50, 80, 100, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]
        brake_5_array =  [0,  0, 30, 60,  85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        expected_array = [0, 30, 50, 80, 100, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]

        brake_1 = P(name='Brake (L) (1) Temp', array=brake_1_array, frequency=1,
                 offset=0.1)

        brake_5 = P(name='Brake (R) (1) Temp', array=brake_5_array, frequency=1,
                         offset=0.5)

        node = self.node_class()
        node.derive(brake_1, None, brake_5, None, None, None, None, None, None, None, None, None, None, None, None)
        assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.3)

    def test_r(self):
        brake_R_array =   [0,  0, 30, 60,   85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        brake_R = P(name='Brake (R) Temp', array=brake_R_array, frequency=1,
                         offset=0.3)
        node = self.node_class()
        node.derive(None, None, None, None, None, None, None, None, None, None, brake_R, None, None, None, None)
        assert_array_equal(node.array, brake_R_array)

class TestBrake_TempMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Brake_TempMin

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertEqual(len(poss_combs), 2**15-1)

    def test_derive(self):
        brake_1_array =  [0, 30, 50, 80, 100, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]
        brake_5_array =  [0,  0, 30, 60,  85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        expected_array = [0,  0, 30, 60,  85, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]

        brake_1 = P(name='Brake (L) (1) Temp', array=brake_1_array, frequency=1,
                 offset=0.1)

        brake_5 = P(name='Brake (R) (3) Temp', array=brake_5_array, frequency=1,
                         offset=0.5)

        node = self.node_class()
        node.derive(brake_1, None, brake_5, None, None, None, None, None, None, None, None, None, None, None, None)

        assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.3)

    def test_r(self):
        brake_R_array =   [0,  0, 30, 60,   85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        brake_R = P(name='Brake (R) Temp', array=brake_R_array, frequency=1,
                         offset=0.3)
        node = self.node_class()
        node.derive(None, None, None, None, None, None, None, None, None, None, brake_R, None, None, None, None)
        assert_array_equal(node.array, brake_R_array)

class TestBrakePressure(unittest.TestCase):
    def test_can_operate(self):
        two_sources = ('Brake (L) Press', 'Brake (R) Press')
        four_sources = ('Brake (L) Inboard Press',
                        'Brake (L) Outboard Press',
                        'Brake (R) Inboard Press',
                        'Brake (R) Outboard Press')
        opts = BrakePressure.get_operational_combinations()
        self.assertTrue(two_sources in opts)
        self.assertTrue(four_sources in opts)

    def test_basic_two_params(self):
        brake_left = P('Brake (L) Press', np.ma.array([0,1,0,0,0]))
        brake_right = P('Brake (R) Press', np.ma.array([0,0,0,1,0]))
        brakes = BrakePressure()
        brakes.derive(brake_left, brake_right)
        expected = np.ma.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                               mask = [0,0,0,0,0,0,0,0,0,1])
        assert_array_equal(brakes.array, expected)

    def test_basic_four_params(self):
        brake_li = P('Brake (L) Inboard Press', np.ma.array([0,0.75,1,0.75,0]))
        brake_lo = P('Brake (L) Outboard Press', np.ma.array([0,0.75,1,0.75,0]))
        brake_ri = P('Brake (R) Inboard Press', np.ma.array([0,0.75,1,0.75,0]))
        brake_ro = P('Brake (R) Outboard Press', np.ma.array([0,0.75,1,0.75,0]))
        brakes = BrakePressure()
        brakes.derive(None, None, brake_li, brake_lo, brake_ri, brake_ro)
        self.assertAlmostEqual(brakes.array[4], 0.75)
        self.assertAlmostEqual(brakes.array[8], 1.0)


class TestCabinAltitude(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Cabin Press',)]
        opts = CabinAltitude.get_operational_combinations()
        self.assertEqual(opts,expected)

    def test_basic(self):
        cp = P(name='Cabin Press',
               array=np.ma.array([14.696, 10.108, 4.3727, 2.1490]),
               units=ut.PSI)
        ca = CabinAltitude()
        ca.derive(cp)
        expected = np.ma.array([0.0, 10000, 30000, 45000])
        ma_test.assert_masked_array_almost_equal (ca.array, expected, decimal=-3)


class TestClimbForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed','Fast')]
        opts = ClimbForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climb_for_flight_phases_basic(self):
        up_and_down_data = np.ma.array([0,0,2,5,3,] + [3]*10 + [2,5,6,8,0])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([0]+[100]*18+[0])))
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD Smoothed', up_and_down_data), phase_fast)
        expected = np.ma.array([0,0,2,5,0,] + [0]*10 + [0,3,4,6,0])
        ma_test.assert_masked_array_approx_equal(climb.array, expected)


class TestControlColumn(unittest.TestCase):

    def setUp(self):
        ccc = np.ma.array(data=[])
        self.ccc = P('Control Column (Capt)', ccc)
        ccf = np.ma.array(data=[])
        self.ccf = P('Control Column (FO)', ccf)

    def test_can_operate(self):
        expected = [('Control Column (Capt)', 'Control Column (FO)')]
        opts = ControlColumn.get_operational_combinations()
        self.assertEqual(opts, expected)

    @patch('analysis_engine.derived_parameters.blend_two_parameters')
    def test_control_column(self, blend_two_parameters):
        blend_two_parameters.return_value = [None, None, None]
        cc = ControlColumn()
        cc.derive(self.ccc, self.ccf)
        blend_two_parameters.assert_called_once_with(self.ccc, self.ccf)


class TestControlColumnForce(unittest.TestCase):

    def test_can_operate(self):
        expected = [('Control Column Force (Capt)',
                     'Control Column Force (FO)')]
        opts = ControlColumnForce.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_control_column_force(self):
        ccf = ControlColumnForce()
        ccf.derive(
            ControlColumnForce('Control Column Force (Capt)',
                               np.ma.array([0,1,2,3], mask=[True, False, True, False])),
            ControlColumnForce('Control Column Force (FO)',
                               np.ma.array([4,5,6,7], mask=[False, True, True, False])))
        self.assertEqual(ccf.array.tolist(), [4, 1, None, 10])


class TestControlWheel(unittest.TestCase):

    def setUp(self):
        cwc = np.ma.array(data=[])
        self.cwc = P('Control Wheel (Capt)', cwc)
        cwf = np.ma.array(data=[])
        self.cwf = P('Control Wheel (FO)', cwf)

    def test_can_operate(self):
        expected = ('Control Wheel (Capt)',
                    'Control Wheel (FO)',
                    'Control Wheel Synchro',
                    'Control Wheel Potentiometer')
        opts = ControlWheel.get_operational_combinations()
        self.assertIn(('Control Wheel Synchro',), opts)
        self.assertIn(('Control Wheel Potentiometer',), opts)
        self.assertIn(('Control Wheel (Capt)', 'Control Wheel (FO)'), opts)
        self.assertEqual(opts[-1], expected)
        self.assertEqual(len(opts), 13)

    @patch('analysis_engine.derived_parameters.blend_two_parameters')
    def test_control_wheel(self, blend_two_parameters):
        blend_two_parameters.return_value = [None, None, None]
        cw = ControlWheel()
        cw.derive(self.cwc, self.cwf)
        blend_two_parameters.assert_called_once_with(self.cwc, self.cwf)


class TestControlWheelForce(unittest.TestCase):

    def test_can_operate(self):
        expected = [('Control Wheel Force (Capt)',
                     'Control Wheel Force (FO)')]
        opts = ControlWheelForce.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_control_wheel_force(self):
        ccf = ControlWheelForce()
        ccf.derive(
            ControlWheelForce('Control Wheel Force (Capt)', np.ma.arange(10)),
            ControlWheelForce('Control Wheel Force (FO)', np.ma.arange(10)))
        np.testing.assert_array_almost_equal(ccf.array, np.ma.arange(0, 20, 2))


class TestControlColumnForceAtControlWheelCapt(unittest.TestCase):
    def setUp(self):
        self.node_class = ControlColumnForceAtControlWheelCapt

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Control Column Force (Capt)', 'Control Column (Capt)',),
                                                    family=A('Family', 'ATR-72')))

    def test_derive(self):
        cc_force_array = np.linspace(0, 100, 10)
        cc_force = P('Control Column Force (Capt)', (cc_force_array*np.sin(cc_force_array)))
        cc_angle_array = np.linspace(0, 25, 10)
        cc_angle = P('Control Column Force (Capt)', (cc_angle_array*np.sin(cc_angle_array)))

        node = self.node_class()
        node.derive(cc_force, cc_angle)
        self.assertEqual(node.name, 'Control Column Force At Control Wheel (Capt)')
        assert_array_almost_equal(node.array, [0, -4.05, -1.86, 11.42, 7.15, -16.79, -15.17, 19.45, 26.01, -18.59],
                                  decimal=2)


class TestControlColumnForceAtControlWheelFO(unittest.TestCase):
    def setUp(self):
        self.node_class = ControlColumnForceAtControlWheelFO

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Control Column Force (FO)', 'Control Column (FO)',),
                                                    family=A('Family', 'ATR-72')))

    def test_derive(self):
        cc_force_array = np.linspace(0, 100, 10)
        cc_force = P('Control Column Force (FO)', (cc_force_array*np.sin(cc_force_array)))
        cc_angle_array = np.linspace(0, 25, 10)
        cc_angle = P('Control Column Force (FO)', (cc_angle_array*np.sin(cc_angle_array)))
        node = self.node_class()
        node.derive(cc_force, cc_angle)
        self.assertEqual(node.name, 'Control Column Force At Control Wheel (FO)')
        assert_array_almost_equal(node.array, [0, -4.05, -1.86, 11.42, 7.15, -16.79, -15.17, 19.45, 26.01, -18.59],
                                  decimal=2)


class TestDescendForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed', 'Fast')]
        opts = DescendForFlightPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descend_for_flight_phases_basic(self):
        down_and_up_data = np.ma.array([0,0,12,5,3,] + [12]*10 + [12,15,10,7,0])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.concatenate((np.zeros(1), np.ones(18) * 100, np.zeros(1)))))
        descend = DescendForFlightPhases()
        descend.derive(Parameter('Altitude STD Smoothed', down_and_up_data), phase_fast)
        expected = np.ma.array([0,0,0,-7,-9,] + [0]*10 + [0,0,-5,-8,0])
        ma_test.assert_masked_array_approx_equal(descend.array, expected)


class TestSidestickAngleCapt(NodeTest, unittest.TestCase):
    def setUp(self):
        self.node_class = SidestickAngleCapt
        self.operational_combinations = [
            ('Sidestick Pitch (Capt)', 'Sidestick Roll (Capt)'),
        ]

    def test_derive(self):
        pitch_array = np.ma.arange(20)
        roll_array = pitch_array[::-1]
        pitch = P('Sidestick Pitch (Capt)', pitch_array)
        roll = P('Sidestick Roll (Capt)', roll_array)
        node = self.node_class()
        node.derive(pitch, roll)

        expected_array = np.ma.sqrt(pitch_array ** 2 + roll_array ** 2)
        assert_array_equal(node.array, expected_array)

    def test_derive_from_hdf(self):
        [pitch, roll, sidestick], phase = self.get_params_from_hdf(
            os.path.join(test_data_path, 'dual_input.hdf5'),
            ['Pitch Command (Capt)', 'Roll Command (Capt)', # old names
             self.node_class.get_name()])

        roll.array = align(roll, pitch)

        node = self.node_class()
        node.derive(pitch, roll)
        expected_array = np.ma.sqrt(pitch.array ** 2 + roll.array ** 2)
        assert_array_equal(node.array, expected_array)

        assert_array_equal(node.array, sidestick.array)


class TestSidestickAngleFO(NodeTest, unittest.TestCase):
    def setUp(self):
        self.node_class = SidestickAngleFO
        self.operational_combinations = [
            ('Sidestick Pitch (FO)', 'Sidestick Roll (FO)'),
        ]

    def test_derive(self):
        pitch_array = np.ma.arange(20)
        roll_array = pitch_array[::-1]
        pitch = P('Sidestick Pitch (FO)', pitch_array)
        roll = P('Sidestick Roll (FO)', roll_array)
        node = self.node_class()
        node.derive(pitch, roll)

        expected_array = np.ma.sqrt(pitch_array ** 2 + roll_array ** 2)
        assert_array_equal(node.array, expected_array)

    def test_derive_from_hdf(self):
        [pitch, roll, sidestick], phase = self.get_params_from_hdf(
            os.path.join(test_data_path, 'dual_input.hdf5'),
            ['Pitch Command (FO)', 'Roll Command (FO)',  # old names
             self.node_class.get_name()])

        roll.array = align(roll, pitch)

        node = self.node_class()
        node.derive(pitch, roll)
        expected_array = np.ma.sqrt(pitch.array ** 2 + roll.array ** 2)
        assert_array_equal(node.array, expected_array)

        np.testing.assert_array_almost_equal(node.array, sidestick.array)


class TestDistanceToLanding(unittest.TestCase):

    def test_can_operate(self):
        expected = [('Distance Travelled', 'Touchdown')]
        opts = DistanceToLanding.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_derive(self):
        distance_travelled = P('Distance Travelled', array=np.ma.arange(0, 100))
        tdwns = KTI('Touchdown', items=[KeyTimeInstance(70, 'Touchdown'),
                                        KeyTimeInstance(95, 'Touchdown')])

        expected_result = np.ma.concatenate((np.arange(70, -1, -1), np.arange(24, -1, -1), np.arange(1, 5, 1)))
        dtl = DistanceToLanding()
        dtl.derive(distance_travelled, tdwns)
        assert_array_equal(dtl.array, expected_result)

    def test_no_touchdown(self):
        distance_travelled = P('Distance Travelled', array=np.ma.arange(0, 10))
        tdwns = []
        dtl = DistanceToLanding()
        dtl.derive(distance_travelled, tdwns)
        expected_result = np.ma.array(data=np.zeros(10), mask=True)
        ma_test.assert_masked_array_equal(dtl.array, expected_result)


class TestDistanceFlown(unittest.TestCase):

    def setUp(self):
        self.node_class = DistanceFlown

    def test_can_operate(self):
        ops = self.node_class.get_operational_combinations()
        self.assertEqual(ops, [('Airspeed True', 'Airborne')])

    def test_derive_basic(self):
        tas=P('Airspeed True', array=np.ma.ones(20) * 360.0)
        airborne = buildsections('Airborne', [5, 17])
        node = self.node_class()
        node.get_derived((tas, airborne))
        expected = np.ma.concatenate((np.zeros(5), np.ma.arange(0.0, 1.0, 0.1), np.ones(5) * 1.1))
        assert_array_almost_equal(node.array, expected, decimal=1)


class TestDistanceTravelled(unittest.TestCase):

    def test_can_operate(self):
        expected = [('Groundspeed',)]
        opts = DistanceTravelled.get_operational_combinations()
        self.assertEqual(opts, expected)

    @patch('analysis_engine.derived_parameters.integrate')
    @patch('analysis_engine.derived_parameters.repair_mask')
    def test_derive(self, repair_mask, integrate):
        gndspeed = Mock()
        gndspeed.array = Mock()
        gndspeed.frequency = Mock()
        repair_mask.return_value = Mock()
        DistanceTravelled().derive(gndspeed)
        repair_mask.assert_called_once_with(gndspeed.array, gndspeed.frequency,
                                            repair_duration=None)
        integrate.assert_called_once_with(repair_mask.return_value, gndspeed.frequency,
                                          scale=1.0 / 3600)


class TestDrift(unittest.TestCase):

    def setUp(self):
        self.node_class = Drift

    def test_can_operate(self):
        self.assertTrue(Drift.can_operate(('Drift (1)',)))
        self.assertTrue(Drift.can_operate(('Drift (2)',)))
        self.assertTrue(Drift.can_operate(('Drift (1)', 'Drift (2)')))
        self.assertTrue(Drift.can_operate(('Track', 'Heading Continuous')))

    def test_combining_drift_signals(self):
        d1 = P('Drift (1)', array=np.ma.ones(10) * 1.0, offset = 0.2)
        d2 = P('Drift (2)', array=np.ma.ones(10) * -3.0, offset = 0.7)
        # blend_two_parameters complains if they are invariant!
        d1.array[5] = 0.0
        d2.array[5] = 0.0
        drift = Drift()
        drift.derive(d1, d2, None, None)
        self.assertEqual(drift.array[0], -1.0)

    def test_combining_track_and_heading(self):
        trk = P('Track', array=np.ma.ones(10) * 5.0)
        hdg = P('Heading Continuous', array=np.ma.zeros(10))
        drift = Drift()
        drift.derive(None, None, trk, hdg)
        self.assertEqual(drift.array[0], 5.0)

        trk = P('Track', array=np.ma.zeros(10))
        hdg = P('Heading Continuous', array=np.ma.ones(10) * 3.0)
        drift = Drift()
        drift.derive(None, None, trk, hdg)
        self.assertEqual(drift.array[0], -3.0)

        trk = P('Track', array=np.ma.ones(10) * 270.0)
        hdg = P('Heading Continuous', array=np.ma.ones(10) * (270.0 + 360.0))
        drift = Drift()
        drift.derive(None, None, trk, hdg)
        self.assertEqual(drift.array[0], 0.0)

    def test_derive_hdg_trk(self):
        hdg = P('Heading Continuous', array=np.ma.array([10, 10, 350, 350, 360, 360, 370, 370]))
        track = P('Track', array=np.ma.array([30, 350, 10, 330, 20, 340, 30, 350]))
        node = self.node_class()
        node.derive(None, None, track, hdg)

        assert_array_equal(node.array, np.ma.array([20, -20] * 4))
        self.assertEqual(node.hz, 1.0)
        self.assertEqual(node.offset, 0.0)

    def test_derive_drifts(self):
        array = np.ma.array([10, 20, 30, 40, 50])
        drift_1 = P('Drift (1)', array=array, frequency=0.5, offset=0.0)
        drift_2 = P('Drift (2)', array=array+5, frequency=0.5, offset=1.0)
        node = self.node_class()
        node.derive(drift_1, drift_2, None, None)

        expected = np.ma.array([12.5, 15, 20, 25, 30, 35, 40, 45, 50, 52.5])
        assert_array_almost_equal(node.array, expected)
        self.assertEqual(node.hz, 1.0)
        self.assertEqual(node.offset, 0.0)


class TestEng_EPRAvg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRAvg
        self.operational_combinations = [
            ('Eng (1) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR', 'Eng (4) EPR',),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_EPRMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRMax
        self.operational_combinations = [
            ('Eng (1) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR', 'Eng (4) EPR',),
        ]

    def test_derive(self):
        eng_1_array = np.ma.array([ 1.02,  1.09,  1.16,  1.23,  1.3 ,  1.37,  1.44,  1.51,  1.58,
                                    1.65,  1.72,  1.79])

        eng_2_array = np.ma.array([ 1.04,  1.11,  1.16,  1.2 ,  1.33,  1.34,  1.44,  1.51,  1.6,
                                    1.63,  1.71,  1.79])

        expected_array = eng_1_array
        expected_array[[0,1,4,8]] = eng_2_array[[0,1,4,8]]

        eng_1 = P(name='Eng (1) EPR', array=eng_1_array, frequency=1, offset=0.02734375)
        eng_2 = P(name='Eng (2) EPR', array=eng_2_array, frequency=1, offset=0.52734375)

        node = self.node_class()
        node.derive(eng_1, eng_2, None, None)

        assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.27734375)


class TestEng_EPRMin(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRMin
        self.operational_combinations = [
            ('Eng (1) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR',),
            ('Eng (1) EPR', 'Eng (2) EPR', 'Eng (3) EPR', 'Eng (4) EPR',),
        ]

    def test_derive(self):
        eng_1_array = np.ma.array([ 1.02,  1.09,  1.16,  1.23,  1.3 ,  1.37,  1.44,  1.51,  1.58,
                                    1.65,  1.72,  1.79])

        eng_2_array = np.ma.array([ 1.04,  1.11,  1.16,  1.2 ,  1.33,  1.34,  1.44,  1.51,  1.6,
                                    1.63,  1.71,  1.79])

        expected_array = eng_2_array
        expected_array[[0,1,4,8]] = eng_1_array[[0,1,4,8]]

        eng_1 = P(name='Eng (1) EPR', array=eng_1_array, frequency=1, offset=0.02734375)
        eng_2 = P(name='Eng (2) EPR', array=eng_2_array, frequency=1, offset=0.52734375)

        node = self.node_class()
        node.derive(eng_1, eng_2, None, None)

        assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.27734375)


class TestEng_EPRMinFor5Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_EPRMinFor5Sec
        self.operational_combinations = [('Eng (*) EPR Min',)]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_N1Avg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1Avg
        self.operational_combinations = [
            ('Eng (1) N1',),
            ('Eng (1) N1', 'Eng (2) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_N1Avg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )


class TestEng_N1Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1Max
        self.operational_combinations = [
            ('Eng (1) N1',),
            ('Eng (1) N1', 'Eng (2) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N1Max()
        eng.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )

    def test_derive_two_engines_offset(self):
        # this tests that average is performed on data sampled alternately.
        a = np.ma.arange(50, 55)
        b = np.ma.arange(54, 49, -1) + 0.2
        eng = Eng_N1Max()
        eng.derive(P('Eng (1)',a,offset=0.25), P('Eng (2)',b, offset=0.75), None, None)
        assert_array_equal(eng.array,np.ma.array([54.2, 53.2, 52.2, 53, 54]))
        self.assertEqual(eng.offset, 0)


class TestEng_N1Min(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1Min
        self.operational_combinations = [
            ('Eng (1) N1',),
            ('Eng (1) N1', 'Eng (2) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1',),
            ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10,20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N1Min()
        eng.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )


class TestEng_N1MinFor5Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N1MinFor5Sec
        self.operational_combinations = [('Eng (*) N1 Min',)]

    def test_derive(self):
        test_data = np.ma.concatenate((np.arange(5, 0, -1), np.arange(0, 5), np.ones(5) * 5))
        param=P('Eng (*) N1 Min', array=test_data, frequency=1.0)
        min5s = Eng_N1MinFor5Sec()
        min5s.derive(param)
        expected = np.ma.array(data=[5,4,3,3,3,3,3,3,3,4]+[5]*5,
                               mask=[0]*9+[1]*6)
        ma_test.assert_masked_array_equal(min5s.array, expected)

    @unittest.expectedFailure
    def test_real_data(self):
        test_data = np.ma.array([56,53.5,49.6,47.2,41.9,37.3,33.8,31.6,30.2,
                                 29.9,30.1,30,30.1,30.1,30.1,30.2,30.2,30.9,
                                 32.9,37.2,40.1,39.7,37.6,36.1,36,38.1,42.1,
                                 44.3,44.8,44.5,46.3,50.5,55.1,55.5,54.9,54.4])
        param=P('Eng (*) N1 Min', array=test_data, frequency=1.0)
        min5s = Eng_N1MinFor5Sec()
        min5s.derive(param)
        #The second_window process is not sensitive to min/max hence tracks
        #this data incorrectly. The peak at sample 20 should set the level
        #for 5 seconds, and the trough that follows should not go lower than
        #the highest value seen for 5 seconds.
        self.assertTrue(np.all(min5s.array[18:23] == np.ma.array([40.1]*5)))
        self.assertAlmostEqual(min5s.array[24], 39.7)

class TestEng_N2Avg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N2Avg
        self.operational_combinations = [
            ('Eng (1) N2',),
            ('Eng (1) N2', 'Eng (2) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_N2Avg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )


class TestEng_N2Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N2Max
        self.operational_combinations = [
            ('Eng (1) N2',),
            ('Eng (1) N2', 'Eng (2) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N2Max()
        eng.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )


class TestEng_N2Min(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N2Min
        self.operational_combinations = [
            ('Eng (1) N2',),
            ('Eng (1) N2', 'Eng (2) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2',),
            ('Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N2Min()
        eng.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )


class TestEng_N3Avg(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N3Avg
        self.operational_combinations = [
            ('Eng (1) N3',),
            ('Eng (1) N3', 'Eng (2) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_N3Avg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )


class TestEng_N3Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N3Max
        self.operational_combinations = [
            ('Eng (1) N3',),
            ('Eng (1) N3', 'Eng (2) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N3Max()
        eng.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )


class TestEng_N3Min(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_N3Min
        self.operational_combinations = [
            ('Eng (1) N3',),
            ('Eng (1) N3', 'Eng (2) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3',),
            ('Eng (1) N3', 'Eng (2) N3', 'Eng (3) N3', 'Eng (4) N3',),
        ]

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_N3Min()
        eng.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )


class TestEng_NpAvg(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_NpAvg.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) Np',))
        self.assertEqual(opts[-1], ('Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!


    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng_avg = Eng_NpAvg()
        eng_avg.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng_avg.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      6,7,8,9,10,11,12,13, # unmasked avg of two engines
                      9]) # only second engine value masked
        )


class TestEng_NpMax(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_NpMax.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) Np',))
        self.assertEqual(opts[-1], ('Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_NpMax()
        eng.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      11,12,13,14,15,16,17,18,9])
        )


class TestEng_NpMin(unittest.TestCase):
    def test_can_operate(self):
        opts = Eng_NpMin.get_operational_combinations()
        self.assertEqual(opts[0], ('Eng (1) Np',))
        self.assertEqual(opts[-1], ('Eng (1) Np', 'Eng (2) Np', 'Eng (3) Np', 'Eng (4) Np'))
        self.assertEqual(len(opts), 15) # 15 combinations accepted!

    def test_derive_two_engines(self):
        # this tests that average is performed on incomplete dependencies and
        # more than one dependency provided.
        a = np.ma.arange(0, 10)
        b = np.ma.arange(10, 20)
        a[0] = np.ma.masked
        b[0] = np.ma.masked
        b[-1] = np.ma.masked
        eng = Eng_NpMin()
        eng.derive(P('a',a), P('b',b), None, None)
        assert_array_equal(
            np.ma.filled(eng.array, fill_value=999),
            np.array([999, # both masked, so filled with 999
                      1,2,3,4,5,6,7,8,9])
        )


class TestFuelQty(unittest.TestCase):
    def test_can_operate(self):
        # testing for number of combinations possible, will operate with at
        # least one of the listed parameters. Not listing all operational
        # combinations as this can get very large (2**(n-1)-1) where n is the
        # number of parameters n-1 as both left and right are required if
        # either is avalibale (-1 as none is not a option)
        self.assertEqual(len(FuelQty.get_operational_combinations()), 2**7-2)
        opts = FuelQty.get_operational_combinations()
        self.assertTrue(('Fuel Qty (L)', 'Fuel Qty (C)', 'Fuel Qty (R)',
                         'Fuel Qty (Trim)', 'Fuel Qty (Aux)',
                         'Fuel Qty (Tail)', 'Fuel Qty (Stab)') in opts)

    def test_three_tanks(self):
        fuel_qty1 = P('Fuel Qty (L)',
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (C)',
                      array=np.ma.array([2,4,6], mask=[False, False, False]))
        # Mask will be interpolated by repair_mask.
        fuel_qty3 = P('Fuel Qty (R)',
                      array=np.ma.array([3,6,9], mask=[False, True, False]))
        fuel_qty_node = FuelQty()
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, fuel_qty3, None, None,
                             None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([6, 12, 18]))
        # Works without all parameters.
        fuel_qty_node.derive(fuel_qty1, *[None,]*6)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([1, 2, 3]))

    def test_four_tanks(self):
        fuel_qty1 = P('Fuel Qty (L)',
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (C)',
                      array=np.ma.array([2,4,6], mask=[False, False, False]))
        # Mask will be interpolated by repair_mask.
        fuel_qty3 = P('Fuel Qty (R)',
                      array=np.ma.array([3,6,9], mask=[False, True, False]))
        fuel_qty_a = P('Fuel Qty (Aux)',
                      array=np.ma.array([11,12,13], mask=[False, False, False]))
        fuel_qty_node = FuelQty()
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, fuel_qty3, fuel_qty_a,
                             None, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([17, 24, 31]))

    def test_masked_tank(self):
        fuel_qty1 = P('Fuel Qty (L)',
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (R)',
                      array=np.ma.array([2,4,6], mask=[True, True, True]))
        # Mask will be interpolated by repair_mask.
        fuel_qty_node = FuelQty()
        fuel_qty_node.derive(fuel_qty1, None, fuel_qty2, None, None, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([1, 2, 3]))

class TestFuelQtyC(unittest.TestCase):

    def setUp(self):
        self.node_class = FuelQtyC

    def test_can_operate(self):
        # testing for number of combinations possible, will operate with at
        # least one of the listed parameters. Not listing all operational
        # combinations as this can get very large (2**n-1) where n is the
        # number of parameters (-1 as none is not a option)
        opts = self.node_class.get_operational_combinations()
        self.assertTrue(('Fuel Qty (C) (1)',) in opts)
        self.assertTrue(('Fuel Qty (C) (2)',) in opts)
        self.assertTrue(('Fuel Qty (C) (3)',) in opts)
        self.assertTrue(('Fuel Qty (C) (1)', 'Fuel Qty (C) (2)',
                         'Fuel Qty (C) (3)') in opts)
        self.assertTrue(('Fuel Qty (C) (1)', 'Fuel Qty (C) (2)',
                         'Fuel Qty (C) (3)', 'Fuel Qty (C) (4)') in opts)
        self.assertEqual(len(opts), 15) # 2**4-1

    def test_three_tanks(self):
        fuel_qty1 = P('Fuel Qty (C) (1)',
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (C) (2)',
                      array=np.ma.array([2,4,6], mask=[False, False, False]))
        fuel_qty3 = P('Fuel Qty (C) (3)',
                      array=np.ma.array([3,6,9], mask=[False, True, False]))
        fuel_qty_node = self.node_class()
        fuel_qty_node.derive(fuel_qty1, None, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([1, 2, 3]))
        fuel_qty_node.derive(None, fuel_qty2, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([2, 4, 6]))
        fuel_qty_node.derive(None, fuel_qty2, fuel_qty3, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([5, 10, 15],
                                                  mask=[False, True, False]))
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, fuel_qty3, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([6, 12, 18],
                                                  mask=[False, True, False]))

class TestFuelQtyL(unittest.TestCase):

    def setUp(self):
        self.node_class = FuelQtyL

    def test_can_operate(self):
        # testing for number of combinations possible, will operate with at
        # least one of the listed parameters. Not listing all operational
        # combinations as this can get very large (2**n-1) where n is the
        # number of parameters (-1 as none is not a option)
        opts = self.node_class.get_operational_combinations()
        self.assertTrue(('Fuel Qty (L) (1)',) in opts)
        self.assertTrue(('Fuel Qty (L) (2)',) in opts)
        self.assertTrue(('Fuel Qty (L) (3)',) in opts)
        self.assertTrue(('Fuel Qty (L) (1)', 'Fuel Qty (L) (2)',
                         'Fuel Qty (L) (3)') in opts)
        self.assertTrue(('Fuel Qty (L) (1)', 'Fuel Qty (L) (2)',
                         'Fuel Qty (L) (3)', 'Fuel Qty (L) (4)') in opts)
        self.assertEqual(len(opts), 31) # 2**5-1

    def test_three_tanks(self):
        fuel_qty1 = P('Fuel Qty (L) (1)',
                      array=np.ma.array([1,2,3], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (L) (2)',
                      array=np.ma.array([2,4,6], mask=[False, False, False]))
        fuel_qty3 = P('Fuel Qty (L) (3)',
                      array=np.ma.array([3,6,9], mask=[False, True, False]))
        fuel_qty_node = self.node_class()
        fuel_qty_node.derive(fuel_qty1, None, None, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([1, 2, 3]))
        fuel_qty_node.derive(None, fuel_qty2, None, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([2, 4, 6]))
        fuel_qty_node.derive(None, fuel_qty2, fuel_qty3, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([5, 10, 15],
                                                  mask=[False, True, False]))
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, fuel_qty3, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([6, 12, 18],
                                                  mask=[False, True, False]))


class TestFuelQtyR(unittest.TestCase):

    def setUp(self):
        self.node_class = FuelQtyR

    def test_can_operate(self):
        # testing for number of combinations possible, will operate with at
        # least one of the listed parameters. Not listing all operational
        # combinations as this can get very large (2**n-1) where n is the
        # number of parameters (-1 as none is not a option)
        opts = self.node_class.get_operational_combinations()
        self.assertTrue(('Fuel Qty (R) (1)',) in opts)
        self.assertTrue(('Fuel Qty (R) (2)',) in opts)
        self.assertTrue(('Fuel Qty (R) (3)',) in opts)
        self.assertTrue(('Fuel Qty (R) (1)', 'Fuel Qty (R) (2)',
                         'Fuel Qty (R) (3)') in opts)
        self.assertTrue(('Fuel Qty (R) (1)', 'Fuel Qty (R) (2)',
                         'Fuel Qty (R) (3)', 'Fuel Qty (R) (4)') in opts)
        self.assertEqual(len(opts), 31) # 2**5-1

    def test_three_tanks(self):
        fuel_qty1 = P('Fuel Qty (R) (1)',
                      array=np.ma.array([3,2,1], mask=[False, False, False]))
        fuel_qty2 = P('Fuel Qty (R) (2)',
                      array=np.ma.array([6,4,2], mask=[False, False, False]))
        fuel_qty3 = P('Fuel Qty (R) (3)',
                      array=np.ma.array([9,6,3], mask=[False, True, False]))
        fuel_qty_node = self.node_class()
        fuel_qty_node.derive(fuel_qty1, None, None, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([3, 2, 1]))
        fuel_qty_node.derive(None, fuel_qty2, None, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([6, 4, 2]))
        fuel_qty_node.derive(None, fuel_qty2, fuel_qty3, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([15, 10, 5],
                                                  mask=[False, True, False]))
        fuel_qty_node.derive(fuel_qty1, fuel_qty2, fuel_qty3, None, None)
        assert_array_equal(fuel_qty_node.array,
                                      np.ma.array([18, 12, 6],
                                                  mask=[False, True, False]))


class TestFuelQtyAux(unittest.TestCase):

    def setUp(self):
        self.node_class = FuelQtyAux

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertIn(('Fuel Qty (Aux) (1)', 'Fuel Qty (Aux) (2)'), opts)
        self.assertIn(('Fuel Qty (Aux) (1)',), opts)
        self.assertIn(('Fuel Qty (Aux) (2)',), opts)
        self.assertEqual(len(opts), 3)

    def test_derive(self):
        fq1 = P('Fuel Qty (Aux) (1)', np.ma.array([40,30,20,10]))
        fq2 = P('Fuel Qty (Aux) (2)', np.ma.array([10,20,30,40]))

        dfq = self.node_class()
        dfq.derive(fq1, None)
        assert_array_equal(dfq.array, fq1.array)

        dfq.derive(None, fq2)
        assert_array_equal(dfq.array, fq2.array)

        dfq.derive(fq1, fq2)
        assert_array_equal(dfq.array, np.ma.ones(4) * 50)

class TestGrossWeightSmoothed(unittest.TestCase):

    def test_can_operate(self):
        expected = ('Eng (*) Fuel Flow','Gross Weight', 'Climbing',
                     'Descending', 'Airborne')
        self.assertTrue(GrossWeightSmoothed.can_operate(expected))
        self.assertTrue(GrossWeightSmoothed.can_operate(('Gross Weight')))

    def test_gw_real_data_1(self):
        ff = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_1_ff.nod'))
        gw = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_1_gw.nod'))
        gw_orig = gw.array.copy()
        try:
            climbs = load(os.path.join(test_data_path,
                                       'gross_weight_smoothed_1_climbs.nod'))
        except AttributeError: # Python 3
            climbs = buildsection('Climbing', 1018, 1463, 1017.671875,
                                  1462.671875)
        try:
            descends = load(os.path.join(test_data_path,
                                         'gross_weight_smoothed_1_descends.nod'))
        except AttributeError: # Python 3
            descends = buildsections('Descending',
                                     [1602, 2219, 1601.671875, 2218.671875],
                                     [2230, 2248, 2229.671875, 2247.671875],
                                     [2254, 2263, 2253.671875, 2262.671875],
                                     [2269, 2278, 2268.671875, 2277.671875],
                                     [2280, 2365, 2279.671875, 2364.671875],
                                     [2416, 2607, 2415.671875, 2606.671875],)
        try:
            fast = load(os.path.join(test_data_path,
                                     'gross_weight_smoothed_1_fast.nod'))
        except AttributeError: # Python 3
            fast = buildsection('Fast', 991, 2630, 990.53125, 2629.53125)
        gws = GrossWeightSmoothed()
        gws.derive(ff, gw, climbs, descends, fast)
        # Start is similar.
        self.assertTrue(abs(gws.array[640] - gw_orig[640]) < 30)
        # Climbing diverges.
        self.assertTrue(abs(gws.array[1150] - gw_orig[1150]) < 260)
        # End is similar.
        self.assertTrue(abs(gws.array[2500] - gw_orig[2500]) < 30)

    def test_gw_real_data_2(self):
        ff = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_2_ff.nod'))
        gw = load(os.path.join(test_data_path,
                               'gross_weight_smoothed_2_gw.nod'))
        gw_orig = gw.array.copy()
        try:
            climbs = load(os.path.join(test_data_path,
                                       'gross_weight_smoothed_2_climbs.nod'))
        except AttributeError: # Python 3
            climbs = buildsection('Climbing',712, 1725, 711.671875,
                                  1724.671875)
        try:
            descends = load(os.path.join(test_data_path,
                                         'gross_weight_smoothed_2_descends.nod'))
        except AttributeError: # Python 3
            descends = buildsections('Descending',
                                     [4739, 6109, 4738.671875, 6108.671875],
                                     [6123, 6136, 6122.671875, 6135.671875],
                                     [6138, 6174, 6137.671875, 6173.671875],
                                     [6180, 6342, 6179.671875, 6341.671875],
                                     [6392, 6523, 6391.671875, 6522.671875],)

        try:
            fast = load(os.path.join(test_data_path,
                                     'gross_weight_smoothed_2_fast.nod'))
        except AttributeError: # Python 3
            fast = buildsection('Fast', 693, 6552, 692.53125, 6551.53125)
        gws = GrossWeightSmoothed()
        gws.derive(ff, gw, climbs, descends, fast)
        # Start is similar.
        self.assertTrue(abs(gws.array[600] - gw_orig[600]) < 35)
        # Climbing diverges.
        self.assertTrue(abs(gws.array[1500] - gw_orig[1500]) < 180)
        # Descending diverges.
        self.assertTrue(abs(gws.array[5800] - gw_orig[5800]) < 120)

    def test_gw_masked(self):
        weight = P('Gross Weight',np.ma.array([292,228,164,100],dtype=float),offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.ones(256) * 3600, offset=0.0, frequency=1.0)
        weight_aligned = align(weight, fuel_flow)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 40, 50)
        fast = buildsection('Fast', None, None)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        assert_equal(result.array, weight_aligned)

    def test_gw_formula(self):
        weight = P('Gross Weight',np.ma.array([292,228,164,100],dtype=float),offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.ones(256) * 3600, offset=0.0, frequency=1.0)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 40, 50)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 292.0)
        self.assertEqual(result.array[-1], 37.0)

    def test_gw_formula_with_many_samples(self):
        weight = P('Gross Weight', np.ma.arange(56400, 50000, -64),
                   offset=0.0, frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.ones(6400) * 3600,
                      offset=0.0, frequency=1.0)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 50, 60)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[1], 56400-1)

    def test_gw_formula_with_good_data(self):
        weight = P('Gross Weight', np.ma.array(data=[484, 420, 356, 292, 228, 164, 100],
                                               mask=[1, 0, 0, 0, 0, 1, 0], dtype=float),
                   offset=0.0, frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.ones(448) * 3600,
                      offset=0.0, frequency=1.0)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 60, 70)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)

    def test_gw_formula_with_rising_endpoint(self):
        weight = P('Gross Weight', np.ma.array(data=[484, 420, 356, 292, 228, 164, 500],
                                               mask=[1, 0, 0, 0, 0, 0, 0], dtype=float),
                   offset=0.0, frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.ones(448) * 3600,
                      offset=0.0, frequency=1.0)
        climb = buildsection('Climbing', 10, 20)
        descend = buildsection('Descending', 60, 70)
        airs = buildsection('Airborne', 10, 350)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, airs])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)

    def test_gw_formula_climbing(self):
        weight = P('Gross Weight',np.ma.array(data=[484, 420, 356, 292, 228, 164, 100],
                                              mask=[1,0,0,0,0,1,0],dtype=float),
                   offset=0.0,frequency=1/64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.ones(448) * 3600)
        climb = buildsection('Climbing', 1, 4)
        descend = buildsection('Descending', 20, 30)
        fast = buildsection('Fast', 10, len(fuel_flow.array))
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)

    def test_gw_descending(self):
        weight = P('Gross Weight',np.ma.array(
            data=[484, 420, 356, 292, 228, 164, 100],
            mask=[1, 0, 0, 0, 0, 1, 0], dtype=float),
                   frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.ones(448) * 3600)
        gws = GrossWeightSmoothed()
        climb = S('Climbing')
        descend = buildsection('Descending', 3, 5)
        fast = buildsection('Fast', 50, 450)
        gws = GrossWeightSmoothed()
        result = gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(result.array[0], 484.0)
        self.assertEqual(result.array[-1], 37.0)

    def test_gw_one_masked_data_point(self):
        weight = P('Gross Weight',np.ma.array(data=[0],
                                              mask=[1],dtype=float),
                   frequency=1 / 64.0)
        fuel_flow = P('Eng (*) Fuel Flow', np.ma.zeros(64))
        gws = GrossWeightSmoothed()
        climb = S('Climbing')
        descend = S('Descending')
        fast = buildsection('Fast', 0, 1)
        gws = GrossWeightSmoothed()
        gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(len(gws.array),64)
        self.assertEqual(gws.frequency, fuel_flow.frequency)
        self.assertEqual(gws.offset, fuel_flow.offset)

    def test_gw_no_fuel_flow(self):
        weight = P('Gross Weight',np.ma.array(data=[0],
                                              mask=[1], dtype=float),
                   frequency=1 / 64.0)
        fuel_flow = None
        gws = GrossWeightSmoothed()
        climb = S('Climbing')
        descend = S('Descending')
        fast = buildsection('Fast', 0, 1)
        gws = GrossWeightSmoothed()
        gws.get_derived([fuel_flow, weight, climb, descend, fast])
        self.assertEqual(gws.array, weight.array)
        self.assertEqual(gws.frequency, weight.frequency)
        self.assertEqual(gws.offset, weight.offset)


class TestGroundspeed(unittest.TestCase):

    def setUp(self):
        self.node_class = Groundspeed
        self.precise = A('Precise Positioning', value=True)

    def test_can_operate(self):
        self.assertFalse(Groundspeed.can_operate([], ac_type=aeroplane))
        self.assertTrue(Groundspeed.can_operate(['Groundspeed (1)'],
                                                ac_type=aeroplane))
        self.assertTrue(Groundspeed.can_operate(['Groundspeed (2)'],
                                                ac_type=aeroplane))
        self.assertTrue(Groundspeed.can_operate(['Groundspeed (1)',
                                                 'Groundspeed (2)'],
                                                ac_type=aeroplane))
        self.assertFalse(Groundspeed.can_operate([], ac_type=helicopter))
        self.assertTrue(Groundspeed.can_operate(['Groundspeed (1)',
                                                 'Groundspeed (2)'],
                                                ac_type=helicopter))
        self.assertFalse(Groundspeed.can_operate(['Latitude Prepared'],
                                                 ac_type=helicopter))
        self.assertFalse(Groundspeed.can_operate(['Longitude Prepared'],
                                                 ac_type=helicopter))
        self.assertFalse(Groundspeed.can_operate(['Latitude Prepared',
                                                  'Longitude Prepared'],
                                                 ac_type=helicopter))
        self.assertTrue(Groundspeed.can_operate(['Latitude Prepared',
                                                 'Longitude Prepared'],
                                                ac_type=helicopter,
                                                precise=self.precise ))

    def test_basic(self):
        one = P('Groundspeed (1)', np.ma.array([100,200,300]), frequency=0.5, offset=0.0)
        two = P('Groundspeed (2)', np.ma.array([150,250,350]), frequency=0.5, offset=1.0)
        gs = Groundspeed()
        gs.derive(one, two, None, None, aeroplane)
        # Note: end samples are not 100 & 350 due to method of merging.
        assert_array_equal(gs.array[1:-1], np.array([150, 200, 250, 300]))
        self.assertEqual(gs.frequency, 1.0)
        self.assertEqual(gs.offset, 0.0)

    def test_derive_helicopter(self):
        '''
        Test data based on synthesized constant acceleration North at 0.1g for 10 seconds
        '''
        latitude_array = [0,0,0,8.82829E-06,2.64849E-05,5.29698E-05,
                          8.82829E-05,0.000132424,0.000185394,0.000247192,
                          0.000317819,0.000397273,0.000485556,0.000573839,
                          0.000662122,0.000750405,0.000838688,0.000926971,
                          0.001015254]
        latitude = P('latitude', array=np.ma.array(latitude_array))
        longitude_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        longitude = P('latitude', array=np.ma.array(longitude_array))

        node = self.node_class()
        node.derive(None, None, latitude, longitude, helicopter)

        expected = [0,0,0.95428,2.862842,4.771405,6.679961089,
                    8.588521401,10.49708171,12.40564202,14.31420233,
                    16.22276265,18.13132296] + [19.08560311]*7
        assert_array_almost_equal(node.array, expected, decimal=1)

    def test_derive_helicopter_east_at_60(self):
        '''
        Similar data East at 60deg North results in half the velocity due to curvature of the Earth.
        '''
        latitude_array = [60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60]
        latitude = P('latitude', array=np.ma.array(latitude_array))
        longitude_array = [0,0,0,8.82829E-06,2.64849E-05,5.29698E-05,
                          8.82829E-05,0.000132424,0.000185394,0.000247192,
                          0.000317819,0.000397273,0.000485556,0.000573839,
                          0.000662122,0.000750405,0.000838688,0.000926971,
                          0.001015254]
        longitude = P('latitude', array=np.ma.array(longitude_array))

        node = self.node_class()
        node.derive(None, None, latitude, longitude, helicopter)

        expected = [0,0,0.954280156,2.862840467,4.771400778,6.679961089,
                    8.588521401,10.49708171,12.40564202,14.31420233,
                    16.22276265,18.13132296] + [19.08560311]*7
        # At 60degN, the degrees longitude cover half the distance, so the expected speeds are also halved.
        assert_array_almost_equal(node.array, [e/2.0 for e in expected], decimal=3)

    def test_derive_helicopter_null_island_trap(self):
        # In case (0,0) locations get through...
        latitude_array = [0,0,0,1,1,1.0]
        latitude = P('latitude', array=np.ma.array(latitude_array))
        longitude_array = [0,0,0,0,0,0]
        longitude = P('latitude', array=np.ma.array(longitude_array))

        node = self.node_class()
        node.derive(None, None, latitude, longitude, helicopter)

        expected = np.ma.array(data=np.zeros(6), mask=[0,0,1,1,0,0])
        assert_array_equal(node.array, expected)
        assert_array_equal(node.array.mask, expected.mask)


class TestGroundspeedSigned(unittest.TestCase):

    def setUp(self):
        self.node_class = GroundspeedSigned

    def test_can_operate(self):
        self.assertTrue(GroundspeedSigned.can_operate(['Groundspeed', 'Eng (*) Any Running',
                                                       'Aircraft Type', 'Precise Positioning', 'Taxiing',
                                                       'Latitude Prepared', 'Longitude Prepared']))


    def test_basic(self):
        gspd = P('Groundspeed', np.ma.ones(30))
        running = P('Eng (*) Any Running', np.ma.concatenate((np.zeros(15), np.ones(15))))
        precision = A(name='Precise Positioning', value=False)
        taxiing = buildsection('Taxiing', 0, 15)
        lat = P('Latitude Prepared', np.ma.zeros(30))
        lon = P('Longitude Prepared', np.ma.zeros(30))
        gs = GroundspeedSigned()
        gs.derive(gspd, running, precision, taxiing, lat, lon)
        assert_equal(gs.array[4], -1.0)
        assert_equal(gs.array[24], 1.0)
        # Make sure we did not mutate gspd
        assert_equal(gspd.array, np.ma.ones(30))

    def test_early_start(self):
        gspd = P('Groundspeed', np.ma.concatenate((np.ones(15), np.zeros(10), np.arange(15))))
        running = P('Eng (*) Any Running', np.ma.concatenate((np.zeros(20), np.ones(20))))
        gs = GroundspeedSigned()
        gs.derive(gspd, running)
        assert_equal(gs.array[4], -1.0)
        assert_equal(gs.array[22], 0.0)
        assert_equal(gs.array[27], 2.0)

    def test_no_pushback(self):
        gspd = P('Groundspeed', np.ma.concatenate((np.ones(15), np.zeros(10), np.arange(15))))
        running = P('Eng (*) Any Running', np.ma.ones(40))
        gs = GroundspeedSigned()
        gs.derive(gspd, running)
        assert_equal(gs.array, gspd.array)

    def test_scaling_correction(self):
        lat_data = []
        lon_data = []
        gspd_data = []
        this_test_data_path = os.path.join(test_data_path,
                                           'Groundspeed_test_data_Entebbe.csv')
        with open(this_test_data_path, 'rt') as csvfile:
            self.reader = csv.DictReader(csvfile)
            for row in self.reader:
                lat_data.append(float(row['Latitude']))
                lon_data.append(float(row['Longitude']))
                gspd_data.append(float(row['Groundspeed']))
        gspd = P('Groundspeed', gspd_data)
        taxiing = buildsection('Taxiing', 0, len(lat_data))
        ac_type = A(name='Aircraft Type', value = 'aeroplane')
        precision = A(name='Precise Positioning', value = True)
        lat = P('Latitude Prepared', lat_data)
        lon = P('Longitude Prepared', lon_data)
        running=P('Eng (*) Any Running', np_ma_ones_like(lat.array))
        gs = GroundspeedSigned()
        gs.derive(gspd, running, ac_type, precision, taxiing, lat, lon)
        # This case reported a groundspeed exceedance of over 60kt (ref 11151421)
        self.assertGreater(np.max(gspd_data), 40)
        # but the real speed was about 32kt
        self.assertLess(np.max(gs.array), 40)
        # and check the aircraft type
        ac_type = A(name='Aircraft Type', value = 'balloon')
        gspd = P('Groundspeed', gspd_data)
        gs.derive(gspd, running, ac_type, precision, taxiing, lat, lon)
        self.assertGreater(np.max(gs.array), 60)
        # and check not precise positioning
        ac_type = A(name='Aircraft Type', value = 'aeroplane')
        gspd = P('Groundspeed', gspd_data)
        precision = A(name='Precise Positioning', value = False)
        gs.derive(gspd, running, ac_type, precision, taxiing, lat, lon)
        self.assertGreater(np.max(gs.array), 60)

    def test_scaling_correction_ineffective_if_not_precise(self):
        lat_data=[]
        lon_data=[]
        gspd_data=[]
        this_test_data_path = os.path.join(test_data_path,
                                           'Groundspeed_test_data_Entebbe.csv')
        with open(this_test_data_path, 'rt') as csvfile:
            self.reader = csv.DictReader(csvfile)
            for row in self.reader:
                lat_data.append(float(row['Latitude']))
                lon_data.append(float(row['Longitude']))
                gspd_data.append(float(row['Groundspeed']))
        gspd = P('Groundspeed', gspd_data)
        taxiing = buildsection('Taxiing', 0, len(lat_data))
        ac_type = A(name='Aircraft Type', value = 'aeroplane')
        precision = A(name='Precise Positioning', value = False)
        lat = P('Latitude Prepared', lat_data)
        lon = P('Longitude Prepared', lon_data)
        running=P('Eng (*) Any Running', np_ma_ones_like(lat.array))
        gs = GroundspeedSigned()
        gs.derive(gspd, running, ac_type, precision, taxiing, lat, lon)
        self.assertGreater(np.max(gs.array), 40)

'''
class TestGroundspeedAlongTrack(unittest.TestCase):

    @unittest.skip('Commented out until new computation of sliding motion')
    def test_can_operate(self):
        expected = [('Groundspeed','Acceleration Along Track', 'Altitude AAL',
                     'ILS Glideslope')]
        opts = GroundspeedAlongTrack.get_operational_combinations()
        self.assertEqual(opts, expected)

    @unittest.skip('Commented out until new computation of sliding motion')
    def test_groundspeed_along_track_basic(self):
        gat = GroundspeedAlongTrack()
        gspd = P('Groundspeed', np.ma.concatenate((np.ones(2) * 100, np.ones(18) * 120)))
        accel = P('Acceleration Along Track', np.ma.zeros(20))
        gat.derive(gspd, accel)
        # A first order lag of 6 sec time constant rising from 100 to 120
        # will pass through 110 knots between 13 & 14 seconds after the step
        # rise.
        self.assertLess(gat.array[5], 56.5)
        self.assertGreater(gat.array[6], 56.5)

    @unittest.skip('Commented out until new computation of sliding motion')
    def test_groundspeed_along_track_accel_term(self):
        gat = GroundspeedAlongTrack()
        gspd = P('Groundspeed',np.ma.ones(200) * 100, frequency=1)
        accel = P('Acceleration Along Track', np.ma.ones(200) * 0.1, frequency=1)
        accel.array[0] = 0.0
        gat.derive(gspd, accel)
        # The resulting waveform takes time to start going...
        self.assertLess(gat.array[4],55.0)
        # ...then rises under the influence of the lag...
        self.assertGreater(gat.array[16],56.0)
        # ...to a peak...
        self.assertGreater(np.ma.max(gat.array.data),16)
        # ...and finally decays as the longer washout time constant takes effect.
        self.assertLess(gat.array[199],52.0)
'''

#class TestHeadingContinuous(unittest.TestCase):
    #def test_can_operate(self):
        #expected = [('Heading',)]
        #opts = HeadingContinuous.get_operational_combinations()
        #self.assertEqual(opts, expected)

    #def test_heading_continuous(self):
        #head = HeadingContinuous()
        #head.derive(P('Heading',np.ma.remainder(
            #np.ma.arange(10)+355,360.0)))

        #answer = np.ma.array(data=[355.0, 356.0, 357.0, 358.0, 359.0, 360.0,
                                   #361.0, 362.0, 363.0, 364.0],
                             #dtype=np.float, mask=False)

        ##ma_test.assert_masked_array_approx_equal(res, answer)
        #assert_array_equal(head.array.data, answer.data)

class TestHeadingContinuous(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = HeadingContinuous
        self.operational_combinations = [('Heading',),
                                         ('Heading (Capt)', 'Heading (FO)'),
                                         ('Heading', 'Heading (Capt)',),
                                         ('Heading', 'Heading (FO)'),
                                         ('Heading', 'Heading (Capt)', 'Heading (FO)'),
                                         ('Heading','Frame'),
                                         ('Heading (Capt)', 'Heading (FO)','Frame'),
                                         ('Heading', 'Heading (Capt)','Frame'),
                                         ('Heading', 'Heading (FO)','Frame'),
                                         ('Heading', 'Heading (Capt)', 'Heading (FO)','Frame')
                                         ]

    def test_heading_continuous_basic(self):
        hdg = P('Heading',np.ma.remainder(np.ma.arange(10)+355,360.0))
        hdg.array[2] = np.ma.masked
        node = self.node_class()
        node.derive(hdg, None, None)
        expected = np.ma.array(data=[355.0, 356.0, 357.0, 358.0, 359.0, 360.0,
                                     361.0, 362.0, 363.0, 364.0],
                               dtype=np.float, mask=False)
        assert_equal(node.array, expected)

    def test_heading_continuous_merged(self):
        hdg = P('Heading',np.ma.remainder(np.ma.arange(10)+355,360.0))
        hdg_ca = P('Heading (Capt)',np.ma.array([5,6,7,8,9.0]),offset=0.1,frequency=0.5)
        hdg_fo = P('Heading (FO)',np.ma.array([15,16,17,18,19.0]),offset=1.1,frequency=0.5)
        node = self.node_class()
        node.derive(hdg, hdg_ca, hdg_fo)
        expected = np.ma.array(data=np.arange(10)/2.0+9.75,
                               dtype=np.float, mask=False)
        expected[0]=10.0
        expected[-1]=14.0
        assert_equal(node.array, expected)
        self.assertEqual(node.offset, 0.1)
        self.assertEqual(node.frequency, 1,0)

    def test_heading_continuous_merged_rollover(self):
        hdg = P('Heading',np.ma.remainder(np.ma.arange(10)+355,360.0))
        hdg_ca = P('Heading (Capt)',np.ma.array([358,2,6,10, 14.0]),offset=0.1,frequency=0.5)
        hdg_ca.array[2]=np.ma.masked
        hdg_fo = P('Heading (FO)',np.ma.array([346,350,354,358,2.0]),offset=1.1,frequency=0.5)
        hdg_fo.array[3]=np.ma.masked
        node = self.node_class()
        node.derive(hdg, hdg_ca, hdg_fo)
        expected = np.ma.array(data=np.arange(10)*2.0+351.0,
                               dtype=np.float, mask=False)
        expected[0]=352.0
        expected[-1]=368.0
        assert_equal(node.array, expected)
        self.assertEqual(node.offset, 0.1)
        self.assertEqual(node.frequency, 1,0)

    def test_heading_continuous_not_hercules(self):
        hdg = P('Heading',np.ma.array(data=[10]*60, mask=[0]*20+[1]*20+[0]*20, dtype=float))
        con_hdg = HeadingContinuous()
        con_hdg.derive(hdg, None, None, None)
        # REPAIR_DURATION is limited to 10 seconds, so this should not be repaired.
        self.assertEqual(np.ma.count(con_hdg.array), 40)

    def test_heading_continuous_hercules(self):
        hdg = P('Heading', np.ma.array(data=[10]*60, mask=[0]*20+[1]*20+[0]*20, dtype=float))
        con_hdg = HeadingContinuous()
        herc = A('Frame', 'L382-Hercules')
        con_hdg.derive(hdg, None, None, herc)
        self.assertEqual(np.ma.count(con_hdg.array), 60)

    def test_heading_continuous_starting_north(self):
        hdg_ca = P('Heading (Capt)',np.ma.array([358,358,0.0,1.0]))
        hdg_fo = P('Heading (FO)',np.ma.array([0.0,1.0,359,357]),offset=0.5)
        node = self.node_class()
        node.derive(None, hdg_ca, hdg_fo, None)
        expected = np.ma.array([359,359,359.25,360,360,359.75,359.5,359 ])
        assert_equal(node.array, expected)


class TestTrack(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Track
        self.operational_combinations = [('Heading', 'Drift')]

    def test_derive_basic(self):
        heading = P('Heading',
                    array=np.ma.array([14.1, 29.2, 59.3, 119.4, 239.5, 359.6, 449.7, 539.8, 629.9, 720.0]))
        drift = P('Drift',
                  array=np.ma.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]))
        node = self.node_class()
        node.derive(heading, drift)
        expected = [15.0, 30.0, 60.0, 120.0, 240.0, 0.0, 90.0, 180.0, 270.0, 0.0]
        assert_equal(node.array, expected)


class TestTrackTrue(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TrackTrue
        self.operational_combinations = [('Heading True', 'Drift')]

    def test_derive_basic(self):
        heading = P('Heading True',
                    array=np.ma.array([14.1, 29.2, 59.3, 119.4, 239.5, 359.6, 449.7, 539.8, 629.9, 720.0]))
        drift = P('Drift',
                      array=np.ma.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]))
        node = self.node_class()
        node.derive(heading, drift)
        expected = [15.0, 30.0, 60.0, 120.0, 240.0, 0.0, 90.0, 180.0, 270.0, 0.0]
        assert_equal(node.array, expected)

    #def test_derive_basic(self):
        #heading = Parameter('Heading True', array=np.ma.arange(0, 100, 10, dtype=np.float64))
        #drift = Parameter('Drift', array=np.ma.arange(0, 1, 0.1))
        #node = self.node_class()
        #node.derive(heading, None)
        #expected = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        #assert_equal(node.array, expected)
        #node.derive(heading, drift)
        #expected = [0, 10.1, 20.2, 30.3, 40.4, 50.5, 60.6, 70.7, 80.8, 90.9]
        #assert_equal(node.array, expected)

    def test_derive_heading_extra(self):
        # Compare IRU Track Angle True (recorded) against the derived:
        heading = load(os.path.join(test_data_path, 'HeadingTrack_Heading_True.nod'))
        drift = load(os.path.join(test_data_path, 'HeadingTrack_Drift.nod'))
        node = self.node_class()
        node.derive(heading, drift)
        expected = load(os.path.join(test_data_path, 'HeadingTrack_IRU_Track_Angle_Recorded.nod'))
        assert_array_within_tolerance(node.array % 360, expected.array, 10, 98)


class TestTrackContinuous(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TrackContinuous
        self.operational_combinations = [('Heading Continuous', 'Drift')]

    def test_derive_basic(self):
        heading = P('Heading Continuous',
                    array=np.ma.array([14.1, 29.2, 59.3, 119.4, 239.5, 359.6, 449.7, 539.8, 629.9, 720.0]))
        drift = P('Drift',
                  array=np.ma.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]))
        node = self.node_class()
        node.derive(heading, drift)
        expected = [15.0, 30.0, 60.0, 120.0, 240.0, 360.0, 450.0, 540.0, 630.0, 720.0]
        assert_equal(node.array, expected)


class TestTrackTrueContinuous(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TrackTrueContinuous
        self.operational_combinations = [('Heading True Continuous', 'Drift')]

    def test_derive_basic(self):
        heading = P('Heading True Continuous',
                    array=np.ma.array([14.1, 29.2, 59.3, 119.4, 239.5, 359.6, 449.7, 539.8, 629.9, 720.0]))
        drift = P('Drift',
                      array=np.ma.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]))
        node = self.node_class()
        node.derive(heading, drift)
        expected = [15.0, 30.0, 60.0, 120.0, 240.0, 360.0, 450.0, 540.0, 630.0, 720.0]
        assert_equal(node.array, expected)


class TestTrackDeviationFromRunway(unittest.TestCase):

    # TODO: For all these combination there is just one testcase calling a derive once
    def test_can_operate(self):
        self.assertEqual(
            TrackDeviationFromRunway.get_operational_combinations(),
            [('Track True Continuous', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Approach Information'),
             ('Track Continuous', 'FDR Takeoff Runway'),
             ('Track Continuous', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Track Continuous', 'Approach Information'),
             ('Track True Continuous', 'Takeoff Roll Or Rejected Takeoff', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Takeoff Roll Or Rejected Takeoff', 'Approach Information'),
             ('Track True Continuous', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track Continuous', 'Takeoff Roll Or Rejected Takeoff', 'FDR Takeoff Runway'),
             ('Track Continuous', 'Takeoff Roll Or Rejected Takeoff', 'Approach Information'),
             ('Track Continuous', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'Takeoff Roll Or Rejected Takeoff', 'FDR Takeoff Runway'),
             ('Track True Continuous', 'Track Continuous', 'Takeoff Roll Or Rejected Takeoff', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True Continuous', 'Takeoff Roll Or Rejected Takeoff', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track Continuous', 'Takeoff Roll Or Rejected Takeoff', 'FDR Takeoff Runway', 'Approach Information'),
             ('Track True Continuous', 'Track Continuous', 'Takeoff Roll Or Rejected Takeoff', 'FDR Takeoff Runway', 'Approach Information')
             ]
        )

    def test_deviation(self):
        apps = App(items=[ApproachItem(
            'LANDING', slice(8763, 9037),
            airport={'code': {'iata': 'FRA', 'icao': 'EDDF'},
                     'distance': 2.2981699358981746,
                     'id': 2289,
                     'latitude': 50.0264,
                     'location': {'city': 'Frankfurt-Am-Main',
                                  'country': 'Germany'},
                     'longitude': 8.54313,
                     'magnetic_variation': 'E000459 0106',
                     'name': 'Frankfurt Am Main'},
            landing_runway={'end': {'latitude': 50.027542, 'longitude': 8.534175},
                    'glideslope': {'angle': 3.0,
                                   'latitude': 50.037992,
                                   'longitude': 8.582733,
                                   'threshold_distance': 1098},
                    'id': 4992,
                    'identifier': '25L',
                    'localizer': {'beam_width': 4.5,
                                  'frequency': 110700.0,
                                  'heading': 249,
                                  'latitude': 50.026722,
                                  'longitude': 8.53075},
                    'magnetic_heading': 248.0,
                    'start': {'latitude': 50.040053, 'longitude': 8.586531},
                    'strip': {'id': 2496,
                              'length': 13123,
                              'surface': 'CON',
                              'width': 147}},
            turnoff=8998.2717013888887)])
        heading_track = load(os.path.join(test_data_path, 'HeadingDeviationFromRunway_heading_track.nod'))
        to_runway = load(os.path.join(test_data_path, 'HeadingDeviationFromRunway_runway.nod'))
        takeoff = buildsections('Takeoff', [963.22222222222217, 1145.751417357174])
        takeoff.frequency = 2.0
        takeoff.offset=0.1328125
        deviation = TrackDeviationFromRunway()
        deviation.get_derived((heading_track, None, takeoff, to_runway, apps))
        # check average stays close to 0
        err = abs(np.ma.average(deviation.array[8775:8975]))
        self.assertTrue(err < 1.5)
        self.assertAlmostEqual(np.ma.min(deviation.array[8775:8975]), -10.5, places = 1)
        self.assertAlmostEqual(np.ma.max(deviation.array[8775:8975]), 12.3, places = 1)

    def test_derive__multiple_approaches(self):

        apps = App(items=[
            ApproachItem(
                    'GO_AROUND', slice(500, 1500),
                    airport={},
                    approach_runway={
                        'magnetic_heading': 45.0}),
            ApproachItem(
                    'LANDING', slice(1500, 2200),
                    airport={},
                    approach_runway={
                        'magnetic_heading': 90.0},
                    landing_runway={
                        'magnetic_heading': 90.0})])
        heading_track = P('Track', array=np.ma.concatenate((np.ones(1500) * 45, np.ones(700) * 90)))

        deviation = TrackDeviationFromRunway()
        deviation.derive(None, heading_track, None, None, apps)

        np.testing.assert_array_equal(deviation.array, np.ma.zeros(2200))

    def test_derive_north_rwy_app(self):
        hdg_con = P(name='Track True Continuous',
                    array=np.ma.array([
                        719.7430146, 719.8749494, 720.0068841, 719.741058,
                        719.5834213, 719.4516828, 719.3199441, 718.9452616,
                        718.9287504, 718.6679204, 718.665269, 718.6860868,
                        718.9742777, 719.3039603, 719.3287719, 719.6427257,
                        719.4436262, 719.4629417, 719.8338194, 720.2436481,
                        720.3361067, 720.3801377, 720.1192977, 720.1398589,
                        719.9397382, 719.6981202, 719.7613724,
                        ]))
        expected = [
            0.690980747, 0.822915588, 0.954850318, 0.689024181, 0.531387448,
            0.399648935, 0.267910243, -0.106772212, -0.123283449, -0.384113379,
            -0.386764844, -0.36594698, -0.077756105, 0.251926457, 0.276738102,
            0.590691878, 0.391592334, 0.4109079, 0.781785586, 1.191614312,
            1.284072892, 1.328103887, 1.06726383, 1.087825049, 0.887704327,
            0.646086399, 0.709338563,
        ]
        apps = App(
            items=[
            ApproachItem(
                type='LANDING',
                slice=slice(0, 27, None),
                landing_runway={
                    u'end': {
                        u'latitude': 4.205357754937445,
                        u'elevation': 6,
                        u'longitude': 73.52892655092602
                        },
                    u'start': {
                        u'latitude': 4.179942425877438,
                        u'elevation': 5,
                        u'longitude': 73.52934822453923},
                    u'magnetic_heading': 2.0,
                }),
            ])

        deviation = TrackDeviationFromRunway()
        deviation.derive(hdg_con, None, None, None, apps)

        self.assertEqual(len(deviation.array), 27)
        for first, second in zip(deviation.array, expected):
            self.assertAlmostEqual(first, second, places=2)

'''
class TestHeadingIncreasing(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous',)]
        opts = HeadingIncreasing.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_heading_increasing(self):
        head = P('Heading Continuous', array=np.ma.array([0.0,1.0,-2.0]),
                 frequency=0.5)
        head_inc=HeadingIncreasing()
        head_inc.derive(head)
        expected = np.ma.array([0.0, 0.5, 2.5])
        assert_array_equal(head_inc.array, expected)


    def test_latitude_smoothing_basic(self):
        lat = P('Latitude',np.ma.array([0,0,1,2,1,0,0],dtype=float))
        lon = P('Longitude', np.ma.array([0,0,0,0,0,0,0.001],dtype=float))
        smoother = LatitudePreparedLatLon()
        smoother.get_derived([lon, lat, aeroplane])
        # An output warning of smooth cost function closing with cost > 1 is
        # normal and arises because the data sample is short.
        expected = [0.0, 0.0, 0.00088, 0.00088, 0.00088, 0.0, 0.0]
        np.testing.assert_almost_equal(smoother.array, expected, decimal=5)

    def test_latitude_smoothing_masks_static_data(self):
        lat = P('Latitude',np.ma.array([0,0,1,2,1,0,0],dtype=float))
        lon = P('Longitude', np.ma.zeros(7,dtype=float))
        smoother = LatitudePreparedLatLon()
        smoother.get_derived([lon, lat, aeroplane])
        self.assertEqual(np.ma.count(smoother.array),0) # No non-masked values.

    #@unittest.skip('Need proper assertions')
    def test_latitude_smoothing_short_array(self):
        lat = P('Latitude',np.ma.array([0,0],dtype=float))
        lon = P('Longitude', np.ma.zeros(2,dtype=float))
        smoother = LatitudePreparedLatLon()
        smoother.get_derived([lon, lat, aeroplane])
        self.assertTrue(smoother.array.mask.all())

    def test_longitude_smoothing_basic(self):
        lat = P('Latitude',np.ma.array([0,0,1,2,1,0,0],dtype=float))
        lon = P('Longitude', np.ma.array([0,0,-2,-4,-2,0,0],dtype=float))
        smoother = LongitudePreparedLatLon()
        smoother.get_derived([lon, lat, aeroplane])
        # An output warning of smooth cost function closing with cost > 1 is
        # normal and arises because the data sample is short.
        expected = [0.0, 0.0, -0.00176, -0.00176, -0.00176, 0.0, 0.0]
        np.testing.assert_almost_equal(smoother.array, expected, decimal=5)
'''

class TestHeading(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Heading.get_operational_combinations(),
            [('Heading True Continuous', 'Magnetic Variation')])

    def test_basic(self):
        true = P('Heading True Continuous', np.ma.array([0,5,6,355,356]))
        var = P('Magnetic Variation',np.ma.array([2,3,-8,-7,9]))
        head = Heading()
        head.derive(true, var)
        expected = P('Heading True', np.ma.array([358.0, 2.0, 14.0, 2.0, 347.0]))
        assert_array_equal(head.array, expected.array)


class TestHeadingTrue(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(HeadingTrue.get_operational_combinations(),
            [('Heading Continuous', 'Magnetic Variation From Runway'),])

    def test_from_runway_used_in_preference(self):
        head = P('Heading Continuous', np.ma.array([0,5,6,355,356]))
        rwy_var = P('Magnetic Variation From Runway',np.ma.array([0,1,2,3,4]))
        true = HeadingTrue()
        true.derive(head, rwy_var)
        expected = P('Heading True', np.ma.array([0, 6, 8, 358, 0]))
        assert_array_equal(true.array, expected.array)


class TestILSFrequency(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS (1) Frequency', 'ILS (2) Frequency',),
                    ('ILS-VOR (1) Frequency', 'ILS-VOR (2) Frequency',),
                    ('ILS (1) Frequency', 'ILS (2) Frequency',
                     'ILS-VOR (1) Frequency', 'ILS-VOR (2) Frequency',)]
        opts = ILSFrequency.get_operational_combinations()
        self.assertTrue([e in opts for e in expected])

    def test_ils_vor_frequency_in_range(self):
        f1 = P('ILS-VOR (1) Frequency',
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS-VOR (2) Frequency',
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, f2])
        expected_array = np.ma.array(
            data=[1,2,108.10,108.15,111.95,112.00],
             mask=[True,True,False,False,False,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)

    def test_ils_vor_1_frequency_in_range(self):
        f1 = P('ILS-VOR (1) Frequency',
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS-VOR (2) Frequency',
               np.ma.array([1,2,3,4,5,6]),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, f2])
        expected_array = np.ma.array(
            data=[1,2,108.10,108.15,111.95,112.00],
             mask=[True,True,False,False,False,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)

    def test_ils_vor_2_frequency_in_range(self):
        f1 = P('ILS-VOR (1) Frequency',
               np.ma.array([1,2,3,4,5,6]),
               offset = 1.1, frequency = 0.5)
        f2 = P('ILS-VOR (2) Frequency',
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 0.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, f2])
        expected_array = np.ma.array(
            data=[1, 108.10, 108.15, 111.95, 112.00, 6],
             mask=[True,False,False,False,True,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)

    def test_ils_vor_1_takes_precedence(self):
        f1 = P('ILS-VOR (1) Frequency',
               np.ma.array([0,108.10,108.10,0,108.10,112.00]),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS-VOR (2) Frequency',
               np.ma.array([110.50,110.50,110.50,110.50,110.50,110.50]),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, f2])
        expected_array = np.ma.array(
            data=[110.50,108.10,108.10,110.50,108.10,110.50],
             mask=[True,False,False,False,False,False])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)



    def test_single_ils_vor_frequency_in_range(self):
        f1 = P('ILS-VOR (1) Frequency',
               np.ma.array(data=[1,2,108.10,108.15,111.95,112.00],
                           mask=[True,True,False,False,False,True]),
               offset = 0.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, None])
        expected_array = np.ma.array(
            data=[1,2,108.10,108.15,111.95,112.00],
             mask=[True,True,False,False,False,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)

    def test_ils_frequency_in_range(self):
        f1 = P('ILS (1) Frequency',
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS (2) Frequency',
               np.ma.array([1,2,108.10,108.15,111.95,112.00]),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([f1, f2, None, None])
        expected_array = np.ma.array(
            data=[1,2,108.10,108.15,111.95,112.00],
             mask=[True,True,False,False,False,True])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)

    def test_ils_frequency_matched(self):
        f1 = P('ILS-VOR (1) Frequency',
               np.ma.array([108.10]*3+[111.95]*3),
               offset = 0.1, frequency = 0.5)
        f2 = P('ILS-VOR (2) Frequency',
               np.ma.array([108.10,111.95]*3),
               offset = 1.1, frequency = 0.5)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, f2])
        expected_array = f1.array
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)

    def test_ils_frequency_different_sample_rates(self):
        f1 = P('ILS-VOR (1) Frequency',
               np.ma.concatenate((np.ones(2) * 110.5, np.zeros(2), np.ones(2) * 111.95)),
               frequency = 0.5,
               offset = 0.423828125)
        f2 = P('ILS-VOR (2) Frequency',
               np.ma.ones(3) * 108.10,
               frequency = 0.25,
               offset = 1.423828125)
        ils = ILSFrequency()
        result = ils.get_derived([None, None, f1, f2])
        expected_array = np.ma.array(
            data= [110.5, 110.5, 108.10, 108.10, 111.95, 111.95],
             mask=[  False,  False,  False,   False, False,   False])
        ma_test.assert_masked_array_approx_equal(result.array, expected_array)


class TestILSLocalizerRange(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestPitch(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Pitch (1)', 'Pitch (2)')]
        opts = Pitch.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_pitch_combination(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.arange(5, dtype=float), 1, 0.1),
                   P('Pitch (2)', np.ma.arange(10, 15, dtype=float), 1, 0.6)
                  )
        answer = np.ma.array(data=([5.0,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.0]))
        combo = P('Pitch',answer,frequency=2,offset=0.1)
        assert_array_equal(pch.array, combo.array)
        self.assertEqual(pch.frequency, combo.frequency)
        self.assertEqual(pch.offset, combo.offset)

    def test_pitch_reverse_combination(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.arange(5, dtype=float)+1, 1,0.95),
                   P('Pitch (2)', np.ma.arange(5, dtype=float)+10, 1,0.45))
        answer = np.ma.array(data=np.arange(10), mask=([1]+[0]*9))/2.0+5.0
        assert_array_equal(pch.array, answer.data)

    def test_pitch_error_different_rates(self):
        pch = Pitch()
        self.assertRaises(AssertionError, pch.derive,
                          P('Pitch (1)', np.ma.arange(5, dtype=float), 2,0.1),
                          P('Pitch (2)', np.ma.arange(10, dtype=float)+10, 4,0.6))

    def test_pitch_different_offsets(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.arange(5, dtype=float), 1,0.11),
                   P('Pitch (2)', np.ma.arange(5, dtype=float), 1,0.6))
        # This originally produced an error, but with amended merge processes
        # this is not necessary. Simply check the result is the right length.
        self.assertEqual(len(pch.array),10)


class TestPitchFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = PitchFor3Sec
        self.operational_combinations = [
            ('Pitch',),
        ]
        self.pitch = P(
            name='Pitch',
            array=np.ma.repeat((10, 11, 12, 10), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_name_and_units(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Pitch For 3 Sec')
        self.assertEqual(node.units, ut.DEGREE)

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.pitch])
        expected = np.ma.repeat((10, 11, 10), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.pitch.frequency = 1
        node = self.node_class()
        node.get_derived([self.pitch])
        expected = np.ma.repeat((10, 10.5, 11, 10), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestVerticalSpeed(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(VerticalSpeed.get_operational_combinations(),
                         [('Altitude STD Smoothed',),
                           ('Altitude STD Smoothed', 'Frame')])

    def test_vertical_speed_basic(self):
        alt_std = P('Altitude STD Smoothed', np.ma.ones(10) * 100)
        vert_spd = VerticalSpeed()
        vert_spd.derive(alt_std, None)
        expected = np.ma.zeros(10)
        ma_test.assert_masked_array_approx_equal(vert_spd.array, expected)

    def test_vertical_speed_alt_std_only(self):
        alt_std = P('Altitude STD Smoothed', np.ma.arange(100, 200, 10))
        vert_spd = VerticalSpeed()
        vert_spd.derive(alt_std, None)
        expected = np.ma.ones(10) * 600 #  10 ft/sec = 600 fpm
        ma_test.assert_masked_array_approx_equal(vert_spd.array, expected)


class TestVerticalSpeedForFlightPhases(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            VerticalSpeedForFlightPhases.get_operational_combinations(),
            [('Altitude STD Smoothed',)],
        )

    def test_vertical_speed_for_flight_phases_basic(self):
        alt_std = P('Altitude STD Smoothed', np.ma.arange(10))
        vert_spd = VerticalSpeedForFlightPhases()
        vert_spd.derive(alt_std)
        assert_array_equal(vert_spd.array, np.ma.ones(10) * 60)

    def test_vertical_speed_for_flight_phases_level_flight(self):
        alt_std = P('Altitude STD Smoothed', np.ma.ones(10) * 100)
        vert_spd = VerticalSpeedForFlightPhases()
        vert_spd.derive(alt_std)
        assert_array_equal(vert_spd.array, np.ma.zeros(10))

class TestVerticalSpeedFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = VerticalSpeedFor3Sec
        self.operational_combinations = [
            ('Vertical Speed',),
        ]
        self.vert_spd = P(
            name='Vertical Speed',
            array=np.ma.repeat((1000, 1100, 1200, 1000), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_name_and_units(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Vertical Speed For 3 Sec')
        self.assertEqual(node.units, ut.FPM)

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.vert_spd])
        expected = np.ma.repeat((1000, 1100, 1000), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.vert_spd.frequency = 1
        node = self.node_class()
        node.get_derived([self.vert_spd])
        expected = np.ma.repeat((1000, 1050, 1100, 1000), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestHeadingRate(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous',)]
        opts = HeadingRate.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_rate_of_turn(self):
        rot = HeadingRate()
        rot.derive(P('Heading Continuous', np.ma.arange(10)))
        answer = np.ma.array(data=[1]*10, dtype=np.float)
        assert_array_equal(rot.array, answer) # Tests data only; NOT mask

    def test_rate_of_turn_phase_stability(self):
        rot = HeadingRate()
        rot.derive(P('Heading Continuous', np.ma.array([0,0,2,4,2,0,0],
                                                          dtype=float)))
        answer = np.ma.array([0,1.95,0.5,0,-0.5,-1.95,0])
        ma_test.assert_masked_array_approx_equal(rot.array, answer)

    def test_sample_long_gentle_turn(self):
        # Sample taken from a long circling hold pattern
        path = os.path.join(test_data_path, 'heading_continuous_in_hold.npz')
        head_cont = P(array=np.ma.array(load_compressed(path)), frequency=2)
        rot = HeadingRate()
        rot.get_derived((head_cont,))
        np.testing.assert_allclose(rot.array[50:1150],
                                   np.ones(1100, dtype=float)*2.1, rtol=0.1)


class TestMach(unittest.TestCase):
    def test_can_operate(self):
        opts = Mach.get_operational_combinations()
        self.assertEqual(opts, [('Airspeed', 'Altitude STD Smoothed')])

    def test_all_cases(self):
        cas = P('Airspeed', np.ma.array(data=[0, 100, 200, 200, 200, 500, 200],
                                        mask=[0,0,0,0,1,0,0], dtype=float))
        alt = P('Altitude STD Smoothed', np.ma.array(data=[0, 10000, 20000, 30000, 30000, 45000, 20000],
                                        mask=[0,0,0,0,0,0,1], dtype=float))
        mach = Mach()
        mach.derive(cas, alt)
        expected = np.ma.array(data=[0, 0.182, 0.4402, 0.5407, 0.5407, 1.6825, 45000],
                                        mask=[0,0,0,0,1,1,1], dtype=float)
        ma_test.assert_masked_array_approx_equal(mach.array, expected, decimal=2)


class TestHeadwind(unittest.TestCase):
    def test_can_operate(self):
        opts = [set(opt) for opt in Headwind.get_operational_combinations()]
        wind_based = {'Wind Speed', 'Wind Direction', 'Heading True', 'Altitude AAL'}
        aspd_based = {'Airspeed True', 'Groundspeed'}
        for opt in opts:
            self.assertTrue(wind_based.issubset(opt) or aspd_based.issubset(opt))

    def test_real_example(self):
        ws = P('Wind Speed', np.ma.array([84.0,]))
        wd = P('Wind Direction', np.ma.array([350]))
        head = P('Heading True', np.ma.array([50]))
        alt_aal = P('Altitude AAL', np.ma.ones(5) * 2000)
        hw = Headwind()
        hw.derive(None, ws, wd, head, alt_aal, None)
        expected = np.ma.array([42.])
        ma_test.assert_masked_array_almost_equal(hw.array, expected)

    def test_odd_angles(self):
        ws = P('Wind Speed', np.ma.ones(5) * 20)
        # using only multiplies of 60 as a difference as that ensures we get
        # nice, round values from cos()
        wd = P('Wind Direction', np.ma.array([340, 270, 270, 350, 5], dtype=float))
        head=P('Heading True', np.ma.array([340, 90, 210, 50, 245], dtype=float))
        alt_all = P('Altitude AAL', np.ma.ones(5) * 2000)
        hw = Headwind()
        hw.derive(None, ws, wd, head, alt_all, None)
        ma_test.assert_masked_array_almost_equal (hw.array, np.ma.array([20, -20, 10, 10, -10]))

    def test_headwind_below_100ft(self):
        # create consistent 20 kt windspeed on the tail
        wspd = P('Wind Speed', np.ma.ones(40) * 20)
        wdir = P('Wind Direction', np.ma.ones(40) * 180.0)
        head = P('Heading True', np.ma.zeros(40))
        # create a 40 kt difference between Airspeed and speed over ground
        gspd = P('Groundspeed', np.ma.ones(40) * 220)
        gspd.array[4] = 200.0
        aspd = P('Airpseed True', np.ma.ones(40) * 180.0)
        # first 5 and last 5 samples are below 100ft
        alt = P('Altitude AAL', np.ma.concatenate((np.ones(10) * 50.0, np.ones(20) * 5000.0, np.ones(10) * 40.0)))
        hw = Headwind()
        hw.derive(aspd, wspd, wdir, head, alt, gspd)
        # below 100ft
        np.testing.assert_equal(hw.array[:10], [-40.0, -40.0, -36.0, -36.0, -36.0, -36.0, -36.0, -40.0, -40.0, -40.0])
        # above 100ft
        np.testing.assert_equal(hw.array[10:30], np.ones(20) * -20)
        # below 100ft
        np.testing.assert_equal(hw.array[30:], np.ones(10) * -40)

    def test_no_wind_parameters(self):
        # create a 40 kt difference between Airspeed and speed over ground
        gspd = P('Groundspeed', np.ma.ones(20) * 220)
        idx = 4
        gspd.array[idx] = 200.0
        aspd = P('Airpseed True', np.ma.ones(20) * 180.0)

        hw = Headwind()
        hw.derive(aspd, None, None, None, None, gspd)

        expected = np.ones(20) * -40
        expected[idx - 2 : idx + 3] += 20 / 5
        np.testing.assert_equal(hw.array, expected)


class TestWindAcrossLandingRunway(unittest.TestCase):
    def test_can_operate(self):
        opts = WindAcrossLandingRunway.get_operational_combinations()
        expected = [('Wind Speed', 'Wind Direction Continuous', 'FDR Landing Runway'),
                    ('Wind Speed', 'Wind Direction Magnetic Continuous', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction Continuous', 'Wind Direction Magnetic Continuous', 'FDR Landing Runway'),
                    ('Wind Speed', 'Wind Direction Continuous', 'Wind Direction Magnetic Continuous', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction Continuous', 'FDR Landing Runway', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction Magnetic Continuous', 'FDR Landing Runway', 'Heading During Landing'),
                    ('Wind Speed', 'Wind Direction Continuous', 'Wind Direction Magnetic Continuous', 'FDR Landing Runway', 'Heading During Landing')]
        self.assertEqual(opts, expected)

    def test_real_example(self):
        ws = P('Wind Speed', np.ma.array([84.0]))
        wd = P('Wind Direction Continuous', np.ma.array([-21]))
        land_rwy = A('FDR Landing Runway')
        land_rwy.value = {'start': {'latitude': 60.18499999999998,
                                    'longitude': 11.073744},
                          'end': {'latitude': 60.216066999999995,
                                  'longitude': 11.091663999999993}}

        walr = WindAcrossLandingRunway()
        walr.derive(ws,wd,None,land_rwy,None)
        expected = np.ma.array([50.55619778])
        np.testing.assert_array_almost_equal(walr.array.data, expected.data, 1)

    def test_error_cases(self):
        ws = P('Wind Speed', np.ma.array([84.0]))
        wd = P('Wind Direction Continuous', np.ma.array([-21]))
        land_rwy = A('FDR Landing Runway')
        land_rwy.value = {}
        walr = WindAcrossLandingRunway()

        walr.derive(ws,wd,None,land_rwy,None)
        self.assertEqual(len(walr.array.data), len(ws.array.data))
        self.assertEqual(walr.array.data[0],0.0)
        self.assertEqual(walr.array.mask[0],1)

        walr.derive(ws,wd,None)
        self.assertEqual(len(walr.array.data), len(ws.array.data))
        self.assertEqual(walr.array.data[0],0.0)
        self.assertEqual(walr.array.mask[0],1)


class TestAOA(unittest.TestCase):
    def test_can_operate(self):
        opts = AOA.get_operational_combinations()
        self.assertEqual(opts, [
            ('AOA (L)',),
            ('AOA (R)',),
            ('AOA (L)', 'AOA (R)'),
            ('AOA (L)', 'Model'),
            ('AOA (R)', 'Model'),
            ('AOA (L)', 'AOA (R)', 'Model'),
        ])

    def test_derive(self):
        aoa_l = P('AOA (L)', [4.921875, 4.5703125, 4.5703125, 4.5703125,
                              4.570315, 4.5703125, 4.5703125, 4.9213875],
                  frequency=1.0, offset=0.1484375)

        aoa_r = P('AOA (R)', [4.881875, 4.5703125, 4.5712125, 4.544125],
                          frequency=0.5, offset=0.6484375)
        aoa = AOA()
        res = aoa.get_derived((aoa_l, aoa_r, None))
        self.assertEqual(aoa.hz, 1)
        self.assertEqual(aoa.offset, 0.1484375)

    def test_Derive_only_left(self):
        aoa_l = P('AOA (L)', [4.921875, 4.5703125, 4.5703125, 4.5703125,
                              4.570315, 4.5703125, 4.5703125, 4.9213875],
                  frequency=1.0, offset=0.1484375)

        aoa = AOA()
        res = aoa.get_derived((aoa_l, None, None))
        self.assertEqual(aoa.hz, 1)
        self.assertEqual(aoa.offset, 0.1484375)

    def test_cl_600_2B19_model(self):
        aoa_r = P('AOA (R)', [0.0, 1.0])
        model = A('Model', 'CRJ-200ER (CL-600-2B19)')
        aoa = AOA()
        aoa.get_derived([None, aoa_r, model])
        self.assertAlmostEqual(aoa.array[0], -1.404)
        self.assertAlmostEqual(aoa.array[1], 0.257)


class TestAccelerationLateralOffsetRemoved(unittest.TestCase):

    def setUp(self):
        self.node_class = AccelerationLateralOffsetRemoved
        self.acc_lateral=P('Acceleration Lateral', array=np.ma.concatenate((np.zeros(20), np.ones(20) * 0.5, np.zeros(10))), frequency=1.0)

    def test_can_operate(self):
        opts = AccelerationLateralOffsetRemoved.get_operational_combinations()
        self.assertEqual(opts, [('Acceleration Lateral',), ('Acceleration Lateral', 'Acceleration Lateral Offset')])

    def test_derive_positive_offset(self):
        self.acc_lat_offset = KPV([KeyPointValue(index=0, value=0.2, name='Acceleration Lateral Offset')])
        acc = AccelerationLateralOffsetRemoved()
        acc.derive(self.acc_lateral, self.acc_lat_offset)
        assert_array_equal(acc.array, self.acc_lateral.array - self.acc_lat_offset[0].value)

    def test_derive_negative_offset(self):
        self.acc_lat_offset = KPV([KeyPointValue(index=0, value=-0.2, name='Acceleration Lateral Offset')])
        acc = AccelerationLateralOffsetRemoved()
        acc.derive(self.acc_lateral, self.acc_lat_offset)
        assert_array_equal(acc.array, self.acc_lateral.array - self.acc_lat_offset[0].value)

    def test_derive_no_offset(self):
        self.acc_lat_offset = KPV([KeyPointValue(index=0, value=0.0, name='Acceleration Lateral Offset')])
        acc = AccelerationLateralOffsetRemoved()
        acc.derive(self.acc_lateral, self.acc_lat_offset)
        assert_array_equal(acc.array, self.acc_lateral.array - self.acc_lat_offset[0].value)


class TestAccelerationLongitudinalOffsetRemoved(unittest.TestCase):

    def setUp(self):
        self.node_class = AccelerationLongitudinalOffsetRemoved
        self.acc_long=P('Acceleration Longitudinal', array=np.ma.concatenate((np.zeros(20), np.ones(20) * 0.5, np.zeros(10))))

    def test_can_operate(self):
        opts = AccelerationLongitudinalOffsetRemoved.get_operational_combinations()
        self.assertEqual(opts, [('Acceleration Longitudinal', 'Acceleration Longitudinal Offset')])

    def test_derive_positive_offset(self):
        self.acc_long_offset = KPV([KeyPointValue(index=0, value=0.2, name='Acceleration Longitudinal Offset')])

        acc = AccelerationLongitudinalOffsetRemoved()
        acc.derive(self.acc_long, self.acc_long_offset)
        assert_array_equal(acc.array, self.acc_long.array - self.acc_long_offset[0].value)

    def test_derive_negative_offset(self):
        self.acc_long_offset = KPV([KeyPointValue(index=0, value=-0.2, name='Acceleration Longitudinal Offset')])

        acc = AccelerationLongitudinalOffsetRemoved()
        acc.derive(self.acc_long, self.acc_long_offset)
        assert_array_equal(acc.array, self.acc_long.array - self.acc_long_offset[0].value)

    def test_derive_no_offset(self):
        self.acc_long_offset = KPV([KeyPointValue(index=0, value=0.0, name='Acceleration Longitudinal Offset')])

        acc = AccelerationLongitudinalOffsetRemoved()
        acc.derive(self.acc_long, self.acc_long_offset)
        assert_array_equal(acc.array, self.acc_long.array - self.acc_long_offset[0].value)


class TestAccelerationNormalOffsetRemoved(unittest.TestCase):

    def setUp(self):
        self.node_class = AccelerationNormalOffsetRemoved
        self.acc_normal=P('Acceleration Normal', array=np.ma.concatenate((np.ones(20), np.ones(20) * 1.5, np.ones(10))), frequency=1.0)

    def test_can_operate(self):
        opts = AccelerationNormalOffsetRemoved.get_operational_combinations()
        self.assertEqual(opts, [('Acceleration Normal',), ('Acceleration Normal', 'Acceleration Normal Offset')])

    def test_derive_positive_offset(self):
        self.acc_norm_offset = KPV([KeyPointValue(index=0, value=1.2, name='Acceleration Normal Offset')])

        acc = AccelerationNormalOffsetRemoved()
        acc.derive(self.acc_normal, self.acc_norm_offset)
        assert_array_equal(acc.array, self.acc_normal.array - self.acc_norm_offset[0].value + 1.0)
        # could as well be acceleration_normal.array - 0.2

    def test_derive_negative_offset(self):
        self.acc_norm_offset = KPV([KeyPointValue(index=0, value=-1.2, name='Acceleration Normal Offset')])

        acc = AccelerationNormalOffsetRemoved()
        acc.derive(self.acc_normal, self.acc_norm_offset)
        assert_array_equal(acc.array, self.acc_normal.array - self.acc_norm_offset[0].value + 1.0)

    def test_derive_no_offset(self):
        self.acc_norm_offset = KPV([KeyPointValue(index=0, value=0.0, name='Acceleration Normal Offset')])

        acc = AccelerationNormalOffsetRemoved()
        acc.derive(self.acc_normal, self.acc_norm_offset)
        assert_array_equal(acc.array, self.acc_normal.array - self.acc_norm_offset[0].value + 1.0)


class TestAileron(unittest.TestCase):

    def test_can_operate(self):
        opts = Aileron.get_operational_combinations()
        self.assertTrue(opts,
                        [('Aileron (L)',),
                         ('Aileron (R)',),
                         ('Aileron (L)', 'Aileron (R)'),
                        ])

    def test_normal_two_sensors(self):
        left = P('Aileron (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        right = P('Aileron (R)', np.ma.array([2.0]*2+[1.0]*2), frequency=0.5, offset=1.1)
        aileron = Aileron()
        aileron.get_derived([left, right])
        expected_data = np.ma.array([0, 1.5, 1.55, 1.75, 1.7, 1.5, 0, 0])
        expected_data[0] = np.ma.masked
        expected_data[-2:] = np.ma.masked
        assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 1)
        self.assertAlmostEqual(aileron.offset, 0.3, places=1)

    def test_left_only(self):
        left = P('Aileron (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        aileron = Aileron()
        aileron.get_derived([left, None])
        expected_data = left.array
        assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 0.5)
        self.assertEqual(aileron.offset, 0.1)

    def test_right_only(self):
        right = P('Aileron (R)', np.ma.array([3.0]*2+[2.0]*2), frequency=2.0, offset = 0.3)
        aileron = Aileron()
        aileron.get_derived([None, right])
        expected_data = right.array
        assert_array_equal(aileron.array, expected_data)
        self.assertEqual(aileron.frequency, 2.0)
        self.assertEqual(aileron.offset, 0.3)

    def test_aileron_with_flaperon(self):
        al = load(os.path.join(test_data_path, 'aileron_left.nod'))
        ar = load(os.path.join(test_data_path, 'aileron_right.nod'))
        ail = Aileron()
        ail.derive(al, ar)
        # this section is averaging 4.833 degrees on the way in
        self.assertAlmostEqual(np.ma.average(ail.array[160:600]), -0.04, 1)
        # this section is averaging 9.106 degrees, ensure it gets moved to 0
        #self.assertAlmostEqual(np.ma.average(ail.array[800:1000]), 0.2, 1)
        assert_array_within_tolerance(ail.array[800:1000], 0, 4, 80)


class TestAileronLeft(unittest.TestCase):

    def test_can_operate(self):
        opts = AileronLeft.get_operational_combinations()
        self.assertTrue(('Aileron (L) Synchro',) in opts)
        self.assertTrue(('Aileron (L) Potentiometer',) in opts)
        self.assertTrue(('Aileron (L) Inboard',) in opts)
        self.assertTrue(('Aileron (L) Outboard',) in opts)

    def test_synchro_working(self):
        # Includes offset potentiometer test case
        synchro = P('Aileron (L) Synchro', np.ma.array([0.0, 1.0, -1.0, 0.0]))
        pot = P('Aileron (L) Potentiometer', np.ma.array([30.0, 30.01, 29.99, 30.0]))
        aileronleft = AileronLeft()
        aileronleft.get_derived([pot, synchro, None, None])
        assert_array_equal(aileronleft.array, synchro.array)

    def test_pot_working(self):
        synchro = P('Aileron (L) Synchro', np.ma.array([0.0, 0.01, -0.01, 0.0]))
        pot = P('Aileron (L) Potentiometer', np.ma.array([0.0, 1.0, -1.0, 0.0]))
        aileronleft = AileronLeft()
        aileronleft.get_derived([pot, synchro, None, None])
        assert_array_equal(aileronleft.array, pot.array)

    def test_inboard_only(self):
        inboard = P('Aileron (L) Inboard', np.ma.array([0.0, 0.01, -0.01, 0.0]))
        aileronleft = AileronLeft()
        aileronleft.get_derived([None, None, inboard, None])
        assert_array_equal(aileronleft.array, inboard.array)

    def test_outboard_only(self):
        outboard = P('Aileron (L) Outboard', np.ma.array([0.0, 0.01, -0.01, 0.0]))
        aileronleft = AileronLeft()
        aileronleft.get_derived([None, None, None, outboard])
        assert_array_equal(aileronleft.array, outboard.array)

    def test_inboard_preferred(self):
        inboard = P('Aileron (L) Inboard', np.ma.array([0.0, 1.0, -1.0, 0.0]))
        outboard = P('Aileron (L) Outboard', np.ma.array([0.0, 2.0, -2.0, 0.0]))
        aileronleft = AileronLeft()
        aileronleft.get_derived([None, None, inboard, outboard])
        assert_array_equal(aileronleft.array, inboard.array)


class TestAileronRight(unittest.TestCase):

    def test_can_operate(self):
        opts = AileronRight.get_operational_combinations()
        self.assertTrue(('Aileron (R) Synchro',) in opts)
        self.assertTrue(('Aileron (R) Potentiometer',) in opts)
        self.assertTrue(('Aileron (R) Inboard',) in opts)
        self.assertTrue(('Aileron (R) Outboard',) in opts)

    def test_synchro_working(self):
        # Includes potentiometer close to zero test case
        synchro = P('Aileron (R) Synchro', np.ma.array([0.0, 1.0, -1.0, 0.0]))
        pot = P('Aileron (R) Potentiometer', np.ma.array([0.0, 0.01, -0.01, 0.0]))
        aileronright = AileronRight()
        aileronright.get_derived([pot, synchro, None, None])
        assert_array_equal(aileronright.array, synchro.array)

    def test_pot_working(self):
        synchro = P('Aileron (R) Synchro', np.ma.array([0.0, 0.01, -0.01, 0.0]))
        pot = P('Aileron (R) Potentiometer', np.ma.array([0.0, 1.0, -1.0, 0.0]))
        aileronright = AileronRight()
        aileronright.get_derived([pot, synchro, None, None])
        assert_array_equal(aileronright.array, pot.array)

    def test_inboard_only(self):
        inboard = P('Aileron (R) Inboard', np.ma.array([0.0, 0.01, -0.01, 0.0]))
        aileronright = AileronRight()
        aileronright.get_derived([None, None, inboard, None])
        assert_array_equal(aileronright.array, inboard.array)

    def test_outboard_only(self):
        outboard = P('Aileron (R) Outboard', np.ma.array([0.0, 0.01, -0.01, 0.0]))
        aileronright = AileronRight()
        aileronright.get_derived([None, None, None, outboard])
        assert_array_equal(aileronright.array, outboard.array)

    def test_inboard_preferred(self):
        inboard = P('Aileron (R) Inboard', np.ma.array([0.0, 1.0, -1.0, 0.0]))
        outboard = P('Aileron (R) Outboard', np.ma.array([0.0, 2.0, -2.0, 0.0]))
        aileronright = AileronLeft()
        aileronright.get_derived([None, None, inboard, outboard])
        assert_array_equal(aileronright.array, inboard.array)


class TestAileronTrim(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeSTD(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestElevator(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Elevator.get_operational_combinations(),
                         [('Elevator (L)',),
                          ('Elevator (R)',),
                          ('Elevator (L)', 'Elevator (R)'),
                          ])

    def test_normal_two_sensors(self):
        left = P('Elevator (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        right = P('Elevator (R)', np.ma.array([2.0]*2+[1.0]*2), frequency=0.5, offset=1.1)
        elevator = Elevator()
        elevator.derive(left, right)
        expected_data = np.ma.array([1.5]*3+[1.75]*2+[1.5]*3)
        assert_array_equal(elevator.array, expected_data)
        self.assertEqual(elevator.frequency, 1.0)
        self.assertEqual(elevator.offset, 0.1)

    def test_left_only(self):
        left = P('Elevator (L)', np.ma.array([1.0]*2+[2.0]*2), frequency=0.5, offset=0.1)
        elevator = Elevator()
        elevator.derive(left, None)
        expected_data = left.array
        assert_array_equal(elevator.array, expected_data)
        self.assertEqual(elevator.frequency, 0.5)
        self.assertEqual(elevator.offset, 0.1)

    def test_right_only(self):
        right = P('Elevator (R)', np.ma.array([3.0]*2+[2.0]*2), frequency=2.0, offset=0.3)
        elevator = Elevator()
        elevator.derive(None, right)
        expected_data = right.array
        assert_array_equal(elevator.array, expected_data)
        self.assertEqual(elevator.frequency, 2.0)
        self.assertEqual(elevator.offset, 0.3)

class TestElevatorLeft(unittest.TestCase):
    def test_can_operate(self):
        opts = ElevatorLeft.get_operational_combinations()
        self.assertTrue(('Elevator (L) Synchro',) in opts)
        self.assertTrue(('Elevator (L) Potentiometer',) in opts)
        self.assertTrue(('Elevator (L) Inboard',) in opts)
        self.assertTrue(('Elevator (L) Outboard',) in opts)

    def test_synchro(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorLeft()
        elevator.derive(None, syn, None, None)
        assert_array_equal(elevator.array, syn.array)

    def test_pot(self):
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,0,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, None, None, None)
        assert_array_equal(elevator.array, pot.array)

    def test_both_prefer_syn(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,1,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, syn, None, None)
        assert_array_equal(elevator.array, syn.array)

    def test_both_prefer_pot(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[1,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, syn, None, None)
        assert_array_equal(elevator.array, pot.array)

    def test_both_equally_good(self):
        # Where there is no advantage, adopt the synchro which should be a better transducer.
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,0,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,0,0]))
        elevator=ElevatorLeft()
        elevator.derive(pot, syn, None, None)
        assert_array_equal(elevator.array, syn.array)

    def test_inboard_and_outboard(self):
        l_in=P('Elevator (L) Inboard', np.ma.array([1,2,3]))
        l_out=P('Elevator (L) Outboard', np.ma.array([5,4,3]))
        elevator=ElevatorLeft()
        elevator.get_derived([None, None, l_in, l_out])
        assert_array_equal(elevator.array, l_in.array)

    def test_outboard_only(self):
        l_out=P('Elevator (L) Outboard', np.ma.array([5,4,3]))
        elevator=ElevatorLeft()
        elevator.get_derived([None, None, None, l_out])
        assert_array_equal(elevator.array, l_out.array)

class TestElevatorRight(unittest.TestCase):
    def test_can_operate(self):
        opts = ElevatorRight.get_operational_combinations()
        self.assertTrue(('Elevator (R) Synchro',) in opts)
        self.assertTrue(('Elevator (R) Potentiometer',) in opts)
        self.assertTrue(('Elevator (R) Inboard',) in opts)
        self.assertTrue(('Elevator (R) Outboard',) in opts)

    def test_synchro(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorRight()
        elevator.derive(None, syn, None, None)
        assert_array_equal(elevator.array, syn.array)

    def test_pot(self):
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,0,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, None, None, None)
        assert_array_equal(elevator.array, pot.array)

    def test_both_prefer_syn(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,1,1,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, syn, None, None)
        assert_array_equal(elevator.array, syn.array)

    def test_both_prefer_pot(self):
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[1,0,1,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,1,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, syn, None, None)
        assert_array_equal(elevator.array, pot.array)

    def test_both_equally_good(self):
        # Where there is no advantage, adopt the synchro which should be a better transducer.
        syn=P('Elevator (L) Synchro', np.ma.array(data=[1,2,3,4],
                                                  mask=[0,0,0,0]))
        pot=P('Elevator (L) Potentiometer', np.ma.array(data=[5,6,7,8],
                                                  mask=[0,0,0,0]))
        elevator=ElevatorRight()
        elevator.derive(pot, syn, None, None)
        assert_array_equal(elevator.array, syn.array)

    def test_inboard_and_outboard(self):
        r_in=P('Elevator (R) Inboard', np.ma.array([1,2,3]))
        r_out=P('Elevator (R) Outboard', np.ma.array([5,4,3]))
        elevator=ElevatorRight()
        elevator.get_derived([None, None, r_in, r_out])
        assert_array_equal(elevator.array, r_in.array)

    def test_outboard_only(self):
        r_out=P('Elevator (R) Outboard', np.ma.array([5,4,3]))
        elevator=ElevatorRight()
        elevator.get_derived([None, None, None, r_out])
        assert_array_equal(elevator.array, r_out.array)


class TestEng_FuelFlow(unittest.TestCase):
    def test_can_operate(self):
        for n in range(1, 5):
            self.assertTrue(Eng_FuelFlow.can_operate(
                ('Eng (%d) Fuel Flow' % n,)))

    def test_derive(self):
        ff1_arr = np.ma.arange(100)
        ff2_arr = np.ma.arange(100)[::-1]
        ff1 = P(name='Eng (1) Fuel Flow', array=ff1_arr, frequency=1)
        ff2 = P(name='Eng (2) Fuel Flow', array=ff2_arr, frequency=1)

        node = Eng_FuelFlow()
        node.derive(ff1, ff2, None, None)

        assert_array_equal(node.array, np.ma.ones(100) * 99)


class TestEng_FuelFlowMax(unittest.TestCase):
    def test_can_operate(self):
        for n in range(1, 5):
            self.assertTrue(Eng_FuelFlowMax.can_operate(
                ('Eng (%d) Fuel Flow' % n,)))

    def test_derive(self):
        ff1_arr = np.ma.arange(100)
        ff2_arr = np.ma.arange(100)[::-1]
        ff1 = P(name='Eng (1) Fuel Flow', array=ff1_arr, frequency=1)
        ff2 = P(name='Eng (2) Fuel Flow', array=ff2_arr, frequency=1)

        expected_arr = np.ma.concatenate([np.arange(99, 49, -1),
                                          np.arange(50, 100)])

        node = Eng_FuelFlowMax()
        node.derive(ff1, ff2, None, None)

        assert_array_equal(node.array, expected_arr)


class TestEng_FuelFlowMin(unittest.TestCase):
    def test_can_operate(self):
        for n in range(1, 5):
            self.assertTrue(Eng_FuelFlowMin.can_operate(
                ('Eng (%d) Fuel Flow' % n,)))

    def test_derive(self):
        ff1_arr = np.ma.arange(100)
        ff2_arr = np.ma.arange(100)[::-1]
        ff1 = P(name='Eng (1) Fuel Flow', array=ff1_arr, frequency=1)
        ff2 = P(name='Eng (2) Fuel Flow', array=ff2_arr, frequency=1)

        expected_arr = np.ma.concatenate([np.arange(50),
                                          np.arange(49, -1, -1)])

        node = Eng_FuelFlowMin()
        node.derive(ff1, ff2, None, None)

        assert_array_equal(node.array, expected_arr)


class TestEng_1_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_1_FuelBurn
        self.operational_combinations = [('Eng (1) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_2_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_2_FuelBurn
        self.operational_combinations = [('Eng (2) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_3_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_3_FuelBurn
        self.operational_combinations = [('Eng (3) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_4_FuelBurn(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_4_FuelBurn
        self.operational_combinations = [('Eng (4) Fuel Flow', )]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_FuelBurn(unittest.TestCase):

    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_GasTempAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_GasTempMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_GasTempMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilPressAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilPressMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilPressMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilQtyAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilQtyMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilQtyMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilTempAvg(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilTempMax(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_OilTempMin(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_TorqueAvg(unittest.TestCase):

    def setUp(self):
        self.node_class = Eng_TorqueAvg

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertTrue(('Eng (1) Torque',) in poss_combs)
        self.assertTrue(('Eng (2) Torque',) in poss_combs)
        self.assertTrue(('Eng (3) Torque',) in poss_combs)
        self.assertTrue(('Eng (4) Torque',) in poss_combs)

    def test_derive(self):
        eng_1_array =    [0, 30, 50, 80, 100,   100, 70, 70, 70, 50, 50, 10,  0,  0, 0]
        eng_3_array =    [0,  0, 30, 60,  85,   100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        expected_array = [0, 15, 40, 70,  92.5, 100, 70, 70, 70, 50, 50, 20,  5,  5, 0]

        eng_1 = P(name='Eng (1) Torque', array=eng_1_array, frequency=1,
                 offset=0.1)

        eng_3 = P(name='Eng (3) Torque', array=eng_3_array, frequency=1,
                         offset=0.5)

        node = self.node_class()
        node.derive(eng_1, None, eng_3, None)

        assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.3)


class TestEng_TorqueMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Eng_TorqueMax

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertTrue(('Eng (1) Torque',) in poss_combs)
        self.assertTrue(('Eng (2) Torque',) in poss_combs)
        self.assertTrue(('Eng (3) Torque',) in poss_combs)
        self.assertTrue(('Eng (4) Torque',) in poss_combs)


    def test_derive(self):
        eng_1_array =    [0, 30, 50, 80, 100, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]
        eng_3_array =    [0,  0, 30, 60,  85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        expected_array = [0, 30, 50, 80, 100, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]

        eng_1 = P(name='Eng (1) Torque', array=eng_1_array, frequency=1,
                 offset=0.1)

        eng_3 = P(name='Eng (3) Torque', array=eng_3_array, frequency=1,
                offset=0.5)

        node = self.node_class()
        node.derive(eng_1, None, eng_3, None)

        assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.3)


class TestEng_TorqueMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Eng_TorqueMin

    def test_can_operate(self):
        poss_combs = self.node_class.get_operational_combinations()
        self.assertTrue(('Eng (1) Torque',) in poss_combs)
        self.assertTrue(('Eng (2) Torque',) in poss_combs)
        self.assertTrue(('Eng (3) Torque',) in poss_combs)
        self.assertTrue(('Eng (4) Torque',) in poss_combs)


    def test_derive(self):
        eng_1_array =    [0, 30, 50, 80, 100, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]
        eng_3_array =    [0,  0, 30, 60,  85, 100, 70, 70, 70, 50, 50, 30, 10, 10, 0]
        expected_array = [0,  0, 30, 60,  85, 100, 70, 70, 70, 50, 50, 10,  0,  0, 0]

        eng_1 = P(name='Eng (1) Torque', array=eng_1_array, frequency=1,
                 offset=0.1)

        eng_3 = P(name='Eng (3) Torque', array=eng_3_array, frequency=1,
                         offset=0.5)

        node = self.node_class()
        node.derive(eng_1, None, eng_3, None)

        assert_array_equal(node.array, expected_array)
        self.assertEqual(node.offset, 0.3)


class TestEng_VibBroadbandMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibBroadbandMax
        self.operational_combinations = [
            ('Eng (1) Vib Broadband',),
            ('Eng (1) Vib Broadband Accel A',),
            ('Eng (1) Vib Broadband Accel B',),
            ('Eng (1) Vib Broadband', 'Eng (2) Vib Broadband', 'Eng (3) Vib Broadband', 'Eng (4) Vib Broadband'),
            ('Eng (1) Vib Broadband Accel A', 'Eng (2) Vib Broadband Accel A', 'Eng (3) Vib Broadband Accel A', 'Eng (4) Vib Broadband Accel A'),
            ('Eng (1) Vib Broadband Accel B', 'Eng (2) Vib Broadband Accel B', 'Eng (3) Vib Broadband Accel B', 'Eng (4) Vib Broadband Accel B',),
        ]

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibN1Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibN1Max
        self.operational_combination_length = 65535 # 2**16-1
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibN2Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibN2Max
        self.operational_combination_length = 1023 # 2**10-1
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibN3Max(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibN3Max
        self.operational_combination_length = 255 # 2**8-1
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibAMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibAMax
        self.operational_combination_length = 15
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibBMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibBMax
        self.operational_combination_length = 15
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibCMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibCMax
        self.operational_combination_length = 15
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEng_VibNpMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Eng_VibNpMax
        self.operational_combination_length = 1023 # 2**10-1
        self.check_operational_combination_length_only = True

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngTPRLimitDifference(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(EngTPRLimitDifference.get_operational_combinations(),
                         [('Eng (*) TPR Max', 'Eng TPR Limit Max')])

    def test_derive_basic(self):
        eng_tpr_max_array = np.ma.concatenate([
            np.arange(0, 150, 10), np.arange(150, 0, -10)])
        eng_tpr_limit_array = np.ma.concatenate([
            np.arange(10, 110, 10), np.ones(10) * 110, np.arange(110, 10, -10)])
        eng_tpr_max = P('Eng (*) TPR Max', array=eng_tpr_max_array)
        eng_tpr_limit = P('Eng (*) TPR Limit Max', array=eng_tpr_limit_array)
        node = EngTPRLimitDifference()
        node.derive(eng_tpr_max, eng_tpr_limit)
        self.assertEqual(
            node.array.tolist(),
            [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 0, 10, 20,
             30, 40, 30, 20, 10, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10,
             -10])


class TestFlapAngle(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapAngle
        self.operational_combinations = [
            ('Flap Angle (L)',),
            ('Flap Angle (R)',),
            ('Flap Angle (L) Inboard',),
            ('Flap Angle (R) Inboard',),
            ('Flap Angle (L)', 'Flap Angle (R)'),
            ('Flap Angle (L) Inboard', 'Flap Angle (R) Inboard'),
            ('Flap Angle (L)', 'Flap Angle (R)', 'Flap Angle (C)', 'Flap Angle (MCP)'),
            ('Flap Angle (L)', 'Flap Angle (R)', 'Flap Angle (L) Inboard', 'Flap Angle (R) Inboard'),
        ]

    def test_hercules(self):
        parameter = P(array=np.ma.concatenate((np.arange(0, 5000, 100), np.arange(5000, 0, -200))))
        ##frame = A('Frame', 'L382-Hercules')
        node = self.node_class()
        node.derive(parameter, None, None, None, None, None)
        self.assertAlmostEqual(node.array[25], 2500, 1)


    def test_single_parameter(self):
        fr = P('Flap Angle (R)', array=np.ma.arange(10), offset=0.123, frequency=2)
        fa = FlapAngle()
        fa.get_derived([None, fr, None, None, None, None])
        self.assertEqual(fa.offset, fr.offset)
        self.assertEqual(fa.frequency, fr.frequency)
        assert_array_equal(fa.array, fr.array)

    def test_with_same_offsets(self):
        fl = P('Flap Angle (L)', array=np.ma.arange(10, 20, 2), offset=0.123, frequency=0.5)
        fr = P('Flap Angle (R)', array=np.ma.arange(11, 21, 2), offset=1.123, frequency=0.5)
        fa = FlapAngle()
        fa.apply_median_filter = False
        fa.get_derived([fl, fr, None, None, None, None])
        self.assertEqual(fa.offset, 0.123)
        self.assertEqual(fa.frequency, 1)
        np.testing.assert_array_almost_equal(fa.array, np.ma.arange(10, 20, 1), decimal=0)

    def test_with_different_offsets(self):
        fl = P('Flap Angle (L)', array=np.ma.arange(10, 20, 1), offset=0.0001, frequency=1)
        fr = P('Flap Angle (R)', array=np.ma.arange(10, 20, 1), offset=0, frequency=1)
        # 3rd flap is ignored
        fc = P('Flap Angle (C)', array=np.ma.arange(11, 21, 2), offset=0.123, frequency=1)
        fa = FlapAngle()
        fa.apply_median_filter = False
        fa.get_derived([fl, fr, fc, None, None, None])
        self.assertAlmostEqual(fa.offset, 5e-05)
        self.assertEqual(fa.frequency, 2)
        # ignore last value as we have nothing to interpolate too
        assert_array_within_tolerance(fa.array[:-1], np.ma.arange(10, 19.5, 0.5), 0.001, 99.9)
        self.assertEqual(fa.array[-2], 19)
        self.assertEqual(fa.array.mask[-1], True)

    def test_late_offset(self):
        '''
        Minimum Flap Angle offset is in the second half of the subframe at 1Hz.
        Upscaling the frequency to 2Hz without also changing the offset will result in an invalid parameter.
        '''
        fl = P('Flap Angle (L)', array=np.ma.arange(10, 20, 1), offset=0.693359375, frequency=1)
        fr = P('Flap Angle (R)', array=np.ma.arange(10, 20, 1), offset=0.697265625, frequency=1)
        fa = FlapAngle()
        fa.get_derived([fl, fr, None, None, None, None])
        self.assertEqual(fa.frequency, 2)
        offset = (fl.offset + fr.offset)/2.0 -0.5
        self.assertEqual(fa.offset, offset)


class TestFlapSynchroAsymmetry(unittest.TestCase):
    # very simple test but it ensures we aren't getting negative
    # values as that would affect the FlapSynchroAsymmetryMax KPV
    def test_basic(self):
        synchro_l = P('Flap Angle (L) Synchro', [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5])
        synchro_r = P('Flap Angle (R) Synchro', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        asym = FlapSynchroAsymmetry()
        asym.get_derived((synchro_l, synchro_r))
        self.assertEqual(np.max(asym.array), 3)

    def test_second_window(self):
        synchro_l = P('Flap Angle (L) Synchro', [12, 11, 10, 9,  8,  7, 6, 5, 4, 3, 2, 1])
        synchro_r = P('Flap Angle (R) Synchro', [12, 12, 12, 12, 12, 12, 6, 5, 4, 3, 2, 1])
        asym = FlapSynchroAsymmetry()
        asym.get_derived((synchro_l, synchro_r))
        self.assertEqual(np.max(asym.array), 1)


class TestHeadingTrueContinuous(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')

    def test_fully_masked(self):
        hdg = P('Heading True', array=np.ma.masked_all((40,)))
        node = HeadingTrueContinuous()
        node.derive(hdg)

        np.testing.assert_array_equal(node.array.mask, np.ones(40))


class TestILSGlideslope(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestILSLocalizer(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLatitudePrepared(unittest.TestCase):
    def test_can_operate(self):
        combinations = LatitudePrepared.get_operational_combinations()
        expected_combinations = [
            ('Heading', 'Airspeed True', 'Latitude At Liftoff', 'Longitude At Liftoff',
             'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading True', 'Airspeed True', 'Latitude At Liftoff', 'Longitude At Liftoff',
             'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Heading True', 'Airspeed True', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Airspeed True', 'Groundspeed', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Airspeed True', 'Altitude AAL', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading True', 'Airspeed True', 'Groundspeed', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading True', 'Airspeed True', 'Altitude AAL', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Heading True', 'Airspeed True', 'Groundspeed', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Heading True', 'Airspeed True', 'Altitude AAL', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Airspeed True', 'Groundspeed', 'Altitude AAL', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading True', 'Airspeed True', 'Groundspeed', 'Altitude AAL', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Heading True', 'Airspeed True', 'Groundspeed', 'Altitude AAL',
             'Latitude At Liftoff', 'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown')
        ]
        self.assertEqual(combinations, expected_combinations)

    @patch("analysis_engine.derived_parameters.air_track", return_value=(None, None))
    def test_derive(self, air_track):
        hdg_mag = P("Heading")
        gspd = P("Groundspeed")
        tas = P("Airspeed True", frequency=2)
        alt_aal = P("Altitude AAL")
        lat_lift = KPV(
            "Latitude At Liftoff",
            items=[KeyPointValue(name="Latitude At Liftoff", index=10, value=20)],
        )
        lon_lift = KPV(
            "Longitude At Liftoff",
            items=[KeyPointValue(name="Longitude At Liftoff", index=10, value=120)],
        )
        lat_land = KPV(
            "Latitude At Touchdown",
            items=[KeyPointValue(name="Latitude At Touchdown", index=200, value=50)],
        )
        lon_land = KPV(
            "Longitude At Touchdown",
            items=[KeyPointValue(name="Longitude At Touchdown", index=200, value=70)],
        )
        node = LongitudePrepared()
        node.derive(
            hdg_mag, None, tas, gspd, alt_aal, lat_lift, lon_lift, lat_land, lon_land
        )
        air_track.assert_called_with(
            20, 120, 50, 70, gspd.array, hdg_mag.array, alt_aal.array, 2
        )

    @patch("analysis_engine.derived_parameters.air_track", return_value=(None, None))
    def test_derive_empty_liftoff_coords(self, air_track):
        hdg_mag = P("Heading")
        gspd = P("Groundspeed")
        tas = P("Airspeed True", frequency=2)
        alt_aal = P("Altitude AAL")
        lat_lift = KPV("Latitude At Liftoff")
        lon_lift = KPV("Longitude At Liftoff")
        lat_land = KPV(
            "Latitude At Touchdown",
            items=[KeyPointValue(name="Latitude At Touchdown", index=200, value=50)],
        )
        lon_land = KPV(
            "Longitude At Touchdown",
            items=[KeyPointValue(name="Longitude At Touchdown", index=200, value=70)],
        )
        node = LongitudePrepared()
        node.derive(
            hdg_mag, None, tas, gspd, alt_aal, lat_lift, lon_lift, lat_land, lon_land
        )
        air_track.assert_not_called()
        self.assertFalse(node.array > 0)


class TestLatitudeSmoothed(unittest.TestCase):
    def test_can_operate(self):
        combinations = LatitudeSmoothed.get_operational_combinations()
        self.assertTrue(all('Latitude Prepared' in c for c in combinations))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestLongitudePrepared(unittest.TestCase):
    def test_can_operate(self):
        combinations = LongitudePrepared.get_operational_combinations()
        expected_combinations = [
            ('Heading', 'Airspeed True', 'Latitude At Liftoff', 'Longitude At Liftoff',
             'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading True', 'Airspeed True', 'Latitude At Liftoff', 'Longitude At Liftoff',
             'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Heading True', 'Airspeed True', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Airspeed True', 'Groundspeed', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Airspeed True', 'Altitude AAL', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading True', 'Airspeed True', 'Groundspeed', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading True', 'Airspeed True', 'Altitude AAL', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Heading True', 'Airspeed True', 'Groundspeed', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Heading True', 'Airspeed True', 'Altitude AAL', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Airspeed True', 'Groundspeed', 'Altitude AAL', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading True', 'Airspeed True', 'Groundspeed', 'Altitude AAL', 'Latitude At Liftoff',
             'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown'),
            ('Heading', 'Heading True', 'Airspeed True', 'Groundspeed', 'Altitude AAL',
             'Latitude At Liftoff', 'Longitude At Liftoff', 'Latitude At Touchdown', 'Longitude At Touchdown')
        ]
        self.assertEqual(combinations, expected_combinations)

    @patch("analysis_engine.derived_parameters.air_track", return_value=(None, None))
    def test_derive(self, air_track):
        hdg_mag = P("Heading")
        gspd = P("Groundspeed")
        tas = P("Airspeed True", frequency=2)
        alt_aal = P("Altitude AAL")
        lat_lift = KPV(
            "Latitude At Liftoff",
            items=[KeyPointValue(name="Latitude At Liftoff", index=10, value=20)],
        )
        lon_lift = KPV(
            "Longitude At Liftoff",
            items=[KeyPointValue(name="Longitude At Liftoff", index=10, value=120)],
        )
        lat_land = KPV(
            "Latitude At Touchdown",
            items=[KeyPointValue(name="Latitude At Touchdown", index=200, value=50)],
        )
        lon_land = KPV(
            "Longitude At Touchdown",
            items=[KeyPointValue(name="Longitude At Touchdown", index=200, value=70)],
        )
        node = LongitudePrepared()
        node.derive(
            hdg_mag, None, tas, gspd, alt_aal, lat_lift, lon_lift, lat_land, lon_land
        )
        air_track.assert_called_with(
            20, 120, 50, 70, gspd.array, hdg_mag.array, alt_aal.array, 2
        )

    @patch("analysis_engine.derived_parameters.air_track", return_value=(None, None))
    def test_derive_empty_liftoff_coords(self, air_track):
        hdg_mag = P("Heading")
        gspd = P("Groundspeed")
        tas = P("Airspeed True", frequency=2)
        alt_aal = P("Altitude AAL")
        lat_lift = KPV("Latitude At Liftoff")
        lon_lift = KPV("Longitude At Liftoff")
        lat_land = KPV(
            "Latitude At Touchdown",
            items=[KeyPointValue(name="Latitude At Touchdown", index=200, value=50)],
        )
        lon_land = KPV(
            "Longitude At Touchdown",
            items=[KeyPointValue(name="Longitude At Touchdown", index=200, value=70)],
        )
        node = LongitudePrepared()
        node.derive(
            hdg_mag, None, tas, gspd, alt_aal, lat_lift, lon_lift, lat_land, lon_land
        )
        air_track.assert_not_called()
        self.assertFalse(node.array > 0)


class TestLongitudeSmoothed(unittest.TestCase):
    def test_can_operate(self):
        combinations = LongitudeSmoothed.get_operational_combinations()
        self.assertTrue(all('Longitude Prepared' in c for c in combinations))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestMagneticVariation(unittest.TestCase):
    def test_can_operate(self):
        combinations = MagneticVariation.get_operational_combinations()
        self.assertTrue(
            ('Latitude', 'Longitude', 'Altitude AAL', 'Start Datetime') in combinations)
        self.assertTrue(
            ('Latitude (Coarse)', 'Longitude (Coarse)', 'Altitude AAL', 'Start Datetime') in combinations)
        self.assertTrue(
            ('Latitude', 'Latitude (Coarse)', 'Longitude', 'Longitude (Coarse)', 'Altitude AAL', 'Start Datetime') in combinations)

    def test_derive(self):
        mag_var = MagneticVariation()
        lat = P('Latitude', array=np.ma.arange(10, 14, 0.01))
        lat.array[3] = np.ma.masked
        lon = P('Longitude', array=np.ma.arange(-10, -14, -0.01))
        lon.array[2:4] = np.ma.masked
        alt_aal = P('Altitude AAL', array=np.ma.arange(20000, 24000, 10))
        alt_aal.array[4] = np.ma.masked
        start_datetime = A('Start Datetime',
                           value=datetime.datetime(2013, 3, 23))
        mag_var.derive(lat, None, lon, None, alt_aal, start_datetime)
        ma_test.assert_masked_array_almost_equal(
            mag_var.array[0:10],
            np.ma.array([-5.989, -5.99, -5.991, -5.992, -5.994, -5.995, -5.996,
                         -5.997, -5.998, -5.999]), decimal=3)
        # Test with Coarse parameters.
        mag_var.derive(None, lat, None, lon, alt_aal, start_datetime)
        ma_test.assert_masked_array_almost_equal(
            mag_var.array[300:310],
            np.ma.array([-6.425, -6.427, -6.428, -6.43, -6.432, -6.434, -6.435,
                         -6.437, -6.439, -6.44]), decimal=3)


class TestMagneticVariationFromRunway(unittest.TestCase):
    def test_can_operate(self):
        opts = MagneticVariationFromRunway.get_operational_combinations()
        self.assertEqual(opts,
                    [('Magnetic Variation',
                     'Heading During Takeoff',
                     'Heading During Landing',
                     'FDR Takeoff Runway',
                     'FDR Landing Runway',
                     )])

    def test_derive_both_runways(self):
        # Updates to the derived parameter. New data from
        # segment: d9785fefb912dc77b6e936a4c16de96e5fad868c8cc4cff4e568fb1cff8b87a5
        toff_rwy = {'end': {'elevation': 76,
                            'latitude': 28.45938337469715,
                            'longitude': -13.863540217017873},
                    'id': 9893,
                    'identifier': '01',
                    'magnetic_heading': 6.0,
                    'start': {'elevation': 24,
                              'latitude': 28.441860630699686,
                              'longitude': -13.86412042467909},
                    'strip': {'id': 4947, 'length': 11175,
                              'surface': 'ASP', 'width': 147}}
        land_rwy = {'end': {'elevation': -12,
                            'latitude': 52.32863339288183,
                            'longitude': 4.70886201465568},
                    'id': 8089,
                    'identifier': '18R',
                    'magnetic_heading': 183.0,
                    'start': {'elevation': -13,
                              'latitude': 52.36023790740991,
                              'longitude': 4.7117260390982185},
                    'strip': {'id': 4045, 'length': 12467,
                              'surface': 'ASP', 'width': 183}}
        mag_var = load(os.path.join(test_data_path, 'magnetic_variation_from_runway__mag_vari.nod'))
        mag_var_rwy = MagneticVariationFromRunway()
        mag_var_rwy.derive(
            mag_var,
            KPV([KeyPointValue(index=222.0546875, value=9.4921875, name='Heading During Takeoff')]),
            KPV([KeyPointValue(index=3626.3046875, value=185.9765625, name='Heading During Landing')]),
            A('FDR Takeoff Runway', toff_rwy),
            A('FDR Landing Runway', land_rwy)
        )
        # 0 to takeoff index variation
        self.assertAlmostEqual(mag_var_rwy.array[0], -7.83, places=2)
        self.assertAlmostEqual(mag_var_rwy.array[222], -7.82, places=2)
        # landing index to end
        self.assertAlmostEqual(mag_var_rwy.array[3626], -2.81, places=2)
        self.assertAlmostEqual(mag_var_rwy.array[-1], -2.79, places=2)

    def test_derive_only_takeoff_available(self):
        toff_rwy = {'end': {'elevation': 10,
                            'latitude': 52.7100630002283,
                            'longitude': -8.907803520515461},
                    'start': {'elevation': 43,
                              'latitude': 52.69327604095164,
                              'longitude': -8.943465355819775},
                    'strip': {'id': 2014, 'length': 10495,
                              'surface': 'ASP', 'width': 147}}
        land_rwy = {# MISSING VITAL LAT/LONG INFORMATION
                    'strip': {'id': 2322, 'length': 8858,
                              'surface': 'ASP', 'width': 197}}
        mag_var = P('Magnetic Variation', np.ma.zeros(14272))
        mag_var_rwy = MagneticVariationFromRunway()
        mag_var_rwy.derive(
            mag_var,
            KPV([KeyPointValue(index=62.143, value=58.014, name='Heading During Takeoff')]),
            KPV([KeyPointValue(index=213.869, value=266.5128, name='Heading During Landing')]),
            A('FDR Takeoff Runway', toff_rwy),
            A('FDR Landing Runway', land_rwy)
        )
        # 0 to takeoff index variation
        self.assertAlmostEqual(mag_var_rwy.array[0], -5.87, places=2)
        self.assertAlmostEqual(mag_var_rwy.array[62], -5.87, places=2)
        # landing index to end
        self.assertAlmostEqual(mag_var_rwy.array[213], -5.87, places=2)
        self.assertAlmostEqual(mag_var_rwy.array[-1], -5.87, places=2)

    def test_derive_landing_rollover(self):
        toff_rwy = {'end': {'elevation': 354, 'latitude': 32.697986, 'longitude': -83.644942},
                    'id': 13451,
                    'identifier': '05',
                    'magnetic_heading': 52.9,
                    'start': {'elevation': 354, 'latitude': 32.686022, 'longitude': -83.660633},
                    'strip': {'id': 6726, 'length': 6501, 'surface': 'ASP', 'width': 150}}
        land_rwy = {'end': {'elevation': 1289,
                            'latitude': 36.9130564484,
                            'longitude': -94.012504957},
                    'id': 19503,
                    'identifier': '36',
                    'magnetic_heading': 359.0,
                    'start': {'elevation': 1289,
                              'latitude': 36.899355805,
                              'longitude': -94.0129958013},
                    'strip': {'id': 12132, 'length': 5000, 'surface': 'ASPH', 'width': 75}}
        mag_var = P('Magnetic Variation', np.ma.array([0.0]*9336))
        mag_var_rwy = MagneticVariationFromRunway()
        mag_var_rwy.derive(
            mag_var,
            KPV([KeyPointValue(index=148.9833984375, value=53.00950000000029, name='Heading During Takeoff')]),
            KPV([KeyPointValue(index=1580.7333984375, value=359.8320000000002, name='Heading During Landing')]),
            A('FDR Takeoff Runway', toff_rwy),
            A('FDR Landing Runway', land_rwy)
        )
        # 0 to takeoff index variation
        self.assertTrue(np.all(mag_var_rwy.array[:149] == -5.1902489670943481))
        # landing index to end
        self.assertTrue(np.all(mag_var_rwy.array[1580:] == 1.8087827938368832))


class TestPitchRate(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRelief(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRoll(unittest.TestCase):
    def test_can_operate(self):
        opts = Roll.get_operational_combinations()
        self.assertTrue(('Heading Continuous', 'Altitude AAL',) in opts)
        self.assertTrue(('Roll (1)', 'Roll (2)',) in opts)

    def test_derive(self):
        time = np.arange(100)
        two_time = np.arange(200)
        zero = np.array([0]*100)
        ht_values = np.concatenate([zero, 2000.0*(1.0-np.cos(two_time*np.pi*0.01)), zero])
        ht=P('Altitude AAL', array=np.ma.array(ht_values), frequency=2.0)
        hdg_values = np.concatenate([20.0*(np.sin(time*np.pi*0.03)), zero])
        hdg_values += 120 # Datum heading offset
        hdg=P('Heading', array=np.ma.array(hdg_values), frequency=1.0)
        derroll=Roll()
        derroll.derive(None, None, hdg, ht)
        self.assertLess(derroll.array[40], 0.25)
        self.assertLess(np.ma.max(derroll.array),13.0)
        self.assertGreater(np.ma.max(derroll.array),11.0)


class TestRollFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = RollFor3Sec
        self.operational_combinations = [
            ('Roll',),
        ]
        self.roll = P(
            name='Roll',
            array=np.ma.repeat((10, 11, 12, 10), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_name_and_units(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Roll For 3 Sec')
        self.assertEqual(node.units, ut.DEGREE)

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.roll])
        expected = np.ma.repeat((10, 11, 10), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.roll.frequency = 1
        node = self.node_class()
        node.get_derived([self.roll])
        expected = np.ma.repeat((10, 10.5, 11, 10), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestRollRate(unittest.TestCase):
    def test_can_operate(self):
        opts = RollRate.get_operational_combinations()
        self.assertTrue(('Roll',) in opts)

    def test_derive(self):
        roll = P(array=[0,2,4,6,8,10,12], name='Roll', frequency=2.0)
        rr = RollRate()
        rr.derive(roll)
        expected=np_ma_ones_like(roll.array)*4.0
        assert_array_equal(expected[2:4], rr.array[2:4]) # Differential process blurs ends of the array, so just test the core part.


class TestRollRateForTouchdown(unittest.TestCase):
    def test_derive(self):
        roll_rate=P('Roll', np.array([-3.2, -0.5, 2.3, 0.9, 0.7, 0.7, 0.5, 0.5]), frequency=2)
        node = RollRateForTouchdown()
        node.derive(roll_rate)
        assert_array_almost_equal(node.array, [0, 5.4, 5.6, -2.8, -0.4, 0, -0.4, 0])


class TestRollRateAtTouchdownLimit(unittest.TestCase):
    def test_derive(self):
        gw = P('Gross Weight Smoothed', np.ma.masked_array(
            [10000, 20000, 20000, 21800, 34000, 38790, 40000, 50000],
            mask=[False, True, False, False, False, False, False, False]))
        expected_result = np.ma.masked_array(
            [0, 0, 0, 12.4, 8.02, 6.3, 0, 0],
            mask=[True, True, True, False, False, False, True, True])
        node = RollRateAtTouchdownLimit()
        node.derive(gw)
        assert_array_almost_equal(node.array, expected_result, decimal=3)


class TestRudderPedalCapt(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(RudderPedalCapt.get_operational_combinations(),
                         [('Rudder Pedal (Capt) (1)',),
                          ('Rudder Pedal (Capt) (2)',),
                          ('Rudder Pedal (Capt) (1)', 'Rudder Pedal (Capt) (2)')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRudderPedalFO(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(RudderPedalFO.get_operational_combinations(),
                         [('Rudder Pedal (FO) (1)',),
                          ('Rudder Pedal (FO) (2)',),
                          ('Rudder Pedal (FO) (1)', 'Rudder Pedal (FO) (2)')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestRudderPedal(unittest.TestCase):
    def test_can_operate(self):
        opts = RudderPedal.get_operational_combinations()
        self.assertTrue(('Rudder Pedal (Capt)',) in opts)
        self.assertTrue(('Rudder Pedal (FO)',) in opts)
        self.assertTrue(('Rudder Pedal (Capt)', 'Rudder Pedal (FO)') in opts)
        self.assertTrue(('Rudder Pedal Potentiometer',) in opts)
        self.assertTrue(('Rudder Pedal Synchro',) in opts)
        self.assertTrue(('Rudder Pedal Potentiometer', 'Rudder Pedal Synchro') in opts)


    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSlatAngle(unittest.TestCase):

    def setUp(self):
        self.node_class = SlatAngle

        ac_model = A('Model', 'B777-212')
        ac_series = A('Series', 'B777-200')
        ac_family = A('Family', 'B777')
        self.ac_info_attributes = (ac_model, ac_series, ac_family)



    @patch('analysis_engine.derived_parameters.at')
    def test_can_operate(self, at):
        self.assertTrue(self.node_class().can_operate(('Slat Angle (L)',)))
        self.assertTrue(self.node_class().can_operate(('Slat Angle (R)',)))
        self.assertTrue(self.node_class().can_operate(('Slat Angle (L)', 'Slat Angle (R)')))

        at.get_slat_map.side_effect = ({}, KeyError)
        self.assertTrue(self.node_class.can_operate(
            ('Slat Fully Extended', 'Model', 'Series', 'Family'),
            model=A('Model', 'CRJ900 (CL-600-2D24)'),
            series=A('Series', 'CRJ900'),
            family=A('Family', 'CL-600'),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Slat Fully Extended', 'Model', 'Series', 'Family'),
            model=A('Model', 'B737-333'),
            series=A('Series', 'B737-300'),
            family=A('Family', 'B737 Classic'),
        ))
        self.assertTrue(self.node_class().can_operate(('Slat Angle Recorded',)))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')

    @patch('analysis_engine.derived_parameters.at')
    def test_derive_discretes(self, at):
        at.get_slat_map.return_value = {f: str(f) for f in (0, 22, 32)}

        slat_full = M(
            name='Slat Fully Extended',
            array=np.ma.repeat((0, 0, 1, 0, 0), 10),
            values_mapping={0: '-', 1: 'Extended'},
        )
        slat_part = M(
            name='Slat Part Extended',
            array=np.ma.repeat((0, 1, 0, 1, 0), 10),
            values_mapping={0: '-', 1: 'Part Extended'},
        )
        slat_retracted = M(
            name='Slat Retracted',
            array=np.ma.repeat((1, 0, 0, 0, 1), 10),
            values_mapping={0: '-', 1: 'Retracted'},
        )
        node = self.node_class()
        node.derive(None, None,
                   slat_full,
                   slat_part,
                   slat_retracted,
                   None,
                   *self.ac_info_attributes)
        expected = np.ma.repeat((0, 22, 32, 22, 0), 10)
        assert_almost_equal(node.array, expected)



class TestSlopeToLanding(unittest.TestCase):
    def setUp(self):
        self.node_class = SlopeToLanding

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertEqual(len(combinations), 1)
        combination = set(combinations[0])
        self.assertSetEqual(combination, {'Altitude AAL', 'Distance To Landing',
                                          'Altitude STD', 'SAT', 'Approach'})

    def test_derive(self):
        alt = np.ma.array([3000 , 2000, 1000, 0], dtype=float)
        # Airport is at 3000 ft elevation -> ISA temp is 9 deg
        # Make SAT -1 deg -> 10 deg below ISA
        # 4% height correction per 10 deg ISA deviation
        sat = P('SAT', array=np.ma.ones(4) * -1)
         # Imagine we fly on a 3 deg glideslope. Altimeter would read 4% higher
         # due to low temperature.
        alt_aal = P('Altitude AAL', array=alt * 1.04)
        alt_std = P('Altitude STD', array=alt + 3000)
        apps = buildsection('Approach And Landing', 1, 4)
        dist = P('Distance To Landing', array=np.ma.array([9.42, 6.28, 3.14, 0.0]))
        node = self.node_class()
        node.derive(alt_aal, dist, alt_std, sat, apps)

        expected = np.ma.array(
            np.tan(np.deg2rad(np.ones(4) * 3)),
            mask=[True, False, False, False]
        )
        assert_array_almost_equal(node.array, expected, decimal=3)


class TestSlopeAngleToLanding(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            SlopeAngleToLanding.get_operational_combinations(),
            [('Slope To Landing',)],
        )

    def test_derive(self):
        slope = P('Slope To Landing', np.ma.array([0.5, 1, 2, 4, 8, 16]))
        node = SlopeAngleToLanding()
        node.derive(slope)
        assert_array_almost_equal(node.array, [26.6, 45.0, 63.43, 76.0, 82.9, 86.4], decimal=1)


class TestSlopeToAimingPoint(unittest.TestCase):
    def setUp(self):
        self.node_class = SlopeToAimingPoint

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertEqual(len(combinations), 1)
        combination = set(combinations[0])
        self.assertSetEqual(combination, {'Altitude AAL', 'Aiming Point Range',
                                          'Altitude STD', 'SAT', 'Approach'})

    def test_derive(self):
        alt = np.ma.array([3000 , 2000, 1000, 0], dtype=float)
        # Airport is at 3000 ft elevation -> ISA temp is 9 deg
        # Make SAT -1 deg -> 10 deg below ISA
        # 4% height correction per 10 deg ISA deviation
        sat = P('SAT', array=np.ma.ones(4) * -1)
         # Imagine we fly on a 3 deg glideslope. Altimeter would read 4% higher
         # due to low temperature.
        alt_aal = P('Altitude AAL', array=alt * 1.04)
        alt_std = P('Altitude STD', array=alt + 3000)
        apps = buildsection('Approach And Landing', 1, 4)
        dist = P('Aiming Point Range', array=np.ma.array([9.42, 6.28, 3.14, 0.0]))
        node = self.node_class()
        node.derive(alt_aal, dist, alt_std, sat, apps)

        expected = np.ma.array(
            np.tan(np.deg2rad(np.ones(4) * 3)),
            mask=[True, False, False, False]
        )
        assert_array_almost_equal(node.array, expected, decimal=3)


class TestSlopeAngleToAimingPoint(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            SlopeAngleToAimingPoint.get_operational_combinations(),
            [('Slope To Aiming Point',)],
        )

    def test_derive(self):
        slope = P('Slope To Aiming Point', np.ma.array([0.5, 1, 2, 4, 8, 16]))
        node = SlopeAngleToAimingPoint()
        node.derive(slope)
        assert_array_almost_equal(node.array, [26.6, 45.0, 63.43, 76.0, 82.9, 86.4], decimal=1)


class TestApproachFlightPathAngle(unittest.TestCase):
    def setUp(self):
        self.node_class = ApproachFlightPathAngle

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertAlmostEqual(len(opts), 3)
        for opt in opts:
            self.assertIn('Altitude AAL', opt)
            self.assertIn('SAT', opt)
            self.assertIn('Approach And Landing', opt)
            self.assertTrue('Aiming Point Range' in opt or
                            'Distance To Landing' in opt)

    def test_derive(self):

        dist_land = P('Distance To Landing', np.ma.array([
            9.568, 9.451, 9.334, 9.218, 9.102, 8.987, 8.871, 8.756,
            8.641, 8.527, 8.413, 8.300, 8.187, 8.075, 7.964, 7.853,
            7.741, 7.629, 7.518, 7.406, 7.295, 7.184, 7.072, 6.961,
            6.850, 6.739, 6.629, 6.518, 6.408, 6.298, 6.189, 6.079,
            5.969, 5.859, 5.748, 5.638, 5.528, 5.418, 5.308, 5.198,
            5.088, 4.979, 4.868, 4.758, 4.648, 4.538, 4.429, 4.321,
            4.214, 4.107, 4.000, 3.893, 3.788, 3.683, 3.580, 3.479,
            3.379, 3.281, 3.184, 3.087, 2.989, 2.892, 2.795, 2.697,
            2.599, 2.501, 2.402, 2.304, 2.206, 2.108, 2.010, 1.913,
            1.815, 1.718, 1.620, 1.523, 1.426, 1.329, 1.232, 1.135,
            1.039, 0.942, 0.846, 0.750, 0.654, 0.558, 0.462, 0.366,
            0.270, 0.175, 0.079, 0.051, 0.146, 0.236, 0.317, 0.388,
            0.448, 0.499, 0.541, 0.576, 0.605, 0.629, 0.648, 0.666,
            0.684, 0.702, 0.718,
        ]))

        sat=P('SAT', np.ma.array([
            12.490, 12.461, 12.487, 12.310, 12.238, 12.043, 12.279, 12.286,
            12.311, 12.349, 12.249, 12.302, 12.430, 12.193, 12.475, 12.236,
            11.982, 11.980, 12.249, 12.392, 12.169, 12.132, 12.133, 12.022,
            12.122, 12.429, 12.460, 12.495, 12.547, 12.566, 12.542, 12.666,
            12.780, 12.796, 13.035, 13.150, 13.302, 13.394, 13.591, 13.533,
            13.564, 13.838, 13.783, 13.785, 14.025, 14.026, 14.229, 14.373,
            14.366, 14.620, 14.679, 14.861, 15.069, 15.101, 15.310, 15.583,
            15.670, 15.694, 15.742, 15.720, 15.729, 16.025, 16.116, 16.229,
            16.475, 16.637, 16.626, 16.718, 16.790, 17.025, 16.956, 17.180,
            17.452, 17.479, 17.414, 17.646, 17.714, 17.727, 17.816, 17.978,
            18.179, 18.398, 18.513, 18.671, 18.745, 18.789, 18.895, 18.792,
            19.087, 19.225, 19.281, 19.346, 19.629, 20.805, 21.091, 21.255,
            21.410, 21.412, 21.266, 21.208, 21.286, 20.394, 205.047, 131.322,
            57.646, 279.008, 205.395,
        ], mask=[False]*101 + [True]*6))

        alt_aal = P('Altitude AAL', np.ma.array([
            3005.337, 2962.893, 2920.658, 2879.115, 2838.726, 2799.473, 2761.183, 2723.319,
            2685.825, 2649.282, 2613.708, 2579.813, 2547.529, 2516.689, 2486.646, 2456.238,
            2424.368, 2391.121, 2357.134, 2323.072, 2289.134, 2254.652, 2218.837, 2181.590,
            2143.621, 2105.374, 2067.640, 2030.584, 1994.313, 1959.072, 1924.319, 1889.634,
            1854.362, 1818.152, 1781.745, 1745.967, 1711.448, 1678.343, 1645.720, 1612.825,
            1579.269, 1545.072, 1510.893, 1477.201, 1443.782, 1411.201, 1379.059, 1347.337,
            1315.603, 1283.041, 1249.047, 1213.201, 1176.393, 1139.553, 1104.535, 1072.325,
            1042.911, 1015.566, 988.621, 961.078, 932.362, 902.714, 872.510, 842.146,
            811.325, 779.343, 745.201, 709.257, 672.887, 638.232, 606.584, 577.813,
            550.837, 524.096, 496.380, 467.294, 436.479, 404.911, 373.275, 341.609,
            310.127, 278.436, 246.899, 215.763, 184.930, 154.251, 124.029, 93.305,
            64.424, 41.786, 15.034, 0., 0., 0., 0., 0. ,
            0., 0., 0., 0., 0., 0., 0., 0. ,
            0., 0., 0.,
        ]))

        node = self.node_class()
        node.derive(alt_aal, None, dist_land, sat, buildsection('Approach And Landing', 1, 106))

        self.assertEqual(node.name, "Approach Flight Path Angle")
        self.assertEqual(len(node.array.compressed()), 83)
        self.assertAlmostEqual(node.array.max(), 3.14, places=2)
        self.assertAlmostEqual(node.array.min(), 3.02, places=2)


class TestSpeedbrake(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(Speedbrake.can_operate(('Spoiler (L) (4)', 'Spoiler (R) (4)'),
                                               family=A('Family', 'B737 Classic')))
        self.assertTrue(Speedbrake.can_operate(('Spoiler (L) (4)', 'Spoiler (R) (4)'),
                                               family=A('Family', 'B737 NG')))
        self.assertTrue(Speedbrake.can_operate(('Spoiler (L) (3)', 'Spoiler (R) (3)'),
                                                    family=A('Family', 'B737 MAX')))
        self.assertTrue(Speedbrake.can_operate(('Spoiler (L) (3)', 'Spoiler (R) (3)'),
                                               family=A('Family', 'A320')))
        self.assertTrue(Speedbrake.can_operate(('Spoiler (L) (7)', 'Spoiler (R) (7)'),
                                               family=A('Family', 'B787')))
        self.assertTrue(Speedbrake.can_operate(('Spoiler (L)', 'Spoiler (R)'),
                                               family=A('Family', 'Learjet')))
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L) (3)', 'Spoiler (L) (4)',
             'Spoiler (R) (3)', 'Spoiler (R) (4)'), family=A('Family', 'CRJ 900')))
        self.assertTrue(Speedbrake.can_operate(('Spoiler (L)', 'Spoiler (R)'),
                                               family=A('Family', 'Phenom 300')))
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L) (3)', 'Spoiler (L) (4)', 'Spoiler (L) (5)',
             'Spoiler (R) (3)', 'Spoiler (R) (4)', 'Spoiler (R) (5)'),
            family=A('Family', 'ERJ-190/195')))
        family = A(name='Family', value='CL-600')
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L) (3)', 'Spoiler (L) (4)',
             'Spoiler (R) (3)', 'Spoiler (R) (4)',), family=family))
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L) (3)', 'Spoiler (R) (3)',), family=family))
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L) (2)', 'Spoiler (R) (2)',), family=family))
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L) (1)', 'Spoiler (R) (1)',), family=A('Family', 'G-VI')))
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L)', 'Spoiler (R)',), family=A('Family', 'Citation')))
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L)', 'Spoiler (R)',), family=A('Family', 'Citation VLJ')))
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L)', 'Spoiler (R)',), family=A('Family', 'Phenom 300')))
        self.assertTrue(Speedbrake.can_operate(
            ('Spoiler (L)', 'Spoiler (R)',), family=A('Family', 'Pilatus-PC')))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestSAT(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_basic(self):
        pass
        # Note: the core function machtat2sat is tested by the library test.


class TestSAT_ISA(unittest.TestCase):
    # Note: the core function machtat2sat is tested by the library test.

    def test_can_operate(self):
        self.assertEqual(
            SAT_ISA.get_operational_combinations(),
            [('Altitude STD Smoothed',)])

    def test_basic_altitude(self):
        # -1000 ft = 16.9812 C
        # 0 ft = 15C
        # 5000 ft = 5.09400 C
        # 10000 ft = -4.81200 C
        # 20000 ft = -24.6240 C
        # 30000 ft = -44.4360 C
        # 40000 ft = -56.5000 C
        # 50000 ft = -56.5000 C

        alt=P('Altitude STD Smoothed',
              array=np.array([-1,0,5,10,20,30,40,50])*1000.0)
        sat = SAT_ISA()
        sat.derive(alt)
        expected = np.ma.array([16.9812, 15.0, 5.09400, -4.812, -24.624,
                                -44.4360, -56.5, -56.5])
        assert_array_almost_equal(sat.array, expected)


class TestTAT(unittest.TestCase):
    def test_can_operate(self):
        opts = TAT.get_operational_combinations()
        self.assertIn(('TAT (1)', 'TAT (2)'), opts)
        self.assertIn(('SAT', 'Mach'), opts)

    def test_combination(self):
        t1 = P('TAT (1)', array = [1,3,5])
        t2 = P('TAT (2)', array = [2,4,6])
        tat = TAT()
        tat.derive(t1, t2, None)
        expected = np.ma.arange(1, 7) + 0.5
        # This test correctly ignores the 6th overrun sample which is masked.
        assert_array_almost_equal(tat.array, expected)

    def test_conversion(self):
        sat = P('SAT', array=[0.0, -30.0])
        mach = P('Mach', array=[0.5, 0.8])
        tat = TAT()
        tat.derive(None, None, sat, mach)
        # TAT = SAT (1 + (1.4-1)/2M^2) = 1 + 0.2M^2
        expected = np.ma.array([13.6575, 1.1232])
        assert_array_almost_equal(tat.array, expected)

    def test_tat_has_precedence(self):
        t1 = P('TAT (1)', array=[1,3,5], frequency=0.5)
        t2 = P('TAT (2)', array=[2,4,6], frequency=0.5)
        sat = P('SAT', array=np.ma.ones(6) * (-30.0))
        mach = P('Mach', array=np.ma.ones(6) * (0.8))
        tat = TAT()
        tat.get_derived((t1, t2, sat, mach))
        expected = np.ma.arange(1, 7) + 0.5
        # This test correctly ignores the 6th overrun sample which is masked.
        assert_array_almost_equal(tat.array, expected)
        self.assertEqual(tat.frequency, 0.5 * 2)


class TestTailwind(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestThrustAsymmetry(unittest.TestCase):
    def test_can_operate(self):
        opts = ThrustAsymmetry.get_operational_combinations()
        expected = [
            ('Eng (*) EPR Max', 'Eng (*) EPR Min'),
            ('Eng (*) N1 Max', 'Eng (*) N1 Min'),
            ]
        for exp in expected:
            self.assertIn(exp, opts, msg=exp)

    def test_derive_n1(self):
        n1_max = P('Eng (*) N1 Max', np.arange(10, 30))
        n1_min = P('Eng (*) N1 Min', np.arange(8, 28))
        asym = ThrustAsymmetry()
        asym.derive(None, None, n1_max=n1_max, n1_min=n1_min)
        self.assertEqual(len(asym.array), len(n1_max.array))
        uniq = unique_values(asym.array.astype(int))
        # there should be all 20 values being 2 out
        self.assertEqual(uniq, {2: 20})

    def test_derive_epr(self):
        epr_max = P('Eng (*) EPR Max', np.arange(0.0, 1.8, step=0.2))
        epr_min = P('Eng (*) EPR Min', np.arange(0.2, 2.0, step=0.2))
        asym = ThrustAsymmetry()
        asym.get_derived((epr_max, epr_min, None, None))
        self.assertEqual(len(asym.array), len(epr_max.array))
        self.assertEqual(asym.array[2], -20)
        self.assertAlmostEqual(asym.array[-3], -20)

    def test_derive_epr_and_n1(self):
        n1_max = P('Eng (*) N1 Max', np.arange(10, 30))
        n1_min = P('Eng (*) N1 Min', np.arange(8, 28))
        # create decimal array in the range of 0 to 2
        epr_max = P('Eng (*) EPR Max', np.arange(0.0, 1.8, step=0.2))
        epr_min = P('Eng (*) EPR Min', np.arange(0.2, 2.0, step=0.2))
        asym = ThrustAsymmetry()
        asym.get_derived((epr_max, epr_min, n1_max, n1_min))
        self.assertEqual(len(asym.array), len(n1_max.array))
        self.assertEqual(asym.array[2], 2)


class TestThrottleLevers(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(ThrottleLevers.get_operational_combinations(),
                         [('Eng (1) Throttle Lever',),
                          ('Eng (2) Throttle Lever',),
                          ('Eng (1) Throttle Lever', 'Eng (2) Throttle Lever')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTorqueAsymmetry(unittest.TestCase):

    def setUp(self):
        self.node_class = TorqueAsymmetry

    def test_can_operate(self):
        prop = A('Engine Propulsion', 'PROP')
        jet = A('Engine Propulsion', 'JET')
        self.assertTrue(self.node_class.can_operate(('Eng (*) Torque Max', 'Eng (*) Torque Min'), eng_type=prop))
        self.assertFalse(self.node_class.can_operate(('Eng (*) Torque Max', 'Eng (*) Torque Min'), eng_type=jet))
        self.assertTrue(self.node_class.can_operate(('Eng (*) Torque Max', 'Eng (*) Torque Min'), ac_type=helicopter))


    def test_derive(self):
        torque_max = P('Eng (*) Torque Max', np.arange(10, 30))
        torque_min = P('Eng (*) Torque Min', np.arange(8, 28))

        node = self.node_class()
        node.derive(torque_max, torque_min)

        self.assertEqual(len(node.array), len(torque_max.array))
        uniq = unique_values(node.array.astype(int))
        # there should be all 20 values being 2 out
        self.assertEqual(uniq, {2: 20})


class TestTurbulence(unittest.TestCase):
    def test_can_operate(self):
        opts = Turbulence.get_operational_combinations()
        self.assertEqual(opts, [('Acceleration Vertical',)])

    def test_derive(self):
        accel = np.ma.concatenate((np.ones(40), [2], np.ones(40)))
        turb = Turbulence()
        turb.derive(P('Acceleration Vertical', accel, frequency=8))
        expected = np.ma.concatenate((np.zeros(20), np.ones(41) * 0.156173762, np.zeros(20)))
        np.testing.assert_array_almost_equal(expected, turb.array.data)


class TestVOR1Frequency(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestVOR2Frequency(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestVerticalSpeedInertial(unittest.TestCase):
    def test_can_operate(self):
        opts = VerticalSpeedInertial.get_operational_combinations()
        self.assertEqual(opts, [('Acceleration Vertical', 'Altitude STD Smoothed',
                                 'Altitude Radio Offset Removed', 'Fast',
                                 'Aircraft Type')])

    def test_derive(self):
        time = np.arange(100)
        zero = np.array([0]*50)
        acc_values = np.concatenate([zero, np.cos(time*np.pi*0.02), zero])
        vel_values = np.concatenate([zero, np.sin(time*np.pi*0.02), zero])
        ht_values = np.concatenate([zero, 1.0-np.cos(time*np.pi*0.02), zero])

        # For a 0-400ft leap over 100 seconds, the scaling is 200ft amplitude and 2*pi/100 for each differentiation.
        amplitude = 200.0
        diff = 2.0 * np.pi / 100.0
        ht_values *= amplitude
        vel_values *= amplitude * diff * 60.0 # fpm
        acc_values *= amplitude * diff**2.0 / GRAVITY_IMPERIAL # g

        #import wx
        #import matplotlib.pyplot as plt
        #plt.plot(acc_values,'k')
        #plt.plot(vel_values,'b')
        #plt.plot(ht_values,'r')
        #plt.show()

        az = P('Acceleration Vertical', acc_values)
        alt_std = P('Altitude STD Smoothed', ht_values + 30.0) # Pressure offset
        alt_rad = P('Altitude STD Smoothed', ht_values-2.0) #Oleo compression
        fast = buildsection('Fast', 10, len(acc_values)-10)
        ac_type = A(name='Aircraft Type', value = 'aeroplane')

        vsi = VerticalSpeedInertial()
        vsi.derive(az, alt_std, alt_rad, fast, ac_type)

        expected = vel_values
        np.testing.assert_almost_equal(vsi.array, expected, decimal=-2)

    def test_check_go_around(self):
        # We build two cycles with a "touch and go" in between to check that the
        # algorithm does not treat the first as the taxi out for the second, and the
        # second the taxi in for the first.
        time = np.arange(100)
        zero = np.array([0]*50)
        acc_values = np.concatenate([zero,
                                     np.cos(time*np.pi*0.02),
                                     zero,
                                     np.cos(time*np.pi*0.02),
                                     zero                                     ])
        vel_values = np.concatenate([zero,
                                     np.sin(time*np.pi*0.02),
                                     zero,
                                     np.sin(time*np.pi*0.02),
                                     zero                                     ])
        ht_values = np.concatenate([zero,
                                    1.0-np.cos(time*np.pi*0.02),
                                    zero,
                                    1.0-np.cos(time*np.pi*0.02),
                                    zero                                    ])

        # For a 0-400ft leap over 100 seconds, the scaling is 200ft amplitude and 2*pi/100 for each differentiation.
        amplitude = 200.0
        diff = 2.0 * np.pi / 100.0
        ht_values *= amplitude
        vel_values *= amplitude * diff * 60.0 # fpm
        acc_values *= amplitude * diff**2.0 / GRAVITY_IMPERIAL # g

        az = P('Acceleration Vertical', acc_values)
        alt_std = P('Altitude STD Smoothed', ht_values + 30.0) # Pressure offset
        alt_rad = P('Altitude STD Smoothed', ht_values-2.0) #Oleo compression
        fast = buildsection('Fast', 10, len(acc_values)-10)
        ac_type = A(name='Aircraft Type', value = 'aeroplane')

        vsi = VerticalSpeedInertial()
        vsi.derive(az, alt_std, alt_rad, fast, ac_type)

        # The mean absolute signal should be 269. In the fault case we are checking, it drops
        # to 4, hence 100 is a clear divide between success and failure.
        self.assertGreater(np.ma.average(np.ma.abs(vsi.array)), 100.0)

    def test_masked_alt_std(self):
        time = np.arange(100)
        zero = np.array([0]*50)
        acc_values = np.concatenate([zero, np.cos(time*np.pi*0.02), zero])
        vel_values = np.concatenate([zero, np.sin(time*np.pi*0.02), zero])
        ht_values = np.concatenate([zero, 1.0-np.cos(time*np.pi*0.02), zero])

        # For a 0-400ft leap over 100 seconds, the scaling is 200ft amplitude and 2*pi/100 for each differentiation.
        amplitude = 200.0
        diff = 2.0 * np.pi / 100.0
        ht_values *= amplitude
        vel_values *= amplitude * diff * 60.0 # fpm
        acc_values *= amplitude * diff**2.0 / GRAVITY_IMPERIAL # g

        az = P('Acceleration Vertical', acc_values)
        alt_std = P('Altitude STD Smoothed', ht_values + 30.0) # Pressure offset
        alt_rad = P('Altitude STD Smoothed', ht_values-2.0) #Oleo compression
        fast = buildsection('Fast', 10, len(acc_values)-10)
        # Mask alt_std in Fast section
        alt_std.array[fast[0].slice] = np.ma.masked
        ac_type = A(name='Aircraft Type', value = 'aeroplane')

        vsi = VerticalSpeedInertial()
        vsi.derive(az, alt_std, alt_rad, fast, ac_type)

        np.testing.assert_array_equal(vsi.array.mask, np.ones(vsi.array.shape))


class TestRudder(unittest.TestCase):
    def test_can_operate(self):
        opts = Rudder.get_operational_combinations()
        # Set up currently for 777 which has to have three parts.
        #self.assertIn(('Rudder (Upper)'), opts)
        #self.assertIn(('Rudder (Middle)'), opts)
        #self.assertIn(('Rudder (Lower)'), opts)
        self.assertIn(('Rudder (Upper)', 'Rudder (Middle)', 'Rudder (Lower)'), opts)

    def test_derive(self):
        upper = P('Rudder (Upper)', array=np.ma.zeros(7))
        upper.array[1] = 3
        middle = P('Rudder (Middle)', array=np.ma.zeros(7))
        middle.array[3] = 6
        lower = P('Rudder (Lower)', array=np.ma.zeros(7))
        lower.array[5] = 9
        rud = Rudder()
        rud.derive(upper, middle, lower)
        np.testing.assert_almost_equal(rud.array.data, [0,1,0,2,0,3,0])

class TestStabilizer(unittest.TestCase):
    def test_can_operate(self):
        opts = Stabilizer.get_operational_combinations()
        # Set up currently for 777 which has to have four parts.
        self.assertIn(('Stabilizer (1)', 'Stabilizer (2)', 'Stabilizer (3)', 'Frame'), opts)

    def test_basic(self):
        s1 = P('Stabilizer (1)', array=np.ma.zeros(7))
        s1.array[1] = 3
        s2 = P('Stabilizer (2)', array=np.ma.zeros(7))
        s2.array[3] = 6
        s3 = P('Stabilizer (3)', array=np.ma.zeros(7))
        s3.array[5] = 9
        frame = A('Frame', '777')
        stab = Stabilizer()
        stab.derive(s1, s2, s3, frame)
        result = (np.array([0,1,0,2,0,3,0]) * 0.0503) - 3.4629
        # Blend will null the end values, so only test excluding these.
        np.testing.assert_almost_equal(stab.array.data[1:-1], result[1:-1])

    def test_wrong_frame(self):
        s1 = P('Stabilizer (1)', array=np.ma.zeros(7))
        frame = A('Frame', '787')
        stab = Stabilizer()
        self.assertRaises(ValueError, stab.derive, s1, s1, s1, frame)


class TestWheelSpeed(unittest.TestCase):
    def test_can_operate(self):
        opts = WheelSpeed.get_operational_combinations()
        self.assertEqual(opts,
                         [('Wheel Speed (L)', 'Wheel Speed (R)'),
                          #('Wheel Speed (L)', 'Wheel Speed (C)', 'Wheel Speed (R)'),
                          ])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestWheelSpeedLeft(unittest.TestCase):
    def test_can_operate(self):
        opts = WheelSpeedLeft.get_operational_combinations()
        self.assertIn(('Wheel Speed (L) (1)', 'Wheel Speed (L) (2)'), opts)
        self.assertIn(('Wheel Speed (L) (1)', 'Wheel Speed (L) (2)', 'Wheel Speed (L) (3)', 'Wheel Speed (L) (4)'), opts)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestWheelSpeedRight(unittest.TestCase):
    def test_can_operate(self):
        opts = WheelSpeedRight.get_operational_combinations()
        self.assertIn(('Wheel Speed (R) (1)', 'Wheel Speed (R) (2)'), opts)
        self.assertIn(('Wheel Speed (R) (1)', 'Wheel Speed (R) (2)', 'Wheel Speed (R) (3)', 'Wheel Speed (R) (4)'), opts)

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        pass


class TestWindDirectionContinuous(unittest.TestCase):
    @unittest.skip('Test Not Implemented')
    def test_can_operate(self):
        self.assertTrue(False, msg='Test not implemented.')

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestWindDirection(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(WindDirection.can_operate(('Wind Direction (1)',)))
        self.assertTrue(WindDirection.can_operate(('Wind Direction (2)',)))
        self.assertTrue(WindDirection.can_operate(('Wind Direction (1)',
                                                   'Wind Direction (2)',)))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestCoordinatesSmoothed(TemporaryFileTest, unittest.TestCase):
    def setUp(self):
        self.approaches = App('Approach Information',
            items=[ApproachItem('GO_AROUND', slice(3198.0, 3422.0),
                            ils_freq=108.55,
                            gs_est=slice(3200, 3390),
                            loc_est=slice(3199, 3445),
                            airport={'code': {'iata': 'KDH', 'icao': 'OAKN'},
                                     'distance': 2.483270162497824,
                                     'elevation': 3301,
                                     'id': 3279,
                                     'latitude': 31.5058,
                                     'location': {'country': 'Afghanistan'},
                                     'longitude': 65.8478,
                                     'magnetic_variation': 'E001590 0506',
                                     'name': 'Kandahar'},
                            landing_runway={'end': {'elevation': 3294,
                                            'latitude': 31.497511,
                                            'longitude': 65.833933},
                                    'id': 44,
                                    'identifier': '23',
                                    'magnetic_heading': 232.9,
                                    'start': {'elevation': 3320,
                                              'latitude': 31.513997,
                                              'longitude': 65.861714},
                                    'strip': {'id': 22,
                                              'length': 10532,
                                              'surface': 'ASP',
                                              'width': 147}}),
                   ApproachItem('LANDING', slice(12928.0, 13440.0),
                            ils_freq=111.3,
                            gs_est=slice(13034, 13262),
                            loc_est=slice(12929, 13347),
                            turnoff=13362.455208333333,
                            airport={'code': {'iata': 'DXB', 'icao': 'OMDB'},
                                     'distance': 1.6842014290716794,
                                     'id': 3302,
                                     'latitude': 25.2528,
                                     'location': {'city': 'Dubai',
                                                  'country': 'United Arab Emirates'},
                                     'longitude': 55.3644,
                                     'magnetic_variation': 'E001315 0706',
                                     'name': 'Dubai Intl'},
                            landing_runway={'end': {'latitude': 25.262131, 'longitude': 55.347572},
                                    'glideslope': {'angle': 3.0,
                                                   'latitude': 25.246333,
                                                   'longitude': 55.378417,
                                                   'threshold_distance': 1508},
                                    'id': 22,
                                    'identifier': '30L',
                                    'localizer': {'beam_width': 4.5,
                                                  'frequency': 111300.0,
                                                  'heading': 300,
                                                  'latitude': 25.263139,
                                                  'longitude': 55.345722},
                                    'magnetic_heading': 299.7,
                                    'start': {'latitude': 25.243322, 'longitude': 55.381519},
                                    'strip': {'id': 11,
                                              'length': 13124,
                                              'surface': 'ASP',
                                              'width': 150}})])

        self.toff = [Section(name='Takeoff',
                             slice=slice(372, 414, None),
                             start_edge=371.32242063492066,
                             stop_edge=413.12204760355382)]

        self.toff_rwy = A(name = 'FDR Takeoff Runway',
                          value = {'end': {'elevation': 4843,
                                           'latitude': 34.957972,
                                           'longitude': 69.272944},
                                   'id': 41,
                                   'identifier': '03',
                                   'magnetic_heading': 26.0,
                                   'start': {'elevation': 4862,
                                             'latitude': 34.934306,
                                             'longitude': 69.257},
                                   'strip': {'id': 21,
                                             'length': 9852,
                                             'surface': 'CON',
                                             'width': 179}})

        self.source_file_path = os.path.join(
            test_data_path, 'flight_with_go_around_and_landing.hdf5')
        super(TestCoordinatesSmoothed, self).setUp()

    # Skipped by DJ's advice: too many changes withoud updating the test
    @unittest.skip('Test Out Of Date')
    def test__adjust_track_precise(self):
        with hdf_file(self.test_file_path) as hdf:
            lon = hdf['Longitude']
            lat = hdf['Latitude']
            ils_loc =hdf['ILS Localizer']
            app_range = hdf['ILS Localizer Range']
            gspd = hdf['Groundspeed']
            hdg = hdf['Heading True Continuous']
            tas = hdf['Airspeed True']
            rot = hdf['Rate Of Turn']

        precision = A(name='Precise Positioning', value = True)
        mobile = Mobile()
        mobile.get_derived((rot, gspd))

        cs = CoordinatesSmoothed()
        lat_new, lon_new = cs._adjust_track(
            lon, lat, ils_loc, app_range, hdg, gspd, tas,
            self.toff, self.toff_rwy, self.approaches, mobile, precision)

        chunks = np.ma.clump_unmasked(lat_new)
        self.assertEqual(len(chunks),3)
        self.assertEqual(chunks,[slice(44, 372, None),
                                 slice(3200, 3445, None),
                                 slice(12930, 13424, None)])

    # Skipped by DJ's advice: too many changes withoud updating the test
    @unittest.skip('Test Out Of Date')
    def test__adjust_track_imprecise(self):
        with hdf_file(self.test_file_path) as hdf:
            lon = hdf['Longitude']
            lat = hdf['Latitude']
            ils_loc =hdf['ILS Localizer']
            app_range = hdf['ILS Localizer Range']
            gspd = hdf['Groundspeed']
            hdg = hdf['Heading True Continuous']
            tas = hdf['Airspeed True']
            rot = hdf['Rate Of Turn']

        precision = A(name='Precise Positioning', value = False)

        mobile = Mobile()
        mobile.get_derived((rot, gspd))
        cs = CoordinatesSmoothed()
        lat_new, lon_new = cs._adjust_track(
            lon, lat, ils_loc, app_range, hdg, gspd, tas,
            self.toff, self.toff_rwy, self.approaches, mobile, precision)

        chunks = np.ma.clump_unmasked(lat_new)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks,[slice(44,414),slice(12930,13424)])


        #import matplotlib.pyplot as plt
        #plt.plot(lat_new, lon_new)
        #plt.show()
        #plt.plot(lon.array, lat.array)
        #plt.show()

    # Skipped by DJ's advice: too many changes withoud updating the test
    @unittest.skip('Test Out Of Date')
    def test__adjust_track_visual(self):
        with hdf_file(self.test_file_path) as hdf:
            lon = hdf['Longitude']
            lat = hdf['Latitude']
            ils_loc =hdf['ILS Localizer']
            app_range = hdf['ILS Localizer Range']
            gspd = hdf['Groundspeed']
            hdg = hdf['Heading True Continuous']
            tas = hdf['Airspeed True']
            rot = hdf['Rate Of Turn']

        precision = A(name='Precise Positioning', value = False)
        mobile = Mobile()
        mobile.get_derived((rot, gspd))

        self.approaches.value[0].pop('ILS localizer established')
        self.approaches.value[1].pop('ILS localizer established')
        # Don't need to pop the glideslopes as these won't be looked for.
        cs = CoordinatesSmoothed()
        lat_new, lon_new = cs._adjust_track(
            lon, lat, ils_loc, app_range, hdg, gspd, tas,
            self.toff, self.toff_rwy, self.approaches, mobile, precision)

        chunks = np.ma.clump_unmasked(lat_new)
        self.assertEqual(len(chunks),2)
        self.assertEqual(chunks,[slice(44,414),slice(12930,13424)])


class TestApproachRange(TemporaryFileTest, unittest.TestCase):
    def setUp(self):
        self.approaches = App(items=[
            ApproachItem('GO_AROUND', slice(3198, 3422),
                     ils_freq=108.55,
                     gs_est=slice(3200, 3390),
                     loc_est=slice(3199, 3445),
                     airport={'code': {'iata': 'KDH', 'icao': 'OAKN'},
                              'distance': 2.483270162497824,
                              'elevation': 3301,
                              'id': 3279,
                              'latitude': 31.5058,
                              'location': {'country': 'Afghanistan'},
                              'longitude': 65.8478,
                              'magnetic_variation': 'E001590 0506',
                              'name': 'Kandahar'},
                     landing_runway={'end': {'elevation': 3294,
                                     'latitude': 31.497511,
                                     'longitude': 65.833933},
                             'id': 44,
                             'identifier': '23',
                             'magnetic_heading': 232.9,
                             'start': {'elevation': 3320,
                                       'latitude': 31.513997,
                                       'longitude': 65.861714},
                             'strip': {'id': 22,
                                       'length': 10532,
                                       'surface': 'ASP',
                                       'width': 147}}),
            ApproachItem('LANDING', slice(12928, 13440),
                     ils_freq=111.3,
                     gs_est=slice(13034, 13262),
                     loc_est=slice(12929, 13347),
                     turnoff=13362.455208333333,
                     airport={'code': {'iata': 'DXB', 'icao': 'OMDB'},
                              'distance': 1.6842014290716794,
                              'id': 3302,
                              'latitude': 25.2528,
                              'location': {'city': 'Dubai',
                                           'country': 'United Arab Emirates'},
                              'longitude': 55.3644,
                              'magnetic_variation': 'E001315 0706',
                              'name': 'Dubai Intl'},
                     landing_runway={'end': {'latitude': 25.262131, 'longitude': 55.347572},
                             'glideslope': {'angle': 3.0,
                                            'latitude': 25.246333,
                                            'longitude': 55.378417,
                                            'threshold_distance': 1508},
                             'id': 22,
                             'identifier': '30L',
                             'localizer': {'beam_width': 4.5,
                                           'frequency': 111300.0,
                                           'heading': 300,
                                           'latitude': 25.263139,
                                           'longitude': 55.345722},
                             'magnetic_heading': 299.7,
                             'start': {'latitude': 25.243322, 'longitude': 55.381519},
                             'strip': {'id': 11,
                                       'length': 13124,
                                       'surface': 'ASP',
                                       'width': 150}})])

        self.toff = Section(name='Takeoff',
                       slice=slice(372, 414, None),
                       start_edge=371.32242063492066,
                       stop_edge=413.12204760355382)

        self.toff_rwy = A(name='FDR Takeoff Runway',
                          value={'end': {'elevation': 4843,
                                         'latitude': 34.957972,
                                         'longitude': 69.272944},
                                 'id': 41,
                                 'identifier': '03',
                                 'magnetic_heading': 26.0,
                                 'start': {'elevation': 4862,
                                           'latitude': 34.934306,
                                           'longitude': 69.257},
                                 'strip': {'id': 21,
                                           'length': 9852,
                                           'surface': 'CON',
                                           'width': 179}})

        self.source_file_path = os.path.join(
            test_data_path, 'flight_with_go_around_and_landing.hdf5')
        super(TestApproachRange, self).setUp()

    def test_can_operate(self):
        operational_combinations = ApproachRange.get_operational_combinations()
        self.assertTrue(('Heading True', 'Airspeed True', 'Altitude AAL', 'Approach Information') in operational_combinations, msg="Missing 'Heading True' combination")
        self.assertTrue(('Track True', 'Airspeed True', 'Altitude AAL', 'Approach Information') in operational_combinations, msg="Missing 'Track True' combination")
        self.assertTrue(('Track', 'Airspeed True', 'Altitude AAL', 'Approach Information') in operational_combinations, msg="Missing 'Track' combination")
        self.assertTrue(('Heading', 'Airspeed True', 'Altitude AAL', 'Approach Information') in operational_combinations, msg="Missing 'Heading' combination")

    def test_range_basic(self):
        with hdf_file(self.test_file_path) as hdf:
            hdg = hdf['Heading True']
            tas = hdf['Airspeed True']
            alt = hdf['Altitude AAL']
            glide = hdf['ILS Glideslope']

        ar = ApproachRange()
        ar.derive(None, glide, None, None, None, hdg, tas, alt, self.approaches)
        result = ar.array
        chunks = np.ma.clump_unmasked(result)
        self.assertEqual(len(chunks),1)
        self.assertEqual(chunks,[slice(12928, 13440, None)])

    def test_range_full_param_set(self):
        with hdf_file(self.test_file_path) as hdf:
            hdg = hdf['Track True']
            tas = hdf['Airspeed True']
            alt = hdf['Altitude AAL']
            glide = hdf['ILS Glideslope']
            gspd = hdf['Groundspeed']

        ar = ApproachRange()
        ar.derive(gspd, glide, None, None, hdg, None, tas, alt, self.approaches)
        result = ar.array
        chunks = np.ma.clump_unmasked(result)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks,[slice(12928, 13440, None)])


class TestZeroFuelWeight(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ZeroFuelWeight
        self.operational_combinations = [
            ('HDF Duration', 'Fuel Qty', 'Gross Weight'),
            ('HDF Duration', 'Dry Operating Weight',),
            ('HDF Duration', 'Dry Operating Weight', 'Payload'),
        ]
        self.duration = A('HDF Duration', 10)

    def test_derive_fuel_qty_gross_wgt(self):
        fuel_qty = P('Fuel Qty', np.ma.array([1, 2, 3, 4]))
        gross_wgt = P('Gross Weight', np.ma.array([11, 12, 13, 14]))
        zfw = ZeroFuelWeight()
        zfw.derive(fuel_qty, gross_wgt, None, None, self.duration)
        self.assertTrue((zfw.array == 10).all())

    def test_derive_dry_operating_wgt(self):
        dry_operating_wgt = A('Dry Operating Weight', 100000)
        zfw = ZeroFuelWeight()
        zfw.derive(None, None, dry_operating_wgt, None, self.duration)
        self.assertTrue((zfw.array == dry_operating_wgt.value).all())

    def test_derive_dry_operating_wgt_payload(self):
        dry_operating_wgt = A('Dry Operating Weight', 100000)
        payload = A('Payload', None)
        zfw = ZeroFuelWeight()
        zfw.derive(None, None, dry_operating_wgt, payload, self.duration)
        self.assertTrue((zfw.array == dry_operating_wgt.value).all())

        payload = A('Payload', 1000)
        zfw = ZeroFuelWeight()
        zfw.derive(None, None, dry_operating_wgt, payload, self.duration)
        self.assertTrue((zfw.array == 101000).all())


class TestGrossWeight(unittest.TestCase):
    def test_can_operate(self):
        combinations = GrossWeight.get_operational_combinations()
        self.assertTrue(('Zero Fuel Weight', 'Fuel Qty') in combinations)
        self.assertTrue(('HDF Duration', 'AFR Landing Gross Weight',) in combinations)
        self.assertTrue(('HDF Duration',
                         'AFR Landing Gross Weight',
                         'AFR Takeoff Gross Weight',
                         'Touchdown',
                         'Liftoff') in combinations)
        self.assertTrue(('HDF Duration',
                         'AFR Takeoff Gross Weight',
                         'AFR Landing Fuel',
                         'AFR Takeoff Fuel') in combinations)

    def test_derive_fuel_qty_zfw(self):
        fq = P('Fuel Qty', array=np.ma.array([40,30,20,10]))
        zfw = P('Zero Fuel Weight', array=np.ma.array([990, 1000, 1000, 1100]))
        node = GrossWeight()
        node.derive(zfw, fq, None, None, None, None, None, None, None)
        self.assertEqual(node.array.tolist(), [1040, 1030, 1020, 1010])

    def test_derive_afr_land_wgt(self):
        duration = A('HDF Duration', 100)
        afr_land_wgt = A('AFR Landing Gross Weight', 1000)

        node = GrossWeight()
        node.derive(None, None, duration, afr_land_wgt, None, None, None, None, None)

        assert_array_equal(node.array, np.ones(100) * 1000)

    def test_derive_afr_interpolated_wgt(self):
        duration = A('HDF Duration', 100)
        afr_takeoff_wgt = A('AFR Takeoff Gross Weight', 2000)
        afr_land_wgt = A('AFR Landing Gross Weight', 1000)
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(10)])
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(90)])

        node = GrossWeight()
        node.derive(None, None, duration, afr_land_wgt, afr_takeoff_wgt,
                    None, None, touchdowns, liftoffs)

        self.assertTrue(node.array.mask[:10].all())
        self.assertTrue(node.array.mask[-9:].all())
        self.assertEqual(node.array[10], 2000)
        self.assertEqual(node.array[-10], 1000)
        self.assertTrue(node.array.mask[:10].all())
        self.assertAlmostEqual(node.array[50], 1500.0, 1)

    def test_using_zero_fuel_weight(self):
        fuel_qty_array = np.ma.arange(10000, 4000, -100)
        fuel_qty_array[10] *= 0.9
        fuel_qty = P('Fuel Qty', array=fuel_qty_array)

        zfw_array = np_ma_ones_like(fuel_qty_array) * 17400
        zfw = P('Zero Fuel Weight', array=zfw_array)

        gw = GrossWeight()
        gw.derive(zfw, fuel_qty, None, None, None, None, None, None, None)

        self.assertEqual(len(gw.array), 60)
        self.assertEqual(gw.array[4], (10000-(4*100)+17400))

        # check fuel quantity decrease at liftoff by 90%
        # only has limited 4% effect on array
        self.assertLess(gw.array[10], (10000-(10*100)+17400))
        self.assertGreater(gw.array[10], (10000-(10*100)+17400)*0.96)
        self.assertEqual(gw.array[25], (10000-(25*100)+17400))

    def test_derive_afr_interpolated_ldg_wgt(self):
        duration = A('HDF Duration', 300)
        afr_takeoff_wgt = A('AFR Takeoff Gross Weight', 20000)
        afr_takeoff_fuel = A('AFR Takeoff Fuel', 5000)
        afr_land_fuel = A('AFR Landing Fuel', 2500)
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(25)])
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(275)])

        node = GrossWeight()
        node.derive(None, None, duration, None, afr_takeoff_wgt,
                    afr_land_fuel, afr_takeoff_fuel, touchdowns, liftoffs)

        self.assertTrue(node.array[10] is np.ma.masked)
        self.assertEqual(node.array[25], 20000)
        self.assertAlmostEqual(node.array[50], 19750, delta=10) # delta 10 = 1 sample
        self.assertAlmostEqual(node.array[250], 17750, delta=10) # delta 10 = 1 sample
        self.assertEqual(node.array[275], 17500)
        self.assertTrue(node.array[290] is np.ma.masked)

##############################################################################
# Velocity Speeds


########################################
# Takeoff Safety Speed (V2)


#class TestV2(unittest.TestCase, NodeTest):

    #def setUp(self):
        #self.node_class = V2
        #self.airspeed = P('Airspeed', np.ma.repeat(200, 1280))
        #self.liftoffs = KTI(name='Liftoff', items=[
            #KeyTimeInstance(name='Liftoff', index=20),
            #KeyTimeInstance(name='Liftoff', index=860),
        #])
        #self.climbs = KTI(name='Climb Start', items=[
            #KeyTimeInstance(name='Climb Start', index=420),
            #KeyTimeInstance(name='Climb Start', index=1060),
        #])

    #def test_can_operate(self):
        ## AFR:
        #self.assertTrue(self.node_class.can_operate(
            #('Airspeed', 'AFR V2', 'Liftoff', 'Climb Start'),
            #afr_v2=A('AFR V2', 120),
        #))
        #self.assertFalse(self.node_class.can_operate(
            #('Airspeed', 'AFR V2', 'Liftoff', 'Climb Start'),
            #afr_v2=A('AFR V2', 70),
        #))
        ## Embraer:
        #self.assertTrue(self.node_class.can_operate(
            #('Airspeed', 'V2-Vac', 'Liftoff', 'Climb Start'),
        #))
        ## Airbus:
        #self.assertTrue(self.node_class.can_operate(
            #('Airspeed', 'Airspeed Selected', 'Speed Control', 'Liftoff', 'Climb Start', 'Manufacturer'),
            #manufacturer=A('Manufacturer', 'Airbus'),
        #))

    #def test_derive__nothing(self):
        #node = self.node_class()
        #node.derive(self.airspeed, None, None, None, None, self.liftoffs, self.climbs, None)
        #expected = np.ma.repeat(0, 1280)
        #expected[expected == 0] = np.ma.masked
        #ma_test.assert_masked_array_equal(node.array, expected)

    #def test_derive__afr_v2(self):
        #afr_v2 = A('AFR V2', 120)
        #node = self.node_class()
        #node.derive(self.airspeed, None, None, None, afr_v2, self.liftoffs, self.climbs, None)
        #expected = np.ma.repeat((120, 0, 120, 0), (420, 120, 520, 220))
        #expected[expected == 0] = np.ma.masked
        #ma_test.assert_masked_array_equal(node.array, expected)

    #def test_derive__airbus(self):
        #manufacturer = A(name='Manufacturer', value='Airbus')
        #spd_ctl = M('Speed Control', np.ma.repeat((1, 0), (320, 960)), values_mapping={0: 'Manual', 1: 'Auto'})
        #spd_sel = P('Airspeed Selected', np.ma.repeat((400, 120, 170), (10, 630, 640)))
        #spd_sel.array[:10] = np.ma.masked
        #node = self.node_class()
        #node.derive(self.airspeed, None, spd_sel, spd_ctl, None, self.liftoffs, self.climbs, manufacturer)
        #expected = np.ma.repeat((120, 0), (420, 860))
        #expected[expected == 0] = np.ma.masked
        #ma_test.assert_masked_array_equal(node.array, expected)

    #def test_derive__embraer(self):
        #manufacturer = A(name='Manufacturer', value='Embraer')
        #v2_vac = P('V2-Vac', np.ma.repeat(150, 1280))
        #node = self.node_class()
        #node.derive(self.airspeed, v2_vac, None, None, None, self.liftoffs, self.climbs, manufacturer)
        #expected = np.ma.repeat((150, 0, 150, 0), (420, 120, 520, 220))
        #expected[expected == 0] = np.ma.masked
        #ma_test.assert_masked_array_equal(node.array, expected)


#class TestV2Lookup(unittest.TestCase, NodeTest):

    #class VSX(VelocitySpeed):
        #'''
        #Table for aircraft with undefined V2.
        #'''
        #tables = {}

    #class VSC0(VelocitySpeed):
        #'''
        #Table for aircraft with configuration and fallback tables.
        #'''
        #weight_unit = ut.TONNE
        #tables = {'v2': {
            #'weight': ( 35,  40,  45,  50,  55,  60,  65),
           #'Lever 1': (113, 119, 126, 132, 139, 145, 152),
        #}}
        #fallback = {'v2': {'Lever 2': 140}}

    #class VSC1(VelocitySpeed):
        #'''
        #Table for aircraft with configuration and fallback tables only.
        #'''
        #weight_unit = None
        #fallback = {'v2': {'Lever 1': 135}}

    #class VSF0(VelocitySpeed):
        #'''
        #Table for aircraft with flap and fallback tables.
        #'''
        #weight_unit = ut.TONNE
        #tables = {'v2': {
            #'weight': ( 35,  40,  45,  50,  55,  60,  65),
                #'15': (113, 119, 126, 132, 139, 145, 152),
        #}}
        #fallback = {'v2': {'5': 140}}

    #class VSF1(VelocitySpeed):
        #'''
        #Table for aircraft with flap and fallback tables only.
        #'''
        #weight_unit = None
        #fallback = {'v2': {'17.5': 135}}

    #def setUp(self):
        #self.node_class = V2Lookup
        #self.airspeed = P('Airspeed', np.ma.repeat(200, 1280))
        #self.weight = KPV(name='Gross Weight At Liftoff', items=[
            #KeyPointValue(name='Gross Weight At Liftoff', index=20, value=54192.06),
            #KeyPointValue(name='Gross Weight At Liftoff', index=860, value=44192.06),
        #])
        #self.liftoffs = KTI(name='Liftoff', items=[
            #KeyTimeInstance(name='Liftoff', index=20),
            #KeyTimeInstance(name='Liftoff', index=860),
        #])
        #self.climbs = KTI(name='Climb Start', items=[
            #KeyTimeInstance(name='Climb Start', index=420),
            #KeyTimeInstance(name='Climb Start', index=1060),
        #])

    #@patch('analysis_engine.library.at')
    #def test_can_operate(self, at):
        #nodes = ('Airspeed', 'Liftoff', 'Climb Start',
                 #'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        #keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        #airbus = dict(zip(keys, self.generate_attributes('airbus')))
        #boeing = dict(zip(keys, self.generate_attributes('boeing')))
        #beechcraft = dict(zip(keys, self.generate_attributes('beechcraft')))
        ## Assume that lookup tables are found correctly...
        #at.get_vspeed_map.return_value = self.VSF0
        ## Flap Lever w/ Weight:
        #available = nodes + ('Flap Lever', 'Gross Weight At Liftoff')
        #self.assertTrue(self.node_class.can_operate(available, **boeing))
        ## Flap Lever (Synthetic) w/ Weight:
        #available = nodes + ('Flap Lever (Synthetic)', 'Gross Weight At Liftoff')
        #self.assertTrue(self.node_class.can_operate(available, **airbus))
        ## Flap Lever w/o Weight:
        #available = nodes + ('Flap Lever',)
        #self.assertTrue(self.node_class.can_operate(available, **beechcraft))
        ## Flap Lever (Synthetic) w/o Weight:
        #available = nodes + ('Flap Lever (Synthetic)',)
        #self.assertTrue(self.node_class.can_operate(available, **airbus))
        ## Assume that lookup tables are not found correctly...
        #at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        #available = nodes + ('Flap Lever', 'Gross Weight At Liftoff')
        #for i in range(2):
            #self.assertFalse(self.node_class.can_operate(available, **boeing))

    #@patch('analysis_engine.library.at')
    #def test_derive__flap_with_weight__standard(self, at):
        #mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        #flap_lever = M('Flap Lever', np.ma.repeat(15, 1280), values_mapping=mapping)

        #attributes = self.generate_attributes('boeing')
        #at.get_vspeed_map.return_value = self.VSF0

        #node = self.node_class()
        #node.derive(flap_lever, None, self.airspeed, self.weight,
                    #self.liftoffs, self.climbs, *attributes)

        #attributes = (a.value for a in attributes)
        #at.get_vspeed_map.assert_called_once_with(*attributes)
        #expected = np.ma.repeat((138, 0, 125, 0), (420, 120, 520, 220))
        #expected[expected == 0] = np.ma.masked
        #ma_test.assert_masked_array_equal(node.array, expected)

    #@patch('analysis_engine.library.at')
    #def test_derive__flap_with_weight__fallback(self, at):
        #mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        #flap_lever = M('Flap Lever', np.ma.repeat(5, 1280), values_mapping=mapping)

        #attributes = self.generate_attributes('boeing')
        #at.get_vspeed_map.return_value = self.VSF0

        #node = self.node_class()
        #node.derive(flap_lever, None, self.airspeed, self.weight,
                    #self.liftoffs, self.climbs, *attributes)

        #attributes = (a.value for a in attributes)
        #at.get_vspeed_map.assert_called_once_with(*attributes)
        #expected = np.ma.repeat((140, 0, 140, 0), (420, 120, 520, 220))
        #expected[expected == 0] = np.ma.masked
        #ma_test.assert_masked_array_equal(node.array, expected)

    #@patch('analysis_engine.library.at')
    #def test_derive__flap_without_weight__standard(self, at):
        #mapping = {f: str(f) for f in (0, 17.5, 35)}
        #flap_lever = M('Flap Lever', np.ma.repeat(17.5, 1280), values_mapping=mapping)

        #attributes = self.generate_attributes('beechcraft')
        #at.get_vspeed_map.return_value = self.VSF0

        #node = self.node_class()
        #node.derive(flap_lever, None, self.airspeed, None, self.liftoffs,
                    #self.climbs, *attributes)

        #attributes = (a.value for a in attributes)
        #at.get_vspeed_map.assert_called_once_with(*attributes)
        #expected = np.ma.repeat(0, 1280)
        #expected[expected == 0] = np.ma.masked
        #ma_test.assert_masked_array_equal(node.array, expected)

    #@patch('analysis_engine.library.at')
    #def test_derive__flap_without_weight__fallback(self, at):
        #mapping = {f: str(f) for f in (0, 17.5, 35)}
        #flap_lever = M('Flap Lever', np.ma.repeat(17.5, 1280), values_mapping=mapping)

        #attributes = self.generate_attributes('beechcraft')
        #at.get_vspeed_map.return_value = self.VSF1

        #node = self.node_class()
        #node.derive(flap_lever, None, self.airspeed, None, self.liftoffs,
                    #self.climbs, *attributes)

        #attributes = (a.value for a in attributes)
        #at.get_vspeed_map.assert_called_once_with(*attributes)
        #expected = np.ma.repeat((135, 0, 135, 0), (420, 120, 520, 220))
        #expected[expected == 0] = np.ma.masked
        #ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Reference Speed (Vref)


class TestVref(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Vref
        self.airspeed = P('Airspeed', np.ma.repeat(200, 128))
        self.approaches = buildsections('Approach And Landing', (2, 41), (66, 105))

    def test_can_operate(self):
        # AFR:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'AFR Vref', 'Approach And Landing'),
            afr_vref=A('AFR Vref', 120),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Airspeed', 'AFR Vref', 'Approach And Landing'),
            afr_vref=A('AFR Vref', 70),
        ))
        # Embraer:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'V1-Vref', 'Approach And Landing'),
        ))

    def test_derive__nothing(self):
        node = self.node_class()
        node.derive(self.airspeed, None, None, self.approaches)
        expected = np.ma.repeat(0, 128)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__afr_vref(self):
        afr_vref = A('AFR Vref', 120)
        node = self.node_class()
        node.derive(self.airspeed, None, afr_vref, self.approaches)
        expected = np.ma.repeat((0, 120, 0, 120, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__embraer(self):
        v1_vref = P('V1-Vref', np.ma.repeat(150, 128))
        node = self.node_class()
        node.derive(self.airspeed, v1_vref, None, self.approaches)
        expected = np.ma.repeat((0, 150, 0, 150, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestVrefLookup(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined Vref.
        '''
        tables = {}

    class VSC0(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'vref': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
        'Lever Full': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'vref': {'Lever 3': 140}}

    class VSC1(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables only.
        '''
        weight_unit = None
        fallback = {'vref': {'Lever Full': 135}}

    class VSF0(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'vref': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
                '40': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'vref': {'30': 140}}

    class VSF1(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables only.
        '''
        weight_unit = None
        fallback = {'vref': {'35': 135}}

    def setUp(self):
        self.node_class = VrefLookup
        self.airspeed = P('Airspeed', np.ma.repeat(200, 128))
        self.weight = P('Gross Weight Smoothed', np.ma.repeat((54192.06, 44192.06), 64))
        self.touchdowns = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=7),
            KeyTimeInstance(name='Touchdown', index=71),
        ])
        self.approaches = buildsections('Approach And Landing', (2, 41.1), (66, 105.1))

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        nodes = ('Airspeed', 'Approach And Landing', 'Touchdown',
                 'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        airbus = dict(zip(keys, self.generate_attributes('airbus')))
        boeing = dict(zip(keys, self.generate_attributes('boeing')))
        beechcraft = dict(zip(keys, self.generate_attributes('beechcraft')))
        # Assume that lookup tables are found correctly...
        at.get_vspeed_map.return_value = self.VSF0
        # Flap Lever w/ Weight:
        available = nodes + ('Flap Lever', 'Gross Weight Smoothed')
        self.assertTrue(self.node_class.can_operate(available, **boeing))
        # Flap Lever (Synthetic) w/ Weight:
        available = nodes + ('Flap Lever (Synthetic)', 'Gross Weight Smoothed')
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Flap Lever w/o Weight:
        available = nodes + ('Flap Lever',)
        self.assertTrue(self.node_class.can_operate(available, **beechcraft))
        # Flap Lever (Synthetic) w/o Weight:
        available = nodes + ('Flap Lever (Synthetic)',)
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Assume that lookup tables are not found correctly...
        at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        available = nodes + ('Flap Lever', 'Gross Weight Smoothed')
        for i in range(2):
            self.assertFalse(self.node_class.can_operate(available, **boeing))

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(40, 128), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, self.weight,
                    self.approaches, self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 138, 0, 125, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(30, 128), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, self.weight,
                    self.approaches, self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 140, 0, 140, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(35, 128), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, None, self.approaches,
                    self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0, 128)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(35, 128), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF1

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, None, self.approaches,
                    self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 135, 0, 135, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Lowest Selectable Speed (VLS)


class TestVLSLookup(unittest.TestCase, NodeTest):
    class VSNoCG(VelocitySpeed):
        '''
        Table for aircraft with configuration-only lookup (A319, A320, A321).
        '''
        weight_unit = ut.TONNE
        tables = {
            'vls': {
                      'weight':        ( 44,  47,  51,  55,  59,  60,  67,  71,  76),
                     'Lever 3': {None: (113, 117, 122, 127, 131, 136, 140, 144, 149)},
                  'Lever Full': {None: (111, 113, 116, 121, 125, 129, 133, 137, 142)},
                    },
        }


    class VSCG(VelocitySpeed):
        '''
        Table for aircraft with configuration + cg lookup (A330, A340) and VLS clean.
        '''
        weight_unit = ut.TONNE
        tables = {
            'vls': {
                'weight':      (180, 190, 210, 230, 250, 270, 290, 310, 330, 350, 380),
               'Lever 3': {
                           13: (132, 133, 141, 147, 154, 160, 166, 171, 176, 182, 189),
                           35: (132, 133, 137, 143, 150, 156, 162, 167, 172, 177, 185),
                           },
            'Lever Full': {
                           13: (132, 133, 136, 143, 149, 154, 160, 166, 171, 176, 183),
                           35: (132, 133, 134, 139, 145, 150, 156, 162, 167, 172, 179),
                           },
                    },
            'vls_clean': {
                'altitude': (0  , 10000, 20000, 30000, 40000),
                'weight': {
                    240:    (209,   222,   237,   245, np.nan),
                    220:    (200,   207,   226,   233, np.nan),
                    200:    (190,   192,   214,   220,   243),
                    180:    (181,   181,   200,   206,   221),
                    160:    (170,   171,   182,   194,   203),
                    140:    (159,   159,   163,   181,   187),
                    120:    (148,   148,   148,   164,   171),
                }
            },
        }


    def setUp(self):
        self.node_class = VLSLookup
        self.approach = buildsection('Approach And Landing', 2, 13)
        self.airborne = buildsection('Airborne', 1, 15)
        self.airspeed = P('Airspeed', np.ma.repeat(200, 16))
        self.flap_lever = M('Flap Lever',
                  np.ma.concatenate((np.zeros(4), np.ones(4) * 3, np.ones(4) * 4, np.ones(4) * 2)),
                  values_mapping={0: 'Lever 0', 1: 'Lever 1', 2: 'Lever 2', 3: 'Lever 3', 4: 'Lever Full'})

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        airbus = {'series': Attribute('Series', 'A320-200'),
                   'model': Attribute('Model', 'A320-212'),
             'engine_type': Attribute('Engine Type', 'CFM56-5A3'),
                  'family': Attribute('Family', 'A320'),
           'engine_series': Attribute('Engine Series', 'CFM56-5A')}

        at.get_vspeed_map.return_value = self.VSNoCG

        nodes = ('Airspeed', 'Approach And Landing', 'Gross Weight Smoothed',
                 'Flap Lever', 'Series', 'Model', 'Engine Type', 'Family',
                 'Engine Series', 'Airborne')

        self.assertTrue(self.node_class.can_operate(nodes, **airbus))


    @patch('analysis_engine.library.at')
    def test_derive_no_cg(self, at):
        at.get_vspeed_map.return_value = self.VSNoCG

        model = A('Model', 'A320-212')
        series = A('Series', 'A320-200')
        family = A('Family', 'A320')
        engine_type = A('Engine Type', 'CFM56-5A3')
        engine_series = A('Engine Series', 'CFM56-5A')

        node = self.node_class()

        gw = P('Gross Weight Smoothed', np.ones(16) * 64300)

        node.derive(self.flap_lever, None, self.airspeed, gw, self.approach,
                    model, series, family, engine_type, engine_series, None,
                    self.airborne)

        expected = np.ma.array([None, None, None, None,
                               138, 138, 138, 138,
                               131, 131, 131, 131,
                               None, None, None, None],
                               mask = [1, 1, 1, 1,
                                       0, 0, 0, 0,
                                       0, 0, 0, 0,
                                       1, 1, 1, 1,])

        ma_test.assert_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive_with_cg(self, at):
        at.get_vspeed_map.return_value = self.VSCG

        model = A('Model', 'A340-642X')
        series = A('Series', 'A340-600')
        family = A('Family', 'A340')
        engine_type = A('Engine Type', 'Trent 556A2-61')
        engine_series = A('Engine Series', 'Trent 500')

        node = self.node_class()

        gw = P('Gross Weight Smoothed', np.ones(16) * 264300)
        cg = P('Center Of Gravity', np.ones(16) * 22.5)

        node.derive(self.flap_lever, None, self.airspeed, gw, self.approach,
                    model, series, family, engine_type, engine_series, cg, None,
                    self.airborne)

        expected = np.ma.array([None, None, None, None,
                               157, 157, 157, 157,
                               151, 151, 151, 151,
                               None, None, None, None],
                               mask = [1, 1, 1, 1,
                                       0, 0, 0, 0,
                                       0, 0, 0, 0,
                                       1, 1, 1, 1,])

        ma_test.assert_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive_vls_clean(self, at):
        at.get_vspeed_map.return_value = self.VSCG

        model = A('Model', 'A330-202')
        series = A('Series', 'A330-200')
        family = A('Family', 'A330')
        engine_type = A('Engine Type', 'General Electric CF6-80E1A4')
        engine_series = A('Engine Series', 'General Electric CF6-80E')

        node = self.node_class()

        gw = P('Gross Weight Smoothed', np.ones(16) * 220000)
        alt_std = P('Altitude STD Smoothed', np.ones(16) * 20000)

        node.derive(self.flap_lever, None, self.airspeed, gw, self.approach,
                    model, series, family, engine_type, engine_series, None,
                    alt_std, self.airborne)

        expected = np.ma.array([226, 226, 226, 226,
                               None, None, None, None,
                               None, None, None, None,
                               None, None, None, None],
                               mask = [1, 0, 0, 0,
                                       1, 1, 1, 1,
                                       1, 1, 1, 1,
                                       1, 1, 1, 1,])

        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Approach Speed (VAPP)


class TestVapp(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Vapp
        self.airspeed = P('Airspeed', np.ma.repeat(200, 128))
        self.approaches = buildsections('Approach And Landing', (2, 41), (66, 105))

    def test_can_operate(self):
        # AFR:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'AFR Vapp', 'Approach And Landing'),
            afr_vapp=A('AFR Vapp', 120),
        ))
        self.assertFalse(self.node_class.can_operate(
            ('Airspeed', 'AFR Vapp', 'Approach And Landing'),
            afr_vapp=A('AFR Vapp', 70),
        ))
        # Embraer:
        self.assertTrue(self.node_class.can_operate(
            ('Airspeed', 'VR-Vapp', 'Approach And Landing'),
        ))

    def test_derive__nothing(self):
        node = self.node_class()
        node.derive(self.airspeed, None, None, self.approaches)
        expected = np.ma.repeat(0, 128)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__afr_vapp(self):
        afr_vapp = A('AFR Vapp', 120)
        node = self.node_class()
        node.derive(self.airspeed, None, afr_vapp, self.approaches)
        expected = np.ma.repeat((0, 120, 0, 120, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__embraer(self):
        vr_vapp = P('VR-Vapp', np.ma.repeat(150, 128))
        node = self.node_class()
        node.derive(self.airspeed, vr_vapp, None, self.approaches)
        expected = np.ma.repeat((0, 150, 0, 150, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestVappLookup(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined Vapp.
        '''
        tables = {}

    class VSC0(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'vapp': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
        'Lever Full': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'vapp': {'Lever 3': 140}}

    class VSC1(VelocitySpeed):
        '''
        Table for aircraft with configuration and fallback tables only.
        '''
        weight_unit = None
        fallback = {'vapp': {'Lever Full': 135}}

    class VSF0(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables.
        '''
        weight_unit = ut.TONNE
        tables = {'vapp': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
                '40': (113, 119, 126, 132, 139, 145, 152),
        }}
        fallback = {'vapp': {'30': 140}}

    class VSF1(VelocitySpeed):
        '''
        Table for aircraft with flap and fallback tables only.
        '''
        weight_unit = None
        fallback = {'vapp': {'35': 135}}

    def setUp(self):
        self.node_class = VappLookup
        self.airspeed = P('Airspeed', np.ma.repeat(200, 128))
        self.weight = P('Gross Weight Smoothed', np.ma.repeat((54192.06, 44192.06), 64))
        self.touchdowns = KTI(name='Touchdown', items=[
            KeyTimeInstance(name='Touchdown', index=7),
            KeyTimeInstance(name='Touchdown', index=71),
        ])
        self.approaches = buildsections('Approach And Landing', (2, 41), (66, 105))

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        nodes = ('Airspeed', 'Approach And Landing', 'Touchdown',
                 'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        airbus = dict(zip(keys, self.generate_attributes('airbus')))
        boeing = dict(zip(keys, self.generate_attributes('boeing')))
        beechcraft = dict(zip(keys, self.generate_attributes('beechcraft')))
        # Assume that lookup tables are found correctly...
        at.get_vspeed_map.return_value = self.VSF0
        # Flap Lever w/ Weight:
        available = nodes + ('Flap Lever', 'Gross Weight Smoothed')
        self.assertTrue(self.node_class.can_operate(available, **boeing))
        # Flap Lever (Synthetic) w/ Weight:
        available = nodes + ('Flap Lever (Synthetic)', 'Gross Weight Smoothed')
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Flap Lever w/o Weight:
        available = nodes + ('Flap Lever',)
        self.assertTrue(self.node_class.can_operate(available, **beechcraft))
        # Flap Lever (Synthetic) w/o Weight:
        available = nodes + ('Flap Lever (Synthetic)',)
        self.assertTrue(self.node_class.can_operate(available, **airbus))
        # Assume that lookup tables are not found correctly...
        at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        available = nodes + ('Flap Lever', 'Gross Weight Smoothed')
        for i in range(2):
            self.assertFalse(self.node_class.can_operate(available, **boeing))

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(40, 128), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, self.weight,
                    self.approaches, self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 138, 0, 125, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_with_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)}
        flap_lever = M('Flap Lever', np.ma.repeat(30, 128), values_mapping=mapping)

        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, self.weight,
                    self.approaches, self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 140, 0, 140, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__standard(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(35, 128), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF0

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, None, self.approaches,
                    self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0, 128)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__flap_without_weight__fallback(self, at):
        mapping = {f: str(f) for f in (0, 17.5, 35)}
        flap_lever = M('Flap Lever', np.ma.repeat(35, 128), values_mapping=mapping)

        attributes = self.generate_attributes('beechcraft')
        at.get_vspeed_map.return_value = self.VSF1

        node = self.node_class()
        node.derive(flap_lever, None, self.airspeed, None, self.approaches,
                    self.touchdowns, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat((0, 135, 0, 135, 0), (2, 40, 24, 40, 22))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Maximum Operating Speed (VMO)


class TestVMOLookup(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined VMO.
        '''
        tables = {}

    class VS0(VelocitySpeed):
        '''
        Table for aircraft that don't have a VMO.
        '''
        tables = {'vmo': None}

    class VS1(VelocitySpeed):
        '''
        Table for aircraft that have a fixed value for VMO.
        '''
        tables = {'vmo': 350}

    class VS2(VelocitySpeed):
        '''
        Table for aircraft that have linear interpolated values for VMO.
        '''
        tables = {'vmo': {
            'altitude': (  0, 15000, 25000, 40000),
               'speed': (350,   350,   300,   300),
        }}

    class VS3(VelocitySpeed):
        '''
        Table for aircraft that have stepped values for VMO.
        '''
        tables = {'vmo': {
            'altitude': (  0, 20000, 20000, 40000),
               'speed': (350,   350,   300,   300),
        }}

    def setUp(self):
        self.node_class = VMOLookup
        self.altitude = P('Altitude', np.ma.arange(0, 50000, 1000))

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        nodes = ('Altitude STD Smoothed', 'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        boeing = dict(zip(keys, self.generate_attributes('boeing')))
        # Assume that lookup tables are found correctly...
        at.get_vspeed_map.return_value = self.VS0
        self.assertTrue(self.node_class.can_operate(nodes, **boeing))
        # Assume that lookup tables are not found correctly...
        at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        for i in range(2):
            self.assertFalse(self.node_class.can_operate(nodes, **boeing))

    @patch('analysis_engine.library.at')
    def test_derive__none(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS0

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0, 50)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__fixed(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS1

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(350, 50)
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__linear(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS2

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.concatenate((
            np.repeat(350, 16),
            np.arange(345, 295, -5),
            np.repeat(300, 15),
            np.repeat(000, 9),
        ))
        # numpy 1.10 results in following
        # expected[0] = np.ma.masked
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__stepped(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS3

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.concatenate((
            np.repeat(350, 20),
            np.repeat(300, 21),
            np.repeat(000, 9),
        ))
        # numpy 1.10 results in following
        # np.ma.repeat(350, 21),
        # np.ma.repeat(300, 20),
        # expected[0] = np.ma.masked
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Maximum Operating Mach (MMO)


class TestMMOLookup(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined MMO.
        '''
        tables = {}

    class VS0(VelocitySpeed):
        '''
        Table for aircraft that don't have a MMO.
        '''
        tables = {'mmo': None}

    class VS1(VelocitySpeed):
        '''
        Table for aircraft that have a fixed value for MMO.
        '''
        tables = {'mmo': 0.850}

    class VS2(VelocitySpeed):
        '''
        Table for aircraft that have linear interpolated values for MMO.
        '''
        tables = {'mmo': {
            'altitude': (    0, 15000, 25000, 40000),
               'speed': (0.850, 0.850, 0.800, 0.800),
        }}

    class VS3(VelocitySpeed):
        '''
        Table for aircraft that have stepped values for MMO.
        '''
        tables = {'mmo': {
            'altitude': (    0, 20000, 20000, 40000),
               'speed': (0.850, 0.850, 0.800, 0.800),
        }}


    def setUp(self):
        self.node_class = MMOLookup
        self.altitude = P('Altitude', np.ma.arange(0, 50000, 1000))

    @patch('analysis_engine.library.at')
    def test_can_operate(self, at):
        nodes = ('Altitude STD Smoothed', 'Model', 'Series', 'Family', 'Engine Series', 'Engine Type')
        keys = ('model', 'series', 'family', 'engine_type', 'engine_series')
        boeing = dict(zip(keys, self.generate_attributes('boeing')))
        # Assume that lookup tables are found correctly...
        at.get_vspeed_map.return_value = self.VS0
        self.assertTrue(self.node_class.can_operate(nodes, **boeing))
        # Assume that lookup tables are not found correctly...
        at.get_vspeed_map.side_effect = (KeyError, self.VSX)
        for i in range(2):
            self.assertFalse(self.node_class.can_operate(nodes, **boeing))

    @patch('analysis_engine.library.at')
    def test_derive__none(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS0

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0, 50)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__fixed(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS1

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.repeat(0.850, 50)
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.library.at')
    def test_derive__linear(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS2

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.concatenate((
            np.repeat(0.850, 16),
            np.arange(0.845, 0.795, -0.005),
            np.repeat(0.800, 15),
            np.repeat(0.000, 9),
        ))
        # numpy 1.10 results in following
        # expected[0] = np.ma.masked
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_almost_equal(node.array, expected, decimal=3)

    @patch('analysis_engine.library.at')
    def test_derive__stepped(self, at):
        attributes = self.generate_attributes('boeing')
        at.get_vspeed_map.return_value = self.VS3

        node = self.node_class()
        node.derive(self.altitude, *attributes)

        attributes = (a.value for a in attributes)
        at.get_vspeed_map.assert_called_once_with(*attributes)
        expected = np.ma.concatenate((
            np.repeat(0.850, 20),
            np.repeat(0.800, 21),
            np.repeat(0.000, 9),
        ))
        # numpy 1.10 results in following
        # np.ma.repeat(350, 21),
        # np.ma.repeat(300, 20),
        # expected[0] = np.ma.masked
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Minimum Airspeed


class TestMinimumAirspeed(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MinimumAirspeed
        self.operational_combinations = [
            ('Airborne', 'Airspeed', 'VLS'),
            ('Airborne', 'Airspeed', 'VLS Lookup'),
            ('Airborne', 'Airspeed', 'Min Operating Speed'),
            ('Airborne', 'Airspeed', 'FC Min Operating Speed'),
            ('Airborne', 'Airspeed', 'Minimum Clean Lookup', 'Flap Lever'),
            ('Airborne', 'Airspeed', 'Minimum Clean Lookup', 'Flap Lever (Synthetic)'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(250, 100))
        self.airborne = buildsection('Airborne', 10, 90)

    def test_derive__invalid(self):
        node = self.node_class()
        node.derive(self.airspeed, None, None, None, None, None, None, None, self.airborne)
        expected = np.ma.repeat(0, 100)
        expected.mask = True
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__vls(self):
        vls = P('VLS', np.ma.repeat(180, 100))
        node = self.node_class()
        node.derive(self.airspeed, None, None, vls, None, None, None, None, self.airborne)
        expected = np.ma.array(vls.array)
        expected.mask = np.repeat((1, 0, 0, 0, 0, 0, 0, 0, 0, 1), 10)
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__vls_lookup(self):
        vls_lookup = P('VLS Lookup', np.ma.repeat(180, 100))
        node = self.node_class()
        node.derive(self.airspeed, None, None, None, vls_lookup, None, None, None, self.airborne)
        expected = np.ma.array(vls_lookup.array)
        expected.mask = np.repeat((1, 0, 0, 0, 0, 0, 0, 0, 0, 1), 10)
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__mos(self):
        mos = P('Min Operating Speed', np.ma.repeat(190, 100))
        node = self.node_class()
        node.derive(self.airspeed, None, mos, None, None, None, None, None, self.airborne)
        expected = np.ma.array(mos.array)
        expected.mask = np.repeat((1, 0, 0, 0, 0, 0, 0, 0, 0, 1), 10)
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__mos_fc(self):
        mos_fc = P('FC Min Operating Speed', np.ma.repeat(200, 100))
        node = self.node_class()
        node.derive(self.airspeed, mos_fc, None, None, None, None, None, None, self.airborne)
        expected = np.ma.array(mos_fc.array)
        expected.mask = np.repeat((1, 0, 0, 0, 0, 0, 0, 0, 0, 1), 10)
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__min_clean(self):
        min_clean= P('Minimum Clean Lookup', np.ma.repeat(210, 100))
        array = np.ma.repeat((20, 10, 10, 0, 0, 0, 0, 10, 10, 20), 10)
        flap = M('Flap Lever', array, values_mapping={0: '0', 10: '10', 20: '20'})
        node = self.node_class()
        node.derive(self.airspeed, None, None, None, None, min_clean, flap, None, self.airborne)
        expected = np.ma.array(min_clean.array)
        expected.mask = np.repeat((1, 1, 1, 0, 0, 0, 0, 1, 1, 1), 10)
        ma_test.assert_masked_array_equal(node.array, expected)


class TestMinimumCleanLookup(unittest.TestCase):

    class VS0(VelocitySpeed):
        '''
        Table for aircraft with flap and weight.
        '''
        weight_unit = ut.TONNE
        tables = {
            'vref': {
                'weight': ( 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190),
                    '20': (122, 128, 135, 141, 146, 151, 157, 162, 168, 173, 179),
                    '25': (117, 123, 129, 135, 141, 146, 151, 156, 161, 166, 170),
                    '30': (113, 119, 125, 131, 137, 142, 148, 156, 164, 171, 179),
            },
            'vmo': 360,
            'mmo': 0.860,
        }


    def setUp(self):
        self.node_class = MinimumCleanLookup
        self.alt = P('Altitude STD Smoothed',
                np.ma.repeat((15000, 20000, 26000, 27000), 10))
        self.airborne = buildsection('Airborne', 5, 35)
        self.air_spd = P('Airspeed', np.ma.repeat((170, 180, 190, 230), 10))

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], A('Family', 'B757')))
        self.assertTrue(self.node_class.can_operate(self.node_class.get_dependency_names(), A('Family', 'B757')))
        self.assertTrue(self.node_class.can_operate(self.node_class.get_dependency_names(), A('Family', 'B767')))
        self.assertFalse(self.node_class.can_operate(self.node_class.get_dependency_names(), A('Family', 'A320')))

    @patch('analysis_engine.derived_parameters.at')
    @patch('analysis_engine.library.at')
    def test_767(self, at0, at1):
        at0.get_vspeed_map.return_value = self.VS0
        at1.get_fms_map.return_value = {}
        gw = P('Gross Weight Smoothed', np.ma.repeat((130000, 150000, 180000, 190000), 10))
        model = A('Model', 'B767-3JHF(ER)')
        series = A('Series', 'B767-300')
        family = A('Family', 'B767')
        eng_type = A('Engine Type', 'CF6-80C2B7F')
        eng_series = A('Engine Series', 'CF6-80C2')

        node = self.node_class()
        node.derive(self.air_spd, gw, self.airborne, model, series, family,
                    eng_type, eng_series, self.alt)
        expected = np.ma.repeat((217, 228, 271, 279), 10)
        expected.mask = np.array([1]*5 + [0]*30 + [1]*5)
        ma_test.assert_masked_array_equal(node.array, expected)

    @patch('analysis_engine.derived_parameters.at')
    @patch('analysis_engine.library.at')
    def test_767_cruise(self, at0, at1):
        at0.get_vspeed_map.return_value = self.VS0
        at1.get_fms_map.return_value = {}
        gw = P('Gross Weight Smoothed', np.ma.repeat((130000, 150000, 180000, 190000), 10))
        crz = buildsection('Cruise', 10,35)
        model = A('Model', 'B767-3JHF(ER)')
        series = A('Series', 'B767-300')
        family = A('Family', 'B767')
        eng_type = A('Engine Type', 'CF6-80C2B7F')
        eng_series = A('Engine Series', 'CF6-80C2')

        node = self.node_class()
        node.derive(self.air_spd, gw, self.airborne, model, series, family,
                    eng_type, eng_series, self.alt, crz)
        expected = np.ma.repeat((217, 228, 271, 279), 10)
        expected.mask = np.array([1]*5 + [0]*30 + [1]*5)
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Flap Manoeuvre Speed


class TestFlapManoeuvreSpeed(unittest.TestCase, NodeTest):

    class VSX(VelocitySpeed):
        '''
        Table for aircraft with undefined Vref.
        '''
        tables = {}

    class VS0(VelocitySpeed):
        '''
        Table for aircraft with flap and weight.
        '''
        weight_unit = ut.TONNE
        tables = {'vref': {
            'weight': ( 35,  40,  45,  50,  55,  60,  65),
                '30': (111, 119, 127, 134, 141, 147, 154),
                '40': (107, 115, 123, 131, 138, 146, 153),
        },
        'vmo': 340,
        'mmo': 0.82,
    }

    def setUp(self):
        self.node_class = FlapManoeuvreSpeed
        self.airspeed = P('Airspeed', np.ma.repeat(200, 90))
        self.weight = P('Gross Weight Smoothed', np.ma.repeat((50000, 60000), 45))
        self.altitude = P('Altitude STD Smoothed', np.ma.arange(20010, 19920, -1))
        self.descents = buildsections('Descent To Flare', (2, 42), (47, 87))
        self.flap_lever = M(
            name='Flap Lever',
            array=np.ma.repeat((0, 1, 2, 5, 10, 15, 25, 30, 40), 10),
            values_mapping={f: str(f) for f in (0, 1, 2, 5, 10, 15, 25, 30, 40)},
        )
        self.flap_lever.array.mask = np.repeat(False, 90)  # expand mask.

    @patch('analysis_engine.derived_parameters.at')
    @patch('analysis_engine.library.at')
    def test_can_operate(self, at0, at1):
        nodes = ('Airspeed', 'Altitude STD Smoothed', 'Descent To Flare',
                 'Gross Weight Smoothed', 'Model', 'Series', 'Family',
                 'Engine Series', 'Engine Type')
        attrs = dict(zip(
            ('model', 'series', 'family', 'engine_type', 'engine_series'),
            self.generate_attributes('boeing'),
        ))
        attrs.update(manufacturer=A('Manufacturer', 'Boeing'))
        # Assume that lookup tables are found correctly...
        at0.get_vspeed_map.return_value = self.VS0
        at1.get_fms_map.return_value = {}
        # Flap Lever:
        available = nodes + ('Flap Lever',)
        self.assertTrue(self.node_class.can_operate(available, **attrs))
        # Flap Lever (Synthetic):
        available = nodes + ('Flap Lever (Synthetic)',)
        self.assertTrue(self.node_class.can_operate(available, **attrs))
        # Recorded Vref:
        available = nodes + ('Flap Lever', 'Vref (25)', 'Vref (30)')
        self.assertTrue(self.node_class.can_operate(available, **attrs))
        # Assume that lookup tables are not found correctly...
        # Please be careful if changing the below side effect iterables!
        at0.get_vspeed_map.side_effect = (KeyError, self.VSX)
        at1.get_fms_map.side_effect = ({}, {}, KeyError)
        available = nodes + ('Flap Lever',)
        for i in range(3):
            self.assertFalse(self.node_class.can_operate(available, **attrs))

    @patch('analysis_engine.derived_parameters.at')
    @patch('analysis_engine.library.at')
    def test_derive__vref_plus_offset(self, at0, at1):
        attributes = self.generate_attributes('boeing')
        at0.get_vspeed_map.return_value = self.VS0
        at1.get_fms_map.return_value = {
            '0':  ('40', 70),
            '1':  ('40', 50),
            '2':  None,
            '5':  ('40', 30),
            '10': ('40', 30),
            '15': ('40', 20),
            '25': ('40', 10),
            '30': ('30', 0),
            '40': ('40', 0),
        }

        node = self.node_class()
        node.derive(self.airspeed, self.flap_lever, None, self.weight,
                    None, None, self.altitude, *attributes)

        attributes = [a.value for a in attributes]
        at0.get_vspeed_map.assert_called_once_with(*attributes)
        at1.get_fms_map.assert_called_once_with(*attributes[:3])
        expected = np.ma.repeat(
            (201, 181, 0, 161, 161, 176, 166, 156, 147, 146),
            (10, 10, 10, 10, 5, 5, 10, 10, 10, 10)
        )
        for s in slice(0, 10), slice(20, 30):
            expected[s] = np.ma.masked
        ma_test.assert_masked_array_almost_equal(node.array, expected, decimal=0)

    @patch('analysis_engine.derived_parameters.at')
    @patch('analysis_engine.library.at')
    def test_derive__fixed_speeds_in_weight_bands(self, at0, at1):
        attributes = self.generate_attributes('boeing')
        at0.get_vspeed_map.return_value = self.VS0
        at1.get_fms_map.return_value = {
            '0':  ((53070, 210), (62823, 220), (99999, 230)),
            '1':  ((53070, 190), (62823, 200), (99999, 210)),
            '2':  None,
            '5':  ((53070, 170), (62823, 180), (99999, 190)),
            '10': ((53070, 160), (62823, 170), (99999, 180)),
            '15': ((53070, 150), (62823, 160), (99999, 170)),
            '25': ((53070, 140), (62823, 150), (99999, 160)),
            '30': ('30', 0),
            '40': ('40', 0),
        }

        node = self.node_class()
        node.derive(self.airspeed, self.flap_lever, None, self.weight,
                    None, None, self.altitude, *attributes)

        attributes = [a.value for a in attributes]
        at0.get_vspeed_map.assert_called_once_with(*attributes)
        at1.get_fms_map.assert_called_once_with(*attributes[:3])
        expected = np.ma.repeat(
            (210, 190, 0, 170, 160, 170, 160, 150, 147, 146),
            (10, 10, 10, 10, 5, 5, 10, 10, 10, 10)
        )
        for s in slice(0, 10), slice(20, 30):
            expected[s] = np.ma.masked
        ma_test.assert_masked_array_almost_equal(node.array, expected, decimal=0)


    @patch('analysis_engine.derived_parameters.at')
    @patch('analysis_engine.library.at')
    def test_derive__use_recorded_vref(self, at0, at1):
        attributes = self.generate_attributes('boeing')
        at0.get_vspeed_map.return_value = self.VS0
        at1.get_fms_map.return_value = {
            '0':  ('30', 80),
            '1':  ('30', 60),
            '5':  ('30', 40),
            '15': ('30', 20),
            '20': ('30', 20),
            '25': ('25', 0),
            '30': ('30', 0),
        }

        # We need to change some of the test data:
        self.airspeed = P('Airspeed', np.ma.repeat(200, 70))
        self.weight = P('Gross Weight Smoothed', np.ma.repeat((50000, 60000), 35))
        self.descents = buildsections('Descent To Flare', (2, 32), (37, 67))
        self.flap_lever = M(
            name='Flap Lever',
            array=np.ma.repeat((0, 1, 5, 15, 20, 25, 30), 10),
            values_mapping={f: str(f) for f in (0, 1, 5, 15, 20, 25, 30)},
        )
        self.flap_lever.array.mask = np.repeat(False, 70)  # expand mask.
        self.vref_25 = P('Vref (25)', np.ma.repeat(155, 70))
        self.vref_30 = P('Vref (30)', np.ma.repeat(150, 70))

        node = self.node_class()
        node.derive(self.airspeed, self.flap_lever, None, self.weight,
                    self.vref_25, self.vref_30, self.altitude, *attributes)

        attributes = [a.value for a in attributes]
        at0.get_vspeed_map.assert_called_once_with(*attributes)
        at1.get_fms_map.assert_called_once_with(*attributes[:3])
        expected = np.ma.repeat((230, 210, 190, 170, 170, 155, 150), 10)
        for s in (slice(0, 10),):
            expected[s] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


##############################################################################
# Relative Airspeeds

########################################
# Airspeed Minus Airspeed Selected (FMS)


class TestAirspeedMinusAirspeedSelectedFMS(unittest.TestCase):
    def setUp(self):
        self.node_class = AirspeedMinusAirspeedSelectedFMS
        self.airspeed = P('Airspeed', np.ma.repeat(102, 2000))
        self.air_sel = P('Airspeed Selected (FMS)',
                         np.ma.repeat((90, 120), 1000))
        self.approaches = buildsection('Approach And Landing', 500, 999.5)
        self.assertEqual(self.node_class.name,
                         'Airspeed Minus Airspeed Selected (FMS)',)

    def test_can_operate(self):
        o = self.node_class.get_operational_combinations()
        self.assertEqual([('Airspeed',
                           'Airspeed Selected (FMS)',
                           'Approach And Landing',)], o)

    def test_derive(self):
        node = self.node_class()
        node.derive(self.airspeed, self.air_sel, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

class TestAirspeedMinusAirspeedSelectedFMSFor3Sec(unittest.TestCase):
    def setUp(self):
        self.node_class = AirspeedMinusAirspeedSelectedFMSFor3Sec
        self.airspeed = P(
            'Airspeed Minus Airspeed Selected (FMS)',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )
        self.assertEqual(self.node_class.name,
                         'Airspeed Minus Airspeed Selected (FMS) For 3 Sec',)

    def test_can_operate(self):
        o = self.node_class.get_operational_combinations()
        self.assertEqual([('Airspeed Minus Airspeed Selected (FMS)',)], o)

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Minus V2


class TestAirspeedMinusV2(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2
        self.operational_combinations = [
            ('Airspeed', 'V2 At Liftoff', 'Liftoff', 'Climb Start', 'Climb Acceleration Start', 'Grounded'),
            ('Airspeed', 'Airspeed Selected At Takeoff Acceleration Start', 'Liftoff', 'Climb Start', 'Climb Acceleration Start', 'Grounded'),
            ('Airspeed', 'V2 At Liftoff', 'Airspeed Selected At Takeoff Acceleration Start', 'Liftoff', 'Climb Start', 'Climb Acceleration Start', 'Grounded'),
            ('Airspeed', 'Airspeed Selected At Takeoff Acceleration Start', 'V2 Lookup At Liftoff', 'Liftoff', 'Climb Start', 'Climb Acceleration Start', 'Grounded'),
            ('Airspeed', 'V2 At Liftoff', 'Airspeed Selected At Takeoff Acceleration Start', 'V2 Lookup At Liftoff', 'Liftoff', 'Climb Start', 'Climb Acceleration Start', 'Grounded'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 2000))
        self.v2_record = KPV([KeyPointValue(index=500, value=90, name='V2 At Liftoff')])
        self.airspeed_selected = KPV([KeyPointValue(index=500, value=110, name='Airspeed Selected At Takeoff Acceleration Start')])
        self.v2_lookup = KPV([KeyPointValue(index=500, value=95, name='V2 Lookup At Liftoff')])
        self.liftoffs = KTI(name='Liftoff', items=[
            KeyTimeInstance(name='Liftoff', index=500),
        ])
        self.climb_accels = KTI(name='Climb Acceleration Start', items=[
            KeyTimeInstance(name='Climb Acceleration Start', index = 999.5)
        ])
        self.climbs = KTI(name='Climb Start', items=[
            KeyTimeInstance(name='Climb Start', index=899.5),
        ])
        self.grounded = buildsection('Grounded', 0, 500)

    def test_derive__recorded_only(self):
        node = self.node_class()
        node.derive(self.airspeed, self.v2_record, None, None, self.liftoffs, self.climbs, self.climb_accels, self.grounded)
        expected = np.ma.repeat((0, 12, 0), (180, 820, 1000))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__lookup_only(self):
        node = self.node_class()
        node.derive(self.airspeed, None, None, self.v2_lookup, self.liftoffs, self.climbs, self.climb_accels, self.grounded)
        expected = np.ma.repeat((0, 7, 0), (180, 820, 1000))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__prefer_recorded(self):
        node = self.node_class()
        node.derive(self.airspeed, self.v2_record, self.airspeed_selected, self.v2_lookup, self.liftoffs, self.climbs, self.climb_accels, self.grounded)
        expected = np.ma.repeat((0, 12, 0), (180, 820, 1000))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__prefer_selected(self):
        node = self.node_class()
        node.derive(self.airspeed, None, self.airspeed_selected, self.v2_lookup, self.liftoffs, self.climbs, self.climb_accels, self.grounded)
        expected = np.ma.repeat((0, -8, 0), (180, 820, 1000))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__recorded_masked(self):
        node = self.node_class()
        node.derive(self.airspeed, KPV(items=[]), None, self.v2_lookup, self.liftoffs, self.climbs, self.climb_accels, self.grounded)
        expected = np.ma.repeat((0, 7, 0), (180, 820, 1000))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__recorded_and_lookup_masked(self):
        node = self.node_class()
        node.derive(self.airspeed, KPV(items=[]), None, KPV(items=[]), self.liftoffs, self.climbs, self.climb_accels, self.grounded)
        expected = np.ma.repeat(0, 2000)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__masked_within_phase(self):
        node = self.node_class()
        node.derive(self.airspeed, KPV(items=[]), None, KPV(items=[]), self.liftoffs, self.climbs, self.climb_accels, self.grounded)
        expected = np_ma_masked_zeros_like(self.airspeed.array)
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__early_climb_accel_start(self):
        climb_accels = KTI(name='Climb Acceleration Start', items=[
            KeyTimeInstance(name='Climb Acceleration Start', index = 700.0)
        ])
        node = self.node_class()
        node.derive(self.airspeed, self.v2_record, None, None, self.liftoffs, self.climbs, climb_accels, self.grounded)
        expected = np.ma.repeat((0, 12, 0), (180, 720, 1100))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__empty_climb_accel_start(self):
        climb_accels = KTI(name='Climb Acceleration Start', items=[])
        node = self.node_class()
        node.derive(self.airspeed, self.v2_record, None, None, self.liftoffs, self.climbs, climb_accels, self.grounded)
        expected = np.ma.repeat((0, 12, 0), (180, 720, 1100))
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestAirspeedMinusV2For3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusV2For3Sec
        self.operational_combinations = [
            ('Airspeed Minus V2',),
        ]
        self.airspeed = P(
            name='Airspeed Minus V2',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Minus Vref


class TestAirspeedMinusVref(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVref
        self.operational_combinations = [
            ('Airspeed', 'Vref', 'Approach And Landing'),
            ('Airspeed', 'Vref Lookup', 'Approach And Landing'),
            ('Airspeed', 'Vref', 'Vref Lookup', 'Approach And Landing'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 2000))
        self.vref_record = P('Vref', np.ma.repeat((90, 120), 1000))
        self.vref_lookup = P('Vref Lookup', np.ma.repeat((95, 125), 1000))
        self.approaches = buildsection('Approach And Landing', 500, 1000)

    def test_derive__record_only(self):
        node = self.node_class()
        node.derive(self.airspeed, self.vref_record, None, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__lookup_only(self):
        node = self.node_class()
        node.derive(self.airspeed, None, self.vref_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__prefer_record(self):
        node = self.node_class()
        node.derive(self.airspeed, self.vref_record, self.vref_lookup, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__record_masked(self):
        self.vref_record.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.vref_record, self.vref_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__both_masked(self):
        self.vref_record.array.mask = True
        self.vref_lookup.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.vref_record, self.vref_lookup, self.approaches)
        expected = np.ma.repeat(0, 2000)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__masked_within_phase(self):
        self.vref_record.array[:-1] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, self.vref_record, self.vref_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__superframe(self):
        self.vref_record.array = np.tile(np.ma.repeat((100, 100, 100, 120), 4), 2)
        self.vref_record.frequency = 1 / 64.0
        node = self.node_class()
        node.get_derived([self.airspeed, self.vref_record, None, self.approaches])
        expected = np.ma.repeat((0, 2, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestAirspeedMinusVrefFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVrefFor3Sec
        self.operational_combinations = [
            ('Airspeed Minus Vref',),
        ]
        self.airspeed = P(
            name='Airspeed Minus Vref',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.concatenate((np.ones(11) * 100.0, np.ones(1) * 105.0, np.ones(16) * 110.0, np.ones(12) * 100.0))
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Minus VLS

class TestAirspeedMinusVLS(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLS
        self.operational_combinations = [
            ('Airspeed', 'VLS', 'Approach And Landing',),
            ('Airspeed', 'VLS Lookup', 'Approach And Landing',)
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 2000))
        self.vls_record = P('VLS', np.ma.repeat((90, 120), 1000))
        self.vls_lookup = P('VLS Lookup', np.ma.repeat((90, 120), 1000))
        self.approaches = buildsection('Approach And Landing', 500, 1000)

    def test_derive(self):
        node = self.node_class()
        node.derive(self.airspeed, self.vls_record, None, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_lookup(self):
        node = self.node_class()
        node.derive(self.airspeed, None, self.vls_lookup, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


class TestAirspeedMinusVLSFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVLSFor3Sec
        self.operational_combinations = [
            ('Airspeed Minus VLS',),
        ]
        self.airspeed = P(
            name='Airspeed Minus VLS',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.concatenate((np.ones(11) * 100.0, np.ones(1) * 105.0, np.ones(16) * 110.0, np.ones(12) * 100.0))
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Minus Vapp


class TestAirspeedMinusVapp(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVapp
        self.operational_combinations = [
            ('Airspeed', 'Vapp', 'Approach And Landing'),
            ('Airspeed', 'Vapp Lookup', 'Approach And Landing'),
            ('Airspeed', 'Vapp', 'Vapp Lookup', 'Approach And Landing'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 2000))
        self.vapp_record = P('Vapp', np.ma.repeat((90, 120), 1000))
        self.vapp_lookup = P('Vapp Lookup', np.ma.repeat((95, 125), 1000))
        self.approaches = buildsection('Approach And Landing', 500, 1000)

    def test_derive__record_only(self):
        node = self.node_class()
        node.derive(self.airspeed, self.vapp_record, None, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__lookup_only(self):
        node = self.node_class()
        node.derive(self.airspeed, None, self.vapp_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__prefer_record(self):
        node = self.node_class()
        node.derive(self.airspeed, self.vapp_record, self.vapp_lookup, self.approaches)
        expected = np.ma.repeat((0, 12, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__record_masked(self):
        self.vapp_record.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.vapp_record, self.vapp_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__both_masked(self):
        self.vapp_record.array.mask = True
        self.vapp_lookup.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.vapp_record, self.vapp_lookup, self.approaches)
        expected = np.ma.repeat(0, 2000)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__masked_within_phase(self):
        self.vapp_record.array[:-1] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, self.vapp_record, self.vapp_lookup, self.approaches)
        expected = np.ma.repeat((0, 7, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    """
    Test not required. Vapp is always sampled at 0.25 Hz or 1Hz. Lower sample rates would be meaningless.
    def test_derive__superframe(self):
        self.vapp_record.array = np.tile(np.ma.repeat((100, 100, 100, 120), 4), 2)
        self.vapp_record.frequency = 1 / 64.0
        node = self.node_class()
        node.get_derived([self.airspeed, self.vapp_record, None, self.approaches])
        expected = np.ma.repeat((0, 2, 0, 0), 500)
        expected[expected == 0] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)
    """

class TestAirspeedMinusVappFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusVappFor3Sec
        self.operational_combinations = [
            ('Airspeed Minus Vapp',),
        ]
        self.airspeed = P(
            name='Airspeed Minus Vapp',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Minus Minimum Airspeed


class TestAirspeedMinusMinimumAirspeed(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusMinimumAirspeed
        self.operational_combinations = [
            ('Airspeed', 'Minimum Airspeed'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 6))
        self.minimum_airspeed = P('Minimum Airspeed', np.ma.arange(80, 92, 2))

    def test_derive__basic(self):
        self.minimum_airspeed.array[3] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, self.minimum_airspeed)
        expected = np.ma.arange(22, 10, -2)
        expected[3] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__masked(self):
        self.minimum_airspeed.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.minimum_airspeed)
        expected = np.ma.arange(22, 10, -2)
        expected.mask = True
        ma_test.assert_masked_array_equal(node.array, expected)


class TestAirspeedMinusMinimumAirspeedFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusMinimumAirspeedFor3Sec
        self.operational_combinations = [
            ('Airspeed Minus Minimum Airspeed',),
        ]
        self.airspeed = P(
            name='Airspeed Minus Minimum Airspeed',
            array=np.ma.repeat((100, 110, 120, 100), (6, 7, 1, 6)),
            frequency=2,
        )

    def test_derive_basic(self):
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 110, 100), (6, 8, 6))
        expected[-6:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_align(self):
        self.airspeed.frequency = 1
        node = self.node_class()
        node.get_derived([self.airspeed])
        expected = np.ma.repeat((100, 105, 110, 100), (11, 1, 16, 12))
        expected[-7:] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Minus Flap Manoeuvre Speed


class TestAirspeedMinusFlapManoeuvreSpeed(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedMinusFlapManoeuvreSpeed
        self.operational_combinations = [
            ('Airspeed', 'Flap Manoeuvre Speed'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 6))
        self.flap_mvr_spd = P('Flap Manoeuvre Speed', np.ma.arange(80, 92, 2))

    def test_derive__basic(self):
        self.flap_mvr_spd.array[3] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, self.flap_mvr_spd)
        expected = np.ma.arange(22, 10, -2)
        expected[3] = np.ma.masked
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive__masked(self):
        self.flap_mvr_spd.array.mask = True
        node = self.node_class()
        node.derive(self.airspeed, self.flap_mvr_spd)
        expected = np.ma.arange(22, 10, -2)
        expected.mask = True
        ma_test.assert_masked_array_equal(node.array, expected)


class TestAirspeedMinusFlapManoeuvreSpeedFor3Sec(unittest.TestCase, NodeTest):
    # This test was missing, so copied from above and amended, but for some
    # reason fails in second_window with the error "Invalid Seconds for
    # frequency" which I do not understand. DJ 17/10/14.
    def setUp(self):
        self.node_class = AirspeedMinusFlapManoeuvreSpeedFor3Sec
        self.operational_combinations = [
            ('Airspeed', 'Flap Manoeuvre Speed'),
        ]
        self.airspeed = P('Airspeed', np.ma.repeat(102, 12))
        self.flap_mvr_spd = P('Flap Manoeuvre Speed',
                              np.ma.concatenate((np.arange(80,87), np.arange(85,80,-1))))

    def test_derive__basic(self):
        self.flap_mvr_spd.array[3] = np.ma.masked
        node = self.node_class()
        node.derive(self.airspeed, self.flap_mvr_spd)
        expected = np.ma.array(data=np.concatenate((np.zeros(4), np.ones(4) * 18, np.zeros(4))),
                               mask=np.concatenate((np.ones(4), np.zeros(4), np.ones(4))))
        ma_test.assert_masked_array_equal(node.array, expected)


########################################
# Airspeed Relative


class TestAirspeedRelative(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelative
        self.operational_combinations = [
            ('Airspeed Minus V2',),
            ('Airspeed Minus Vapp',),
            ('Airspeed Minus Vref',),
            ('Airspeed Minus Vapp', 'Airspeed Minus Vref'),
        ]
        self.v2 = P('Airspeed Minus V2', np.ma.array(data=np.arange(20,120),
                                                     mask=np.concatenate((np.ones(5), np.zeros(10), np.ones(85)))))
        self.vapp = P('Airspeed Minus Vapp', np.ma.array(data=np.arange(30,130),
                                                         mask=np.concatenate((np.ones(80), np.zeros(10), np.ones(10)))))
        self.vref = P('Airspeed Minus Vref', np.ma.array(data=np.arange(40,140),
                                                      mask=np.concatenate((np.ones(80), np.zeros(10), np.ones(10)))))

    def test_derive__vapp(self):
        node = self.node_class()
        node.derive(None, self.vapp, None)
        ma_test.assert_masked_array_equal(node.array, self.vapp.array)

    def test_derive__vref(self):
        node = self.node_class()
        node.derive(None, None, self.vref)
        ma_test.assert_masked_array_equal(node.array, self.vref.array)

    def test_derive__both_approach(self):
        node = self.node_class()
        node.derive(None, self.vapp, self.vref)
        ma_test.assert_masked_array_equal(node.array, self.vapp.array)

    def test_derive_all_three(self):
        node = self.node_class()
        node.derive(self.v2, self.vapp, self.vref)
        expected = np.ma.array(data=np.concatenate((np.arange(20,70), np.arange(80,130))),
                               mask=np.concatenate((np.ones(5), np.zeros(10), np.ones(65), np.zeros(10), np.ones(10))))
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_v2(self):
        node = self.node_class()
        node.derive(self.v2, None, None)
        ma_test.assert_masked_array_equal(node.array, self.v2.array)


class TestAirspeedRelativeFor3Sec(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedRelativeFor3Sec
        self.operational_combinations = [
            ('Airspeed Minus V2 For 3 Sec',),
            ('Airspeed Minus Vapp For 3 Sec',),
            ('Airspeed Minus Vref For 3 Sec',),
            ('Airspeed Minus Vapp For 3 Sec', 'Airspeed Minus Vref For 3 Sec'),
        ]
        self.v2 = P('Airspeed Minus V2 For 3 Sec', np.ma.array(data=np.arange(20,120),
                                                               mask=np.concatenate((np.ones(5), np.zeros(10), np.ones(85)))))
        self.vapp = P('Airspeed Minus Vapp For 3 Sec', np.ma.array(data=np.arange(30,130),
                                                                   mask=np.concatenate((np.ones(80), np.zeros(10), np.ones(10)))))
        self.vref = P('Airspeed Minus Vref For 3 Sec', np.ma.array(data=np.arange(40,140),
                                                                   mask=np.concatenate((np.ones(80), np.zeros(10), np.ones(10)))))

    def test_derive__vapp(self):
        node = self.node_class()
        node.derive(None, self.vapp, None)
        ma_test.assert_masked_array_equal(node.array, self.vapp.array)

    def test_derive__vref(self):
        node = self.node_class()
        node.derive(None, None, self.vref)
        ma_test.assert_masked_array_equal(node.array, self.vref.array)

    def test_derive__both_approach(self):
        node = self.node_class()
        node.derive(None, self.vapp, self.vref)
        ma_test.assert_masked_array_equal(node.array, self.vapp.array)

    def test_derive_all_three(self):
        node = self.node_class()
        node.derive(self.v2, self.vapp, self.vref)
        expected = np.ma.array(data=np.concatenate((np.arange(20,70), np.arange(80,130))),
                               mask=np.ma.concatenate((np.ones(5), np.zeros(10), np.ones(65), np.zeros(10), np.ones(10))))
        ma_test.assert_masked_array_equal(node.array, expected)

    def test_derive_v2(self):
        node = self.node_class()
        node.derive(self.v2, None, None)
        ma_test.assert_masked_array_equal(node.array, self.v2.array)



class TestAirspeedMinusAirspeedSelectedFor3Sec(unittest.TestCase):
    def test_can_operate(self):
        o = AirspeedMinusAirspeedSelectedFor3Sec.get_operational_combinations()
        self.assertEqual([('Airspeed', 'Airspeed Selected For Approaches')], o)

    def test_derive(self):
        aspd = P('Airspeed', array=np.ma.ones(250) * 256)
        aspd_sel = P('Airspeed Selected', array=np.ma.concatenate((np.ones(10) * 999, np.zeros(90), np.ones(150) * 250)))
        aspd_minus_sel = AirspeedMinusAirspeedSelectedFor3Sec()
        aspd_minus_sel.get_derived([aspd, aspd_sel])
        assert_array_equal(
            aspd_minus_sel.array[-150:], np.ma.ones(150)*6)

##############################################################################

########################################
# Aircraft Energy


class TestKineticEnergy(unittest.TestCase):
    def test_derive(self):

        airspeed_array = np.ma.concatenate((np.zeros(20),
                                            np.arange(20,170,1.25),
                                            np.ones(1000) * 170,
                                            np.arange(170,20,-0.25),
                                            np.zeros(260)))
        airspeed = P(name='Airspeed True', array=airspeed_array, frequency=1,
                       offset=0)

        mass_array=np.ma.arange(23000,21000,-1)
        mass = P(name='Gross Weight Smoothed', array=mass_array, frequency=1,
                offset=0)

        ke = KineticEnergy()
        ke.derive(airspeed,mass)

        # take element 100, 120 knots,
        # velocity = airspeed in m/s
        # velocity = 120 * 0.5144
        # airspeed at element 100 = 120
        # mass at element 100 = 22900
        # result = 0.5 * 22900 * velocity **2
        # result / 10 ** 6

        self.assertAlmostEqual(ke.array[100], 43.6359, places=4)

    def test_can_operate(self):
        result = KineticEnergy.can_operate(['Airspeed True','Gross Weight Smoothed'])
        self.assertTrue(result)   # positive test
        result = KineticEnergy.can_operate(['Airspeed True',''])
        self.assertFalse(result)  # partial 1 test
        result = KineticEnergy.can_operate(['Gross Weight Smoothed',])
        self.assertFalse(result)  # missing other test


class TestPotentialEnergy(unittest.TestCase):
    def test_derive(self):
        gross_weight_smoothed_array = np.ma.arange(23000, 21000, -1)
        gross_weight_smoothed = P(name='Gross Weight Smoothed', array=gross_weight_smoothed_array)

        altitude_aal_array = np.ma.concatenate((np.zeros(50),
                                                np.arange(0, 10000, 100),
                                                np.ones(1700) * 10000,
                                                np.arange(10000, 0, -100),
                                                np.zeros(50)))
        altitude_aal = P(name='Altitude AAL', array=altitude_aal_array)

        self.assertEqual(len(gross_weight_smoothed_array), len(altitude_aal_array))

        pe = PotentialEnergy()
        pe.derive(gross_weight_smoothed, altitude_aal)

        # gross_weight_smoothed[1000] = 22000
        # altitude_aal[1000] = 10000
        # converstion FT / METER = 0.3048
        # gravity meter = 9.81
        # potential energy[1000] = 657.81936

        self.assertAlmostEqual(pe.array[1000],657.81936, places=5)

    def test_can_operate(self):
        result = PotentialEnergy.can_operate(['Altitude AAL','Gross Weight Smoothed'])
        self.assertTrue(result)   # positive test


class TestAircraftEnergy(unittest.TestCase):
    def test_derive(self):

        #kinetic_energy_array = np.ma.concatenate((np.zeros(90),
                                                  #np.arange(0, 1000, 100),
                                                  #np.ones(800) * 1000,
                                                  #np.arange(1000, 0, -100),
                                                  #np.zeros(90)))
        kinetic_energy_array = np.ma.concatenate(( [0]*90,
                                                  np.ma.arange( 0, 1000, 100),
                                                  [1000]*800,
                                                  np.ma.arange(1000,0,-100),
                                                  [0]*90 ))

        kinetic_energy = P('Kinetic Energy',
                           array=kinetic_energy_array)

        potential_energy_array = np.ma.concatenate(([0]*50,
                                                    np.ma.arange( 0, 10000, 50),
                                                    [10000]*500,
                                                    np.ma.arange(10000,0,-50),
                                                    [0]*50 ))
        #potential_energy_array = np.ma.concatenate((np.zeros(50),
                                                    #np.arange(0, 10000, 50),
                                                    #np.zeros(500) * 10000,
                                                    #np.arange(10000, 0, -50),
                                                    #np.zeros(50)))

        potential_energy = P('Potential Energy',
                             array=potential_energy_array)

        ae = AircraftEnergy()
        ae.derive(kinetic_energy, potential_energy)

        #60 Pot: 500, Kin:0, Total: 500
        #92 Pot: 2100 Kin: 200, Total: 2300
        #500 Pot: 10000, Kin:1000, Total: 11000
        self.assertAlmostEqual( ae.array[60], 500 )
        self.assertAlmostEqual( ae.array[92], 2300 )
        self.assertAlmostEqual( ae.array[500], 11000 )


if __name__ == '__main__':
    unittest.main()
