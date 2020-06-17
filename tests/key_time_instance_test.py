import numpy as np
import os
import sys
import unittest

from flightdatautilities.array_operations import load_compressed

from hdfaccess.parameter import MappedArray

from analysis_engine.node import (
    A, M, P, S, KTI, aeroplane, ApproachItem, helicopter,
    KeyTimeInstance, load, Parameter, Section
)

from analysis_engine.key_time_instances import (
    AltitudePeak,
    AltitudeWhenClimbing,
    AltitudeWhenDescending,
    AltitudeBeforeLevelFlightWhenClimbing,
    AltitudeBeforeLevelFlightWhenDescending,
    APDisengagedSelection,
    APEngagedSelection,
    APUStart,
    APUStop,
    ATDisengagedSelection,
    ATEngagedSelection,
    Autoland,
    BottomOfDescent,
    ClimbAccelerationStart,
    ClimbStart,
    ClimbThrustDerateDeselected,
    DistanceFromLandingAirport,
    DistanceFromTakeoffAirport,
    DistanceFromThreshold,
    EngFireExtinguisher,
    EngStart,
    EngStop,
    EnterHold,
    ExitHold,
    FirstEngFuelFlowStart,
    FirstEngStartBeforeLiftoff,
    LastEngStartBeforeLiftoff,
    FirstFlapExtensionWhileAirborne,
    FlapExtensionWhileAirborne,
    FlapLeverSet,
    FlapAlternateArmedSet,
    FlapLoadReliefSet,
    FlapRetractionWhileAirborne,
    FlapRetractionDuringGoAround,
    GearDownSelection,
    GearUpSelection,
    GearUpSelectionDuringGoAround,
    GoAround,
    InitialClimbStart,
    LandingDecelerationEnd,
    LandingStart,
    LandingTurnOffRunway,
    LastEngFuelFlowStop,
    LastEngStopAfterTouchdown,
    Liftoff,
    LocalizerEstablishedEnd,
    LocalizerEstablishedStart,
    LowestAltitudeDuringApproach,
    MinsToTouchdown,
    MovementStart,
    MovementStop,
    OffBlocks,
    OnBlocks,
    OffshoreTouchdown,
    OnshoreTouchdown,
    SecsToTouchdown,
    DistanceToTouchdown,
    SlatAlternateArmedSet,
    TakeoffAccelerationStart,
    TakeoffPeakAcceleration,
    TakeoffTurnOntoRunway,
    TAWSGlideslopeCancelPressed,
    TAWSTerrainOverridePressed,
    TAWSMinimumsTriggered,
    TopOfClimb,
    TopOfDescent,
    TouchAndGo,
    Touchdown,
    Transmit
)

from analysis_engine.test_utils import buildsection, buildsections

debug = sys.gettrace() is not None

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')

##############################################################################
# Superclasses


class NodeTest(object):

    def test_can_operate(self):
        if not hasattr(self, 'node_class'):
            return
        kwargs = getattr(self, 'can_operate_kwargs', {})
        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations(**kwargs)),
                self.operational_combination_length,
            )
        else:
            combinations = list(map(set, self.node_class.get_operational_combinations(**kwargs)))
            for combination in map(set, self.operational_combinations):
                self.assertIn(combination, combinations)


##############################################################################


class TestAltitudePeak(unittest.TestCase):
    def setUp(self):
        self.alt_aal = P(name='Altitude AAL', array=np.ma.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
        ]))

    def test_can_operate(self):
        expected = [('Altitude AAL',)]
        self.assertEqual(AltitudePeak.get_operational_combinations(), expected)

    def test_derive(self):
        alt_peak = AltitudePeak()
        alt_peak.derive(self.alt_aal)
        expected = [KeyTimeInstance(name='Altitude Peak', index=9)]
        self.assertEqual(alt_peak, expected)


class TestBottomOfDescent(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            BottomOfDescent.get_operational_combinations(),
            [('Altitude STD Smoothed', 'Climb Cruise Descent',)],
        )

    def test_bottom_of_descent_basic(self):
        testwave = np.cos(np.arange(3.2, 9.6, 0.1)) * (2500) + 2560
        alt_std = Parameter('Altitude STD Smoothed', np.ma.array(testwave))
        from analysis_engine.flight_phase import ClimbCruiseDescent
        airs = S('Airborne', items=[Section('Airborne', slice(0, 64), 0, 64)])
        ccd = ClimbCruiseDescent()
        #air = buildsection('Airborne', 0,50)
        #duration = A('HDF Duration', 63)
        ccd.derive(alt_std, airs)
        bod = BottomOfDescent()
        bod.derive(alt_std, ccd)
        expected = [KeyTimeInstance(index=62, name='Bottom Of Descent')]
        self.assertEqual(bod, expected)

    def test_bottom_of_descent_complex(self):
        testwave = np.cos(np.arange(3.2, 9.6, 0.1)) * (2500) + 2560
        alt_std = Parameter('Altitude STD Smoothed', np.ma.array(testwave))
        #airs = buildsections('Airborne', [896, 1654], [1688, 2055])
        ccds = buildsections('Climb Cruise Descent', [897, 1253], [1291, 1651], [1689, 2054])
        #duration = A('HDF Duration', 3000)
        bod = BottomOfDescent()
        bod.derive(alt_std, ccds)
        self.assertEqual(len(bod), 3)
        self.assertEqual(bod[0].index, 1254)

        expected = [KeyTimeInstance(index=1254, name='Bottom Of Descent'),
                    KeyTimeInstance(index=1652, name='Bottom Of Descent'),
                    KeyTimeInstance(index=2055, name='Bottom Of Descent')]
        self.assertEqual(bod, expected)

    def test_bod_ccd_only(self):
        testwave = np.cos(np.arange(3.2, 9.6, 0.1)) * (2500) + 2560
        alt_std = Parameter('Altitude STD Smoothed', np.ma.array(testwave))
        ccds = buildsection('Climb Cruise Descent', 897, 1253)
        bod = BottomOfDescent()
        bod.derive(alt_std, ccds)
        self.assertEqual(len(bod), 1)
        self.assertEqual(bod[0].index, 1254)

    def test_bod_end(self):
        testwave = np.cos(np.arange(3.2, 9.6, 0.1)) * (2500) + 2560
        alt_std = Parameter('Altitude STD Smoothed', np.ma.array(testwave))
        ccds = buildsection('Climb Cruise Descent', 897, 2000)
        bod = BottomOfDescent()
        bod.derive(alt_std, ccds)
        expected = [KeyTimeInstance(index=2001, name='Bottom Of Descent')]
        self.assertEqual(bod, expected)

    def test_bod_end_none(self):
        testwave = np.cos(np.arange(3.2, 9.6, 0.1)) * (2500) + 2560
        alt_std = Parameter('Altitude STD Smoothed', np.ma.array(testwave))
        ccds = buildsection('Climb Cruise Descent', 897, None)
        bod = BottomOfDescent()
        bod.derive(alt_std, ccds)
        expected = []
        self.assertEqual(bod, expected)


class TestClimbStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases', 'Liftoff', 'Top Of Climb')]
        opts = ClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climb_start_basic(self):
        #vert_spd = Parameter('Vertical Speed', np.ma.array([1200]*8))
        #climb = Climbing()
        #climb.derive(vert_spd, [Section('Fast',slice(0,8,None),0,8)])
        alt_aal = Parameter('Altitude AAL', np.ma.arange(0, 1600, 220))
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(0, 'Liftoff')])
        tocs = KTI('Top Of Climb',
                   items=[KeyTimeInstance(len(alt_aal.array) - 1, 'Top Of Climb')])
        kti = ClimbStart()
        kti.derive(alt_aal, liftoffs, tocs)
        # These values give an result with an index of 4.5454 recurring.
        self.assertEqual(len(kti), 1)
        self.assertAlmostEqual(kti[0].index, 4.5, 1)

    def test_climb_start_slow_climb(self):
        # Failed when ClimbStart was based on Climbing.
        alt_aal = load(os.path.join(test_data_path,
                                    'ClimbStart_AltitudeAAL_1.nod'))
        liftoffs = load(os.path.join(test_data_path,
                                    'ClimbStart_Liftoff_1.nod'))
        tocs = load(os.path.join(test_data_path,
                                 'ClimbStart_TopOfClimb_1.nod'))
        kti = ClimbStart()
        kti.derive(alt_aal, liftoffs, tocs)
        self.assertEqual(len(kti), 1)
        self.assertAlmostEqual(kti[0].index, 1384, 0)

    def test_climb_start_for_helicopter_operation(self):
        # This test case is borne out of actual helicopter data.
        alt_aal = Parameter('Altitude AAL', np.ma.array([0]*17+[2000]*50+[0]*26+[2000]*50))
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(0, 'Liftoff'),
                                         KeyTimeInstance(3, 'Liftoff'),
                                         KeyTimeInstance(7, 'Liftoff'),
                                         KeyTimeInstance(12, 'Liftoff'),
                                         KeyTimeInstance(15, 'Liftoff'),
                                         KeyTimeInstance(91, 'Liftoff')])
        tocs = KTI('Top Of Climb',
                   items=[KeyTimeInstance(20, 'Top Of Climb'),
                          KeyTimeInstance(30, 'Top Of Climb'),
                          KeyTimeInstance(77, 'Top Of Climb'),
                          KeyTimeInstance(84, 'Top Of Climb'),
                          KeyTimeInstance(94, 'Top Of Climb')])
        kti = ClimbStart()
        kti.derive(alt_aal, liftoffs, tocs)
        self.assertEqual(len(kti), 2)
        self.assertEqual(kti[0].index, 16.5)
        self.assertEqual(kti[1].index, 92.5)

class TestClimbAccelerationStart(unittest.TestCase):

    def setUp(self):
        self.node_class = ClimbAccelerationStart

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Airspeed Selected',
                                                     'Initial Climb',
                                                     'Altitude When Climbing')))
        self.assertTrue(self.node_class.can_operate(('Airspeed Selected',
                                                     'Airspeed Target',
                                                     'Initial Climb',
                                                     'Altitude When Climbing')))
        self.assertTrue(self.node_class.can_operate(('Airspeed',
                                                     'Flap Lever',
                                                     'Initial Climb',
                                                     'Altitude When Climbing')))
        self.assertTrue(self.node_class.can_operate(('Airspeed',
                                                     'Flap Lever (Synthetic)',
                                                     'Initial Climb',
                                                     'Altitude When Climbing')))
        self.assertTrue(self.node_class.can_operate(('Altitude AAL For Flight Phases',
                                                     'Engine Propulsion',
                                                     'Initial Climb',
                                                     'Altitude When Climbing')))
        jet = A('Engine Propulsion', 'JET')
        self.assertTrue(self.node_class.can_operate(('Throttle Levers',
                                                     'Initial Climb',
                                                     'Altitude When Climbing'),
                                                    eng_type=jet))
        prop = A('Engine Propulsion', 'PROP')
        self.assertTrue(self.node_class.can_operate(('Eng (*) Np Max',
                                                     'Initial Climb',
                                                     'Altitude When Climbing'),
                                                    eng_type=prop))

    def test_derive_basic(self):
        array = np.ma.concatenate((np.ones(15) * 110, np.ones(20) * 180))
        spd_sel = Parameter('Airspeed Selected', array=array)
        init_climbs = buildsection('Initial Climb', 5, 19)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(29, name='4000 Ft Climbing')]
        )
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel,
                    None, None, None, None, None, None, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 14)

    def test_derive_spd_analog(self):
        spd_sel = load(os.path.join(test_data_path, 'climb_acceleration_start_spd_sel_analog.nod'))
        init_climbs = buildsection('Initial Climb', 714, 741)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(761, name='4000 Ft Climbing')]
        )
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel,
                    None, None, None, None, None, None, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 1461, places=0)

    def test_derive_spd_unchanged(self):
        array = np.ma.array([155]*35)
        spd_sel = Parameter('Airspeed Selected', array=array)
        init_climbs = buildsection('Initial Climb', 5, 29)
        alt_climbing = AltitudeWhenClimbing()
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel,
                    None, None, None, None, None, None, None, None, None)
        self.assertEqual(len(node), 0)

    def test_derive_spd_masked(self):
        array = np.ma.concatenate((np.ones(15) * 155, np.ones(20) * 180))
        array[5:30] = np.ma.masked
        spd_sel = Parameter('Airspeed Selected', array=array)
        init_climbs = buildsection('Initial Climb', 5, 29)
        alt_climbing = AltitudeWhenClimbing()
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel,
                    None, None, None, None, None, None, None, None, None)
        self.assertEqual(len(node), 0)

    def test_derive_spd_sel_fmc_first(self):
        array = np.ma.concatenate((np.ones(15) * 110, np.ones(20) * 180))
        array[5:30] = np.ma.masked
        spd_sel = Parameter('Airspeed Selected', array=array)
        array = np.ma.concatenate((np.ones(15) * 130, np.ones(20) * 180))
        array[20:30] = np.ma.masked
        spd_sel_fmc = Parameter('Airspeed Selected (FMC)', array=array)
        init_climbs = buildsection('Initial Climb', 5, 19)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(29, name='4000 Ft Climbing')]
        )
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel, None, spd_sel_fmc,
                    None, None, None, None, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 14)

    def test_derive_spd_sel_fmc_last_lower_freq(self):
        array = np.ma.concatenate((np.ones(20) * 110, np.ones(15) * 180))
        spd_sel = Parameter('Airspeed Selected', array=array)
        array = np.ma.concatenate((np.ones(12) * 130, np.ones(5) * 180))
        spd_sel_fmc = Parameter('Airspeed Selected (FMC)', array=array, frequency=0.5)
        init_climbs = buildsection('Initial Climb', 5, 19)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(34, name='4000 Ft Climbing')]
        )
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel, None, spd_sel_fmc,
                    None, None, None, None, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 19)

    def test_derive_engine_propulsion(self):
        jet = A('Engine Propulsion', value='JET')
        alt_aal = P('Altitude AAL For Flight Phases', array=np.ma.arange(1000))
        init_climbs = buildsection('Initial Climb', 35, 1000)
        alt_climbing = AltitudeWhenClimbing()
        masked_array = np.ma.array(data=np.zeros(1000), mask=True)
        throttle = P('Throttle Levers', array=masked_array)
        node = self.node_class()
        node.derive(alt_aal, init_climbs, alt_climbing, None, None, None, jet,
                    None, throttle, None, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 800)
        prop = A('Engine Propulsion', value='PROP')
        eng_np = P('Eng (*) Np Max', array=masked_array)
        node = self.node_class()
        node.derive(alt_aal, init_climbs, alt_climbing, None, None, None, prop, eng_np,
                    None, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 400)

    def test_derive_spd(self):
        array = np.ma.concatenate((np.ones(29) * 111, np.arange(110, 181)))
        spd = Parameter('Airspeed', array=array)
        flap = M(
            'Flap Lever (Synthetic)',
            array=MappedArray(
                np.repeat((1, 0), (80, 20)),
                values_mapping={0: 'Lever 0', 1: 'Lever 1'}
            )
        )
        init_climbs = buildsection('Initial Climb', 5, 40)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(99, name='5000 Ft Climbing')]
        )
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, None, None, None, None, None, None,
                    spd, flap, None, None)
        self.assertEqual(len(node), 1)
        # With moving average over 11 samples, half window is 5 samples to the left
        # And we move an additional 3 seconds before
        self.assertEqual(node[0].index, 30-5-3)

    def test_derive_spd_flap_at_0_in_climb(self):
        array = np.ma.concatenate((np.ones(29) * 111, np.arange(110, 181)))
        spd = Parameter('Airspeed', array=array)
        flap = M(
            'Flap Lever (Synthetic)',
            array=MappedArray(
                np.repeat((0, 1), (80, 20)),
                values_mapping={0: 'Lever 0', 1: 'Lever 1'}
            )
        )
        init_climbs = buildsection('Initial Climb', 5, 40)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(60, name='5000 Ft Climbing')]
        )
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, None, None, None, None, None, None,
                    spd, flap, None, None)
        self.assertEqual(len(node), 0)

    def test_derive_spd_flap_extension(self):
        array = np.ma.concatenate((np.ones(29) * 111, np.arange(110, 181)))
        spd = Parameter('Airspeed', array=array)
        flap = M(
            'Flap Lever (Synthetic)',
            array=MappedArray(
                np.repeat((1, 2), (80, 20)),
                values_mapping={1: 'Lever 1', 2: 'Lever 2'}
            )
        )
        init_climbs = buildsection('Initial Climb', 5, 40)
        alt_climbing = AltitudeWhenClimbing(items=[
            KeyTimeInstance(90, name='4000 Ft Climbing'),
            KeyTimeInstance(99, name='5000 Ft Climbing'),
        ])
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, None, None, None, None, None, None,
                    spd, flap, None, None)
        self.assertEqual(len(node), 0)

    def test_derive_eng_np(self):
        initial_climbs = buildsection('Initial Climb', 887, 926)
        initial_climbs.frequency = 0.5
        alt_climbing = AltitudeWhenClimbing()
        prop = A('Engine Propulsion', value='PROP')
        eng_np = load(os.path.join(
            test_data_path, 'climb_acceleration_start_eng_np.nod'))
        node = self.node_class()
        node.derive(None, initial_climbs, alt_climbing, None, None, None, prop, eng_np,
                    None, None, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 917, places=0)

    def test_derive_eng_np_noise(self):
        '''
        Noisy Eng Np signal is smoothed allowing fallback to 400 Ft.
        '''
        initial_climbs = buildsection('Initial Climb', 419, 439)
        initial_climbs.frequency = 1
        alt_climbing = AltitudeWhenClimbing()
        prop = A('Engine Propulsion', value='PROP')
        alt_aal = load(os.path.join(
            test_data_path, 'climb_acceleration_start_alt_aal_noise.nod'))
        eng_np = load(os.path.join(
            test_data_path, 'climb_acceleration_start_eng_np_noise.nod'))
        # Make masked speed and flap array to default to Eng Np
        node = self.node_class()
        node.derive(alt_aal, initial_climbs, alt_climbing,
                    None, None, None, prop, eng_np, None, None, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 854, places=0)

    def test_derive_throttle_levers_fallback(self):
        initial_climbs = buildsection('Initial Climb', 511, 531)
        alt_climbing = AltitudeWhenClimbing()
        jet = A('Engine Propulsion', value='JET')
        alt_aal = load(os.path.join(
            test_data_path, 'climb_acceleration_start_alt_aal_fallback.nod'))
        throttle_levers = load(os.path.join(
            test_data_path, 'climb_acceleration_start_throttle_levers_fallback.nod'))
        node = self.node_class()
        node.derive(alt_aal, initial_climbs,alt_climbing, None, None, None, jet, None,
                    throttle_levers, None, None, None, None)
        self.assertEqual(len(node), 1)
        # Falls back to 800 Ft
        self.assertAlmostEqual(node[0].index, 527, places=0)

    def test_empty_initial_climb(self):
        array = np.ma.concatenate((np.ones(15) * 110, np.ones(20) * 180))
        spd_sel = Parameter('Airspeed Selected', array=array)
        init_climbs = S('Initial Climb')
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(29, name='4000 Ft Climbing')]
        )
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel,
                    None, None, None, None, None, None, None, None)
        self.assertEqual(len(node), 0)

    def test_derive_spd_sel_after_flap_retraction(self):
        array = np.ma.concatenate((np.ones(29) * 111, np.arange(110, 181)))
        spd = Parameter('Airspeed', array=array)
        spd_sel = Parameter(
            'Airspeed Selected',
            array=np.ma.concatenate((np.ones(90) * 111, np.ones(10) * 180))
        )
        flap_array = MappedArray(
            np.repeat((1, 0), (60, 40)),
            values_mapping={0: '0', 1: '1'}
        )
        flap = M('Flap Lever', array=flap_array)
        init_climbs = buildsection('Initial Climb', 5, 40)
        alt_climbing = AltitudeWhenClimbing(items=[
            KeyTimeInstance(90, name='4000 Ft Climbing'),
            KeyTimeInstance(99, name='5000 Ft Climbing'),
        ])
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel,
                    None, None, None, None, None, spd, flap, None, None)
        self.assertEqual(len(node), 1)
        # With moving average over 11 samples, half window is 5 samples to the left
        # And we move an additional 3 seconds before
        self.assertEqual(node[0].index, 30-5-3)

    def test_derive_spd_flap_small_acceleration(self):
        array = np.ma.concatenate((np.ones(29) * 111, np.arange(110, 115), np.ones(65) * 115))
        spd = Parameter('Airspeed', array=array)
        flap = M(
            'Flap Lever',
            array=MappedArray(
                np.repeat((5, 0), (80, 20)),
                values_mapping={0: '0', 5: '5'}
            )
        )
        init_climbs = buildsection('Initial Climb', 5, 40)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(99, name='4000 Ft Climbing')]
        )
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing,
                    None, None, None, None, None, None, spd, None, flap, None)
        # Too small acceleration. Bailed out from spd and flap algorithm.
        self.assertEqual(len(node), 0)

    def test_derive_spd_high_freq(self):
        array = np.ma.concatenate((np.ones(29) * 111, np.arange(110, 181)))
        spd = Parameter('Airspeed', array=array, frequency=2)
        flap = M(
            'Flap Lever',
            array=MappedArray(
                np.repeat((5, 0), (40, 60)),
                values_mapping={0: '0', 5: '5'}
            )
        )
        init_climbs = buildsection('Initial Climb', 2, 20)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(49, name='5000 Ft Climbing')]
        )
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing,
                    None, None, None, None, None, None, spd, flap, None, None)
        self.assertEqual(len(node), 1)
        # With moving average over 11 samples, half window is 5 samples to the left
        # And we move an additional 3 seconds before -> 6 samples at 2 Hz
        self.assertEqual(node[0].index, 30-5-6)

    def test_derive_airspeed_target(self):
        array = np.ma.concatenate((np.ones(15) * 110, np.ones(20) * 180))
        spd_sel = Parameter('Airspeed Selected', array=array)
        array = np.ma.concatenate((np.ones(10) * 110, np.ones(25) * 180))
        spd_tgt = Parameter('Airspeed Target', array=array)
        init_climbs = buildsection('Initial Climb', 5, 19)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(29, name='4000 Ft Climbing')]
        )
        family = A('Family', value='B787')
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel, spd_tgt,
                    None, None, None, None, None, None, None, family)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 9)

    def test_derive_spd_tgt_fluctuating(self):
        array = np.ma.concatenate((
            np.ones(15) * 155,
            np.ones(10) * 165,
            np.ones(10) * 220
        ))
        spd_sel = Parameter('Airspeed Selected', array=array)
        array = np.ma.concatenate((
            np.ones(10) * 155,
            np.array([155, 159, 159, 155, 157, 161, 159, 165, 160, 164]),
            np.ones(5) * 165,
            np.ones(10) * 220
        ))
        spd_tgt = Parameter('Airspeed Target', array=array)
        init_climbs = buildsection('Initial Climb', 1, 30)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(33, name='4000 Ft Climbing')]
        )
        family = A('Family', value='B787')
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel, spd_tgt,
                    None, None, None, None, None, None, None, family)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 24)

    def test_derive_spd_tgt_masked_during_jump(self):
        array = np.ma.concatenate((np.ones(15) * 110, np.ones(20) * 180))
        spd_sel = Parameter('Airspeed Selected', array=array)
        array = np.ma.concatenate((np.ones(10) * 110, np.ones(25) * 180))
        array[8:12] = np.ma.masked
        spd_tgt = Parameter('Airspeed Target', array=array)
        init_climbs = buildsection('Initial Climb', 5, 19)
        alt_climbing = AltitudeWhenClimbing(
            items=[KeyTimeInstance(29, name='4000 Ft Climbing')]
        )
        family = A('Family', value='B787')
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel, spd_tgt,
                    None, None, None, None, None, None, None, family)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 11)

    def test_derive_2_spd_sel_changes_before_flap_retraction(self):
        array = np.ma.concatenate((np.ones(29) * 111, np.arange(110, 181)))
        spd = Parameter('Airspeed', array=array)
        spd_sel = Parameter(
            'Airspeed Selected',
            array=np.ma.concatenate((
                np.ones(30) * 111, np.ones(40) * 160, np.ones(30) * 180
            ))
        )
        flap_array = MappedArray(
            np.repeat((1, 0), (80, 20)),
            values_mapping={0: '0', 1: '1'}
        )
        flap = M('Flap Lever', array=flap_array)
        init_climbs = buildsection('Initial Climb', 5, 40)
        alt_climbing = AltitudeWhenClimbing(items=[
            KeyTimeInstance(90, name='4000 Ft Climbing'),
            KeyTimeInstance(99, name='5000 Ft Climbing'),
        ])
        node = self.node_class()
        node.derive(None, init_climbs, alt_climbing, spd_sel,
                    None, None, None, None, None, spd, flap, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 69)


class TestClimbThrustDerateDeselected(unittest.TestCase):
    def test_can_operate(self):
        ac_family = A('Family', 'B787')
        expected = ('AT Climb 1 Derate', 'AT Climb 2 Derate')
        self.assertTrue(ClimbThrustDerateDeselected.can_operate(expected, ac_family=ac_family))
        self.assertFalse(ClimbThrustDerateDeselected.can_operate(expected, ac_family=None))

    def test_derive_basic(self):
        values_mapping = {0: 'Not Latched', 1: 'Latched'}
        climb_derate_1 = M('AT Climb 1 Derate',
                           array=np.ma.array([0,0,1,1,0,0,0,1,0,0]),
                           values_mapping=values_mapping)
        climb_derate_2 = M('AT Climb 2 Derate',
                           array=np.ma.array([0,0,0,1,1,0,0,0,0,0]),
                           values_mapping=values_mapping)
        node = ClimbThrustDerateDeselected()
        node.derive(climb_derate_1, climb_derate_2)
        self.assertEqual(node, [KeyTimeInstance(index=4.5, name='Climb Thrust Derate Deselected'),
                                KeyTimeInstance(index=7.5, name='Climb Thrust Derate Deselected')])


class TestGoAround(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GoAround.get_operational_combinations(),
                    [('Descent Low Climb', 'Altitude AAL For Flight Phases'),
                     ('Descent Low Climb', 'Altitude AAL For Flight Phases',
                      'Altitude Radio')])

    def test_go_around_basic(self):
        dlc = [Section('Descent Low Climb',slice(10,18),10,18)]
        alt = Parameter('Altitude AAL',
                        np.ma.array(list(range(0,4000,500))+
                                    list(range(4000,0,-500))+
                                    list(range(0,1000,501))))
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        goa.derive(dlc,alt,alt)
        expected = [KeyTimeInstance(index=16, name='Go Around')]
        self.assertEqual(goa, expected)

    def test_multiple_go_arounds(self):
        # This tests for three go-arounds, but the fourth part of the curve
        # does not produce a go-around as it ends in mid-descent.
        alt = Parameter('Altitude AAL',
                        np.ma.array(np.cos(
                            np.arange(0,21,0.02))*(1000)+2500))
        ####if debug:
        ####    from analysis_engine.plot_flight import plot_parameter
        ####    plot_parameter(alt.array)

        dlc = buildsections('Descent Low Climb',[50,260],[360,570],[670,890])

        ## Merge with analysis_engine refactoring
            #from analysis_engine.plot_flight import plot_parameter
            #plot_parameter(alt)

        #aal = ApproachAndLanding()
        #aal.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   #Parameter('Altitude Radio For Flight Phases',alt))

        #climb = ClimbForFlightPhases()
        #climb.derive(Parameter('Altitude STD Smoothed', alt),
                     #[Section('Fast',slice(0,len(alt),None))])

        goa = GoAround()
        goa.derive(dlc,alt,alt)

        expected = [KeyTimeInstance(index=157, name='Go Around'),
                    KeyTimeInstance(index=471, name='Go Around'),
                    KeyTimeInstance(index=785, name='Go Around')]
        self.assertEqual(goa, expected)

    def test_go_around_no_rad_alt(self):
        # This tests that the go-around works without a radio altimeter.
        dlc = [Section('Descent Low Climb',slice(10,18),10,18)]
        alt = Parameter('Altitude AAL',\
                        np.ma.array(list(range(0,4000,500))+
                                    list(range(4000,0,-500))+
                                    list(range(0,1000,501))))
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        goa.derive(dlc,alt,None)
        expected = [KeyTimeInstance(index=16, name='Go Around')]
        self.assertEqual(goa, expected)


    def test_go_around_with_rad_alt(self):
        # This tests that the go-around works without a radio altimeter.
        alt = Parameter('Altitude AAL',
                        np.ma.array(np.cos(
                            np.arange(0,21,0.02))*(1000)+2500))
        alt_rad = Parameter('Altitude Radio',\
                        alt.array-range(len(alt.array)))
        ####if debug:
        ####    from analysis_engine.plot_flight import plot_parameter
        ####    plot_parameter(alt_rad.array)
        # The sloping graph has shifted minima. We only need to check one to
        # show it's using the rad alt signal.
        dlc = [Section('Descent Low Climb',slice( 50,260),50,260)]

        goa = GoAround()
        goa.derive(dlc,alt,alt_rad)
        expected = [KeyTimeInstance(index=160, name='Go Around')]
        self.assertEqual(goa, expected)



"""
class TestAltitudeInApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeInApproach.get_operational_combinations(),
                         [('Approach', 'Altitude AAL')])

    def test_derive(self):
        approaches = S('Approach', items=[Section('a', slice(4, 7)),
                                                      Section('b', slice(10, 20))])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(list(range(1950, 0, -200)) +
                                       list(range(1950, 0, -200))))
        altitude_in_approach = AltitudeInApproach()
        altitude_in_approach.derive(approaches, alt_aal)
        self.assertEqual(list(altitude_in_approach),
          [KeyTimeInstance(index=4.75, name='1000 Ft In Approach',
                           datetime=None, latitude=None, longitude=None),
           KeyTimeInstance(index=14.75, name='1000 Ft In Approach',
                           datetime=None, latitude=None, longitude=None),
           KeyTimeInstance(index=12.25, name='1500 Ft In Approach',
                           datetime=None, latitude=None, longitude=None)])
"""

"""
class TestAltitudeInFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeInFinalApproach.get_operational_combinations(),
                         [('Approach', 'Altitude AAL')])

    def test_derive(self):
        approaches = S('Approach',
                       items=[Section('a', slice(2, 7)),
                              Section('b', slice(10, 20))])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(list(range(950, 0, -100)) +
                                       list(range(950, 0, -100))))
        altitude_in_approach = AltitudeInFinalApproach()
        altitude_in_approach.derive(approaches, alt_aal)

        self.assertEqual(list(altitude_in_approach),
          [KeyTimeInstance(index=4.5,
                           name='500 Ft In Final Approach', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=18.512820512820511,
                           name='100 Ft In Final Approach',
                           datetime=None, latitude=None, longitude=None),
           KeyTimeInstance(index=14.5,
                           name='500 Ft In Final Approach', datetime=None,
                           latitude=None, longitude=None)])
"""


class TestAltitudeWhenClimbing(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AltitudeWhenClimbing.get_operational_combinations(),
            [('Takeoff', 'Initial Climb', 'Climb', 'Altitude AAL', 'Altitude STD Smoothed')])

    def test_derive(self):
        takeoff = S(
            'Takeoff', items=[Section('Takeoff', slice(4, 13), 4, 13)])
        initial_climb = S(
            'Initial Climb', items=[Section('Initial Climb', slice(14, 30), 14, 30)])
        climb = S('Climb')
        arr = np.ma.arange(20, 200, 5)
        alt_aal = P('Altitude AAL', arr)
        altitude_when_climbing = AltitudeWhenClimbing()
        altitude_when_climbing.derive(takeoff, initial_climb, climb, alt_aal)
        self.assertEqual(
            list(altitude_when_climbing),
            [KeyTimeInstance(index=6, name='50 Ft Climbing'),
             KeyTimeInstance(index=11, name='75 Ft Climbing'),
             KeyTimeInstance(index=16, name='100 Ft Climbing'),
             KeyTimeInstance(index=26, name='150 Ft Climbing')])


class TestAltitudeWhenDescending(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            AltitudeWhenDescending.get_operational_combinations(),
            [('Descent', 'Altitude AAL', 'Altitude STD Smoothed')])

    def test_derive(self):
        descending = buildsections('Descent', [0, 20])
        alt_aal = P(
            'Altitude AAL',
            np.ma.masked_array(
                range(100, 0, -10), mask=[False] * 6 + [True] * 3 + [False]))
        altitude_when_descending = AltitudeWhenDescending()
        altitude_when_descending.derive(descending, alt_aal)
        self.assertEqual(
            list(altitude_when_descending),
            [KeyTimeInstance(index=2.5, name='75 Ft Descending'),
             KeyTimeInstance(index=5.0, name='50 Ft Descending'),
             KeyTimeInstance(index=6.5, name='35 Ft Descending'),
             KeyTimeInstance(index=8.0, name='20 Ft Descending'),
             KeyTimeInstance(index=9.0, name='10 Ft Descending')])


class TestAltitudeBeforeLevelFlightWhenClimbing(unittest.TestCase):

    def setUp(self):
        self.node_class = AltitudeBeforeLevelFlightWhenClimbing

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Altitude STD Smoothed',
                                 'Level Flight',
                                 'Climb')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestAltitudeBeforeLevelFlightWhenDescending(unittest.TestCase):

    def setUp(self):
        self.node_class = AltitudeBeforeLevelFlightWhenDescending

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, [('Altitude STD Smoothed',
                                 'Level Flight',
                                 'Descending')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


#class TestAltitudeSTDWhenDescending(unittest.TestCase):
    #def test_can_operate(self):
        #self.assertEqual(AltitudeSTDWhenDescending.get_operational_combinations(),
                         #[('Descending', 'Altitude AAL', 'Altitude STD Smoothed')])

    #def test_derive(self):
        #descending = buildsections('Descending', [0, 10], [11, 20])
        #alt_aal = P('Altitude STD',
                    #np.ma.masked_array(range(100, 0, -10),
                                       #mask=[False] * 6 + [True] * 3 + [False]))
        #altitude_when_descending = AltitudeSTDWhenDescending()
        #altitude_when_descending.derive(descending, alt_aal)
        #self.assertEqual(list(altitude_when_descending),
          #[KeyTimeInstance(index=2.5, name='75 Ft Descending'),
           #KeyTimeInstance(index=5.0, name='50 Ft Descending'),
        #])


class TestInitialClimbStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff',)]
        opts = InitialClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_initial_climb_start_basic(self):
        instance = InitialClimbStart()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Takeoff',slice(0,4,None),0,3.5)])
        expected = [KeyTimeInstance(index=3.5, name='Initial Climb Start')]
        self.assertEqual(instance, expected)

class TestLandingDecelerationEnd(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed','Landing')]
        opts = LandingDecelerationEnd.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_landing_end_deceleration(self):
        landing = [Section('Landing',slice(2,40,None),2,40)]
        speed = np.ma.array([79,77.5,76,73.9,73,70.3,68.8,67.6,66.4,63.4,62.8,
                             61.6,61.9,61,60.1,56.8,53.8,49.6,47.5,46,44.5,43.6,
                             42.7,42.4,41.8,41.5,40.6,39.7,39.4,38.5,37.9,38.5,
                             38.5,38.8,38.5,37.9,37.9,37.9,37.9,37.9])
        aspd = P('Airspeed',speed)
        kpv = LandingDecelerationEnd()
        kpv.derive(aspd, landing)
        expected = [KeyTimeInstance(index=21.0, name='Landing Deceleration End')]
        self.assertEqual(kpv, expected)


class TestTakeoffPeakAcceleration(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffPeakAcceleration.get_operational_combinations(),
                         [('Takeoff', 'Acceleration Longitudinal')])

    def test_takeoff_peak_acceleration_basic(self):
        acc = P('Acceleration Longitudinal',
                np.ma.array([0,0,.1,.1,.2,.1,0,0]))
        landing = [Section('Takeoff',slice(2,5,None),2,5)]
        kti = TakeoffPeakAcceleration()
        kti.derive(landing, acc)
        expected = [KeyTimeInstance(index=4, name='Takeoff Peak Acceleration')]
        self.assertEqual(kti, expected)


class TestLandingStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing',)]
        opts = LandingStart.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_initial_landing_start_basic(self):
        instance = LandingStart()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Landing',slice(66,77,None),66,77)])
        expected = [KeyTimeInstance(index=66, name='Landing Start')]
        self.assertEqual(instance, expected)


class TestLandingTurnOffRunway(unittest.TestCase):
    node_class = LandingTurnOffRunway
    def test_can_operate(self):
        expected = [('Approach Information',)]
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_landing_turn_off_runway_basic(self):
        node = self.node_class()
        apps = [ApproachItem('LANDING', slice(0, 5), turnoff=26)]
        node.derive(apps)
        expected = [KeyTimeInstance(index=26, name='Landing Turn Off Runway')]
        self.assertEqual(node, expected)


class TestLiftoff(unittest.TestCase):
    #TODO: Extend test coverage. This algorithm was developed using lots of
    #test data and graphical inspection, but needs a formal test framework.
    def test_can_operate(self):
        self.assertTrue(('Airborne',) in Liftoff.get_operational_combinations())

    def test_liftoff_basic(self):
        # Linearly increasing climb rate with the 5 fpm threshold set between
        # the 5th and 6th sample points.
        vert_spd = Parameter('Vertical Speed Inertial',
                             (np.ma.arange(10) - 0.5) * 40)
        # Airborne section encloses the test point.
        airs = buildsection('Airborne', 6, None)
        lift=Liftoff()
        lift.derive(vert_spd, None, None, None, None, airs, None)
        expected = [KeyTimeInstance(index=6, name='Liftoff')]
        self.assertEqual(lift, expected)

    def test_liftoff_no_vert_spd_detected(self):
        # Check the backstop setting.
        vert_spd = Parameter('Vertical Speed Inertial',
                             (np.ma.array([0] * 40)))
        airs = buildsection('Airborne', 6, None)
        lift=Liftoff()
        lift.derive(vert_spd, None, None, None, None, airs, None)
        expected = [KeyTimeInstance(index=6, name='Liftoff')]
        self.assertEqual(lift, expected)

    def test_liftoff_already_airborne(self):
        vert_spd = Parameter('Vertical Speed Inertial',
                             (np.ma.array([0] * 40)))
        airs = buildsection('Airborne', None, 10)
        lift=Liftoff()
        lift.derive(vert_spd, airs)
        expected = []
        self.assertEqual(lift, expected)


class TestTakeoffAccelerationStart(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(\
            TakeoffAccelerationStart.get_operational_combinations(),
            [('Airspeed', 'Takeoff'),
             ('Airspeed', 'Takeoff', 'Acceleration Longitudinal Offset Removed'),
             ('Airspeed', 'Takeoff', 'Acceleration Longitudinal Offset Removed', 'Acceleration Longitudinal')])

    def test_takeoff_acceleration_start(self):
        # This test uses the same airspeed data as the library routine test,
        # so should give the same answer!
        airspeed_data = np.ma.array(data=[37.9,37.9,37.9,37.9,37.9,37.9,37.9,
                                          37.9,38.2,38.2,38.2,38.2,38.8,38.2,
                                          38.8,39.1,39.7,40.6,41.5,42.7,43.6,
                                          44.5,46,47.5,49.6,52,53.2,54.7,57.4,
                                          60.7,61.9,64.3,66.1,69.4,70.6,74.2,
                                          74.8],
                                    mask=[1]*22+[0]*15
                                    )
        takeoff = buildsection('Takeoff',3,len(airspeed_data))
        aspd = P('Airspeed', airspeed_data)
        instance = TakeoffAccelerationStart()
        instance.derive(aspd, takeoff,None)
        self.assertLess(instance[0].index, 1.0)
        self.assertGreater(instance[0].index, 0.5)

    def test_takeoff_acceleration_start_truncated(self):
        # This test uses the same airspeed data as the library routine test,
        # so should give the same answer!
        airspeed_data = np.ma.array(data=[37.9,37.9,37.9,37.9,37.9,
                                          37.9,38.2,38.2,38.2,38.2,38.8,38.2,
                                          38.8,39.1,39.7,40.6,41.5,42.7,43.6,
                                          44.5,46,47.5,49.6,52,53.2,54.7,57.4,
                                          60.7,61.9,64.3,66.1,69.4,70.6,74.2,
                                          74.8],
                                    mask=[1]*20+[0]*15
                                    )
        takeoff = buildsection('Takeoff',3,len(airspeed_data))
        aspd = P('Airspeed', airspeed_data)
        instance = TakeoffAccelerationStart()
        instance.derive(aspd, takeoff,None)
        self.assertEqual(instance[0].index, 0.0)

    def test_takeoff_acceleration_start_airspeed_masked(self):
        airspeed_data = np.ma.array(np.ones(50) * 250)
        airspeed_data[:40] = np.ma.masked
        takeoff = buildsection('Takeoff', 3, len(airspeed_data))
        aspd = P('Airspeed', airspeed_data)
        instance = TakeoffAccelerationStart()
        instance.derive(aspd, takeoff,None)
        self.assertEqual(len(instance), 0)


class TestTakeoffTurnOntoRunway(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Continuous','Takeoff','Fast')]
        opts = TakeoffTurnOntoRunway.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_takeoff_turn_onto_runway_basic(self):
        instance = TakeoffTurnOntoRunway()
        head = P('Heading Continuous',np.ma.arange(5))
        takeoff = buildsection('Takeoff',1.7,5.5)
        fast = buildsection('Fast',3.7,7)
        instance.derive(head, takeoff, fast)
        expected = [KeyTimeInstance(index=1.7, name='Takeoff Turn Onto Runway')]
        self.assertEqual(instance, expected)

    def test_takeoff_turn_onto_runway_curved(self):
        instance = TakeoffTurnOntoRunway()
        head = P('Heading Continuous',np.ma.array(list(range(20))+[20]*70))
        fast = buildsection('Fast',40,90)
        takeoff = buildsection('Takeoff',4,75)
        instance.derive(head, takeoff, fast)
        expected = [KeyTimeInstance(index=21.5, name='Takeoff Turn Onto Runway')]
        self.assertEqual(instance, expected)

    def test_takeoff_turn_onto_runway_no_fast_section_after_takeoff(self):
        instance = TakeoffTurnOntoRunway()
        head = P('Heading Continuous',np.ma.arange(5))
        takeoff = buildsection('Takeoff',1.7,5.5)
        fast = buildsection('Fast', 0.1, 1.5)
        instance.derive(head, takeoff, fast)
        expected = [KeyTimeInstance(index=1.7, name='Takeoff Turn Onto Runway')]
        self.assertEqual(instance, expected)


class TestTAWSGlideslopeCancelPressed(unittest.TestCase):

    def test_basic(self):
        tgc = M('TAWS Glideslope Cancel', ['Cancel', '-', '-', 'Cancel', 'Cancel', '-', '-'],
               values_mapping={0: '-', 1: 'Cancel'})
        air = buildsection('Airborne', 2, 8)
        glide = TAWSGlideslopeCancelPressed()
        glide.derive(tgc, air)
        expected = [KeyTimeInstance(index=2.5, name='TAWS Glideslope Cancel Pressed')]
        self.assertEqual(glide, expected)


class TestTAWSMinimumsTriggered(unittest.TestCase):

    def test_basic(self):
        tto = M('TAWS Minimums Triggered',
                ['-', '-', '-', 'Minimums', 'Minimums', '-', '-'],
                values_mapping={0: '-', 1: 'Minimums'})
        air = buildsection('Airborne', 2, 8)
        glide = TAWSMinimumsTriggered()
        glide.derive(tto, air)
        expected = [KeyTimeInstance(index=2.5,
                                    name='TAWS Minimums Triggered')]
        self.assertEqual(glide, expected)


class TestTAWSTerrainOverridePressed(unittest.TestCase):

    def test_basic(self):
        tto = M('TAWS Terrain Override Pressed',
                ['-', '-', '-', 'Override', 'Override', '-', '-'],
                values_mapping={0: '-', 1: 'Override'})
        air = buildsection('Airborne', 2, 8)
        glide = TAWSTerrainOverridePressed()
        glide.derive(tto, air)
        expected = [KeyTimeInstance(index=2.5,
                                    name='TAWS Terrain Override Pressed')]
        self.assertEqual(glide, expected)


class TestTopOfClimb(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed','Climb Cruise Descent')]
        opts = TopOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_top_of_climb_basic(self):
        alt_data = np.ma.array(list(range(0,800,100))+[800]*5+list(range(800,0,-100)))
        alt = Parameter('Altitude STD Smoothed', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=8, name='Top Of Climb')]
        self.assertEqual(phase, expected)

    def test_top_of_climb_truncated_start(self):
        alt_data = np.ma.array([800]*5+list(range(800,0,-100)))
        alt = Parameter('Altitude STD Smoothed', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = []
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),0)

    def test_top_of_climb_truncated_end(self):
        alt_data = np.ma.array(list(range(0,800,100))+[800]*5)
        alt = Parameter('Altitude STD Smoothed', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=8, name='Top Of Climb')]
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),1)


class TestTopOfDescent(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed', 'Climb Cruise Descent')]
        opts = TopOfDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_top_of_descent_basic(self):
        alt_data = np.ma.array(list(range(0,800,100))+[800]*5+list(range(800,0,-100)))
        alt = Parameter('Altitude STD Smoothed', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=13, name='Top Of Descent')]
        self.assertEqual(phase, expected)

    def test_top_of_descent_truncated_start(self):
        alt_data = np.ma.array([800]*5+list(range(800,0,-100)))
        alt = Parameter('Altitude STD Smoothed', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=5, name='Top Of Descent')]
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),1)

    def test_top_of_descent_truncated_end(self):
        alt_data = np.ma.array(list(range(0,800,100))+[800]*5)
        alt = Parameter('Altitude STD Smoothed', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = buildsection('Climb Cruise Descent',0,len(alt.array)-1)
        phase.derive(alt, in_air)
        expected = []
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),0)


class TestTouchdown(unittest.TestCase):
    def test_can_operate(self):
        opts = Touchdown.get_operational_combinations()
        # Minimal case
        self.assertIn(('Altitude AAL', 'Landing'), opts)
        self.assertIn(('Acceleration Normal', 'Acceleration Longitudinal Offset Removed', 'Altitude AAL', 'Gear On Ground', 'Landing'), opts)

    def test_touchdown_with_minimum_requirements(self):
        # Test 1
        altitude = Parameter('Altitude AAL',
                             np.ma.array(data=[28, 21, 15, 10, 6, 3, 1, 0, 0,  0],
                                         mask = False))
        lands = buildsection('Landing', 2, 9)
        tdwn = Touchdown()
        tdwn.derive(None, None, altitude, None, None, lands)
        expected = [KeyTimeInstance(index=7, name='Touchdown')]
        self.assertEqual(tdwn, expected)

    def test_touchdown_using_alt(self):
        '''
        test to check index where altitude becomes 0 is used instead of
        inertial landing index. Gear on Ground index indicates height at 21
        feet.
        '''
        alt = load(os.path.join(test_data_path,
                                    'TestTouchdown-alt.nod'))
        gog = load(os.path.join(test_data_path,
                                    'TestTouchdown-gog.nod'))
        #FIXME: MappedArray should take values_mapping and apply it itself
        gog.array.values_mapping = gog.values_mapping
        lands = buildsection('Landing', 23279, 23361)
        tdwn = Touchdown()
        tdwn.derive(None, None, alt, alt, gog, lands)
        self.assertEqual(tdwn.get_first().index, 23292.0)


class TestOffshoreTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = OffshoreTouchdown
        self.operational_combinations = [('Touchdown', 'Offshore')]

    def test_derived(self):
        offshore = buildsections('Offshore', [10, 20], [30, 50], [60, 70])
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(15, 'Touchdown'),
                                            KeyTimeInstance(25, 'Touchdown'),
                                            KeyTimeInstance(35, 'Touchdown'),
                                            KeyTimeInstance(45, 'Touchdown'),
                                            KeyTimeInstance(55, 'Touchdown'),
                                            KeyTimeInstance(65, 'Touchdown')])

        node = self.node_class()
        node.derive(touchdown, offshore)

        expected = KTI('Offshore Touchdown', items=[KeyTimeInstance(15, 'Offshore Touchdown'),
                                                    KeyTimeInstance(35, 'Offshore Touchdown'),
                                                    KeyTimeInstance(45, 'Offshore Touchdown'),
                                                    KeyTimeInstance(65, 'Offshore Touchdown')])

        self.assertEqual(node, expected)


class TestOnshoreTouchdown(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = OnshoreTouchdown
        self.operational_combinations = [('Touchdown', 'Offshore')]

    def test_derived(self):
        offshore = buildsections('Offshore', [10, 20], [30, 50], [60, 70])
        touchdown = KTI('Touchdown', items=[KeyTimeInstance(15, 'Touchdown'),
                                            KeyTimeInstance(25, 'Touchdown'),
                                            KeyTimeInstance(35, 'Touchdown'),
                                            KeyTimeInstance(45, 'Touchdown'),
                                            KeyTimeInstance(55, 'Touchdown'),
                                            KeyTimeInstance(65, 'Touchdown')])

        node = self.node_class()
        node.derive(touchdown, offshore)

        expected = KTI('Onshore Touchdown', items=[KeyTimeInstance(25, 'Onshore Touchdown'),
                                                    KeyTimeInstance(55, 'Onshore Touchdown'),])

        self.assertEqual(node, expected)


##############################################################################
# Automated Systems


class TestAPEngagedSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = APEngagedSelection
        self.operational_combinations = [('AP Engaged', 'Fast')]

    def test_derive(self):
        ap = M(
            name='AP Engaged',
            array=['-', '-', '-', 'Engaged', '-', '-', '-'],
            values_mapping={0: '-', 1: 'Engaged'},
        )
        aes = APEngagedSelection()
        fast = buildsection('Fast', 2, 5)
        aes.derive(ap, fast)
        expected = [KeyTimeInstance(index=2.5, name='AP Engaged Selection')]
        self.assertEqual(aes, expected)


class TestAPDisengagedSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = APEngagedSelection
        self.operational_combinations = [('AP Engaged', 'Fast')]

    def test_derive(self):
        ap = M(
            name='AP Engaged',
            array=['-', '-', '-', 'Engaged', '-', '-', '-'],
            values_mapping={0: '-', 1: 'Engaged'},
        )
        ads = APDisengagedSelection()
        fast = buildsection('Fast', 2, 5)
        ads.derive(ap, fast)
        expected = [KeyTimeInstance(index=3.5, name='AP Disengaged Selection')]
        self.assertEqual(ads, expected)


class TestATEngagedSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ATEngagedSelection
        self.operational_combinations = [('AT Engaged', 'Airborne')]

    def test_derive(self):
        at = M(
            name='AT Engaged',
            array=['-', '-', '-', 'Engaged', '-', '-', '-'],
            values_mapping={0: '-', 1: 'Engaged'},
        )
        aes = ATEngagedSelection()
        air = buildsection('Airborne', 2, 5)
        aes.derive(at, air)
        expected = [KeyTimeInstance(index=2.5, name='AT Engaged Selection')]
        self.assertEqual(aes, expected)


class TestATDisengagedSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = ATEngagedSelection
        self.operational_combinations = [('AT Engaged', 'Airborne')]

    def test_derive(self):
        at = M(
            name='AT Engaged',
            array=['-', '-', '-', 'Engaged', '-', '-', '-'],
            values_mapping={0: '-', 1: 'Engaged'},
        )
        ads = ATDisengagedSelection()
        air = buildsection('Airborne', 2, 5)
        ads.derive(at, air)
        expected = [KeyTimeInstance(index=3.5, name='AT Disengaged Selection')]
        self.assertEqual(ads, expected)


##############################################################################

# Engine Start and Stop - may run into the ends of the valid recording.

class TestEngStart(unittest.TestCase):

    def test_can_operate(self):
        combinations = EngStart.get_operational_combinations()
        self.assertTrue(('Eng (1) N2',) in combinations)
        self.assertTrue(('Eng (2) N2',) in combinations)
        self.assertTrue(('Eng (3) N2',) in combinations)
        self.assertTrue(('Eng (4) N2',) in combinations)
        self.assertTrue(('Eng (1) N2', 'Eng (2) N2',
                         'Eng (3) N2', 'Eng (4) N2') in combinations)
        self.assertTrue(('Eng (1) N3',) in combinations)
        self.assertTrue(('Eng (2) N3',) in combinations)
        self.assertTrue(('Eng (3) N3',) in combinations)
        self.assertTrue(('Eng (4) N3',) in combinations)
        self.assertTrue(('Eng (1) N3', 'Eng (2) N3',
                         'Eng (3) N3', 'Eng (4) N3') in combinations)

    def test_basic(self):
        eng2 = Parameter('Eng (2) N2', np.ma.array([0, 20, 40, 60]))
        eng1 = Parameter(
            'Eng (1) N2',
            np.ma.array(data=[0, 0, 99, 99, 60, 60, 60], mask=[1, 1, 1, 1, 1, 1, 0]))
        es = EngStart()
        es.derive(
            None, None, None, None,
            eng1, eng2, None, None,
            None, None, None, None,
            None, None, None, None,
        )
        self.assertEqual(es[1].name, 'Eng (2) Start')
        self.assertEqual(es[1].index, 2)

    def test_prefer_N2(self):
        eng_N1 = Parameter('Eng (1) N1', np.ma.array([50, 50, 50, 50]))
        eng_N2 = Parameter('Eng (1) N2', np.ma.array(data=[0, 0, 0, 60]))
        es = EngStart()
        es.derive(eng_N1, None, None, None, eng_N2, None, None, None, None, None, None, None)
        self.assertEqual(es[0].name, 'Eng (1) Start')
        self.assertEqual(es[0].index, 3)
        self.assertEqual(len(es), 1)

    def test_three_spool(self):
        eng22 = Parameter(
            'Eng (2) N2', np.ma.array([0, 20, 40, 60, 0, 20, 40, 60]))
        eng12 = Parameter(
            'Eng (1) N2', np.ma.array(
                data=[0, 0, 99, 99, 60, 60, 60, 60], mask=[1, 1, 1, 1, 0, 0, 0, 0]))
        eng23 = Parameter('Eng (2) N3', np.ma.array([0, 40, 60, 60, 0, 0, 30, 60]))
        eng13 = Parameter(
            'Eng (1) N3', np.ma.array(
                data=[0, 0, 99, 99, 60, 60, 60, 60], mask=[1, 1, 1, 1, 0, 0, 0, 0]))
        es = EngStart()
        es.derive(None, None, None, None, eng12, eng22, None, None, eng13, eng23, None, None)
        self.assertEqual(len(es), 2)
        self.assertEqual(es[1].name, 'Eng (2) Start')
        self.assertEqual(es[1].index, 1)

    def test_N1_only(self):
        eng_N1 = Parameter('Eng (1) N1', np.ma.array([0, 5, 10, 11]))
        es = EngStart()
        es.derive(
            eng_N1, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
        )
        self.assertEqual(len(es), 1)
        self.assertEqual(es[0].name, 'Eng (1) Start')
        self.assertEqual(es[0].index, 2)

    def test_short_dip(self):
        eng_1_n3 = load(os.path.join(test_data_path, 'eng_start_eng_1_n3.nod'))
        eng_2_n3 = load(os.path.join(test_data_path, 'eng_start_eng_2_n3.nod'))
        node = EngStart()
        node.derive(
            None, None, None, None,
            None, None, None, None,
            eng_1_n3, eng_2_n3, None, None,
            None, None, None, None,
        )
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].name, 'Eng (1) Start')
        self.assertEqual(node[0].index, 161)
        self.assertEqual(node[1].name, 'Eng (2) Start')
        self.assertEqual(node[1].index, 94)


class TestFirstEngStartBeforeLiftoff(unittest.TestCase):

    def test_can_operate(self):
        combinations = FirstEngStartBeforeLiftoff.get_operational_combinations()
        self.assertEqual(combinations,
                         [('Eng Start', 'Engine Count', 'Liftoff'),])

    def test_derive_basic(self):
        eng_count = A('Engine Count', 3)
        eng_starts = EngStart('Eng Start',
                              items=[KeyTimeInstance(10, name='Eng (1) Start'),
                                     KeyTimeInstance(20, name='Eng (2) Start'),
                                     KeyTimeInstance(30, name='Eng (3) Start'),
                                     KeyTimeInstance(40, name='Eng (1) Start'),
                                     KeyTimeInstance(50, name='Eng (2) Start'),
                                     KeyTimeInstance(60, name='Eng (3) Start'),
                                     KeyTimeInstance(70, name='Eng (1) Start'),
                                     KeyTimeInstance(80, name='Eng (2) Start'),
                                     KeyTimeInstance(360, name='Eng (1) Start'),
                                     KeyTimeInstance(370, name='Eng (1) Start'),
                                     KeyTimeInstance(380, name='Eng (2) Start')])
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(100, 'Liftoff')])
        node = FirstEngStartBeforeLiftoff()
        node.derive(eng_starts, eng_count, liftoffs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 60)


class TestLastEngStartBeforeLiftoff(unittest.TestCase):

    def setUp(self):
        self.node_class = LastEngStartBeforeLiftoff

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertEqual(combinations,
                         [('Eng Start', 'Engine Count', 'Liftoff'),])

    def test_derive(self):
        eng_count = A('Engine Count', 3)
        eng_starts = EngStart('Eng Start',items=[
            KeyTimeInstance(10, name='Eng (1) Start'),
            KeyTimeInstance(20, name='Eng (2) Start'),
            KeyTimeInstance(30, name='Eng (3) Start'),
            KeyTimeInstance(40, name='Eng (1) Start'),
            KeyTimeInstance(50, name='Eng (2) Start'),
            KeyTimeInstance(60, name='Eng (3) Start'),
            KeyTimeInstance(70, name='Eng (1) Start'),
            KeyTimeInstance(80, name='Eng (2) Start'),
            KeyTimeInstance(360, name='Eng (1) Start'),
            KeyTimeInstance(370, name='Eng (1) Start'),
            KeyTimeInstance(380, name='Eng (2) Start'),
        ])
        liftoffs = KTI('Liftoff', items=[KeyTimeInstance(100, 'Liftoff')])
        node = self.node_class()
        node.derive(eng_starts, eng_count, liftoffs)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 80)


class TestEngStop(unittest.TestCase):

    def test_can_operate(self):
        combinations = EngStop.get_operational_combinations()
        self.assertTrue(('Eng (1) N2', 'Eng Start') in combinations)
        self.assertTrue(('Eng (2) N2', 'Eng Start') in combinations)
        self.assertTrue(('Eng (3) N2', 'Eng Start') in combinations)
        self.assertTrue(('Eng (4) N2', 'Eng Start') in combinations)
        self.assertTrue(('Eng (1) N2', 'Eng (2) N2',
                         'Eng (3) N2', 'Eng (4) N2', 'Eng Start') in combinations)

    def test_basic(self):
        eng1 = Parameter(
            'Eng (1) N2', np.ma.array(
                data=[60, 50, 40, 99, 99, 0, 0], mask=[0, 0, 0, 1, 1, 1, 1]),
            frequency=1 / 64.0)
        eng2 = Parameter(
            'Eng (2) N2', np.ma.array([60, 40, 20, 0]), frequency=1 / 64.0)
        eng_start = EngStart(name='Eng Start', items=[
            KeyTimeInstance(2, 'Engine (1) Start'),
        ])
        es = EngStop()
        es.get_derived([
            None, None, None, None,
            eng1, eng2, None, None,
            None, None, None, None,
            None, None, None, None, eng_start, None
        ])
        self.assertEqual(len(es), 2)
        self.assertEqual(es[0].name, 'Eng (1) Stop')
        self.assertEqual(es[0].index, 6)
        self.assertEqual(es[1].name, 'Eng (2) Stop')
        self.assertEqual(es[1].index, 2)

    def test_short_dip(self):
        eng_1_n3 = load(os.path.join(test_data_path, 'eng_start_eng_1_n3.nod'))
        eng_2_n3 = load(os.path.join(test_data_path, 'eng_start_eng_2_n3.nod'))
        node = EngStop()
        eng_start = EngStart(name='Eng Start')
        node.derive(
            None, None, None, None,
            None, None, None, None,
            eng_1_n3, eng_2_n3, None, None,
            None, None, None, None, eng_start, None,
        )
        self.assertEqual(len(node), 2)

    def test_basic__running_at_end_of_data(self):
        eng1 = Parameter(
            'Eng (1) N2', np.ma.array(
                data=list(range(60, 0, -10)) + [0]*5 + list(range(0, 90, 10)) + [99, 99]),
            frequency=1 / 64.0)
        eng_start = EngStart(name='Eng Start', items=[
            KeyTimeInstance(14.5*64, 'Eng (1) Start'),
        ])
        es = EngStop()
        es.get_derived([
            None, None, None, None,
            eng1, None, None, None,
            None, None, None, None,
            None, None, None, None, eng_start, None
        ])
        self.assertEqual(len(es), 2)
        self.assertEqual(es[0].name, 'Eng (1) Stop')
        self.assertEqual(es[0].index, 3)
        self.assertEqual(es[1].name, 'Eng (1) Stop')
        self.assertEqual(es[1].index, 21)

    def test_stop_at_end_of_data(self):
        eng1 = Parameter(
            'Eng (1) N2', np.ma.array(
                data=[60, 50, 40, 35, 30, 0, 0], mask=[0, 0, 0, 0, 0, 1, 1]))
        eng_start = EngStart(name='Eng Start', items=[
            KeyTimeInstance(2, 'Engine (1) Start'),
        ])
        es = EngStop()
        es.get_derived([
            None, None, None, None,
            eng1, None, None, None,
            None, None, None, None,
            None, None, None, None, eng_start, None
        ])
        self.assertEqual(len(es), 1)
        self.assertEqual(es[0].name, 'Eng (1) Stop')
        self.assertEqual(es[0].index, 4)


class TestLastEngStopAfterTouchdown(unittest.TestCase):

    def test_can_operate(self):
        combinations = LastEngStopAfterTouchdown.get_operational_combinations()
        self.assertEqual(
            combinations,
            [('Eng Stop', 'Engine Count', 'Touchdown', 'HDF Duration')])

    def test_derive_basic(self):
        eng_count = A('Engine Count', 3)
        eng_stops = EngStop('Eng Stop',
                            items=[KeyTimeInstance(10, name='Eng (1) Stop'),
                                   KeyTimeInstance(20, name='Eng (2) Stop'),
                                   KeyTimeInstance(30, name='Eng (3) Stop'),
                                   KeyTimeInstance(110, name='Eng (1) Stop'),
                                   KeyTimeInstance(120, name='Eng (2) Stop'),
                                   KeyTimeInstance(130, name='Eng (3) Stop'),
                                   KeyTimeInstance(140, name='Eng (1) Stop'),
                                   KeyTimeInstance(150, name='Eng (2) Stop'),
                                   KeyTimeInstance(160, name='Eng (1) Stop'),
                                   KeyTimeInstance(170, name='Eng (1) Stop'),
                                   KeyTimeInstance(180, name='Eng (2) Stop'),])
        touchdowns = KTI('Touchdown', items=[KeyTimeInstance(100, 'Touchdown')])
        node = LastEngStopAfterTouchdown()
        node.derive(eng_stops, eng_count, touchdowns)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 130)


class TestEnterHold(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Holding',)]
        self.assertEqual(expected, EnterHold.get_operational_combinations())

    def test_derive(self):
        hold = buildsection('Holding', 2, 5)
        expected = [KeyTimeInstance(index=2, name='Enter Hold')]
        eh = EnterHold()
        eh.derive(hold)
        self.assertEqual(eh, expected)


class TestExitHold(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Holding',)]
        self.assertEqual(expected, ExitHold.get_operational_combinations())

    def test_derive(self):
        hold = buildsection('Holding', 2, 5)
        expected = [KeyTimeInstance(index=2, name='Enter Hold')]
        eh = EnterHold()
        eh.derive(hold)
        self.assertEqual(eh, expected)


class TestAPUStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('APU Running',)]
        self.assertEqual(expected, APUStart.get_operational_combinations())

    def test_derive(self):
        array = ['-'] * 3 + ['Running'] * 2 + ['-'] * 2
        mapping = {0: '-', 1: 'Running'}
        apu = M('APU Running', array, values_mapping=mapping)
        astart = APUStart()
        astart.derive(apu)
        self.assertEqual(len(astart), 1)
        self.assertEqual(astart[0].index, 2.5)

    def test_derive_at_start(self):
        # APURunning at the beginning of the data, should be detected
        array = ['Running'] * 2 + ['-'] * 5
        mapping = {0: '-', 1: 'Running'}
        apu = M('APU Running', array, values_mapping=mapping)
        astart = APUStart()
        astart.derive(apu)
        self.assertEqual(len(astart), 1)
        self.assertEqual(astart[0].index, 0)
        # masked first value
        apu.array[0] = np.ma.masked
        astart = APUStart()
        astart.derive(apu)
        self.assertEqual(len(astart), 1)
        self.assertEqual(astart[0].index, 1)


class TestAPUStop(unittest.TestCase):
    def test_can_operate(self):
        expected = [('APU Running',)]
        self.assertEqual(expected, APUStop.get_operational_combinations())

    def test_derive(self):
        array = ['-'] * 3 + ['Running'] * 2 + ['-'] * 2
        mapping = {0: '-', 1: 'Running'}
        apu = M('APU Running', array, values_mapping=mapping)
        astop = APUStop()
        astop.derive(apu)
        self.assertEqual(len(astop), 1)
        self.assertEqual(astop[0].index, 4.5)

    def test_derive_at_end(self):
        # APURunning at the end of the data, should be detected
        array = ['-'] * 5 + ['Running'] * 2
        mapping = {0: '-', 1: 'Running'}
        apu = M('APU Running', array, values_mapping=mapping)
        astop = APUStop()
        astop.derive(apu)
        self.assertEqual(len(astop), 1)
        self.assertEqual(astop[0].index, 6)
        # masked first value
        apu.array[6] = np.ma.masked
        astop = APUStop()
        astop.derive(apu)
        self.assertEqual(len(astop), 1)
        self.assertEqual(astop[0].index, 5)


##############################################################################
# Flap & Slat


class TestFlapLoadReliefSet(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapLoadReliefSet
        self.operational_combinations = [('Flap Load Relief',)]

    def test_derive(self):
        array = ['Normal'] * 3 + ['Load Relief'] * 2 + ['Normal'] * 2
        mapping = {0: 'Normal', 1: 'Load Relief'}
        flr = M('Flap Load Relief', array, values_mapping=mapping)
        node = self.node_class()
        node.derive(flr)
        expected = [KeyTimeInstance(index=2.5, name=self.node_class.get_name())]
        self.assertEqual(node, expected)


class TestFlapAlternateArmed(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapAlternateArmedSet
        self.operational_combinations = [('Flap Alternate Armed',)]

    def test_derive(self):
        array = ['-'] * 3 + ['Armed'] * 3 + ['-'] * 1
        mapping = {0: '-', 1: 'Armed'}
        faa = M('Flap Alternate Armed', array, values_mapping=mapping)
        node = self.node_class()
        node.derive(faa)
        expected = [KeyTimeInstance(index=2.5, name=self.node_class.get_name())]
        self.assertEqual(node, expected)


class TestSlatAlternateArmedSet(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = SlatAlternateArmedSet
        self.operational_combinations = [('Slat Alternate Armed',)]

    def test_derive(self):
        array = ['-'] * 2 + ['Armed'] * 3 + ['-'] * 2
        mapping = {0: '-', 1: 'Armed'}
        saa = M('Slat Alternate Armed', array, values_mapping=mapping)
        node = self.node_class()
        node.derive(saa)
        expected = [KeyTimeInstance(index=1.5, name=self.node_class.get_name())]
        self.assertEqual(node, expected)


class TestFlapLeverSet(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapLeverSet
        # Testing decimal flap lever 17.5 not available in public model information.
        self.node_class.NAME_VALUES['flap'].append('17.5')
        self.operational_combinations = [
            ('Flap Lever',),
            ('Flap Lever (Synthetic)',),
            ('Flap Lever', 'Flap Lever (Synthetic)'),
        ]
        array = np.ma.array((0, 0, 5, 5, 10, 10, 15, 10, 10, 5, 5, 0, 0, 17.5))
        mapping = {f0: str(f0) for f0 in (int(f1)
                   if float(f1).is_integer() else f1 for f1 in np.ma.unique(array))}
        self.flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        array = np.ma.array((1, 1, 1, 5, 5, 10, 10, 15, 10, 10, 5, 5, 1, 1))
        mapping = {f0: 'Lever %s' % i for i, f0 in enumerate(int(f1)
                   if float(f1).is_integer() else f1 for f1 in np.ma.unique(array))}
        self.flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)

    def test_derive_basic(self):
        name = self.node_class.get_name()
        node = self.node_class()

        node.derive(self.flap_lever, None)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=10.5, name='Flap 0 Set'),
            KeyTimeInstance(index=1.5, name='Flap 5 Set'),
            KeyTimeInstance(index=8.5, name='Flap 5 Set'),
            KeyTimeInstance(index=3.5, name='Flap 10 Set'),
            KeyTimeInstance(index=6.5, name='Flap 10 Set'),
            KeyTimeInstance(index=5.5, name='Flap 15 Set'),
            KeyTimeInstance(index=12.5, name='Flap 17.5 Set'),
        ]))
        node = self.node_class()
        node.derive(None, self.flap_synth)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=11.5, name='Flap Lever 0 Set'),
            KeyTimeInstance(index=2.5, name='Flap Lever 1 Set'),
            KeyTimeInstance(index=9.5, name='Flap Lever 1 Set'),
            KeyTimeInstance(index=4.5, name='Flap Lever 2 Set'),
            KeyTimeInstance(index=7.5, name='Flap Lever 2 Set'),
            KeyTimeInstance(index=6.5, name='Flap Lever 3 Set'),
        ]))
        node = self.node_class()
        node.derive(self.flap_lever, self.flap_synth)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=10.5, name='Flap 0 Set'),
            KeyTimeInstance(index=1.5, name='Flap 5 Set'),
            KeyTimeInstance(index=8.5, name='Flap 5 Set'),
            KeyTimeInstance(index=3.5, name='Flap 10 Set'),
            KeyTimeInstance(index=6.5, name='Flap 10 Set'),
            KeyTimeInstance(index=5.5, name='Flap 15 Set'),
            KeyTimeInstance(index=12.5, name='Flap 17.5 Set'),
        ]))


class TestFlapExtensionWhileAirborne(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapExtensionWhileAirborne
        self.operational_combinations = [
            ('Flap Lever', 'Airborne'),
            ('Flap Lever (Synthetic)', 'Airborne'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Airborne'),
        ]
        array = np.ma.array((0, 0, 5, 5, 10, 10, 15, 10, 10, 5, 5, 0, 0))
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        self.flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        array = np.ma.array((1, 1, 1, 5, 5, 10, 10, 15, 10, 10, 5, 5, 1))
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        self.flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)

    def test_derive(self):
        airborne = buildsection('Airborne', 1, 12)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.flap_lever, None, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=1.5, name=name),
            KeyTimeInstance(index=3.5, name=name),
            KeyTimeInstance(index=5.5, name=name),
        ]))
        node = self.node_class()
        node.derive(None, self.flap_synth, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=2.5, name=name),
            KeyTimeInstance(index=4.5, name=name),
            KeyTimeInstance(index=6.5, name=name),
        ]))
        node = self.node_class()
        node.derive(self.flap_lever, self.flap_synth, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=1.5, name=name),
            KeyTimeInstance(index=3.5, name=name),
            KeyTimeInstance(index=5.5, name=name),
        ]))


class TestEngFireExtinguisher(unittest.TestCase):

    def test_basic(self):
        e1f = P(name = 'Eng (1) Fire Extinguisher',
                array = np.ma.array(data=[0,0,0,0,0,0,1,0,0,0]),
                frequency=1, offset=0,)
        e2f = P(name = 'Eng (2) Fire Extinguisher',
                array = np.ma.array([0]*10),
                frequency=1, offset=0,)
        air = buildsection('Airborne', 2, 8)
        pull = EngFireExtinguisher()
        pull.derive(e1f, e2f, air)
        self.assertEqual(pull, [
            KeyTimeInstance(index=6, name='Eng Fire Extinguisher'),
            ])

    def test_none(self):
        e1f = P(name = 'Eng (1) Fire Extinguisher',
                array = np.ma.array(data=[0,0,0,0,0,0,0,0,0,0]),
                frequency=1, offset=0,)
        e2f = P(name = 'Eng (2) Fire Extinguisher',
                array = np.ma.array([0]*10),
                frequency=1, offset=0,)
        air = buildsection('Airborne', 2, 8)
        pull = EngFireExtinguisher()
        pull.derive(e1f, e2f, air)
        self.assertEqual(pull, [])

    def test_either(self):
        e2f = P(name = 'Eng (2) Fire Extinguisher',
                array = np.ma.array(data=[0,0,0,0,0,1,1,1,0,0]),
                frequency=1, offset=0,)
        e1f = P(name = 'Eng (1) Fire Extinguisher',
                array = np.ma.array([0]*10),
                frequency=1, offset=0,)
        air = buildsection('Airborne', 2, 8)
        pull = EngFireExtinguisher()
        pull.derive(e1f, e2f, air)
        self.assertEqual(pull, [
            KeyTimeInstance(index=5, name='Eng Fire Extinguisher'),
            ])

    def test_both(self):
        e1f = P(name = 'Eng (1) Fire Extinguisher',
                array = np.ma.array(data=[0,0,0,1,0,1,1,1,0,0]),
                frequency=1, offset=0,)
        e2f = P(name = 'Eng (2) Fire Extinguisher',
                array = np.ma.array(data=[0,0,0,1,1,1,1,1,0,0]),
                frequency=1, offset=0,)
        air = buildsection('Airborne', 1, 5)
        pull = EngFireExtinguisher()
        pull.derive(e1f, e2f, air)
        self.assertEqual(pull, [
            KeyTimeInstance(index=3, name='Eng Fire Extinguisher'),
            ])


class TestFirstFlapExtensionWhileAirborne(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FirstFlapExtensionWhileAirborne
        self.operational_combinations = [
            ('Flap Lever', 'Airborne'),
            ('Flap Lever (Synthetic)', 'Airborne'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Airborne'),
        ]
        array = np.ma.array([0, 0, 5, 5, 10, 10, 15, 10, 10, 5, 5, 0, 0])
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        self.flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        array = np.ma.array([1, 1, 1, 5, 10, 10, 15, 10, 10, 5, 5, 1, 1])
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        self.flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)

    def test_derive(self):
        airborne = buildsection('Airborne', 1, 12)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.flap_lever, None, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=1.5, name=name),
        ]))
        node = self.node_class()
        node.derive(None, self.flap_synth, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=2.5, name=name),
        ]))
        node = self.node_class()
        node.derive(self.flap_lever, self.flap_synth, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=1.5, name=name),
        ]))

    def test_corrupt_flap_signal(self):

        array = np.ma.array(data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0],
                            mask=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        mapping = {int(f): str(f) for f in np.ma.unique(array.data)}
        self.flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        airborne = buildsection('Airborne', 1, 12)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.flap_lever, None, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=8.5, name=name),
        ]))


class TestFlapRetractionWhileAirborne(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapRetractionWhileAirborne
        self.operational_combinations = [
            ('Flap Lever', 'Airborne'),
            ('Flap Lever (Synthetic)', 'Airborne'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Airborne'),
        ]
        array = np.ma.array((0, 0, 5, 5, 10, 10, 15, 10, 10, 5, 5, 0, 0))
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        self.flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        array = np.ma.array((1, 1, 1, 5, 5, 10, 10, 15, 10, 10, 5, 5, 1))
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        self.flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)

    def test_derive(self):
        airborne = buildsection('Airborne', 2, 11)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.flap_lever, None, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=6.5, name=name),
            KeyTimeInstance(index=8.5, name=name),
            KeyTimeInstance(index=10.5, name=name),
        ]))
        node = self.node_class()
        node.derive(None, self.flap_synth, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=7.5, name=name),
            KeyTimeInstance(index=9.5, name=name),
        ]))
        node = self.node_class()
        node.derive(self.flap_lever, self.flap_synth, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=6.5, name=name),
            KeyTimeInstance(index=8.5, name=name),
            KeyTimeInstance(index=10.5, name=name),
        ]))


class TestFlapRetractionDuringGoAround(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = FlapRetractionDuringGoAround
        self.operational_combinations = [
            ('Flap Lever', 'Go Around And Climbout'),
            ('Flap Lever (Synthetic)', 'Go Around And Climbout'),
            ('Flap Lever', 'Flap Lever (Synthetic)', 'Go Around And Climbout'),
        ]
        array = np.ma.array((0, 0, 5, 5, 10, 10, 15, 10, 10, 5, 5, 0, 0))
        mapping = {int(f): str(f) for f in np.ma.unique(array)}
        self.flap_lever = M(name='Flap Lever', array=array, values_mapping=mapping)
        array = np.ma.array((1, 1, 1, 5, 5, 10, 10, 15, 10, 10, 5, 5, 1))
        mapping = {int(f): 'Lever %s' % i for i, f in enumerate(np.ma.unique(array))}
        self.flap_synth = M(name='Flap Lever (Synthetic)', array=array, values_mapping=mapping)

    def test_derive(self):
        airborne = buildsection('Go Around', 2, 11)
        name = self.node_class.get_name()
        node = self.node_class()
        node.derive(self.flap_lever, None, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=6.5, name=name),
            KeyTimeInstance(index=8.5, name=name),
            KeyTimeInstance(index=10.5, name=name),
        ]))
        node = self.node_class()
        node.derive(None, self.flap_synth, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=7.5, name=name),
            KeyTimeInstance(index=9.5, name=name),
        ]))
        node = self.node_class()
        node.derive(self.flap_lever, self.flap_synth, airborne)
        self.assertEqual(node, KTI(name=name, items=[
            KeyTimeInstance(index=6.5, name=name),
            KeyTimeInstance(index=8.5, name=name),
            KeyTimeInstance(index=10.5, name=name),
        ]))


##############################################################################
# Gear


class TestGearDownSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GearDownSelection
        self.operational_combinations = [('Gear Down Selected', 'Airborne')]
        self.gear_dn_sel = M(
            name='Gear Down Selected',
            array=['Down'] * 3 + ['Up'] * 2 + ['Down'] * 2,
            values_mapping={0: 'Up', 1: 'Down'},
        )
        self.airborne = buildsection('Airborne', 0, 7)

    def test_derive(self):
        node = GearDownSelection()
        node.derive(self.gear_dn_sel, self.airborne)
        self.assertEqual(node, [
            KeyTimeInstance(index=4.5, name='Gear Down Selection'),
        ])


class TestGearUpSelection(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GearUpSelection
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.operational_combinations = [('Gear Up Selected', 'Airborne', 'Go Around And Climbout')]
        self.gear_up_sel = M(
            name='Gear Up Selected',
            array=['Down'] * 3 + ['Up'] * 2 + ['Down'] * 2,
            values_mapping={0: 'Down', 1: 'Up'},
        )
        self.airborne = buildsection('Airborne', 0, 7)

    def can_operate_helicopter(self):
        operational_combinations = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(operational_combinations, [('Gear Up Selected', 'Airborne')])

    def test_normal_operation(self):
        go_arounds = buildsection('Go Around And Climbout', 6, 7)
        node = GearUpSelection()
        node.derive(self.gear_up_sel, self.airborne, go_arounds)
        self.assertEqual(node, [
            KeyTimeInstance(index=2.5, name='Gear Up Selection'),
        ])

    def test_during_go_around(self):
        go_arounds = buildsection('Go Around And Climbout', 2, 4)
        node = GearUpSelection()
        node.derive(self.gear_up_sel, self.airborne, go_arounds)
        self.assertEqual(node, [])

    def test_low_hz(self):
        '''
        Gear Up Selection was not being triggered due to using
        the slice index 119 instead of the start_edge 118.0625.
        '''
        gear_up_sel = load(os.path.join(test_data_path,
                                        'GearUpSelection_gear_up_sel_1.nod'))
        airborne = buildsection('Airborne', 119, 1381, 118.0625, 1380.4375)
        go_arounds = S('Go Around')
        node = GearUpSelection()
        node.derive(gear_up_sel, airborne, go_arounds)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 118.5)

    def test_airborne_start_index(self):
        array = load_compressed(os.path.join(test_data_path, 'GearUpSelection_gear_up_sel_2.npz'))
        gear_up_sel = M('Gear Up Selected', array)
        airborne = buildsection('Airborne', 497, 6808, 496.6328125, 6807.6328125)
        go_arounds = S('Go Around')
        node = GearUpSelection()
        node.derive(gear_up_sel, airborne, go_arounds)
        self.assertEqual(len(node), 1)
        # The calculated KTI index is shifted forward slightly to match the slice start index.
        # This fixes a discrepancy whereby Gear Up Selection occurs just before Liftoff.
        self.assertEqual(node[0].index, 496.6328125)


class TestGearUpSelectionDuringGoAround(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = GearUpSelectionDuringGoAround
        self.operational_combinations = [('Gear Up Selected', 'Go Around And Climbout')]
        self.gear_up_sel = M(
            name='Gear Up Selected',
            array=['Down'] * 3 + ['Up'] * 2 + ['Down'] * 2,
            values_mapping={0: 'Down', 1: 'Up'},
        )

    def test_normal_operation(self):
        go_arounds = buildsection('Go Around And Climbout', 6, 7)
        node = GearUpSelectionDuringGoAround()
        node.derive(self.gear_up_sel, go_arounds)
        self.assertEqual(node, [])

    def test_during_go_around(self):
        go_arounds = buildsection('Go Around And Climbout', 2, 4)
        node = GearUpSelectionDuringGoAround()
        node.derive(self.gear_up_sel, go_arounds)
        self.assertEqual(node, [
            KeyTimeInstance(index=2.5, name='Gear Up Selection During Go Around'),
        ])


##############################################################################


class TestLocalizerEstablishedEnd(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer Established',)]
        self.assertEqual(
            expected,
            LocalizerEstablishedEnd.get_operational_combinations())

    def test_derive(self):
        ils = buildsection('ILS Localizer Established', 10, 19)
        expected = [
            KeyTimeInstance(index=20, name='Localizer Established End')]
        les = LocalizerEstablishedEnd()
        les.derive(ils)
        self.assertEqual(les, expected)


class TestLocalizerEstablishedStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('ILS Localizer Established',)]
        self.assertEqual(
            expected,
            LocalizerEstablishedStart.get_operational_combinations())

    def test_derive(self):
        ils = buildsection('ILS Localizer Established', 10, 20)
        expected = [
            KeyTimeInstance(index=10, name='Localizer Established Start')]
        les = LocalizerEstablishedStart()
        les.derive(ils)
        self.assertEqual(les, expected)


class TestLowestAltitudeDuringApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LowestAltitudeDuringApproach
        self.operational_combinations = [('Altitude AAL', 'Altitude Radio', 'Approach And Landing')]

    def test_derive(self):
        alt_aal = P(
            name='Altitude AAL',
            array=np.ma.array([5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        )
        alt_rad = P(
            name='Altitude Radio',
            array=np.ma.array([5, 5, 4, 4, 3, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        )
        approaches = buildsection('Approach And Landing', 2, 15)
        node = self.node_class()
        node.derive(alt_aal, alt_rad, approaches)
        self.assertEqual(node, [
            KeyTimeInstance(index=7, name='Lowest Altitude During Approach'),
        ])


class TestMinsToTouchdown(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Touchdown', 'Liftoff')]
        self.assertEqual(
            expected,
            MinsToTouchdown.get_operational_combinations())

    def test_derive(self):
        td = [KeyTimeInstance(index=500, name='Touchdown')]
        lo = KTI('Liftoff', items=[KeyTimeInstance(10, 'Liftoff')])
        sttd = MinsToTouchdown()
        sttd.derive(td, lo)
        self.assertEqual(
            sttd,
            [
                KeyTimeInstance(index=200, name='5 Mins To Touchdown'),
                KeyTimeInstance(index=260, name='4 Mins To Touchdown'),
                KeyTimeInstance(index=320, name='3 Mins To Touchdown'),
                KeyTimeInstance(index=380, name='2 Mins To Touchdown'),
                KeyTimeInstance(index=440, name='1 Mins To Touchdown'),
            ]
        )

    def test_overlap(self):
        td = [KeyTimeInstance(index=500, name='Touchdown')]
        lo = KTI('Liftoff', items=[KeyTimeInstance(300, 'Liftoff')])
        sttd = MinsToTouchdown()
        sttd.derive(td, lo)
        self.assertEqual(len(sttd), 3)


class TestSecsToTouchdown(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Touchdown', 'Liftoff')]
        self.assertEqual(
            expected,
            SecsToTouchdown.get_operational_combinations())

    def test_derive(self):
        td = [KeyTimeInstance(index=100, name='Touchdown')]
        lo = KTI('Liftoff', items=[KeyTimeInstance(1, 'Liftoff')])
        sttd = SecsToTouchdown()
        sttd.derive(td, lo)
        self.assertEqual(len(sttd), 3)
        self.assertEqual(
            sttd,
            [
                KeyTimeInstance(index=10, name='90 Secs To Touchdown'),
                KeyTimeInstance(index=70, name='30 Secs To Touchdown'),
                KeyTimeInstance(index=80, name='20 Secs To Touchdown'),
            ]
        )

    def test_overlap(self):
        td = [KeyTimeInstance(index=100, name='Touchdown')]
        lo = KTI('Liftoff', items=[KeyTimeInstance(30, 'Liftoff')])
        sttd = SecsToTouchdown()
        sttd.derive(td, lo)
        self.assertEqual(len(sttd), 2)
        self.assertEqual(
            sttd,
            [
                KeyTimeInstance(index=70, name='30 Secs To Touchdown'),
                KeyTimeInstance(index=80, name='20 Secs To Touchdown'),
            ]
        )


class TestDistanceToTouchdown(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(DistanceToTouchdown.get_operational_combinations(),
                         [('Distance To Landing', 'Touchdown')])

    def test_derive(self):
        td = [KeyTimeInstance(index=49, name='Touchdown')]
        dtl=P('Distance To Landing', np.ma.array(np.linspace(5.0, 0.0)))
        dtt = DistanceToTouchdown()
        dtt.derive(dtl, td)
        self.assertEqual(len(dtt), 4)
        self.assertAlmostEqual(dtt[0].index, 41.2, places=1)
        self.assertEqual(dtt[0].name, '0.8 NM To Touchdown')
        self.assertAlmostEqual(dtt[1].index, 39.2, places=1)
        self.assertEqual(dtt[1].name, '1.0 NM To Touchdown')
        self.assertAlmostEqual(dtt[2].index, 34.3, places=1)
        self.assertEqual(dtt[2].name, '1.5 NM To Touchdown')
        self.assertAlmostEqual(dtt[3].index, 29.4, places=1)
        self.assertEqual(dtt[3].name, '2.0 NM To Touchdown')

    def test_derive__multiple_touchdowns(self):
        touchdown_name = 'Touchdown'
        td = KTI(touchdown_name,
                 items=[
                     KeyTimeInstance(index=50.51, name=touchdown_name),
                     KeyTimeInstance(index=101.53, name=touchdown_name)
                     ]
                 )
        dtl=P('Distance To Landing', np.ma.concatenate((np.arange(10, -0.2, -0.2), np.arange(10, -0.2, -0.2), np.arange(0.2, 2, 0.2))))
        dtt = DistanceToTouchdown()
        dtt.derive(dtl, td)
        self.assertEqual(len(dtt), 8)
        self.assertAlmostEqual(dtt[0].index, 46.0, places=1)
        self.assertEqual(dtt[0].name, '0.8 NM To Touchdown')
        self.assertAlmostEqual(dtt[1].index, 45.0, places=1)
        self.assertEqual(dtt[1].name, '1.0 NM To Touchdown')
        self.assertAlmostEqual(dtt[2].index, 42.5, places=1)
        self.assertEqual(dtt[2].name, '1.5 NM To Touchdown')
        self.assertAlmostEqual(dtt[3].index, 40.0, places=1)
        self.assertEqual(dtt[3].name, '2.0 NM To Touchdown')
        self.assertAlmostEqual(dtt[4].index, 97.0, places=1)
        self.assertEqual(dtt[4].name, '0.8 NM To Touchdown')
        self.assertAlmostEqual(dtt[5].index, 96.0, places=1)
        self.assertEqual(dtt[5].name, '1.0 NM To Touchdown')
        self.assertAlmostEqual(dtt[6].index, 93.5, places=1)
        self.assertEqual(dtt[6].name, '1.5 NM To Touchdown')
        self.assertAlmostEqual(dtt[7].index, 91.0, places=1)
        self.assertEqual(dtt[7].name, '2.0 NM To Touchdown')


class TestDistanceFromTakeoffAirport(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            DistanceFromTakeoffAirport.get_operational_combinations(),
            [('Longitude Smoothed', 'Latitude Smoothed', 'Airborne', 'FDR Takeoff Airport')])

    def test_derive(self):
        apt = A(name='FDR Takeoff Airport', value={'latitude': 0.0, 'longitude': 0.0})
        airs = buildsection('Airborne', 0, 9000)
        # 300 NM at 30 sec per NM
        test = np.ma.arange(9030)
        lat = P('Latitude', [0.0]*(len(test)))
        lon = P('Longitude', (test-30)/(30.0*60))
        dfta = DistanceFromTakeoffAirport()
        dfta.derive(lon, lat, airs, apt)

        self.assertEqual(len(dfta), 2)
        self.assertAlmostEqual(dfta[0].index, 4527, places=0)
        self.assertAlmostEqual(dfta[1].index, 7525, places=0)

    def test_first_occurrance(self):
        apt = A(name='FDR Takeoff Airport', value={'latitude': 0.0, 'longitude': 0.0})
        airs = buildsection('Airborne', 0, 9000)
        # 300 NM at 30 sec per NM
        dist = list(range(6000,4000,-1))+list(range(4000,6000))+list(range(6000,-30,-1))
        test = np.ma.array(dist[::-1]) # Copied from landing case  :o)
        lat = P('Latitude', [0.0]*(len(test)))
        lon = P('Longitude', test/(30.0*60))
        dfta = DistanceFromTakeoffAirport()
        dfta.derive(lon, lat, airs, apt)
        self.assertEqual(len(dfta), 1)
        self.assertAlmostEqual(dfta[0].index, 4526, places=0)


    def test_masked(self):
        apt = A(name='FDR Takeoff Airport', value={'latitude': 0.0, 'longitude': 0.0})
        airs = buildsection('Airborne', 0, 9000)
        # 300 NM at 30 sec per NM
        dist = list(range(6000,4000,-1))+list(range(4000,6000))+list(range(6000,-30,-1))
        test = np.ma.array(dist[::-1]) # Copied from landing case  :o)
        lat = P('Latitude', [0.0]*(len(test)))
        lat.array[4520:4530] = np.ma.masked
        lon = P('Longitude', test/(30.0*60))
        lat.array[4500:4530] = np.ma.masked
        dfta = DistanceFromTakeoffAirport()
        dfta.derive(lon, lat, airs, apt)
        self.assertEqual(len(dfta), 1)
        self.assertAlmostEqual(dfta[0].index, 4526, places=0)


class TestDistanceFromLandingAirport(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            DistanceFromLandingAirport.get_operational_combinations(),
            [('Longitude Smoothed', 'Latitude Smoothed', 'Airborne', 'FDR Landing Airport')])

    def test_derive(self):
        apt = A(name='FDR Landing Airport', value={'latitude': 0.0, 'longitude': 0.0})
        airs = buildsection('Airborne', 0, 9000)
        # 300 NM at 30 sec per NM
        test = np.ma.arange(9030, -1, -1)
        lat = P('Latitude', [0.0]*(len(test))) # On equator
        lon = P('Longitude', (test-30)/(30.0*60))
        dfla = DistanceFromLandingAirport()
        dfla.derive(lon, lat, airs, apt)

        self.assertEqual(len(dfla), 2)
        self.assertAlmostEqual(dfla[0].index, 4503, places=0)
        self.assertAlmostEqual(dfla[1].index, 1505, places=0)

    def test_first_occurrance(self):
        apt = A(name='FDR Landing Airport', value={'latitude': 0.0, 'longitude': 0.0})
        airs = buildsection('Airborne', 0, 9000)
        # 300 NM at 30 sec per NM
        dist = list(range(6000,4000,-1))+list(range(4000,6000))+list(range(6000,-30,-1))
        test = np.ma.array(dist)
        lat = P('Latitude', [0.0]*(len(test))) # On equator
        lon = P('Longitude', test/(30.0*60))
        dfla = DistanceFromLandingAirport()
        dfla.derive(lon, lat, airs, apt)
        self.assertEqual(len(dfla), 1)
        self.assertAlmostEqual(dfla[0].index, 1503, places=0)


    def test_masked(self):
        apt = A(name='FDR Landing Airport', value={'latitude': 0.0, 'longitude': 0.0})
        airs = buildsection('Airborne', 0, 9000)
        # 300 NM at 30 sec per NM
        dist = list(range(6000,4000,-1))+list(range(4000,6000))+list(range(6000,-30,-1))
        test = np.ma.array(dist)
        lat = P('Latitude', [0.0]*(len(test))) # On equator
        # mask a few samples
        lat.array[1500:1510] = np.ma.masked
        lon = P('Longitude', test/(30.0*60))
        lon.array[1490:1505] = np.ma.masked
        dfla = DistanceFromLandingAirport()
        dfla.derive(lon, lat, airs, apt)
        self.assertEqual(len(dfla), 1)
        self.assertAlmostEqual(dfla[0].index, 1503, places=0)


class TestDistanceFromThreshold(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            DistanceFromThreshold.get_operational_combinations(),
            [('Longitude Smoothed', 'Latitude Smoothed', 'Airborne', 'FDR Landing Runway')])

    def test_derive(self):
        # The runway threshold is offset a little from the equator
        # as the aircraft will never pass perfectly over the threshold.
        rwy = A(name='FDR Landing Runway', value={
            'start': {'latitude': 0.0, 'longitude': 0.0001},
            'end': {'latitude': 0.0, 'longitude': -0.03},
        })
        airs = buildsection('Airborne', 0, 99)
        test = np.ma.arange(128, -1, -1)
        # 3 NM to threshold in first 96 samples
        lat = P('Latitude', np.ma.zeros(len(test), dtype=np.float)) # On equator
        lon = P('Longitude', (test - 32) / (32.0 * 60))
        #lon.array = np.ma.concatenate([lon.array])
        dft = DistanceFromThreshold()
        dft.derive(lon, lat, airs, rwy)

        self.assertEqual(dft[0].index, 96)
        self.assertAlmostEqual(dft[1].index, 64, places=0)
        self.assertAlmostEqual(dft[2].index,32, places=0)

    def test_derive_beyond_array(self):
        rwy = A(name='FDR Landing Runway', value={
            'start': {'latitude': 0, 'longitude': 0},
            'end': {'latitude': 0, 'longitude': -0.03},
        })
        airs = buildsection('Airborne', 0, 99)
        test = np.ma.arange(128, -1, -1)
        # 1.5 NM to zero in first 96 samples
        lat = P('Latitude', np.ma.zeros(len(test), dtype=np.float)) # On equator
        lon = P('Longitude', (test - 32) / (64.0 * 60))
        dft = DistanceFromThreshold()
        dft.derive(lon, lat, airs, rwy)

        self.assertEqual(dft[0].index, 96)
        # Note shifted 1nm point as flying at half the speed
        self.assertAlmostEqual(dft[1].index, 32, places=0)
        # Array is too short to include 2nm point
        self.assertRaises(ValueError)

    def test_derive_close_to_landing(self):
        # An error in one version detected points closer to takeoff where the
        # aircraft flew a circuit. This tests that the last points are found.
        rwy = A(name='FDR Landing Runway', value={
            'start': {'latitude': 0, 'longitude': 0},
            'end': {'latitude': 0, 'longitude': -0.03},
        })
        airs = buildsection('Airborne', 30, 230)
        test = np.ma.array(list(range(0,129))+list(range(128,-1,-1)))
        # 3 NM to zero in 96 samples
        lat = P('Latitude', [0.0]*(len(test))) # On equator
        lon = P('Longitude', (test-32)/(32.0*60))
        dft = DistanceFromThreshold()
        dft.derive(lon, lat, airs, rwy)

        self.assertEqual(dft[0].index, 225)
        self.assertAlmostEqual(dft[1].index, 193, places=0)
        self.assertAlmostEqual(dft[2].index, 161, places=0)

class TestAutoland(unittest.TestCase):
    def test_can_operate(self):
        expected = [('AP Channels Engaged', 'Touchdown'),
                    ('AP Channels Engaged', 'Touchdown', 'Family')]
        opts = Autoland.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_derive_autoland_dual(self):
        # test with no family
        td = [KeyTimeInstance(index=5, name='Touchdown')]
        ap = M('AP Channels Engaged',
               array=['-', '-', '-', 'Dual', 'Dual', 'Dual', '-', '-', '-'],
               values_mapping={0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'})
        node = Autoland()
        node.derive(ap, td, None)
        expected = [KeyTimeInstance(index=5, name='Autoland')]
        self.assertEqual(node, expected)
        # test with B737 Classic creates no autoland as requires Triple mode
        node = Autoland()
        node.derive(ap, td, A('Family', 'B737 Classic'))
        self.assertEqual(node, [])

    def test_derive_autoland(self):
        # simulate each of the states at touchdown using the indexes
        td = [KeyTimeInstance(index=3, name='Touchdown'),
              KeyTimeInstance(index=4, name='Touchdown'),
              KeyTimeInstance(index=5, name='Touchdown'),
              KeyTimeInstance(index=6, name='Touchdown')]
        ap = M('AP Channels Engaged',
               array=['-', '-', '-', 'Triple', 'Dual', 'Single', '-', '-', '-'],
               values_mapping={0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'})
        # test with no family
        node = Autoland()
        node.derive(ap, td, None)
        self.assertEqual([n.index for n in node], [3, 4])
        # test with no A330
        node = Autoland()
        node.derive(ap, td, A('Family', 'A330'))
        self.assertEqual([n.index for n in node], [3, 4])
        # test with no A330
        node = Autoland()
        node.derive(ap, td, A('Family', 'B757'))
        self.assertEqual([n.index for n in node], [3])


class TestTouchAndGo(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TouchAndGo
        self.operational_combinations = [
            ('Altitude AAL', 'Go Around And Climbout'),
            ('Altitude AAL', 'Go Around And Climbout', 'Gear On Ground'),
        ]

    def test_derive_one_go_around_and_climbout(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.array([
            5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        ]))

        go_around_and_climbout = buildsection('Go Around And Climbout', 7, 13)

        node = self.node_class()
        node.derive(alt_aal, go_around_and_climbout, None)

        expected = [KeyTimeInstance(index=10, name='Touch And Go')]
        self.assertEqual(node, expected)

    def test_derive_multiple_go_around_and_climbout(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.array([
            5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            8, 7, 6, 5, 4, 3, 2, 1, 0, 1,
            2, 3, 4, 5, 6]))

        go_around_and_climbout = buildsections('Go Around And Climbout', (7, 13), (25, 30))

        node = self.node_class()
        node.derive(alt_aal, go_around_and_climbout, None)

        expected = [KeyTimeInstance(index=10, name='Touch And Go'), KeyTimeInstance(index=28, name='Touch And Go')]
        self.assertEqual(node, expected)

    def test_derive_with_gear_on_ground(self):
        alt_aal = P(name='Altitude AAL', array=np.ma.array([
            5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
            2, 2, 2, 3, 4, 5, 6, 7, 8, 9,
        ]))

        go_around_and_climbout = buildsection('Go Around And Climbout', 7, 13)
        gog = P(name='Gear On Ground', array=np.ma.array(['Air'] * 10 + ['Ground'] + ['Air'] * 9))

        node = self.node_class()
        node.derive(alt_aal, go_around_and_climbout, gog)

        expected = [KeyTimeInstance(index=10, name='Touch And Go')]
        self.assertEqual(node, expected)


class TestTransmit(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Transmit
        self.operational_combinations = [('Transmitting',)]

    def test_derive(self):
        transmitting = M('Transmitting', MappedArray([0, 0, 0, 1, 0, 0, 0],
                                                     values_mapping={0: '-', 1: 'Transmit'}))
        node = self.node_class()
        node.derive(transmitting)
        expected = [KeyTimeInstance(index=2.5, name='Transmit')]
        self.assertEqual(node, expected)


class TestMovementStart(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MovementStart
        self.operational_combinations = [('Stationary',)]

    def test_derive(self):
        stationary = buildsections('Stationary', [0, 5], [10, 15])
        node = self.node_class()
        node.derive(stationary)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 6)
        self.assertEqual(node[1].index, 16)


class TestMovementStop(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = MovementStop
        self.operational_combinations = [('Stationary',)]

    def test_derive(self):
        stationary = buildsections('Stationary', [0, 5], [10, 15])
        node_class = MovementStop()
        node_class.derive(stationary)
        self.assertEqual(len(node_class), 1)
        self.assertEqual(node_class[0].index, 10)


class TestOffBlocks(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = OffBlocks
        self.operational_combinations = [('Mobile',)]

    def test_basic(self):
        mobile = buildsections('Mobile', [5, 10], [15, None])
        node = self.node_class()
        node.derive(mobile)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 5)

    def test_none_start(self):
        mobile = buildsection('Mobile', None, 10)
        node = self.node_class()
        node.derive(mobile)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 0)


class TestOnBlocks(unittest.TestCase, NodeTest):
    def setUp(self):
        self.node_class = OnBlocks
        self.operational_combinations = [('Mobile', 'Heading')]

    def test_basic(self):
        mobile = buildsections('Mobile', [5, None])
        hdg = P('Heading', array=np.ma.arange(0, 360, 10, dtype=np.float))
        node = self.node_class()
        node.derive(mobile, hdg)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 36)


class TestFirstEngFuelFlowStart(unittest.TestCase):
    def test_derive(self):
        ff = P(
            'Eng (*) Fuel Flow',
            array=np.ma.array([0, 0, 1, 3, 5, 5, 5, 5, 5, 5, 3, 2, 2, 0, 0]))
        feffs = FirstEngFuelFlowStart()
        feffs.derive(ff)
        self.assertEqual(feffs[0].index, 2)


class TestLastEngFuelFlowStop(unittest.TestCase):
    def test_derive(self):
        ff = P(
            'Eng (*) Fuel Flow',
            array=np.ma.array([0, 0, 1, 3, 5, 5, 5, 5, 5, 5, 3, 2, 2, 0, 0]))
        leffs = LastEngFuelFlowStop()
        leffs.derive(ff)
        self.assertEqual(leffs[0].index, 13)
