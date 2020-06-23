import numpy as np
import os
import unittest

from hdfaccess.parameter import MappedArray
from flightdatautilities.array_operations import load_compressed

from analysis_engine.node import (
    A, M, P, S, KPV, KTI, aeroplane, App, ApproachItem, helicopter,
    KeyPointValue, KeyTimeInstance, load, Parameter, Section, SectionNode
)

from analysis_engine.library import (
    integrate, np_ma_zeros_like, np_ma_ones_like
)

from analysis_engine.helicopter.flight_phase import OnDeck

from analysis_engine.flight_phase import (
    Airborne,
    AirborneRadarApproach,
    Approach,
    ApproachAndLanding,
    BaroDifference,
    BouncedLanding,
    ClimbCruiseDescent,
    Climbing,
    Cruise,
    Descending,
    Descent,
    DescentLowClimb,
    DescentToFlare,
    EngHotelMode,
    Fast,
    FinalApproach,
    GearExtended,
    GearExtending,
    GearRetracted,
    GearRetracting,
    GoAroundAndClimbout,
    GoAround5MinRating,
    Grounded,
    Holding,
    IANFinalApproachCourseEstablished,
    IANGlidepathEstablished,
    ILSGlideslopeEstablished,
    ILSLocalizerEstablished,
    InitialApproach,
    InitialClimb,
    InitialCruise,
    Landing,
    LandingRoll,
    LevelFlight,
    MaximumContinuousPower,
    Mobile,
    RejectedTakeoff,
    ShuttlingApproach,
    Stationary,
    StraightAndLevel,
    Takeoff,
    Takeoff5MinRating,
    TakeoffRoll,
    TakeoffRollOrRejectedTakeoff,
    TakeoffRotation,
    TakeoffRunwayHeading,
    Taxiing,
    TaxiIn,
    TaxiOut,
    TCASOperational,
    TCASResolutionAdvisory,
    TCASTrafficAdvisory,
    TurningInAir,
    TurningOnGround
)

from analysis_engine.key_time_instances import BottomOfDescent, TopOfClimb, TopOfDescent

from analysis_engine.test_utils import (
    buildsection,
    buildsections,
    build_kti
)

from analysis_engine.settings import AIRSPEED_THRESHOLD
from analysis_engine.utils import open_node_container


test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data')


##############################################################################
# Superclasses


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

    def get_params_from_hdf(self, hdf_path, param_names, _slice=None,
                            phase_name='Phase'):
        import shutil
        import tempfile
        from hdfaccess.file import hdf_file
        from analysis_engine.node import derived_param_from_hdf

        params = []
        phase = None

        with tempfile.NamedTemporaryFile() as temp_file:
            shutil.copy(hdf_path, temp_file.name)

            with hdf_file(temp_file.name) as hdf:
                for param_name in param_names:
                    p = hdf.get(param_name)
                    if p is not None:
                        p = derived_param_from_hdf(p)
                    params.append(p)

        if _slice:
            phase = S(name=phase_name, frequency=1)
            phase.create_section(_slice)
            phase = phase.get_aligned(params[0])

        return params, phase


##############################################################################


class TestAirborne(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        node = Airborne
        available = ('Altitude AAL For Flight Phases', 'Fast')
        self.assertTrue(node.can_operate(available, seg_type=A('Segment Type', 'START_AND_STOP')))
        self.assertFalse(node.can_operate(available, seg_type=A('Segment Type', 'GROUND_ONLY')))

    def test_airborne_aircraft_basic(self):
        # First sample with altitude more than zero is 6, last with high speed is 80.
        vert_spd_data = np.ma.concatenate((np.zeros(5), np.arange(0,400,20), np.arange(400,-400,-20), np.arange(-400,50,20)))
        altitude = Parameter('Altitude AAL For Flight Phases', integrate(vert_spd_data, 1, 0, 1.0/60.0))
        fast = SectionNode('Fast', items=[Section(name='Airborne', slice=slice(3, 80, None), start_edge=3, stop_edge=80)])
        air = Airborne()
        air.derive(altitude, fast)
        expected = [Section(name='Airborne', slice=slice(8.714285714285714, 80, None), start_edge=8.714285714285714, stop_edge=80)]

        self.assertEqual(len(air), 1)
        self.assertAlmostEqual(air[0].slice.start, expected[0].slice.start, places=3)
        self.assertEqual(air[0].slice.stop, expected[0].slice.stop)
        self.assertAlmostEqual(air[0].start_edge, expected[0].start_edge, places=3)
        self.assertEqual(air[0].stop_edge, expected[0].stop_edge)

    def test_airborne_aircraft_not_fast(self):
        altitude_data = np.ma.arange(0, 10)
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = SectionNode('Fast')
        air = Airborne()
        air.derive(alt_aal, fast)
        self.assertEqual(air, [])

    @unittest.skip('TODO: Test to be amended')
    def test_airborne_aircraft_started_midflight(self):
        altitude_data = np.ma.array([100]*20+[60,30,10]+[0]*4)
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = buildsection('Fast', None, 25)
        air = Airborne()
        air.derive(alt_aal, fast)
        # The problem here is that buildsection now returns a slice to 24 and
        # a stop_edge to 23. Probably not worth fixing this test if we are
        # going to turn over to segments.
        expected = buildsection('Airborne', None, 23)
        self.assertEqual(air, expected)

    def test_airborne_aircraft_ends_in_midflight(self):
        altitude_data = np.ma.array([0]*5+[30,80]+[100]*20)
        alt_aal = Parameter('Altitude AAL For Flight Phases', altitude_data)
        fast = buildsection('Fast', 2, None)
        air = Airborne()
        air.derive(alt_aal, fast)
        expected = buildsection('Airborne', 5.4333, None)
        self.assertEqual(len(air), 1)
        self.assertAlmostEqual(air[0].slice.start, expected[0].slice.start, places=3)
        self.assertEqual(air[0].slice.stop, expected[0].slice.stop)
        self.assertAlmostEqual(air[0].start_edge, expected[0].start_edge, places=3)
        self.assertEqual(air[0].stop_edge, expected[0].stop_edge)


    def test_airborne_aircraft_fast_with_gaps(self):
        alt_aal = P('Altitude AAL For Flight Phases',
                    np.ma.arange(60)+10000,frequency=0.1)
        fast = buildsections('Fast', [1,10],[15,24],[30,36],[40,50],[55,59])
        fast.frequency = 0.1
        air = Airborne()
        air.derive(alt_aal, fast)
        self.assertEqual(len(air), 2)
        self.assertEqual(air[0].slice.start, 1)
        self.assertEqual(air[0].slice.stop, 24)
        self.assertEqual(air[1].slice.start, 30)
        self.assertEqual(air[1].slice.stop, 59)

    def test_airborne_aircraft_no_height_change(self):
        alt_aal = P('Altitude AAL For Flight Phases',
                    np.ma.ones(60) * 10000,frequency=0.1)
        fast = buildsections('Fast', [1,10],[15,24],[30,36],[40,50],[55,59])
        fast.frequency = 0.1
        air = Airborne()
        air.derive(alt_aal, fast)
        self.assertEqual(len(air), 0)

    def test_transient_on_takeoff(self):
        alt_aal = P('Altitude AAL For Flight Phases',
                    np.ma.concatenate([
                        np.zeros(10),
                        np.arange(0, 5, 2),
                        np.arange(4, -1, -2),
                        np.zeros(10),
                        np.arange(0, 1000, 10),
                        np.arange(1000, -10, -10),
                        np.zeros(10)]))
        fast = buildsection('Fast', 8, 235)
        air = Airborne()
        air.derive(alt_aal, fast)
        self.assertGreater(air.get_first().slice.start, 25)


class TestAirborneRadarApproach(unittest.TestCase):

    def setUp(self):
        self.node_class = AirborneRadarApproach

    def test_can_operate(self):
        expected = [('Approach Information',)]
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_derive_one_airborne_radar_approach(self):

        approaches = App()
        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('SHUTTLING',
                                   slice(35, 48, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)
        node = self.node_class()
        node.derive(approaches)
        expected = slice(19, 29)

        self.assertEqual(len(node), 1)
        self.assertEqual('Airborne Radar Approach', node.get_name())
        self.assertEqual(expected, node.get_slices()[0])

    def test_derive_two_airborne_radar_approaches(self):

        approaches = App()
        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(35, 48, None),
                                        runway_change=False,
                                        offset_ils=False,
                                        airport=None,
                                        landing_runway=None,
                                        approach_runway=None,
                                        gs_est=None,
                                        loc_est=None,
                                        ils_freq=None,
                                        turnoff=None,
                                        lowest_lat=-19.92955434,
                                        lowest_lon=115.385025548,
                                        lowest_hdg=206.713600159)
        node = self.node_class()
        node.derive(approaches)
        expected = [slice(19, 29), slice(35, 48)]

        self.assertEqual(len(node), 2)
        self.assertEqual('Airborne Radar Approach', node.get_name())
        self.assertEqual(expected[0], node.get_slices()[0])
        self.assertEqual('Airborne Radar Approach', node.get_name())
        self.assertEqual(expected[1], node.get_slices()[1])

    def test_derive_no_airborne_radar_approaches(self):

        approaches = App()
        approaches.create_approach('SHUTTLING',
                                   slice(35, 48, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)
        node = self.node_class()
        node.derive(approaches)

        self.assertEqual(len(node), 0)


class TestApproachAndLanding(unittest.TestCase):
    def test_can_operate(self):
        node = ApproachAndLanding
        start_stop = A('Segment Type', 'START_AND_STOP')
        ground_only = A('Segment Type', 'GROUND_ONLY')
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight', 'Landing',),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases',),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Landing'),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                         ac_type=aeroplane, seg_type=start_stop))

        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                          ac_type=aeroplane, seg_type=ground_only))
        # aircraft deps for helicopter invalid
        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases', 'Landing'),
                                          ac_type=helicopter, seg_type=start_stop))
        self.assertTrue(node.can_operate(('Approach', 'Landing'),
                                         ac_type=helicopter, seg_type=start_stop))

    def test_approach_and_landing_aircraft_basic(self):
        alt = np.ma.concatenate((np.arange(5000, 500, -500), np.zeros(10)))
        # No Go-arounds detected
        app = ApproachAndLanding()
        app.derive(aeroplane, Parameter('Altitude AAL For Flight Phases', alt, 0.5), None, None, None)
        self.assertEqual(app.get_slices(), [slice(4.0, 9)])

    def test_approach_and_landing_aircraft_go_around_below_1500ft(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'GoAroundAndClimbout_alt_aal.nod'))
        app_ldg = ApproachAndLanding()
        app_ldg.derive(aeroplane, alt_aal, None, None, None)
        self.assertEqual(len(app_ldg), 6)
        self.assertAlmostEqual(app_ldg[0].slice.start, 1005, places=0)
        self.assertAlmostEqual(app_ldg[0].slice.stop, 1111, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.start, 1378, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.stop, 1458, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.start, 1676, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.stop, 1783, places=0)
        self.assertAlmostEqual(app_ldg[3].slice.start, 2021, places=0)
        self.assertAlmostEqual(app_ldg[3].slice.stop, 2116, places=0)
        self.assertAlmostEqual(app_ldg[4].slice.start, 2208, places=0)
        self.assertAlmostEqual(app_ldg[4].slice.stop, 2468, places=0)
        self.assertAlmostEqual(app_ldg[5].slice.start, 2680, places=0)
        self.assertAlmostEqual(app_ldg[5].slice.stop, 2806, places=0)

    def test_approach_and_landing_aircraft_go_around_2(self):
        alt_aal = load(os.path.join(test_data_path, 'alt_aal_goaround.nod'))
        level_flights = SectionNode('Level Flight')
        level_flights.create_sections([
            slice(1629.0, 2299.0, None),
            slice(3722.0, 4708.0, None),
            slice(4726.0, 4805.0, None),
            slice(5009.0, 5071.0, None),
            slice(5168.0, 6883.0, None),
            slice(8433.0, 9058.0, None)])
        landings = buildsection('Landing', 10500, 10749)
        app_ldg = ApproachAndLanding()
        app_ldg.derive(aeroplane, alt_aal, level_flights, None, landings)
        self.assertEqual(len(app_ldg), 4)
        self.assertAlmostEqual(app_ldg[0].slice.start, 3425, places=0)
        self.assertAlmostEqual(app_ldg[0].slice.stop, 3632, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.start, 4805, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.stop, 4941, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.start, 6883, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.stop, 7171, places=0)
        self.assertAlmostEqual(app_ldg[3].slice.start, 10362, places=0)
        self.assertAlmostEqual(app_ldg[3].slice.stop, 10750, places=0)

    def test_approach_and_landing_aircraft_with_go_around_and_climbout_atr42_data(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'AltitudeAAL_ATR42_two_goarounds.nod'))
        app_ldg = ApproachAndLanding()
        app_ldg.derive(aeroplane, alt_aal, None, None, None)
        self.assertEqual(len(app_ldg), 3)
        self.assertAlmostEqual(app_ldg[0].slice.start, 9771, places=0)
        self.assertAlmostEqual(app_ldg[0].slice.stop, 10810, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.start, 12050, places=0)
        self.assertAlmostEqual(app_ldg[1].slice.stop, 12631, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.start, 26926, places=0)
        self.assertAlmostEqual(app_ldg[2].slice.stop, 27359, places=0)

    @unittest.skip('Algorithm does not successfully this noisy signal.')
    def test_approach_and_landing_aircraft_146_oscillating_1(self):
        # Example flight with noisy alt aal
        array = load_compressed(os.path.join(test_data_path, 'find_low_alts_alt_aal_1.npz'))
        alt_aal = P('Altitude AAL For Flight Phases', frequency=2, array=array)

        level_flights = buildsections('Level Flight',
            (1856.0, 2392.0),
            (4062.0, 4382.0),
            (4432.0, 4584.0),
            (4606.0, 4856.0),
            (5210.0, 5562.0),
            (5576.0, 5700.0),
            (5840.0, 5994.0),
            (6152.0, 6598.0),
            (7268.0, 7768.0),
            (8908.0, 9124.0),
            (9752.0, 9898.0),
            (9944.0, 10210.0),
            (10814.0, 11098.0),
            (11150.0, 11332.0),
            (11352.0, 11676.0),
            (12122.0, 12346.0),
            (12814.0, 12998.0),
            (13028.0, 13194.0),
            (13432.0, 13560.0),
            (13716.0, 13888.0),
            (13904.0, 14080.0),
            (14122.0, 14348.0),
            (14408.0, 14570.0),
            (14596.0, 14786.0),
            (15092.0, 15356.0),
            (15364.0, 15544.0),
            (15936.0, 16066.0),
            (16078.0, 16250.0),
            (16258.0, 16512.0),
            (16632.0, 16782.0),
            (16854.0, 16982.0),
            (17924.0, 18112.0),
            (18376.0, 18514.0),
            (18654.0, 20582.0),
            (21184.0, 21932.0),
        )
        level_flights.hz = 2

        lands = buildsection('Landing', 22296, 22400)
        lands.hz = 2

        app_lands = ApproachAndLanding(frequency=2)

        app_lands.get_derived([aeroplane, alt_aal, level_flights, None, lands])
        app_lands = app_lands.get_slices()
        self.assertEqual(len(app_lands), 3)
        self.assertAlmostEqual(app_lands[0].start, 3037, places=0)
        self.assertAlmostEqual(app_lands[0].stop, 4294, places=0)
        self.assertAlmostEqual(app_lands[1].start, 8176, places=0)
        self.assertAlmostEqual(app_lands[1].stop, 8920, places=0)
        self.assertAlmostEqual(app_lands[2].start, 21932, places=0)
        self.assertAlmostEqual(app_lands[2].stop, 22400, places=0)

    @unittest.skip('Algorithm is confused by oscillating altitude.')
    def test_approach_and_landing_aircraft_146_oscillating_2(self):
        # Example flight with noisy alt aal
        alt_aal = load(os.path.join(test_data_path, 'ApproachAndLanding_alt_aal_1.nod'))
        level_flights = load(os.path.join(test_data_path, 'ApproachAndLanding_level_flights_1.nod'))
        landings = load(os.path.join(test_data_path, 'ApproachAndLanding_landings_1.nod'))

        app_lands = ApproachAndLanding(frequency=2)

        app_lands.get_derived([aeroplane, alt_aal, level_flights, None, landings])
        app_lands = app_lands.get_slices()
        self.assertEqual(len(app_lands), 2)
        self.assertAlmostEqual(app_lands[0].start, 8824, places=0)
        self.assertAlmostEqual(app_lands[0].stop, 9980, places=0)
        self.assertAlmostEqual(app_lands[1].start, 29173, places=0)
        self.assertAlmostEqual(app_lands[1].stop, 29726, places=0)

    def test_approach_and_landing_aircraft_brief_alt_dip(self):
        '''
        Section is not created for brief altitude dip (2 samples).
        '''
        level_flights = buildsections('Level Flight', (1656, 1982), (2144, 2276), (3040, 3198), (3328, 3572), (4972, 6004), (6716, 7716))
        lands = buildsections('Landing', (2039, 2591), (6345, 6654), (8137, 8212))
        alt_aal = P('Altitude AAL', load_compressed(os.path.join(test_data_path, 'ApproachAndLanding_alt_aal_2.npz')), 2)
        node = ApproachAndLanding()
        node.derive(aeroplane, alt_aal, level_flights, None, lands)
        self.assertEqual(len(node), 3)
        self.assertAlmostEqual(node[0].slice.start, 4156, places=0)
        self.assertAlmostEqual(node[0].slice.stop, 4732, places=0)
        self.assertAlmostEqual(node[1].slice.start, 6004, places=0)
        self.assertAlmostEqual(node[1].slice.stop, 6359, places=0)
        self.assertAlmostEqual(node[2].slice.start, 7716, places=0)
        self.assertAlmostEqual(node[2].slice.stop, 8213, places=0)

    def test_approach_and_landing_helicopter(self):
        apps = buildsection('Approach', 2, 5)
        lands = buildsection('Landing', 4, 7)
        node = ApproachAndLanding()
        node.derive(helicopter, None, None, apps, lands)
        self.assertEqual(len(node), 1)
        self.assertEqual(node.get_slices()[0].start, 2)
        self.assertEqual(node.get_slices()[0].stop, 10) # land phase stop + 2 samples




class TestApproach(unittest.TestCase):

    def test_can_operate(self):
        node = Approach
        start_stop = A('Segment Type', 'START_AND_STOP')
        ground_only = A('Segment Type', 'GROUND_ONLY')
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight', 'Landing',),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases',),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Landing'),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                          ac_type=aeroplane, seg_type=ground_only))
        self.assertTrue(node.can_operate(('Altitude AGL', 'Altitude STD'),
                                          ac_type=helicopter, seg_type=start_stop))

    def test_approach_aircraft_basic(self):
        alt = np.ma.concatenate((np.arange(5000, 500, -500), [50], np.zeros(10)))
        app = Approach()
        land = buildsection('Landing', 11,14)
        app.derive(aeroplane, Parameter('Altitude AAL For Flight Phases', alt), None, land)
        self.assertEqual(app.get_slices(), [slice(4.0, 9)])

    def test_approach_aircraft_ignore_takeoff(self):
        alt = np.ma.concatenate((np.zeros(5), np.arange(0,5000,500), np.arange(5000,500,-500), np.zeros(5)))
        app = Approach()
        land = buildsection('Landing', 11,14)
        app.derive(aeroplane, Parameter('Altitude AAL For Flight Phases', alt), None, land)
        self.assertEqual(len(app), 1)

    def test_approach_aircraft_with_go_around_and_climbout_atr42_data(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'AltitudeAAL_ATR42_two_goarounds.nod'))
        #alt_aal.array[9860:10180] = 2560
        landing = buildsection('Landing', 27350, 27400)
        level_flights = buildsections('Level Flight',
            (8754, 8971),
            (9844, 10207),
            (11108, 11241),
            (11256, 11489),
            (14887, 20037),
            (20127, 22946),
            (22947, 23587),
            (23701, 24428),
            (25208, 25356),
            (25376, 25837),
        )
        app = Approach()
        app.derive(aeroplane, alt_aal, level_flights, landing)

        self.assertEqual(len(app), 3)
        self.assertAlmostEqual(app[0].slice.start,10207, places=0)
        self.assertAlmostEqual(app[0].slice.stop, 10810, places=0)
        self.assertAlmostEqual(app[1].slice.start,12050, places=0)
        self.assertAlmostEqual(app[1].slice.stop, 12631, places=0)
        self.assertAlmostEqual(app[2].slice.start,26926, places=0)
        self.assertAlmostEqual(app[2].slice.stop, 27342, places=0)

    def test_approach__helicopter_multiple(self):
        alt_array = np.ma.concatenate((np.ones(5) * 1500, np.arange(1500,50,-25), np.arange(50,1100,25), np.arange(1100,0,-25), np.zeros(6)))
        alt = Parameter('Altitude AGL', alt_array)
        app = Approach()
        app.derive(helicopter, None, None, None, alt, alt)

        self.assertEqual(len(app), 2)
        self.assertAlmostEqual(app[0].slice.start,8, places=0)
        self.assertAlmostEqual(app[0].slice.stop, 63, places=0)
        self.assertAlmostEqual(app[1].slice.start,106.5, places=0)
        self.assertAlmostEqual(app[1].slice.stop, 149, places=0)


class TestBouncedLanding(unittest.TestCase):
    def setUp(self):
        self.node_class = BouncedLanding

    def test_can_operate(self):
        self.assertTrue(BouncedLanding.can_operate(
            ('Altitude AAL For Flight Phases', 'Airborne'),
            ac_type=aeroplane))

    def test_bounce_basic(self):
        airborne = buildsection('Airborne', 3,10)
        alt = np.ma.array([0,0,0,2,10,30,10,2,0,0,0,0,0,0])
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL For Flight Phases', alt), airborne, None)
        expected = []
        self.assertEqual(bl, expected)

    def test_bounce_with_bounce(self):
        airborne = buildsection('Airborne', 3,11)
        alt = np.ma.array([0,0,0,2,10,30,10,2,0,3,3,0,0,0])
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL For Flight Phases', alt), airborne, None)
        self.assertEqual(bl[0].slice, slice(9, 11))

    def test_bounce_with_double_bounce(self):
        airborne = buildsection('Airborne', 3,12)
        alt = np.ma.array([0,0,0,2,10,30,10,2,0,3,0,5,0])
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL For Flight Phases', alt), airborne, None)
        self.assertEqual(bl[0].slice, slice(9, 12))

    def test_bounce_with_bounce_with_gog(self):
        airborne = buildsection('Airborne', 3,11)
        alt = np.ma.array([0,0,0,2,10,30,10,2,0,3,3,0,0,0])
        gog = M('Gear On Ground', np.ma.array([1]*3 + [0]*5 + [1]*6), values_mapping = {0:'Air', 1:'Ground'})
        bl = BouncedLanding()
        bl.derive(Parameter('Altitude AAL For Flight Phases', alt), airborne, gog)
        self.assertEqual(len(bl.get_slices()), 0) # Only first bounce has wheels off ground

    def test_bounce_not_detected_with_multiple_touch_and_go(self):
        # test data is a training flight with many touch and go
        bl = BouncedLanding()
        aal = load(os.path.join(test_data_path, 'alt_aal_training.nod'))
        airs = load(os.path.join(test_data_path, 'airborne_training.nod'))
        bl.derive(aal, airs, None)
        # should not create any bounced landings (used to create 20 at 8000ft)
        self.assertEqual(len(bl), 0)


class TestIANFinalApproachCourseEstablished(unittest.TestCase):

    def setUp(self):
        self.node_class = IANFinalApproachCourseEstablished
        ian_array = load_compressed(os.path.join(test_data_path, 'ian_established-ian_app_course.npz'))
        self.ian_app_corse = Parameter('IAN Final Approach Course', ian_array)
        aal_array = load_compressed(os.path.join(test_data_path, 'ian_established-alt_aal.npz'))
        self.alt_aal = Parameter('Altitude AAL For Flight Phases', aal_array)
        values_mapping = {0: 'No Source', 1: 'FMC', 5: 'LOC/FMC', 6: 'GLS', 7: 'ILS'}
        self.app_src_capt = M('Displayed App Source (Capt)', np_ma_zeros_like(aal_array), values_mapping=values_mapping)
        self.app_src_fo = M('Displayed App Source (FO)', np_ma_zeros_like(aal_array), values_mapping=values_mapping)

    def test_derive__basic(self):
        self.app_src_capt.array[slice(28710, 30480)] = 'FMC'
        self.app_src_fo.array[slice(28709, 30481)] = 'FMC'
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(30238, 30537), loc_est=None),])
        node = self.node_class()
        node.derive(self.ian_app_corse,
                    self.alt_aal,
                    apps,
                    None,
                    self.app_src_capt,
                    self.app_src_fo)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 30238, delta=1)
        self.assertAlmostEqual(node[0].slice.stop, 30438, delta=1) #TODO: check stop index

    def test_derive__ils_approach(self):
        self.app_src_capt.array[slice(28710, 30480)] = 'ILS'
        self.app_src_fo.array[slice(28709, 30481)] = 'ILS'
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(30238, 30537), loc_est=True),])
        node = self.node_class()
        node.derive(self.ian_app_corse,
                    self.alt_aal,
                    apps,
                    None,
                    self.app_src_capt,
                    self.app_src_fo)
        self.assertEqual(len(node), 0)

    def test_derive__no_source(self):
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(30238, 30537), loc_est=None),])
        node = self.node_class()
        node.derive(self.ian_app_corse,
                    self.alt_aal,
                    apps,
                    None,
                    self.app_src_capt,
                    self.app_src_fo)
        self.assertEqual(len(node), 0)


class TestIANGlidepathEstablished(unittest.TestCase):

    def setUp(self):
        self.node_class = IANGlidepathEstablished
        ian_array = load_compressed(os.path.join(test_data_path, 'ian_established-ian_glidepath.npz'))
        self.ian_glidepath = Parameter('IAN Glidepath', ian_array)
        aal_array = load_compressed(os.path.join(test_data_path, 'ian_established-alt_aal.npz'))
        self.alt_aal = Parameter('Altitude AAL For Flight Phases', aal_array)
        values_mapping = {0: 'No Source', 1: 'FMC', 5: 'LOC/FMC', 6: 'GLS', 7: 'ILS'}
        self.app_src_capt = Parameter('Displayed App Source (Capt)',
                                      MappedArray(np_ma_zeros_like(aal_array), values_mapping=values_mapping))
        self.app_src_fo = Parameter('Displayed App Source (FO)',
                                    MappedArray(np_ma_zeros_like(aal_array), values_mapping=values_mapping))

    def test_derive__basic(self):
        self.app_src_capt.array[slice(28710, 30480)] = 'FMC'
        self.app_src_fo.array[slice(28709, 30481)] = 'FMC'
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(30238, 30537), gs_est=None),])
        node = self.node_class()
        node.derive(self.ian_glidepath,
                    self.alt_aal,
                    apps,
                    self.app_src_capt,
                    self.app_src_fo)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 30346, delta=1)
        self.assertAlmostEqual(node[0].slice.stop, 30419, delta=1)

    def test_derive__ils_approach(self):
        self.app_src_capt.array[slice(28710, 30480)] = 'ILS'
        self.app_src_fo.array[slice(28709, 30481)] = 'ILS'
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(30238, 30537), gs_est=True),])
        node = self.node_class()
        node.derive(self.ian_glidepath,
                    self.alt_aal,
                    apps,
                    self.app_src_capt,
                    self.app_src_fo)
        self.assertEqual(len(node), 0)

    def test_derive__no_source(self):
        apps = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(30238, 30537), gs_est=None),])
        node = self.node_class()
        node.derive(self.ian_glidepath,
                    self.alt_aal,
                    apps,
                    self.app_src_capt,
                    self.app_src_fo)
        self.assertEqual(len(node), 0)



class TestILSGlideslopeEstablished(unittest.TestCase):

    def setUp(self):
        self.node_class = ILSGlideslopeEstablished

    def test_can_operate(self):
        expected = [('Approach Information',)]
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_landing_turn_off_runway_basic(self):
        node = self.node_class()
        apps = [ApproachItem('LANDING', slice(0, 5), gs_est=slice(33, 48))]
        node.derive(apps)
        expected = [Section('ILS Glideslope Established', slice(33, 48), 33, 48)]
        self.assertEqual(node, expected)


class TestILSLocalizerEstablished(unittest.TestCase):

    def setUp(self):
        self.node_class = ILSLocalizerEstablished

    def test_can_operate(self):
        expected = [('Approach Information',)]
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_landing_turn_off_runway_basic(self):
        node = self.node_class()
        apps = [ApproachItem('LANDING', slice(0, 5), loc_est=slice(33, 48))]
        node.derive(apps)
        expected = [Section('ILS Localizer Established', slice(33, 48), 33, 48)]
        self.assertEqual(node, expected)


class TestInitialApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases',
                     'Approach')]
        opts = InitialApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_initial_approach_phase_basic(self):
        alt = np.ma.concatenate((np.arange(4000,0,-500), np.arange(0,4000,500)))
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(2, 8), 2, 8)])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, app_land)
        expected = [Section('Initial Approach', slice(2, 6,), 2, 6)]
        self.assertEqual(app, expected)

    def test_initial_approach_phase_over_high_ground(self):
        alt_aal = np.ma.concatenate((np.arange(0,4000,500), np.arange(4000,0,-500)))
        # Raising the ground makes the radio altitude trigger one sample sooner.
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt_aal)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(10, 16, None), 10, 16)])
        app.derive(alt_aal, app_land)
        expected = [Section('Initial Approach', slice(10, 14), 10, 14)]
        self.assertEqual(app, expected)

    def test_initial_approach_phase_with_go_around(self):
        alt = np.ma.concatenate((np.arange(4000,2000,-500), np.arange(2000,4000,500)))
        app = InitialApproach()
        alt_aal = Parameter('Altitude AAL For Flight Phases',alt)
        app_land = SectionNode('Approach',
            items=[Section('Approach', slice(2, 5), 2, 5)])
        # Pretend we are flying over flat ground, so the altitudes are equal.
        app.derive(alt_aal, app_land)
        expected = [Section('Initial Approach', slice(2, 4), 2, 4)]
        self.assertEqual(app, expected)


'''
class TestCombinedClimb(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Top Of Climb', 'Go Around', 'Liftoff', 'Touchdown')]
        opts = CombinedClimb.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_derive(self):
        toc_name = 'Top Of Climb'
        toc = KTI(toc_name, items=[KeyTimeInstance(4344, toc_name),
                                   KeyTimeInstance(5496, toc_name),
                                   KeyTimeInstance(7414, toc_name)])
        ga_name = 'Go Around'
        ga = KTI(ga_name, items=[KeyTimeInstance(5404.4375, ga_name),
                                 KeyTimeInstance(6314.9375, ga_name)])
        lo = KTI('Liftoff', items=[KeyTimeInstance(3988.9375, 'Liftoff')])
        node = CombinedClimb()
        node.derive(toc, ga, lo)
        climb_name = 'Combined Climb'
        expected = [
            Section(name='Combined Climb', slice=slice(3988.9375, 4344, None), start_edge=3988.9375, stop_edge=4344),
            Section(name='Combined Climb', slice=slice(5404.4375, 5496, None), start_edge=5404.4375, stop_edge=5496),
            Section(name='Combined Climb', slice=slice(6314.9375, 7414, None), start_edge=6314.9375, stop_edge=7414),
        ]

        self.assertEqual(list(node), expected)
'''


class TestClimbCruiseDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD Smoothed','Airborne')]
        opts = ClimbCruiseDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climb_cruise_descent_start_midflight(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.concatenate((np.ones(5) * 15000, np.arange(15000, 1000, -1000)))
        alt_aal = Parameter('Altitude STD Smoothed',
                            testwave)
        air=buildsection('Airborne', None, 18)
        camel.derive(alt_aal, air)
        self.assertEqual(camel[0].slice, slice(None, 18))

    def test_climb_cruise_descent_end_midflight(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.concatenate((np.arange(1000,15000,1000), np.ones(5) * 15000))
        alt_aal = Parameter('Altitude STD Smoothed', testwave)
        air=buildsection('Airborne',0, None)
        camel.derive(alt_aal, air)
        expected = buildsection('Climb Cruise Descent', 0, None)
        self.assertEqual(camel, expected)

    def test_climb_cruise_descent_all_high(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.ones(5) * 15000
        # plot_parameter (testwave)
        air=buildsection('Airborne',0,5)
        camel.derive(Parameter('Altitude STD Smoothed', testwave), air)
        expected = []
        self.assertEqual(camel, expected)

    def test_climb_cruise_descent_one_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 2, 0.1)) * -3000 + 12500
        air = buildsection('Airborne', 0, 62)
        camel.derive(Parameter('Altitude STD Smoothed', testwave), air)
        self.assertEqual(len(camel), 1)

    def test_climb_cruise_descent_one_hump_with_airborne_delayed(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 2, 0.1)) * -3000 + 12500
        air = buildsection('Airborne', 5, 62)
        camel.derive(Parameter('Altitude STD Smoothed', testwave), air)
        self.assertEqual(len(camel), 1)
        self.assertEqual(camel[0].slice, slice(5, 62, None))

    def test_climb_cruise_descent_two_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 4, 0.1)) * -3000 + 12500
        # plot_parameter (testwave)
        air = buildsection('Airborne',0,122)
        camel.derive(Parameter('Altitude STD Smoothed', testwave), air)
        self.assertEqual(len(camel), 2)

    def test_climb_cruise_descent_three_humps(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 6, 0.1)) * (-3000) + 12500
        # plot_parameter (testwave)
        air = buildsection('Airborne',0,186)
        camel.derive(Parameter('Altitude STD Smoothed', testwave), air)
        self.assertEqual(len(camel), 3)

    def test_climb_cruise_descent_masked(self):
        # This test will find out if we can separate the two humps on this camel
        camel = ClimbCruiseDescent()
        # Needs to get above 15000ft and below 10000ft to create this phase.
        testwave = np.ma.cos(np.arange(0, 3.14 * 6, 0.1)) * (-3000) + 12500
        testwave[63:125] = np.ma.masked
        # import matplotlib.pyplot as plt
        # plt.plot(testwave)
        # plt.show()
        air = buildsection('Airborne',0,186)
        camel.derive(Parameter('Altitude STD Smoothed', testwave), air)
        self.assertEqual(len(camel), 2)

    def test_climb_cruise_descent_repair_mask(self):
        # If the Altitude STD mask isn't repaired, a spurious cycle is reported
        # by cycle_finder which results in an infinite loop.
        camel = ClimbCruiseDescent()
        air = buildsection('Airborne', 619, 5325)
        camel.derive(Parameter('Altitude STD Smoothed',
                               load_compressed(os.path.join(test_data_path, 'climb_cruise_descent_alt_std.npz'))),
                     air)
        self.assertEqual(len(camel), 1)



'''
# ClimbFromBottomOfDescent is commented out in flight_phase.py
class TestClimbFromBottomOfDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Top Of Climb', 'Climb Start', 'Bottom Of Descent')]
        opts = ClimbFromBottomOfDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descent_to_bottom_of_descent_basic(self):
        testwave = np.cos(np.arange(0,12.6,0.1))*(-3000)+12500
        alt_data = np.ma.array(testwave)

        #===========================================================
        # This block of code replicates normal opeartion and ensures
        # that the cruise/climb/descent, top of climb and top of
        # descent data matches the cruise phase under test.
        #===========================================================
        # Use the same test data for flight phases and measured altitude.
        alt = Parameter('Altitude STD', alt_data)

        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Flight Phases', alt_data))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        dlc = DescentLowClimb()
        dlc.derive(alt)
        bod = BottomOfDescent()
        bod.derive(alt, dlc)

        descent_phase = ClimbFromBottomOfDescent()
        descent_phase.derive(toc, [], bod) # TODO: include start of climb instance
        expected = [Section(name='Climb From Bottom Of Descent',slice=slice(63, 94, None))]
        self.assertEqual(descent_phase, expected)
'''


class TestClimbing(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Vertical Speed For Flight Phases', 'Airborne')]
        opts = Climbing.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_climbing_basic(self):
        vert_spd_data = np.ma.concatenate((np.arange(500,1200,100), np.arange(1200,-1200,-200), np.arange(-1200,500,100)))
        vert_spd = Parameter('Vertical Speed For Flight Phases',
                             np.ma.array(vert_spd_data))
        air = buildsection('Airborne', 2, 7)
        up = Climbing()
        up.derive(vert_spd, air)
        self.assertEqual(up[0].slice, slice(3,8))


class TestCruise(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Climb Cruise Descent',
                     'Top Of Climb', 'Top Of Descent',
                     'Airspeed')]
        opts = Cruise.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_cruise_phase_basic(self):
        alt_data = np.ma.array(
            np.cos(np.arange(0, 12.6, 0.1)) * -3000 + 12500)

        #===========================================================
        # This block of code replicates normal opeartion and ensures
        # that the cruise/climb/descent, top of climb and top of
        # descent data matches the cruise phase under test.
        #===========================================================
        # Use the same test data for flight phases and measured altitude.
        alt_p = Parameter('Altitude STD', alt_data)
        # Transform the "recorded" altitude into the CCD input data.
        ccd = ClimbCruiseDescent()
        ccd.derive(alt_p, buildsection('Airborne', 0, len(alt_data)-1))
        toc = TopOfClimb()
        toc.derive(alt_p, ccd)
        tod = TopOfDescent()
        tod.derive(alt_p, ccd)
        air_spd = Parameter('Altitude STD', np_ma_ones_like(alt_data) * 60)

        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod, air_spd)
        #===========================================================

        # With this test waveform, the peak at 31:32 is just flat enough
        # for the climb and descent to be a second apart, whereas the peak
        # at 94 genuinely has no interval with a level cruise.
        expected = [slice(31, 32),slice(94, 95)]
        self.assertEqual(test_phase.get_slices(), list(expected))

    def test_cruise_truncated_start(self):
        alt_data = np.ma.concatenate((np.ones(5) * 15000, np.arange(15000,2000,-4000)))
        #===========================================================
        alt = Parameter('Altitude STD', alt_data)
        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Climb Cruise Descent', alt_data),
                   buildsection('Airborne', 0, len(alt_data)))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        air_spd = Parameter('Altitude STD', np_ma_ones_like(alt_data) * 60)

        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod, air_spd)
        #===========================================================
        self.assertEqual(test_phase[0].slice, slice(None,5))
        self.assertEqual(len(toc), 0)
        self.assertEqual(len(tod), 1)

    def test_cruise_truncated_end(self):
        alt_data = np.ma.concatenate((np.arange(300,36000,6000), np.ones(4) * 36000))
        #===========================================================
        alt = Parameter('Altitude STD', alt_data)
        ccd = ClimbCruiseDescent()
        ccd.derive(Parameter('Altitude For Climb Cruise Descent', alt_data),
                   buildsection('Airborne', 0, len(alt_data)))
        toc = TopOfClimb()
        toc.derive(alt, ccd)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        air_spd = Parameter('Altitude STD', np_ma_ones_like(alt_data) * [60])

        test_phase = Cruise()
        test_phase.derive(ccd, toc, tod, air_spd)
        #===========================================================
        expected = Cruise()
        expected.create_section(slice(6, None), 'Cruise')
        self.assertEqual(test_phase, expected)
        self.assertEqual(len(toc), 1)
        self.assertEqual(len(tod), 0)


class TestInitialClimb(unittest.TestCase):

    def test_can_operate(self):
        expected = [('Takeoff', 'Climb Start', 'Top Of Climb', 'Altitude STD', 'Aircraft Type')]
        opts = InitialClimb.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_basic(self):
        ini_clb = InitialClimb()
        toff = buildsection('Takeoff', 10, 20)
        clb_start = build_kti('Climb Start', 30)
        toc = build_kti('Top Of Climb', 40)
        alt = P('Altitude STD', array=np.ma.arange(0, 20000, 20))
        ini_clb.derive(toff, clb_start, toc, alt)
        self.assertEqual(len(ini_clb.get_slices()), 1)
        self.assertEqual(ini_clb.get_first().slice.start, 20)
        self.assertEqual(ini_clb.get_first().slice.stop, 30)

    def test_short_climb(self):
        # in this case we don't want the phase to be created
        # as if the max(alt) is above 1000 feet, that most likely
        # means that we're missing or have masked some of the data during
        # takeoff and creating the phase in this case will result in
        # Initial Climb to be up to ToC point which is usually
        # over 30.000ft raising invalid events
        ini_clb = InitialClimb()
        toff = buildsection('Takeoff', 10, 20)
        clb_start = build_kti('Climb Start', None)
        toc = build_kti('Top Of Climb', 40)
        alt = P('Altitude STD', array=np.ma.arange(0, 20000, 20))
        ini_clb.derive(toff, clb_start, toc, alt)
        self.assertEqual(len(ini_clb.get_slices()), 0)

    def test_initial_climb_for_helicopter_operation(self):
        # This test case is borne out of actual helicopter data.
        toffs = buildsections('Takeoff', [5, 10], [15, 20])
        climbs = build_kti('Climb Start', 30)
        toc = build_kti('Top Of Climb', 40)
        ini_clb = InitialClimb()
        alt = P('Altitude STD', array=np.ma.arange(0, 20000, 20))
        ini_clb.derive(toffs, climbs, toc, alt)
        self.assertEqual(len(ini_clb), 1)

    def test_max_alt_below_1000ft(self):
        ini_clb = InitialClimb()
        toff = buildsection('Takeoff', 10, 20)
        clb_start = build_kti('Climb Start', 30)
        toc = build_kti('Top Of Climb', 40)
        alt = P('Altitude STD', array=np.ma.arange(0, 950, 5))
        ini_clb.derive(toff, clb_start, toc, alt)
        self.assertEqual(len(ini_clb.get_slices()), 1)
        self.assertEqual(ini_clb.get_first().slice.start, 20)
        self.assertEqual(ini_clb.get_first().slice.stop, 30)

    def test_no_alt_data_below_1000ft(self):
        ini_clb = InitialClimb()
        toff = buildsection('Takeoff', 10, 20)
        clb_start = build_kti('Climb Start', 30)
        toc = build_kti('Top Of Climb', 40)
        alt = P('Altitude STD', array=np.ma.arange(1500, 17000, 10))
        ini_clb.derive(toff, clb_start, toc, alt)
        self.assertEqual(len(ini_clb.get_slices()), 1)
        self.assertEqual(ini_clb.get_first().slice.start, 20)
        self.assertEqual(ini_clb.get_first().slice.stop, 30)


class TestInitialCruise(unittest.TestCase):

    def test_basic(self):
        cruise=buildsection('Cruise', 1000,1500)
        ini_cru = InitialCruise()
        ini_cru.derive(cruise)
        self.assertEqual(ini_cru[0].slice, slice(1300,1330))

    def test_short_cruise(self):
        short_cruise=buildsection('Cruise', 1000,1329)
        ini_cru = InitialCruise()
        ini_cru.derive(short_cruise)
        self.assertEqual(len(ini_cru), 0)

    def test_multiple_cruises(self):
        short_cruise=buildsections('Cruise', [1000,1339], [2000,3000])
        ini_cru = InitialCruise()
        ini_cru.derive(short_cruise)
        self.assertEqual(len(ini_cru), 1)
        self.assertEqual(ini_cru[0].slice, slice(1300,1330))


class TestDescentLowClimb(unittest.TestCase):
    def test_can_operate(self):
        node = DescentLowClimb
        start_stop = A('Segment Type', 'START_AND_STOP')
        ground_only = A('Segment Type', 'GROUND_ONLY')
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases',),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                             seg_type=start_stop))

        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases', 'Level Flight'),
                                              seg_type=ground_only))

    def test_descent_low_climb_basic(self):
        # Wave is 5000ft to 0 ft and back up, with climb of 5000ft.
        testwave = np.ma.cos(np.arange(0, 6.3, 0.1)) * (2500) + 2500
        dsc = testwave - testwave[0]
        dsc[32:] = 0.0
        clb = testwave - min(testwave)
        clb[:31] = 0.0
        alt_aal = Parameter('Altitude AAL For Flight Phases', testwave)
        #descend = Parameter('Descend For Flight Phases', np.ma.array(dsc))
        #climb = Parameter('Climb For Flight Phases', np.ma.array(clb))
        dlc = DescentLowClimb()
        dlc.derive(alt_aal)
        self.assertEqual(len(dlc), 1)
        self.assertAlmostEqual(dlc[0].slice.start, 14, places=0)
        self.assertAlmostEqual(dlc[0].slice.stop, 38, places=0)

    def test_descent_low_climb_inadequate_climb(self):
        testwave = np.ma.cos(np.arange(0, 6.3, 0.1)) * (240) + 2500 # 480ft climb
        clb = testwave - min(testwave)
        clb[:31] = 0.0
        alt_aal = Parameter('Altitude AAL For Flight Phases', testwave)
        dlc = DescentLowClimb()
        dlc.derive(alt_aal)
        self.assertEqual(len(dlc), 0)


class TestDescending(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Vertical Speed For Flight Phases', 'Airborne')]
        opts = Descending.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descending_basic(self):
        vert_spd = Parameter('Vertical Speed For Flight Phases',
                             np.ma.concatenate((np.zeros(2), np.ones(5) * 1000, np.ones(12) * -500, np.zeros(2))))
        air = buildsection('Airborne',3,20)
        phase = Descending()
        phase.derive(vert_spd, air)
        self.assertEqual(phase[0].slice, slice(7,19))

class TestDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Top Of Descent', 'Bottom Of Descent')]
        opts = Descent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_descent_basic(self):
        alt_data = np.ma.cos(np.arange(0,12.6,0.1))*(-3000)+12500
        air = buildsection('Airborne', 0, len(alt_data))

        #===========================================================
        # This block of code replicates normal opeartion and ensures
        # that the cruise/climb/descent, top of climb and top of
        # descent data matches the cruise phase under test.
        #===========================================================
        # Use the same test data for flight phases and measured altitude.
        alt = Parameter('Altitude STD', alt_data)

        ccd = ClimbCruiseDescent()
        ccd.derive(alt, air)
        tod = TopOfDescent()
        tod.derive(alt, ccd)
        bod = BottomOfDescent()
        bod.derive(alt, ccd)

        descent_phase = Descent()
        descent_phase.derive(tod, bod)
        expected = [Section(name='Descent',slice=slice(32,63,None), start_edge = 32, stop_edge = 63),
                    Section(name='Descent',slice=slice(94,125,None), start_edge = 94, stop_edge = 125)]
        self.assertEqual(descent_phase, expected)

class TestFast(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(Fast.can_operate(
            ('Airspeed',),
            ac_type=aeroplane,
            seg_type=A('Segment Type', 'START_AND_STOP')))
        self.assertTrue(Fast.can_operate(
            ('Nr',),
            ac_type=helicopter))
        self.assertFalse(Fast.can_operate(
            ('Airspeed',),
            ac_type=helicopter))

    def test_fast_phase_basic(self):
        slow_and_fast_data = np.ma.concatenate((np.arange(60, 120, 10), np.ones(300) * 120, np.arange(120, 50, -10)))
        ias = Parameter('Airspeed', slow_and_fast_data,1,0)
        phase_fast = Fast()
        phase_fast.derive(ias, None, None, None)
        if AIRSPEED_THRESHOLD == 80:
            expected = buildsection('Fast', 2, 311)
        if AIRSPEED_THRESHOLD == 70:
            expected = buildsection('Fast', 1, 312)
        self.assertEqual(phase_fast.get_slices(), expected.get_slices())

    def test_fast_all_fast(self):
        fast_data = np.ma.ones(20) * 120
        ias = Parameter('Airspeed', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias, None, None, None)
        # Q: Should we really create no fast sections?
        expected = buildsection('Fast', 0, 20)
        self.assertEqual(phase_fast.get_slices(), expected.get_slices())

    def test_fast_all_slow(self):
        fast_data = np.ma.ones(10) * 12
        ias = Parameter('Airspeed', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias, None, None, None)
        self.assertEqual(phase_fast, [])

    def test_fast_slowing_only(self):
        fast_data = np.ma.arange(120, 60, -2)
        ias = Parameter('Airspeed', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias, None, None, None)
        expected = buildsection('Fast', 0, 21)
        self.assertEqual(phase_fast.get_slices(), expected.get_slices())

    def test_fast_speeding_only(self):
        fast_data = np.ma.arange(60, 120, 2)
        ias = Parameter('Airspeed', fast_data, 1, 0)
        phase_fast = Fast()
        phase_fast.derive(ias, None, None, None)
        expected = buildsection('Fast', 10, 30)
        self.assertEqual(phase_fast.get_slices(), expected.get_slices())

    def test_fast_phase_prop(self):
        slow_and_fast_data = np.ma.concatenate((np.arange(40, 120, 10), np.ones(300) * 120, np.arange(120, 30, -10)))
        ias = Parameter('Airspeed', slow_and_fast_data)
        phase_fast = Fast()
        phase_fast.derive(ias, None, None, A('Engine Propulsion', 'PROP'))

        expected = buildsection('Fast', 2, 315)
        self.assertEqual(phase_fast.get_slices(), expected.get_slices())

    def test_fast_real_data_1(self):
        airspeed = load(os.path.join(test_data_path, 'Fast_airspeed.nod'))
        node = Fast()
        node.derive(airspeed, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 2258)
        self.assertEqual(node[0].slice.stop, 14976)

        # Test short masked section does not create two fast slices.
        airspeed.array[5000:5029] = np.ma.masked
        node = Fast()
        node.derive(airspeed, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 2258)
        self.assertEqual(node[0].slice.stop, 14976)

        # Test long masked section splits up fast into two slices.
        airspeed.array[6000:10000] = np.ma.masked
        node = Fast()
        node.derive(airspeed, None, None, None)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 2258)
        self.assertEqual(node[0].slice.stop, 6000)
        self.assertEqual(node[1].slice.start, 10000)
        self.assertEqual(node[1].slice.stop, 14976)

    def test_spike_airspd(self):
        spd_array = np.ma.arange(0, 120)
        spd_array[:60] = np.ma.masked
        spd_array[2] = 100  # Spike
        ias = Parameter('Airspeed', spd_array, 1.0, 0.0)
        phase_fast = Fast()
        phase_fast.derive(ias, None, None, None)
        expected = buildsection('Fast', 80, 120)
        self.assertEqual(phase_fast.get_slices(), expected.get_slices())


class TestGrounded(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(Grounded.can_operate(('HDF Duration',), ac_type=aeroplane))
        self.assertTrue(Grounded.can_operate(('Airborne', 'HDF Duration'), ac_type=aeroplane))
        self.assertTrue(Grounded.can_operate(('Airspeed', 'HDF Duration'), ac_type=aeroplane))
        self.assertTrue(Grounded.can_operate(('Airborne', 'Airspeed', 'HDF Duration'), ac_type=aeroplane))
        self.assertFalse(Grounded.can_operate(('HDF Duration',), ac_type=helicopter))
        self.assertTrue(Grounded.can_operate(('Airborne', 'Airspeed'), ac_type=helicopter))

    def test_grounded_aircraft_phase_basic(self):
        slow_and_fast_data = \
            np.ma.concatenate((np.arange(60, 120, 10), np.ones(300) * 120, np.arange(120, 50, -10)))
        ias = Parameter('Airspeed', slow_and_fast_data, 1, 0)
        duration = A('HDF Duration', len(ias.array)/ias.frequency)
        air = buildsection('Airborne', 2, 311)
        phase_grounded = Grounded()
        phase_grounded.derive(ias, air, aeroplane, duration)
        expected = buildsections('Grounded', [0, 2], [311, 313])
        self.assertEqual(phase_grounded.get_slices(), expected.get_slices())

    def test_grounded_aircraft_all_fast(self):
        grounded_data = np.ma.ones(10) * 120
        ias = Parameter('Airspeed', grounded_data, 1, 0)
        duration = A('HDF Duration', len(ias.array)/ias.frequency)
        air = buildsection('Airborne', None, None)
        phase_grounded = Grounded()
        phase_grounded.derive(ias, air, aeroplane, duration)
        expected = buildsection('Grounded', None, None)
        self.assertEqual(phase_grounded.get_slices(), expected.get_slices())

    def test_grounded_aircraft_all_slow(self):
        grounded_data = np.ma.ones(10) * 12
        ias = Parameter('Airspeed', grounded_data, 1, 0)
        duration = A('HDF Duration', len(ias.array)/ias.frequency)
        air = buildsection('Airborne', None, None)
        phase_grounded = Grounded()
        phase_grounded.derive(ias, air, aeroplane, duration)
        expected = buildsection('Grounded', 0, 9)
        self.assertEqual(phase_grounded.get_first().slice, expected[0].slice)

    def test_grounded_aircraft_landing_only(self):
        grounded_data = np.ma.arange(110,60,-10)
        ias = Parameter('Airspeed', grounded_data,1,0)
        duration = A('HDF Duration', len(ias.array)/ias.frequency)
        air = buildsection('Airborne',None,4)
        phase_grounded = Grounded()
        phase_grounded.derive(ias, air, aeroplane, duration)
        expected = buildsection('Grounded',4,4)
        self.assertEqual(phase_grounded.get_first().slice, expected[0].slice)

    def test_grounded_aircraft_speeding_only(self):
        grounded_data = np.ma.arange(60,120,10)
        ias = Parameter('Airspeed', grounded_data,1,0)
        duration = A('HDF Duration', len(ias.array)/ias.frequency)
        air = buildsection('Airborne',2,None)
        phase_grounded = Grounded()
        phase_grounded.derive(ias, air, aeroplane, duration)
        expected = buildsection('Grounded',0,1)
        self.assertEqual(phase_grounded.get_first().slice, expected[0].slice)


class TestFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL For Flight Phases', 'Airborne')]
        opts = FinalApproach.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_approach_phase_basic(self):
        alt = np.ma.concatenate((np.arange(0,1200,100), np.arange(1500,500,-100), np.arange(400,0,-40), np.zeros(3)))
        alt_aal = Parameter('Altitude AAL For Flight Phases', array=alt)
        expected = buildsection('Final Approach', 18, 31)
        fapp=FinalApproach()
        fapp.derive(alt_aal)
        self.assertEqual(fapp.get_slices(), expected.get_slices())


class TestGearRetracting(unittest.TestCase):
    def setUp(self):
        self.node_class = GearRetracting

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts,[
                         ('Gear Up In Transit', 'Airborne')])

    def test_derive(self):
        up_trans = M('Gear Up In Transit', array=np.ma.concatenate((np.zeros(5), np.ones(10), np.zeros(45))),
                      values_mapping={0: '-', 1: 'Retracting'})
        airs = buildsection('Airborne', 2, 58)

        node = self.node_class()
        node.derive(up_trans, airs)
        expected=buildsection('Gear Retracting', 5, 15)
        self.assertEqual(node.get_slices(), expected.get_slices())


class TestGoAroundAndClimbout(unittest.TestCase):

    def test_can_operate(self):
        node = GoAroundAndClimbout
        start_stop = A('Segment Type', 'START_AND_STOP')
        ground_only = A('Segment Type', 'GROUND_ONLY')
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases','Level Flight'),
                                             seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases',),
                                             seg_type=start_stop))
        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases',),
                                              seg_type=ground_only))

    def test_go_around_and_climbout_basic(self):
        # The Go-Around phase starts 500ft before the minimum altitude is
        # reached, and ends after 2000ft climb....
        height = np.ma.concatenate((np.arange(0,40), np.arange(40,20,-1), np.arange(20,45), np.ones(5) * 45, np.arange(45,0,-1))) * 100.0
        alt = Parameter('Altitude For Flight Phases', height)
        levels = buildsection('Level Flight', 85, 91)
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt, levels)
        expected = buildsection('Go Around And Climbout', 55, 80)
        self.assertEqual(len(ga_phase), 1)
        self.assertEqual(ga_phase.get_first().start_edge, expected[0].start_edge)
        self.assertEqual(ga_phase.get_first().stop_edge, expected[0].stop_edge)

    def test_go_around_and_climbout_level_off(self):
        # The Go-Around phase starts 500ft before the minimum altitude is
        # reached, and ends ... or until level-off.
        height = np.ma.concatenate((np.arange(0,40), np.arange(40,20,-1), np.arange(20,30), np.ones(5) * 30, np.arange(30,0,-1))) * 100.0
        alt = Parameter('Altitude For Flight Phases', height)
        levels = buildsection('Level Flight', 70, 76)
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt, levels)
        # Level flight reached at ~70
        expected = buildsection('Go Around And Climbout', 55, 68)
        self.assertEqual(len(ga_phase), 1)
        self.assertEqual(ga_phase.get_first().start_edge, expected[0].start_edge)
        self.assertEqual(ga_phase.get_first().stop_edge, expected[0].stop_edge)

    def test_go_around_and_climbout_phase_not_reaching_2000ft(self):
        '''
        down = np.ma.array(list(range(4000,1000,-490))+[1000]*7) - 4000
        up = np.ma.array([1000]*7+list(range(1000,4500,490))) - 1500
        ga_kti = KTI('Go Around', items=[KeyTimeInstance(index=7, name='Go Around')])
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(Parameter('Descend For Flight Phases',down),
                        Parameter('Climb For Flight Phases',up),
                        ga_kti)
        expected = buildsection('Go Around And Climbout', 4.9795918367346941,
                                12.102040816326531)
        self.assertEqual(len(ga_phase), 1)
        self.assertEqual(ga_phase.get_first().start_edge, expected[0].start_edge)
        self.assertEqual(ga_phase.get_first().stop_edge, expected[0].stop_edge)
        '''
        alt_aal = load(os.path.join(test_data_path, 'alt_aal_goaround.nod'))
        level_flights = SectionNode('Level Flight')
        level_flights.create_sections([
            slice(1629.0, 2299.0, None),
            slice(3722.0, 4708.0, None),
            slice(4726.0, 4807.0, None),
            slice(5009.0, 5071.0, None),
            slice(5168.0, 6883.0, None),
            slice(8433.0, 9058.0, None)])
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt_aal, level_flights)
        self.assertEqual(len(ga_phase), 3)
        self.assertAlmostEqual(ga_phase[0].slice.start, 3586, places=0)
        self.assertAlmostEqual(ga_phase[0].slice.stop, 3722, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.start, 4895, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.stop, 5009, places=0)
        self.assertAlmostEqual(ga_phase[2].slice.start, 7124, places=0)
        self.assertAlmostEqual(ga_phase[2].slice.stop, 7265, places=0)

    def test_go_around_and_climbout_below_3000ft(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'GoAroundAndClimbout_AltitudeAAL.nod'))
        level_flights = load(os.path.join(test_data_path,
                                          'GoAroundAndClimbout_LevelFlights.nod'))
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt_aal, level_flights)
        self.assertEqual(len(ga_phase), 1)
        self.assertAlmostEqual(ga_phase[0].slice.start, 10697, places=0)
        self.assertAlmostEqual(ga_phase[0].slice.stop, 10968, places=0)

    def test_go_around_and_climbout_real_data(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'GoAroundAndClimbout_alt_aal.nod'))

        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt_aal, None)
        self.assertEqual(len(ga_phase), 5)
        #self.assertEqual(ga_phase.get_slices(), [
            #slice(1005.0, 1170.0, None),
            #slice(1378.0, 1502.0, None),
            #slice(1676.0, 1836.0, None),
            #slice(2021.0, 2206.0, None),
            #slice(2208.0, 2502.0, None)])
        self.assertAlmostEqual(ga_phase[0].slice.start, 1057, places=0)
        self.assertAlmostEqual(ga_phase[0].slice.stop, 1169, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.start, 1393, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.stop, 1502, places=0)
        # For some reason, places=0 breaks for 1721.5 == 1721 or 1722..
        self.assertAlmostEqual(ga_phase[2].slice.start, 1722, delta=0.5)
        self.assertAlmostEqual(ga_phase[2].slice.stop, 1836, places=0)
        self.assertAlmostEqual(ga_phase[3].slice.start, 2071, places=0)
        self.assertAlmostEqual(ga_phase[3].slice.stop, 2205, places=0)
        self.assertAlmostEqual(ga_phase[4].slice.start, 2392, places=0)
        self.assertAlmostEqual(ga_phase[4].slice.stop, 2502, places=0)

    def test_two_go_arounds_for_atr42(self):
        alt_aal = load(os.path.join(test_data_path,
                                    'AltitudeAAL_ATR42_two_goarounds.nod'))
        ga_phase = GoAroundAndClimbout()
        ga_phase.derive(alt_aal, None)
        self.assertEqual(len(ga_phase), 2)
        self.assertAlmostEqual(ga_phase[0].slice.start, 10703, places=0)
        self.assertAlmostEqual(ga_phase[0].slice.stop, 10948, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.start, 12529, places=0)
        self.assertAlmostEqual(ga_phase[1].slice.stop, 12749, places=0)



class TestHolding(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(
            Holding.get_operational_combinations(ac_type=aeroplane),
            [('Altitude AAL For Flight Phases',
             'Heading Continuous', 'Altitude Max',
             'Latitude Smoothed', 'Longitude Smoothed')])

    def test_straightish_not_detected(self):
        rot=np.ma.concatenate((
            np.zeros(600),
            np.tile(np.concatenate((np.ones(60) * 0.3, np.zeros(60))), 6),
            np.zeros(2180),
            np.tile(np.concatenate((np.ones(60) * 0.6, np.zeros(60))), 6),
            np.zeros(2180),
            np.tile(np.concatenate((np.ones(120) * 0.5, np.zeros(90))), 6),
            np.zeros(2180),
            np.tile(np.concatenate((np.ones(120) * 0.5, np.zeros(120))), 6),
            np.zeros(600),
        ))
        alt=P('Altitude AAL For Flight Phases', np.ones(11880) * 4000)
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt_max=KPV('Altitude Max', items=[
            KeyPointValue(index=200, value=40000.0),])
        lat=P('Latitude Smoothed', np.ma.ones(11880) * 24.0)
        lon=P('Longitude Smoothed', np.ma.ones(11880) * 24.0)
        hold=Holding()
        hold.derive(alt, hdg, alt_max, lat, lon)
        self.assertEqual(len(hold), 0)

    def test_rejected_outside_height_range(self):
        rot=np.ma.concatenate((
            np.zeros(600),
            np.tile(np.concatenate((np.ones(60) * 3.0, np.zeros(60))), 6),
            np.zeros(600),
        ))
        alt=P('Altitude AAL For Flight Phases', np.ones(11880) * 4000)
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt_max=KPV('Altitude Max', items=[
            KeyPointValue(index=1700, value=40.0),])
        lat=P('Latitude Smoothed', np.ma.ones(1920) * 24.0)
        lon=P('Longitude Smoothed', np.ma.ones(1920) * 24.0)
        hold=Holding()
        hold.derive(alt, hdg, alt_max, lat, lon)
        self.assertEqual(len(hold), 0)

    def test_hold_detected(self):
        rot=np.ma.concatenate((
            np.zeros(600),
            np.tile(np.concatenate((np.ones(60) * 3, np.zeros(60))), 6),
            np.zeros(2180),
            np.tile(np.concatenate((np.ones(60) * 3, np.zeros(60))), 6),
            np.zeros(2180),
            np.tile(np.concatenate((np.ones(120) * 1.5, np.zeros(120))), 6),
            np.zeros(2180),
            np.tile(np.concatenate((np.ones(120) * 1.5, np.zeros(120))), 6),
            np.zeros(600),
        ))
        alt=P('Altitude AAL For Flight Phases', np.ones(11880) * 4000)
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt_max=KPV('Altitude Max', items=[
            KeyPointValue(index=200, value=40000.0),])
        lat=P('Latitude Smoothed', np.ma.ones(11880) * 24.0)
        lon=P('Longitude Smoothed', np.ma.ones(11880) * 24.0)
        hold=Holding()
        hold.derive(alt, hdg, alt_max, lat, lon)
        self.assertAlmostEqual(hold[0].slice.start, 600, delta=3)
        self.assertAlmostEqual(hold[0].slice.stop, 1260, delta=3)
        self.assertAlmostEqual(hold[1].slice.start, 3500, delta=3)
        self.assertAlmostEqual(hold[1].slice.stop, 4160, delta=3)
        self.assertAlmostEqual(hold[2].slice.start, 6400, delta=3)
        self.assertAlmostEqual(hold[2].slice.stop, 7720, delta=3)
        self.assertAlmostEqual(hold[3].slice.start, 10020, delta=3)
        self.assertAlmostEqual(hold[3].slice.stop, 11340, delta=3)

    def test_hold_detected_2_hz(self):
        rot=np.ma.concatenate((
            np.zeros(1200),
            np.tile(np.concatenate((np.ones(120) * 1.5, np.zeros(120))), 6),
            np.zeros(4360),
            np.tile(np.concatenate((np.ones(120) * 1.5, np.zeros(120))), 6),
            np.zeros(4360),
            np.tile(np.concatenate((np.ones(240) * 0.75, np.zeros(240))), 6),
            np.zeros(4360),
            np.tile(np.concatenate((np.ones(240) * 0.75, np.zeros(240))), 6),
            np.zeros(1200),
        ))
        alt=P('Altitude AAL For Flight Phases', np.ones(len(rot)) * 4000)
        hdg=P('Heading Increasing', integrate(rot, 1.0), frequency=2.0)
        alt_max=KPV('Altitude Max', items=[
            KeyPointValue(index=400, value=40000.0),])
        lat=P('Latitude Smoothed', np.ma.ones(11880*2) * 24.0)
        lon=P('Longitude Smoothed', np.ma.ones(11880*2) * 24.0)
        hold=Holding()
        hold.derive(alt, hdg, alt_max, lat, lon)
        self.assertAlmostEqual(hold[0].slice.start, 600 * 2, delta=3)
        self.assertAlmostEqual(hold[0].slice.stop, 1260 * 2, delta=3)
        self.assertAlmostEqual(hold[1].slice.start, 3500 * 2, delta=3)
        self.assertAlmostEqual(hold[1].slice.stop, 4160 * 2, delta=3)
        self.assertAlmostEqual(hold[2].slice.start, 6400 * 2, delta=3)
        self.assertAlmostEqual(hold[2].slice.stop, 7720 * 2, delta=3)
        self.assertAlmostEqual(hold[3].slice.start, 10020 * 2, delta=3)
        self.assertAlmostEqual(hold[3].slice.stop, 11340 * 2, delta=3)

    def test_hold_rejected_if_travelling(self):
        rot=np.ma.concatenate((
            np.zeros(600),
            np.tile(np.concatenate((np.ones(60) * 3.0, np.zeros(60))), 6),
            np.zeros(600),
        ))
        alt=P('Altitude AAL For Flight Phases', np.ones(11880) * 4000)
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt_max=KPV('Altitude Max', items=[
            KeyPointValue(index=100, value=40.0),])
        lat=P('Latitude Smoothed', np.ma.ones(1920) * 24.0)
        lon=P('Longitude Smoothed', np.array(range(1920)) * 0.01)
        hold=Holding()
        hold.derive(alt, hdg, alt_max, lat, lon)
        self.assertEqual(len(hold), 0)

    def test_single_turn_rejected(self):
        rot=np.ma.concatenate((
            np.zeros(600),
            np.tile(np.concatenate((np.ones(60) * 3.0, np.zeros(60))), 1),
            np.zeros(600),
        ))
        alt=P('Altitude AAL For Flight Phases', np.ones(11880) * 4000)
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt_max=KPV('Altitude Max', items=[
            KeyPointValue(index=100, value=40.0),])
        lat=P('Latitude Smoothed', np.ma.ones(1920) * 24.0)
        lon=P('Longitude Smoothed', np.ma.ones(1920) * 24.0)
        hold=Holding()
        hold.derive(alt, hdg, alt_max, lat, lon)
        self.assertEqual(len(hold), 0)

    def test_slow_turns_on_limit_of_calculation(self):
        alt=P('Altitude AAL For Flight Phases', np.ones(15488) * 20000)
        hdg_array = load_compressed(os.path.join(test_data_path, 'holding_high_over_florida.npz'))
        hdg = P('Heading Increasing', hdg_array)
        alt_max=KPV('Altitude Max', items=[
            KeyPointValue(index=2030, value=36015.0),])
        lat=P('Latitude Smoothed', np.ma.ones(15488) * 24.0)
        lon=P('Longitude Smoothed', np.ma.ones(15488) * 24.0)
        hold=Holding()
        hold.derive(alt, hdg, alt_max, lat, lon)
        self.assertEqual(len(hold), 1)

    def test_hold_detected_with_masked_lat_long(self):
        rot=np.ma.concatenate((
            np.zeros(600),
            np.tile(np.concatenate((np.ones(60) * 3, np.zeros(60))), 6),
            np.zeros(2180),
            np.tile(np.concatenate((np.ones(60) * 3, np.zeros(60))), 6),
            np.zeros(2180),
            np.tile(np.concatenate((np.ones(120) * 1.5, np.zeros(120))), 6),
            np.zeros(2180),
            np.tile(np.concatenate((np.ones(120) * 1.5, np.zeros(120))), 6),
            np.zeros(600),
        ))
        alt=P('Altitude AAL For Flight Phases', np.ones(11880) * 4000)
        hdg=P('Heading Increasing', integrate(rot,1.0))
        alt_max=KPV('Altitude Max', items=[
            KeyPointValue(index=200, value=40000.0),])
        lat=P('Latitude Smoothed', np.ma.ones(11880) * 24.0)
        lat.array[:] = np.ma.masked
        lon=P('Longitude Smoothed', np.ma.ones(11880) * 24.0)
        lon.array[:] = np.ma.masked
        hold=Holding()
        hold.derive(alt, hdg, alt_max, lat, lon)
        self.assertAlmostEqual(hold[0].slice.start, 600, delta=3)
        self.assertAlmostEqual(hold[0].slice.stop, 1260, delta=3)
        self.assertAlmostEqual(hold[1].slice.start, 3500, delta=3)
        self.assertAlmostEqual(hold[1].slice.stop, 4160, delta=3)
        self.assertAlmostEqual(hold[2].slice.start, 6400, delta=3)
        self.assertAlmostEqual(hold[2].slice.stop, 7720, delta=3)
        self.assertAlmostEqual(hold[3].slice.start, 10020, delta=3)
        self.assertAlmostEqual(hold[3].slice.stop, 11340, delta=3)


class TestLanding(unittest.TestCase):
    def test_can_operate(self):
        node = Landing
        start_stop = A('Segment Type', 'START_AND_STOP')
        ground_only = A('Segment Type', 'GROUND_ONLY')
        self.assertTrue(node.can_operate(('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast',),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertTrue(node.can_operate(('Heading Continuous', 'Altitude AAL For Flight Phases',),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases', 'Fast'),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertTrue(node.can_operate(('Altitude AAL For Flight Phases'),
                                         ac_type=aeroplane, seg_type=start_stop))
        self.assertFalse(node.can_operate(('Altitude AAL For Flight Phases'),
                                          ac_type=aeroplane, seg_type=ground_only))
        self.assertTrue(node.can_operate(('Altitude AGL', 'Collective', 'Airborne'),
                                         ac_type=helicopter, seg_type=start_stop))

    def test_landing_aeroplane_basic(self):
        head = np.ma.array([20]*8+[10,0])
        alt_aal = np.ma.array([100,80,60,40,20]+[0]*6)
        phase_fast = buildsection('Fast', 0, 5)
        landing = Landing()
        landing.derive(aeroplane,
                       P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases', alt_aal),
                       phase_fast,
                       None)
        expected = buildsection('Landing', 2.5, 9)
        self.assertEqual(landing.get_slices(), expected.get_slices())

    def test_landing_aeroplane_turnoff(self):
        head = np.ma.array([20]*15+list(range(20,0,-2)))
        alt_aal = np.ma.array([100,80,60,40,20]+[0]*26)
        phase_fast = buildsection('Fast',0,5)
        landing = Landing()
        landing.derive(aeroplane,
                       P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases', alt_aal),
                       phase_fast,
                       None)
        expected = buildsection('Landing', 2.5, 24)
        self.assertEqual(landing.get_slices(), expected.get_slices())

    def test_landing_aeroplane_turnoff_left(self):
        head = np.ma.array([20]*15+list(range(20,0,-2)))*-1.0
        alt_aal = np.ma.array([100,80,60,40,20]+[0]*26)
        phase_fast = buildsection('Fast', 0, 5)
        landing = Landing()
        landing.derive(aeroplane,
                       P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases', alt_aal),
                       phase_fast,
                       None)
        expected = buildsection('Landing', 2.5, 24)
        self.assertEqual(landing.get_slices(), expected.get_slices())

    def test_landing_aeroplane_with_multiple_fast(self):
        # ensure that the result is a single phase!
        head = np.ma.array([20]*15+list(range(20,0,-2)))
        alt_aal = np.ma.array(list(range(140,0,-10))+[0]*26)
        # first test the first section that is not within the landing heights
        phase_fast = buildsections('Fast', [2, 5], [7, 10])
        landing = Landing()
        landing.derive(aeroplane,
                       P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases', alt_aal),
                       phase_fast,
                       None)
        self.assertEqual(len(landing), 1)
        self.assertEqual(landing[0].slice.start, 9)
        self.assertEqual(landing[0].slice.stop, 24)

        # second, test both sections are within the landing section of data
        phase_fast = buildsections('Fast', [0, 12], [14, 15])
        landing = Landing()
        landing.derive(aeroplane,
                       P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases', alt_aal),
                       phase_fast,
                       None)
        self.assertEqual(len(landing), 1)
        self.assertEqual(landing[0].slice.start, 9)
        self.assertEqual(landing[0].slice.stop, 24)

    def test_landing_aeroplane_mobile_stop(self):
        alt_aal = load(os.path.join(test_data_path, 'Landing_alt_aal_1.nod'))
        head = load(os.path.join(test_data_path, 'Landing_head_1.nod'))
        mobile = buildsection('Mobile', 139, 7510)
        fast = buildsection('Fast', 399, 7438)
        landing = Landing()
        landing.derive(aeroplane, head, alt_aal, fast, mobile)
        self.assertEqual(len(landing), 1)
        self.assertAlmostEqual(landing[0].slice.start, 7434, places=0)
        self.assertAlmostEqual(landing[0].slice.stop, 7511, places=0)

    def test_landing_aeroplane_extended_slice(self):
        # slice needs to be slightly extended to pick up 50 ft transition.
        alt_aal = load(os.path.join(test_data_path, 'Landing_alt_aal_2.nod'))
        head = load(os.path.join(test_data_path, 'Landing_head_2.nod'))
        mobile = buildsection('Mobile', 84, 14426)
        fast = buildsection('Fast', 310, 14315)
        landing = Landing()
        landing.derive(aeroplane, head, alt_aal, fast, mobile)
        self.assertEqual(len(landing), 1)
        self.assertAlmostEqual(landing[0].slice.start, 14315, places=0)
        self.assertAlmostEqual(landing[0].slice.stop, 14375, places=0)

    def test_landing_helicopter_basic(self):
        alt_agl = P(name='Altitude AGL', array=np.ma.array([50]*5+list(range(45,0,-5))+[0]*5,dtype=float))
        coll = P(name='Collective', array=np.ma.array([53]*5+list(range(48,3,-5))+[3]*5,dtype=float))
        airs=buildsection('Airborne', 4, 15)
        node = Landing()
        node.derive(helicopter, None, None, None, None, alt_agl, coll, airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 10)
        self.assertEqual(node[0].slice.stop, 15)

    def test_derive__aeroplane_short_goaround(self):
        # reflective of real jetstream goaround

        alt_aal = np.ma.concatenate((np.arange(10000, 0, -25), np.arange(0, 1500, 25), np.arange(1500, 0, -25),[0]*80))  # len 600
        head = np.ma.array([-50]*570+list(range(-50,0,5))+[0]*20)
        phase_fast = buildsection('Fast', 0, 550)
        landing = Landing()
        landing.derive(aeroplane,
                       P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases', alt_aal),
                       phase_fast,
                       None)
        expected = buildsection('Landing', 518, 579)
        self.assertEqual(landing.get_slices(), expected.get_slices())


class TestLandingRoll(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(LandingRoll.get_operational_combinations(ac_type=aeroplane),
                         [('Groundspeed', 'Landing'),
                          ('Airspeed True', 'Landing'),
                          ('Pitch', 'Groundspeed', 'Landing'),
                          ('Pitch', 'Airspeed True', 'Landing'),
                          ('Groundspeed', 'Airspeed True', 'Landing'),
                          ('Pitch', 'Groundspeed', 'Airspeed True', 'Landing')])
        self.assertEqual(LandingRoll.get_operational_combinations(ac_type=helicopter), [])

class TestMobile(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Mobile
        self.operational_combinations = [
            ('Heading Rate',),
            ('Heading Rate', 'Groundspeed'),
            ('Heading Rate', 'Airborne'),
            ('Heading Rate', 'Fast'),
            ('Heading Rate', 'Groundspeed', 'Airborne'),
            ('Heading Rate', 'Groundspeed', 'Fast'),
            ('Heading Rate', 'Airborne', 'Fast'),
            ('Heading Rate', 'Groundspeed', 'Airborne', 'Fast'),
        ]

    def test_gspd(self):
        rot = np.ma.array([0, 0, 0, 0, 0, 0, 0])
        gspd = np.ma.array([1, 2, 3, 5, 5, 2, 1])
        airs = buildsection('Airborne',3,5)
        move = Mobile()
        move.derive(P('Heading Rate', rot), P('Groundspeed', gspd), airs, None)
        expected = buildsection('Mobile', 1, 6)
        self.assertEqual(move.get_slices(), expected.get_slices())

    def test_rot(self):
        rot = np.ma.array([0, 4, 8, 8, 8, 4, 0])
        gspd = np.ma.array([0, 0, 0, 0, 0, 0, 0])
        airs = buildsection('Airborne',3,5)
        move = Mobile()
        move.derive(P('Heading Rate', rot), P('Groundspeed', gspd), airs, None)
        expected = buildsection('Mobile', 1, 6)
        self.assertEqual(move.get_slices(), expected.get_slices())

    def test_airborne(self):
        rot = np.ma.array([0, 0, 0, 0, 0, 0, 0])
        gspd = np.ma.array([0, 0, 0, 0, 0, 0, 0])
        airs = buildsection('Airborne',3,5)
        move = Mobile()
        move.derive(P('Heading Rate', rot), P('Groundspeed', gspd), airs, None)
        expected = buildsection('Mobile', 3, 6)
        self.assertEqual(move.get_slices(), expected.get_slices())

    def test_mobile_from_start(self):
        rot = np.ma.array([0, 0, 0, 0, 0, 0, 0])
        gspd = np.ma.array([2, 2, 3, 5, 2, 0, 0])
        airs = buildsection('Airborne',2,4)
        move = Mobile()
        move.derive(P('Heading Rate', rot), P('Groundspeed', gspd), airs, None)
        expected = buildsection('Mobile', 0, 5)
        self.assertEqual(move.get_slices(), expected.get_slices())

    def test_no_gs(self):
        rot = np.ma.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        airs = buildsection('Airborne', 2, 4)
        fast = buildsection('Fast', 1, 6)
        move = Mobile()
        move.derive(P('Heading Rate', rot), None, airs, fast)
        expected = buildsection('Mobile', 1, 7)
        self.assertEqual(move.get_slices(), expected.get_slices())


class TestLevelFlight(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = LevelFlight
        self.operational_combinations = [('Airborne', 'Vertical Speed For Flight Phases',
                                          'Altitude AAL')]

    def test_level_flight_phase_basic(self):
        data = list(range(0, 400, 1)) + list(range(400, -450, -1)) +\
            list(range(-450, 50, 1))
        vrt_spd = Parameter(
            name='Vertical Speed For Flight Phases',
            array=np.ma.array(data),
        )
        alt_aal = Parameter('Altitude AAL', np_ma_ones_like(vrt_spd.array) * 1000.0)
        airborne = SectionNode('Airborne', items=[
            Section('Airborne', slice(0, 3600, None), 0, 3600),
        ])
        level = LevelFlight()
        level.derive(airborne, vrt_spd, alt_aal)
        self.assertEqual(level, [
            Section('Level Flight', slice(0, 301, None), 0, 301),
            Section('Level Flight', slice(500, 1101, None), 500, 1101),
            Section('Level Flight', slice(1400, 1750, None), 1400, 1750)])

    def test_level_flight_phase_not_airborne_basic(self):
        data = list(range(0, 400, 1)) + list(range(400, -450, -1)) +\
            list(range(-450, 50, 1))
        vrt_spd = Parameter(
            name='Vertical Speed For Flight Phases',
            array=np.ma.array(data),
        )
        alt_aal = Parameter('Altitude AAL', np_ma_ones_like(vrt_spd.array) * 1000.0)
        airborne = SectionNode('Airborne', items=[
            Section('Airborne', slice(550, 1200, None), 550, 1200),
        ])
        level = LevelFlight()
        level.derive(airborne, vrt_spd, alt_aal)
        self.assertEqual(level, [
            Section('Level Flight', slice(550, 1101, None), 550, 1101)
        ])

    def test_rejects_short_segments(self):
        data = [400]*50+[0]*20+[400]*50+[0]*80+[-400]*40+[4]*10+[500]*70
        vrt_spd = Parameter(
            name='Vertical Speed For Flight Phases',
            array=np.ma.array(data),
            frequency=1.0
        )
        alt_aal = Parameter('Altitude AAL', np_ma_ones_like(vrt_spd.array) * 1000.0)
        airborne = SectionNode('Airborne', items=[
            Section('Airborne', slice(0, 320), 0, 320),
        ])
        level = LevelFlight()
        level.derive(airborne, vrt_spd, alt_aal)
        self.assertEqual(level, [
            Section('Level Flight', slice(120, 200, None), 120, 200)
        ])

    def test_rejects_on_gound(self):
        aal = Parameter('Altitude AAL', array=np.ma.array([200]*120 + [0]*60 + [200]*120))
        vs = Parameter('Vertical Speed For Flight Phases', array=np.ma.array([0.0]*280))
        airs=buildsection('Airborne', 1, 280)
        level = LevelFlight()
        level.derive(airs, vs, aal)
        self.assertEqual(level.get_slices(), [slice(1, 120, None), slice(180, 280, None)])


class TestStationary(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = Stationary
        self.operational_combinations = [('Groundspeed',)]

    def test_stationary_basic(self):
        data = [0]*30 + list(range(0, 60, 1)) +\
            list(range(60,0,-2)) + [0]*20
        gspd = Parameter('Groundspeed',array=np.ma.array(data))
        station = Stationary()
        station.derive(gspd)
        self.assertEqual(station[0].slice, slice(0, 32, None))
        self.assertEqual(station[1].slice, slice(120, 140, None))

    def test_stationary_reject_short_periods(self):
        data = [0]*10+[5]*4+[0]*6+[5]*6+[0]*14
        gspd = Parameter('Groundspeed',array=np.ma.array(data))
        station = Stationary()
        station.derive(gspd)
        self.assertEqual(len(station), 2)
        self.assertEqual(station[0].slice, slice(0, 20, None))
        self.assertEqual(station[1].slice, slice(26, 40, None))


class TestStraightAndLevel(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = StraightAndLevel
        self.operational_combinations = [('Level Flight', 'Heading')]

    def test_straight_and_level_basic(self):
        data = [0]*60+ list(range(0, 360, 1)) +\
            list(range(360,0,-2)) + [0]*120
        hdg = Parameter('Heading',array=np.ma.array(data, dtype=np.float64))
        level = buildsection('Level Flight', 0, 900)
        s_and_l = StraightAndLevel()
        s_and_l.derive(level, hdg)
        self.assertEqual(s_and_l[1].slice, slice(600, 720, None))
        self.assertEqual(s_and_l[0].slice, slice(0, 426, None))


class TestShuttlingApproach(unittest.TestCase):

    def setUp(self):
        self.node_class = ShuttlingApproach

    def test_can_operate(self):
        expected = [('Approach Information',)]
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_derive_one_shuttling_approach(self):

        approaches = App()
        approaches.create_approach('SHUTTLING',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('LANDING',
                                   slice(35, 48, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)
        node = self.node_class()
        node.derive(approaches)
        expected = slice(19, 29)

        self.assertEqual(len(node), 1)
        self.assertEqual('Shuttling Approach', node.get_name())
        self.assertEqual(expected, node.get_slices()[0])

    def test_derive_two_shuttling_approaches(self):

        approaches = App()
        approaches.create_approach('SHUTTLING',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('SHUTTLING',
                                   slice(35, 48, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        node = self.node_class()
        node.derive(approaches)
        expected = [slice(19, 29), slice(35, 48)]

        self.assertEqual(len(node), 2)
        self.assertEqual('Shuttling Approach', node.get_name())
        self.assertEqual(expected[0], node.get_slices()[0])
        self.assertEqual('Shuttling Approach', node.get_name())
        self.assertEqual(expected[1], node.get_slices()[1])


    def test_no_shuttling_approaches(self):

        approaches = App()
        approaches.create_approach('LANDING',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        node = self.node_class()
        node.derive(approaches)

        self.assertEqual(len(node), 0)


class TestRejectedTakeoff(unittest.TestCase):
    '''
    The test was originally written for an acceleration threshold of 0.1g.
    This was later increased to 0.15g and the test amended to match that
    scaling, hence the *1.5 factors.
    '''
    def test_can_operate(self):
        expected = [
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running',
             'Grounded', 'Takeoff Runway Heading', 'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running',
             'Grounded', 'Eng (*) N1 Max', 'Takeoff Runway Heading', 'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running',
             'Grounded', 'Takeoff Acceleration Start', 'Takeoff Runway Heading', 'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running',
             'Grounded', 'Takeoff Runway Heading', 'Segment Type', 'Takeoff And Go Around'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded',
             'Eng (*) N1 Max', 'Takeoff Acceleration Start', 'Takeoff Runway Heading', 'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running',
             'Grounded', 'Eng (*) N1 Max', 'Takeoff Runway Heading', 'Segment Type', 'Takeoff And Go Around'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running',
             'Grounded', 'Takeoff Acceleration Start', 'Takeoff Runway Heading', 'Segment Type',
             'Takeoff And Go Around'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running',
             'Grounded', 'Eng (*) N1 Max', 'Takeoff Acceleration Start', 'Takeoff Runway Heading',
             'Segment Type', 'Takeoff And Go Around')
        ]
        self.assertEqual(
            RejectedTakeoff.get_operational_combinations(
                seg_type=A('Segment Type', 'START_AND_STOP')),
            expected
        )

        expected = [
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Eng (*) N1 Max',
             'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded',
             'Takeoff Acceleration Start', 'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Takeoff Runway Heading',
             'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Segment Type',
             'Takeoff And Go Around'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Eng (*) N1 Max',
             'Takeoff Acceleration Start', 'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Eng (*) N1 Max',
             'Takeoff Runway Heading', 'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Eng (*) N1 Max',
             'Segment Type', 'Takeoff And Go Around'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded',
             'Takeoff Acceleration Start', 'Takeoff Runway Heading', 'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded',
             'Takeoff Acceleration Start', 'Segment Type', 'Takeoff And Go Around'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Takeoff Runway Heading',
             'Segment Type', 'Takeoff And Go Around'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Eng (*) N1 Max',
             'Takeoff Acceleration Start', 'Takeoff Runway Heading', 'Segment Type'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Eng (*) N1 Max',
             'Takeoff Acceleration Start', 'Segment Type', 'Takeoff And Go Around'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Eng (*) N1 Max',
             'Takeoff Runway Heading', 'Segment Type', 'Takeoff And Go Around'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded',
             'Takeoff Acceleration Start', 'Takeoff Runway Heading', 'Segment Type', 'Takeoff And Go Around'),
            ('Acceleration Longitudinal Offset Removed', 'Eng (*) All Running', 'Grounded', 'Eng (*) N1 Max',
             'Takeoff Acceleration Start', 'Takeoff Runway Heading', 'Segment Type', 'Takeoff And Go Around'
            )
        ]
        self.assertEqual(
            RejectedTakeoff.get_operational_combinations(
                seg_type=A('Segment Type', 'GROUND_ONLY')),
            expected
        )

        self.assertEqual(
            RejectedTakeoff.get_operational_combinations(),
            []
        )

    def test_derive_one_rejected_takeoff(self):
        accel_lon = P('Acceleration Longitudinal Offset Removed',
                      np.ma.array([0] * 3 + [0.02, 0.05, 0.02, 0, -0.17,] + [0] * 7 +
                                  [0.2, 0.4, 0.1] + [0.11] * 4 + [0] * 6 + [-2] +
                                  [0] * 5 + [0.02, 0.08, 0.08, 0.08, 0.08] + [0] * 20)*1.5)
        grounded = buildsections('Grounded', [0,len(accel_lon.array)/2.0],[len(accel_lon.array)/2.0, len(accel_lon.array)])
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                        values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading',[0, 30])

        node = RejectedTakeoff()
        # Set a low frequency to pass slice duration checks.
        node.frequency = 1/64.0
        node.derive(accel_lon, eng_running, grounded, None, None,
                    toff_rwy_hdg, A('Segment Type', 'START_AND_STOP'), None)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 15, 0)
        self.assertAlmostEqual(node[0].slice.stop, 27, 0)

    def test_derive_two_rejected_takeoffs(self):

        accel_lon = P('Acceleration Longitudinal Offset Removed',
                      np.ma.array([0] * 3 + [0.02, 0.05, 0.11, 0, -0.17,] + [0] * 7 +
                                  [0.2, 0.4, 0.1] + [0.11] * 4 + [0] * 6 + [-2] +
                                  [0] * 5 + [0.02, 0.08, 0.08, 0.08, 0.08] + [0] * 20)*1.5)
        grounded = buildsections('Grounded', [0,len(accel_lon.array)/2.0],[len(accel_lon.array)/2.0, len(accel_lon.array)])
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                            values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading',[0, 30])

        node = RejectedTakeoff()
        # Set a low frequency to pass slice duration checks.
        node.frequency = 1/64.0
        node.derive(accel_lon, eng_running, grounded, None, None,
                    toff_rwy_hdg, A('Segment Type', 'START_AND_STOP'), None)
        self.assertEqual(len(node), 2)
        self.assertAlmostEqual(node[0].slice.start, 5, 0)
        self.assertAlmostEqual(node[0].slice.stop, 6, 0)
        self.assertAlmostEqual(node[1].slice.start, 15, 0)
        self.assertAlmostEqual(node[1].slice.stop, 27, 0)


    def test_derive_one_rejected_takeoff_with_two_acceleration_spikes(self):
        accel_lon = P('Acceleration Longitudinal Offset Removed',
                          np.ma.array([0] * 3 + [0.02, 0.05, 0.02, 0, -0.17,] + [0] * 7 +
                                      [0.2, 0.4, 0.1] + [0.11] * 4 + [0] * 6 + [0.2, 0.4, 0.1] +
                                      [0.11] * 4 + [0] * 6  + [-0.2] + [0] * 5 +
                                      [0.02, 0.08, 0.08, 0.08, 0.08] + [0] * 20)*1.5)
        grounded = buildsections('Grounded', [0,len(accel_lon.array) - 20],[len(accel_lon.array) - 20, len(accel_lon.array)])
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                            values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading',[0, 50])

        node = RejectedTakeoff()
        # Set a low frequency to pass slice duration checks.
        node.frequency = 1/64.0
        node.derive(accel_lon, eng_running, grounded, None, None,
                    toff_rwy_hdg, A('Segment Type', 'START_AND_STOP'), None)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 15, 0)
        self.assertAlmostEqual(node[0].slice.stop, 40, 0)

    def test_derive_flight_with_rejected_takeoff_1(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_2.nod'))
        accel_lon.array *= 1.5
        grounded = load(os.path.join(test_data_path,
                                     'RejectedTakeoff_Grounded_2.nod'))
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                        values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading',
                                     [3500, 4000], [5000, 5447])
        node = RejectedTakeoff()
        node.derive(accel_lon, eng_running, grounded, None, None,
                    toff_rwy_hdg, A('Segment Type', 'START_AND_STOP'), None)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 3622, 0)
        self.assertAlmostEqual(node[0].slice.stop, 3663, 0)

    def test_derive_flight_with_two_rejected_takeoff_1(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'rejectedTakeoffAccelLon.nod'))
        eng_running = load(os.path.join(test_data_path,
                                            'rejectedTakeoffEngRunning.nod'))
        groundeds = load(os.path.join(test_data_path,
                                     'rejectedTakeoffGroundeds.nod'))
        eng_n1 = load(os.path.join(test_data_path,
                                       'rejectedTakeoffEngN1.nod'))
        toff_rwy_hdg = buildsections('Takeoff Runway Heading',
                                     [2500, 4000], [7000,8000], [10000, 15305])
        node = RejectedTakeoff()
        node.derive(accel_lon, eng_running, groundeds, eng_n1, None,
                    toff_rwy_hdg, A('Segment Type', 'START_AND_STOP'), None)
        self.assertEqual(len(node), 2)
        self.assertAlmostEqual(node[0].slice.start, 2970, 0)
        self.assertAlmostEqual(node[0].slice.stop, 3015, 0)
        self.assertAlmostEqual(node[1].slice.start, 7689, 0)
        self.assertAlmostEqual(node[1].slice.stop, 7727, 0)

    def test_derive_flight_with_rejected_takeoff_2(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_5.nod'))
        accel_lon.array *= 1.5
        grounded = load(os.path.join(test_data_path,
                                     'RejectedTakeoff_Grounded_5.nod'))
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                            values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading',
                                     [2300, 2500], [3000, 3731])
        node = RejectedTakeoff()
        node.derive(accel_lon, eng_running, grounded, None, None,
                    toff_rwy_hdg, A('Segment Type', 'START_AND_STOP'), None)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 2383, 0)
        self.assertAlmostEqual(node[0].slice.stop, 2413, 0)

    def test_derive_flight_with_rejected_takeoff_short(self):
        '''
        test derived from genuine low speed rejected takeoff FDS hash 452728ea2768
        '''
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_Short.nod'))
        grounded = buildsections('Grounded', [0, 3796], [23516, 24576])
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                            values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading',
                                     [1900, 2000], [3500, 3796])
        node = RejectedTakeoff(frequency=4)
        node.derive(accel_lon, eng_running, grounded, None, None,
                    toff_rwy_hdg, A('Segment Type', 'START_AND_ONLY'), None)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 1917, 0)
        self.assertAlmostEqual(node[0].slice.stop, 1969, 0)

    def test_derive_flight_without_rejected_takeoff_3(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_4.nod'))
        grounded = load(os.path.join(test_data_path,
                                     'RejectedTakeoff_Grounded_4.nod'))
        accel_lon.array *= 1.5
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                            values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading', [2300, 2500], [3000, 3731])
        node = RejectedTakeoff()
        node.derive(accel_lon, eng_running, grounded, None, None, toff_rwy_hdg,
                    A('Segment Type','START_AND_STOP'), None)
        self.assertEqual(len(node), 0)

    def test_derive_flight_without_rejected_takeoff_1(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_1.nod'))
        grounded = load(os.path.join(test_data_path,
                                     'RejectedTakeoff_Grounded_1.nod'))
        accel_lon.array *= 1.5
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                            values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading', [2300, 2500], [3000, 3731])
        node = RejectedTakeoff()
        node.derive(accel_lon, eng_running, grounded, None, None, toff_rwy_hdg,
                    A('Segment Type', 'START_AND_STOP'), None)
        self.assertEqual(len(node), 0)

    def test_derive_flight_without_rejected_takeoff_2(self):
        accel_lon = load(os.path.join(
            test_data_path,
            'RejectedTakeoff_AccelerationLongitudinalOffsetRemoved_3.nod'))
        accel_lon.array *= 1.5
        grounded = load(os.path.join(test_data_path,
                                     'RejectedTakeoff_Grounded_3.nod'))
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                            values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading', [2300, 2500], [3000, 3731])
        node = RejectedTakeoff()
        node.derive(accel_lon, eng_running, grounded, None, None, toff_rwy_hdg,
                    A('Segment Type', 'START_AND_STOP'), None)
        self.assertEqual(len(node), 0)

    def test_derive_flight_with_toga_and_rejected_takeoff(self):
        accel_lon = P('Acceleration Longitudinal Offset Removed',
                      np.ma.array([0] * 3 + [0.02, 0.05, 0.02, 0, -0.17, -0.18, -0.19] + [0] * 5 +
                                  [0.2, 0.4, 0.1] + [0.11] * 4 + [0] * 6 + [-2] +
                                  [0] * 5 + [0.02, 0.08, 0.08, 0.08, 0.08] + [0] * 20)*1.5)
        grounded = buildsections('Grounded', [0,len(accel_lon.array)/2.0],[len(accel_lon.array)/2.0, len(accel_lon.array)])
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                        values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading', [0, 30])
        toga = M('Takeoff And Go Around', np.ma.array([1]*30 + [0]*29))
        eng_n1 = P('Eng (1) N1 Max', np.ma.array([0]*7 + [80]*3 + [0]*49))

        node = RejectedTakeoff()
        # Set a low frequency to pass slice duration checks.
        node.frequency = 1/64.0
        node.derive(accel_lon, eng_running, grounded, eng_n1, None,
                    toff_rwy_hdg, A('Segment Type', 'START_AND_STOP'), toga)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 7, 0)
        self.assertAlmostEqual(node[0].slice.stop, 10, 0)

    def test_derive_flight_with_toga_and_no_rejected_takeoff(self):
        accel_lon = P('Acceleration Longitudinal Offset Removed', np.ma.arange(0, 0.5, 0.1))
        grounded = buildsections('Grounded', [0,len(accel_lon.array)/2.0],[len(accel_lon.array)/2.0, len(accel_lon.array)])
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                        values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading', [0, 30])
        toga = M('Takeoff And Go Around', np.ma.array([1]*30 + [0]*20))
        eng_n1 = P('Eng (1) N1 Max', np.ma.array([0]*7 + [0]*52))

        node = RejectedTakeoff()
        # Set a low frequency to pass slice duration checks.
        node.frequency = 1/64.0
        node.derive(accel_lon, eng_running, grounded, eng_n1, None,
                    toff_rwy_hdg, A('Segment Type', 'START_AND_STOP'), toga)
        self.assertEqual(len(node), 0)

    def test_derive_toga_button_press(self):
        accel_lon = P('Acceleration Longitudinal Offset Removed',
                      np.ma.array([0] * 3 + [0.02, 0.05, 0.02, 0, -0.17, -0.18, -0.19] + [0] * 5 +
                                  [0.2, 0.4, 0.1] + [0.11] * 4 + [0] * 6 + [-2] +
                                  [0] * 5 + [0.02, 0.08, 0.08, 0.08, 0.08] + [0] * 20)*1.5)
        grounded = buildsections('Grounded', [0,len(accel_lon.array)/2.0],[len(accel_lon.array)/2.0, len(accel_lon.array)])
        eng_running = M('Eng (*) All Running', np_ma_ones_like(accel_lon.array),
                        values_mapping={0: 'Not Running', 1: 'Running'})
        toff_rwy_hdg = buildsections('Takeoff Runway Heading', [0, 30])
        toga = M('Takeoff And Go Around', np.ma.array([1]*1 + [0]*58))
        eng_n1 = P('Eng (*) N1 Max', np.ma.array([0]*7 + [80]*3 + [0]*49))

        node = RejectedTakeoff()
        node.frequency = 1
        node.derive(accel_lon, eng_running, grounded, eng_n1, None,
                    toff_rwy_hdg, A('Segment Type', 'START_AND_STOP'), toga)
        self.assertEqual(len(node), 0)


class TestTakeoff(unittest.TestCase):
    def test_can_operate(self):
        # Airborne dependency added to avoid trying to derive takeoff when
        # aircraft never airborne
        available = ('Heading Continuous', 'Altitude AAL For Flight Phases', 'Fast', 'Airborne')
        seg_type = A('Segment Type', 'START_AND_STOP')
        self.assertTrue(Takeoff.can_operate(available, seg_type=seg_type))
        seg_type.value = 'NO_MOVEMENT'
        self.assertFalse(Takeoff.can_operate(available, seg_type=seg_type))

    def test_takeoff_basic(self):
        head = np.ma.array([ 0,0,10,20,20,20,20,20,20,20,20])
        alt_aal = np.ma.array([0,0,0,0,0,0,0,0,10,30,70])
        phase_fast = buildsection('Fast', 6.5, 10)
        takeoff = Takeoff()
        takeoff.derive(P('Heading Continuous', head),
                       P('Altitude AAL For Flight Phases', alt_aal),
                       phase_fast)
        expected = buildsection('Takeoff', 1.5, 9.125)
        self.assertEqual(takeoff.get_slices(), expected.get_slices())

    def test_takeoff_basic_short(self):
        head = np.ma.array([ 0,0,10,20,20,20,20,20,20,20,20])
        alt_aal = np.ma.array([0,0,0,0,0,0,0,0,10,30,70])
        phase_fast = buildsection('Fast', 6.5, 16.5)
        takeoff = Takeoff()
        takeoff.derive(P('Heading Continuous', np.ma.concatenate((head, head[::-1]))),
                       P('Altitude AAL For Flight Phases', np.ma.concatenate((alt_aal, alt_aal[::-1]))),
                       phase_fast)
        expected = buildsection('Takeoff', 1.5, 9.125)
        self.assertEqual(takeoff.get_slices(), expected.get_slices())

    def test_takeoff_real_short(self):
        takeoff = Takeoff()
        takeoff.derive(
            P('Heading Continuous', load_compressed(os.path.join(test_data_path, 'Takeoff_HeadingContinuous_2.npz'))),
            P('Altitude AAL For Flight Phases',
              load_compressed(os.path.join(test_data_path, 'Takeoff_AltitudeAAL_2.npz'))),
            buildsection('Fast', 451, 2797),
            buildsection('Airborne', 461, 2783),
        )
        self.assertEqual(len(takeoff), 1)
        self.assertAlmostEqual(takeoff[0].slice.start, 426.01, places=2)
        self.assertAlmostEqual(takeoff[0].slice.stop, 463.22, places=2)

    def test_takeoff_with_zero_slices(self):
        '''
        A zero slice was causing the derive method to raise an exception.
        This test aims to replicate the problem, and shows that with a None
        slice an empty takeoff phase is produced.
        '''
        head = np.ma.array([0,0,10,20,20,20,20,20,20,20,20])
        alt_aal = np.ma.array([0,0,0,0,0,0,0,0,10,30,70])
        phase_fast = buildsection('Fast', None, None)
        takeoff = Takeoff()
        takeoff.derive(P('Heading Continuous',head),
                       P('Altitude AAL For Flight Phases', alt_aal),
                       phase_fast)
        expected = []
        self.assertEqual(takeoff.get_slices(), expected)

    def test_takeoff_alt_spike(self):
        '''
        Altitude spike below rate of change limit before liftoff was truncating
        Takeoff slice.
        '''
        takeoff = Takeoff(frequency=4)
        head = load(os.path.join(
            test_data_path,
            'Takeoff_HeadingContinuous_1.nod'))
        alt_aal = load(os.path.join(
            test_data_path,
            'Takeoff_AltitudeAAL_1.nod'))
        fast = buildsection('Fast', 14063, 107663)
        airs = buildsection('Airborne', 14187, 107591)
        takeoff.derive(head, alt_aal, fast, airs)
        slices = takeoff.get_slices()
        self.assertEqual(len(slices), 1)
        self.assertEqual(round(slices[0].start), 13923)
        self.assertEqual(round(slices[0].stop), 14200)


class TestTaxiOut(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Mobile', 'Takeoff'),
                    ('Mobile', 'Takeoff', 'First Eng Start Before Liftoff')]
        self.assertEqual(TaxiOut.get_operational_combinations(), expected)

    def test_taxi_out(self):
        gnd = buildsections('Mobile', [0, 1], [3, 9])
        toff = buildsection('Takeoff', 8, 11)
        first_eng_starts = KTI(
            'First Eng Start Before Liftoff',
            items=[KeyTimeInstance(2, 'First Eng Start Before Liftoff')])
        tout = TaxiOut()
        tout.derive(gnd, toff, first_eng_starts)
        expected = buildsection('Taxi Out', 4, 7)
        self.assertEqual(tout.get_slices(), expected.get_slices())
        first_eng_starts = KTI(
            'First Eng Start Before Liftoff',
            items=[KeyTimeInstance(5, 'First Eng Start Before Liftoff')])
        tout = TaxiOut()
        tout.derive(gnd, toff, first_eng_starts)
        expected = buildsection('Taxi Out', 5, 7)
        self.assertEqual(tout.get_slices(), expected.get_slices())

    def test_taxi_out_empty(self):
        gnd = buildsections('Mobile', [0, 1], [3, 9])
        toff = buildsection('Takeoff', 1, 11)
        first_eng_starts = KTI(
            'First Eng Start Before Liftoff',
            items=[KeyTimeInstance(2, 'First Eng Start Before Liftoff')])
        tout = TaxiOut()
        tout.derive(gnd, toff, first_eng_starts)
        self.assertEqual(len(tout), 0)

    def test_taxi_out_late_eng(self):
        first_eng_starts = KTI('First Eng Start Before Liftoff', items=[KeyTimeInstance(557, 'First Eng Start Before Liftoff')])
        gnd = buildsection('Mobile', 386, 3737)
        toff = buildsection('Takeoff', 527, 582)
        tout = TaxiOut()
        tout.derive(gnd, toff, first_eng_starts)
        self.assertEqual(len(tout), 1)
        self.assertEqual(tout[0].slice.start, 387)
        self.assertEqual(tout[0].slice.stop, 526)

    def test_taxi_out_empty_2(self):
        gnd = buildsection('Mobile', 4816, 6681)
        toff = buildsection('Takeoff', 4611, 4926)
        first_eng_starts = KTI('First Eng Start Before Liftoff', items=[KeyTimeInstance(4754, 'First Eng Start Before Liftoff')])
        tout = TaxiOut()
        tout.derive(gnd, toff, first_eng_starts)
        self.assertEqual(len(tout), 0)


class TestTaxiIn(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Mobile', 'Landing', 'Last Eng Stop After Touchdown')]
        self.assertEqual(TaxiIn.get_operational_combinations(), expected)

    def test_taxi_in_1(self):
        gnd = buildsections('Mobile', [7, 13], [16, 17])
        landing = buildsection('Landing', 5, 11)
        last_eng_stops = KTI(
            'Last Eng Stop After Touchdown',
            items=[KeyTimeInstance(15, 'Last Eng Stop After Touchdown')])
        t_in = TaxiIn()
        t_in.derive(gnd, landing, last_eng_stops)
        expected = buildsection('Taxi In', 12, 14)
        self.assertEqual(t_in.get_slices(),expected.get_slices())

    def test_taxi_in_2(self):
        gnd = buildsection('Mobile', 488, 7007)
        landing = buildsection('Landing', 3389, 3437)
        last_eng_stops = KTI(
            'Last Eng Stop After Touchdown',
            items=[KeyTimeInstance(3734, 'Last Eng Stop After Touchdown')])
        t_in = TaxiIn()
        t_in.derive(gnd, landing, last_eng_stops)
        self.assertEqual(len(t_in), 1)
        self.assertEqual(t_in[0].slice.start, 3438)
        self.assertEqual(t_in[0].slice.stop, 3734)

    def test_taxi_in_early_eng_stop(self):
        gnd = buildsection('Mobile', 139, 7510)
        landing = buildsection('Landing', 7434, 7510)
        last_eng_stops = KTI(
            'Last Eng Stop After Touchdown',
            items=[KeyTimeInstance(7634, 'Last Eng Stop After Touchdown')])
        t_in = TaxiIn()
        t_in.derive(gnd, landing, last_eng_stops)
        self.assertEqual(len(t_in), 0)


class TestTaxiing(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(Taxiing.get_operational_combinations(), [
            ('Mobile', 'Takeoff', 'Landing', 'Airborne'),
            ('Mobile', 'Groundspeed', 'Takeoff', 'Landing', 'Airborne'),
            ('Mobile', 'Takeoff', 'Landing', 'Rejected Takeoff', 'Airborne'),
            ('Mobile', 'Groundspeed', 'Takeoff', 'Landing', 'Rejected Takeoff', 'Airborne'),
        ])

    def test_taxiing_mobile_airborne(self):
        mobiles = buildsection('Mobile', 10, 90)
        airs = buildsection('Airborne', 20, 80)
        node = Taxiing()
        node.derive(mobiles, None, None, None, None, airs)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 10)
        self.assertEqual(node[0].slice.stop, 20)
        self.assertEqual(node[1].slice.start, 80)
        self.assertEqual(node[1].slice.stop, 90)

    def test_taxiing_mobile_takeoff_landing(self):
        mobiles = buildsection('Mobile', 10, 90)
        toffs = buildsection('Takeoff', 20, 30)
        lands = buildsection('Landing', 70, 80)
        airs = buildsection('Airborne', 25, 75)
        node = Taxiing()
        node.derive(mobiles, None, toffs, lands, None, airs)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 10)
        self.assertEqual(node[0].slice.stop, 20)
        self.assertEqual(node[1].slice.start, 80)
        self.assertEqual(node[1].slice.stop, 90)

    def test_taxiing_mobile_takeoff_landing_gspd(self):
        mobiles = buildsection('Mobile', 10, 100)
        gspd_array = np.ma.concatenate((np.zeros(15), np.ones(70) * 10, np.zeros(5), np.ones(5) * 10, np.zeros(5)))
        gspd = P('Groundspeed', array=gspd_array)
        toffs = buildsection('Takeoff', 20, 30)
        lands = buildsection('Landing', 60, 70)
        airs = buildsection('Airborne', 25, 65)
        node = Taxiing()
        node.derive(mobiles, gspd, toffs, lands, None, airs)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].slice.start, 15)
        self.assertEqual(node[0].slice.stop, 20)
        self.assertEqual(node[1].slice.start, 70)
        self.assertEqual(node[1].slice.stop, 85)
        self.assertEqual(node[2].slice.start, 90)
        self.assertEqual(node[2].slice.stop, 95)


    def test_taxiing_including_rejected_takeoff(self):
        mobiles = buildsection('Mobile', 3, 100)
        gspd_array = np.ma.concatenate((np.zeros(10), np.ones(30) * 10, np.zeros(5), np.ones(50) * 10, np.zeros(5)))
        gspd = P('Groundspeed', array=gspd_array)
        toffs = buildsection('Takeoff',  50, 60)
        lands = buildsection('Landing', 80, 90)
        airs = buildsection('Airborne', 55, 85)
        rtos = buildsection('Rejected Takeoff', 20, 30)
        node = Taxiing()
        node.derive(mobiles, gspd, toffs, lands, rtos, airs)
        self.assertEqual(len(node), 4)
        self.assertEqual(node[0].slice.start, 10)
        self.assertEqual(node[0].slice.stop, 20)
        self.assertEqual(node[1].slice.start, 30)
        self.assertEqual(node[1].slice.stop, 40)
        self.assertEqual(node[2].slice.start, 45)
        self.assertEqual(node[2].slice.stop, 50)
        self.assertEqual(node[3].slice.start, 90)
        self.assertEqual(node[3].slice.stop, 95)


class TestTurningInAir(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Rate', 'Airborne', 'Aircraft Type')]
        self.assertEqual(TurningInAir.get_operational_combinations(), expected)

    def test_turning_in_air_phase_basic(self):
        rate_of_turn_data = np.arange(-4, 4.4, 0.4)
        rate_of_turn = Parameter('Heading Rate', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,21)
        turning_in_air = TurningInAir()
        turning_in_air.derive(rate_of_turn, airborne)
        expected = buildsections('Turning In Air',[0, 6],[16,21])
        self.assertEqual(turning_in_air.get_slices(), expected.get_slices())

    def test_turning_in_air_phase_with_mask(self):
        rate_of_turn_data = np.ma.arange(-4, 4.4, 0.4)
        rate_of_turn_data[6] = np.ma.masked
        rate_of_turn_data[16] = np.ma.masked
        rate_of_turn = Parameter('Heading Rate', np.ma.array(rate_of_turn_data))
        airborne = buildsection('Airborne',0,21)
        turning_in_air = TurningInAir()
        turning_in_air.derive(rate_of_turn, airborne)
        expected = buildsections('Turning In Air',[0, 6],[16,21])
        self.assertEqual(turning_in_air.get_slices(), expected.get_slices())


class TestTurningOnGround(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Rate', 'Taxiing')]
        self.assertEqual(TurningOnGround.get_operational_combinations(), expected)

    def test_turning_on_ground_phase_basic(self):
        rate_of_turn_data = np.ma.arange(-12, 12, 1)
        rate_of_turn = Parameter('Heading Rate', np.ma.array(rate_of_turn_data))
        grounded = buildsection('Grounded', 0, 24)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, grounded)
        expected = buildsections('Turning On Ground',[0, 7], [18,24])
        self.assertEqual(turning_on_ground.get_slices(), expected.get_slices())

    def test_turning_on_ground_phase_with_mask(self):
        rate_of_turn_data = np.ma.arange(-12, 12, 1)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Heading Rate', np.ma.array(rate_of_turn_data))
        grounded = buildsection('Grounded', 0, 24)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, grounded)
        # Masked inside is exclusive of the range outer limits, this behaviour
        # is not consistent with TurningInAir test which is inclusive of the
        # start of the range.
        expected = buildsections('Turning On Ground',[0, 7], [18,24])
        self.assertEqual(turning_on_ground.get_slices(), expected.get_slices())

    def test_turning_on_ground_after_takeoff_inhibited(self):
        rate_of_turn_data = np.ma.arange(-12, 12, 1)
        rate_of_turn_data[10] = np.ma.masked
        rate_of_turn_data[18] = np.ma.masked
        rate_of_turn = Parameter('Heading Rate', np.ma.array(rate_of_turn_data))
        grounded = buildsection('Grounded', 0,10)
        turning_on_ground = TurningOnGround()
        turning_on_ground.derive(rate_of_turn, grounded)
        expected = buildsections('Turning On Ground',[0, 7])
        self.assertEqual(turning_on_ground.get_slices(), expected.get_slices())


class TestDescentToFlare(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(DescentToFlare.get_operational_combinations(),
                         [('Descent', 'Altitude AAL For Flight Phases')])

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestEngHotelMode(unittest.TestCase):

    def setUp(self):
        self.node_class = EngHotelMode

    def test_can_operate(self):
        family = A('Family', value='ATR-42')
        available = ['Eng (2) Np', 'Eng (1) N1', 'Eng (2) N1', 'Grounded', 'Propeller Brake']
        self.assertTrue(self.node_class.can_operate(available, ac_type=aeroplane, family=family))
        self.assertFalse(self.node_class.can_operate(available, ac_type=helicopter, family=family))
        family.value = 'B737'
        self.assertFalse(self.node_class.can_operate(available, ac_type=aeroplane, family=family))

    def test_derive_basic(self):
        range_array = np.arange(0, 50, 5)
        eng2_n1 = Parameter('Eng (2) N1',
                            array=np.ma.concatenate((
                                range_array,
                                np.ones(40) * 50,
                                range_array[::-1],)))
        eng1_n1 = Parameter('Eng (1) N1', array=np.ma.concatenate(
            (np.zeros(10),
             range_array,
             np.ones(20) * 50,
             range_array[::-1],
             np.zeros(10))))
        eng2_np = Parameter('Eng (1) N1',
                            array=np.ma.concatenate((
                                np.zeros(20),
                                np.ones(20) * 100,
                                np.zeros(20))))
        groundeds = buildsections('Grounded', (0, 22), (38, None))
        prop_brake_values_mapping = {1: 'On', 0: '-'}
        prop_brake = M('Propeller Brake', array=np.ma.concatenate((np.ones(16), np.zeros(24), np.ones(20))), values_mapping=prop_brake_values_mapping)

        node = self.node_class()
        node.derive(eng2_np, eng1_n1, eng2_n1, groundeds, prop_brake)

        expected = [Section(name='Eng Hotel Mode', slice=slice(10, 16, None), start_edge=10, stop_edge=16),
                    Section(name='Eng Hotel Mode', slice=slice(42, 50, None), start_edge=42, stop_edge=50)]
        self.assertEqual(list(node), expected)

    def test_derive(self):
        eng2_np = load(os.path.join(test_data_path, 'eng_hotel_mode_eng2_np.nod'))
        eng1_n1 = load(os.path.join(test_data_path, 'eng_hotel_mode_eng1_n1.nod'))
        eng2_n1 = load(os.path.join(test_data_path, 'eng_hotel_mode_eng2_n1.nod'))
        prop_brake = load(os.path.join(test_data_path, 'eng_hotel_mode_prop_brake.nod'))
        groundeds = buildsections('Grounded', (0, 9203.6875), (17481.6875, 19011.6875))

        node = self.node_class()
        node.derive(eng2_np, eng1_n1, eng2_n1, groundeds, prop_brake)

        self.assertEqual(node.get_slices(), [slice(7760, 7784), slice(8248, 8632), slice(17696, 17770)])


class TestGearExtending(unittest.TestCase):
    def setUp(self):
        self.node_class = GearExtending

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations()
        self.assertEqual(opts,[
                         ('Gear Down In Transit', 'Airborne')])


    def test_derive(self):
        down_transit = M('Gear Down In Transit', array=np.ma.array([0]*45 + [1]*10 + [0]*5),
                 values_mapping={0: '-', 1: 'Extending'})
        airs = buildsection('Airborne', 2, 58)


        node = self.node_class()
        node.derive(down_transit, airs)
        expected=buildsection('Gear Extending', 45, 55)
        self.assertEqual(node.get_slices(), expected.get_slices())


class TestGearExtended(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GearExtended.get_operational_combinations(),
                         [('Gear Down',)])

    def test_basic(self):
        gear = M(
            name='Gear Down',
            array=np.ma.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]),
            values_mapping={0: 'Up', 1: 'Down'})
        gear_ext = GearExtended()
        gear_ext.derive(gear)
        self.assertEqual(gear_ext[0].slice, slice(0, 5))
        self.assertEqual(gear_ext[1].slice, slice(13,16))


class TestGearRetracted(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(GearRetracted.get_operational_combinations(),
                         [('Gear Up',)])

    def test_basic(self):
        gear = M(
            name='Gear Up',
            array=np.ma.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0]),
            values_mapping={0: 'Down', 1: 'Up'})
        gear_ext = GearRetracted()
        gear_ext.derive(gear)
        self.assertEqual(gear_ext[0].slice, slice(5, 14))


class TestGoAround5MinRating(unittest.TestCase):
    def setUp(self):
        self.node_class = GoAround5MinRating
        self.jet = A('Engine Propulsion', value='JET')
        self.prop = A('Engine Propulsion', value='PROP')
        self.ga = KTI('Go Around', items=[KeyTimeInstance(index=50)])
        self.duration = A('HDF Duration', value=360)

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Go Around', 'Eng (*) Np Avg', 'HDF Duration', 'Engine Propulsion'),
                                                    eng_type=self.prop))
        self.assertTrue(self.node_class.can_operate(('Go Around', 'HDF Duration')))

    def test_derive_prop(self):
        array = np.ma.array([70]*40 + [75, 80, 85, 90, 95] + [100]*120 + [95, 90, 85, 80] + [75]*191)
        eng_np = P('Eng (*) Np Avg', array=array)
        node = GoAround5MinRating()
        node.derive(self.ga, eng_np, self.duration,
                    eng_type=self.prop)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 50)
        self.assertEqual(node[0].slice.stop, 170)

    def test_derive_jet(self):
        array = np.ma.array([70]*40 + [75, 80, 85, 90, 95] + [100]*120 + [95, 90, 85, 80] + [75]*191)
        eng_np = P('Eng (*) Np Avg', array=array)
        node = GoAround5MinRating()
        node.derive(self.ga, eng_np, self.duration,
                    eng_type=self.jet)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 50)
        self.assertEqual(node[0].slice.stop, 350)


class TestMaximumContinuousPower(unittest.TestCase):
    def setUp(self):
        self.node_class = MaximumContinuousPower

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(),
                         [('Airborne', 'Takeoff 5 Min Rating', 'Go Around 5 Min Rating')])

    def test_derive(self):
        air = buildsection('Airborne', 25, 1000)
        toff = buildsection('Takeoff 5 Min Rating', 15, 100)
        ga = buildsection('Go Around 5 Min Rating', 500, 550)
        node = self.node_class()
        node.derive(air, toff, ga)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 100)
        self.assertEqual(node[0].slice.stop, 500)
        self.assertEqual(node[1].slice.start, 550)
        self.assertEqual(node[1].slice.stop, 1000)


class TestTakeoff5MinRating(unittest.TestCase):
    def setUp(self):
        self.node_class = Takeoff5MinRating
        self.jet = A('Engine Propulsion', value='JET')
        self.prop = A('Engine Propulsion', value='PROP')

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Takeoff Acceleration Start', 'Liftoff', 'Eng (*) Np Avg', 'Engine Propulsion', 'HDF Duration'), eng_type=self.prop))
        self.assertTrue(self.node_class.can_operate(('Takeoff Acceleration Start', 'HDF Duration'), eng_type=self.jet))
        self.assertTrue(self.node_class.can_operate(('Liftoff', 'HDF Duration'), ac_type=helicopter))

    def test_derive_basic_jet(self):
        toffs = KTI('Takeoff Acceleration Start',
                    items=[KeyTimeInstance(index=100)])
        duration = A('HDF Duration', value=1200)
        node = Takeoff5MinRating()
        node.derive(toffs, None, None, duration, eng_type=self.jet)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 100)
        self.assertEqual(node[0].slice.stop, 400)

    def test_derive_basic_prop(self):
        toffs = KTI('Takeoff Acceleration Start',
                    items=[KeyTimeInstance(index=20)])
        lifts = KTI('Liftoff',
                    items=[KeyTimeInstance(index=40)])
        duration = A('HDF Duration', value=189)
        array = np.ma.array([71]*20 + [68, 65, 80, 90] + [101]*60 + list(range(101, 84, -4)) + [86]*100)
        eng_np = P('Eng (*) Np Avg', array=array)
        node = Takeoff5MinRating()
        node.derive(toffs, lifts, eng_np, duration, eng_type=self.prop)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 20)
        self.assertEqual(node[0].slice.stop, 89)

    def test_derive_basic_helicopter(self):
        lifts = KTI('Liftoff',
                    items=[KeyTimeInstance(index=50),
                           KeyTimeInstance(index=340),
                           KeyTimeInstance(index=630),
                           KeyTimeInstance(index=980)])
        duration = A('HDF Duration', value=1200)
        node = Takeoff5MinRating()
        node.derive(None, lifts, None, duration, ac_type=helicopter)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 50)
        self.assertEqual(node[0].slice.stop, 930)
        self.assertEqual(node[1].slice.start, 980)
        self.assertEqual(node[1].slice.stop, 1200)

class TestTakeoffRoll(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffRoll.get_operational_combinations(),
                         [('Takeoff', 'Takeoff Acceleration Start',),('Takeoff', 'Takeoff Acceleration Start', 'Pitch',)])

    def test_derive(self):
        accel_start = KTI('Takeoff Acceleration Start', items=[
                    KeyTimeInstance(967.92513157006306, 'Takeoff Acceleration Start'),
                ])
        takeoffs = S(items=[Section('Takeoff', slice(953, 995), 953, 995)])
        pitch = load(os.path.join(test_data_path,
                                    'TakeoffRoll-pitch.nod'))
        node = TakeoffRoll()
        node.derive(toffs=takeoffs,
                   acc_starts=accel_start,
                   pitch=pitch)
        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].slice.start, 967.92, places=1)
        self.assertAlmostEqual(node[0].slice.stop, 990.27, places=1)


class TestTakeoffRollOrRejectedTakeoff(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(TakeoffRollOrRejectedTakeoff.get_operational_combinations(),
                         [('Takeoff Roll',),
                          ('Rejected Takeoff',),
                          ('Transition Hover To Flight',),
                          ('Takeoff Roll', 'Rejected Takeoff'),
                          ('Takeoff Roll', 'Transition Hover To Flight'),
                          ('Rejected Takeoff', 'Transition Hover To Flight'),
                          ('Takeoff Roll', 'Rejected Takeoff', 'Transition Hover To Flight')]
                         )

    def test_derive(self):
        rolls = buildsections('Takeoff Roll', [5,15], [105.2,115.4])
        rejs = buildsections('Rejected Takeoff', [50,65])
        expected = buildsections('Takeoff Roll Or Rejected Takeoff', [5,15], [50,65], [105.2,115.4])
        phase = TakeoffRollOrRejectedTakeoff()
        phase.derive(rolls, rejs)
        self.assertEqual([s.slice for s in sorted(list(expected))],
                         [s.slice for s in list(expected)])


class TestTakeoffRotation(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(TakeoffRotation.can_operate(('Liftoff',), ac_type=aeroplane))
        self.assertFalse(TakeoffRotation.can_operate(('Liftoff',), ac_type=helicopter))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTwoDegPitchTo35Ft(unittest.TestCase):
    def test_can_operate(self):
        self.assertTrue(OnDeck.can_operate(('Grounded', 'Pitch', 'Roll'),
                                           ac_type=helicopter))

    @unittest.skip('Test Not Implemented')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestTCASOperational(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASOperational
        self.operational_combinations = [
            ('Altitude AAL', ),
            ('Altitude AAL', 'TCAS Combined Control'),
            ('Altitude AAL', 'TCAS Status'),
            ('Altitude AAL', 'TCAS Valid'),
            ('Altitude AAL', 'TCAS Failure'),
            ('Altitude AAL', 'TCAS Combined Control', 'TCAS Status'),
            ('Altitude AAL', 'TCAS Combined Control', 'TCAS Valid'),
            ('Altitude AAL', 'TCAS Combined Control', 'TCAS Failure'),
            ('Altitude AAL', 'TCAS Status', 'TCAS Valid'),
            ('Altitude AAL', 'TCAS Status', 'TCAS Failure'),
            ('Altitude AAL', 'TCAS Valid', 'TCAS Failure'),
            ('Altitude AAL', 'TCAS Combined Control', 'TCAS Status', 'TCAS Valid'),
            ('Altitude AAL', 'TCAS Combined Control', 'TCAS Status', 'TCAS Failure'),
            ('Altitude AAL', 'TCAS Combined Control', 'TCAS Valid', 'TCAS Failure'),
            ('Altitude AAL', 'TCAS Status', 'TCAS Valid', 'TCAS Failure'),
            ('Altitude AAL', 'TCAS Combined Control', 'TCAS Status', 'TCAS Valid', 'TCAS Failure'),
        ]
        self.values_mapping_cc = {  # Values from ARINC 735
            0: 'No Advisory',
            1: 'Clear of Conflict',
            2: 'Spare',
            3: 'Spare',
            4: 'Up Advisory Corrective',
            5: 'Down Advisory Corrective',
            6: 'Preventive',
            7: 'Not Used',
        }
        self.status_mapping = {
            0: 'Normal Operation',
            1: 'TCAS Computer Unit',
            2: 'TCAS System Status',
            3: 'Both Failed',
        }

    def test_normal_operation(self):
        tcas_cc = M('TCAS Combined Control',
                    array=np.ma.concatenate([np.zeros(500), np.ones(10) * 4, np.ones(2), np.zeros(500)]),
                    values_mapping=self.values_mapping_cc)
        alt_aal=P('Altitude AAL', array=np.ma.concatenate([np.arange(0, 1000, 200), np.ones(1000) * 1000, np.arange(1000, -100, -200)]))
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, None, None, None)
        self.assertEqual(node.get_first().slice, slice(5, 1006))
        self.assertEqual(node.get_first().name, 'TCAS Operational')

    def test_one_sector(self):
        tcas_cc = M('TCAS Combined Control', array=np.ma.concatenate([np.tile([0,0,6,6], 253), np.zeros(1010)]),
                    values_mapping=self.values_mapping_cc)
        up_down = np.concatenate([np.arange(0, 1000, 200), np.ones(1000) * 1000, np.arange(1000, -100, -200)])
        alt_aal=P('Altitude AAL', array=np.ma.concatenate([up_down, up_down]))
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, None, None, None)
        self.assertEqual(node.get_first().slice, slice(1016, 2017))
        self.assertEqual(node.get_first().name, 'TCAS Operational')

    def test_not_low(self):
        # Following a change in altitude datum for TCAS operation, this test was "elevated" to show that the
        # effect was due to altitude. Other tests were changed more simply by resetting the slice limits.
        alt_aal=P('Altitude AAL', array=
                  np.ma.concatenate([np.arange(500, 1500, 100), np.ones(10) * 1500, np.arange(1400, 400, -100)]))
        node = self.node_class()
        node.derive(alt_aal, None, None, None, None)
        self.assertEqual(node.get_slices()[0], slice(4, 26))

    def test_not_constant(self):
        tcas_cc = M('TCAS Combined Control',
                    array=np.ma.ones(1012) * 4,
                    values_mapping=self.values_mapping_cc)
        alt_aal = P('Altitude AAL', array=np.ma.concatenate([np.arange(0, 1000, 200), np.ones(1000) * 1000, np.arange(1000, -100, -200)]))
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, None, None, None)
        self.assertEqual(node, [])

    def test_not_out_of_scope(self):
        array = np.ma.concatenate([
            np.zeros(200),
            np.ones(20) * 2,
            np.zeros(30),
            np.ones(20) * 3,
            np.zeros(30),
            np.ones(20) * 7,
            np.zeros(660),
        ])
        tcas_cc = M('TCAS Combined Control', array=array, values_mapping=self.values_mapping_cc)
        alt_aal=P('Altitude AAL', array=np.ma.concatenate([np.arange(0, 1000, 200), np.ones(1000) * 1000, np.arange(1000, -100, -200)]))
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, None, None, None)
        self.assertEqual(node.get_slices(), [slice(5, 200, None), slice(220, 250, None), slice(270, 300, None), slice(320, 1006, None)])

    def test_not_if_status_wrong(self):
        # Embraer map status zero to Normal Operation
        tcas_cc = M('TCAS Combined Control', array=np.ma.concatenate([np.zeros(500), np.ones(10) * 5, np.zeros(490)]),
                    values_mapping=self.values_mapping_cc)
        alt_aal = P('Altitude AAL', array=np.ma.concatenate([np.arange(0, 1000, 200), np.ones(989) * 1000, np.arange(1000, -100, -200)]))
        status = M('TCAS Status', array=np.ma.zeros(1000),
                   values_mapping=self.status_mapping)
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, status, None, None)
        self.assertEqual(node.get_first().slice.start, 5)

        status = M('TCAS Status', array=np.ma.ones(1000) * 2,
                   values_mapping=self.status_mapping)
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, status, None, None)
        self.assertEqual(node, [])

        # Airbus map Status one to TCAS Active
        status = M('TCAS Status', array=np.ma.concatenate([np.zeros(490), np.ones(15), np.zeros(495)]),
                   values_mapping={0:'-', 1:'TCAS Active'})
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, status, None, None)
        # As the invalid status period overlaps the TCAS RA, we ignore it...
        self.assertEqual(node.get_first().slice, slice(490, 500))

        # Check we can handle the unknown
        alt_aal=P('Altitude AAL', array=np.ma.concatenate([np.arange(0, 1000, 200), np.ones(989) * 1000, np.arange(1000, -100, -200)]))
        status = M('TCAS Status',
                   array=np.ma.concatenate([np.zeros(490), np.ones(15), np.zeros(495)]),
                   values_mapping={0:'anything', 1:'nonsense'})
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, status, None, None)
        # The unrecognised status does not mask the TCAS RA...
        self.assertEqual(node.get_first().slice.start, 5)

    def test_not_if_valid_wrong(self):
        tcas_cc = M('TCAS Combined Control', array=np.ma.concatenate([np.zeros(500), np.ones(10) * 5, np.zeros(490)]),
                   values_mapping=self.values_mapping_cc)
        alt_aal=P('Altitude AAL', array=np.ma.concatenate([np.arange(0, 1000, 200), np.ones(989) * 1000, np.arange(1000, -100, -200)]))
        valid = M('TCAS Valid', array=np.ma.ones(1000), values_mapping={0:'-', 1:'Valid'})
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, None, valid, None)
        self.assertEqual(node.get_first().slice.start, 5)

        valid = M('TCAS Valid', array=np.ma.zeros(1000), values_mapping={0:'-', 1:'Valid'})
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, None, valid, None)
        self.assertEqual(node, [])

        valid = M('TCAS Valid', array=np.ma.concatenate([np.ones(490), np.zeros(15), np.ones(495)]), values_mapping={0:'-', 1:'Valid'})
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, None, valid, None)
        self.assertEqual(node.get_slices(), [slice(5, 490, None), slice(510, 995, None)])

    def test_not_if_failure(self):
        tcas_cc = M('TCAS Combined Control', array=np.ma.concatenate([np.zeros(500), np.ones(10) * 5, np.zeros(490)]),
                   values_mapping=self.values_mapping_cc)
        alt_aal=P('Altitude AAL', array=np.ma.concatenate([np.arange(0, 1000, 200), np.ones(989) * 1000, np.arange(1000, -100, -200)]))
        fail = M('TCAS Failure', array=np.ma.zeros(1000), values_mapping={0:'-', 1:'Failed'})
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, None, None, fail)
        self.assertEqual(node.get_first().slice.start, 5)

        fail = M('TCAS Failure', array=np.ma.ones(1000), values_mapping={0:'-', 1:'Failed'})
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, None, None, fail)
        self.assertEqual(node, [])

        fail = M('TCAS Failure', array=np.ma.concatenate([np.zeros(490), np.ones(15), np.zeros(495)]), values_mapping={0:'-', 1:'Failed'})
        node = self.node_class()
        node.derive(alt_aal, tcas_cc, None, None, fail)
        self.assertEqual(node.get_slices(), [slice(5, 490, None), slice(510, 995, None)])


    def test_masked_1(self):
        # This replicates the format seen from real data.
        tcas_cc = M('TCAS Combined Control',
                    array=np.ma.array(data=[0, 1, 2, 3, 4, 5, 4, 5, 6, 5],
                                      mask=[0, 1] * 5),
                    values_mapping=self.values_mapping_cc)
        alt_aal=P('Altitude AAL', array=np.ma.concatenate([np.arange(0, 1000, 200), [1000], np.arange(1000, -100, -200)]))
        node = self.node_class()
        node.derive(tcas_cc, alt_aal, None, None)
        self.assertEqual(node, [])

    def test_masked_2(self):
        # This replicates the format seen from real data.
        tcas_cc = M('TCAS Combined Control',
                    array=np.ma.array(data=np.concatenate([np.zeros(50), np.ones(50) * 5, np.zeros(410)]),
                                      mask=np.tile([0, 1], 255)),
                    values_mapping=self.values_mapping_cc)
        alt_aal=P('Altitude AAL', array=np.ma.concatenate([np.arange(0, 1000, 200), np.ones(500) * 1000, np.arange(1000, -100, -200)]))
        node = self.node_class()
        node.derive(tcas_cc, alt_aal, None, None)
        self.assertEqual(node, [])

        # This is another format seen from real data.
        tcas_cc = M('TCAS Combined Control',
                    array=np.ma.array(data=np.concatenate([np.zeros(50), np.ones(20) * 5, np.zeros(1810)]),
                                      mask=np.concatenate([np.zeros(50), np.ones(20), np.zeros(1810)])),
                    values_mapping=self.values_mapping_cc)
        alt_aal=P('Altitude AAL', array=np.ma.concatenate([np.arange(0, 1000, 200), np.ones(1900) * 1000, np.arange(1000, -100, -200)]))
        node = self.node_class()
        node.derive(tcas_cc, alt_aal, None, None)
        self.assertEqual(node, [])

    def test_corrupt(self):
        tcas_cc = M('TCAS Combined Control',
                    array=np.ma.concatenate([np.random.randint(2, size=20) * 6, np.ones(5), np.ones(6) * 4, np.zeros(5)]))
        alt_aal = P('Altitude AAL', array=np.ma.ones(20) * 5000)
        node = self.node_class()
        node.derive(tcas_cc, alt_aal, None, None)
        self.assertEqual(node, [])


class TestTCASResolutionAdvisory(unittest.TestCase):

    def test_can_operate(self):
        self.assertTrue(TCASResolutionAdvisory.can_operate(('TCAS Combined Control',
                                                            'TCAS Operational')))
        self.assertTrue(TCASResolutionAdvisory.can_operate(('TCAS RA',
                                                            'TCAS Operational')))

    def test_derive_cc(self):
        tcas_cc = M('TCAS Combined Control', array=np.ma.array([0,0,0,0,4,5,4,5,5,5,4,3,2,1]+498*[0]),
                    values_mapping={0: 'No Advisory',
                                    1: 'Clear of Conflict',
                                    2: 'Spare',
                                    3: 'Spare',
                                    4: 'Up Advisory Corrective',
                                    5: 'Down Advisory Corrective',
                                    6: 'Preventive',
                                    7: 'Not Used'})
        tcas_op = buildsection('TCAS Operating', 3, 480)
        node = TCASResolutionAdvisory()
        node.derive(tcas_cc, tcas_op)
        self.assertEqual(node.get_first().name, 'TCAS Resolution Advisory')
        self.assertEqual(node.get_ordered_by_index()[0].slice, slice(4, 11))

    def test_derive_ra(self):
        self.assertTrue(TCASResolutionAdvisory.can_operate(('TCAS Operational', 'TCAS RA')))
        tcas_ra = M('TCAS RA', array=np.ma.array([0]*4 + [1]*10 + 498*[0]),
                    values_mapping={0: '-', 1: 'RA'})
        tcas_op = buildsection('TCAS Operating', 3, 480)
        node = TCASResolutionAdvisory()
        node.derive(None, tcas_op, tcas_ra)
        self.assertEqual(node.get_first().name, 'TCAS Resolution Advisory')
        self.assertEqual(node.get_ordered_by_index()[0].slice, slice(4, 14))

class TestTCASTrafficAdvisory(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = TCASTrafficAdvisory
        self.operational_combinations = [
            ('TCAS Operational', 'TCAS TA'),
            ('TCAS Operational', 'TCAS All Threat Traffic'),
            ('TCAS Operational', 'TCAS Traffic Alert'),
            ('TCAS Operational', 'TCAS TA (1)'),
            ('TCAS Operational', 'TCAS TA', 'TCAS Resolution Advisory'),
            ('TCAS Operational', 'TCAS All Threat Traffic', 'TCAS Resolution Advisory'),
            ('TCAS Operational', 'TCAS Traffic Alert', 'TCAS Resolution Advisory'),
            ('TCAS Operational', 'TCAS TA (1)', 'TCAS Resolution Advisory'),
        ]

    def test_normal_operation(self):
        tcas_ops = buildsection('TCAS Operational', 5, 25)
        ta = M('TCAS TA', array=np.ma.array([0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0]),
               values_mapping={0: '-', 1: 'TA'})
        node = self.node_class()
        node.derive(tcas_ops, ta, None, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node.get_first().slice, slice(7, 13))
        node = self.node_class()
        node.derive(tcas_ops, None, ta, None, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node.get_first().slice, slice(7, 13))
        node = self.node_class()
        node.derive(tcas_ops, None, None, ta, None, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node.get_first().slice, slice(7, 13))
        node = self.node_class()
        node.derive(tcas_ops, None, None, None, ta, None)
        self.assertEqual(len(node), 1)
        self.assertEqual(node.get_first().slice, slice(7, 13))

    def test_not_close_to_ra(self):
        tcas_ops = buildsection('TCAS Operational', 3, 21)
        ta = M('TCAS TA', array=np.ma.array([0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]),
               values_mapping={0: '-', 1: 'TA'})
        ra = buildsection('TCAS Resolution Advisory', 20, 21)
        node = self.node_class()
        node.derive(tcas_ops, None, ta, None, None, ra)
        self.assertEqual(len(node), 1)
        ra = buildsection('TCAS Resolution Advisory', 13, 21)
        node = self.node_class()
        node.derive(tcas_ops, None, ta, None, None, ra)
        self.assertEqual(len(node), 0)


class TestTakeoffRunwayHeading(unittest.TestCase):

    def setUp(self):
        self.node_class = TakeoffRunwayHeading

    def test_derive_rto_turnaround(self):
        for flight_pk, nodes, attrs in open_node_container(
              os.path.join(test_data_path, 'runway_takeoff_heading.zip')):
            hdg_con = nodes['Heading Continuous']
            groundeds = nodes['Grounded']
            toffs = nodes['Takeoff Roll']

        hdg_array = hdg_con.array %360
        hdg = P('Heading', hdg_array, frequency=hdg_con.frequency)
        gnd_aligned = groundeds.get_aligned(hdg)
        toff_aligned = toffs.get_aligned(hdg)

        node = self.node_class()
        node.derive(hdg, gnd_aligned, toff_aligned)

        self.assertEqual(len(node),3)
        self.assertEqual(node.get_slices(), [
            slice(1627, 2102, None),
            slice(2119, 2123, None),
            slice(2525, 2632, None)
        ])

class TestBaroDifference(unittest.TestCase):

    def setUp(self):
        self.node_class = BaroDifference
        self.operational_combinations = [
            ('Baro Correction (Capt)', 'Baro Correction (FO)', 'Baro Setting Selection'),
            ('Baro Correction (Capt)', 'Baro Correction (FO)', 'Baro Correction (ISIS)'),
        ]
        self.can_operate_kwargs = {'manufacturer': A('Manufacturer', 'Airbus')}
        self.fast = buildsection('Fast', 0, 80)

    def test_can_operate_airbus(self):
        manufacturer = A('Manufacturer', 'Airbus')
        combinations = list(map(set, self.node_class.get_operational_combinations(manufacturer=manufacturer)))
        expected = [
            ('Baro Correction (Capt)', 'Baro Correction (FO)', 'Baro Setting Selection'),
            ('Baro Correction (Capt)', 'Baro Correction (FO)', 'Baro Correction (ISIS)'),
            ('Baro Correction (Capt)', 'Baro Correction (FO)', 'Baro Setting Selection (Capt)', 'Baro Setting Selection (FO)'),
        ]
        for combination in map(set, expected):
            self.assertIn(combination, combinations)
        self.assertNotIn({'Baro Correction (Capt)', 'Baro Correction (FO)'}, combinations)
        self.assertNotIn({'Baro Correction (Capt)', 'Baro Correction (FO)', 'Baro Setting Selection (Capt)'}, combinations)

    def test_can_operate(self):
        manufacturer = A('Manufacturer', 'Boeing')
        combinations = list(map(set, self.node_class.get_operational_combinations(manufacturer=manufacturer)))
        expected = [
            ('Baro Correction (Capt)', 'Baro Correction (FO)')
        ]
        for combination in map(set, expected):
            self.assertIn(combination, combinations)

    def test_derive_baro_difference(self):
        baro_capt = P('Baro Correction (Capt)',
                      array=np.ma.concatenate([
                          np.ma.ones(3) * 995,
                          np.ma.ones(7) * 998,
                          np.ma.ones(10) * 1013
                          ]),
                      frequency=1./4)
        baro_fo = P('Baro Correction (FO)',
                    array=np.ma.concatenate([
                        np.ma.ones(15) * 998,
                        np.ma.ones(5) * 1013
                        ]),
                    frequency=1./4)

        node = self.node_class()
        node.get_derived((baro_capt, baro_fo, None, None, None, None, self.fast))
        self.assertEqual(len(node), 2)
        self.assertEqual(node.get_slices(), [
            slice(0, 3, None),
            slice(10, 15, None)
        ])

        self.assertEqual(node.frequency, 1./4.)

    def test_no_difference(self):
        baro_capt = P('Baro Correction (Capt)',
                      array=np.ma.concatenate([
                          np.ma.ones(10) * 998,
                          np.ma.ones(10) * 1013
                          ]),
                      frequency=1./4)
        baro_fo = P('Baro Correction (FO)',
                    array=np.ma.concatenate([
                        np.ma.ones(10) * 998,
                        np.ma.ones(10) * 1013
                        ]),
                    frequency=1./4)

        node = self.node_class()
        node.get_derived((baro_capt, baro_fo, None, None, None, None, self.fast))
        self.assertEqual(len(node), 0)

    def test_short_duration_difference(self):
        baro_capt = P('Baro Correction (Capt)',
                      array=np.ma.concatenate([
                          np.ma.array([989]),
                          np.ma.ones(9) * 998,
                          np.ma.ones(10) * 1013
                          ]),
                      frequency=1./4)
        baro_fo = P('Baro Correction (FO)',
                    array=np.ma.concatenate([
                        np.ma.ones(10) * 998,
                        np.ma.ones(9) * 1013,
                        np.ma.array([1021])
                        ]),
                    frequency=1./4)

        node = self.node_class()
        node.get_derived((baro_capt, baro_fo, None, None, None, None, self.fast))
        self.assertEqual(len(node), 0)


    def test_bar_sel(self):
        baro_capt = P('Baro Correction (Capt)',
                      array=np.ma.concatenate([
                          np.ma.ones(3) * 995,
                          np.ma.ones(7) * 998,
                          np.ma.ones(10) * 1013
                          ]),
                      frequency=1./4)
        baro_fo = P('Baro Correction (FO)',
                    array=np.ma.concatenate([
                        np.ma.ones(15) * 998,
                        np.ma.ones(5) * 1013
                        ]),
                    frequency=1./4)
        baro_sel = M(
            'Baro Setting Selection',
            array=np.ma.concatenate([
                np.ones(10, dtype=np.int) * 2,
                np.ones(10, dtype=np.int) * 1,
            ]),
            values_mapping={0: 'ALT QFE', 1: 'ALT STD', 2: 'ALT QNH'},
            frequency=1./4
        )

        node = self.node_class()
        node.get_derived((baro_capt, baro_fo, baro_sel, None, None, None, self.fast))

        self.assertEqual(len(node), 1)
        self.assertEqual(node.get_slices(), [slice(0, 3, None)])

        self.assertEqual(node.frequency, 1./4.)

    def test_bar_sel_cpt_fo(self):
        baro_capt = P('Baro Correction (Capt)',
                      array=np.ma.concatenate([
                          np.ma.ones(3) * 995,
                          np.ma.ones(7) * 998,
                          np.ma.ones(10) * 1013
                          ]),
                      frequency=1./4)
        baro_fo = P('Baro Correction (FO)',
                    array=np.ma.concatenate([
                        np.ma.ones(15) * 998,
                        np.ma.ones(5) * 1013
                        ]),
                    frequency=1./4)
        baro_sel_cpt = M(
            'Baro Setting Selection (Capt)',
            array=np.ma.concatenate([
                np.ones(10, dtype=np.int) * 2,
                np.ones(10, dtype=np.int) * 1,
            ]),
            values_mapping={0: 'QFE', 1: 'STD', 2: 'QNH', 3: 'Not Used'},
            frequency=1./4
        )
        baro_sel_fo = M(
            'Baro Setting Selection (FO)',
            array=np.ma.concatenate([
                np.ones(10, dtype=np.int) * 2,
                np.ones(10, dtype=np.int) * 1,
            ]),
            values_mapping={0: 'QFE', 1: 'STD', 2: 'QNH', 3: 'Not Used'},
            frequency=1./4
        )

        node = self.node_class()
        node.get_derived((baro_capt, baro_fo, None, baro_sel_cpt, baro_sel_fo, None, self.fast))

        self.assertEqual(len(node), 1)
        self.assertEqual(node.get_slices(), [slice(0, 3, None)])

        self.assertEqual(node.frequency, 1./4.)

    def test_bar_cor_isis(self):
        baro_capt = P('Baro Correction (Capt)',
                      array=np.ma.concatenate([
                          np.ma.ones(10) * 995,
                          np.ma.ones(10) * 1013
                          ]),
                      frequency=1./4)
        baro_fo = P('Baro Correction (FO)',
                    array=np.ma.concatenate([
                        np.ma.ones(2) * 995,
                        np.ma.ones(8) * 998,
                        np.ma.ones(10) * 1013
                        ]),
                    frequency=1./4)
        baro_cor_isis = P(
            'Baro Correction (ISIS)',
            array=np.ma.concatenate([
                np.ones(8, dtype=np.int) * 995,
                np.ones(12, dtype=np.int) * 1013,
            ]),
            frequency=1./4
        )

        node = self.node_class()
        node.get_derived((baro_capt, baro_fo, None, None, None, baro_cor_isis, self.fast))

        self.assertEqual(len(node), 1)
        self.assertEqual(node.get_slices(), [slice(2, 8, None)])

        self.assertEqual(node.frequency, 1./4.)

    def test_outside_fast_section(self):
        baro_capt = P(
            'Baro Correction (Capt)',
            array=np.ma.concatenate([
                np.ma.ones(3) * 995,
                np.ma.ones(7) * 998,
                np.ma.ones(10) * 1013
                ]),
            frequency=1./4)
        baro_fo = P(
            'Baro Correction (FO)',
            array=np.ma.concatenate([
                np.ma.ones(15) * 998,
                np.ma.ones(5) * 1013
                ]),
            frequency=1./4)
        fast = buildsection('Fast', 15, 70)

        node = self.node_class()
        node.get_derived((baro_capt, baro_fo, None, None, None, None, fast))

        self.assertEqual(len(node), 1)
        self.assertEqual(node.get_slices(), [slice(10, 15, None)])
