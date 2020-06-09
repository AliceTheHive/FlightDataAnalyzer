import numpy as np
import os
import unittest
import yaml

from mock import call, Mock, patch

from analysis_engine.approaches import ApproachInformation, is_heliport
from analysis_engine.exceptions import AFRMissmatchError
from analysis_engine.node import (
    A, ApproachItem, aeroplane, helicopter, KPV, KeyPointValue, P, S, Section, KTI, KeyTimeInstance,
    load)


test_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'test_data')
approaches_path = os.path.join(test_data_path, 'approaches')

airports = yaml.load(open(os.path.join(test_data_path, 'airports.yaml'), 'rb'), Loader=yaml.FullLoader)


class TestIsHeliport(unittest.TestCase):

    def test_is_heliport(self):
        self.assertFalse(is_heliport(aeroplane, airports['gatwick'], airports['gatwick']['runways'][0]))
        self.assertTrue(is_heliport(helicopter, None, None))
        helipad = {'identifier':'H', 'strip': {'length': 0}}
        heliport = {'name':'Vangard helipad', 'runways':[helipad]}
        self.assertTrue(is_heliport(helicopter, heliport, helipad))
        self.assertTrue(is_heliport(helicopter, {}, helipad))
        self.assertFalse(is_heliport(helicopter, airports['gatwick'], airports['gatwick']['runways'][0]))

class TestApproachInformation(unittest.TestCase):

    def setUp(self):
        self.node_class = ApproachInformation
        self.gatwick = [airports['gatwick']]

    def test_can_operate(self):
        self.assertFalse(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AAL'), ac_type=aeroplane))
        self.assertTrue(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AAL', 'Takeoff', 'Touchdown'), ac_type=aeroplane))
        self.assertTrue(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AGL', 'Takeoff', 'Touchdown'), ac_type=helicopter))
        self.assertTrue(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AAL', 'Latitude Prepared', 'Longitude Prepared',
             'Takeoff', 'Touchdown'), ac_type=aeroplane))
        self.assertFalse(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AAL', 'Latitude Prepared', 'Takeoff', 'Touchdown'),
            ac_type=aeroplane))
        self.assertFalse(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AAL', 'Longitude Prepared', 'Takeoff', 'Touchdown'),
            ac_type=aeroplane))

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_basic(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0, -10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(0, 100), 0, 100)]),
                          P('Heading Continuous', np.ma.ones(100)*260),
                          None,
                          None,
                          P('ILS Localizer', np.ma.concatenate((np.arange(-2.5, 0, 0.05), np.ones(50) * -0.15))),
                          P('ILS Glideslope', np.ma.zeros(100)),
                          P('ILS Frequency', np.ma.ones(100) * 110.90),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          S(items=[Section('Fast', slice(10,90), 10, 90)]),
                          None, None, None, None, None, None, None,
                          KTI('Touchdown', items=[KeyTimeInstance(index=19)]),
                          None,
                          S(items=[Section('Takeoff', slice(1,10), 1, 10)]))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(41, 100))

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_frequency_masked(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0, -10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(0, 100), 0, 100)]),
                          P('Heading Continuous', np.ma.ones(100) * 260.0),
                          None,
                          None,
                          P('ILS Localizer', np.ma.concatenate((np.arange(-2.5, 0, 0.05), np.ones(50) * -0.15))),
                          P('ILS Glideslope', np.ma.zeros(100)),
                          P('ILS Frequency', np.ma.array(data=np.ones(100) * 110.90, mask=np.ones(100))),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_frequency_wrong_frequency(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0, -10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(0, 100), 0, 100)]),
                          P('Heading Continuous', np.ma.ones(100) * 260.0),
                          None,
                          None,
                          P('ILS Localizer', np.ma.concatenate((np.arange(-2.5, 0, 0.05), np.ones(50) * -0.15))),
                          P('ILS Glideslope', np.ma.zeros(100)),
                          P('ILS Frequency', np.ma.ones(100) * 110.95),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_masked_preamble(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        ils_array = np.ma.concatenate((np.arange(-2.5, 0, 0.05), np.ones(50) * -0.15))
        ils_array.mask = np.ma.getmaskarray(ils_array)
        ils_array.mask[0:45] = True
        ils_array.mask[70:] = True
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0, -10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(0, 100), 0, 100)]),
                          P('Heading Continuous', np.ma.ones(100) * 260),
                          None,
                          None,
                          P('ILS Localizer', ils_array),
                          P('ILS Glideslope', np.ma.zeros(100)),
                          P('ILS Frequency', np.ma.ones(100) * 110.90),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(45, 70))

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_never_on_loc(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0, -10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(0, 100), 0, 100)]),
                          P('Heading Continuous', np.ma.ones(100)*260),
                          None,
                          None,
                          P('ILS Localizer', np.ma.ones(100)*3),
                          P('ILS Glideslope', np.ma.zeros(100)),
                          P('ILS Frequency', np.ma.ones(100)*110.90),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)
        self.assertEqual(approaches[0].gs_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_only_last_segment(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(20, 90), 20, 90)]),
                          P('Heading Continuous', np.ma.ones(100)*260),
                          None,
                          None,
                          P('ILS Localizer',np.ma.repeat([0,0,0,1,3,3,2,1,0,0], 10), frequency=0.5),
                          P('ILS Glideslope',np.ma.zeros(100)),
                          P('ILS Frequency', np.ma.ones(100)*110.90),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=80, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=80, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        # Slice changed from original test to reflect new way of determining localizer established phase.
        # Not looking at loc signal, but approach phase and established startpoint, and runway turnoff endpoint.
        self.assertEqual(approaches[0].loc_est, slice(20, 70))

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_stays_established_with_large_visible_deviations(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(10, 90), 10, 90)]),
                          P('Heading Continuous', np.ma.ones(100)*260),
                          None,
                          None,
                          P('ILS Localizer',np.ma.array(np.repeat([0,0,0,1,2.3,2.3,2,1,0,0], 10))),
                          None,
                          None,
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(10, 70)) # 70 reflects the 2 dot endpoint

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_insensitive_to_few_masked_values(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        ils_array = np.ma.array(np.repeat([0,0,0,1,2.3,2.3,2,1,0,0], 10))
        ils_array.mask = np.ma.getmaskarray(ils_array)
        ils_array.mask[60:62] = True
        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(10, 90), 10, 90)]),
                          P('Heading Continuous', np.ma.ones(100)*260),
                          None,
                          None,
                          P('ILS Localizer', ils_array),
                          None,
                          None,
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(10, 70)) # 70 reflects the 2 dot endpoint

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_skips_too_many_masked_values(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(10, 90), 10, 90)]),
                          P('Heading Continuous', np.ma.ones(100)*260),
                          None,
                          None,
                          P('ILS Localizer', np.ma.array(data=np.zeros(20), mask=np.tile([0,1], 10))),
                          None,
                          None,
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_skips_too_few_values(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-50)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(4, 18), 4, 18)]),
                          P('Heading Continuous', np.ma.ones(100)*260),
                          None,
                          None,
                          P('ILS Localizer', np.ma.array(data=np.zeros(20), mask=np.concatenate((np.ones(10), np.zeros(5), np.ones(5))))),
                          None,
                          None,
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=17, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=17, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_all_masked(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-50)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(2, 19), 2, 19)]),
                          P('Heading Continuous', np.ma.ones(20)*260),
                          None,
                          None,
                          P('ILS Localizer', np.ma.array(data=np.zeros(20), mask=True)),
                          None,
                          None,
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=17, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=17, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_always_on_loc(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0,-50)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(2, 19), 2, 19)]),
                          P('Heading Continuous', np.ma.ones(20)*260),
                          None,
                          None,
                          P('ILS Localizer',np.ma.ones(20)*-0.2),
                          P('ILS Glideslope',np.ma.zeros(20)),
                          P('ILS Frequency', np.ma.ones(20)*110.90),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=17, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=17, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(2,19,None))

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_not_above_1500ft(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(2000, 0,-100)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(2, 19), 2, 19)]),
                          P('Heading Continuous', np.ma.ones(20)*260),
                          None,
                          None,
                          P('ILS Localizer', np.ma.zeros(20)),
                          P('ILS Glideslope', np.ma.zeros(20)),
                          P('ILS Frequency', np.ma.ones(20)*110.90),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=17, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=17, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(5,19,None))


    # This test was "test_ils_localizer_established_not_below_1000ft" but revised requirement made it permissable to establish
    # below 1000ft AAL.
    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_below_1000ft(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(2000, 0,-50)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(2, 39), 2, 39)]),
                          P('Heading Continuous', np.ma.ones(40)*260),
                          None,
                          None,
                          P('ILS Localizer', np.ma.concatenate((np.ones(21) * 2.0, np.zeros(12), np.ones(7) * -2.0))),
                          P('ILS Glideslope', np.ma.zeros(40)),
                          P('ILS Frequency', np.ma.ones(40) * 110.90),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=17, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=17, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True))
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(22,39.5,None))


    #@patch('analysis_engine.api_handler.FileHandler.get_nearest_airport')
    @unittest.skip('superceded')
    def test_derive(self, get_nearest_airport):
        approaches = self.node_class()
        approaches._lookup_airport_and_runway = Mock()
        approaches._lookup_airport_and_runway.return_value = [None, None]

        # No approaches if no approach sections in the flight:
        approaches.derive(self.app, self.alt_aal, self.fast)
        self.assertEqual(approaches, [])
        # Test the different approach types:
        slices = [slice(0, 5), slice(10, 15), slice(20, 25)]
        self.app.create_phases(slices)

        approaches.derive(self.app, self.alt_aal, self.fast,
                          land_afr_apt=self.land_afr_apt_none,
                          land_afr_rwy=self.land_afr_rwy_none)
        self.assertEqual(approaches,
                         [ApproachItem('TOUCH_AND_GO', slice(0, 5)),
                          ApproachItem('GO_AROUND', slice(10, 15)),
                          ApproachItem('LANDING', slice(20, 25))])
        #approaches.set_flight_attr.assert_called_once_with()
        #approaches.set_flight_attr.reset_mock()
        approaches._lookup_airport_and_runway.assert_has_calls([
            call(_slice=slices[0], appr_ils_freq=None, precise=False,
                 lowest_lat=None, lowest_lon=None, lowest_hdg=None),
            call(_slice=slices[1], appr_ils_freq=None, precise=False,
                 lowest_lat=None, lowest_lon=None, lowest_hdg=None),
            call(_slice=slices[2], appr_ils_freq=None, precise=False,
                 lowest_lat=None, lowest_lon=None, lowest_hdg=None,
                 land_afr_apt=self.land_afr_apt_none,
                 land_afr_rwy=self.land_afr_rwy_none, hint='landing'),
        ])
        del approaches[:]
        approaches._lookup_airport_and_runway.reset_mock()
        # Test that landing lat/lon/hdg used for landing only, else use approach
        # lat/lon/hdg:
        approaches.derive(self.app, self.alt_aal, self.fast, self.land_hdg, self.land_lat,
                          self.land_lon, self.appr_hdg, self.appr_lat, self.appr_lon,
                          land_afr_apt=self.land_afr_apt_none,
                          land_afr_rwy=self.land_afr_rwy_none)
        self.assertEqual(approaches,
                         [ApproachItem('TOUCH_AND_GO', slice(0, 5)),
                          ApproachItem('GO_AROUND', slice(10, 15),
                                       lowest_hdg=self.appr_hdg[1].value),
                          ApproachItem('LANDING', slice(20, 25),
                                       lowest_lat=self.land_lat[0].value,
                                       lowest_lon=self.land_lon[0].value,
                                       lowest_hdg=self.land_hdg[0].value)])
        approaches._lookup_airport_and_runway.assert_has_calls([
            call(_slice=slices[0], lowest_hdg=None, lowest_lat=None,
                 lowest_lon=None, appr_ils_freq=None, precise=False),
            call(_slice=slices[1], lowest_hdg=self.appr_hdg[1].value,
                 lowest_lat=None, lowest_lon=None, appr_ils_freq=None,
                 precise=False),
            call(_slice=slices[2], lowest_hdg=self.land_hdg[0].value,
                 lowest_lat=self.land_lat[0].value, lowest_lon=self.land_lon[0].value,
                 appr_ils_freq=None, precise=False,
                 land_afr_apt=self.land_afr_apt_none, land_afr_rwy=self.land_afr_rwy_none,
                 hint='landing'),
        ])
        approaches._lookup_airport_and_runway.reset_mock()

        # FIXME: Finish implementing these tests to check that using the API
        #        works correctly and any fall back values are used as
        #        appropriate.


    @unittest.skip('Covered by TestHerat')
    def test_derive_afr_fallback(self):
        self.assertTrue(False, msg='Test not implemented.')

    @patch('analysis_engine.approaches.api')
    def test_landing_turn_off_runway_basic(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.side_effect = api.NotFoundError()
        get_handler.get_airport.return_value = self.gatwick[0]
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.zeros(30)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(10, 27), 10, 26)]),
                          P('Heading Continuous', np.ma.ones(30)*260),
                          None,
                          None,
                          None,
                          None,
                          None,
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          None,
                          None,
                          A('Precise Positioning', True))
        get_handler.get_airport.assert_called_with(2379)
        self.assertEqual(approaches[0].turnoff, 26)

    @patch('analysis_engine.approaches.api')
    def test_landing_turn_off_runway_curved(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.side_effect = api.NotFoundError()
        get_handler.get_airport.return_value = self.gatwick[0]
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.concatenate((np.arange(50, 0, -1), np.zeros(40)))),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(30, 80), 30, 80)]),
                          P('Heading Continuous', np.ma.concatenate((np.ones(70) * 260, np.arange(260, 280)))),
                          None,
                          None,
                          None,
                          None,
                          None,
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          None,
                          None,
                          A('Precise Positioning', True),
                          )
        get_handler.get_airport.assert_called_with(2379)
        self.assertEqual(approaches[0].turnoff, 70)

    @patch('analysis_engine.approaches.api')
    def test_landing_turn_off_runway_curved_left(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.side_effect = api.NotFoundError()
        get_handler.get_airport.return_value = self.gatwick[0]
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.concatenate((np.arange(50, 0, -1), np.zeros(40)))),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(30, 80), 30, 80)]),
                          P('Heading Continuous', np.ma.concatenate((np.ones(70) * 260, np.arange(260, 240, -1)))),
                          None,
                          None,
                          None,
                          None,
                          None,
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          None,
                          None,
                          A('Precise Positioning', True))
        get_handler.get_airport.assert_called_with(2379)
        self.assertEqual(approaches[0].turnoff, 70)

class TestCloseAirprots(unittest.TestCase):
    @patch('analysis_engine.approaches.api')
    def test_ils_freq(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['almaza'], airports['cairo'], airports['embaba']]
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()

        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.concatenate((np.arange(50, 0, -1), np.zeros(40)))),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(30, 80), 30, 80)]),
                          P('Heading Continuous', np.ma.ones(90) * 45.703),
                          None,
                          None,
                          P('ILS Localizer', array=np.ma.ones(90)),
                          P('ILS Glideslope', array=np.ma.ones(90)),
                          P('ILS Frequency', np.ma.ones(90) * 109.9),
                          None,
                          None,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=50, value=30.08465, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=50, value=31.3969768, name='Longitude At Touchdown')]),
                          A('Precise Positioning', False))

        self.assertEqual(len(approaches), 1)
        self.assertEqual(approaches[0].airport, airports['cairo'])
        self.assertEqual(approaches[0].landing_runway, airports['cairo']['runways'][2])
        self.assertEqual(approaches[0].approach_runway, airports['cairo']['runways'][2])


    @patch('analysis_engine.approaches.api')
    def test_no_ils_freq(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['almaza'], airports['cairo'], airports['embaba']]
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()

        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.concatenate((np.arange(50, 0, -1), np.zeros(40)))),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(30, 80), 30, 80)]),
                          P('Heading Continuous', np.ma.ones(90) * 45.703),
                          None,
                          None,
                          P('ILS Localizer', array=np.ma.ones(90)),
                          P('ILS Glideslope', array=np.ma.ones(90)),
                          None,
                          None,
                          None,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=50, value=30.08465, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=50, value=31.3969768, name='Longitude At Touchdown')]),
                          A('Precise Positioning', False))

        self.assertEqual(len(approaches), 1)
        self.assertEqual(approaches[0].airport, airports['cairo'])
        self.assertEqual(approaches[0].landing_runway, airports['cairo']['runways'][4])
        self.assertEqual(approaches[0].approach_runway, airports['cairo']['runways'][4])


    @patch('analysis_engine.approaches.api')
    def test_afr_match(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['almaza'], airports['cairo'], airports['embaba']]
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()

        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.concatenate((np.arange(50, 0, -1), np.zeros(40)))),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(30, 80), 30, 80)]),
                          P('Heading Continuous', np.ma.ones(90) * 45.703),
                          None,
                          None,
                          P('ILS Localizer', array=np.ma.ones(90)),
                          P('ILS Glideslope', array=np.ma.ones(90)),
                          None,
                          A(name='AFR Landing Airport', value={'id':4461}),
                          None,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=50, value=30.08465, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=50, value=31.3969768, name='Longitude At Touchdown')]),
                          A('Precise Positioning', False))

        self.assertEqual(len(approaches), 1)
        self.assertEqual(approaches[0].airport, airports['almaza'])
        self.assertEqual(approaches[0].landing_runway, airports['almaza']['runways'][2])
        self.assertEqual(approaches[0].approach_runway, airports['almaza']['runways'][2])


    @patch('analysis_engine.approaches.api')
    def test_afr_missmatch(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['almaza'], airports['cairo'], airports['embaba']]
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()

        with self.assertRaises(AFRMissmatchError):
            approaches.derive(P('Altitude AAL For Flight Phases', np.ma.concatenate((np.arange(50,0,-1), np.zeros(40)))),
                              None,
                              A('Aircraft Type', 'aeroplane'),
                              S(items=[Section('Approach', slice(30, 80), 30, 80)]),
                              P('Heading Continuous', np.ma.ones(90) * 45.703),
                              None,
                              None,
                              P('ILS Localizer', array=np.ma.ones(90)),
                              P('ILS Glideslope', array=np.ma.ones(90)),
                              None,
                              A(name='AFR Landing Airport', value={'id':1234, 'code': {'iata': 'xxx'}}),
                              None,
                              KPV('Latitude At Touchdown', items=[KeyPointValue(index=50, value=30.08465, name='Latitude At Touchdown')]),
                              KPV('Longitude At Touchdown', items=[KeyPointValue(index=50, value=31.3969768, name='Longitude At Touchdown')]),
                              A('Precise Positioning', True))


class TestAlicante(unittest.TestCase):

    # There is no ILS from this direction.
    @patch('analysis_engine.approaches.api')
    def test_no_ils(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['alicante']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_10091047_')
        app_start = 13261*2 # 13261 taken from CSV output file, but AppInfo runs at 2Hz.
        app_end = 13535*2

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True))
        self.assertEqual(approaches[0].airport['name'], 'Alicante')
        self.assertEqual(approaches[0].landing_runway['identifier'], '28')
        self.assertEqual(approaches[0].gs_est, None)
        self.assertEqual(approaches[0].loc_est, None)


class TestBardufoss(unittest.TestCase):

    @patch('analysis_engine.approaches.api')
    def test_slightly_offset_localizer(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['bardufoss']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_9928419_')
        app_start = 11352
        app_end = 11800

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True))
        self.assertEqual(approaches[0].airport['name'], 'Bardufoss')
        self.assertEqual(approaches[0].landing_runway['identifier'], '10')
        # The aircraft was never established on the glidepath
        # (started OK, but went outside 0.5 dots within 10 seconds of acquiring the localizer).
        self.assertEqual(approaches[0].gs_est, None)
        # ...but was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 11541)
        self.assertEqual(int(approaches[0].loc_est.stop), 11684)


class TestBodo(unittest.TestCase):

    @patch('analysis_engine.approaches.api')
    def test_offset_localizer(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['bodo']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_10076024_')
        app_start = 8780
        app_end = 8861

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True))
        self.assertEqual(approaches[0].airport['name'], 'Bodo')
        self.assertEqual(approaches[0].landing_runway['identifier'], '25')
        # The aircraft was never established on the glidepath
        # (started OK, but went outside 0.5 dots within 10 seconds of acquiring the localizer).
        self.assertEqual(approaches[0].gs_est, None)
        # ...but was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), app_start)
        self.assertEqual(int(approaches[0].loc_est.stop), 8861)


class TestChania(unittest.TestCase):

    # There is no ILS from this direction.
    @patch('analysis_engine.approaches.api')
    def test_no_ils(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['chania']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_9936188_')
        app_start = 12029*2 # 13261 taken from CSV output file, but AppInfo runs at 2Hz.
        app_end = 12324*2

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True))
        self.assertEqual(approaches[0].airport['name'], 'Chania')
        self.assertEqual(approaches[0].landing_runway['identifier'], '29')
        self.assertEqual(approaches[0].gs_est, None)
        self.assertEqual(approaches[0].loc_est, None)
        self.assertEqual(approaches[0].ils_freq, None)


class TestCapeTown(unittest.TestCase):
    # There is no ILS from this direction.
    @patch('analysis_engine.approaches.api')
    def test_cpt(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['cape_town']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None

        root = os.path.join(approaches_path, 'Landing_test_21952859_')

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          None, # Altitude AGL
                          A('Aircraft Type', 'aeroplane'),
                          fetch('Approach And Landing'),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          P('ILS Localizer'),
                          S('ILS Glideslope'),
                          None, # ILS Frequency
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None),
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          fetch('Fast'),
                          None, # Airspeed
                          None, # Groundspeed
                          None, # Altitude ADH
                          None, # Vertical Speed
                          None, # Roll
                          fetch('Heading'),)
        self.assertEqual(approaches[0].airport['name'], 'Cape Town Intl')
        self.assertEqual(approaches[0].landing_runway['identifier'], '19')
        self.assertEqual(approaches[0].gs_est, None)
        self.assertEqual(approaches[0].loc_est, None)
        self.assertEqual(approaches[0].ils_freq, None)


class TestDallasFortWorth(unittest.TestCase):

    # A change in runways late on the approach.
    @patch('analysis_engine.approaches.api')
    def test_runway_change_at_DFW(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['dallas_fort_worth']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_9821340_')
        app_start = 3187 * 2
        app_end = 3542 * 2

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True))
        self.assertEqual(approaches[0].airport['name'], 'Dallas Fort Worth Intl')
        # We approached runway 17C...
        self.assertEqual(approaches[0].approach_runway['identifier'], '17C')
        # ...but landed on 17R...
        self.assertEqual(approaches[0].landing_runway['identifier'], '17R')
        # The aircraft was established on the glidepath
        self.assertEqual(int(approaches[0].gs_est.start), 6523)
        self.assertEqual(int(approaches[0].gs_est.stop), 6590)
        # ...but was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 6523)
        self.assertEqual(int(approaches[0].loc_est.stop), 6590)



class TestFortWorth(unittest.TestCase):

    # This is the Fort Worth that it NOT at Dallas !!!
    @patch('analysis_engine.approaches.api')
    def test_ils_corrected_data(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['fort_worth']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_9094175_')
        app_start = 17834
        app_end = 18218

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True))
        self.assertEqual(approaches[0].airport['name'], 'Fort Worth Alliance')
        self.assertEqual(approaches[0].landing_runway['identifier'], '16L')
        # The aircraft was established on the glidepath slightly after the approach start
        self.assertEqual(int(approaches[0].gs_est.start), 17870)
        self.assertEqual(int(approaches[0].gs_est.stop), 18076)
        # ...but was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), app_start)
        self.assertEqual(int(approaches[0].loc_est.stop), app_end)


class TestHeraklion(unittest.TestCase):
    # There is no ILS from this direction.
    @patch('analysis_engine.approaches.api')
    def test_her(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['heraklion']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None

        root = os.path.join(approaches_path, 'Landing_test_22159938_')

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          None, # Altitude AGL
                          A('Aircraft Type', 'aeroplane'),
                          fetch('Approach And Landing'),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None),
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', False),
                          fetch('Fast'),
                          None, # Airspeed
                          None, # Groundspeed
                          None, # Altitude ADH
                          None, # Vertical Speed
                          None, # Roll
                          fetch('Heading'),)
        self.assertEqual(approaches[0].airport['name'], 'Iraklion - Nizos Kazantzakis')
        self.assertEqual(approaches[0].landing_runway['identifier'], '27')
        self.assertEqual(approaches[0].gs_est, None)
        self.assertEqual(approaches[0].loc_est, None)
        self.assertEqual(approaches[0].ils_freq, None)


class TestHerat(unittest.TestCase):

    # This aircraft does not record latitude and longitude
    @patch('analysis_engine.approaches.api')
    def test_no_ils_but_signal_looks_ok(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.side_effect = api.NotFoundError()
        get_handler.get_airport.return_value = airports['herat']
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_10411379_')
        app_start = 10792.6
        app_end = 11329.0

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value={'id': 3275}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown',  items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True))
        self.assertEqual(approaches[0].airport['name'], 'Herat')
        self.assertEqual(approaches[0].landing_runway['identifier'], '36')
        # There is no ILS on this runway.
        self.assertEqual(approaches[0].gs_est, None)
        self.assertEqual(approaches[0].loc_est, None)


class TestKirkenes(unittest.TestCase):

    # There is an ILS on the reverse runway, not on this approach.
    @patch('analysis_engine.approaches.api')
    def test_ils_on_reciprocal_runway(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['kirkenes']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_10066360_')
        app_start = 13803.5
        app_end = 14246.0

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True))
        self.assertEqual(approaches[0].airport['name'], 'Kirkenes')
        self.assertEqual(approaches[0].landing_runway['identifier'], '06')
        # There is no ILS on this runway.
        self.assertEqual(approaches[0].gs_est, None)
        self.assertEqual(approaches[0].loc_est, None)
        self.assertEqual(approaches[0].ils_freq, None)


class TestScatsta(unittest.TestCase):

    # This airport has a localizer but no glideslope antenna
    @patch('analysis_engine.approaches.api')
    def test_no_ils_glidepath(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['scatsta']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_10174453_')
        app_start = 6996.0
        app_end = 7327.0

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None),
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=7060, value=60.433, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=7060, value=-1.292, name='Longitude At Touchdown')]),
                          A('Precise Positioning', False))
        self.assertEqual(approaches[0].airport['name'], 'Scatsta')
        self.assertEqual(approaches[0].landing_runway['identifier'], '24')
        # We should be established on the (offset) localizer:
        self.assertEqual(int(approaches[0].loc_est.start), 7055)
        self.assertEqual(int(approaches[0].loc_est.stop), 7086)
        # ...but there is no glideslope on this runway.
        self.assertEqual(approaches[0].gs_est, None)


class TestSirSeretseKhama(unittest.TestCase):

    # Note: This aircraft does not have a recorded ILS Frequency
    @patch('analysis_engine.approaches.api')
    def test_late_turn_and_glideslope_established(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['sir_seretse_khama']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_10089954_')
        app_start = 8256
        app_end = 8942

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0].airport['name'], 'Sir Seretse Khama Intl')
        self.assertEqual(approaches[0].landing_runway['identifier'], '08')
        # The aircraft did get established on the glidepath
        self.assertEqual(int(approaches[0].gs_est.start), 8679)
        self.assertEqual(int(approaches[0].gs_est.stop), 8882)
        # ...and was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 8679)
        self.assertEqual(int(approaches[0].loc_est.stop), app_end)


class TestSplit(unittest.TestCase):

    # This flight gave an erroneous smoothed track when originally processed.
    # (The go-around was not aligned to the runway while the landing was).
    @patch('analysis_engine.approaches.api')
    def test_ga_at_split(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['split']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_15109245_')
        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(19951, 20518),
                                           start_edge=19951,
                                           stop_edge=20518),
                                   Section(name='Approach And Landing',
                                           slice=slice(20876, 21470),
                                           start_edge=20876,
                                           stop_edge=21470),
                                   ]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', False))
        self.assertEqual(approaches[0].airport['name'], 'Split')
        self.assertEqual(approaches[0].landing_runway['identifier'], '23')
        self.assertEqual(approaches[1].airport['name'], 'Split')
        self.assertEqual(approaches[1].landing_runway['identifier'], '23')


class TestWashingtonNational(unittest.TestCase):

    # This runway has a huge offset to the ILS - 40deg!
    @patch('analysis_engine.approaches.api')
    def test_huge_offset(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['washington_national']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_10260422_')
        app_start = 8284
        app_end = 8832

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0].airport['name'], 'Washington National')
        self.assertEqual(approaches[0].landing_runway['identifier'], '19')
        # The aircraft did get established on the glidepath
        self.assertEqual(approaches[0].gs_est, None)
        # ...and was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 8534)
        self.assertEqual(int(approaches[0].loc_est.stop), 8703)


class TestZaventem(unittest.TestCase):

    # This was a go-around and landing.
    @patch('analysis_engine.approaches.api')
    def test_go_around_and_landing(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['zaventem']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_10180313_')
        app_start = 11754
        app_end = 12346

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, stop_edge=app_end),
                                   Section(name='Approach And Landing',
                                           slice=slice(13500, 13898),
                                           start_edge=13500, stop_edge=13898)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', False),
                          )
        self.assertEqual(len(approaches), 2)
        self.assertEqual(approaches[0].airport['name'], 'Brussels Airport')
        self.assertEqual(approaches[0].landing_runway['identifier'], '01')
        self.assertEqual(approaches[0].type, 'GO_AROUND')
        self.assertEqual(approaches[1].type, 'LANDING')
        self.assertEqual(int(approaches[0].gs_est.start), 12106)
        self.assertEqual(int(approaches[1].gs_est.start), 13554)
        # ...and was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 12106)
        self.assertEqual(int(approaches[1].loc_est.start), 13554)


class TestBarcelona(unittest.TestCase):
    '''
    c60cf86bb146
    '''

    # A change in runways late on the approach.
    @patch('analysis_engine.approaches.api')
    def test_no_runway_change_at_barcelona(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['barcelona']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_004_')

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(18919, 19340),
                                           start_edge=18919,
                                           stop_edge=19340),
                            Section(name='Approach And Landing',
                                    slice=slice(20826, 21386),
                                    start_edge=20826,
                                    stop_edge=21386)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', False),
                          )
        # Go Around
        self.assertEqual(approaches[0].type, 'GO_AROUND')
        self.assertEqual(approaches[0].approach_runway['identifier'], '07L')
        self.assertEqual(approaches[0].landing_runway['identifier'], '07L')
        self.assertEqual(int(approaches[0].gs_est.start), 19085)
        self.assertEqual(int(approaches[0].gs_est.stop), 19312)
        self.assertEqual(int(approaches[0].loc_est.start), 19085)
        self.assertEqual(int(approaches[0].loc_est.stop), 19340)
        self.assertEqual(approaches[0].runway_change, False)

        # Landing
        self.assertEqual(approaches[1].type, 'LANDING')
        self.assertEqual(approaches[1].approach_runway['identifier'], '07L')
        self.assertEqual(approaches[1].landing_runway['identifier'], '07L')
        self.assertEqual(int(approaches[1].gs_est.start), 21025)
        self.assertEqual(int(approaches[1].gs_est.stop), 21251)
        self.assertEqual(int(approaches[1].loc_est.start), 21025)
        self.assertEqual(int(approaches[1].loc_est.stop), 21371)
        self.assertEqual(approaches[1].runway_change, False)



class TestNice(unittest.TestCase):
    '''
    bb25dd623581
    '''

    # A change in runways late on the approach.
    @patch('analysis_engine.approaches.api')
    def test__no_runway_change_at_nice(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['nice']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_003_')
        app_start = 3312 * 2
        app_end = 3606 * 2

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', False),
                          )

        self.assertEqual(approaches[0].type, 'LANDING')
        self.assertEqual(approaches[0].approach_runway['identifier'], '04R')
        self.assertEqual(approaches[0].landing_runway['identifier'], '04R')
        self.assertEqual(int(approaches[0].gs_est.start), 6818)
        self.assertEqual(int(approaches[0].gs_est.stop), 7025)
        self.assertEqual(int(approaches[0].loc_est.start), 6818)
        self.assertEqual(int(approaches[0].loc_est.stop), 7055)
        self.assertEqual(approaches[0].runway_change, False)


class TestGranCanaria(unittest.TestCase):
    '''
    0d61c465aad2
    '''

    # A change in runways late on the approach.
    @patch('analysis_engine.approaches.api')
    def test_no_runway_change_at_gran_canaria(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['gran_canaria']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_002_')
        app_start = 10949 * 2
        app_end = 11318 * 2

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', False),
                          )

        self.assertEqual(approaches[0].type, 'LANDING')
        self.assertEqual(approaches[0].approach_runway['identifier'], '03L')
        self.assertEqual(approaches[0].landing_runway['identifier'], '03R')
        self.assertEqual(int(approaches[0].gs_est.start), 22197)
        self.assertEqual(int(approaches[0].gs_est.stop), 22229)
        self.assertEqual(int(approaches[0].loc_est.start), 22177)
        self.assertEqual(int(approaches[0].loc_est.stop), 22229)
        self.assertEqual(approaches[0].runway_change, True)


class TestMunich(unittest.TestCase):
    '''
    15b80862f048
    '''

    # A change in runways late on the approach.
    @patch('analysis_engine.approaches.api')
    def test_no_runway_change_at_munich(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['munich']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_001_')
        app_start = 5732 * 2
        app_end = 6070 * 2

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', False),
                          )

        self.assertEqual(approaches[0].type, 'LANDING')
        # We approached runway 26L...
        self.assertEqual(approaches[0].approach_runway['identifier'], '26L')
        # ...and landed on 26L...
        self.assertEqual(approaches[0].landing_runway['identifier'], '26L')
        # The aircraft was established on the glidepath
        self.assertEqual(int(approaches[0].gs_est.start), 11717)
        self.assertEqual(int(approaches[0].gs_est.stop), 11936)
        # ...but was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 11717)
        self.assertEqual(int(approaches[0].loc_est.stop), 12050)
        # There was no runway change
        self.assertEqual(approaches[0].runway_change, False)


class TestNewDelhi(unittest.TestCase):
    '''
    Indira Ghandi airport has two runways, 09 and 10, and the test case lands on 10 but 09 is detected as the landing runway.
    The code was amended to ensure that runway changes are for parallel runways.
    '''

    # Not a change in runways.
    @patch('analysis_engine.approaches.api')
    def test_no_runway_change_at_new_delhi(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['new_delhi']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_10839725_')
        app_start = 10841
        app_end = 11675

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', False),
                          )

        self.assertEqual(approaches[0].type, 'LANDING')
        self.assertEqual(approaches[0].approach_runway['identifier'], '10')
        self.assertEqual(approaches[0].landing_runway['identifier'], '09')
        # There was no runway change
        self.assertEqual(approaches[0].runway_change, False)


class TestPalmaDeMallorca(unittest.TestCase):
    '''
    14100867 aka bfd:1a1cef312a74
    '''

    # A change in runways early on the approach.
    @patch('analysis_engine.approaches.api')
    def test_runway_change_at_Palma_De_Mallorca(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['palma_de_mallorca']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'ILS_test_14100867_')
        app_start = 17853
        app_end = 18678

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing',
                            frequency=2.0,
                            items=[Section(name='Approach And Landing',
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start,
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          P('ILS Glideslope', array=np.ma.ones(20160)),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          S(name='Fast',
                            frequency=2.0,
                            items=[Section(name='Fast',
                                           slice=slice(5507, 18515),
                                           start_edge=5507,
                                           stop_edge=18515)],
                          ))

        self.assertEqual(approaches[0].type, 'LANDING')
        # We approached runway 24L...
        self.assertEqual(approaches[0].approach_runway['identifier'], '24L')
        # ...and landed on 24R...
        self.assertEqual(approaches[0].landing_runway['identifier'], '24R')
        # There was a runway change
        self.assertEqual(approaches[0].runway_change, True)


class TestVaranusIslandHelicopter(unittest.TestCase):
    '''
    abo:011c26985a63
    '''

    # Helicopters heading does not necessarily align with airport runways on approach.
    @patch('analysis_engine.approaches.api')
    def test_helicopter_hdg_differ_from_runway_hdg(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = [airports['varanus island']]
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(approaches_path, 'Landing_heli_abo_011c26985a63_')

        approaches = ApproachInformation()
        approaches.derive(
            fetch('Altitude AAL'),
            fetch('Altitude AGL'),
            A('Aircraft Type', 'helicopter'),
            S(name='Approach And Landing', frequency=2.0,
              items=[
                  Section(
                      name='Approach And Landing', slice=slice(4682, 4898),
                      start_edge=4682, stop_edge=4898
                  )
              ]
            ),
            fetch('Heading Continuous'),
            fetch('Latitude Prepared'),
            fetch('Longitude Prepared'),
            fetch('ILS Localizer'),
            fetch('ILS Glideslope'),
            fetch('ILS Frequency'),
            A(name='AFR Landing Airport', value=None),
            A(name='AFR Landing Runway', value=None) ,
            KPV('Latitude At Touchdown', items=[
                KeyPointValue(index=4694.0, value=-20.659897159785032, name='Latitude At Touchdown'),
                KeyPointValue(index=10636.0, value=-20.71006896905601, name='Latitude At Touchdown'),
            ]),
            KPV('Longitude At Touchdown', items=[
                KeyPointValue(index=4752.0, value=115.58472611010075, name='Longitude At Touchdown'),
                KeyPointValue(index=10802.0, value=116.76602829247713, name='Longitude At Touchdown')
            ]),
            A('Precise Positioning', True),
            S(name='Fast',
              frequency=2.0,
              items=[Section(name='Fast',
                             slice=slice(93, 11373, None),
                             start_edge=92.875,
                             stop_edge=11372.875)],
            ),

            fetch('Airspeed'),
            fetch('Groundspeed'),
            fetch('Altitude ADH'),
            fetch('Vertical Speed'),
            fetch('Roll'),
            fetch('Heading'),
            fetch('Distance To Landing'),
            KTI('Touchdown', items=[
                KeyTimeInstance(index=4877.258035827243, name='Touchdown'),
                KeyTimeInstance(index=10815.393951826029, name='Touchdown'),
            ]),
            S(name='Offshore', frequency=2.0, items=[]),
            S(name='Takeoff', frequency=2.0, items=[
                Section(name='Takeoff', slice=slice(1081, 1101, None), start_edge=1080.3028244307038, stop_edge=1100.3028244307038),
                Section(name='Takeoff', slice=slice(6611, 6631, None), start_edge=6610.377288585346, stop_edge=6630.377288585346)
            ]),
        )

        self.assertEqual(approaches[0].type, 'LANDING')
        self.assertEqual(approaches[0].airport['name'], 'Varanus Island')
        self.assertIsNone(approaches[0].landing_runway)
        self.assertIsNone(approaches[0].approach_runway)


if __name__ == '__main__':
    unittest.main()


