# -*- coding: utf-8 -*-
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
##############################################################################
# Flight Data Analyzer Settings
##############################################################################


import os
import six
import sys

# Note: Create an analyser_custom_settings.py module to override settings for
# your local environment and append customised modules.


_path = os.path.dirname(os.path.realpath(sys.executable if getattr(sys, 'frozen', False) else __file__))


##############################################################################
# General Configuration


# Modules to import all derived Nodes from. Additional modules can be
# appended to this list in analyzer_custom_settings.py by creating a similar list of
# modules with the variable name ending with "_MODULES"
# e.g. MY_EXTRA_MODULES = ['my_package.extra_attributes', 'my_package.extra_params']
NODE_MODULES = [
    'analysis_engine.approaches',
    'analysis_engine.derived_parameters',
    'analysis_engine.multistate_parameters',
    'analysis_engine.key_point_values',
    'analysis_engine.key_time_instances',
    'analysis_engine.flight_attribute',
    'analysis_engine.flight_phase',
]
NODE_HELICOPTER_MODULE_PATHS = [
    #'analysis_engine.helicopter.approaches',
    'analysis_engine.helicopter.derived_parameters',
    'analysis_engine.helicopter.multistate_parameters',
    'analysis_engine.helicopter.key_point_values',
    'analysis_engine.helicopter.key_time_instances',
    #'analysis_engine.helicopter.flight_attribute',
    'analysis_engine.helicopter.flight_phase',
]
PRE_PROCESSING_MODULE_PATHS = [ # Cant end with _MODULES as will be added to NODE_MODULES
    'analysis_engine.pre_processing.merge_multistate_parameters',
    'analysis_engine.pre_processing.merge_parameters',
]

API_HTTP_HANDLER = 'analysis_engine.api_handler.HTTPHandler'
API_HTTP_BASE_URL = None

API_FILE_HANDLER = 'analysis_engine.api_handler.FileHandler'
API_FILE_PATHS = {
    'aircraft': os.path.join(_path, 'config', 'aircraft.yaml'),
    'airports': os.path.join(_path, 'config', 'airports.yaml'),
    'runways': os.path.join(_path, 'config', 'runways.yaml'),
    'exports': os.path.join(_path, 'config', 'exports.yaml'),
}

API_HANDLER = API_FILE_HANDLER

# User's home directory, override in analyser_custom_settings.py
WORKING_DIR = os.path.expanduser('~')

# Cache parameters which are used more than n times in HDF
CACHE_PARAMETER_MIN_USAGE = 0


##############################################################################
# Segment Splitting


# Minimum duration of slow airspeed in seconds to split flights inbetween.
MINIMUM_SPLIT_DURATION = 600
ROTOR_MINIMUM_SPLIT_DURATION = 4

# Minimum duration of a fast airspeed to splt into a segment.
MINIMUM_FAST_DURATION = 60

# When the average normalised value of selected parameters drops below this
# value, a flight split can be made.
MINIMUM_SPLIT_PARAM_VALUE = 0.175

# Threshold for splitting based upon rate of turn. This threshold dictates
# when the aircraft is not considered to be turning.
HEADING_RATE_SPLITTING_THRESHOLD = 0.1

# Parameter names to be normalised for splitting flights.
SPLIT_PARAMETERS = ('Eng (1) N1', 'Eng (2) N1', 'Eng (3) N1', 'Eng (4) N1',
                    'Eng (1) N2', 'Eng (2) N2', 'Eng (3) N2', 'Eng (4) N2',
                    'Eng (1) NP', 'Eng (2) NP', 'Eng (3) NP', 'Eng (4) NP')


##############################################################################
# Node Cache


# Node cache determines whether Nodes will be cached during processing to avoid
# unnecessary array alignment. Caching parameters will increase memory usage.
NODE_CACHE = True

# The number of decimal places which the offset of cached parameters will be
# accurate to. A value of None will retain full accuracy.
NODE_CACHE_OFFSET_DP = None


##############################################################################
# Parameter Analysis

# FlightDataConverter parameters which wrap from 359 to 0 or -180 to 180
WRAPPING_PARAMS = (
    # Engine angles
    'Eng (1) Low Press Turbine Imbalance Angle',
    'Eng (2) Low Press Turbine Imbalance Angle',
    'Eng (1) Fan Imbalance Angle',
    'Eng (2) Fan Imbalance Angle',
    'Eng (1) Imbalance Angle',
    'Eng (2) Imbalance Angle',
    # Heading
    'Heading',
    'Heading (FO)',
    'Heading (Capt)',
    'Heading True',
    'Heading True (FO)',
    'Heading True (Capt)',
    ##'Heading Selected', - is a binary selection, not an angular parameter.
    'IRU Heading (L)',
    'IRU Heading (C)',
    'IRU Heading (R)',
    # Longitude
    'Longitude',  # When flying past the Bering Sea (-180 to +180)
    'Longitude (1)',
    'Longitude (2)',
    'Longitude (3)',
    'Longitude (FO)',
    'Longitude (Capt)',
    'Longitude Recorded',
    # Track
    'Track Angle',
    'Track Angle True',
    'FMC Track Angle True',
    'IRU Track Angle True',
    # Wind
    'Wind Direction',
    'Wind Direction (1)',
    'Wind Direction (2)',
    'Wind Direction True',
    'Wind Direction Magnetic',
    'FMC Wind Direction True',
    # Altitude in wrapped format
    'Altitude STD (Fine)',
)


# The limit of compensation for normal accelerometer errors. For example, to
# allow for an accelerometer to lie in the range 0.8g to 1.2g, enter a value
# of 0.2. An accelerometer with an average reading of 1.205g during the taxi
# phases (in and out) will not be corrected, and all the acceleration KPVs
# will carry that offset.
ACCEL_NORM_OFFSET_LIMIT = 0.3 #g

# The limits for lateral and longitudinal accelerometer errors.
ACCEL_LAT_OFFSET_LIMIT = 0.1 # g
ACCEL_LON_OFFSET_LIMIT = 0.1 # g

# The minimum sensible duration for being airborne, used to reject skips and bounced landings.
AIRBORNE_THRESHOLD_TIME = 60  # secs

# An airspeed or rotor speed below which you just can't possibly be flying.
AIRSPEED_THRESHOLD = 80  # kts

# The minimum sensible duration for being declaring a START_AND_STOP
AIRSPEED_THRESHOLD_TIME = 3 * 60  # secs

# Min number of samples to use when creating hash of airspeed
# 64 samples is enough to exceed arinc and short enough to not affect flights
AIRSPEED_HASH_MIN_SAMPLES = 64

# Transition altitude between Altitude Rad and Altitude STD for Altitude AAL
ALTITUDE_AAL_TRANS_ALT = 100.0

# Altitude AAL complementary filter timeconstant
ALTITUDE_AAL_LAG_TC = 3.0

# Altitude to break flights into separate climb/cruise/descent segments.
# This is applied to altitude with hysteresis, so break will happen when
# climbing above 15000 ft and below 10000 ft.
ALTITUDE_FOR_CLB_CRU_DSC = 12500

# The maximum signal range we accommodate.
ALTITUDE_RADIO_MAX_RANGE = 5000.0 # ft

# The smallest radio altimeter overflow step which will be corrected.
# Most radio altimeters are scaled to overflow at 4096 or 2048ft, but some
# A340s have a resolution of 0.5ft and overflow at 11 bits = 1024ft increments.
ALTITUDE_RADIO_MIN_OVERFLOW = 1024.0

# The maximum radio altimeter offset we are able to compensate for.
# NB: Negative offsets are not corrected, assumed to be from oleo compression.
ALTITUDE_RADIO_OFFSET_LIMIT = 10.0

# Minimum descent height range for an approach and landing phase.
APPROACH_MIN_DESCENT = 500

# Cycle size; the climb needed to end one approach and initiate the next
APPROACH_AND_LANDING_CYCLE_SIZE = 500

# Resolved vertical acceleration washout time constant. This long period
# function removes any standing offset to the resolved acceleration signal
# and is essential in the vertical velocity complementary filter.
AZ_WASHOUT_TC = 60.0

# As above for the along-track resolved acceleration term.
AT_WASHOUT_TC = 60.0

# Minimum threshold for detecting a bounced landing. Bounced landings lower
# than this will not be identified or held in a database. Note: The event
# threshold is higher than this.
BOUNCED_LANDING_THRESHOLD = 2.0
BOUNCED_MAXIMUM_DURATION = 20  # sec

# The duration to be scanned for peak accelerations during takeoff and landing.
# To avoid overlap, accelerations in flight are "trimmed" to avoid this period.
BUMP_HALF_WIDTH = 3.0 # sec

# Force to start checking control stiffness. Intended to be the same setting
# for all three flying controls.
CONTROL_FORCE_THRESHOLD = 3.0  # lb

#Less than 5 mins you can't do a circuit, so we'll presume this is a data
#snippet
FLIGHT_WORTH_ANALYSING_SEC = 300

# Minimum duration of flight in seconds
##DURATION_THRESHOLD = 60  # sec

# Threshold for start of climb phase
CLIMB_THRESHOLD = 1000  # ft AAL

# Minimum period of a climb or descent for testing against thresholds
# (reduces number of KPVs computed in turbulence)
CLIMB_OR_DESCENT_MIN_DURATION = 10  # sec

# Tolerance of controls (Pitch/Roll (Captain/FO)) when in use in degrees.
# Used when trying determine which pilot is actively using the controls.
CONTROLS_IN_USE_TOLERANCE = 1

# Pilot in control - difference between each control force as a ratio
CONTROL_COLUMN_IN_USE_RATIO = 1.30  # %

# Change in altitude to create a Descent Low Climb phase, from which
# approaches, go-around and touch-and-go phases and instances derive.
DESCENT_LOW_CLIMB_THRESHOLD = 500 #ft

# Acceleration due to gravity
GRAVITY_IMPERIAL = 32.2  # ft/sec^2 - used for combining acceleration and height terms

# Acceleration due to gravity
GRAVITY_METRIC = 9.81  # m/sec^2 - used for comibining acceleration and groundspeed terms

# Groundspeed complementary filter time constant.
GROUNDSPEED_LAG_TC = 6.0  # seconds

# Threshold for start and end of Mobile phase when groundspeed is available.
GROUNDSPEED_FOR_MOBILE = 2.0  # kts

# The minimum amount of heading change in degrees that would satisfy movement
# on the ground representative of taxiing. Any flight / taxi will normally
# exceed this value massively.
HEADING_CHANGE_TAXI_THRESHOLD = 60  # deg

# Threshold for start and end of Mobile phase
HEADING_RATE_FOR_MOBILE = 0.5  # deg/sec

# Threshold for straight cruising flight
HEADING_RATE_FOR_STRAIGHT_FLIGHT = 1.0 # deg/sec

# Threshold for turn onto runway at start of takeoff.
HEADING_TURN_ONTO_RUNWAY = 15.0  # deg

#Threshold for turn off runway at end of takeoff. This allows for turning
#onto a rapid exit turnoff, and so we are treating deceleration down the RET
#as part of the landing phase. Notice that the KTI "Landing Turn Off Runway"
#will determine the point of turning off the runway centreline in either
#case, using the peak curvature technique.
# Reduced from 60 to 45 deg after we found some cargo operators whose pattern
# after leaving the runway did not exceed 60 deg.
HEADING_TURN_OFF_RUNWAY = 45.0  # deg

# Holding pattern criteria.
# Minimum time used to detect and as the timebase for differentiation of heading.
HOLDING_MIN_TIME = 5*60  #sec
# Rate of turn on average over this period
HOLDING_ROT = 0.6 # deg/sec
# Maximum groundspeed over the period in the hold.
HOLDING_MAX_GSPD = 110.0  # kts
# The hold must be at least one complete turn (+/- 5 degrees)
HOLDING_MIN_TURN = 355.0 # deg

# Threshold for flight phase altitude hysteresis.
HYSTERESIS_FPALT = 200  # ft

# Hysteresis for engine start & stop. Stops nuisance engine start/stop changes.
HYSTERESIS_ENG_START_STOP = 2.5 #%

# Threshold for flight phase airspeed hysteresis.
HYSTERESIS_FPIAS = 5  # kts

# Threshold for flight phase altitude hysteresis specifically for separating
# Climb Cruise Descent phases.
HYSTERESIS_FPALT_CCD = 500  # ft
# Note: Original value was 2,500ft, based upon normal operations, but
# circuits flown below 2,000ft agl were being processed incorrectly. We
# therefore squash the altitude signal above 10,000ft so that changes of
# altitude to create a new flight phase have to be five times greater; 500ft
# changes below 10,000ft are significant, while above this 2,500ft is more
# meaningful.

# Threshold for radio altimeter hysteresis
# (used for flight phase calculations only)
HYSTERESIS_FP_RAD_ALT = 5  # ft

# Threshold for flight phase vertical speed hysteresis.
# We're going to ignore changes smaller than this to avoid repeatedly changing
# phase if the aircraft is climbing/descending close to a threshold level.
HYSTERESIS_FPROC = 40  # fpm / RMS altitude noise
# The threshold used is scaled in proportion to the altitude noise level, so
# that for the Hercules we can get up to 400 fpm or more, a value which has
# been selected from inspection of test data which is notoriously noisy. By
# measuring the noise, we don't burden "quieter" aircraft unnecessarily.

# Threshold for rate of turn hysteresis.
HYSTERESIS_FPROT = 2  # deg/sec

# ILS Capture threshold. Set lower than the lowest exceedance threshold.
ILS_CAPTURE = 0.5  # dots
ILS_CAPTURE_ROC = 0.1 # dots/sec
ILS_ESTABLISHED_DURATION = 10.0 # sec

# Full scale reading on the ILS
ILS_MAX_SCALE = 2.5  # dots

# Tolerance bands for comparison of two sources
ILS_GS_SPREAD = 0.05
ILS_LOC_SPREAD = 0.1

# Initial approach threshold height
INITIAL_APPROACH_THRESHOLD = 3000  # ft

# Threshold for start of initial climb phase
INITIAL_CLIMB_THRESHOLD = 35  # ft (Radio, where available)

# Threshold for start of braking / reverse thrust on landing.
LANDING_ACCELERATION_THRESHOLD = -0.1  # g
# TODO: Was -0.2g set to -0.1 for Herc testing - revert or not???

# Threshold for start of landing phase
LANDING_THRESHOLD_HEIGHT = 50  # (Radio, where available)

# Speed for end of landing roll (Groundspeed where available, else TAS)
LANDING_ROLL_END_SPEED = 65.0

# Level flight minimum duration
LEVEL_FLIGHT_MIN_DURATION = 20  # sec

# Maximum age of a Segment's timebase in days. A value of None allows any age.
MAX_TIMEBASE_AGE = 365 * 10  # days

# Heading change KPV rejects turns below this threshold. 270deg splits full
# orbits (which we want to measure) from turns in the hold.
MIN_HEADING_CHANGE = 270.0

# Engine core speed for engine starting.
CORE_START_SPEED = 35.0  # %

# Engine core speed for engine to stopping.
# Note: Challenger 300 reports 49.5% after engine start.
CORE_STOP_SPEED = 35.0  # %


# Minimum values for determining a running engine, used in segment type
# detection as well as Eng Running multistates
MIN_FAN_RUNNING = 10.0  # N1/Np %
MIN_CORE_RUNNING = 10.0  # N2/N3/Ng %
MIN_FUEL_FLOW_RUNNING = 50  # kg/hr

# Minimum proportion of valid data for unused fuel tanks.
MIN_VALID_FUEL = 0.25

# Threshold for Longitudinal Acceleration Offset Removed dropping to after
# a Takeoff Acceleration Start.
REJECTED_TAKEOFF_THRESHOLD = 0

# Collective value below this threshold is considered to be on ground. Only
# used for segment type classifiction, as Nr can be high throughout the segment
COLLECTIVE_ON_GROUND_THRESHOLD = 20.0 # %

'''
See experimental KTP LandingStopLimitPointPoorBraking et seq.
# Mu values for good, medium and poor braking action (Boeing definition).
MU_GOOD = 0.2
MU_MEDIUM = 0.1
MU_POOR = 0.05 # dimensionless.
'''

# Pilot action threshold for TCAS responses
TCAS_SCAN_TIME = 10.0 # seconds
TCAS_THRESHOLD = 0.15 # g

# Transition altitude - above use Altitude STD, below use Altitude AAL.
# Note: This affects the generic KTIs and dependent nodes only. Some KPVs are
# "hard wired", e.g. 8000-10000 ft ranges.
TRANSITION_ALTITUDE = 8000  # ft

# Vertical speed limits of 800 fpm and -500 fpm gives good distinction with
# level flight. Separately defined to allow for future adjustment.
VERTICAL_SPEED_FOR_CLIMB_PHASE = 800  # fpm
VERTICAL_SPEED_FOR_DESCENT_PHASE = -500  # fpm

# Vertical speed limits of 300 fpm to identify airborne after takeoff and end
# of descent, when relying solely upon pressure altitude data.
VERTICAL_SPEED_FOR_LEVEL_FLIGHT = 300  # fpm

# Vertical speed for liftoff. This builds upon the intertially smoothed
# vertical speed computation to identify accurately the point of liftoff.
# At 100fpm this can repond to motion on a bumpy runway, hence 150fpm as the threshold.
VERTICAL_SPEED_FOR_LIFTOFF = 150  # fpm

# Vertical speed for touchdown.
VERTICAL_SPEED_FOR_TOUCHDOWN = -100  # fpm

# Vertical speed complementary filter timeconstant
VERTICAL_SPEED_LAG_TC = 5.0  # sec

# Heading Rate (rate of turn) limits for flight.
# (Also used for validation of accelerometers on ground).
HEADING_RATE_FOR_FLIGHT_PHASES_FW = 2.0  # deg per second
HEADING_RATE_FOR_FLIGHT_PHASES_RW = 10.0  # deg per second

# Heading Rate limit for taxi event.
HEADING_RATE_FOR_TAXI_TURNS = 5.0  # deg per second

# Duration of masked data to repair by interpolation for flight phase analysis
REPAIR_DURATION = 10  # seconds

# Minimum engine speed for reverse thrust to be considered effective.
REVERSE_THRUST_EFFECTIVE_EPR = 1.25 # %EPR
REVERSE_THRUST_EFFECTIVE_N1 = 65 # %N1

# Threshold for spoiler deployment when operating as speedbrake in flight.
# See KPV "AirspeedWithSpoilerDeployedMax"
SPOILER_DEPLOYED = 5.0 # deg

# Acceleration forwards at the start of the takeoff roll.
# Was 0.1g, but increased to avoid nuisance triggers during enthusiastic taxiing.
TAKEOFF_ACCELERATION_THRESHOLD = 0.15  # g

# Height in ft where Altitude AAL switches between Radio and STD sources.
# Changed from 100ft to 50ft to remove problems at airports with dips imediatly
# before runway start.
TRANSITION_ALT_RAD_TO_STD = 50

# The takeoff and landing acceleration algorithm linear estimation period
TRUCK_OR_TRAILER_INTERVAL = 3  # samples: should be odd.

# The takeoff and landing acceleration algorithm linear estimation period
TRUCK_OR_TRAILER_PERIOD = 7  # samples

# Top of Climb / Top of Descent Threshold.
"""This threshold was based upon the idea of "Less than 600 fpm for 6 minutes"
This was often OK, but one test data sample had a 4000ft climb 20 mins
after level off. This led to reducing the threshold to 600 fpm in 3
minutes which has been found to give good qualitative segregation
between climb, cruise and descent phases."""
SLOPE_FOR_TOC_TOD = 600 / float(3*60)  # 600fpm in 3 mins

# Tolerance values used when determining the nearest runway:
RUNWAY_HEADING_TOLERANCE = 30  # deg
RUNWAY_ILSFREQ_TOLERANCE = 50  # kHz



"""
POLARIS Settings for helicopter flight data analysis.
"""

# The minimum sensible duration for being airborne, lower than fixed wing!
AIRBORNE_THRESHOLD_TIME_RW = 20  # secs

# Hover taxi co.

HOVER_TAXI_HEIGHT = 20 # ft
HOVER_TAXI_MIN_DURATION = 5 # sec

# Transition altitude between Altitude Rad and Altitude STD for Altitude AAL
ALTITUDE_AGL_SMOOTHING = 10 # Seconds for smoothing period.
ALTITUDE_AGL_TRANS_ALT = 2000.0 # ft

# Cycle size; the climb needed to end one approach and initiate the next
APPROACH_AND_LANDING_CYCLE_SIZE = 200

# Shaft speed split for autorotation
AUTOROTATION_SPLIT = 1.0 # %

# Helicopters fly more slowly so have higher rates of turn.
HEADING_RATE_FOR_FLIGHT_PHASES = 10.0
HEADING_RATE_FOR_TAXI = 5.0

# Maximum groundspeed for hover
HOVER_GROUNDSPEED_LIMIT = 10 # kts
# Maximum height for hover
HOVER_HEIGHT_LIMIT = 300.0 # ft
# Hovers must at some time get below this level.
HOVER_MIN_HEIGHT = 20 # ft
# Minimum duratino for  a hover phase.
HOVER_MIN_DURATION = 5.0 # sec


# Collective threshold for end of landing.
LANDING_COLLECTIVE_PERIOD = 10.0

# Height at start of landing phase.
LANDING_HEIGHT = 20.0

# Period to look back from touchdown for either 20ft or highest point as start of landing.
LANDING_TRACEBACK_PERIOD = 30.0 # sec

# The minimum sensible rotor speed to become airborne. Used for flight splitting (nb: rotor speed can fall below this level in flight).
ROTORSPEED_THRESHOLD = 50  # % Nr

# The minimum sensible duration for being declaring a START_AND_STOP
ROTORSPEED_THRESHOLD_TIME = 3 * 60  # secs

# Rotor speed below which nothing interesting happens (except engine and rotor starts)
ROTORS_TURNING = 50 # %

# Min number of samples to use when creating hash of airspeed (or Rotor Speed)
# 64 samples is enough to exceed arinc and short enough to not affect flights
SPEED_HASH_MIN_SAMPLES = 64

# Time before and after liftoff for estimate of takeoff. Refined by collective where possible.
TAKEOFF_PERIOD = 5.0

# Conditions for transition calculation.
ROTOR_TRANSITION_ALTITUDE = 300 # ft
ROTOR_TRANSITION_SPEED_LOW = 40 # kts
ROTOR_TRANSITION_SPEED_HIGH = 60 # kts
# ROTOR_TRANSITION_SPEED_HIGH reduced from 80kts to allow for lower speed climbs from S76


##############################################################################
# KPV/KTI Name Values

# These are some common frequently used name values defined here to be used in
# multiple key point values or key time instances for consistency.

NAME_VALUES_ENGINE = {'number': [1, 2, 3, 4]}

NAME_VALUES_CLIMB = {'altitude': [
    10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000,
    2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
]}

NAME_VALUES_DESCENT = {'altitude': NAME_VALUES_CLIMB['altitude'][::-1]}

# Review comment in DistanceFromLandingAirport before adding smaller values to this list.
# distance from airport
NAME_VALUES_DISTANCE = {'distance': [150, 250]}

# distance to landing threshold
NAME_VALUES_RANGES = {'distance': [0, 1, 2, 4]}

##############################################################################
# Custom Settings

# Import from custom_settings if exists
try:
    from analyser_custom_settings import *  # NOQA
    # add any new modules to the list of modules
    from copy import copy
    for k, v in six.iteritems(copy(locals())):
        if k.endswith('_MODULES') and k != 'NODE_MODULES':
            NODE_MODULES.extend(v)
    # We want to preserve the order of the modules as later we will want to
    # use the nodes from the additional modules in preference to those in the
    # analyzer.
    seen = set()
    seen_add = seen.add
    NODE_MODULES = [x for x in NODE_MODULES if not (x in seen or seen_add(x))]
except ImportError as err:
    # logger.info preferred, but stack trace is important when trying to
    # determine an unexpected ImportError lower down the line.
    import logging
    logger = logging.getLogger(name=__name__)
    logger.addHandler(logging.NullHandler())
    logger.exception("Unable to import analysis_engine/analyser_custom_settings.py")
    pass

##############################################################################
# KPV/KTI Name Values (#2)

# Note: These must be created after the custom settings have been imported.

from flightdatautilities import aircrafttables as at

NAME_VALUES_FLAP = {'flap': at.get_flap_detents()}

NAME_VALUES_SLAT = {'slat': at.get_slat_detents()}

NAME_VALUES_AILERON = {'aileron': at.get_aileron_detents()}

NAME_VALUES_CONF = {'conf': at.get_conf_detents()}

NAME_VALUES_LEVER = {'flap': at.get_lever_detents()}  # XXX: Key must be 'flap'
