import sys

from utilities.dict_helpers import dict_filter  #TODO: Mark utiltities as a dependency

from analysis import settings
from analysis.dependency_graph import dependency_order
from analysis.hdf_access import hdf_file
from analysis.library import calculate_timebase
from analysis.node import (
    DerivedParameterNode, GeoKeyTimeInstance, KeyPointValue, KeyPointValueNode,
    KeyTimeInstance, KeyTimeInstanceNode, FlightPhaseNode)


def get_required_params(aircraft):
    """
    """
    param_list = [] ##['Rate Of Descent High', 'Top of Climb and Top of Descent']
    return param_list



def geo_locate(hdf, kti_list):
    """
    Translate KeyTimeInstance into GeoKeyTimeInstance namedtuples
    
    TODO: Account for different frequency kti indexes.
    """
    if 'Latitude Smoothed' not in hdf \
       or 'Longitude Smoothed' not in hdf:
        return kti_list
    
    lat_pos = hdf['Latitude Smoothed']
    long_pos = hdf['Longitude Smoothed']
    gkti_list = []
    for kti in kti_list:
        gkti = GeoKeyTimeInstance(kti.index, kti.state,
                                  lat_pos[kti.index], long_pos[kti.index])
        gkti_list.append(gkti)
    return gkti_list


        
def derive_parameters(hdf, nodes, process_order):
    """
    Derives the parameter values and if limits are available, applies
    parameter validation upon each param before storing the resulting masked
    array back into the hdf file.
    
    :param hdf: Data file accessor used to get and save parameter data and attributes
    :type hdf: hdf_file
    :param nodes: Used to determine the type of node in the process_order
    :type nodes: NodeManager
    :param process_order: Parameter / Node class names in the required order to be processed
    :type process_order: list of strings
    """
    params = {} # store all derived params that aren't masked arrays
    kpv_list = [] # duplicate storage, but maintaining types
    kti_list = []
    phase_list = []
    
    for param_name in process_order:
        if param_name in nodes.lfl and settings.POST_LFL_PARAM_PROCESS:
            # perform any post_processing on LFL params
            param = hdf.get_param(param_name)
            _param = settings.POST_LFL_PARAM_PROCESS(hdf, param)
            if _param:
                hdf.set_param(_param)
            continue
        
        node_class = nodes.derived_nodes[param_name]  # raises KeyError if Node is "unknown"
        # retrieve dependencies which are available from hdf (LFL/Derived masked arrays)
        deps = hdf.get_params(node_class.get_dependency_names())
        # update with dependencies already derived (non-masked arrays)
        deps.update( dict_filter(params, keep=node_class.get_dependency_names()) )
        if not deps:
            raise RuntimeError("No dependencies available - Nodes cannot operate without ANY dependencies available! Node: %s" % node_class.__name__)
        # initialise node
        node = node_class(deps)
        # Derive the resulting value
        result = node.get_derived(deps)
        
        if isinstance(node, KeyPointValueNode):
            ### expect a single KPV or a list of KPVs
            #Q: track node instead of result here??
            params[param_name] = result  # keep track
            ##if isinstance(result, KeyPointValue):
                ##kpv_list.append(result)
            ##else:
            kpv_list.extend(result)
        elif isinstance(node, KeyTimeInstanceNode):
            ### expect a single KTI or a list of KTIs
            params[param_name] = result  # keep track
            ##if isinstance(result, KeyTimeInstance):
                ##kti_list.append(result)
            ##else:
            kti_list.extend(result)
        elif isinstance(node, FlightPhaseNode):
            # expect a single slice
            params[param_name] = result  # keep track
            phase_list.extend(result)
        elif isinstance(node, DerivedParameterNode):
            # perform any post_processing
            if settings.POST_DERIVED_PARAM_PROCESS:
                process_result = settings.POST_DERIVED_PARAM_PROCESS(hdf, result)
                if process_result:
                    result = process_result
            if hdf.duration:
                # check that the right number of results were returned
                assert len(result) == hdf.duration * result.frequency
            hdf.set_param(result)
        else:
            raise NotImplementedError("Unknown Type %s" % node.__class__)
        continue

    return kti_list, kpv_list, phase_list


def process_flight(hdf_path, aircraft):
    # open HDF for reading
    with hdf_file(hdf_path) as hdf:
        # get list of KPV and standard parameters to be calculated
        required_params = get_required_params(aircraft)
        # assume that all params in HDF are from LFL(!)
        lfl_params = hdf.get_param_list()
        # calculate dependency tree
        nodes, process_order = dependency_order(lfl_params, required_params, 
                                                draw=sys.platform != 'win32') # False for Windows :-(
        
        # establish timebase for start of data -- Q: When will this be used? Can we do this later on?
        ##start_datetime = calculate_timebase(hdf.years, hdf.months, hdf.days, hdf.hours, hdf.mins, hdf.seconds)
        
        if settings.PRE_FLIGHT_ANALYSIS:
            settings.PRE_FLIGHT_ANALYSIS(hdf, aircraft, process_order)
            
        kti_list, kpv_list, phase_list = derive_parameters(hdf, nodes, process_order)
        
        # go get bonus info at time of KPVs
        ##kpv_info = get_geo_location_etc_for_kpv(kpv_list)
        
        # go get bonus info at time of KPVs
        kti_info = geo_locate(kti_list)
        
        downsampled_params = downsample_for_graphs(graph_params_list)
        store_flight_information(flight_info, kti_info, params['kpvs'])  # in DB (not HDF)
    '''
    
    ##if not force_analysis:
            ### ensure the aircraft's the same one as we're told it is
            ##aircraft_found = validate_aircraft(aircraft, segment['aircraft_ident']) # raises error? or just returns aircraft?
            ##segment['aircraft'] = aircraft #TODO: DO SOMETHING CLEVER!!!
            
    # no longer exists:    
    ##flight_phase_map1 = flight_phases_basic(altitude, airspeed, heading) # inc. runway turn on / off KPTs
    ##hdf.append_table(flight_phase_map1, table='flight_phase') # add to HDF    
    
    
    ##flight_phase_map2 = compute_flight_phases()
    ##hdf.append_table(flight_phase_map2, table='flight_phase') # add to HDF
    
    # establish Airports and lookup to DB for further information
    ## Takeoff, Approach(s), Landing
    ##if lat_long_available:
        ##airport_meta_data = establish_airports(hdf)
        ##hdf.store_meta(airports=airport_meta_data)
    
    # calculate more derived parameters from the params above once confirmed valid
    
    
    #TODO: put into derived params?
    # calculate timebase file or each entire dfc
    
    segment['data_start_time'] = data_seg_start_datetime + timedelta(seconds=segment_slice.start)
    # not required for analysis, used for partial flight matching.
    segment['data_end_time'] = data_seg_start_datetime + timedelta(seconds=segment_slice.stop)

    # generic technique for algorithms (derived params, validity, correlations, etc.
    for algorithm in dependency_ordered(algorithms):
        data = algorithm.function(hdf[algorithm['parents']])
        hdf.append_table(algorithm['type'], data)
    
    
    
    #TODO: Review the order of below - should KPT / KPV be moved up and into
    #derivations?
    
    downsampled_params = downsample_for_graphs(graph_params_list)
    hdf.append_table(downsampled_params, table='table')
    
    kpts = calculate_key_point_times() # KPT -> KTI (Key Time Instances)
    kpv_list = get_required_kpv(aircraft_info)
    kpvs = calculate_key_point_values(hdf, kpv_list)
    store_flight_information(flight_info, kpts, kpvs) # in DB (not HDF)
    
    
    #=============================
    
    # Request to the Database
    request_event_detection(flight)  # uses Nile/web/thresholds.py

    '''

if __name__ == '__main__':
    import sys
    hdf_path = sys.argv[1]
    process_flight(hdf_path, None)
