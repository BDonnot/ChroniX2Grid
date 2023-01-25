# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

# Python built-in modules
import warnings
import os
import json
import pandas as pd

# Chronix2grid modules
from .preprocessing.pivot_KPI import ref_syn_data#pivot_format_ref_data#pivot_format
from .deterministic.kpis import EconomicDispatchValidator
from ..generation import generation_utils as gu
from .. import constants as cst
from .. import utils as ut
from datetime import datetime, timedelta



def main(kpi_input_folder, generation_output_folder, scenario_names,
         kpi_output_folder, year, case, n_scenarios, params,
         loads_charac, prods_charac, scenario_id=None):
    """
        This is the main function for KPI computation. It formats synthetic and reference chronics and then computes KPIs on it with 2 modes (wind solar and load only or full energy mix). It saves plots in png and html files. It writes a json output

        Parameters
        ----------
        kpi_input_folder (str): path to folder of kpi inputs, which contains paramsKPI.json and benchmark folders. paramsKPI.json tells which benchmark to read as reference
        generation_input_folder (str): path to the input folder of generation module (to read chronics before dispatch)
        generation_output_folder (str): path to the output folder of generation module (to read chronics after dispatch)
        images_repo (str): path to the images folder in which KPI plots will be saved
        year (int): year in which chronics have been generated
        case (str): identify the studied case for chronics generation, such as l2rpn_118
        n_scenario (int): number of scenarios to consider (KPIs are computed for each scenario succesively)
        wind_solar_only (boolean): True if the generated chronics contain only wind, solar and load chronics, False otherwise
        params (dict): configuration params computed from params.json, such as timestep or mesh characteristics
        loads_charac (pandas.DataFrame): characteristics of loads node such as Pmax, type of demand and region
        prods_charac (pandas.DataFrame): characteristics of generators such as Pmax, carrier and region

    """


    ut.check_scenario(n_scenarios, scenario_id)

    print('=====================================================================================================================================')
    print('================================================= KPI GENERATION  ===================================================================')
    print('=====================================================================================================================================')

    warnings.filterwarnings("ignore")


    # Create single zone if no zone is given
    if 'zone' not in prods_charac.columns:
        prods_charac['zone'] = 'R1'
        loads_charac['zone'] = 'R1'

    # Format and compute KPI for each scenario
    chronic_dirs=list_dirs_with_chronics(generation_output_folder)
    for scenario_num in range(min(n_scenarios,len(chronic_dirs))):
        #if n_scenarios > 1:
        #    scenario_name = scenario_names(scenario_num)
        #else:
        #    scenario_name = scenario_names(scenario_id)

        #################
        # here go for some existing scenarios if that is the case


        #scenario_generation_output_folder = os.path.join(
        #    generation_output_folder, scenario_name
        #)

        if n_scenarios > 1:
            scenario_generation_output_folder = chronic_dirs[scenario_num]
        else:
            possible_scenario_name=scenario_names(scenario_id)
            dir_id=[i for i in range(len(chronic_dirs)) if possible_scenario_name in chronic_dirs[i]]
            if len(dir_id)!=0:#if time series are being generated and not all chronics folders have been populated
                scenario_generation_output_folder=chronic_dirs[dir_id[0]]
            else:#if from an existing env with populated chronics folders
                scenario_generation_output_folder=chronic_dirs[scenario_id]

        # check there is load, solar and wind:
        has_load, has_solar, has_wind, has_thermal = check_solar_wind_prod_data(scenario_generation_output_folder,
                                                                                prods_charac)

        wind_solar_only = (has_solar or has_wind) and not has_thermal

        if not (has_load and has_wind and has_solar):
            print(
                "missing load or solar or wind generated timeseries in output folder: " + scenario_generation_output_folder)

        scenario_name = os.path.basename(scenario_generation_output_folder)
        print(scenario_name + '...')#need to change start date to chronic date if already generated

        params=update_time_params_scenario(scenario_generation_output_folder,params)

        scenario_image_folder = os.path.join(
            kpi_output_folder, scenario_name, cst.KPI_IMAGES_FOLDER_NAME
        )
        os.makedirs(scenario_image_folder, exist_ok=True)

        ######################

        # Return Warning if KPIs are not computed on full year. Yet, the computation will work
        if params['weeks'] != 52:
            print('Warning: KPI are incomplete. Computation has been made on '+str(params['weeks'])+' weeks, but are meant to be computed on 52 weeks')

        # Read reference and synthetic chronics, but also KPI configuration, in pivot format. 2 modes: with or without full dispatch
        #if wind_solar_only:
        #    # Get reference and synthetic dispatch and loads
        #    (ref_dispatch, ref_consumption, syn_dispatch, syn_consumption,
        #     paramsKPI) = pivot_format(
        #        scenario_generation_output_folder, kpi_input_folder, year,
        #        prods_charac, loads_charac,
        #        params, case,wind_solar_only,kpi_on_syn_data_only)
        #    ref_prices = None
        #    prices = None
        #
        ref_dispatch, ref_load, syn_dispatch, syn_load,ref_prices, prices, paramsKPI,kpi_on_syn_data_only=ref_syn_data(scenario_generation_output_folder,
                                                                                          kpi_input_folder, year,
                                                                                          prods_charac, loads_charac, params, case,
                                                                                          wind_solar_only)
        #else:
        #    # Get reference and synthetic dispatch and loads
        #    (ref_dispatch, ref_consumption, syn_dispatch, syn_consumption,
        #     ref_prices, prices, paramsKPI) = pivot_format(
        #        scenario_generation_output_folder, kpi_input_folder, year,
        #        prods_charac, loads_charac,
        #        params, case,wind_solar_only,kpi_on_syn_data_only)

        ## Start and Run Economic dispatch validator
        # -- + -- + -- + -- + -- + -- + --
        print ('(1) Computing KPI\'s...')
        dispatch_validator = EconomicDispatchValidator(ref_load,
                                                       syn_load,
                                                       ref_dispatch,
                                                       syn_dispatch,
                                                       year,
                                                       scenario_image_folder,
                                                       prods_charac=prods_charac,
                                                       loads_charac=loads_charac,
                                                       ref_prices=ref_prices,
                                                       syn_prices=prices,
                                                       kpi_on_syn_data_only=kpi_on_syn_data_only)


        # Compute dispatch temporal view
        if wind_solar_only:
            max_col = 1
        else:
            max_col = 2
        if not kpi_on_syn_data_only:
            dispatch_validator.plot_carriers_pw(curve='reference', stacked=True, max_col_splot=max_col, save_html=True,
                                                wind_solar_only=wind_solar_only)
        dispatch_validator.plot_carriers_pw(curve='synthetic', stacked=True, max_col_splot=max_col, save_html=True,
                                            wind_solar_only=wind_solar_only)
        if not kpi_on_syn_data_only:
            dispatch_validator.plot_carriers_pw(curve='reference', stacked=False, max_col_splot=max_col, save_html=True,
                                                wind_solar_only=wind_solar_only)
        dispatch_validator.plot_carriers_pw(curve='synthetic', stacked=False, max_col_splot=max_col, save_html=True,
                                            wind_solar_only=wind_solar_only)

        # Get Load KPI
        dispatch_validator.load_kpi()

        # Get Wind KPI
        dispatch_validator.wind_kpi()

        # Get Solar KPI
        cloud_quantile = float(paramsKPI['cloudiness_quantile'])
        cond_below_cloud = float(paramsKPI['cloudiness_factor'])
        hours = paramsKPI["night_hours"]
        monthly_pattern = paramsKPI["seasons"]
        dispatch_validator.solar_kpi(cloud_quantile=cloud_quantile, cond_below_cloud=cond_below_cloud,
                                     monthly_pattern=monthly_pattern, hours=hours)

        # Wind - Solar KPI
        dispatch_validator.wind_load_kpi()

        # These KPI only if dispatch has been made
        if not wind_solar_only:
            # Get Energy Mix
            dispatch_validator.energy_mix()

            # Get Hydro KPI
            dispatch_validator.hydro_kpi()

            # Get Nuclear KPI
            dispatch_validator.nuclear_kpi()

            # Get Thermal KPI
            dispatch_validator.thermal_kpi()
            dispatch_validator.thermal_load_kpi()


        # Write json output file
        # -- + -- + -- + -- + --
        print ('(2) Generating json output file...')

        kpi_scenario_output_folder = os.path.join(
            kpi_output_folder, scenario_name)
        ec_validator_path = os.path.join(
            kpi_scenario_output_folder, 'ec_validator_output.json')
        with open(ec_validator_path, 'w') as json_f:
            json.dump(dispatch_validator.output, json_f)

        print ('-Done-\n')

def list_dirs_with_chronics(generation_output_folder):
    """
    Get start_date and time resolution for a given generated scenario

    Parameters
    ----------
    generation_output_folder: ``str``
        path to folder which contains generated scenario

    Returns
    -------
    chronic_dirs: :class:`list`
        list of scenarios
    """
    chronic_dirs = set()
    for item in os.listdir(generation_output_folder):
        subfolder = os.path.join(generation_output_folder, item)
        if os.path.isfile(os.path.join(subfolder, 'prod_p.csv.bz2')):
            chronic_dirs.add(subfolder)

        if os.path.isfile(os.path.join(subfolder, 'load_p.csv.bz2')):
            chronic_dirs.add(subfolder)
    l_chronic_dirs=list(chronic_dirs)
    chronic_dirs=sorted(l_chronic_dirs, key=lambda i: os.path.splitext(os.path.basename(i))[0])

    return chronic_dirs

def init_date_time(scenario_folder,start_datetime_ref,time_interval_ref):

    """
    Get start_date and time resolution for a given generated scenario

    Parameters
    ----------
    scenario_folder: ``str``
        path to folder which contains generated scenario
    start_datetime_ref: :class:`Timestamps`
        user input reference start date of scenario if not in info file
    time_interval_ref: :class:`int`
        user input reference time resolution in minutes if not in info file

    Returns
    -------
    start_datetime: :class:`Timestamps`
        start date of scenario
    time_interval: :class:`int`
        time resolution in minutes
    """
    start_datetime=start_datetime_ref
    time_interval=time_interval_ref

    if os.path.exists(os.path.join(scenario_folder, "start_datetime.info")):
        with open(os.path.join(scenario_folder, "start_datetime.info"), "r") as f:
            a = f.read().rstrip().lstrip()
        try:
            tmp=pd.to_datetime(a, format='%Y-%m-%d')
        except Exception:
            raise Exception(
                'Impossible to understand the content of "start_datetime.info". Make sure '
                'it\'s composed of only one line with a datetime in the "%Y-%m-%d %H:%M"'
                "format."
            )
        start_datetime = tmp

    if os.path.exists(os.path.join(scenario_folder, "time_interval.info")):
        with open(os.path.join(scenario_folder, "time_interval.info"), "r") as f:
            a = f.read().rstrip().lstrip()
        try:
            tmp = datetime.strptime(a, "%H:%M")
        except ValueError:
            tmp = datetime.strptime(a, "%M")
        except Exception:
            raise Exception(
                'Impossible to understand the content of "time_interval.info". Make sure '
                'it\'s composed of only one line with a datetime in the "%H:%M"'
                "format."
            )
        time_interval = timedelta(hours=tmp.hour, minutes=tmp.minute).total_seconds()/60
    return start_datetime,time_interval

def update_time_params_scenario(scenario_generation_output_folder,params):

    start_datetime, time_interval = init_date_time(scenario_generation_output_folder,params['start_date'],params['dt'])
    params['end_date'] += start_datetime - params['start_date']
    params['start_date'] = start_datetime
    params['dt'] = time_interval
    return params

def check_solar_wind_prod_data(generation_output_folder,prods_charac):
    """
    For given generated scenarios, check which type of timeseries have been generated

    Parameters
    ----------
    generation_output_folder: ``str``
        path to folder which contains subfolders of generated scenarios
     prods_charac: :class:`pandas.DataFrame`
        dataframe with exepcted productions on the related grid with names and types (wind,solar,hydro,nuclear,thermal)

    Returns
    -------
    has_load: :class:`bool`
        Boolean if load timeseries exist

    has_solar: :class:`bool`
        Boolean if solar timeseries exist

    has_wind: :class:`bool`
        Boolean if wind timeseries exist

    has_thermal: :class:`bool`
        Boolean if thermal timeseries exist

    """
    # check there is load, solar and wind:
    has_solar = False
    has_wind = False
    has_load = False

    # check if other prod_p: there should be more generator names than the ones in solar and wind
    has_thermal = False
    all_prod_names = []
    solar_names = []
    wind_names = []
    for root, dirs, filenames in os.walk(generation_output_folder):
        if 'prod' in str(filenames) and len(all_prod_names) == 0:
            if len(all_prod_names) == 0:
                all_prod_names = pd.read_csv(os.path.join(root, 'prod_p.csv.bz2'),
                                             sep=';', decimal='.', index_col=0, nrows=0).columns.tolist()
        if 'solar_p' in str(filenames) and len(solar_names) == 0:
            if len(solar_names) == 0:
                solar_names = pd.read_csv(os.path.join(root, 'solar_p.csv.bz2'),
                                          sep=';', decimal='.', index_col=0, nrows=0).columns.tolist()
                has_solar = True
        if 'wind_p' in str(filenames) and len(wind_names) == 0:
            if len(wind_names) == 0:
                wind_names = pd.read_csv(os.path.join(root, 'solar_p.csv.bz2'),
                                         sep=';', decimal='.', index_col=0, nrows=0).columns.tolist()
                has_wind = True
        if 'load_p' in str(filenames):
            has_load = True

    if not has_solar:
        # check that solar generator names and wind generator names are in prods_p
        possible_solar_names = set(prods_charac["name"][prods_charac.type == "solar"].values)
        solar_names=possible_solar_names.intersection(all_prod_names)
        if len(solar_names) != 0:
            has_solar = True
            if len(solar_names)!=len(possible_solar_names):
                print("these solar farms are not in data: "+str(possible_solar_names-solar_names))

    if not has_wind:
        # check that solar generator names and wind generator names are in prods_p
        possible_wind_names = set(prods_charac["name"][prods_charac.type == "wind"].values)
        wind_names = possible_wind_names.intersection(all_prod_names)
        if len(wind_names) != 0:
            has_wind = True
            if len(wind_names)!=len(possible_wind_names):
                print("these wind farms are not in data: "+str(possible_wind_names-wind_names))

    wind_solar_only = has_solar or has_wind
    n_prods = len(all_prod_names)
    n_solar_winds = len(solar_names) + len(wind_names)
    if n_prods >= 1 and n_prods > n_solar_winds:
        has_thermal = True
        wind_solar_only = False

    if not (has_load and has_wind and has_solar):
        print("missing load or solar or wind generated timeseries in output folder: " + generation_output_folder)

    return has_load,has_solar,has_wind,has_thermal

