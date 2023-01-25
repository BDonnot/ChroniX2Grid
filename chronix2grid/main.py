# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import time
import pathlib
import shutil

import click
import multiprocessing
from functools import partial
from numpy.random import default_rng

from chronix2grid.GeneratorBackend import GeneratorBackend
from chronix2grid import constants as cst
from chronix2grid.generation import generate_chronics as gen
from chronix2grid.generation import generation_utils as gu
from chronix2grid.kpi import main as kpis
from chronix2grid.output_processor import (
    output_processor_to_chunks, write_start_dates_for_chunks)
from chronix2grid.seed_manager import (parse_seed_arg, generate_default_seed,
                                       dump_seeds)
from chronix2grid import utils as ut


@click.command()
@click.option('--case', default='case118_l2rpn_neurips_1x', help='case folder to base generation on')
@click.option('--start-date', default='2012-01-01', help='Start date to generate chronics')
@click.option('--weeks', default=4, help='Number of weeks to generate')
@click.option('--by-n-weeks', default=4, help='Size of the output chunks in weeks')
@click.option('--n_scenarios', default=2, help='Number of scenarios to generate')
@click.option('--mode', default='LRT', help='Steps to execute : '
                                              'L(K) for loads only (and KPI);R(K) for renewables (and KPI) only; '
                                              'LRT (K) for load, renewable and thermic generation (and KPI); '
                                              'LRDT(TK) for load, renewable, loss (dissipation) generation (and thermic and KPI)')
@click.option('--input-folder',
              default=os.path.join(pathlib.Path(__file__).parent.absolute(),
                                   cst.DEFAULT_INPUT_FOLDER_NAME),
              help='Directory to read input files from.')
@click.option('--output-folder',
              default=os.path.join(os.path.normpath(os.getcwd()),
                                   cst.DEFAULT_OUTPUT_FOLDER_NAME),
              help='Directory to store output files.')
@click.option('--seed-for-loads', default=None, help='Input seed to ensure reproducibility of loads generation')
@click.option('--seed-for-res', default=None, help='Input seed to ensure reproducibility of renewables generation')
@click.option('--seed-for-dispatch', default=None, help='Input seed to ensure reproducibility of dispatch')
@click.option('--ignore-warnings', is_flag=True,
              help='Ignore the warnings related to the existence of data files '
                   'in the chosen output directory.')
@click.option('--scenario_name', default='', help='subname to add to the generated scenario output folder, as Scenario_subname_i')
@click.option('--nb_core', default=1, help='number of cores to parallelize the number of scenarios')
def generate_mp(case, start_date, weeks, by_n_weeks, n_scenarios, mode,
             input_folder, output_folder, scenario_name,
             seed_for_loads, seed_for_res, seed_for_dispatch, nb_core, ignore_warnings):
    prng = default_rng()
    generate_mp_core(prng, case, start_date, weeks, by_n_weeks, n_scenarios, mode,
                     input_folder, output_folder, scenario_name,
                     seed_for_loads, seed_for_res, seed_for_dispatch, nb_core, ignore_warnings)


def generate_mp_core(prng, case, start_date, weeks, by_n_weeks, n_scenarios, mode,
             input_folder, output_folder, scenario_name,
             seed_for_loads, seed_for_res, seed_for_dispatch, nb_core, ignore_warnings):

    start_time = time.time()
    print(case)
    # create folders

    # get scenario name ids
    scenario_base_name = cst.SCENARIO_FOLDER_BASE_NAME
    if scenario_name:
        scenario_base_name = '_'.join([scenario_base_name, str(scenario_name)])

    scen_names = gu.folder_name_pattern(scenario_base_name, n_scenarios)

    generation_output_folder, kpi_output_folder = create_directory_tree(
        case, start_date, output_folder, scenario_base_name, n_scenarios, mode,
        warn_user=not ignore_warnings)

    ###if only 'K' mode, the genreation output folder can already be the output folder, check that. but inform that it is not the conventional one
    if mode=='K':
        print("as the time series as already been generated, checking if they are located in "+generation_output_folder+" or in root folder "
              +output_folder)
        if os.path.exists(generation_output_folder) and len(os.listdir(generation_output_folder))!=0:
            print("the data time series folder is identified as "+generation_output_folder)
        else:
            generation_output_folder=output_folder
            kpi_output_folder = os.path.join(output_folder, cst.KPI_FOLDER_NAME)
    ##########


    # seeds
    default_seed = generate_default_seed(prng)
    seed_for_loads = parse_seed_arg(seed_for_loads, '--seed-for-loads',
                                    default_seed)
    seed_for_res = parse_seed_arg(seed_for_res, '--seed-for-res',
                                  default_seed)
    seed_for_dispatch = parse_seed_arg(seed_for_dispatch, '--seed-for-dispatch',
                                       default_seed)

    initial_seeds = dict(
        loads=seed_for_loads,
        renewables=seed_for_res,
        dispatch=seed_for_dispatch
    )

    print('initial_seeds')
    print(initial_seeds)
    if mode!='K':#if data is generated
        dump_seeds(generation_output_folder, initial_seeds, scenario_name)

    if n_scenarios >= 2:
        seeds_for_loads, seeds_for_res, seeds_for_disp = gu.generate_seeds(
            prng, n_scenarios, seed_for_loads, seed_for_res, seed_for_dispatch
        )
    else:
        # in case uou want to reproduce a specific scenario already generated
        # with the seeds to consider
        seeds_for_loads = [seed_for_loads]
        seeds_for_res = [seed_for_res]
        seeds_for_disp = [seed_for_dispatch]
    if nb_core==1:
        for i in range(n_scenarios):
            generate_per_scenario(case, start_date, weeks, by_n_weeks, mode, input_folder,
            kpi_output_folder, generation_output_folder, scen_names,
            seeds_for_loads, seeds_for_res, seeds_for_disp, ignore_warnings,i)
    else:
    # multi-processing
        pool = multiprocessing.Pool(nb_core)
        iterable = [i for i in range(n_scenarios)]
        multiprocessing_func = partial(
            generate_per_scenario,
            case, start_date, weeks, by_n_weeks, mode, input_folder,
            kpi_output_folder, generation_output_folder, scen_names,
            seeds_for_loads, seeds_for_res, seeds_for_disp, ignore_warnings)

        pool.map(multiprocessing_func, iterable)
        pool.close()
        print('multiprocessing done')
        print('Time taken = {} seconds'.format(time.time() - start_time))
        print('removing temporary folders if exist:')
    rm_temporary_folders(input_folder,output_folder, case)

def rm_temporary_folders(input_folder,output_folder, case):
    grid2op_tempo = os.path.join(input_folder, cst.GENERATION_FOLDER_NAME, case, 'chronics')
    if os.path.exists(grid2op_tempo) and grid2op_tempo!=output_folder:
        shutil.rmtree(grid2op_tempo)
        print("--"+str(grid2op_tempo)+" deleted")

def generate_per_scenario(case, start_date, weeks, by_n_weeks, mode,
             input_folder, kpi_output_folder, generation_output_folder, scen_names,
             seeds_for_loads, seeds_for_res, seeds_for_dispatch, ignore_warnings, scenario_id):
    
    n_scenarios_sub_p = 1  # one scenario to compute per process``
    scenario_name = scen_names(scenario_id)

    # get scenario seeds
    seed_for_loads = seeds_for_loads[scenario_id]
    seed_for_res = seeds_for_res[scenario_id]
    seed_for_dispatch = seeds_for_dispatch[scenario_id]
    
    scenario_seeds = dict(
        loads=seed_for_loads,
        renewables=seed_for_res,
        dispatch=seed_for_dispatch
    )
    print('seeds for scenario: '+scenario_name)
    print(scenario_seeds)
    
    # dump scenario seeds
    # noScenarioDirectoryHere=''
    # generation_output_folder, kpi_output_folder = create_directory_tree(
    #    case, start_date, output_folder, noScenarioDirectoryHere,
    #    n_scenarios_sub_p, mode, warn_user=not ignore_warnings)

    scenario_path = os.path.join(generation_output_folder, scenario_name)
    print('scenario_path: '+scenario_path)

    if mode != 'K':  # if data is generated
        dump_seeds(scenario_path, scenario_seeds)

    # go to generate chronics
    generate_inner(
        case, start_date, weeks, by_n_weeks, n_scenarios_sub_p, mode,
        input_folder, kpi_output_folder, generation_output_folder,
        scen_names, seed_for_loads, seed_for_res, seed_for_dispatch, scenario_id)
    

def generate_inner(case, start_date, weeks, by_n_weeks, n_scenarios, mode,
                   input_folder, kpi_output_folder, generation_output_folder,
                   scen_names, seed_for_loads, seed_for_res,
                   seed_for_dispatch, scenario_id=None):

    ut.check_scenario(n_scenarios, scenario_id)
    time_parameters = gu.time_parameters(weeks, start_date)

    generation_input_folder = os.path.join(
        input_folder, cst.GENERATION_FOLDER_NAME
    )
    kpi_input_folder = os.path.join(
        input_folder, cst.KPI_FOLDER_NAME
    )

    year = time_parameters['year']


    # Chronic generation
    if 'L' in mode or 'R' in mode:
        generator = GeneratorBackend()
        params, _, _ = gen.main(generator,case, n_scenarios, generation_input_folder,
                                 generation_output_folder, scen_names, time_parameters,
                                 mode, scenario_id, seed_for_loads, seed_for_res,
                                 seed_for_dispatch)
        scenario_name = scen_names(scenario_id)
        if by_n_weeks is not None and 'T' in mode:
            output_processor_to_chunks(
                generation_output_folder, scenario_name, by_n_weeks,
                n_scenarios, weeks)
            write_start_dates_for_chunks(
                generation_output_folder, scenario_name, weeks, by_n_weeks,
                n_scenarios, start_date, int(params['dt']))

    # KPI formatting and computing
    if 'K' in mode :
        #get params, prods_charac and loads_charac from Generator config
        generator = GeneratorBackend()
        config_manager_dict = generator.get_config_managers(generation_input_folder, case, generation_output_folder, mode)
        params_dict, prods_charac, loads_charac = generator.get_params_charact(time_parameters, config_manager_dict)

        print("WARNING: make sure that your generated data is in folder: "+generation_output_folder)

        #compute KPIs
        kpis.main(kpi_input_folder, generation_output_folder, scen_names,
                  kpi_output_folder, year, case, n_scenarios,
                  params_dict['G'], loads_charac, prods_charac, scenario_id)


def create_directory_tree(case, start_date, output_directory, scenario_name,
                          n_scenarios, mode, warn_user=True):
    gen_path_to_create = os.path.join(
        output_directory, cst.GENERATION_FOLDER_NAME, case, start_date)
    if warn_user and os.path.isdir(gen_path_to_create):
        gu.warn_if_output_folder_not_empty(gen_path_to_create)
    if(mode!='K'):#if time series have not already been generated
        os.makedirs(gen_path_to_create, exist_ok=True)

    kpi_path_to_create = None
    if 'K' in mode:
        kpi_path_to_create = os.path.join(
            output_directory, cst.KPI_FOLDER_NAME, case, start_date)
        if warn_user and os.path.isdir(kpi_path_to_create):
            gu.warn_if_output_folder_not_empty(kpi_path_to_create)
        os.makedirs(kpi_path_to_create, exist_ok=True)

    scen_name_generator = gu.folder_name_pattern(
        scenario_name, n_scenarios)
    for i in range(n_scenarios):
        s_name = scen_name_generator(i)
        scenario_path_to_create = os.path.join(gen_path_to_create, s_name)
        if (mode != 'K'):  # if time series have not already been generated
            os.makedirs(scenario_path_to_create, exist_ok=True)

            if 'K' in mode:
                scenario_kpi_path_to_create = os.path.join(
                    kpi_path_to_create, s_name, cst.KPI_IMAGES_FOLDER_NAME
                )
                os.makedirs(scenario_kpi_path_to_create, exist_ok=True)

    return gen_path_to_create, kpi_path_to_create


if __name__ == "__main__":
    # # # Default arguments for dev mode
    # case = 'case118_l2rpn_neurips_1x' #'case118_l2rpn_wcci' #'case118_l2rpn_neurips_1x' #'case118_l2rpn_neurips_1x_GAN'
    # start_date = '2012-01-01'
    # weeks = 4
    # by_n_weeks = 4
    # n_scenarios = 2
    # mode = 'LRTK'
    # input_folder = 'input_data' #'getting_started/example/input' #'input_data'
    # output_folder =  'output' #'getting_started/example/output' #'output' #'output_gan'
    # scenario_name = "" #"year_wind_solar" "january_wind_solar_dispatch"
    # seed_for_loads = 912206665
    # seed_for_res = 912206665
    # seed_for_dispatch = 912206665
    # nb_core = 1
    # ignore_warnings = True
    #
    # # Run main function (only works with absolute path)
    # cwd = os.getcwd()
    # input_folder = os.path.join(cwd,input_folder)
    # output_folder = os.path.join(cwd, output_folder)
    # generate_mp_core(case, start_date, weeks, by_n_weeks, n_scenarios, mode,
    #                  input_folder, output_folder, scenario_name,
    #                  seed_for_loads, seed_for_res, seed_for_dispatch, nb_core, ignore_warnings)
    #####################################
    generate_mp()




