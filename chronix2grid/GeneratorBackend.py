# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import warnings

import pandas as pd

from chronix2grid import constants
from chronix2grid import utils
from chronix2grid.generation import generation_utils

from chronix2grid.generation.dispatch import EconomicDispatch


# MSG_PYPSA_DEPENDENCY = "Please install PypsaDispatchBackend dependency to launch chronix2grid with T mode. Chronix2grid stopped before dispatch computation. You should launch xithout letter T in mode"
MSG_NO_DISPATCH_BACKEND = "Dispatch Backend is set to None - no dispatch has been computed"

class GeneratorBackend:
    """
    Class that gathers the Backends of the different generation processes.
    It will allow to generate each step successively, thanks to its method :func:`GeneratorBackend.run`.
    It will load and check the parameters for each step thanks to instances of :class:`chronix2grid.config.ConfigManager`
    It will then do the proper generation thanks to its methods :func:`GeneratorBackend.do_l`, :func:`GeneratorBackend.do_r`,
    :func:`GeneratorBackend.do_d` and :func:`GeneratorBackend.do_t`. This methods rely on other specific backends.

    All the attributes are static variables passed via module :class:`chronix2grid.constants`.
    This is where you can choose all the type of objects that will be used by *Chronix2Grid*

    Attributes
    ----------
    general_config_manager: :class:`chronix2grid.config.ConfigManager`
        Class inheriting from ConfigManager that loads and checks the general parameters of the overall generation process, such as time resolution.
    load_config_manager: :class:`chronix2grid.config.ConfigManager`
        Class inheriting from ConfigManager that loads and checks the load generation parameters
    res_config_manager: :class:`chronix2grid.config.ConfigManager`
        Class inheriting from ConfigManager that loads and checks the renewable (solar and wind) production parameters
    loss_config_manager: :class:`chronix2grid.config.ConfigManager`
        Class inheriting from ConfigManager that loads and checks the power loss generation parameters
    dispatch_config_manager: :class:`chronix2grid.config.ConfigManager`
        Class inheriting from ConfigManager that loads and checks the dispatch parameters
    dispatcher_class: :class:`chronix2grid.generation.dispatch.EconomicDispatch.Dispatcher`
        Class inheriting from Dispatcher that provides an API to a DispatchBackend in order to compute a proper dispatch
    consumption_backend_class
        A class that embeds a load generation backend such as :class:`chronix2grid.generation.consumption.ConsumptionGeneratorBackend`
    renewable_backend_class
        A class that embeds a solar and wind generation backend such as :class:`chronix2grid.generation.renewable.RenewableBackend`
    loss_backend_class
        A class that embeds a power loss generation backend such as :class:`chronix2grid.generation.loss.LossBackend`
    dispatch_backend_class
        A class that embeds a dispatch backend such as :class:`chronix2grid.generation.dispatch.DispatchBackend`
    """
    def __init__(self):
        from chronix2grid import default_backend  # lazy import to avoid circular references
        self.general_config_manager = default_backend.GENERAL_CONFIG
        self.load_config_manager = default_backend.LOAD_GENERATION_CONFIG
        self.res_config_manager = default_backend.RENEWABLE_GENERATION_CONFIG
        self.loss_config_manager = default_backend.LOSS_GENERATION_CONFIG
        self.dispatch_config_manager = default_backend.DISPATCH_GENERATION_CONFIG

        self.dispatcher_class = default_backend.DISPATCHER

        self.consumption_backend_class = default_backend.LOAD_GENERATION_BACKEND
        self.dispatch_backend_class = default_backend.DISPATCH_GENERATION_BACKEND
        self.hydro_backend_class = default_backend.HYDRO_GENERATION_BACKEND
        self.renewable_backend_class = default_backend.RENEWABLE_GENERATION_BACKEND
        self.loss_backend_class = default_backend.LOSS_GENERATION_BACKEND

    # Call generation scripts n_scenario times with dedicated random seeds
    def run(self, case, n_scenarios, input_folder, output_folder, scen_names,
            time_params, mode='LRTK', scenario_id=None,
            seed_for_loads=None, seed_for_res=None, seed_for_disp=None):
        """
        Main function for chronics generation. It works with four steps: load generation (L), renewable generation (R, solar and wind), loss generation (D)
        and then dispatch computation (T) to get the whole energy mix. It writes the resulting chronics in the output_path in zipped csv format

        Parameters
        ----------
        case: ``str``
            name of case to study (must be a folder within input_folder)
        n_scenarios: ``int``
            number of desired scenarios to generate for the same timescale
        input_folder: ``str``
            path of folder containing inputs
        output_folder: ``str``
            path where outputs will be written (intermediate folder case/year/scenario will be used)
        scen_names: ``list``
            list of strings of scenario names to generate
        time_params: ``dict``
            dictionary with 'weeks', 'start_date' and 'year' information
        mode: ``str``
            options to launch certain parts of the generation process : L load R renewable T thermal D Loss
        scenario_id: ``int`` or ``None``
            Id of scenario
        seed_for_loads: ``int`` or ``None``
            seed for the load generation process
        seed_for_res: ``int`` or ``None``
            seed for the renewable generation process
        seed_for_disp: ``int`` or ``None``
            seed for the dispatch generation process

        Returns
        -------
        params: ``dict``
            general parameters
        loads_charac: :class:`pandas.DataFrame`
            characteristics of consumption nodes in the grid used in generation
        prods_charac: :class:`pandas.DataFrame`
            characteristics of production nodes in the grid used in generation
        """

        utils.check_scenario(n_scenarios, scenario_id)

        print('=====================================================================================================================================')
        print('============================================== CHRONICS GENERATION ==================================================================')
        print('=====================================================================================================================================')

        # in multiprocessing, n_scenarios=1 here
        if n_scenarios >= 2:
            seeds_for_loads, seeds_for_res, seeds_for_disp = generation_utils.generate_seeds(
                n_scenarios, seed_for_loads, seed_for_res, seed_for_disp
            )
        else:
            seeds_for_loads = [seed_for_loads]
            seeds_for_res = [seed_for_res]
            seeds_for_disp = [seed_for_disp]

        # dispatch_input_folder, dispatch_input_folder_case, dispatch_output_folder = gu.make_generation_input_output_directories(input_folder, case, year, output_folder)
        config_manager_dict=self.get_config_managers(input_folder, case, output_folder, mode)
        params_dict,prods_charac,loads_charac = self.get_params_charact(time_params,config_manager_dict)

        grid_folder = os.path.join(input_folder, case)

        loss = None

        ## Launch proper scenarios generation
        seeds_iterator = zip(seeds_for_loads, seeds_for_res, seeds_for_disp)

        for i, (seed_load, seed_res, seed_disp) in enumerate(seeds_iterator):

            if n_scenarios > 1:
                scenario_name = scen_names(i)
            else:
                scenario_name = scen_names(scenario_id)

            scenario_folder_path = os.path.join(output_folder, scenario_name)

            print("================ Generating " + scenario_name + " ================")
            if 'L' in mode:
                load, load_forecasted = self.do_l(scenario_folder_path, seed_load, params_dict['L'], loads_charac, config_manager_dict['L'])
                #params.update(params_load)
            if 'R' in mode:
                prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted = self.do_r(scenario_folder_path, seed_res, params_dict['R'],
                                                                                               prods_charac,
                                                                                               config_manager_dict['R'])
                #params.update(params_res)
            if 'D' in mode:

                self.do_d(input_folder, scenario_folder_path,
                                     load, prod_solar, prod_wind,
                                     params_dict['G'], config_manager_dict['D'])
            if 'T' in mode:
                if self.dispatch_backend_class is None:
                    warnings.warn(MSG_NO_DISPATCH_BACKEND, UserWarning)
                else:

                    dispatch_results = self.do_t(input_folder, scenario_name, load, prod_solar, prod_wind,
                                                 grid_folder, scenario_folder_path, seed_disp, params_dict['G'], params_dict['T'], loss)

            print('\n')
        return params_dict['G'], loads_charac, prods_charac

    def do_l(self, scenario_folder_path, seed_load, params, loads_charac, load_config_manager):
        """
        Generates load chronics thanks to the backend in ``self.consumption_backend_class``

        Parameters
        ----------
        scenario_folder_path: ``str``
        seed_load :``int``
        params: ``dict``
        loads_charac: :class:`pandas.DataFrame`
        load_config_manager: :class:`chronix2grid.config.ConfigManager`

        Returns
        -------
        loads: :class:`pandas.DataFrame`
            generated loads chronics
        prods_charac: :class:`pandas.DataFrame`
            generated forecasted loads chronics (currently loads chronics with gaussian noise)
        """
        generator_loads = self.consumption_backend_class(scenario_folder_path, seed_load, params, loads_charac, load_config_manager,
                                                         write_results=True)
        load, load_forecasted = generator_loads.run()
        return load, load_forecasted

    def do_r(self, scenario_folder_path, seed_res, params, prods_charac, res_config_manager):
        """
        Generates load chronics thanks to the backend in ``self.renewable_backend_class``

        Parameters
        ----------
        scenario_folder_path: ``str``
        seed_res: ``int``
        params: ``dict``
        prods_charac: :class:`pandas.DataFrame`
        res_config_manager: :class:`chronix2grid.config.ConfigManager`

        Returns
        -------
        prod_solar: :class:`pandas.DataFrame`
            generated solar chronics
        prod_solar_forecasted: :class:`pandas.DataFrame`
            generated forecasted solar chronics
        prod_wind: :class:`pandas.DataFrame`
            generated wind chronics
        prod_wind_forecasted: :class:`pandas.DataFrame`
            generated forecasted wind chronics
        """
        generator_enr = self.renewable_backend_class(scenario_folder_path, seed_res, params,
                                                     prods_charac,
                                                     res_config_manager, write_results=True)

        prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted = generator_enr.run()
        return prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted

    def do_d(self, input_folder, scenario_folder_path,
                                     load, prod_solar, prod_wind,
                                     params, loss_config_manager):
        """
        Generates load chronics thanks to the backend in ``self.renewable_backend_class``

        Parameters
        ----------
        input_folder: ``str``
        scenario_folder_path: ``str``
        load: :class: `pandas.DataFrame`
        prod_solar: :class: `pandas.DataFrame`
        prod_wind: :class: `pandas.DataFrame`
        params: ``dict``
        loss_config_manager: :class:`chronix2grid.config.ConfigManager`

        Returns
        -------
        loss: :class:`pandas.DataFrame`
            generated loss chronics
        """

        generator_loss = self.loss_backend_class(input_folder, scenario_folder_path,
                                     load, prod_solar, prod_wind,
                                     params, loss_config_manager)
        loss = generator_loss.run()
        return loss

    def do_t(self, input_folder, scenario_name, load, prod_solar, prod_wind, grid_folder,
             scenario_folder_path, seed_disp, params, params_opf, loss):
        """
        Computes production chronics based on a dispatch computation. It uses a dispatcher object as an environment for simulation and
        ``self.dispatch_backend_class`` for computation

        Parameters
        ----------
        dispatcher: :class:`chronix2grid.dispatch.EconomicDispatch.Dispatcher`
        scenario_name: ``str``
        load: :class:`pandas.DataFrame`
        prod_solar: :class:`pandas.DataFrame`
        prod_wind: :class:`pandas.DataFrame`
        grid_folder: ``str``
        scenario_folder_path: ``str``
        seed_disp: ``int``
        params: ``dict``
        params_opf: ``dict``
        loss: :class:`pandas.DataFrame`

        Returns
        -------
        dispatch_results: :class:`collection.namedtuple`
            contains resulting production chronics and terminal conditions from the optimization engine
        """

        prods = pd.concat([prod_solar, prod_wind], axis=1)
        res_names = dict(wind=prod_wind.columns, solar=prod_solar.columns)
        grid_path = os.path.join(grid_folder, constants.GRID_FILENAME)
        # grid_path = grid_folder
        dispatcher = EconomicDispatch.init_dispatcher_from_config_dataframe(grid_path, input_folder,self.dispatcher_class, params_opf)
        dispatcher.chronix_scenario = EconomicDispatch.ChroniXScenario(load, prods, res_names,
                                                                       scenario_name, loss)

        generator_dispatch = self.dispatch_backend_class(dispatcher, scenario_folder_path,
                                                 grid_folder, seed_disp, params, params_opf)
        dispatch_results = generator_dispatch.run()
        return dispatch_results

    def get_params_charact(self,time_params,config_manager_dict):
        """
          Function to load config_managers for each activated mode in 'LRTD' as well as the general config manager

          Parameters
          ----------
          config_manager_dict: ``dict``
              config managers for each activated mode 'L','R','T','D' and general config manager at 'G'
          time_params: ``dict``
              dictionary with 'weeks', 'start_date' and 'year' information
          Returns
          -------
          params_dict: ``dict``
              params for each activated mode 'L','R','T','D' and general params at 'G'
          """
        params_dict=dict()
        params,prods_charac,loads_charac = config_manager_dict['G'].read_configuration()
        #params = config_manager_dict['G'].read_configuration()

        params.update(time_params)
        params = generation_utils.updated_time_parameters_with_timestep(params, params['dt'])

        #loads_charac=None
        #prods_charac=None

        if('L' in config_manager_dict.keys()):
            params_load, loads_charac = config_manager_dict['L'].read_configuration()
            params_load.update(params)
            params_dict['L']=params_load

        if('R' in config_manager_dict.keys()):
            params_res, prods_charac = config_manager_dict['R'].read_configuration()
            params_res.update(params)
            params_dict['R'] = params_res

        if('T' in config_manager_dict.keys()):
            params_opf = config_manager_dict['T'].read_configuration()
            #should we update params with params_opf ?
            params_opf.update(params)
            params_dict['T'] = params_opf
        params_dict['G']=params

        return params_dict,prods_charac,loads_charac

    def get_config_managers(self,input_folder,case,output_folder,mode):
        """
          Function to load config_managers for each activated mode in 'LRTD' as well as the general config manager

          Parameters
          ----------
          case: ``str``
              name of case to study (must be a folder within input_folder)
          n_scenarios: ``int``
              number of desired scenarios to generate for the same timescale
          input_folder: ``str``
              path of folder containing inputs
          output_folder: ``str``
              path where outputs will be written (intermediate folder case/year/scenario will be used)
          mode: ``str``
              options to launch certain parts of the generation process : L load R renewable T thermal D Loss

          Returns
          -------
          config_manager_dict: ``dict``
              config managers for each activated mode 'L','R','T','D' and general config manager at 'G'
          """
        config_manager_dict=dict()

        general_config_manager = self.general_config_manager(
            name="Global Generation",
            root_directory=input_folder,
            input_directories=dict(case=case),
            required_input_files=dict(case=['loads_charac.csv','prods_charac.csv','params.json']),
            output_directory=output_folder
        )
        general_config_manager.validate_configuration()
        config_manager_dict['G']=general_config_manager
        if 'L' in mode:
            load_config_manager = self.load_config_manager(
                name="Loads Generation",
                root_directory=input_folder,
                input_directories=dict(case=case, patterns='patterns'),
                required_input_files=dict(case=['loads_charac.csv', 'params_load.json'],
                                          patterns=['load_weekly_pattern.csv']),
                output_directory=output_folder
            )
            load_config_manager.validate_configuration()
            config_manager_dict['L']=load_config_manager

        if 'R' in mode:
            res_config_manager = self.res_config_manager(
                name="Renewables Generation",
                root_directory=input_folder,
                input_directories=dict(case=case, patterns='patterns'),
                required_input_files=dict(case=['prods_charac.csv', 'params_res.json'],
                                          patterns=['solar_pattern.npy']),
                output_directory=output_folder
            )
            config_manager_dict['R']=res_config_manager

        if 'T' in mode:
            dispath_config_manager = self.dispatch_config_manager(
                name="Dispatch",
                root_directory=input_folder,
                output_directory=output_folder,
                input_directories=dict(params=case),
                required_input_files=dict(case=['prods_charac.csv', 'params_res.json'],params=['params_opf.json'])
            )
            dispath_config_manager.validate_configuration()
            config_manager_dict['T']=dispath_config_manager

        if 'D' in mode:
            loss_config_manager = self.loss_config_manager(
                name="Loss",
                root_directory=input_folder,
                output_directory=output_folder,
                input_directories=dict(params=case),
                required_input_files=dict(params=['params_loss.json'])
            )
            config_manager_dict['D']=loss_config_manager

        return config_manager_dict

