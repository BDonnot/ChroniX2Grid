# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

from abc import ABC, abstractmethod
import json
import os

import numpy as np
import pandas as pd


class ConfigManager(ABC):
    """
    Class that validates a configuration setting by checking that the input files are correctly provided.
    It also provided a static method for specific reading of different input files

    Attributes
    ----------
    name: ``str``
        Name of config manager
    root_directory: ``str``
    input_directories: ``str`` or ``dict``
    output_directory: ``str``
    required_input_files: ``list`` or ``None``
        dict with compulsory files to be checked
    """
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        """
        Self initialization

        Parameters
        ----------
        name: ``str``
        root_directory: ``str``
        input_directories: ``str`` or ``dict``
        output_directory: ``str``
        required_input_files: ``str``
        required_input_files: ``dict`` or ``None``
        """
        self.name = name
        self.root_directory = root_directory
        self.input_directories = input_directories
        self.output_directory = output_directory
        self.required_input_files = required_input_files if required_input_files is not None else []

    def is_single_input_dir(self):
        if isinstance(self.input_directories, str):
            return True
        if isinstance(self.input_directories, dict):
            return False
        raise RuntimeError("input_directories must be either a string or a dictionnary")

    def validate_input(self):
        """
        Validate that the input folder exists and contains expected files. Returns an error message with :func:`ConfigManager.error_message`
        if not the case
        """
        if self.is_single_input_dir():
            directory_to_check = os.path.join(self.root_directory, self.input_directories)
            try:
                files_to_check = os.listdir(directory_to_check)
                for config_file in self.required_input_files:
                    if config_file not in files_to_check:
                        return False
            except FileNotFoundError:
                raise FileNotFoundError(self.error_message())
        else:
            for input_name, input_path in self.input_directories.items():
                input_directory_abs_path = os.path.join(self.root_directory, input_path)
                try:
                    files_to_check = os.listdir(input_directory_abs_path)
                    for config_file in self.required_input_files[input_name]:
                        if config_file not in files_to_check:
                            return False
                except FileNotFoundError:
                    raise FileNotFoundError(self.error_message())
        return True

    def validate_output(self):
        """
        Check if the output path exists from root directory
        """
        return os.path.exists(os.path.join(self.root_directory, self.output_directory))

    def error_message(self):
        """
        Returns an error message if inputs and outputs are not valid

        """
        output_directory_abs_path = os.path.join(
            self.root_directory, self.output_directory)
        error_message_header = (
            f"\n{self.name} process requires the following configuration "
            f"(directories and files must exist): \n"
            f"  - Root directory is {self.root_directory}\n"
            f"  - Output directory is {output_directory_abs_path}\n"
        )
        if self.is_single_input_dir():
            input_directory_abs_path = os.path.join(
                self.root_directory, self.input_directories)
            formatted_required_input_files = ""
            for filename in sorted(self.required_input_files):
                formatted_required_input_files += "\n      - " + filename

            error_msg_body = (
                    f"  - Input directory is {input_directory_abs_path}\n"
                    f"    - Expected input files are:" + formatted_required_input_files
            )
        else:
            formatted_required_directories = ""
            for input_name, input_path in self.input_directories.items():
                input_directory_abs_path = os.path.join(self.root_directory, input_path)
                formatted_required_directories += (
                        "\n    - " + input_directory_abs_path +
                        "\n      - with expected input files:"
                )
                for required_file in self.required_input_files[input_name]:
                    formatted_required_directories += "\n        - " + required_file

            error_msg_body = (
                    f"  - Input directories are:" + formatted_required_directories
            )
        return error_message_header + error_msg_body

    def validate_configuration(self):
        if not self.validate_output() or not self.validate_input():
            raise FileNotFoundError(self.error_message())
        return True

    @abstractmethod
    def read_configuration(self):
        pass

class GeneralConfigManager(ConfigManager):
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        super(GeneralConfigManager, self).__init__(name, root_directory, input_directories,
                                                 output_directory, required_input_files)

    def read_configuration(self):
        """
        Reads parameters for the overall generation process

            * *dt* - temporal granularity of the process
            * *planned_std* - noise level for forecast chronics

        Returns
        -------
        params: ``dict``
            dictionary of parameters
        """
        params_file_path = os.path.join(
            self.root_directory,
            self.input_directories['case'], 'params.json')
        with open(params_file_path, 'r') as json1_file:
            json1_str = json1_file.read()
        params = json.loads(json1_str)
        for key, value in params.items():
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = pd.to_datetime(value, format='%Y-%m-%d')

        try:
            loads_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'loads_charac.csv'),
                sep=',')
            names = loads_charac['name']  # to generate error if separator is wrong

        except:
            loads_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'loads_charac.csv'),
                sep=';')
        try:
            prods_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'prods_charac.csv'),
                sep=',')
            names = prods_charac['name']  # to generate error if separator is wrong

        except:
            prods_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'prods_charac.csv'),
                sep=';')

        return params,prods_charac,loads_charac

class LoadsConfigManager(ConfigManager):
    """
    Reads parameters for :class:`chronix2grid.generation.consumption.ConsumptionGeneratorBackend`

        * *Lx* - x dimension of coarse grid for spatially correlated noise
        * *Ly* - y dimension of coarse grid for spatially correlated noise
        * *dx_corr* - x granularity of coarse grid for spatially correlated noise
        * *dy_corr* - y granularity of coarse grid for spatially correlated noise
        * *temperature_corr* - noise level for spatially correlated noise
        * *std_temperature_noise* - noise level for temporally autocorrelated noise

    Returns
    -------
    params_load: ``dict``
        dictionary of parameters
    """
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        super(LoadsConfigManager, self).__init__(name, root_directory, input_directories,
                                                 output_directory, required_input_files)

    def read_configuration(self):
        params_file_path = os.path.join(
            self.root_directory,
            self.input_directories['case'], 'params_load.json')
        with open(params_file_path, 'r') as json1_file:
            json1_str = json1_file.read()
        params = json.loads(json1_str)
        for key, value in params.items():
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = pd.to_datetime(value, format='%Y-%m-%d')
        # Nt_inter = int(params['T'] // params['dt'] + 1)
        try:
            loads_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'loads_charac.csv'),
                sep=',')
            names = loads_charac['name']  # to generate error if separator is wrong

        except:
            loads_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'loads_charac.csv'),
                sep=';')

        return params, loads_charac

    def read_specific(self):
        """
        Reads data frame with loads characteristics

        Returns
        -------
        loads_charac: :class:`pandas.DataFrame`
        """
        load_weekly_pattern = pd.read_csv(
            os.path.join(self.root_directory, self.input_directories['patterns'],
                         'load_weekly_pattern.csv'))
        return load_weekly_pattern

class ResConfigManager(ConfigManager):
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        super(ResConfigManager, self).__init__(name, root_directory, input_directories,
                                               output_directory, required_input_files)

    def read_configuration(self):
        """
        Reads parameters for :class:`chronix2grid.generation.renewable.RenewableBackend`

            * *Lx* - x dimension of coarse grid for spatially correlated noise
            * *Ly* - y dimension of coarse grid for spatially correlated noise
            * *dx_corr* - x granularity of coarse grid for spatially correlated noise
            * *dy_corr* - y granularity of coarse grid for spatially correlated noise
            * *solar_corr*, *short_wind_corr*, *medium_wind_corr*, *long_wind_corr* - noise levels for spatially correlated noises
            * *std_solar_noise*, *std_short_wind_noise*, *std_medium_wind_noise*, *std_long_wind_noise* - noise levels for temporally autocorrelated noises
            * *smoothdist* - independent noise level
            * *year_solar_pattern* - year of provided solar pattern

        Returns
        -------
        params_res: ``dict``
            dictionary of parameters
        """
        params_file_path = os.path.join(
            self.root_directory,
            self.input_directories['case'], 'params_res.json')
        with open(params_file_path, 'r') as params_json:
            params = json.load(params_json)
        for key, value in params.items():
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = pd.to_datetime(value, format='%Y-%m-%d')

        # Nt_inter = int(params['T'] // params['dt'] + 1)
        try:
            prods_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'prods_charac.csv'),
                sep=',')
            names = prods_charac['name']  # to generate error if separator is wrong

        except:
            prods_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'prods_charac.csv'),
                sep=';')

        return params, prods_charac

    def read_specific(self):
        """
        Reads data frame with generator characteristics

        Returns
        -------
        prods_charac: :class:`pandas.DataFrame`
        """
        solar_pattern = np.load(
            os.path.join(self.root_directory, self.input_directories['patterns'],
                         'solar_pattern.npy'))
        return solar_pattern


class DispatchConfigManager(ConfigManager):
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        super(DispatchConfigManager, self).__init__(name, root_directory, input_directories,
                                                 output_directory, required_input_files)

    def read_configuration(self):
        """
        Reads parameters for :class:`chronix2grid.generation.dispatch.DispatchBackend`

            * *step_opf_min* - time resolution of the OPF in minutes. It can be 5, 10, 15, 20, 30 or 60 and has to be superior or equal to dt (generation time resolution). In case it is strictly above, interpolation is done after dispatch resolution
            * *mode_opf* - frequency at which we wan't to solve the OPF
            * *dispatch_by_carrier* - if True, dispatch results will be returned for the whole carrier. If False, it will be returned by generator
            * *ramp_mode* is essentially designed for debug purpose: when your OPF diverges, you may want to relax some constraints to know the reasons why the problem is unfeasible or leads to divergence
                * If *hard*, all the ramp constraints will be taken into account.
                * If *medium*, thermal ramp-constraints are skipped
                * If *easy*, thermal and hydro ramp-constraints are skipped
                * If *none*, thermal, hydro and nuclear ramp-constraints are skipped
            * *reactive_comp* - Factor applied to consumption to compensate reactive part not modelled by linear opf
            * *pyomo* - whether pypsa should use pyomo or not (boolean)
            * *solver_name* - name of solver, that you should have installed in your environment and added in your environment variables.
            * *hydro_ramp_reduction_factor* - optional factor which will divide max ramp up and down to all hydro generators
            * *losses_pct**- if D mode is deactivate, losses are estimated as a percentage of load.

        Optional parameters can be set for grid2op simulation of loss as a final step.
        The production is updated on a slack generator and warnings or errors are returned if this update violates generator constraints

        * *slack_p_max_reduction* - before dispatch, reduce Pmax of slack generator temporary to anticipate loss
        * *slack_ramp_max_reduction* - before dispatch, reduce ramp max (up and down) of slack generator temporary to anticipate loss
        * *loss_grid2op_simulation* - if True, launches grid2Op simulation for loss
        * *idxSlack*, *nameSlack* - identifies slack generator that will be updated
        * *early_stopping_mode* if True returns errors if generator constraints are violated after updates. If False, only returns warnings
        * *agent_type* - Grid2op agent type ti use for simulation. Can be "reco" for RecoPowerLines or "do-nothing"

        .. warning::
            The dispatch optimization can rely on pypsa simulation. If it is the case you should ensure pypsa dependencies are installed

        .. note::
            If no *loss_grid2op_simulation* is provided, chronix2grid follows considering it is False

        Returns
        -------
        params_opf: ``dict``
            dictionary of parameters
        """

        # Basics
        self.validate_configuration()
        params_filepath = os.path.join(
            self.root_directory,
            self.input_directories['params'],
            'params_opf.json')
        with open(params_filepath, 'r') as opf_param_json:
            params_opf = json.load(opf_param_json)
        try:
            if params_opf['mode_opf'] == '':
                params_opf['mode_opf'] = None
        except KeyError:
            raise KeyError('The mode_opf field of params_opf.json is missing.')

        # Grid2op loss simulation (optional)
        bool = False
        try:
            bool = params_opf["loss_grid2op_simulation"]
        except:
            print('Warning: The loss_grid2op_simulation field of params_opf.json is missing. Continuing assuming it is False')
            params_opf["loss_grid2op_simulation"] = False
        if bool:
            oblig_keys = ["idxSlack","nameSlack","agent_type"]
            for key in oblig_keys:
                try:
                    params_opf[key]
                except KeyError:
                    raise KeyError('loss_grid2op_simulation is set to True, key '+str(key) + ' must be provided')

        # Hydro correction
        if "hydro_ramp_reduction_factor" not in list(params_opf.keys()):
            params_opf["hydro_ramp_reduction_factor"] = 1.
        else:
            params_opf["hydro_ramp_reduction_factor"] = float(params_opf["hydro_ramp_reduction_factor"])

        # Slack temporary correction
        for key in ["slack_p_max_reduction", "slack_ramp_max_reduction"]:
            if key not in list(params_opf.keys()):
                params_opf[key] = 0.
            else:
                params_opf[key] = float(params_opf[key])
        return params_opf


def read_all_configuration(files, root_directory, input_directories):
    params = {}
    for file in files:
        params_file_path = os.path.join(
            root_directory,
            input_directories['case'], file)
        with open(params_file_path, 'r') as params_json:
            params_to_add = json.load(params_json)
        for key, value in params_to_add.items():
            try:
                params_to_add[key] = float(value)
            except ValueError:
                params_to_add[key] = pd.to_datetime(value, format='%Y-%m-%d')
        params.update(params_to_add)
    return params

class LossConfigManager(ConfigManager):
    """
            Checks parameters for :class:`chronix2grid.generation.loss.LossBackend`
                * *loss_pattern* - name of loss pattern to read

            Returns
            -------
            params_loss: ``dict``
                dictionary of parameters
            """
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        super(LossConfigManager, self).__init__(name, root_directory, input_directories,
                                                 output_directory, required_input_files)

    def read_configuration(self):
        self.validate_configuration()
        params_filepath = os.path.join(
            self.root_directory,
            self.input_directories['params'],
            'params_loss.json')
        with open(params_filepath, 'r') as loss_param_json:
            params_loss = json.load(loss_param_json)
        try:
            if params_loss['loss_pattern'] == '':
                params_loss['loss_pattern'] = 'loss_pattern.csv'
        except KeyError:
            raise KeyError('The loss_pattern field of params_loss.json is missing.')
        return params_loss
