# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import unittest
import shutil
import numpy as np
from numpy.random import default_rng
import pandas as pd
import pathlib
import datetime as dt

from chronix2grid import main
from chronix2grid import constants as cst
from chronix2grid import default_backend as def_bk
import chronix2grid.generation.generation_utils as gu
from chronix2grid.config import ResConfigManager
from chronix2grid.generation.renewable.RenewableBackend import RenewableBackend

def_bk.RENEWABLE_GENERATION_CONFIG = ResConfigManager
def_bk.RENEWABLE_GENERATION_BACKEND = RenewableBackend


class TestLoadProdCoherence(unittest.TestCase):
    def setUp(self):
        prng = default_rng()
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'input')
        self.output_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'output')
        self.case = 'case118_l2rpn_wcci'
        self.start_date = '2012-01-01'
        self.year = 2012
        self.seed_loads = 1
        self.seed_res = 1
        self.seed_dispatch = 1
        self.n_scenarios = 1
        self.scenario_names = gu.folder_name_pattern(
            cst.SCENARIO_FOLDER_BASE_NAME, self.n_scenarios
        )
        generation_output_folder, kpi_output_folder = main.create_directory_tree(
            self.case, self.start_date, self.output_folder, cst.SCENARIO_FOLDER_BASE_NAME,
            self.n_scenarios, 'LRK', warn_user=False)
        self.generation_output_folder = generation_output_folder
        self.kpi_output_folder = kpi_output_folder
        seeds_for_loads, seeds_for_res, seeds_for_disp = gu.generate_seeds(prng,
            self.n_scenarios, self.seed_loads, self.seed_res, self.seed_dispatch
        )
        self.seeds_for_loads = seeds_for_loads
        self.seeds_for_res = seeds_for_res
        self.seeds_for_disp = seeds_for_disp
        self.ignore_warnings = True
        self.weeks = 52
        self.nb_core = 1
        self.by_n_weeks = 52


    def tearDown(self) -> None:
        shutil.rmtree(self.output_folder, ignore_errors=False, onerror=None)

    def test_load_coherence_year(self):

        main.generate_per_scenario(
            case=self.case, start_date='2012-01-01', weeks=self.weeks,
            by_n_weeks=self.by_n_weeks, mode='L',
            input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

        generation_input_folder = os.path.join(
            self.input_folder, cst.GENERATION_FOLDER_NAME
        )

        general_config_manager = def_bk.GENERAL_CONFIG(
            name="Global Generation",
            root_directory=generation_input_folder,
            input_directories=dict(case=self.case),
            required_input_files=dict(case=['params.json']),
            output_directory=self.generation_output_folder
        )

        general_config_manager.validate_configuration()
        params,prods_charac, loads_charac = general_config_manager.read_configuration()

        scenario_path = os.path.join(
            self.generation_output_folder,
            f'{cst.SCENARIO_FOLDER_BASE_NAME}_0'
        )

        loads = pd.read_csv(
            os.path.join(scenario_path, 'load_p.csv.bz2'),
            sep=';')

        # Get datetime index
        start = pd.to_datetime(self.start_date)
        end = start + dt.timedelta(days=7 * int(self.weeks)) - 2 * dt.timedelta(minutes=int(params['dt']))
        datetime_index = pd.date_range(
            start=start,
            end=end,
            freq=str(params['dt']) + 'min')
        wdays = datetime_index.weekday
        loads['Weekday'] = wdays

        # Compute mean load by day of week
        loads = loads.groupby('Weekday').mean()
        loads = loads.mean(axis = 1)
        avg_workingdays = loads.loc[loads.index.isin([0,1,2,3,4])].mean() # Loads at working days
        avg_weekends = loads.loc[loads.index.isin([5,6])].mean() # Loads at working days

        # Make sure production is higher than consumption
        self.assertTrue(avg_workingdays > avg_weekends)
