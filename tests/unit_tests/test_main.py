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
import pandas as pd
import pathlib

from numpy.random import default_rng

from chronix2grid import main
from chronix2grid import constants as cst
from chronix2grid import default_backend as def_bk
import chronix2grid.generation.generation_utils as gu

from chronix2grid.config import ResConfigManager
from chronix2grid.generation.renewable.RenewableBackend import RenewableBackend

def_bk.RENEWABLE_GENERATION_CONFIG = ResConfigManager
def_bk.RENEWABLE_GENERATION_BACKEND = RenewableBackend


class TestMain(unittest.TestCase):
    def setUp(self):
        prng = default_rng()
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'input')
        self.output_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),'data',
            'output')
        self.case = 'case118_l2rpn_wcci'
        self.start_date = '2012-01-01'
        self.year = 2012
        self.seed_loads = 1
        self.seed_res = 1
        self.seed_dispatch = 1
        self.n_scenarios = 2
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
        self.mode='LR'

    def tearDown(self) -> None:
        if self.mode!='K':
            shutil.rmtree(self.generation_output_folder, ignore_errors=False, onerror=None)


    # def test_size_chunks(self):
    #     main.generate_inner(case=self.case, start_date=self.start_date, weeks=8,
    #                         by_n_weeks=4, n_scenarios=2, mode='LRTK',
    #                         root_folder=self.root_folder,
    #                         seed_for_loads=self.seed_loads, seed_for_res=self.seed_res,
    #                         seed_for_dispatch=self.seed_dispatch)
    #     dir_path = os.path.join(self.root_folder,
    #                         'generation', 'output', self.case, str(self.year),
    #                         'Scenario_0')
    #     prod_p = pd.read_csv(os.path.join(dir_path, 'prod_p.csv.bz2'))
    #
    #     prod_p_chunk_0 = pd.read_csv(os.path.join(dir_path, 'chunk_0', 'prod_p.csv.bz2'))
    #     prod_p_chunk_1 = pd.read_csv(os.path.join(dir_path, 'chunk_1', 'prod_p.csv.bz2'))
    #
    #     self.assertTrue(np.array_equal(
    #         prod_p.values,
    #         pd.concat([prod_p_chunk_0, prod_p_chunk_1])
    #     ))

    def test_l(self):
        self.mode = 'L'
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode=self.mode, input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

    def test_r(self):
        self.mode = 'R'
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode=self.mode, input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=1)

    def test_lr(self):
        self.mode='LR'
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode=self.mode, input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

    def test_lrk(self):
        self.mode='LRK'
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode=self.mode, input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=1)

    def test_lrtk(self):
        self.mode='LRTK'
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode=self.mode, input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

    def test_lrdk(self):
        self.mode='LRDK'
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode=self.mode, input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

    def test_lrdtk(self):
        self.mode='LRDTK'
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode=self.mode, input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

    def test_k(self):
        self.mode='K'
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode=self.mode, input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

    def test_k_grid2op_env(self):
        self.mode='K'
        self.case = 'case118_l2rpn_wcci_2022'
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'input')
        self.generation_output_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(), 'data',
            'input', 'generation', self.case, 'chronics')
        self.kpi_output_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(), 'data',
            'output', 'kpi', self.case)
        os.makedirs(self.kpi_output_folder,exist_ok=True)
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode=self.mode, input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

