# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import copy
import os
import json
import unittest
import warnings
import tempfile
import shutil
import numpy as np
import pandas as pd
import datetime
from packaging import version
import grid2op
from grid2op.Chronics import ChangeNothing

from chronix2grid.getting_started.example.input.generation.patterns import ref_pattern_path


class TestGrid2opUtils(unittest.TestCase):
    def setUp(self) -> None:
        if version.parse(grid2op.__version__) < version.parse("1.9.8"):
            # a fix in grid2Op 1.9.8 : the "loads_charac.csv" was not
            # part of the data shipped with the package before
            self.skipTest(f"grid2op version too old {grid2op.__version__} < 1.9.8")
        return super().setUp()

    def test_not_too_high_value_forecasts(self):
        seed = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("l2rpn_idf_2023", test=True) as env:
                path = env.get_path_env()
        tmp_dir = tempfile.TemporaryDirectory()
        new_env_path = os.path.join(tmp_dir.name, "l2rpn_idf_2023")
        shutil.copytree(path, new_env_path)
        shutil.rmtree(os.path.join(new_env_path, "chronics"))
        # keep only the first data (not to generate everything)
        with open(os.path.join(new_env_path, "scenario_params.json"), "r") as f:
            scenario_params = json.load(f)
        scenario_params["all_dates"] = scenario_params["all_dates"][:1]
        with open(os.path.join(new_env_path, "scenario_params.json"), "w") as f:
            json.dump(fp=f, obj=scenario_params)
        env = grid2op.make(new_env_path,
                           chronics_class=ChangeNothing,
                           **grid2op.Opponent.get_kwargs_no_opponent())
        env.generate_data(load_weekly_pattern=None, nb_year=1, seed=seed, save_ref_curve=True)
        gen_p_for_orig = pd.read_csv(os.path.join(new_env_path,
                                                  "chronics",
                                                  "2035-01-01_0",
                                                  "prod_p_forecasted.csv.bz2"),
                                     sep=";")
        assert (gen_p_for_orig.iloc[:,2] <= type(env).gen_pmax[2]).all()
        tmp_dir.cleanup()
            
    def test_load_weekly_pattern(self):
        seed = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("l2rpn_wcci_2022", test=True) as env:
                path = env.get_path_env()
        tmp_dir = tempfile.TemporaryDirectory()
        new_env_path = os.path.join(tmp_dir.name, "l2rpn_wcci_2022")
        shutil.copytree(path, new_env_path)
        shutil.rmtree(os.path.join(new_env_path, "chronics"))
        # keep only the first data (not to generate everything)
        with open(os.path.join(new_env_path, "scenario_params.json"), "r") as f:
            scenario_params = json.load(f)
        scenario_params["all_dates"] = scenario_params["all_dates"][:1]
        with open(os.path.join(new_env_path, "scenario_params.json"), "w") as f:
            json.dump(fp=f, obj=scenario_params)
        env = grid2op.make(new_env_path,
                           chronics_class=ChangeNothing,
                           **grid2op.Opponent.get_kwargs_no_opponent())
        env.generate_data(load_weekly_pattern=None, nb_year=1, seed=seed, save_ref_curve=True)
        load_weekly_pattern = pd.read_csv(os.path.join(ref_pattern_path, "load_weekly_pattern.csv"), sep=",")
        load_ref_orig = np.load(os.path.join(new_env_path, "chronics", "2050-01-03_0", "load_ref.npy"))
        total_demand_orig = np.sum(load_ref_orig, axis=1)
        
        # change the load weekly pattern
        load_weekly_pattern2 = load_weekly_pattern[load_weekly_pattern["datetime"] >= "2018"]
        load_weekly_pattern3 = load_weekly_pattern[load_weekly_pattern["datetime"] < "2018"]
        load_weekly_pattern_new = pd.concat([load_weekly_pattern2, load_weekly_pattern3])
        load_weekly_pattern_new.reset_index(inplace=True)
        load_weekly_pattern_new = load_weekly_pattern_new[["datetime", "test"]]
        load_weekly_pattern_new["datetime"] = copy.deepcopy(load_weekly_pattern["datetime"])
        # delete original data
        shutil.rmtree(os.path.join(new_env_path, "chronics"))
        
        # start a new generation
        env.generate_data(load_weekly_pattern=load_weekly_pattern_new, nb_year=1, seed=seed, save_ref_curve=True)
        load_ref_new = np.load(os.path.join(new_env_path, "chronics", "2050-01-03_0", "load_ref.npy"))
        total_demand_new = np.sum(load_ref_new, axis=1)     
        
        # recompute ref case to make sure it works
        shutil.rmtree(os.path.join(new_env_path, "chronics"))
        env.generate_data(load_weekly_pattern=None, nb_year=1, seed=seed, save_ref_curve=True)
        load_ref_orig2 = np.load(os.path.join(new_env_path, "chronics", "2050-01-03_0", "load_ref.npy"))
        total_demand_orig2 = np.sum(load_ref_orig2, axis=1)
        
        # compate the ref curves and the load data     
        assert np.allclose(total_demand_orig, total_demand_orig2)   
        assert not np.allclose(total_demand_orig, total_demand_new)
        tmp_dir.cleanup()
         