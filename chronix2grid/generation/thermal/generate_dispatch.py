import os
import shutil

import numpy as np

from . import dispatch_utils as disp
from .EDispatch_L2RPN2020 import run_economic_dispatch


def main(dispatcher, input_folder, output_folder, seed, params_opf):

    np.random.seed(seed)

    hydro_constraints = dispatcher.make_hydro_constraints_from_res_load_scenario()
    agg_load_without_renew = dispatcher.net_load(params_opf['losses_pct'],
                                                 name=dispatcher.loads.index[0])

    dispatch_results = dispatcher.run(
        agg_load_without_renew,
        params=params_opf,
        gen_constraints=hydro_constraints,
        ramp_mode=run_economic_dispatch.RampMode.none,
        by_carrier=params_opf['dispatch_by_carrier']
    )

    dispatcher.save_results(output_folder)

    files_to_move = ['load_p_forecasted.csv.bz2', 'load_q_forecasted.csv.bz2',
                     'load_q.csv.bz2', 'prod_v.csv.bz2']
    for file_to_move in files_to_move:
        shutil.copy(os.path.join(input_folder, file_to_move),
                    os.path.join(output_folder, file_to_move))

    return dispatch_results
