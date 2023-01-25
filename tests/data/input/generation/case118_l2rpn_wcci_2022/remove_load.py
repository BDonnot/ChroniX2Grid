import pandapower
import pandapower as pp
import numpy as np
import pdb
grid = pp.from_json("grid.json")
idx_of_element_to_delete = np.isin(grid.load.name,
                                   ["load_7_6",
                                    "load_23_20",
                                    "load_71_56",
                                    "load_72_57",
                                    "load_90_72",
                                    "load_98_80",
                                    "load_112_93",
                                    "load_115_96"])
idx_of_element_to_delete = np.where(idx_of_element_to_delete)[0]
grid.load.drop(idx_of_element_to_delete, inplace=True)
grid.load.reset_index(inplace=True, drop=True)
pp.runpp(grid)
p_mw = 0.
max_e_mwh = 24.
for bus in [22, 41, 44, 58, 76, 95, 112]:
    pp.create_storage(grid, bus, p_mw, max_e_mwh)
pp.runpp(grid)
pp.to_json(grid, "grid.json")
