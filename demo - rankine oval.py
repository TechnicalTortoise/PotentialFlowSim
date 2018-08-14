import potentialflow as pot
import numpy as np

fs = pot.create_flowspace(nx=200, ny=100)

fs.add_uniform_flow(2, 0)
fs.add_source(-10, 2, 0)
fs.add_source(10, -2, 0)

fs.evaluate_sfvp()
fs.evaluate_stag_points(target_n=6)

fs.plot_vp(True, 100)
