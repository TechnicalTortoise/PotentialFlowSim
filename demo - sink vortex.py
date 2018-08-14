import potentialflow as pot
import numpy as np

fs = pot.create_flowspace(nx=200, ny=100)

fs.add_source(-10, 0, 0)
fs.add_vortex(10, 0, 0)

fs.evaluate_sfvp()

fs.plot_sf(True, 30, show_stag_point=False, show_stag_line=False)
