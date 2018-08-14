import potentialflow as pot

fs = pot.create_flowspace(nx=200, ny=100)

fs.add_uniform_flow(2, 0)
fs.add_source(5, 0, 0)

fs.evaluate_sfvp()
fs.evaluate_stag_points(target_n=2)

fs.plot_sf(False, 100, other_streamlines=[-fs.stag_sf])
