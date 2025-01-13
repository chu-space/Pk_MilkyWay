from helpers.SimulationAnalysis import SimulationAnalysis

sim = SimulationAnalysis(trees_dir='/central/groups/carnegie_poc/enadler/ncdm_resims/Halo004/wdm_3/output_wdm_3/rockstar/trees')
host_mb = sim.load_main_branch(1889038)

print(host_mb)

