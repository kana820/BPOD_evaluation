# BPOD_evaluation

evaluation.py: get results on Endpoint Distance along with trajectory plots

usage: evaluation.py [-h] timestamp_path data_path gt_path

evaluaton_local.py: get results on local scale drift errors (the angular difference and the distance)

usage: evaluation_local.py [-h] timestamp_path data_path gt_path algo(photo/dsv)

Both scripts align trajectories at the origin.