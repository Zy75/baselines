
Usage:

mpirun -np 4 python3 -m baselines.ddpg.main ( I use this )

OR

python3 -m baselines.ddpg.main

GPU:GeForce GTX1060 6GB

Memo: 2017/11/1 I did major fix on ddpg/training.py but only slight changes on other files. So easy to track diffs.

Changed Reacher max tsteps to 80 in gym env __init__ file.

Replace reacher.py of gym mujoco env with this repo's.

11/3 
python3 test.py to visualize.

best model in completed_model/

max tsteps to 120.

11/5 Trying PPO Hopper
