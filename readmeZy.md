
Usage:

mpirun -np 4 python3 -m baselines.ddpg.main ( I use this )

OR

python3 -m baselines.ddpg.main

GPU:GeForce GTX1060 6GB

Memo: 2017/11/1 I did major fix on ddpg/training.py but only slight changes on other files. So easy to track diffs.

11/8 Started my own eiviron Reacher3d-v0. Have to add reacher3d.py and assets/reacher3d.xml to gym. Also add environ to gym/envs/__init.py__ (max_tstep is 80 on 11/8, reward thresh is 0.0) and add import line to gym/envs/mujoco/__init.py__. This way of making eivironment is not recommended though.


