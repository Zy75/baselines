
Usage:

mpirun -np 4 python3 -m baselines.ddpg.main

OR

python3 -m baselines.ddpg.main

GPU:GeForce GTX1060 6GB

memo:
print( self.sess.run([var for var in self.target_actor.vars if 'actor/dense_2/kernel:0' in var.name]) )
