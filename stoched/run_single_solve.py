from stoched.get_mpc_objects import get_mpc_object_fsp
import time

mpc = get_mpc_object_fsp(dt=6) 

start = time.time()
mpc.model.solve_fsp(mpc.model.As[0])
print('time to solve one step off {0}'.format(time.time()-start))

start = time.time()
mpc.model.solve_fsp(mpc.model.As[1])
print('time to solve one step on` {0}'.format(time.time()-start))

print(2**4)
