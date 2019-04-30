import jax.numpy as np
from jax import jacfwd, grad
from jax.numpy import cos, sin, tan, tanh
import matplotlib.pyplot as plt

class SimpleVehicleModel(object):
    def __init__(self):
        self.num_actions = 2 ## v and steering
        self.num_states = 4 ## 4 parameters I think  
        self.l_r = 0.02
        self.l_f = 0.015
        self.steering_const = self.l_r / (self.l_r + self.l_f)
        

    def f(self, x, u): ### we are going to define this in continuous time 
        '''
            The states are : x(t), y(t), theta(t), v(t) 
            ctrls are : u[0] : steering [pi/2, pi/2], u[1]: acceleration [-1, 1] normalized 
        '''
        v = x[3]
        theta = x[2]
        tan_u = tan(u[0])
        beta = tanh(self.steering_const * tan_u)
        dxdt = np.array([
                                v * cos(theta + beta),## xdot
                                v * sin(theta + beta),## ydot
                                v * sin(beta)/self.l_r,## thetadot
                                u[1]## vdot 
        ])
        return dxdt

class Objective(object):
    def __init__(self):
        
        self.y_scale = 1.0

    def track_cost(self, x):
#         outer_bnd = tanh( 10.0 * (np.sqrt(x[0]**2 + (x[1]/self.y_scale)**2) - 1.0 ))
#         inner_bnd = tanh( 10.0 * (np.sqrt(x[0]**2 + (x[1]/self.y_scale)**2) - 0.5 ))
#         return outer_bnd - inner_bnd + 2.0
        outer_bnd = np.exp( -10.0 * (np.sqrt(x[0]**2 + (x[1]/self.y_scale)**2) - 2.0 )**2)
        inner_bnd = np.exp( -10.0 * (np.sqrt(x[0]**2 + (x[1]/self.y_scale)**2) - 1.0 )**2)
        
        return 10*(outer_bnd - inner_bnd)
    def plot_track(self):
        X, Y = np.meshgrid(np.linspace(-2,2), np.linspace(-2,2))
        xx = np.vstack((X.ravel(), Y.ravel())).T
        val = np.array([self.track_cost(x) for x in xx])
        val = val.reshape(X.shape)
        plt.contourf(X, Y, val)

    def l(self, x, u):
        v = x[3]
        return self.track_cost(x) + u[0]**2 + u[1]**2 + (v-2.0)**2

class SimpleVehicleEnv2(object):


    def __init__(self, time_step = 0.1): ## default the time
        self.time_step = time_step
        self.model = SimpleVehicleModel()
        self.num_actions = self.model.num_actions ## v and steering
        self.num_states = self.model.num_states ## ?
        self.state = np.array([0., -1.2, 0., 1.])
        self.objective = Objective() 
        self.dstep = grad(self.step)

    def reset(self):
        self.state = np.array([0., -1.2, 0., 1.])
    def true_step(self, u):

        self.state = self.state + self.model.f(self.state, u) * self.time_step
        l = self.objective.l(self.state, u)
        return l
    def step(self, u): ## this is like the dumbest euler step ever.. TODO: add damping
        #dl = self.objective.dl(inputs)
        #df = self.model.df(inputs)
        state = self.state + self.model.f(self.state, u) * self.time_step
        l = self.objective.l(state, u)
        return l

if __name__=='__main__':
    import matplotlib.pyplot as plt 
    env = SimpleVehicleEnv2()
    horizon = 40
    ctrls = [np.zeros(env.num_actions) + 0.0001 for _ in range(horizon)]
    X, Y = np.meshgrid(np.linspace(-2,2), np.linspace(-2,2) )
    outer = np.exp(-10. * (np.sqrt(X**2 + (Y)**2 ) - 2.0)**2)
    inner = np.exp(-10. * (np.sqrt(X**2 + (Y)**2 ) - 1.0)**2)
    prev_cost = np.inf
    threshold = 0.001
    step_size = 0.001
    for k in range(200):
        x_pos = []
        y_pos = []
        cost = 0.0
        env.reset()
        for i, u in enumerate(ctrls):
            du = env.dstep(u)
            l = env.true_step(u)
            cost += l
            #step_size = 0.001 *(0.95**k)#0.01# * (0.95 ** k)
            ctrls[i] = u - step_size * du 
            x_pos.append(env.state.copy()[0])
            y_pos.append(env.state.copy()[1])
        
        if np.abs(prev_cost - cost) < threshold:
            break
        if prev_cost < cost:
            step_size *= 0.95
        prev_cost = cost
        print('iter ', k, ' cost ', cost, ' step size ', step_size)
        if k % 10 == 0:
            plt.clf()
            plt.contourf(X, Y, 10*(outer-inner))
            plt.colorbar()
            plt.plot(x_pos, y_pos)
            plt.pause(0.001)
