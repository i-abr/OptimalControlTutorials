import jax.numpy as np
from jax import jacfwd, grad, jit
from jax.numpy import cos, sin, tan, tanh
import matplotlib.pyplot as plt

class SimpleVehicleModel(object):
    def __init__(self):
        self.num_actions = 2 ## v and steering
        self.num_states = 4 ## 4 parameters I think  
        self.l_r = 1.738
        self.l_f = 1.105
        self.steering_const = self.l_r / (self.l_r + self.l_f)
        
        self.df = jacfwd(self.f) ## this takes the total derivative 

    def f(self, inputs): ### we are going to define this in continuous time 
        '''
            The states are : x(t), y(t), theta(t), v(t) 
            ctrls are : u[0] : steering [pi/2, pi/2], u[1]: acceleration [-1, 1] normalized 
        '''
        x = inputs['x']
        u = inputs['u']
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
        self.dl = grad(self.l)
        self.target_vel = 2.0
        self.target_pos = lambda t: np.array([5.*t, 4.0*np.sin(2*np.pi*(5.*t/100.))])
        self.t = 0. 
    def step(self, t):
        self.t = t
    
    def track_cost(self, x):
        err = x[0:2] - self.target_pos(self.t)
        return np.dot(err, err)

    def l(self, inputs):
        u = inputs['u']
        x = inputs['x']
        v = x[3]
        return self.track_cost(inputs['x']) +  0.01*u[0]**2 + 0.001*u[1]**2

class SimpleVehicleEnv(object):


    def __init__(self, time_step = 0.01): ## default the time
        self.time_step = time_step
        self.model = SimpleVehicleModel()
        self.num_actions = self.model.num_actions ## v and steering
        self.num_states = self.model.num_states ## ?
        self.objective = Objective() 
        self.reset()
    def reset(self):
        self.t = 0.
        self.state = np.array([0., 0., np.pi/4, 2.])
        self.objective.step(self.t)
    def step(self, u): ## TODO: add damping
        inputs = {'x' : self.state, 'u' : u}
        l = self.objective.l(inputs)
        dl = self.objective.dl(inputs)
        df = self.model.df(inputs)
        self.t += self.time_step
        self.state = self.state + self.model.f({'x':self.state, 'u':u}) * self.time_step
        self.objective.step(self.t)
        return self.state.copy(), l, dl, df

#     def step(self, u): ## this is like the dumbest euler step ever.. TODO: add damping

#         self.state = self.state + self.model.f({'x':self.state, 'u':u}) * self.time_step
#         inputs = {'x' : self.state, 'u' : u}
#         l = self.objective.l(inputs)
#         dl = self.objective.dl(inputs)
#         df = self.model.df(inputs)
#         return self.state.copy(), l, dl, df


