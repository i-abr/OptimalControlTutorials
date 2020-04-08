import numpy as np

class VTOLEnv(object):

    def __init__(self):
        self._state = np.array([0., 0., 0., 0., 0., 0.])
        self.dt = 0.1
        self.eps = 0.5
        self.viewer = None
    @property
    def state(self):
        return self._state.copy()
    def reset(self):
        self._state = 0.* self._state
    def _f(self, s, a):
        x, y, th, xt, yt, tht = s
        u1, u2 = a
        xtt = - u1 * np.sin(th) + self.eps * u2 * np.cos(th)
        ytt = u1 * np.cos(th) + self.eps * u2 * np.sin(th) - 9.81
        thtt = u2
        return np.array([xt, yt, tht, xtt, ytt, thtt])

    def step(self, a):
        dx = self._f(self._state, a) * self.dt
        self._state = self._state + dx
        return self.state
    def render(self, mode='human'):

        screen_width = 600
        screen_height = 600

        world_width = 24
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        cartwidth = 20.0
        cartheight = 10.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # l,r,t,b = -cartwidth/2, cartwidth/3, cartheight/3, -cartheight/2
            p1, p2, p3, p4 = (-20, -10), (0, 5), (20,-10), (0,-5)

            # body = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            body = rendering.FilledPolygon([p1, p2, p3, p4])
            self.body_trans = rendering.Transform()
            body.add_attr(self.body_trans)
            self.viewer.add_geom(body)

        if self.state is None: return None

        x, _ = np.split(self.state, 2)
        _xt = x[0]*scale + screen_width/3.0
        _yt = x[1]*scale + screen_height/10.0
        self.body_trans.set_translation(_xt, _yt)
        self.body_trans.set_rotation(x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
