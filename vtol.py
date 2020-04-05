import numpy as np

class VTOLEnv(object):

    def __init__(self):
        self._state = np.array([0., 0., 0., 0., 0., 0.])
        self.dt = 0.01
        self.viewer = None
    @property
    def state(self):
        return self._state.copy()
    def reset(self):

    def _f(s, a):
        x, y, th, xt, yt, tht = s
        u1, u2 = a
        xtt = - u1 * np.sin(th) + self.__eps * u2 * np.cos(th)
        ytt = u1 * np.cos(th) + self.__eps * u2 * np.sin(th) - 9.81
        thtt = u2
        return np.array([xt, yt, tht, xtt, ytt, thtt])

    def step(self, a):
        dx = self._f(self._state, a) * self.dt
        self._state = self._state + dx
        return self.state
    def render(self):

        screen_width = 600
        screen_height = 400

        world_width = 6
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2

            body = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.body_trans = rendering.Transform()
            body.add_attr(self.body_trans)
            self.viewer.add_geom(body)

        if self.state is None: return None

        x, _ = np.split(self.state, 1)

        self.body_trans.set_translation(x[0], x[1])
        self.body_trans.set_rotation(x[2])
