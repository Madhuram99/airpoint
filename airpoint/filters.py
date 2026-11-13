import math


class OneEuroFilter:
    """Casiez, Roussel, Vogel (CHI 2012). fc = fc_min + beta * |speed|."""

    def __init__(self, fc_min=0.80, beta=0.010, d_cutoff=1.0):
        self.fc_min, self.beta, self.d_cutoff = fc_min, beta, d_cutoff
        self.x_prev, self.dx_prev, self.t_prev = None, 0.0, None

    @staticmethod
    def _alpha(fc, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * fc * dt))

    def filter(self, x, t, custom_fc_min=None):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-5)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        edx = a_d * dx + (1.0 - a_d) * self.dx_prev
        fc = (self.fc_min if custom_fc_min is None else custom_fc_min) + self.beta * abs(edx)
        alpha = self._alpha(fc, dt)
        x_out = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_out, edx, t
        return x_out
