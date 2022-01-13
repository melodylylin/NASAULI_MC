import math
import os
import multiprocessing
import string

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy import signal
import scipy.integrate
import scipy.optimize
from scipy.spatial import ConvexHull
import control
import slycot
import pyhull
from pytope import Polytope
import picos


def matrix_exp(A, n=30):
    s = np.zeros((3, 3))
    A_i = np.eye(3)
    for i in range(n):
        s = s + A_i/math.factorial(i)
        A_i = A_i@A
    return s


def check_shape(a, shape):
    if np.shape(a) != shape:
        raise IOError(str(np.shape(a)) + '!=' + str(shape))


def wrap(x):
    return np.where(np.abs(x) >= np.pi, (x + np.pi) % (2 * np.pi) - np.pi, x)


class LieGroup:
    
    def __repr__(self):
        return repr(self.matrix)

    def __mul__(self, other):
        return NotImplementedError('')

    
class LieAlgebra:
    
    def __repr__(self):
        return repr(self.wedge)

    def __mul__(self, other):
        return NotImplementedError('')


class Vector:
    
    def __repr__(self):
        return repr(self.matrix)

    def __mul__(self, other):
        return NotImplementedError('')


class R2(Vector):
    
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    
    @property
    def matrix(self):
        return np.array([[self.x], [self.y]])

    def __neg__(self):
        return R2(x=-self.x, y=-self.y)
    
    def __add__(self, other):
        return R2(x=self.x + other.x, y=self.y + other.y)

    @classmethod
    def from_vector(cls, a):
        a = a.reshape(-1)
        return cls(x=a[0], y=a[1])

    
class so2(LieAlgebra):
    
    def __init__(self, theta):
        self.theta = np.reshape(wrap(theta), ())
    
    @property
    def wedge(self):
        return np.array([
            [0, -self.theta],
            [self.theta, 0]
        ])
    
    @property
    def vee(self):
        return np.array([self.theta])

    @property
    def exp(self):
        return SO2(theta=self.theta)
    

class SO2(LieGroup):
    
    def __init__(self, theta):
        self.theta = np.reshape(wrap(theta), ())
    
    @classmethod
    def one(cls):
        return cls(theta=0)

    @property
    def inv(self):
        return SO2(theta=-self.theta)

    @property
    def matrix(self):
        return np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]
        ])
    
    @property
    def params(self):
        return np.array([self.theta])

    @property
    def log(self):
        return so2(self.theta)
    
    @classmethod
    def from_matrix(cls, a):
        check_shape(a, (2, 2))
        return cls(theta=np.arctan2(a[1, 0], a[0, 0]))

    def __matmul__(self, other):
        if isinstance(other, R2):
            return R2.from_vector(self.matrix@other.matrix)
        elif isinstance(other, SO2):
            return SO2(theta=self.theta + other.theta)


class se2(LieAlgebra):
    
    def __init__(self, x: float, y: float, theta: float):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

    def __neg__(self):
        return se2(-self.x, -self.y, -self.theta)

    @property
    def wedge(self):
        return np.array([
            [0, -self.theta, self.x],
            [self.theta, 0, self.y],
            [0, 0, 0]
        ])
    

    def __add__(self, other):
        return se2(x=self.x + other.x, y=self.y + other.y, theta=self.theta + other.theta)
    
    def __sub__(self, other):
        return se2(x=self.x - other.x, y=self.y - other.y, theta=self.theta - other.theta)
    
    @property
    def vee(self):
        return np.array([self.x, self.y, self.theta])
    
    @classmethod
    def from_vector(cls, a):
        a = a.reshape((3, 1))
        return cls(x=a[0], y=a[1], theta=a[2])
    
    @classmethod
    def from_matrix(cls, a):
        check_shape(a, (3, 3))
        return cls(x=a[0, 2], y=a[1, 2], theta=a[1, 0])

    @property
    def ad_matrix(self):
        x, y, theta = self.x, self.y, self.theta
        return np.array([
            [0, -theta, y],
            [theta, 0, -x],
            [0, 0, 0]
        ])

    def __matmul__(self, other):
        return se2.from_vector(self.ad_matrix@other.vee)

    @property
    def exp(self):
        theta = self.theta
        with np.errstate(divide='ignore',invalid='ignore'):
            a = np.where(np.abs(theta) < 1e-3, 1 - theta**2/6 + theta**4/12, np.sin(theta)/theta)
            b = np.where(np.abs(theta) < 1e-3, theta/2 - theta**3/24 + theta**5/720, (1 - np.cos(theta))/theta)
        V = np.array([[a, -b], [b, a]])
        p = V@np.array([self.x, self.y])
        return SE2(x=p[0], y=p[1], theta=self.theta)

    def __rmul__(self, scalar):
        s = np.reshape(scalar, ())
        return se2(x=self.x*s, y=self.y*s, theta=self.theta*s)


class SE2(LieGroup):
    
    def __init__(self, x: float, y: float, theta: float):
        self.x = float(x)
        self.y = float(y)
        self.theta = wrap(float(theta))
    
    @classmethod
    def one(cls):
        return cls(x=0, y=0, theta=0)

    @property
    def params(self):
        return np.array([self.x, self.y, self.theta])
    
    @property
    def matrix(self):
        x, y, theta = self.x, self.y, self.theta
        return np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1]
        ])

    @property
    def R(self):
        return SO2(theta=self.theta)

    @property
    def p(self):
        return R2(x=self.x, y=self.y)
    
    @property
    def inv(self):
        p = -(self.R.inv@self.p)
        return SE2(x=p.x, y=p.y, theta=-self.theta)

    def __matmul__(self, other: 'SE2'):
        p = self.R@other.p + self.p
        return SE2(x=p.x, y=p.y, theta=self.theta + other.theta)

    @classmethod
    def from_matrix(cls, a: np.array):
        check_shape(a, (3, 3))
        return SE2(theta=np.arctan2(a[1, 0], a[0, 0]),
                   x=a[0, 2], y=a[1, 2])
    
    @classmethod
    def from_vector(cls, a):
        a = a.reshape((3, 1))
        return cls(x=a[0], y=a[1], theta=a[2])

    @property
    def Ad_matrix(self):
        x, y, theta = self.x, self.y, self.theta
        return np.array([
            [np.cos(theta), -np.sin(theta), y],
            [np.sin(theta), np.cos(theta), -x],
            [0, 0, 1]
        ])
    
    def Ad(self, v: 'se2'):
        v2 = self.Ad_matrix@v.vee
        return se2(x=v2[0], y=v2[1], theta=v2[2])

    @property
    def log(self):
        x, y, theta = self.x, self.y, self.theta
        with np.errstate(divide='ignore',invalid='ignore'):
            a = np.where(np.abs(theta) < 1e-3, 1 - theta**2/6 + theta**4/12, np.sin(theta)/theta)
            b = np.where(np.abs(theta) < 1e-3, theta/2 - theta**3/24 + theta**5/720, (1 - np.cos(theta))/theta)
        V_inv = np.array([
            [a, b],
            [-b, a]
        ])/(a**2 + b**2)
        p = V_inv@np.array([x, y])
        return se2(x=p[0], y=p[1], theta=theta)

def diff_correction(e: se2, n=100):
    # computes (1 - exp(-ad_x)/ad_x = sum k=0^infty (-1)^k/(k+1)! (ad_x)^k
    ad = e.ad_matrix
    ad_i = np.eye(3)
    s = np.zeros((3, 3))
    for k in range(n):
        s += ((-1)**k/math.factorial(k+1))*ad_i
        ad_i = ad_i @ ad
    return -np.linalg.inv(s)@((-e).exp.Ad_matrix)

def se2_diff_correction(e: se2):
    x = e.x
    y = e.y
    theta = e.theta
    with np.errstate(divide='ignore',invalid='ignore'):
        a = np.where(abs(theta) > 1e-3, -theta*np.sin(theta)/(2*(np.cos(theta) - 1)), 1 - theta**2/12 - theta**4/720)
        b = np.where(abs(theta) > 1e-3, -(theta*x*np.sin(theta) + (1 - np.cos(theta))*(theta*y - 2*x))/(2*theta*(1 - np.cos(theta))), -y/2 + theta*x/12 - theta**3*x/720)
        c = np.where(abs(theta) > 1e-3, -(theta*y*np.sin(theta) + (1 - np.cos(theta))*(-theta*x - 2*y))/(2*theta*(1 - np.cos(theta))), x/2 + theta*y/12 + theta**3*y/720)
    return -np.array([
        [a, theta/2, b],
        [-theta/2, a, c],
        [0, 0, 1]
    ])

def se2_diff_correction_inv(e: se2):
    x = e.x
    y = e.y
    theta = e.theta
    with np.errstate(divide='ignore',invalid='ignore'):
        a = np.where(abs(theta) > 1e-3, np.sin(theta)/theta, 1 - theta**2/6 + theta**4/120)
        b = np.where(abs(theta) > 1e-3, (1  - np.cos(theta))/theta, theta/2 - theta**3/24)
        c = np.where(abs(theta) > 1e-3, -(x*(theta*np.cos(theta) - theta + np.sin(theta) - np.sin(2*theta)/2) + y*(2*np.cos(theta) - np.cos(2*theta)/2 - 3/2))/(theta**2*(1 - np.cos(theta))), y/2 + theta*x/6 - theta**2*y/24 - theta**3*x/120 + theta**4*y/720)
        d = np.where(abs(theta) > 1e-3, -(x*(-2*np.cos(theta) + np.cos(2*theta)/2 + 3/2) + y*(theta*np.cos(theta) - theta + np.sin(theta) - np.sin(2*theta)/2))/(theta**2*(1 - np.cos(theta))), -x/2 + theta*y/6 + theta**2*x/24 - theta**3*y/120 - theta**4*x/720)
    return -np.array([
        [a, -b, c],
        [b, a, d],
        [0, 0, 1]
    ])

def solve_control_gain():
    ### A needs to be polytopic
    A = -se2(x=1, y=0, theta=0).ad_matrix
    B = np.array([[1, 0], [0, 0], [0, 1]])
    Q = 10*np.eye(3)  # penalize state
    R = 1*np.eye(2)  # penalize input
    K, _, _ = control.lqr(A, B, Q, R)
    K = -K  # rescale K, set negative feedback sign
    A0 = A + B@K
    return K, B, A0

def bode_analysis():
    K = solve_control_gain()[0]
    A = -se2(x=1, y=0, theta=0).ad_matrix
    B = np.array([[1, 0], [0, 0], [0, 1]])
    s1 = control.ss(A, B, np.eye(3), np.zeros((3, 2)))
    s2 = s1.feedback(-K)
    G = control.ss2tf(s2)
    mag, phase, omega = control.bode(
        [G[0, 0], G[1, 1], G[2, 1]],
        omega=np.logspace(-3, 3, 1000), 
        Hz=True, dB=True);
    max_frequencies = [ omega[i][np.argmax(mag[i])]/(2*np.pi) for i in range(len(mag)) ]
    print('max frequencies', max_frequencies)


def control_law(B, K, e):
    L = np.diag(np.array([1, 1, 1]))
    u = L@se2_diff_correction_inv(e)@B@K@e.vee # controller input
    return se2.from_vector(u)

def maxw(sol, x, w1_mag, w2_mag):
    U = se2_diff_correction(x)
    U1 = U[:, :2]
    U2 = U[:, 2]
    P = sol['P']

    w1 = U1.T@P@x.vee
    w1 *= -w1_mag/np.linalg.norm(w1)

    w2 = U2.T@P@x.vee
    w2 *= w2_mag/np.linalg.norm(w2)
    return w1, w2

def dynamics(t, y_vect, freq_d, w1_mag, w2_mag, dist, sol, use_approx):
    X = SE2(x=y_vect[0], y=y_vect[1], theta=y_vect[2])
    X_r = SE2(x=y_vect[3], y=y_vect[4], theta=y_vect[5])
    
    e = se2(x=y_vect[6], y=y_vect[7], theta=y_vect[8])
    eta = X.inv@X_r
    e_nl = eta.log
    
    K, B, _ = solve_control_gain()
    
    # reference input
    v_r = se2(x=1, y=0, theta=4*np.pi/1155*(1 - np.cos(2*np.pi*t/10))**6)
    
    # disturbance
    if dist == 'sine':
        w = se2(x=np.cos(2*np.pi*freq_d*t)*w1_mag, y=np.sin(2*np.pi*freq_d*t)*w1_mag, theta=np.cos(2*np.pi*freq_d*t)*w2_mag)
    # square wave
    elif dist == 'square':
        w = se2(x=0, y=signal.square(2*np.pi*freq_d*t)*w1_mag, theta=signal.square(2*np.pi*freq_d*t)*w2_mag)
    # maximize dV
    elif dist == 'maxdV':
        er = e.vee
        w1, w2 = maxw(sol, e, w1_mag, w2_mag)
        w = se2(w1[0], w1[1], w2)
        #print(w1, w2)
    elif dist == 'const':
        w = se2(0, w1_mag, w2_mag)
        
    # control law applied to non-linear error
    u_nl = control_law(B, K, e_nl)
    v_nl = v_r + u_nl + w
    
    # control law applied to log-linear error
    u = control_law(B, K, e)
    v = v_r + u + w
        
    # log error dynamics
    U = se2_diff_correction(e)
    if use_approx:
        # these dynamics don't hold exactly unless you can move sideways
        e_dot = se2.from_vector((-v_r.ad_matrix + B@K)@e.vee + U@w.vee)
    else:
        # these dynamics, always hold
        e_dot = -v_r@e + se2.from_vector(U@(u + w).vee)
    
    return [
        # actual
        v_nl.x*np.cos(X.theta) - v_nl.y*np.sin(X.theta),
        v_nl.x*np.sin(X.theta) + v_nl.y*np.cos(X.theta),
        v_nl.theta,
        # reference
        v_r.x*np.cos(X_r.theta) - v_r.y*np.sin(X_r.theta),
        v_r.x*np.sin(X_r.theta) + v_r.y*np.cos(X_r.theta),
        v_r.theta,
        # log error
        e_dot.x,
        e_dot.y,
        e_dot.theta
    ]

# function to compute exp of log error
def compute_exp_log_err(e_x, e_y, e_theta, x_r, y_r, theta_r):
    return (SE2(x=x_r, y=y_r, theta=theta_r)@((se2(x=e_x, y=e_y, theta=e_theta).exp).inv)).params


# function to compute log of error using group elements
def compute_log_err(x, y, theta, x_r, y_r, theta_r):
    return (SE2(x=x, y=y, theta=theta).inv@SE2(x=x_r, y=y_r, theta=theta_r)).log.vee

# function to compute error using group elements
def compute_err(x, y, theta, x_r, y_r, theta_r):
    return (SE2(x=x, y=y, theta=theta).inv@SE2(x=x_r, y=y_r, theta=theta_r)).params


def simulate(tf, freq_d, w1_mag, w2_mag, x0, y0, theta0, dist, sol, use_approx):
    t_vect = np.arange(0, tf, 0.05)  # time vector

    X0 = SE2(x=x0, y=y0, theta=theta0)  # initial state
    X0_r = SE2(x=0, y=0, theta=0)  # initial reference state
    x0 = (X0.inv@X0_r).log  # initial log of error

    # solve initial value problem
    res = scipy.integrate.solve_ivp(
        fun=dynamics,
        t_span=[t_vect[0], t_vect[-1]], t_eval=t_vect,
        y0=[X0.x, X0.y, X0.theta,
            X0_r.x, X0_r.y, X0_r.theta,
            x0.x, x0.y, x0.theta], args=[freq_d, w1_mag, w2_mag, dist, sol, use_approx], rtol=1e-12, atol=1e-3)
    return res

def plot(res, name=None, legend=False, save=False):
    if save:
        os.makedirs('figures', exist_ok=True)
        
    y_vect = res['y']
    x, y, theta, x_r, y_r, theta_r, log_e_x, log_e_y, log_e_theta = [y_vect[i, :] for i in range(len(y_vect))]
    exp_log_err = np.array([ compute_exp_log_err(y[6], y[7], y[8], y[3], y[4], y[5]) for y in y_vect.T]).T
    log_err_check = np.array([ compute_log_err(y[0], y[1], y[2], y[3], y[4], y[5]) for y in y_vect.T]).T
    err = np.array([ compute_err(y[0], y[1], y[2], y[3], y[4], y[5]) for y in y_vect.T]).T

    plt.rcParams['figure.figsize'] = (15, 10)
    
    plt.figure(1)
    title = name + 'X-Y Trajectory'
    plt.title(title)
    plt.grid(True)
    plt.plot(x, y, 'b-', label='Lie Group' if legend else None, linewidth=3, alpha=0.5)
    plt.plot(exp_log_err[0, :], exp_log_err[1, :], 'r-', label='Lie Algebra' if legend else None, linewidth=1)
    plt.plot(x_r, y_r, 'y-', alpha=1, linewidth=3, label='Reference' if legend else None)
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    if legend:
        plt.legend()
    plt.axis('equal');
    if save:
        plt.savefig('figures/' + title)

    plt.figure(2)
    title = name + 'X-Y Lie Group Error Relative to Reference Trajectory'
    plt.title(title)
    plt.grid(True)
    plt.plot(err[0, :], err[1, :], 'r-')
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    plt.axis('equal');
    if save:
        plt.savefig('figures/' + title)

    plt.figure(3)
    title = name + 'X-Y Lie Algebra Error Relative to Reference Trajectory'
    plt.title(title)
    plt.grid(True)
    plt.plot(log_err_check[0, :], log_err_check[1, :], 'b-', label='Lie Group' if legend else None, linewidth=1)
    plt.plot(log_e_x, log_e_y, 'r-', label='Lie Algebra'  if legend else None, linewidth=1)
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    plt.axis('equal')
    if save:
        plt.savefig('figures/' + title)

    plt.figure(4)
    title = name + 'Lie Group Exponential Coordinates ($x$, $y$, $\\theta$) Time History'
    plt.title(title)
    plt.grid(True)
    plt.plot(res['t'], log_e_x, 'r--', label='$x_{algebra}$'  if legend else None)
    plt.plot(res['t'], log_err_check[0, :], 'r-', label='$x_{group}$'  if legend else None)
    plt.plot(res['t'], log_e_y, 'g--', label='$y_{algebra}$'  if legend else None)
    plt.plot(res['t'], log_err_check[1, :], 'g-', label='$y_{group}$'  if legend else None)
    plt.plot(res['t'], log_e_theta, 'b--', label='$\\theta_{algebra}$'  if legend else None)
    plt.plot(res['t'], log_err_check[2, :], 'b-', label='$\\theta_{group}$'  if legend else None)
    plt.xlabel('t, sec')
    if legend:
        plt.legend()
    if save:
        plt.savefig('figures/' + title)
    
    K, B, _ = solve_control_gain()

    plt.figure(5)
    u = np.array([ control_law(B, K, se2.from_vector(y[6:9])).vee for y in res['y'].T ])
    plt.plot(u[:, 0], 'r-', label='x' if legend else None)
    plt.plot(u[:, 1], 'g-', label='y' if legend else None)
    plt.plot(u[:, 2], 'b-', label='$\\theta$' if legend else None)
    plt.grid(True)
    plt.xlabel('t, sec')
    plt.ylabel('Lie Algebra Value')
    title = name + 'Lie Algebra Control'
    plt.title(title);
    if legend:
        plt.legend()
    if save:
        plt.savefig('figures/' + title)
    return locals()

def plot_simulated(res, name=None, legend=False, save=False, **plt_kwargs):
    if save:
        os.makedirs('figures', exist_ok=True)
        
    y_vect = res['y']
    x, y, theta, x_r, y_r, theta_r, log_e_x, log_e_y, log_e_theta = [y_vect[i, :] for i in range(len(y_vect))]
    ref = np.array([x,y])

    plt.rcParams['figure.figsize'] = (15, 10)
    
    plt.figure(1, figsize=(8,8))
    label = 'Simulated Trajectory in ' + name
    plt.grid(True)
    plt.plot(x, y, label=label  if legend else None, **plt_kwargs)
    #plt.plot(x_r, y_r, 'y-', alpha=1, linewidth=3, label='Reference' if legend else None)
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    
    if legend:
        plt.legend(loc = 1)
    plt.axis('equal');
    if save:
        plt.savefig('figures/' + title)
        
def plot_simulated_corres(res, name=None, legend=False, save=False):
    if save:
        os.makedirs('figures', exist_ok=True)
        
    y_vect = res['y']
    x, y, theta, x_r, y_r, theta_r, log_e_x, log_e_y, log_e_theta = [y_vect[i, :] for i in range(len(y_vect))]
    exp_log_err = np.array([ compute_exp_log_err(y[6], y[7], y[8], y[3], y[4], y[5]) for y in y_vect.T]).T
    log_err_check = np.array([ compute_log_err(y[0], y[1], y[2], y[3], y[4], y[5]) for y in y_vect.T]).T
    err = np.array([ compute_err(y[0], y[1], y[2], y[3], y[4], y[5]) for y in y_vect.T]).T

    plt.rcParams['figure.figsize'] = (15, 10)
    
    plt.figure(1)
    title = name + 'X-Y Trajectory'
    plt.title(title)
    plt.grid(True)
    plt.plot(x, y, 'b-.', label='Lie Group' if legend else None, linewidth=1, alpha=0.5)
    plt.plot(exp_log_err[0, :], exp_log_err[1, :], 'r-', label='Lie Algebra' if legend else None, linewidth=0.5, alpha = 0.7)
    plt.plot(x_r, y_r, 'y-', alpha=1, linewidth=4, label='Reference' if legend else None)
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    if legend:
        plt.legend()
    plt.axis('equal');
    if save:
        plt.savefig('figures/' + title)

def simulate_and_plot(frequencies, tf, w1_mag, w2_mag):
    def worker(n, freq_d, n_freq, data):
        final = n_freq - 1
        data[n]  = simulate(tf=tf, freq_d=freq_d, w1_mag=w1_mag, w2_mag=w2_mag, x0=0, y0=0, theta0=0, dist='square', sol='', use_approx=False)
        print('#', end='')

    print('simulating: ', end='')
    manager = multiprocessing.Manager()
    data = manager.dict()
    jobs = []
    for n, freq in enumerate(frequencies):
        p = multiprocessing.Process(target=worker, args=(n, freq, len(frequencies), data))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    print(' complete')

    print('plotting: ', end='')
    for n, proc in enumerate(jobs):
        last = (n == len(jobs) - 1)
        plot(data[n], name='', legend=last, save=last);
        print('#', end='')
    print(' complete')


def solve_lmi(alpha, A1, A2, U1, U2, verbosity=0):
    # also maximize dV
    
    prob = picos.Problem()
    P = picos.SymmetricVariable('P', (3, 3))
    P1 = P[:2, :]
    P2 = P[2, :]
    mu1 = picos.RealVariable('mu_1')
    mu2 = picos.RealVariable('mu_2')
    gam = mu1 + mu2
    block_eq1 = picos.block([
         [A1.T*P + P*A1 + alpha*P, P1.T*U1, P1.T*U2 + P2.T],
         [U1.T*P1, -alpha*mu1*np.eye(2), 0],
         [U2.T*P1 + P2, 0, -alpha*mu2]])
    block_eq2 = picos.block([
         [A2.T*P + P*A2 + alpha*P, P1.T*U1, P1.T*U2 + P2.T],
         [U1.T*P1, -alpha*mu1*np.eye(2), 0],
         [U2.T*P1 + P2, 0, -alpha*mu2]])
    prob.add_constraint(block_eq1 << 0) # dV < 0
    prob.add_constraint(block_eq2 << 0)
    prob.add_constraint(P >> 1)
    prob.add_constraint(mu1 >> 0)
    prob.add_constraint(mu2 >> 0)
    prob.set_objective('min', mu1 + mu2)
    try:
        prob.solve(options={'verbosity': verbosity})
        cost = gam.value
    except Exception as e:
        print(e)
        cost = -1
    return {
        'cost': cost,
        'prob': prob,
        'mu1': mu1.value,
        'mu2': mu2.value,
        'P': np.round(np.array(P.value), 3),
        'alpha':alpha,
        'gam': gam
    }

def find_se2_invariant_set(verbosity=0):
    dA = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 0]])    
    A0 = solve_control_gain()[2]
    A1 = A0 + 0*dA
    A2 = A0 + np.pi/4*dA

    # these are the two parts of U(x), split ast U(x) = [U1, U2], where the first impacts the u, v and the last impacts the w disturbance
    # these are the zero order terms of the taylor expansion below
    # TODO: could add polytopic system with bounded input disturbance, U(x) is actually a function of the state not a constant, so this 
    # is an under approximation as is
    U1 = np.eye(2)*np.pi/2 # multiply singular val of U
    U2 = np.array([
        [0],
        [0]])
    
    # we use fmin to solve a line search problem in alpha for minimum gamma
    if verbosity > 0:
        print('line search')
    
    # we perform a line search over alpha to find the largest convergence rate possible
    alpha_1 = np.max(np.array([np.linalg.eig(A1)[0],np.linalg.eig(A2)[0]]))
    alpha_2 = np.max(np.linalg.eig(A2)[0])
    alpha_opt = scipy.optimize.fminbound(lambda alpha: solve_lmi(alpha, A1, A2, U1, U2, verbosity=verbosity)['cost'], x1=0.001, x2=-alpha_1, disp=True if verbosity > 0 else False)
    #print(alpha_opt1)
    #alpha_opt2 = scipy.optimize.fminbound(lambda alpha: solve_lmi(alpha, A1, A2, U1, U2, verbosity=verbosity)['cost'], x1=0.001, x2=-alpha_2, disp=True if verbosity > 0 else False)
    #print(alpha_opt2)
    #alpha_opt = np.min(np.array([alpha_opt1, alpha_opt2]))
    
    sol = solve_lmi(alpha_opt, A1, A2, U1, U2)
    prob = sol['prob']
    if prob.status == 'optimal':
        P = prob.variables['P'].value
        mu1 =  prob.variables['mu_1'].value
        mu2 =  prob.variables['mu_2'].value
        if verbosity > 0:
            print(sol)
    else:
        raise RuntimeError('Optimization failed')
        
    return sol


def se2_lie_algebra_invariant_set_points(sol, t, w1_mag, w2_mag, e0): # w1_mag (x-y direc): wind speed
    P = sol['P']
    # V = xTPx scalar
    beta = (e0.T@P@e0) # V0
    #print('V0', beta)
    val = beta*np.exp(-sol['alpha']*t) + sol['mu1']*w1_mag**2 + sol['mu2']*w2_mag**2 # V(t)
    #print('val', val)
    
    # 1 = xT(P/V(t))x, equation for the ellipse
    evals, evects = np.linalg.eig(P/val)
    radii = 1/np.sqrt(evals)
    R = evects@np.diag(radii)
    R = np.real(R)
    
    # draw sphere
    points = []
    n = 25
    for u in np.linspace(0, 2*np.pi, n):
        for v in np.linspace(0, 2*np.pi, 2*n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    for v in np.linspace(0, 2*np.pi, 2*n):
        for u in np.linspace(0, 2*np.pi, n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    points = np.array(points).T
    #u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:40j]
    #x = np.cos(u) * np.sin(v)
    #y = np.sin(u) * np.sin(v)
    #z = np.cos(v)
    #points = np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
    return R@points

def SE2_exp_wedge(a): 
    assert a.shape == (3,)
    alpha = a[2]
    u1 = a[0]
    u2 = a[1]
    if abs(alpha) < 0.1:
        t1 = 1 - alpha**2/6 + alpha**4/120 
        t2 = alpha/2 - alpha**3/24 - alpha**5/720 
    else:
        t1 = np.sin(alpha)/alpha
        t2 = (1-np.cos(alpha))/alpha
    x = np.array([[t1, -t2],
                  [t2, t1]])@[u1,u2]
    return np.array([x[0], x[1], alpha])

def se2_log(a):
    assert a.shape == (3,)
    theta = a[2]
    x = a[0]
    y = a[1]
    with np.errstate(divide='ignore',invalid='ignore'):
        a = np.where(np.abs(theta) < 1e-3, 1 - theta**2/6 + theta**4/12, np.sin(theta)/theta)
        b = np.where(np.abs(theta) < 1e-3, theta/2 - theta**3/24 + theta**5/720, (1 - np.cos(theta))/theta)
    V_inv = np.array([
        [a, b],
        [-b, a]
    ])/(a**2 + b**2)
    p = V_inv@np.array([x, y])
    return np.array([p[0], p[1], theta])

def plot_invariant_set():
    sol = find_se2_invariant_set()
    w1 = 1
    w2 = 1
    e = np.array([0, 0, 0]) # Lie Group
    e0 = se2_log(e) # Lie Algebra

    t = 0
    points = se2_lie_algebra_invariant_set_points(sol, t, w1, w2, e0) #Lie Algebra

    exp_points = np.zeros((3,points.shape[1]))
    for i in range(points.shape[1]):
        exp_points[:,i] = SE2_exp_wedge(points[:,i])


    plt.figure(figsize=(8,4))
    ax = plt.subplot(121)
    ax.plot(points[0, :], points[1, :], 'g');
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    #ax.plot(e0[0],e0[1],'ro')
    plt.axis('equal')
    plt.grid(True)
    ax2 = plt.subplot(122)
    ax2.plot(exp_points[0, :-1], exp_points[1, :-1], 'g');
    #ax2.plot(e[0],e[1],'ro')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    ax.set_title('Invariant Set in Lie Algebra')
    ax2.set_title('Invariant Set in Lie Group')

    plt.figure(figsize=(8,4))
    ax = plt.subplot(121)
    ax.plot(points[1, :], points[2, :], 'g');
    ax.set_xlabel('y')
    ax.set_ylabel('$\\theta$')
    plt.axis('equal')
    ax2 = plt.subplot(122)
    ax2.plot(exp_points[1, :], exp_points[2, :], 'g');
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.axis('equal')
    plt.tight_layout()

    plt.figure(figsize=(8,4))
    ax = plt.subplot(121)
    ax.plot(points[0, :], points[2, :], 'g');
    ax.set_xlabel('x')
    ax.set_ylabel('$\\theta$')
    plt.axis('equal')
    ax2 = plt.subplot(122)
    ax2.plot(exp_points[0, :], exp_points[2, :], 'g');
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.axis('equal')
    plt.tight_layout()

    plt.figure(figsize=(12,6))
    ax = plt.subplot(121, projection='3d', proj_type='ortho', elev=40, azim=40)
    ax.plot3D(e0[0], e0[1], e0[2], 'ro');
    ax.plot3D(points[0, :], points[1, :], points[2, :],'g');
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$\\theta$')
    plt.axis('auto')
    ax2 = plt.subplot(122, projection='3d', proj_type='ortho', elev=40, azim=40)
    ax2.plot3D(e[0], e[1], e[2], 'ro');
    ax2.plot3D(exp_points[0, :], exp_points[1, :], exp_points[2, :], 'g');
    ax2.plot3D(e[0], e[1], e[2], 'ro');
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.axis('auto')
    plt.tight_layout()

def rotate_point(point, angle):
    new_point = array([point[0] * cos(angle) - point[1] * sin(angle),
                 point[0] * sin(angle) + point[1] * cos(angle)])
    return new_point

def flowpipes(res, tf, n, e0, w1, w2, sol):
    
    t_vect = np.arange(0, tf, 0.05)
    
    y_vect = res['y']
    x, y, theta, x_r, y_r, theta_r, log_e_x, log_e_y, log_e_theta = [y_vect[i, :] for i in range(len(y_vect))]
    
    nom = np.array([x_r,y_r]).T
    flowpipes = []
    intervalhull = []
    if len(t_vect)%n == 0:
        steps = int(len(t_vect)/n)
    else:
        steps = int(len(t_vect)/n + 1)
    a = n      
    
    for i in range(n):
        if (steps*(i+1)) <= len(t_vect):
            nom_i = nom[steps*i:steps*(i+1),:] # steps*2
        else:
            nom_i = nom[steps*i:len(t_vect),:]
            
        # Get interval hull
        hull_points = qhull2D(nom_i)
        # Reverse order of points, to match output from other qhull implementations
        hull_points = hull_points[::-1]
        (rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)
        # add first corner_points to last
        corner_points = np.append(corner_points, corner_points[0].reshape(1,2), axis = 0)
        
        dx = corner_points[0][0]-corner_points[1][0]
        if width > height:
            if dx > .1:
                angle = np.arccos((corner_points[0][0]-corner_points[1][0])/width)
            else:
                angle = np.arccos((corner_points[2][0]-corner_points[1][0])/width)
        else:
            if dx > .1:
                angle = np.arccos((corner_points[0][0]-corner_points[1][0])/height)
            else:
                angle = np.arccos((corner_points[2][0]-corner_points[1][0])/height) 
        # fix mess up part
        if angle > 3:
            if i < int(n/3):
                angle = angle - pi
        if angle < 1:
            if i > int(n/3):
                a = i
        if i >= a:
            angle = angle + pi
        
        t = 0.05*i*steps
        # invariant set in se2
        points = se2_lie_algebra_invariant_set_points(sol, t, w1, w2, e0)
        
        # exp map
        exp_points = np.zeros((3,points.shape[1]))
        for i in range(points.shape[1]):
            exp_points[:,i] = SE2_exp_wedge(points[:,i])
            
        # invariant set in SE2   
        inv_points = rotate_point(exp_points, angle)
        
        P2 = Polytope(inv_points.T)
        # minkowski sum
        P1 = Polytope(corner_points) # interval hull
        P = P1+P2 # sum
        p1_vertices = P1.V
        p_vertices = P.V
        p_vertices = np.append(p_vertices, p_vertices[0].reshape(1,2), axis = 0)
        flowpipes.append(p_vertices)
        intervalhull.append(p1_vertices)
    return flowpipes, intervalhull, nom


def circ_inv(r):
    theta_circ = np.linspace(0, 2*np.pi, 100)
    circ = array([r*np.cos(theta_circ), r*np.sin(theta_circ)]).T
    return circ


def w(sol, x):
    U1 = np.eye(2)*pi/2 # multiply singular val of U
    U2 = np.array([
        [0],
        [0]])
    P = sol['P']
    P1 = P[:2, :]
    P2 = P[2, :]
    mu1 = sol['mu1'] 
    mu2 = sol['mu2']
    alpha = sol['alpha']
    
    w1 = ((U1@P1@x + x@P1.T@U1)/(2*alpha*mu1)).real
    w2 = ((U2.T@P1+P2)@x/(alpha*mu2)).real
    
    return w1, w2


# flow pipes 2d

def traj(x0, u, omega, t):
    
    omega_t = np.array([0,omega,0,omega,0,omega,0,omega])

    n_steps = len(t)
    dt = t[1]-t[0]

    # initial condition
    xs = x0
    out_xr = []
    
    tc = n_steps/(omega_t.shape[0]) # time of changing velocity
    
    for i in range(n_steps):
        
        # change of velocity
        if i % tc == 0:
            w = omega[int(i//tc)] #ut[int(i//tc*2)+1]
            if w == 0:
                v = u
            else:
                v = 1
        
        # store data
        out_xr.append(xs) # reference trajectory
        # Propagate reference
        xs = xs + dt*np.array([v*np.cos(xs[2]), v*np.sin(xs[2]), w])


    x = np.array(out_xr)
    return x

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

def minBoundingRect(hull_points_2d):
    #print "Input convex hull points: "
    #print hull_points_2d

    # Compute edges (x2-x1,y2-y1)
    edges = zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
    for i in range( len(edges) ):
        edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
        edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
        edges[i] = [edge_x,edge_y]
    #print "Edges: \n", edges

    # Calculate edge angles   atan2(y/x)
    edge_angles = zeros( (len(edges)) ) # empty 1 column array
    for i in range( len(edge_angles) ):
        edge_angles[i] = math.atan2( edges[i,1], edges[i,0] )
    #print "Edge angles: \n", edge_angles

    # Check for angles in 1st quadrant
    for i in range( len(edge_angles) ):
        edge_angles[i] = abs( edge_angles[i] % (math.pi/2) ) # want strictly positive answers
    #print "Edge angles in 1st Quadrant: \n", edge_angles

    # Remove duplicate angles
    edge_angles = unique(edge_angles)
    #print "Unique edge angles: \n", edge_angles

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, sys.maxsize, 0, 0, 0, 0, 0, 0) # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    #print("Testing", len(edge_angles), "possible rotations for bounding box... \n")
    for i in range( len(edge_angles) ):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = array([ [ math.cos(edge_angles[i]), math.cos(edge_angles[i]-(math.pi/2)) ], [ math.cos(edge_angles[i]+(math.pi/2)), math.cos(edge_angles[i]) ] ])
        #print "Rotation matrix for ", edge_angles[i], " is \n", R

        # Apply this rotation to convex hull points
        rot_points = dot(R, transpose(hull_points_2d) ) # 2x2 * 2xn
        #print "Rotated hull points are \n", rot_points

        # Find min/max x,y points
        min_x = nanmin(rot_points[0], axis=0)
        max_x = nanmax(rot_points[0], axis=0)
        min_y = nanmin(rot_points[1], axis=0)
        max_y = nanmax(rot_points[1], axis=0)
        #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width*height
        #print "Potential bounding box ", i, ":  width: ", width, " height: ", height, "  area: ", area 

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )
        # Bypass, return the last found rect
        #min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]   
    R = array([ [ math.cos(angle), math.cos(angle-(math.pi/2)) ], [ math.cos(angle+(math.pi/2)), math.cos(angle) ] ])
    #print "Projection matrix: \n", R

    # Project convex hull points onto rotated frame
    proj_points = dot(R, transpose(hull_points_2d) ) # 2x2 * 2xn
    #print "Project hull points are \n", proj_points

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]
    #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

    # Calculate center point and project onto rotated frame
    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2
    center_point = dot( [ center_x, center_y ], R )
    #print "Bounding box center point: \n", center_point

    # Calculate corner points and project onto rotated frame
    corner_points = zeros( (4,2) ) # empty 2 column array
    corner_points[0] = dot( [ max_x, min_y ], R )
    corner_points[1] = dot( [ min_x, min_y ], R )
    corner_points[2] = dot( [ min_x, max_y ], R )
    corner_points[3] = dot( [ max_x, max_y ], R )
    #print "Bounding box corner points: \n", corner_points

    #print "Angle of rotation: ", angle, "rad  ", angle * (180/math.pi), "deg"

    return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point, corner_points) # rot_angle, area, width, height, center_point, corner_points


def qhull2D(sample):
    link = lambda a,b: concatenate((a,b[1:]))
    edge = lambda a,b: concatenate(([a],[b]))
    def dome(sample,base): 
        h, t = base
        dists = dot(sample-h, dot(((0,-1),(1,0)),(t-h)))
        outer = repeat(sample, dists>0, 0)
        if len(outer):
            pivot = sample[argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                    dome(outer, edge(pivot, t)))
        else:
            return base
    if len(sample) > 2:
        axis = sample[:,0]
        base = take(sample, [argmin(axis), argmax(axis)], 0)
        return link(dome(sample, base), dome(sample, base[::-1]))
    else:
        return sample
    
    
def flowpipes_test(y, r, t, n):
    nom = y[:,0:2] # n*2 (x-y direction)
    
    # bound 
    theta_circ = np.linspace(0, 2*np.pi, 100)
    circ = np.array([r*np.cos(theta_circ), r*np.sin(theta_circ)]).T
    P2 = Polytope(circ) # invariant set
    
    flowpipes = []
    intervalhull = []
    steps = int(len(t)/n)
    
    for i in range(n):
        # get traj for certain fixed time interval
        nom_i = nom[steps*i:steps*(i+1),:] # steps*2
        
        # Get interval hull
        hull_points = qhull2D(nom_i)
        # Reverse order of points, to match output from other qhull implementations
        hull_points = hull_points[::-1]
        (rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(hull_points)
        
        # minkowski sum
        P1 = Polytope(corner_points) # interval hull
        P = P1+P2 # sum
        p1_vertices = P1.V
        p_vertices = P.V
        flowpipes.append(p_vertices)
        intervalhull.append(p1_vertices)
    return flowpipes, intervalhull, nom