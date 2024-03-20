# %%
import localsolver
import numpy as np

# ------------------------------------------
# Hexaly
# ------------------------------------------
# Problem Reference: http://datagenetics.com/blog/january32015/index.html

with localsolver.LocalSolver() as ls:
    PI = np.pi

    # Declare the optimization model
    m = ls.model

    # Numerical decisions
    R = m.float(0, 1)
    r = m.float(0, 1)
    h = m.float(0, 1)

    # Surface must not exceed the surface of the plain disc
    surface = PI * r**2 + PI * (R + r) * m.sqrt((R - r) ** 2 + h**2)
    m.constraint(surface <= PI)

    # Maximize the volume
    volume = PI * h / 3 * (R**2 + R * r + r**2)
    m.maximize(volume)

    m.close()

    # Parametrize the solver w.r.t computation time limit in terms of seconds
    ls.param.time_limit = 1

    ls.solve()

    # Print out results
    print(surface.value, volume.value)
    print(R.value, r.value, h.value)

# %%
# ------------------------------------------
# Scipy
# ------------------------------------------
from scipy.optimize import minimize

# (R, r, h) for scipy optimizer and variable
x0 = [0.3, 0.3, 0.3]
PI = np.pi

# add bounds
bnds = ((0.0, 1.0), (0.0, 1.0), (0, 1.0))


# add constrainst: its non-negative constraints
def surface_cons(x):
    return PI - (
        PI * x[1] ** 2 + PI * (x[0] + x[1]) * np.sqrt((x[0] - x[1]) ** 2 + x[2] ** 2)
    )


# define objective
def objective_func(x):
    return -(PI * x[2] / 3 * (x[0] ** 2 + x[0] * x[1] + x[1] ** 2))


# solve
cons = {"type": "ineq", "fun": surface_cons}
res = minimize(
    objective_func,
    x0,
    method="SLSQP",
    bounds=bnds,
    constraints=cons,
    options={"disp": True},
)
print(res.x)
