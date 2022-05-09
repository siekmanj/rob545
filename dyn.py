from sympy import *

var('L m g dθ_1 dθ_2 θ_1 θ_2')
V_1 = Matrix([0, 0, dθ_1, 0, dθ_1*L, 0])
V_2 = Matrix([dθ_2, dθ_1*sin(θ_2), dθ_1*cos(θ_1), -dθ_1 * L * sin(θ_2), dθ_1 * L * cos(θ_2) + dθ_2 * L, -dθ_1 * L * sin(θ_2)])
G_1 = Matrix([[0, 0, 0, 0, 0, 0],
              [0, 4, 0, 0, 0, 0],
              [0, 0, 4, 0, 0, 0,],
              [0, 0, 0, m, 0, 0],
              [0, 0, 0, 0, m, 0],
              [0, 0, 0, 0, 0, m]])
G_2 = Matrix([[4, 0, 0, 0, 0, 0],
              [0, 4, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0,],
              [0, 0, 0, m, 0, 0],
              [0, 0, 0, 0, m, 0],
              [0, 0, 0, 0, 0, m]])

init_printing(use_latex='mathjax')
VGV_1 = (V_1.T.multiply(G_1).multiply(V_1))
VGV_2 = (V_2.T.multiply(G_2).multiply(V_2))
K = 0.0 * (VGV_1 + VGV_2)[0]
P = m * g * (1 - cos(θ_2))
Lagrange = K - P

pprint((diff(Lagrange, dθ_1) - diff(Lagrange, θ_1)).subs({L: 1, m:2, dθ_1: 0, θ_1: pi/4, dθ_2: 0, θ_2: pi/4}))
pprint((diff(Lagrange, dθ_2) - diff(Lagrange, θ_2)).subs({L: 1, m:2, dθ_1: 0, θ_1: pi/4, dθ_2: 0, θ_2: pi/4}))

