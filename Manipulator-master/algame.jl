using ForwardDiff
using LinearAlgebra
using FiniteDiff
using Symbolics
using StaticArrays
using BlockDiagonals
using LinearSolve

include("utils.jl")
include("alsolver.jl")

# Link lengths
const l1 = 6.0
const l2 = 6.0
const l3 = 6.0
const l4 = 6.0
const w = 3

# Grounding distance
const d = 100.0

# Object Matrices
const H1 = [1 0; -1 0; 0 1; 0 -1]
const h1 = [l1/2; l1/2; w/2; w/2]

const H2 = [1 0; -1 0; 0 1; 0 -1]
const h2 = [l2/2; l2/2; w/2; w/2]

const F1 = [1 0; -1 0; 0 1; 0 -1]
const f1 = [l3/2; l3/2; w/2; w/2]

const F2 = [1 0; -1 0; 0 1; 0 -1]
const f2 = [l4/2; l4/2; w/2; w/2]

# Number of constraints
n = 4

# Cost Matrices
R = 0.1*I(4)
Q = 10000*I(4)

# Reference
θ_ref = [pi*2/3, pi*2/3, pi/6, pi/6]

# Discretization
dt = 0.1 # seconds [s]
horizon = 5 # seconds [s]
N = convert(Int64, horizon/dt) # number of time steps

# Lagrangian Multipliers
@variables λ[1:n*N]
lambda = ones(n*N)

# Initial Guess
θ_init = [3*pi/4, 3*pi/4, pi/4, pi/4]

# constraints
states_n = length(θ_init)

x_init = generate_trajectory(θ_init, θ_ref, N, dt)

@variables x[1:N, 1:size(x_init, 2)]

x1 = vcat([x[i, vcat(1:6)]' for i in 1:size(x, 1)]...)
x2 = vcat([x[i, vcat(1:4, 7:8)]' for i in 1:size(x, 1)]...)

x_flat = [x'...]
x1_flat = [x1'...]
x2_flat = [x2'...]

mu_1 = [x[:, 2*states_n+1:2*states_n+4]'...]
mu_2 = [x[:, 3*states_n+1:3*states_n+4]'...]

D = state_transition(x, dt, N, θ_init)
D_mu = dot(mu_1,D) + dot(mu_2,D)

C = constraints(x, N)
C_lambda = dot(λ, C)

J = player_cost(x, θ_ref, R, Q, N)

L = J + D_mu + C_lambda

∇L1 = Symbolics.gradient(L, x1_flat)
∇L2 = Symbolics.gradient(L, x2_flat)

G = []

for i in 1:N
    global G
    ∇L1_i = ∇L1[(i-1)*(states_n + 2) + 1:(i-1)*(states_n + 2) + 6]
    ∇L2_i = ∇L2[(i-1)*(states_n + 2) + 1:(i-1)*(states_n + 2) + 6]
    D_i = D[4*(i-1)+1:4*i]
    G_i = vcat(∇L1_i, ∇L2_i, D_i)
    G = vcat(G, G_i)
end

H = Symbolics.jacobian(G, x_flat)

max_iter = 100

x_converged = inner_loop(x_init, lambda, G, H, N, x_flat, λ, max_iter)
