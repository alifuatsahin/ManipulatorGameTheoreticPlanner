using LinearAlgebra
using Symbolics
using StaticArrays
using BlockDiagonals
using LinearSolve

include("utils.jl")
include("alsolver.jl")
include("plotting.jl")

global reshaped_value = nothing

# Link lengths
const l1 = 6.0
const l2 = 6.0
const l3 = 6.0
const l4 = 6.0
const w = 1

# Grounding distance
const d = 20.0

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
nce = 2
nci = 18
n = nce + nci

# State dims
state_dim = 24

# Cost Matrices
R = [20 0 0 0; 0 20 0 0; 0 0 20 0; 0 0 0 20]
Q = [300 0 0 0; 0 300 0 0; 0 0 300 0; 0 0 0 300]

# Reference
θ_ref = [pi/4, pi/4, 5*pi/6, 5*pi/6]

# Discretization
dt = 0.1 # seconds [s]
horizon = 1 # seconds [s]
N = convert(Int64, horizon/dt) # number of time steps

# Lagrangian Multipliers
@variables λ[1:n*N]
@variables ρ[1:n*N]
lambda = ones(n*N)*0.0
rho  = ones(n*N)

I_rho = Diagonal(ρ)

# Initial Guess
θ_init = [3*pi/4, 3*pi/4, pi/6, pi/6]

# constraints
states_n = 8

x_init = generate_trajectory(θ_init, θ_ref, state_dim, N, dt)

@variables x[1:N, 1:state_dim]

x1 = vcat([x[i, vcat(1:6, 9:12)]' for i in 1:N]...)
x2 = vcat([x[i, vcat(1:4, 7:8, 13:16)]' for i in 1:N]...)

x_flat = [x'...]
x1_flat = [x1'...]
x2_flat = [x2'...]

mu_1 = [x[:, end-7:end-4]'...]
mu_2 = [x[:, end-3:end]'...]

D = state_transition(x, dt, N, θ_init);
D_mu = dot(mu_1,D) + dot(mu_2,D);

C = constraints(x, N, F1 ,f1, F2, f2, H1, h1, H2, h2);
C_lambda = dot(λ, C);

J = player_cost(x, θ_ref, R, Q, N);

L = J + D_mu + C_lambda + 1/2*C'*I_rho*C;

∇L1 = Symbolics.gradient(L, x1_flat);
∇L2 = Symbolics.gradient(L, x2_flat);

println("Symbolic Differentiation Done")
G = []

for i in 1:N
    global G
    ∇L1_i = ∇L1[(i-1)*(states_n + 2) + 1:(i-1)*(states_n + 2) + 10]
    ∇L2_i = ∇L2[(i-1)*(states_n + 2) + 1:(i-1)*(states_n + 2) + 10]
    D_i = D[4*(i-1)+1:4*i]
    G_i = vcat(∇L1_i, ∇L2_i, D_i)
    G = vcat(G, G_i)
end

H = Symbolics.jacobian(G, x_flat);
println("Symbolic Hessian Done")
max_iter = 100

y = alsolver(lambda, rho, x_init, x_flat, λ, ρ, C, G, H, max_iter, nci, nce, N)


int_y_1 = generate_intermediate_points(y[:, 1], 5);
int_y_2 = generate_intermediate_points(y[:, 2], 5);
int_y_3 = generate_intermediate_points(y[:, 3], 5);
int_y_4 = generate_intermediate_points(y[:, 4], 5);

animate_robots(int_y_1, int_y_2, int_y_3, int_y_4, d, l1, l2, l3, l4, w)

animate_robots(y[:, 1], y[:, 2], y[:, 3], y[:, 4], d, l1, l2, l3, l4, w)

