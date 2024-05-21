using LinearAlgebra
using Symbolics
using StaticArrays
using BlockDiagonals
using LinearSolve

include("utils.jl")
include("alsolver.jl")
include("plotting.jl")
include("warm_start.jl")

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
nce = 4
nci = 24
n = nce + nci

# State dims
state_dim = 32

# Cost Matrices
R = [20 0 0 0; 0 20 0 0; 0 0 20 0; 0 0 0 20]
Q1 = [50 0 0 0; 0 50 0 0; 0 0 0 0; 0 0 0 0]
Q2 = [0 0 0 0; 0 0 0 0; 0 0 50 0; 0 0 0 50]

# Reference
θ_ref = [pi/4, pi/4, 5*pi/6, 5*pi/6]

# Discretization
dt = 0.2 # seconds [s]
horizon = 4 # seconds [s]
N = convert(Int64, horizon/dt) # number of time steps

# Lagrangian Multipliers
@variables λ[1:n*N]
@variables ρ[1:n*N]
lambda = ones(n*N)*0.1
rho  = ones(n*N)

I_rho = Diagonal(ρ)

# Initial Guess
θ_init = [3*pi/4, 3*pi/4, pi/6, pi/6]

# constraints
states_n = 14

x_init = generate_trajectory(θ_init, θ_ref, state_dim, N, dt)

x_init = warm_start(x_init, l1, l2, l3, l4, d)

@variables x[1:N, 1:state_dim]

x1 = vcat([x[i, vcat(1:6, 9:16)]' for i in 1:N]...)
x2 = vcat([x[i, vcat(1:4, 7:8, 17:24)]' for i in 1:N]...)

x_flat = [x'...]
x1_flat = [x1'...]
x2_flat = [x2'...]

mu_1 = [x[:, end-7:end-4]'...]
mu_2 = [x[:, end-3:end]'...]

D = state_transition(x, dt, N, θ_init);
D_1mu = dot(mu_1, D);
D_2mu = dot(mu_2, D);
D_mu = D_1mu + D_2mu;

C = constraints(x, N, F1 ,f1, F2, f2, H1, h1, H2, h2);
C_lambda = dot(λ, C);

J1 = player_cost(x, θ_ref, R, Q1, N);
J2 = player_cost(x, θ_ref, R, Q2, N);

L1 = J1 + D_1mu + C_lambda + 1/2*C'*I_rho*C;
L2 = J2 + D_2mu + C_lambda + 1/2*C'*I_rho*C;

∇L1 = Symbolics.gradient(L1, x1_flat);
∇L2 = Symbolics.gradient(L2, x2_flat);

println("Symbolic Differentiation Done")
G = []

for i in 1:N
    global G
    ∇L1_i = ∇L1[(i-1)*(states_n) + 1:(i)*(states_n)]
    ∇L2_i = ∇L2[(i-1)*(states_n) + 1:(i)*(states_n)]
    D_i = D[4*(i-1)+1:4*i]
    G_i = vcat(∇L1_i, ∇L2_i, D_i)
    G = vcat(G, G_i)
end

H = Symbolics.jacobian(G, x_flat);
println("Symbolic Hessian Done")
max_iter = 100

y = alsolver(lambda, rho, x_init, x_flat, λ, ρ, C, G, H, max_iter, nci, nce, N, state_dim)


int_y_1 = generate_intermediate_points(y[:, 1], 2);
int_y_2 = generate_intermediate_points(y[:, 2], 2);
int_y_3 = generate_intermediate_points(y[:, 3], 2);
int_y_4 = generate_intermediate_points(y[:, 4], 2);

animate_robots(int_y_1, int_y_2, int_y_3, int_y_4, d, l1, l2, l3, l4, w)

animate_robots(y[:, 1], y[:, 2], y[:, 3], y[:, 4], d, l1, l2, l3, l4, w)

