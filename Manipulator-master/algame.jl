using LinearAlgebra
using Symbolics
using StaticArrays
using BlockDiagonals
using LinearSolve
using NLsolve

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
const d = 22.0

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
nci = 14
n = nce + nci

# State dims
state_dim = 24

# Cost Matrices
R = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
Q1 = [20 0 0 0; 0 20 0 0; 0 0 0 0; 0 0 0 0]
Q2 = [0 0 0 0; 0 0 0 0; 0 0 20 0; 0 0 0 20]

# Reference
θ_ref = [pi/4, pi/4, 5*pi/6, 5*pi/6]

# Discretization
dt = 0.2 # seconds [s]
horizon = 2  # seconds [s]
N = convert(Int64, horizon/dt) # number of time steps

# Lagrangian Multipliers
@variables λ[1:n*N]
@variables ρ[1:n*N]
lambda = ones(n*N)*0.2
rho  = ones(n*N)*0.2

I_rho = Diagonal(ρ)

# Initial Guess
θ_init = [3*pi/4, 3*pi/4, pi/6, pi/6]

# constraints
states_n = 10

x_init = generate_trajectory(θ_init, θ_ref, state_dim, N, dt)

x_init = warm_start(x_init, l1, l2, l3, l4, d)

@variables x[1:N, 1:state_dim]

x1 = vcat([x[i, vcat(1:6, 9:12)]' for i in 1:N]...)
x2 = vcat([x[i, vcat(1:4, 7:8, 13:16)]' for i in 1:N]...)

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

# vals_lambda = Dict(λ[i] => lambda[i] for i in eachindex(λ))

# vals_rho = Dict(ρ[i] => rho[i] for i in eachindex(ρ))

# vals_rho_lambda = merge(vals_lambda, vals_rho)

# G = Symbolics.value.(substitute.(G, (vals_rho_lambda,)));
# H = Symbolics.value.(substitute.(H, (vals_rho_lambda,)));

# G_func = build_function(G, x_flat; expression=Symbolics.Float64)[1];
# G_num = eval(G_func);

# H_func = build_function(H, x_flat; expression=Symbolics.Float64)[1];
# H_num = eval(H_func);

# Define the function for NLsolve
# function G_nlsolve!(F, x)
#     F .= G_num(x)
# end

# function H_nlsolve!(F, x)
#     F .= H_num(x)
# end

# Define the initial guess
# initial_guess = [x_init'...] # Flatten initial guess if necessary

# Solve the system
# result = nlsolve(G_nlsolve!, H_nlsolve!, initial_guess, method = :trust_region, show_trace = true)


println("Symbolic Hessian Done")
max_iter = 100

y = alsolver(lambda, rho, x_init, x_flat, λ, ρ, C, G, H, max_iter, nci, nce, N, state_dim)


int_y_1 = generate_intermediate_points(y[:, 1], 8);
int_y_2 = generate_intermediate_points(y[:, 2], 8);
int_y_3 = generate_intermediate_points(y[:, 3], 8);
int_y_4 = generate_intermediate_points(y[:, 4], 8);

animate_robots(int_y_1, int_y_2, int_y_3, int_y_4, d, l1, l2, l3, l4, w)

animate_robots(y[:, 1], y[:, 2], y[:, 3], y[:, 4], d, l1, l2, l3, l4, w)

y_trimmed = y[1: 6, :]

int_y_1_t = generate_intermediate_points(y_trimmed[:, 1], 15);
int_y_2_t = generate_intermediate_points(y_trimmed[:, 2], 15);
int_y_3_t = generate_intermediate_points(y_trimmed[:, 3], 15);
int_y_4_t = generate_intermediate_points(y_trimmed[:, 4], 15);

animate_robots(int_y_1_t, int_y_2_t, int_y_3_t, int_y_4_t, d, l1, l2, l3, l4, w)
