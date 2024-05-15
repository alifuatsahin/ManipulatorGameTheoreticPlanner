using ForwardDiff
using LinearAlgebra
using FiniteDiff
using Symbolics
using StaticArrays
using BlockDiagonals
using LinearSolve

include("utils.jl")

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
λ = @variables λ[1:n*N]
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

function inner_loop(x_init, G, H, N, x_flat, max_iter)
    x_flat_val = [x_init'...]
    for i in 1:max_iter
        println("Iteration: ", i)        
        
        x_state_vals = Dict(x_flat[i] => x_flat_val[i] for i in 1:16*N)
        lambda_vals = Dict(λ[i] => lambda[i] for i in 1:4*N)
        
        G_val = convert(Vector{Float64}, Symbolics.value.(substitute.(G, (x_state_vals,lambda_vals,))))
        H_val = convert(Matrix{Float64}, Symbolics.value.(substitute.(H, (x_state_vals,lambda_vals,))))
        println("G_val: ", norm(G_val,1))
        δy = - pinv(H_val) * G_val
    
        α = line_search(x_flat_val, G_val,  δy)
        println("α: ", α)
        
        println("norm of delta: ", norm(δy))   
        x_flat_val += α * δy

        x_state_vals = Dict(x_flat[i] => x_flat_val[i] for i in 1:16*N)
        G_new = convert(Vector{Float64},Symbolics.value.(substitute.(G, (x_state_vals,lambda_vals,))))

        println("G_new", norm(G_new,1))

        if norm(G_new) < 0.01
          return reshape(x_flat_val, 16, N)'
        end
    end
    return reshape(x_flat_val, 16, N)'
end

function line_search(y, G_val, δy, β=0.1, τ=0.9)
    α = 1
    while α > 1e-4  

        y_new = y + α * δy
        y_state_vals = Dict(x_flat[i] => y_new[i] for i in 1:16*N)

        G_alpha = convert(Vector{Float64},Symbolics.value.(substitute.(G, (y_state_vals,lambda_vals,))))

        println("norm of G_alpha: ", norm(G_alpha, 1))
        println("norm of G: ", norm(G_val, 1))

        if norm(G_alpha, 1) < (1 - α * β) * norm(G_val, 1)
            return α
        end
        
        α *= τ
    end
    return α  
end

x_converged = inner_loop(x_init, G, H, N, x_flat, max_iter)

