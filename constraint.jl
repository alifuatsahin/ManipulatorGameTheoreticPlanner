using ForwardDiff
using LinearAlgebra

# Define the function
function f(θ1, θ2, θ3, θ4, μ, t, λ)
    # Define parameters
    l1 = 1.0
    l2 = 1.0
    w = 1.0
    
    # Construct A matrix
    A = [l1 * sin(θ1) + w     -1.0;
         l1 * cos(θ1) + w     -1.0;
         l1 * cos(θ1) + l2 * cos(θ2) + w     -1.0;
         l1 * cos(θ1) + l2 * sin(θ2) + w     -1.0]

    # Construct b vector
    b = [l1 * cos(θ1) + l2 * cos(θ2) + w;
         l1 * cos(θ1) + l2 * sin(θ2) + w;
         l1 * cos(θ1) + l2 * sin(θ2) + w;
         l1 * cos(θ1) + l2 * cos(θ2) + w]

    g = [l1 * cos(θ3) + l2 * cos(θ4) + w;
    l1 * cos(θ3) + l2 * sin(θ4) + w;
    l1 * cos(θ3) + l2 * sin(θ4) + w;
    l1 * cos(θ3) + l2 * cos(θ4) + w]

    g'*μ + dot((A*t-b),λ)
end

# Define parameters
t = [2.0; 3.0]
μ = [1.0, 2.0, 3.0, 4.0]
λ = [5.0, 6.0, 7.0, 8.0]

# Evaluate the function
θ1_val = π/4
θ2_val = π/3
θ3_val = π/6
θ4_val = π/2


Hessian_f = ForwardDiff.hessian(x -> f(θ1_val, θ2_val, θ3_val, θ4_val, μ, t, λ), [θ1_val, θ2_val, θ3_val, θ4_val])

println("Hessian of f at θ1 = $θ1_val, θ2 = $θ2_val, θ3 = $θ3_val, θ4 = $θ4_val:\n", Hessian_f)
