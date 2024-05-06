using ForwardDiff
using LinearAlgebra

# Define the function
function f(x)
    # Define parameters
    l1 = 1.0
    l2 = 1.0
    w = 1.0
    
    # Construct A matrix
    A = [l1 * sin(x[1]) + w     -1.0;
         l1 * cos(x[1]) + w     -1.0;
         l1 * cos(x[1]) + l2 * cos(x[2]) + w     -1.0;
         l1 * cos(x[1]) + l2 * sin(x[2]) + w     -1.0]

    # Construct b vector
    b = [l1 * cos(x[1]) + l2 * cos(x[2]) + w;
         l1 * cos(x[1]) + l2 * sin(x[2]) + w;
         l1 * cos(x[1]) + l2 * sin(x[2]) + w;
         l1 * cos(x[1]) + l2 * cos(x[2]) + w]

    g = [l1 * cos(x[3]) + l2 * cos(x[4]) + w;
    l1 * cos(x[3]) + l2 * sin(x[4]) + w;
    l1 * cos(x[3]) + l2 * sin(x[4]) + w;
    l1 * cos(x[3]) + l2 * cos(x[4]) + w]

    # Define parameters
     t = [2.0; 3.0]
     μ = [1.0, 2.0, 3.0, 4.0]
     λ = [5.0, 6.0, 7.0, 8.0]
    g'*μ + dot((A*t-b),λ)
end

# Evaluate the function
θ1_val = π/4
θ2_val = π/3
θ3_val = π/6
θ4_val = π/2

x = [θ1_val; θ2_val; θ3_val; θ4_val]

Hessian_f = ForwardDiff.hessian(f, x)
gradient_f = ForwardDiff.gradient(f, x)

println(Hessian_f)
println(gradient_f)
