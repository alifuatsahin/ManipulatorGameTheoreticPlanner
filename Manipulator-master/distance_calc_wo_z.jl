using JuMP
using Ipopt
using LinearAlgebra
using Plots
using LazySets
using SCS
using Polyhedra

l = 10.0
w = 3

H = [1 0; -1 0; 0 1; 0 -1]
h = [l/2; l/2; w/2; w/2]

F = [1 0; -1 0; 0 1; 0 -1]
g = [l/2; l/2; w/2; w/2]

th1 = pi/2
th2 = 2*pi/7

R1 = [cos(th1) -sin(th1); sin(th1) cos(th1)]

R2 = [cos(th2) sin(th2); -sin(th2) cos(th2)]

t1 = [0; 0]

t2 = [0; 15]

function objective(x)
    return (-H*R1*t1 - h)'* x[1:4] + (-F*R2*t2 - g)'*x[5:8]
end 

model = Model(SCS.Optimizer)

@variable(model, x[i = 1:8])  

@objective(model, Max, objective(x))

@constraint(model, c[i=1:8], x[i] >= 0)

@constraint(model,[1;R2'* F'*x[5:8]] in SecondOrderCone())

@constraint(model,R1'*H'*x[1:4] + R2'* F'*x[5:8] .== 0)

optimize!(model)

println("Optimal solution:")
println("x[1]:")
println(value.(x[1:4]))
println("x[2]:")
println(value.(x[5:8]))

println("Objective value:")
println(objective(value.(x)))


H_transformed = H * R1
h_transformed = h + H * R1 * t1
F_transformed = F * R2
g_transformed = g + F * R2 * t2

# Create LazySets for transformed sets
H_set = Polyhedra.hrep(H_transformed, h_transformed)
F_set = Polyhedra.hrep(F_transformed, g_transformed)

p = polyhedron(H_set)
p2 = polyhedron(F_set)

plot(p, lab="Hx .<= h")
plot!(p2, lab="Fx .<= g")

plot!(xlims=(-30,30), ylims=(-30, 30))







