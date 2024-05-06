using JuMP
using Ipopt
using LinearAlgebra
using Plots
using LazySets
using SCS

l = 1.0
w = 0.2

H = [1 0; -1 0; 0 1; 0 -1]
h = [l/2; l/2; w/2; w/2]

F = [1 0; -1 0; 0 1; 0 -1]
g = [l/2; l/2; w/2; w/2]

th1 = 0
th2 = 0

R1 = [cos(th1) sin(th1); -sin(th1) cos(th1)]

R2 = [cos(th2) sin(th2); -sin(th2) cos(th2)]

t1 = [0; 0]

t2 = [0; 10]

function objective(x)
    return (H*t1 - h)'* x[1:4] + (F*t2 - g)'*x[5:8]
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







