using JuMP
using Ipopt
using LinearAlgebra
using Plots
using SCS

function solve_optimization(th1, th2, th3, th4, l1=6.0, l2=6.0, l3=6.0, l4=6.0, d=20.0, w=1.0)
    
    threshold = 1e-5

    l = 6.0

    H = [1 0; -1 0; 0 1; 0 -1]
    h = [l/2; l/2; w/2; w/2]

    F = [1 0; -1 0; 0 1; 0 -1]
    g = [l/2; l/2; w/2; w/2]

    R2 = [cos(th2) -sin(th2); sin(th2) cos(th2)]
    R4 = [cos(th4) sin(th4); -sin(th4) cos(th4)]

    t2(x1,x2) = [l1 * cos(x1) + l2 * cos(x2); l1 * sin(x1) + l2 * sin(x2)]

    t4(x3,x4) = [l3 * cos(x3) + l4 * cos(x4); d - l3 * sin(x3) - l4 * sin(x4)]

    function objective(x)
        return (-H*R2*t2(th1, th2) - h)'* x[1:4] + (-F*R4*t4(th3, th4) - g)'*x[5:8]
    end

    model = Model(SCS.Optimizer)

    @variable(model, x[i = 1:8])  

    @objective(model, Max, objective(x))

    @constraint(model, c[i=1:8], x[i] >= 0)

    @constraint(model, [1; R4' * F' * x[5:8]] in SecondOrderCone())

    @constraint(model, R2' * H' * x[1:4] + R4' * F' * x[5:8] .== 0)

    optimize!(model)

    x_sol1 = value.(x[1:4])
    x_sol2 = value.(x[5:8])

    x_sol1 = [abs(val) < threshold ? 0.0 : val for val in x_sol1]
    x_sol2 = [abs(val) < threshold ? 0.0 : val for val in x_sol2]

    obj_value = objective(value.(x))

    return x_sol1, x_sol2, obj_value
end

function warm_start(x_init, l1, l2, l3, l4, d)
    N, state_dim = size(x_init)
    for i in 1:N
        th1, th2, th3, th4 = x_init[i, 1:4]
        
        x_sol1, x_sol2, _ = solve_optimization(th1, th2, th3, th4, l1, l2, l3, l4, d)
        
        x_init[i, 9:12] = x_sol1
        x_init[i, 13:16] = x_sol2
    end
    return x_init
end