struct InputCost
    weight1::Float64
    weight2::Float64
    idx::Int

    function InputCost(idx, weight1=1.0, weight2=1.0)
        new(idx, weight1, weight2)
    end
end

function evaluate(ic::InputCost, x, u)
    return ic.weight1 * u[1]^2 + ic.weight2 * u[2]^2
end

function gradient_x(ic::InputCost, x, u)
    return zeros(length(x))
end

function gradient_u(ic::InputCost, x, u)
    grad_u = zeros(length(u))
    grad_u[1] = 2 * ic.weight1 * u[1]
    grad_u[2] = 2 * ic.weight2 * u[2]
    return grad_u
end

function hessian_x(ic::InputCost, x, u)
    return zeros(length(x), length(x))
end

function hessian_u(ic::InputCost, x, u)
    hess_u = zeros(length(u), length(u))
    hess_u[1, 1] = 2 * ic.weight1
    hess_u[2, 2] = 2 * ic.weight2
    return hess_u
end


# make a trial

ic = InputCost(1, 2, 1)
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
u = [3.0, 4.0]

println(evaluate(ic, x, u))
println(gradient_x(ic, x, u))
println(gradient_u(ic, x, u))
println(hessian_x(ic, x, u))
println(hessian_u(ic, x, u))




