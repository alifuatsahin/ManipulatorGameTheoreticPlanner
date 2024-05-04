struct InputCost
    value::Float64
    gradient_u::Vector{Float64}
    hessian_u::Matrix{Float64}
    R::Matrix{Float64}
end

function evaluate(ic::InputCost, u)
    ic.value = 0.5*(u' * ic.R * u)
end

function gradient_u(ic::InputCost, x, u)
    ic.gradient_u = ic.R * u
end

function hessian_u(ic::InputCost, x, u)
    ic.hessian_u = ic.R
end

mutable struct referenceCost
    value::Float64
    gradient::Vector{Float64}
    hessian::Matrix{Float64}
    Q::Matrix{Float64}
end

function evaluate(rc::referenceCost, x)
    rc.value = 0.5 * x' * rc.Q * x
end

function gradient_x(rc::referenceCost, x)
    rc.gradient = rc.Q * x
end

function hesian_x(rc::referenceCost)
    rc.hessian = rc.Q
end



