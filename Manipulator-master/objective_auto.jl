using ForwardDiff
using LinearAlgebra

include("constraint.jl")


function InputCost(u)
    R = [4.0 0.0; 0.0 4.0]
    return 0.5 * (u' * R * u)
end

function ReferenceCost(x, y)
    Q = [1.0 0.0; 0.0 1.0]
    x_ref = [0.0, 0.0]
    return 0.5 * (x - x_ref)' * Q * (x - x_ref) 
end


u = rand(2, 50)
x = rand(2, 50)

ref_grads = zeros(2, 50)
inp_grads = zeros(2, 50)

ref_hessians = zeros(2, 2, 50)
inp_hessians = zeros(2, 2, 50)

for i = 1:50
    gradient_u = ForwardDiff.gradient(u -> InputCost(u), u[:, i])
    hessian_u = ForwardDiff.hessian(u -> InputCost(u), u[:, i])
    gradient_x = ForwardDiff.gradient(x -> ReferenceCost(x,y), x[:, i])
    hessian_x = ForwardDiff.hessian(x -> ReferenceCost(x,y), x[:, i])

    ref_grads[:, i] = gradient_x
    inp_grads[:, i] = gradient_u

    ref_hessians[:, :, i] = hessian_x
    inp_hessians[:, :, i] = hessian_u

end


println(ref_grads)
println(inp_grads)

