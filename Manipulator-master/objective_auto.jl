using ForwardDiff
using LinearAlgebra

const R = [4.0 0.0; 0.0 4.0]
const Q = I(4)
const x_ref = [0.0, 0.0, 0.0, 0.0]

const l = 6.0
const w = 3

const H = [1 0; -1 0; 0 1; 0 -1]
const h = [l/2; l/2; w/2; w/2]

const F = [1 0; -1 0; 0 1; 0 -1]
const g = [l/2; l/2; w/2; w/2]

function InputCost(x, idx = 1)
    if idx == 1
        return 0.5 * (x[4:5]' * R * x[4:5])
    else
        return 0.5 * (x[6:7]' * R * x[6:7])
    end
end

function ReferenceCost(x)
    return 0.5 * (x[1:4] - x_ref)' * Q * (x[1:4] - x_ref) 
end

function Polyhedral_1(x, idx1 = 1, idx2 = 1)

    t1 = [0; 0]
    t2 = [0; 10]
    if idx1 == 2 && idx2 == 1
        return (H*t1 - h)'* x[9:12] + (F*t2 - g)'*x[13:16]
    elseif idx1 == 2 && idx2 == 3
        return (H*t1 - h)'* x[17:20] + (F*t2 - g)'*x[21:24]
    elseif idx1 == 2 && idx2 == 4
        return (H*t1 - h)'* x[25:28] + (F*t2 - g)'*x[29:32]
    
    end 
end 

function Polyhedral_2(x, idx1 = 1, idx2 = 1)
    R1 = [cos(x[idx1]) sin(x[idx1]); -sin(x[idx1]) cos(x[idx1])]
    R2 = [cos(x[idx2]) sin(x[idx2]); -sin(x[idx2]) cos(x[idx2])]

    t1 = [0; 0]
    t2 = [0; 10]

end


x = rand(20,64)


for i in 1:20

    ∇InputCost = ForwardDiff.gradient(x -> InputCost(x, 1), x[i, :])
    ∇ReferenceCost = ForwardDiff.gradient(x -> ReferenceCost(x), x[i, :])
    ∇Polyhedral_1 = ForwardDiff.gradient(x -> Polyhedral_1(x, 2, 3), x[i, :])

    ∇TotalCost = ∇InputCost + ∇ReferenceCost + ∇Polyhedral_1

    HessianInputCost = ForwardDiff.hessian(x -> InputCost(x, 1), x[i, :])
    HessianReferenceCost = ForwardDiff.hessian(x ->ReferenceCost(x), x[i, :])
    HessianPolyhedral_1 = ForwardDiff.hessian(x -> Polyhedral_1(x, 2, 3), x[i, :])

    HessianTotalCost = HessianInputCost + HessianReferenceCost + HessianPolyhedral_1

end 

println("Gradient of TotalCost:")
println(∇TotalCost)

println("Hessian of TotalCost:")
println(HessianTotalCost)
