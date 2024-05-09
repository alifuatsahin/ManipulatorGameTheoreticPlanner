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

const lambda = rand(1,33)



function InputCost(x, idx = 1)
    if idx == 1
        return 0.5 * (x[5:6]' * R * x[5:6])
    else
        return 0.5 * (x[7:8]' * R * x[7:8])
    end
end

function ReferenceCost(x)
    return 0.5 * (x[1:4] - x_ref)' * Q * (x[1:4] - x_ref) 
end

function Polyhedral_1(x, idx1 = 2, idx2 = 1)

    t1 = [0; 0]
    t2 = [0; 10]
    if idx1 == 2 && idx2 == 1 || idx1 == 3 && idx2 == 4 
        return lambda[1]*(H*t1 - h)'* x[9:12] + (F*t2 - g)'*x[13:16]
    elseif idx1 == 2 && idx2 == 3 || idx1 == 3 && idx2 == 2 
        return lambda[12]*(H*t1 - h)'* x[17:20] + (F*t2 - g)'*x[21:24]
    elseif idx1 == 2 && idx2 == 4 || idx1 == 3 && idx2 == 1 
        return lambda[23]*(H*t1 - h)'* x[25:28] + (F*t2 - g)'*x[29:32]
    
    end 
end 

function Polyhedral_2(x, idx1 = 2, idx2 = 1)

    if idx1 == 2 && idx2 == 1 || idx1 == 3 && idx2 == 4 
        return dot(lambda[2:5], x[9:12]) + dot(lambda[6:9],x[13:16])
    elseif idx1 == 2 && idx2 == 3 || idx1 == 3 && idx2 == 2 
        return dot(lambda[13:16], x[17:20]) + dot(lambda[17:20],x[21:24])
    elseif idx1 == 2 && idx2 == 4 || idx1 == 3 && idx2 == 1 
        return dot(lambda[24:27],x[28:31]) + dot(lambda[30:33],x[29:32])
    end 
end

function Polyhedral_3(x, idx1 = 2, idx2 = 1)

    R2 = [cos(x[idx2]) sin(x[idx2]); -sin(x[idx2]) cos(x[idx2])]

    if idx1 == 2 && idx2 == 1 || idx1 == 3 && idx2 == 4 
        return lambda[10]*(norm(R2'* F'*x[13:16]))
    elseif idx1 == 2 && idx2 == 3 || idx1 == 3 && idx2 == 2
        return lambda[21]*(norm(R2'* F'*x[21:24]))
    elseif idx1 == 2 && idx2 == 4 || idx1 == 3 && idx2 == 1
        return lambda[32]*(norm(R2'* F'*x[29:32]))
    end
end

function Polyhedral_4(x, idx1 = 2, idx2 = 1)
    R1 = [cos(x[idx1]) sin(x[idx1]); -sin(x[idx1]) cos(x[idx1])]
    R2 = [cos(x[idx2]) sin(x[idx2]); -sin(x[idx2]) cos(x[idx2])]

    if idx1 == 2 && idx2 == 1 || idx1 == 3 && idx2 == 4 
        return lambda[11]*(R1'*H'*x[9:12] +R2'* F'*x[13:16])
    elseif idx1 == 2 && idx2 == 3 || idx1 == 3 && idx2 == 2
        return lambda[22]*(R1'*H'*x[17:20] + R2'* F'*x[21:24])
    elseif idx1 == 2 && idx2 == 4 || idx1 == 3 && idx2 == 1
        return lambda[33]*(R1'*H'*x[25:28] + R2'* F'*x[29:32])
    end
end


x = rand(20,32) * 2 * pi

function TotalGradient(x, rob_idx = 1)
    ∇TotalCost = zeros(32,1)
    if rob_idx == 1
        idx1 = 2
        constrained_links = [1, 3, 4]
    else
        idx1 = 3
        constrained_links = [1, 2, 4]
    end
    for i = 1:20
        ∇TotalCost += ForwardDiff.gradient(x -> InputCost(x, rob_idx), x[i,:]) 
        ∇TotalCost += ForwardDiff.gradient(x -> ReferenceCost(x), x[i,:])
        if rob_idx == 1
            for idx2 in constrained_links
                ∇TotalCost += ForwardDiff.gradient(x -> Polyhedral_1(x, idx1, idx2), x[i,:])
                ∇TotalCost += ForwardDiff.gradient(x -> Polyhedral_2(x, idx1, idx2), x[i,:])
                ∇TotalCost += ForwardDiff.gradient(x -> Polyhedral_3(x, idx1, idx2), x[i,:])
                grad4 = ForwardDiff.jacobian(x -> Polyhedral_4(x, idx1, idx2), x[i,:])
                ∇TotalCost += grad4[1, :] + grad4[2, :]
            end 
        end
    end
    return ∇TotalCost
end


print(TotalGradient(x, 1))
print(TotalGradient(x, 2))


