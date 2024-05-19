using ForwardDiff
using LinearAlgebra
using FiniteDiff
using Symbolics

const R = [4.0 0.0; 0.0 4.0]
const Q = I(4)
const x_ref = [0.0, 0.0, 0.0, 0.0]

const l1 = 6.0
const l2 = 6.0
const l3 = 6.0
const l4 = 6.0
const l = 6.0
const w = 3

const d = 10.0

const H = [1 0; -1 0; 0 1; 0 -1]
const h = [l/2; l/2; w/2; w/2]

const F = [1 0; -1 0; 0 1; 0 -1]
const g = [l/2; l/2; w/2; w/2]

const lambda = rand(20,32)
const mu = rand(20,8)

t_1(x) = [l1 * cos(x[1]); l1 * sin(x[1])]

t_2(x) = [l1 * cos(x[1]) + l2 * cos(x[2]); l1 * sin(x[1]) + l2 * sin(x[2])]

t_3(x) = [l3 * cos(x[3]); d - l3 * sin(x[3])]

t_4(x) = [l3 * cos(x[3]) + l4 * cos(x[4]); d - l3 * sin(x[3]) - l4 * sin(x[4])]

R_1(x) = [cos(x) -sin(x); sin(x) cos(x)]
R_2(x) = [cos(x) sin(x); -sin(x) cos(x)]

function InputCost(x, idx = 1)
    if idx == 1
        return 0.5 * (x[5:6]' * R * x[5:6])
    else
        return 0.5 * (x[7:8]' * R * x[7:8])
    end
end

function InputCostAug(x)
    return 0.5 * (x[5:6]' * R * x[5:6])
end

function ReferenceCost(x)
    return 0.5 * (x[1:4] - x_ref)' * Q * (x[1:4] - x_ref) 
end

function Polyhedral_1(x, lambda, idx1 = 2, idx2 = 1, L = 1)

    if idx1 == 1
        t1 = t_1(x)
    elseif idx1 == 2
        t1 = t_2(x)
    end

    if idx2 == 3
        t2 = t_3(x)
    elseif idx2 == 4
        t2 = t_4(x)
    end

    R1 = R_1(x[idx1])
    R2 = R_2(x[idx2])

    if idx1 == 2 && idx2 == 3
        return lambda[1]*((-H*R1*t1 - h)'* x[9:12] + (-F*R2*t2 - g)'*x[13:16])
    elseif idx1 == 2 && idx2 == 4
        if L == 1
            return lambda[11]*((-H*R1*t1 - h)'* x[17:20] + (-F*R2*t2 - g)'*x[21:24])
        elseif L == 2
            return lambda[21]*((-H*R1*t1 - h)'* x[17:20] + (-F*R2*t2 - g)'*x[21:24])
        end
    elseif idx1 == 1 && idx2 == 4 
        return lambda[31]*((-H*R1*t1 - h)'* x[25:28] + (-F*R2*t2 - g)'*x[29:32])
    
    end 
end 

function Polyhedral_1_Aug(x, lambda, idx1 = 2, idx2 = 1)

    if idx1 == 1
        t1 = t_1(x)
    elseif idx1 == 2
        t1 = t_2(x)
    end

    if idx2 == 3
        t2 = t_3(x)
    elseif idx2 == 4
        t2 = t_4(x)
    end

    R1 = R_1(x[idx1])
    R2 = R_2(x[idx2])

    if idx1 == 2 && idx2 == 3 
        return lambda[1]*((-H*R1*t1 - h)'* x[7:10] + (-F*R2*t2 - g)'*x[19:22])
    elseif idx1 == 2 && idx2 == 4
        return lambda[13]*((-H*R1*t1 - h)'* x[11:14] + (-F*R2*t2 - g)'*x[23:26])
    elseif idx1 == 1 && idx2 == 4 
        return lambda[25]*((-H*R1*t1 - h)'* x[15:18] + (-F*R2*t2 - g)'*x[27:30])
    end 
end 

function Polyhedral_2(x, lambda ,idx1 = 2, idx2 = 1, L = 1)

    if idx1 == 2 && idx2 == 3 
        return dot(lambda[2:5], x[9:12]) + dot(lambda[6:9],x[13:16])
    elseif idx1 == 2 && idx2 == 4
        if L == 1
            return dot(lambda[12:15], x[17:20]) + dot(lambda[16:19],x[21:24])
        elseif L == 2
            return dot(lambda[22:25], x[17:20]) + dot(lambda[26:29],x[21:24])
        end
    elseif idx1 == 1 && idx2 == 4
        return dot(lambda[32:35],x[25:28]) + dot(lambda[36:39],x[29:32])
    end 
end

function Polyhedral_2_Aug(x, lambda ,idx1 = 2, idx2 = 1)

    if idx1 == 2 && idx2 == 3 
        return dot(lambda[2:5], x[7:10]) + dot(lambda[6:9],x[19:22])
    elseif idx1 == 2 && idx2 == 4
            return dot(lambda[14:17], x[11:14]) + dot(lambda[18:21],x[23:26])
    elseif idx1 == 1 && idx2 == 4
        return dot(lambda[26:29],x[15:18]) + dot(lambda[30:33],x[27:30])
    end 
end

function Polyhedral_3(x, lambda, idx1 = 2, idx2 = 1, L = 1)

    if idx1 == 1
        t1 = t_1(x)
    elseif idx1 == 2
        t1 = t_2(x)
    end

    if idx2 == 3
        t2 = t_3(x)
    elseif idx2 == 4
        t2 = t_4(x)
    end

    R2 = R_2(x[idx2])

    if idx1 == 2 && idx2 == 3 
        return lambda[10]*(norm(R2'* F'*x[13:16]))
    elseif idx1 == 2 && idx2 == 4
        if L == 1
            return lambda[20]*(norm(R2'* F'*x[21:24]))
        elseif L == 2
            return lambda[30]*(norm(R2'* F'*x[21:24]))
        end
    elseif idx1 == 1 && idx2 == 4
        return lambda[40]*(norm(R2'* F'*x[29:32]))
    end
end

function Polyhedral_3_Aug(x, lambda, idx1 = 2, idx2 = 1)

    if idx1 == 1
        t1 = t_1(x)
    elseif idx1 == 2
        t1 = t_2(x)
    end

    if idx2 == 3
        t2 = t_3(x)
    elseif idx2 == 4
        t2 = t_4(x)
    end

    R2 = R_2(x[idx2])

    if idx1 == 2 && idx2 == 3 
        return lambda[12]*(norm(R2'* F'*x[19:22]))
    elseif idx1 == 2 && idx2 == 4
            return lambda[24]*(norm(R2'* F'*x[23:26]))
    elseif idx1 == 1 && idx2 == 4
        return lambda[36]*(norm(R2'* F'*x[27:30]))
    end
end


function Polyhedral_4(x, mu, idx1 = 2, idx2 = 1, L = 1)
    
    if idx1 == 1
        t1 = t_1(x)
    elseif idx1 == 2
        t1 = t_2(x)
    end

    if idx2 == 3
        t2 = t_3(x)
    elseif idx2 == 4
        t2 = t_4(x)
    end

    R1 = R_1(x[idx1])
    R2 = R_2(x[idx2])

    if idx1 == 2 && idx2 == 3
        return mu[5]*(R1'*H'*x[9:12] +R2'* F'*x[13:16])[1]
    elseif idx1 == 2 && idx2 == 4
        if L == 1
            return mu[7]*(R1'*H'*x[17:20] + R2'* F'*x[21:24])[1]
        elseif L == 2
            return mu[13]*(R1'*H'*x[17:20] + R2'* F'*x[21:24])[1]
        end
    elseif idx1 == 1 && idx2 == 4
        return mu[15]*(R1'*H'*x[25:28] + R2'* F'*x[29:32])[1]
    end
end

function Polyhedral_4_Aug(x, lambda, idx1 = 2, idx2 = 1)
    
    if idx1 == 1
        t1 = t_1(x)
    elseif idx1 == 2
        t1 = t_2(x)
    end

    if idx2 == 3
        t2 = t_3(x)
    elseif idx2 == 4
        t2 = t_4(x)
    end

    R1 = R_1(x[idx1])
    R2 = R_2(x[idx2])

    if idx1 == 2 && idx2 == 3
        return lambda[10]*(R1'*H'*x[7:10] +R2'* F'*x[19:22])[1]
    elseif idx1 == 2 && idx2 == 4
        return lambda[22]*(R1'*H'*x[11:14] + R2'* F'*x[23:26])[1]
    elseif idx1 == 1 && idx2 == 4
        return lambda[34]*(R1'*H'*x[15:18] + R2'* F'*x[27:30])[1]
    end
end

function Polyhedral_5(x, mu, idx1 = 2, idx2 = 1, L = 1)

    R1 = R_1(x[idx1])
    R2 = R_2(x[idx2])

    if idx1 == 2 && idx2 == 3
        return mu[6]*(R1'*H'*x[9:12] +R2'* F'*x[13:16])[2]
    elseif idx1 == 2 && idx2 == 4
        if L == 1
            return mu[8]*(R1'*H'*x[17:20] + R2'* F'*x[21:24])[2]
        elseif L == 2
            return mu[14]*(R1'*H'*x[17:20] + R2'* F'*x[21:24])[2]
        end
    elseif idx1 == 1 && idx2 == 4
        return mu[16]*(R1'*H'*x[25:28] + R2'* F'*x[29:32])[2]
    end
end

function Polyhedral_5_Aug(x, lambda, idx1 = 2, idx2 = 1)
    
    if idx1 == 1
        t1 = t_1(x)
    elseif idx1 == 2
        t1 = t_2(x)
    end

    if idx2 == 3
        t2 = t_3(x)
    elseif idx2 == 4
        t2 = t_4(x)
    end

    R1 = R_1(x[idx1])
    R2 = R_2(x[idx2])

    if idx1 == 2 && idx2 == 3
        return lambda[11]*(R1'*H'*x[7:10] +R2'* F'*x[19:22])[2]
    elseif idx1 == 2 && idx2 == 4
        return lambda[23]*(R1'*H'*x[11:14] + R2'* F'*x[23:26])[2]
    elseif idx1 == 1 && idx2 == 4
        return lambda[35]*(R1'*H'*x[15:18] + R2'* F'*x[27:30])[2]
    end
end

function State_Transition(x, mu, x_prev, dt, L=1)
    if L == 1
        return dot(mu[1:4],(x[1:4] - x_prev[1:4] - dt * x[5:8]))
    else
        return dot(mu[9:12],(x[1:4] - x_prev[1:4] - dt * x[5:8]))
    end
end

# x = rand(20,32) * 2 * pi
x_0 = rand(1,32) * 2 * pi

function TotalGradient(x, x_0, rob_idx = 1, finite_difference = false, dt = 0.1, horizon = 20)

    ∇TotalCost = zeros(32,1)
    if rob_idx == 1
        idx1 = 2
        constrained_links = [3, 4]
    else
        idx2 = 4
        constrained_links = [2, 1]
    end

    if finite_difference == false
        for i = 1:horizon
            ∇TotalCost += ForwardDiff.gradient(x -> InputCost(x, rob_idx), x[i,:]) 
            ∇TotalCost += ForwardDiff.gradient(x -> ReferenceCost(x), x[i,:])
            if rob_idx == 1
                for idx2 in constrained_links
                    ∇TotalCost += ForwardDiff.gradient(x -> Polyhedral_1(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += ForwardDiff.gradient(x -> Polyhedral_2(x, lambda[i,:], idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += ForwardDiff.gradient(x -> Polyhedral_3(x, lambda[i,:], idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += ForwardDiff.gradient(x -> Polyhedral_4(x, mu[i,:], idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += ForwardDiff.gradient(x -> Polyhedral_5(x, mu[i,:] ,idx1, idx2, rob_idx), x[i,:])
                end 
            end
        end
    else    
        for i = 1:horizon
            ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> InputCost(x, rob_idx), x[i,:]) 
            ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> ReferenceCost(x), x[i,:])
            if i == 1
                ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> State_Transition(x, mu[i,:], x_0, dt, rob_idx), x[i,:])
            else
                x_prev = x[i-1,:]
                ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> State_Transition(x, mu[i,:], x_prev, dt, rob_idx), x[i,:])
            end
            if rob_idx == 1
                for idx2 in constrained_links
                    ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> Polyhedral_1(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> Polyhedral_2(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> Polyhedral_3(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> Polyhedral_4(x, mu[i,:], idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> Polyhedral_5(x, mu[i,:], idx1, idx2, rob_idx), x[i,:])
                end
            else
                for idx1 in constrained_links
                    ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> Polyhedral_1(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> Polyhedral_2(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> Polyhedral_3(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> Polyhedral_4(x, mu[i,:], idx1, idx2, rob_idx), x[i,:])
                    ∇TotalCost += FiniteDiff.finite_difference_gradient(x -> Polyhedral_5(x, mu[i,:], idx1, idx2, rob_idx), x[i,:])
                end
            end
        end
    end

    return ∇TotalCost
end

function TotalHessian(x, x_0, rob_idx = 1, finite_difference = false, dt = 0.1, horizon = 20)
    ∇∇TotalCost = zeros(32,32)
    if rob_idx == 1
        idx1 = 2
        constrained_links = [3, 4]
    else
        idx2 = 4
        constrained_links = [2, 1]
    end
    if finite_difference == false
        for i = 1:horizon
            ∇∇TotalCost += ForwardDiff.hessian(x -> InputCost(x, rob_idx), x[i,:]) 
            ∇∇TotalCost += ForwardDiff.hessian(x -> ReferenceCost(x), x[i,:])
            if rob_idx == 1
                for idx2 in constrained_links
                    ∇∇TotalCost += ForwardDiff.hessian(x -> Polyhedral_1(x, idx1, idx2), x[i,:])
                    ∇∇TotalCost += ForwardDiff.hessian(x -> Polyhedral_2(x, idx1, idx2), x[i,:])
                    ∇∇TotalCost += ForwardDiff.hessian(x -> Polyhedral_3(x, idx1, idx2), x[i,:])
                    ∇∇TotalCost += ForwardDiff.hessian(x -> Polyhedral_4(x, idx1, idx2), x[i,:])
                    ∇∇TotalCost += ForwardDiff.hessian(x -> Polyhedral_5(x, idx1, idx2), x[i,:])
                end 
            end
        end
    else
        for i = 1:horizon
            ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> InputCost(x, rob_idx), x[i,:]) 
            ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> ReferenceCost(x), x[i,:])
            if i == 1
                ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> State_Transition(x, mu[i,:], x_0, dt, rob_idx), x[i,:])
            else
                x_prev = x[i-1,:]
                ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> State_Transition(x, mu[i,:], x_prev, dt, rob_idx), x[i,:])
            end
            if rob_idx == 1
                for idx2 in constrained_links
                    ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> Polyhedral_1(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> Polyhedral_2(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> Polyhedral_3(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> Polyhedral_4(x, mu[i,:], idx1, idx2, rob_idx), x[i,:])
                    ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> Polyhedral_5(x, mu[i,:], idx1, idx2, rob_idx), x[i,:])
                end
            else
                for idx1 in constrained_links
                    ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> Polyhedral_1(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> Polyhedral_2(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> Polyhedral_3(x, lambda[i,:] ,idx1, idx2, rob_idx), x[i,:])
                    ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> Polyhedral_4(x, mu[i,:], idx1, idx2, rob_idx), x[i,:])
                    ∇∇TotalCost += FiniteDiff.finite_difference_hessian(x -> Polyhedral_5(x, mu[i,:], idx1, idx2, rob_idx), x[i,:])
                end
            end
        end
    end
    return ∇∇TotalCost
end



# @variables x[1:32]

# Define the function Polyhedral_1_Aug symbolically

Polyhedral_1_Aug_sym(x, lambda, idx1, idx2) = begin
    if idx1 == 1
        t1 = t_1(x)
    elseif idx1 == 2
        t1 = t_2(x)
    end

    if idx2 == 3
        t2 = t_3(x)
    elseif idx2 == 4
        t2 = t_4(x)
    end

    R1 = R_1(x[idx1])
    R2 = R_2(x[idx2])

    if idx1 == 2 && idx2 == 3 
        return lambda[1]*((-H*R1*t1 - h)'* x[7:10] + (-F*R2*t2 - g)'*x[19:22])
    elseif idx1 == 2 && idx2 == 4
        return lambda[13]*((-H*R1*t1 - h)'* x[11:14] + (-F*R2*t2 - g)'*x[23:26])
    elseif idx1 == 1 && idx2 == 4 
        return lambda[25]*((-H*R1*t1 - h)'* x[15:18] + (-F*R2*t2 - g)'*x[27:30])
    end 
end

# y = [x...]

# gradient_sym = Symbolics.gradient(Polyhedral_1_Aug_sym(x, lambda[1,:], 2, 3), y, simplify=true)

total_hess = TotalHessian(x, x_0, 1.0, true, 0.1, 20)