
using ForwardDiff
using LinearAlgebra
using FiniteDiff
using Symbolics
using StaticArrays
using BlockDiagonals

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

const lambda = rand(20,36)
const mu = rand(20,8)

t_1(x) = [l1 * cos(x[1]); l1 * sin(x[1])]

t_2(x) = [l1 * cos(x[1]) + l2 * cos(x[2]); l1 * sin(x[1]) + l2 * sin(x[2])]

t_3(x) = [l3 * cos(x[3]); d - l3 * sin(x[3])]

t_4(x) = [l3 * cos(x[3]) + l4 * cos(x[4]); d - l3 * sin(x[3]) - l4 * sin(x[4])]

R_1(x) = [cos(x) -sin(x); sin(x) cos(x)]
R_2(x) = [cos(x) sin(x); -sin(x) cos(x)]

function symbolic_norm(v)
    return sqrt(sum([vi^2 for vi in v]))
end

@variables x[1:64]

ReferenceCost(x) = begin
    return 0.5 * (x[1:4] - x_ref)' * Q * (x[1:4] - x_ref) 
end

InputCost(x) = begin
    return 0.5 * x[5:6]' * R * x[5:6] + 0.5 * x[7:8]' * R * x[7:8]
end

Polyhedral_1(x, lambda, idx1, idx2) = begin

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
        return lambda[1]*((-H*R1*t1 - h)'* x[9:12] + (-F*R2*t2 - g)'*x[21:24] + (-H*R1*t1 - h)'* x[33:36] + (-F*R2*t2 - g)'*x[45:48])
    elseif idx1 == 2 && idx2 == 4
        return lambda[13]*((-H*R1*t1 - h)'* x[13:16] + (-F*R2*t2 - g)'*x[25:28] + (-H*R1*t1 - h)'* x[37:40] + (-F*R2*t2 - g)'*x[49:52])
    elseif idx1 == 1 && idx2 == 4 
        return lambda[25]*((-H*R1*t1 - h)'* x[17:20] + (-F*R2*t2 - g)'*x[29:32] + (-H*R1*t1 - h)'* x[41:44] + (-F*R2*t2 - g)'*x[53:56])
    end 
end

Polyhedral_2(x, lambda ,idx1 = 2, idx2 = 1) =  begin
    if idx1 == 2 && idx2 == 3 
        return dot(lambda[2:5], x[9:12]) + dot(lambda[6:9],x[21:24]) + dot(lambda[2:5], x[33:36]) + dot(lambda[6:9],x[45:48])
    elseif idx1 == 2 && idx2 == 4
        return dot(lambda[14:17], x[13:16]) + dot(lambda[18:21],x[21:24]) + dot(lambda[14:17], x[37:40]) + dot(lambda[18:21],x[49:52])
    elseif idx1 == 1 && idx2 == 4
        return dot(lambda[26:29],x[17:20]) + dot(lambda[30:33],x[29:32]) + dot(lambda[26:29], x[41:44]) + dot(lambda[30:33],x[53:56])
    end 
end

Polyhedral_3(x, lambda, idx1 = 2, idx2 = 1) = begin 

    R2T = [cos(x[idx2]) -sin(x[idx2]); sin(x[idx2]) cos(x[idx2])]

    if idx1 == 2 && idx2 == 3 
        return lambda[10]*(symbolic_norm(R2T* F'*x[21:24]) + symbolic_norm(R2T* F'*x[45:48]))
    elseif idx1 == 2 && idx2 == 4
         return lambda[22]*(symbolic_norm(R2T* F'*x[25:28]) + symbolic_norm(R2T* F'*x[49:52]))
    elseif idx1 == 1 && idx2 == 4
        return lambda[34]*(symbolic_norm(R2T* F'*x[29:32]) + symbolic_norm(R2T* F'*x[53:56]))
    end
end


Polyhedral_4(x, lambda, idx1 = 2, idx2 = 1) = begin

    R1T = [cos(x[idx1]) sin(x[idx1]); -sin(x[idx1]) cos(x[idx1])]
    R2T = [cos(x[idx2]) -sin(x[idx2]); sin(x[idx2]) cos(x[idx2])]

    if idx1 == 2 && idx2 == 3
        return dot(lambda[11:12],(R1T*H'*x[9:12] +R2T* F'*x[21:24]) + (R1T*H'*x[33:36] +R2T* F'*x[45:48]))
    elseif idx1 == 2 && idx2 == 4
        return dot(lambda[23:24],(R1T*H'*x[13:16] +R2T* F'*x[25:28]) + (R1T*H'*x[37:40] +R2T* F'*x[49:52]))
    elseif idx1 == 1 && idx2 == 4
        return dot(lambda[35:36],(R1T*H'*x[17:20] +R2T* F'*x[29:32]) + (R1T*H'*x[41:44] +R2T* F'*x[53:56]))
    end
end


function State_Transition(x, x_prev, dt)
    return dot(x[57:60],(x[1:4] - x_prev[1:4] - dt * x[5:8])) + dot(x[61:64],(x[1:4] - x_prev[1:4] - dt * x[5:8]))
end

y = [x...]

x1 = [y[1:6]; y[9:32]]
x2 = [y[1:4]; y[7:8]; y[33:56]]



ref_grad_1 = Symbolics.gradient(ReferenceCost(x), x1, simplify=true)
ref_grad_2 = Symbolics.gradient(ReferenceCost(x), x2, simplify=true)

input_grad_1 = Symbolics.gradient(InputCost(x), x1, simplify=true)
input_grad_2 = Symbolics.gradient(InputCost(x), x2, simplify=true)

poly_1_grad_1_23 = Symbolics.gradient(Polyhedral_1(x, lambda[1,:], 2, 3), x1, simplify=true)
poly_1_grad_2_23 = Symbolics.gradient(Polyhedral_1(x, lambda[1,:], 2, 3), x2, simplify=true)
poly_1_grad_1_24 = Symbolics.gradient(Polyhedral_1(x, lambda[1,:], 2, 4), x1, simplify=true)
poly_1_grad_2_24 = Symbolics.gradient(Polyhedral_1(x, lambda[1,:], 2, 4), x2, simplify=true)
poly_1_grad_1_14 = Symbolics.gradient(Polyhedral_1(x, lambda[1,:], 1, 4), x1, simplify=true)
poly_1_grad_2_14 = Symbolics.gradient(Polyhedral_1(x, lambda[1,:], 1, 4), x2, simplify=true)


poly_2_grad_1_23 = Symbolics.gradient(Polyhedral_2(x, lambda[1,:], 2, 3), x1, simplify=true)
poly_2_grad_2_23 = Symbolics.gradient(Polyhedral_2(x, lambda[1,:], 2, 3), x2, simplify=true)
poly_2_grad_1_24 = Symbolics.gradient(Polyhedral_2(x, lambda[1,:], 2, 4), x1, simplify=true)
poly_2_grad_2_24 = Symbolics.gradient(Polyhedral_2(x, lambda[1,:], 2, 4), x2, simplify=true)
poly_2_grad_1_14 = Symbolics.gradient(Polyhedral_2(x, lambda[1,:], 1, 4), x1, simplify=true)
poly_2_grad_2_14 = Symbolics.gradient(Polyhedral_2(x, lambda[1,:], 1, 4), x2, simplify=true)


poly_3_grad_1_23 = Symbolics.gradient(Polyhedral_3(x, lambda[1,:], 2, 3), x1, simplify=true)
poly_3_grad_2_23= Symbolics.gradient(Polyhedral_3(x, lambda[1,:], 2, 3), x2, simplify=true)
poly_3_grad_1_24 = Symbolics.gradient(Polyhedral_3(x, lambda[1,:], 2, 4), x1, simplify=true)
poly_3_grad_2_24 = Symbolics.gradient(Polyhedral_3(x, lambda[1,:], 2, 4), x2, simplify=true)
poly_3_grad_1_14 = Symbolics.gradient(Polyhedral_3(x, lambda[1,:], 1, 4), x1, simplify=true)
poly_3_grad_2_14 = Symbolics.gradient(Polyhedral_3(x, lambda[1,:], 1, 4), x2, simplify=true)


poly_4_grad_1_23 = Symbolics.gradient(Polyhedral_4(x, lambda[1,:], 2, 3), x1, simplify=true)
poly_4_grad_2_23 = Symbolics.gradient(Polyhedral_4(x, lambda[1,:], 2, 3), x2, simplify=true)
poly_4_grad_1_24 = Symbolics.gradient(Polyhedral_4(x, lambda[1,:], 2, 4), x1, simplify=true)
poly_4_grad_2_24 = Symbolics.gradient(Polyhedral_4(x, lambda[1,:], 2, 4), x2, simplify=true)
poly_4_grad_1_14 = Symbolics.gradient(Polyhedral_4(x, lambda[1,:], 1, 4), x1, simplify=true)
poly_4_grad_2_14 = Symbolics.gradient(Polyhedral_4(x, lambda[1,:], 1, 4), x2, simplify=true)


x_traj = rand(20, 64)
x_prev = rand(64)
dt = 0.1
x1_traj = [x_traj[1, 1:6]; x_traj[1, 9:32]]
x2_traj = [x_traj[1, 1:4]; x_traj[1, 7:8]; x_traj[1, 33:56]]


state_transition_1 = Symbolics.gradient(State_Transition(x, x_prev, dt), x1, simplify=true)
state_transition_2 = Symbolics.gradient(State_Transition(x, x_prev, dt), x2, simplify=true)

total_grad_1 = (ref_grad_1 + input_grad_1 + poly_1_grad_1_23 + poly_1_grad_1_24 + poly_1_grad_1_14 +
                 poly_2_grad_1_23 + poly_2_grad_1_24 + poly_2_grad_1_14 + poly_3_grad_1_23 + 
                 poly_3_grad_1_24 + poly_3_grad_1_14 + poly_4_grad_1_23 + poly_4_grad_1_24 + poly_4_grad_1_14 + state_transition_1)

total_grad_2 = (ref_grad_2 + input_grad_2 + poly_1_grad_2_23 + poly_1_grad_2_24 + poly_1_grad_2_14 + 
                poly_2_grad_2_23 + poly_2_grad_2_24 + poly_2_grad_2_14 + poly_3_grad_2_23 + poly_3_grad_2_24 + 
                poly_3_grad_2_14 + poly_4_grad_2_23 + poly_4_grad_2_24 + poly_4_grad_2_14)


x1_vals = Dict(x[i] => x1_traj[i] for i in 1:30)   
x2_vals = Dict(x[i] => x2_traj[i] for i in 1:30)

total_grad_1 = substitute(total_grad_1, (x1_vals))
total_grad_2 = substitute(total_grad_2, (x2_vals))

total_hess_1 = Symbolics.jacobian(total_grad_1, x, simplify=true)
total_hess_2 = Symbolics.jacobian(total_grad_2, x, simplify=true)


function get_H(x_traj)
    H_list1 = []
    H_list2 = []
    for j in 1:20
        x_vals = Dict(x[i] => x_traj[j,i] for i in 1:64)
        V1 = substitute.(total_hess_1, (x_vals,))
        V2 = substitute.(total_hess_2, (x_vals,))
        H1 = reshape(Symbolics.value.(V1), 30, 64)
        H2 = reshape(Symbolics.value.(V2), 30, 64)
        push!(H_list1, H1)
        push!(H_list2, H2)
    end
    H_diag1 = BlockDiagonal([H_list1...])
    H_diag2 = BlockDiagonal([H_list2...])
    H = vcat(H_diag1, H_diag2)
    return H
end

H_a = get_H(x_traj)



