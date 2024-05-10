
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
        return lambda[10]*(symbolic_norm(R2'* F'*x[21:24]) + symbolic_norm(R2'* F'*x[45:48]))
    elseif idx1 == 2 && idx2 == 4
         return lambda[22]*(symbolic_norm(R2'* F'*x[25:28]) + symbolic_norm(R2'* F'*x[49:52]))
    elseif idx1 == 1 && idx2 == 4
        return lambda[34]*(symbolic_norm(R2'* F'*x[29:32]) + symbolic_norm(R2'* F'*x[53:56]))
    end
end


Polyhedral_4(x, lambda, idx1 = 2, idx2 = 1) = begin

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
        return dot(lambda[11:12],(R1'*H'*x[9:12] +R2'* F'*x[21:24]) + (R1'*H'*x[33:36] +R2'* F'*x[45:48]))
    elseif idx1 == 2 && idx2 == 4
        return dot(lambda[23:24],(R1'*H'*x[13:16] +R2'* F'*x[25:28]) + (R1'*H'*x[37:40] +R2'* F'*x[49:52]))
    elseif idx1 == 1 && idx2 == 4
        return dot(lambda[35:36],(R1'*H'*x[17:20] +R2'* F'*x[29:32]) + (R1'*H'*x[41:44] +R2'* F'*x[53:56]))
    end
end


y = [x...]

y1 = [y[1:6]; y[9:32]]
y2 = [y[1:4]; y[7:8]; y[33:56]]

ref_grad_1 = Symbolics.gradient(ReferenceCost(y), y1, simplify=true)
ref_grad_2 = Symbolics.gradient(ReferenceCost(y), y2, simplify=true)

input_grad_1 = Symbolics.gradient(InputCost(y), y1, simplify=true)
input_grad_2 = Symbolics.gradient(InputCost(y), y2, simplify=true)

poly_1_grad_1 = Symbolics.gradient(Polyhedral_1(y, lambda[1,:], 2, 3), y1, simplify=true)
poly_1_grad_2 = Symbolics.gradient(Polyhedral_1(y, lambda[1,:], 2, 3), y2, simplify=true)

poly_2_grad_1 = Symbolics.gradient(Polyhedral_2(y, lambda[1,:], 2, 3), y1, simplify=true)
poly_2_grad_2 = Symbolics.gradient(Polyhedral_2(y, lambda[1,:], 2, 3), y2, simplify=true)

poly_3_grad_1 = Symbolics.gradient(Polyhedral_3(y, lambda[1,:], 2, 3), y1, simplify=true)
poly_3_grad_2 = Symbolics.gradient(Polyhedral_3(y, lambda[1,:], 2, 3), y2, simplify=true)

poly_4_grad_1 = Symbolics.gradient(Polyhedral_4(y, lambda[1,:], 2, 3), y1, simplify=true)
poly_4_grad_2 = Symbolics.gradient(Polyhedral_4(y, lambda[1,:], 2, 3), y2, simplify=true)

total_grad_1 = ref_grad_1 + input_grad_1 + poly_1_grad_1 + poly_2_grad_1 + poly_3_grad_1 + poly_4_grad_1
total_grad_2 = ref_grad_2 + input_grad_2 + poly_1_grad_2 + poly_2_grad_2 + poly_3_grad_2 + poly_4_grad_2



x_traj = rand(1,64)

substitutions_y1 = Dict(y1[i] => x_traj[i] for i in 1:length(y1))
substitutions_y2 = Dict(y2[i] => x_traj[i] for i in 1:length(y2))

ref_grad_1 = [substitute(expr, substitutions_y1) for expr in ref_grad_1]
ref_grad_2 = [substitute(expr, substitutions_y2) for expr in ref_grad_2]

input_grad_1 = [substitute(expr, substitutions_y1) for expr in input_grad_1]
input_grad_2 = [substitute(expr, substitutions_y2) for expr in input_grad_2]

poly_1_grad_1 = [substitute(expr, substitutions_y1) for expr in poly_1_grad_1]
poly_1_grad_2 = [substitute(expr, substitutions_y2) for expr in poly_1_grad_2]

poly_2_grad_1 = [substitute(expr, substitutions_y1) for expr in poly_2_grad_1]
poly_2_grad_2 = [substitute(expr, substitutions_y2) for expr in poly_2_grad_2]

poly_3_grad_1 = [substitute(expr, substitutions_y1) for expr in poly_3_grad_1]
poly_3_grad_2 = [substitute(expr, substitutions_y2) for expr in poly_3_grad_2]

poly_4_grad_1 = [substitute(expr, substitutions_y1) for expr in poly_4_grad_1]
poly_4_grad_2 = [substitute(expr, substitutions_y2) for expr in poly_4_grad_2]

total_grad_1 = [substitute(expr, substitutions_y1) for expr in total_grad_1]
total_grad_2 = [substitute(expr, substitutions_y2) for expr in total_grad_2]









