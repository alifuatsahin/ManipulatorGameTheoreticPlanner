using ForwardDiff
using LinearAlgebra
using FiniteDiff
using Symbolics
using StaticArrays
using BlockDiagonals
using LinearSolve

R = 0.1*[1.0 0.0; 0.0 1.0]
Q = 100*I(4)
x_ref = [pi*2/3,pi*2/3, pi/6, pi/6]

const l1 = 6.0
const l2 = 6.0
const l3 = 6.0
const l4 = 6.0
const l = 6.0
const w = 3

const d = 100.0
horizon = 10

const H = [1 0; -1 0; 0 1; 0 -1]
const h = [l/2; l/2; w/2; w/2]

const F = [1 0; -1 0; 0 1; 0 -1]
const g = [l/2; l/2; w/2; w/2]

const lambda = rand(1,36)*0.0001
const dt = 0.25


const θ_init = [3*pi/4, 3*pi/4, pi/4, pi/4]

random_init = rand(60)
x_init = reshape(vcat(θ_init, random_init), 1, 64)
x_init = repeat(x_init, horizon,1)
const x0 = reshape(vcat(θ_init, rand(60)), 1, 64)

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
        return lambda[1]*((H*R1*t1 + h)'* x[9:12] + (F*R2*t2 + g)'*x[21:24] + (H*R1*t1 + h)'* x[33:36] + (F*R2*t2 + g)'*x[45:48])
    elseif idx1 == 2 && idx2 == 4
        return lambda[13]*((H*R1*t1 + h)'* x[13:16] + (F*R2*t2 + g)'*x[25:28] + (H*R1*t1 + h)'* x[37:40] + (F*R2*t2 + g)'*x[49:52])
    elseif idx1 == 1 && idx2 == 4 
        return lambda[25]*((H*R1*t1 + h)'* x[17:20] + (F*R2*t2 + g)'*x[29:32] + (H*R1*t1 + h)'* x[41:44] + (F*R2*t2 + g)'*x[53:56])
    end 
end

Polyhedral_2(x, lambda ,idx1 = 2, idx2 = 1) =  begin
    if idx1 == 2 && idx2 == 3 
        return -(dot(lambda[2:5], x[9:12]) + dot(lambda[6:9],x[21:24]) + dot(lambda[2:5], x[33:36]) + dot(lambda[6:9],x[45:48]))
    elseif idx1 == 2 && idx2 == 4
        return -(dot(lambda[14:17], x[13:16]) + dot(lambda[18:21],x[21:24]) + dot(lambda[14:17], x[37:40]) + dot(lambda[18:21],x[49:52]))
    elseif idx1 == 1 && idx2 == 4
        return -(dot(lambda[26:29],x[17:20]) + dot(lambda[30:33],x[29:32]) + dot(lambda[26:29], x[41:44]) + dot(lambda[30:33],x[53:56]))
    end 
end

Polyhedral_3(x, lambda, idx1 = 2, idx2 = 1) = begin 

    R2T = [cos(x[idx2]) -sin(x[idx2]); sin(x[idx2]) cos(x[idx2])]

    if idx1 == 2 && idx2 == 3 
        return lambda[10]*(symbolic_norm(R2T* F'*x[21:24]) - 1 + symbolic_norm(R2T* F'*x[45:48]) - 1)
    elseif idx1 == 2 && idx2 == 4
         return lambda[22]*(symbolic_norm(R2T* F'*x[25:28])  - 1 + symbolic_norm(R2T* F'*x[49:52]) - 1)
    elseif idx1 == 1 && idx2 == 4
        return lambda[34]*(symbolic_norm(R2T* F'*x[29:32]) - 1 + symbolic_norm(R2T* F'*x[53:56]) - 1)
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

@variables x_state[1:horizon, 1:64]

x_state_y = [x_state'...]

State_Transition_Mu(x_state_flat) = begin
    D = 0
    x_state = reshape(x_state_flat,64,horizon)'
    if horizon != 1
        for i in 1:horizon
            if i == 1
                D += dot(x_state[i, 57:60],(x_state[i, 1:4] - x0[1, 1:4])) + dot(x_state[i, 61:64],(x_state[i, 1:4] - x0[1, 1:4]))
            else
                D += dot(x_state[i, 57:60],(x_state[i, 1:4] - x_state[i-1, 1:4] - dt * x_state[i, 5:8])) + dot(x_state[i, 61:64],(x_state[i, 1:4] - x_state[i-1, 1:4] - dt * x_state[i, 5:8]))
            end
        end 
    else
        D = dot(x_state[57:60],(x_state[1, 1:4] - x0[1:4])) + dot(x_state[61:64],(x_state[1, 1:4] - x0[1:4]))
    end 

    return D
end

@variables D[1:4*horizon]

State_Transition(x_state_flat) = begin
    x_state = reshape(x_state_flat, 64, horizon)'
    idx = 1
    if horizon != 1
        for i in 1:horizon
            if i ==1
                for j in 1:4
                    D[idx] = x_state[i, j] - x0[1, j] - dt * x_state[i, j+4]
                    idx += 1
                end
            else
                for j in 1:4
                    D[idx] = x_state[i, j] - x_state[i-1, j] - dt * x_state[i, j+4]
                    idx += 1
                end
            end
        end 
    else
        for j in 1:4
            D[idx] = x_state[1, j] - x0[1, j] - dt * x_state[1, j+4]
            idx += 1
        end
    end
    return D
end

State_Transition_Scalar(x_state_flat) = begin
    D = zeros(4*horizon,1)
    x_state = reshape(x_state_flat, 64, horizon)'
    idx = 1
    if horizon != 1
        for i in 1:horizon
            if i ==1
                for j in 1:4
                    D[idx] = x_state[i, j] - x0[1, j] - dt * x_state[i, j+4]
                    idx += 1
                end
            else
                for j in 1:4
                    D[idx] = x_state[i, j] - x_state[i-1, j] - dt * x_state[i, j+4]
                    idx += 1
                end
            end
        end 
    else
        for j in 1:4
            D[idx] = x_state[1, j] - x0[1, j] - dt * x_state[1, j+4]
            idx += 1
        end
    end
    return D
end

x_1_state = vcat([x_state[i, vcat(1:6, 9:32)]' for i in 1:size(x_state, 1)]...)
x_2_state = vcat([x_state[i, vcat(1:4, 7:8, 33:56)]' for i in 1:size(x_state, 1)]...)

x1state_y = [x_1_state'...]
x2state_y = [x_2_state'...]

y = [x'...]

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


state_transition_1 = Symbolics.gradient(State_Transition_Mu(x_state_y), x1state_y, simplify=true)
state_transition_2 = Symbolics.gradient(State_Transition_Mu(x_state_y), x2state_y, simplify=true)

state_transition_1_hess = Symbolics.jacobian(state_transition_1, x_state_y, simplify=true)
state_transition_2_hess = Symbolics.jacobian(state_transition_2, x_state_y, simplify=true)


total_grad_1 = (ref_grad_1 + input_grad_1 + poly_1_grad_1_23 + poly_1_grad_1_24 + poly_1_grad_1_14 +
                 poly_2_grad_1_23 + poly_2_grad_1_24 + poly_2_grad_1_14 + poly_3_grad_1_23 + 
                 poly_3_grad_1_24 + poly_3_grad_1_14 + poly_4_grad_1_23 + poly_4_grad_1_24 + poly_4_grad_1_14)


total_grad_2 = (ref_grad_2 + input_grad_2 + poly_1_grad_2_23 + poly_1_grad_2_24 + poly_1_grad_2_14 + 
                poly_2_grad_2_23 + poly_2_grad_2_24 + poly_2_grad_2_14 + poly_3_grad_2_23 + poly_3_grad_2_24 + 
                poly_3_grad_2_14 + poly_4_grad_2_23 + poly_4_grad_2_24 + poly_4_grad_2_14)
          


total_hess_1 = Symbolics.jacobian(total_grad_1, x, simplify=true)
total_hess_2 = Symbolics.jacobian(total_grad_2, x, simplify=true)

state_transition_inside = Symbolics.jacobian(State_Transition(x_state_y), x_state_y, simplify=true)

function get_H(x_traj)
    H_list1 = []
    H_list2 = []
    H1 = H2 = []
    x_flat = [x_traj'...]
    x_state_flat = [x_state'...]
    x_state_vals = Dict(x_state_flat[i] => x_flat[i] for i in 1:64*horizon)
    S1 = substitute.(state_transition_1_hess, (x_state_vals,))
    S2 = substitute.(state_transition_2_hess, (x_state_vals,))
    ST_wo_mu = substitute.(state_transition_inside, (x_state_vals,))
    for j in 1:horizon
        x_vals = Dict(x[i] => x_traj[j,i] for i in 1:64)
        V1 = substitute.(total_hess_1, (x_vals,))
        V2 = substitute.(total_hess_2, (x_vals,))
        H1 = reshape(Symbolics.value.(V1), 30, 64)
        H2 = reshape(Symbolics.value.(V2), 30, 64)
        push!(H_list1, H1)
        push!(H_list2, H2)
    end
    H_a = []
    for i in 1:horizon
        start_idx = 30*(i-1) + 1
        end_idx = 30*i

        start_idx_2 = 64*(i-1) + 1
        end_idx_2 = 64*i

        start_idx_3 = 4*(i-1) + 1
        end_idx__3 = 4*i
        push!(H_a, vcat(H_list1[i] + Symbolics.value.(S1[start_idx:end_idx, start_idx_2:end_idx_2]), 
                        H_list2[i] + Symbolics.value.(S2[start_idx:end_idx, start_idx_2:end_idx_2]), 
                        Symbolics.value.(ST_wo_mu[start_idx_3:end_idx__3, start_idx_2:end_idx_2])))
    end

    H_a = BlockDiagonal([H_a...])
    H_a = convert(Matrix{Float64}, H_a)

    for i in 1:horizon
        for j in 1:horizon
            if i != j
                new_entry = vcat(Symbolics.value.(S1[30*(i-1)+1:30*i, 64*(j-1)+1:64*j]),
                Symbolics.value.(S2[30*(i-1)+1:30*i, 64*(j-1)+1:64*j]),
                Symbolics.value.(ST_wo_mu[4*(i-1)+1:4*i, 64*(j-1)+1:64*j]))

                new_entry = convert(Matrix{Float64}, new_entry)
                H_a[64*(i-1)+1:64*i, 64*(j-1)+1:64*j] .= new_entry
            end 
        end 
    end

    return convert(Matrix{Float64}, H_a)
end

function get_G(x_traj)
    G_list1 = []
    G_list2 = []
    x_flat = [x_traj'...]
    x_state_flat = [x_state'...]
    x_state_vals = Dict(x_state_flat[i] => x_flat[i] for i in 1:64*horizon)
    S1 = substitute.(state_transition_1, (x_state_vals,))
    S2 = substitute.(state_transition_2, (x_state_vals,))
    ST_wo_mu = State_Transition_Scalar(x_traj)
    for j in 1:horizon
        x_vals = Dict(x[i] => x_traj[j,i] for i in 1:64)
        V1 = substitute.(total_grad_1, (x_vals,))
        V2 = substitute.(total_grad_2, (x_vals,))
        G1 = Symbolics.value.(V1)
        G2 = Symbolics.value.(V2)
        push!(G_list1, G1)
        push!(G_list2, G2)

    end
    G1 = vcat(G_list1...) + Symbolics.value.(S1)
    G2 = vcat(G_list2...) + Symbolics.value.(S2)
    G = []
    for i in 1:30:length(G1)
        start_index = i
        end_index = min(i + 29, length(G1))
        slice_G1 = G1[start_index:end_index]
        slice_G2 = G2[start_index:end_index]
        st_index = 4 * div(i - 1, 30) + 1
        slice_ST_wo_mu = ST_wo_mu[st_index:st_index+3]
        push!(G, vcat(slice_G1, slice_G2, slice_ST_wo_mu))
    end
    return vcat(G...)
end

function inner_loop(x_init)
    x_traj = x_init
    x_prev = x_init
    i = 1
    while true | i < 500
        println("Iteration: ", i)
        i += 1
        H_a = get_H(x_traj)
        G_a = get_G(x_traj)
    
        QR = qr(H_a)
    
        Hinv = pinv(QR.R) * QR.Q'
        
        Δtraj = - Hinv * G_a

        x_prev = x_traj
        x_traj_flat = [x_traj'...]

        # α = line_search(x_traj_flat, G_a, Δtraj)

        α = 1.0
        print("α: ", α)
        println("norm of delta: ", norm(Δtraj))   
        x_traj_flat += α * Δtraj

        x_traj = reshape(x_traj_flat, 64, horizon)'
        G_new = get_G(x_traj)
        println("G_new", norm(G_new))

        if norm(G_new) < 0.01
          break
        end
    end
    return x_traj
end


function line_search(y, G, δy, β=0.01, τ=0.5)
    α = 1
    while α > 1e-4  
        y_new = y + α * δy
        y_new = reshape(y_new, 64, horizon)'
        G_alpha = get_G(y_new)

        if norm(G_alpha, 1) <= (1 - α * β) * norm(G, 1)
            return α
        end
        
        α *= τ
    end
    return α  
end

x_converged = inner_loop(x_init)

get_G(x_converged)



