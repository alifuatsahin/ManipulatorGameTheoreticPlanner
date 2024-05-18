using ForwardDiff
using LinearAlgebra
using FiniteDiff
using Symbolics
using StaticArrays
using BlockDiagonals
using LinearSolve

R = 0.1*I(4)
Q = 10000*I(4)
x_ref = [pi*2/3,pi*2/3, pi/6, pi/6]

const l1 = 6.0
const l2 = 6.0
const l3 = 6.0
const l4 = 6.0
const l = 6.0
const w = 3

const d = 100.0
horizon = 1

const H = [1 0; -1 0; 0 1; 0 -1]
const h = [l/2; l/2; w/2; w/2]

const F = [1 0; -1 0; 0 1; 0 -1]
const g = [l/2; l/2; w/2; w/2]

lambda = rand(1,36)*0.1
dt = 5


const θ_init = [3*pi/4, 3*pi/4, pi/4, pi/4]

random_init = rand(60)*0
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

@variables x[1:horizon, 1:64]

x1 = vcat([x[i, vcat(1:6, 9:32)]' for i in 1:size(x, 1)]...)
x2 = vcat([x[i, vcat(1:4, 7:8, 33:56)]' for i in 1:size(x, 1)]...)

x1y = [x1'...]
x2y = [x2'...]

x_flat_sym = [x'...]

ReferenceCost(x_state_flat) = begin
    x_state = reshape(x_state_flat,64,horizon)'
    r = 0
    for i in 1:horizon
        r += 0.5 * (x_state[i, 1:4] - x_ref)' * Q * (x_state[i, 1:4] - x_ref)
    end
    return r
end

InputCost(x_state_flat) = begin
    x_state = reshape(x_state_flat,64,horizon)'
    u = 0
    for i in 1:horizon
        u += 0.5 * x_state[i, 5:8]' * R * x_state[i, 5:8] 
    end
    return u
end

Polyhedral_1(x_flat, lambda, idx1, idx2) = begin
    x = reshape(x_flat,64,horizon)'
    p1 = 0
    for i in 1:horizon
        if idx1 == 1
            t1 = t_1(x[i,:])
        elseif idx1 == 2
            t1 = t_2(x[i,:])
        end

        if idx2 == 3
            t2 = t_3(x[i,:])
        elseif idx2 == 4
            t2 = t_4(x[i,:])
        end

        R1 = R_1(x[i,idx1])
        R2 = R_2(x[i,idx2])

        if idx1 == 2 && idx2 == 3 
            p1 += lambda[1]*((H*R1*t1 + h)'* x[i,9:12] + (F*R2*t2 + g)'*x[i,21:24] + (H*R1*t1 + h)'* x[i,33:36] + (F*R2*t2 + g)'*x[i,45:48])
        elseif idx1 == 2 && idx2 == 4
            p1 += lambda[13]*((H*R1*t1 + h)'* x[i,13:16] + (F*R2*t2 + g)'*x[i,25:28] + (H*R1*t1 + h)'* x[i,37:40] + (F*R2*t2 + g)'*x[i,49:52])
        elseif idx1 == 1 && idx2 == 4 
            p1 += lambda[25]*((H*R1*t1 + h)'* x[i,17:20] + (F*R2*t2 + g)'*x[i,29:32] + (H*R1*t1 + h)'* x[i,41:44] + (F*R2*t2 + g)'*x[i,53:56])
        end 
    end
    return p1
end

Polyhedral_2(x_flat, lambda ,idx1 = 2, idx2 = 1) =  begin
    x = reshape(x_flat,64,horizon)'
    p2 = 0

    for i in 1:horizon
        if idx1 == 2 && idx2 == 3 
            p2 += -(dot(lambda[2:5], x[i,9:12]) + dot(lambda[6:9],x[i,21:24]) + dot(lambda[2:5], x[i,33:36]) + dot(lambda[6:9],x[i,45:48]))
        elseif idx1 == 2 && idx2 == 4
            p2 += -(dot(lambda[14:17], x[i,13:16]) + dot(lambda[18:21],x[i,21:24]) + dot(lambda[14:17], x[i,37:40]) + dot(lambda[18:21],x[i,49:52]))
        elseif idx1 == 1 && idx2 == 4
            p2 += -(dot(lambda[26:29],x[i,17:20]) + dot(lambda[30:33],x[i,29:32]) + dot(lambda[26:29], x[i,41:44]) + dot(lambda[30:33],x[i,53:56]))
        end 
    end
end

Polyhedral_3(x_flat, lambda, idx1 = 2, idx2 = 1) = begin 
    x = reshape(x_flat,64,horizon)'
    p3 = 0
    for i in 1:horizon
        R2T = [cos(x[i,idx2]) -sin(x[i,idx2]); sin(x[i,idx2]) cos(x[i,idx2])]

        if idx1 == 2 && idx2 == 3 
            p3 += lambda[10]*(symbolic_norm(R2T* F'*x[i,21:24]) - 1 + symbolic_norm(R2T* F'*x[i,45:48]) - 1)
        elseif idx1 == 2 && idx2 == 4
            p3 += lambda[22]*(symbolic_norm(R2T* F'*x[i,25:28])  - 1 + symbolic_norm(R2T* F'*x[i,49:52]) - 1)
        elseif idx1 == 1 && idx2 == 4
            p3 += lambda[34]*(symbolic_norm(R2T* F'*x[i,29:32]) - 1 + symbolic_norm(R2T* F'*x[i,53:56]) - 1)
        end
    end 
end


Polyhedral_4(x_flat, lambda, idx1 = 2, idx2 = 1) = begin
    x = reshape(x_flat,64,horizon)'
    p4 = 0
    for i in 1:horizon
        R1T = [cos(x[i,idx1]) sin(x[i,idx1]); -sin(x[i,idx1]) cos(x[i,idx1])]
        R2T = [cos(x[i,idx2]) -sin(x[i,idx2]); sin(x[i,idx2]) cos(x[i,idx2])]

        if idx1 == 2 && idx2 == 3
            p4 += dot(lambda[11:12],(R1T*H'*x[i,9:12] +R2T* F'*x[i,21:24]) + (R1T*H'*x[i,33:36] +R2T* F'*x[i,45:48]))
        elseif idx1 == 2 && idx2 == 4
            p4 += dot(lambda[23:24],(R1T*H'*x[i,13:16] +R2T* F'*x[i,25:28]) + (R1T*H'*x[i,37:40] +R2T* F'*x[i,49:52]))
        elseif idx1 == 1 && idx2 == 4
            p4 += dot(lambda[35:36],(R1T*H'*x[i,17:20] +R2T* F'*x[i,29:32]) + (R1T*H'*x[i,41:44] +R2T* F'*x[i,53:56]))
        end
    end 
end


State_Transition_Mu(x_flat) = begin
    D = 0
    x_state = reshape(x_flat,64,horizon)'
    if horizon != 1
        for i in 1:horizon
            if i == 1
                D += dot(x_state[i, 57:60],(x_state[i, 1:4] - x0[1, 1:4] - dt * x_state[1, 1:4])) + dot(x_state[i, 61:64],(x_state[i, 1:4] - x0[1, 1:4]- dt * x_state[1, 1:4]))
            else
                D += dot(x_state[i, 57:60],(x_state[i, 1:4] - x_state[i-1, 1:4] - dt * x_state[i, 5:8])) + dot(x_state[i, 61:64],(x_state[i, 1:4] - x_state[i-1, 1:4] - dt * x_state[i, 5:8]))
            end
        end 
    else
        D = dot(x_state[57:60],(x_state[1, 1:4] - x0[1:4] - dt * x_state[1, 1:4])) + dot(x_state[61:64],(x_state[1, 1:4] - x0[1:4] - dt * x_state[1, 1:4]))
    end 

    return D
end

@variables D[1:4*horizon]

State_Transition(x_flat) = begin
    x_state = reshape(x_flat, 64, horizon)'
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

State_Transition_Scalar(x_flat) = begin
    D = zeros(4*horizon,1)
    x_state = reshape(x_flat, 64, horizon)'
    idx = 1
    if horizon != 1
        for i in 1:horizon
            if i == 1
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


ref_grad_1 = Symbolics.gradient(ReferenceCost(x_flat_sym), x1y, simplify=true)
ref_grad_2 = Symbolics.gradient(ReferenceCost(x_flat_sym), x2y, simplify=true)

input_grad_1 = Symbolics.gradient(InputCost(x_flat_sym), x1y, simplify=true)
input_grad_2 = Symbolics.gradient(InputCost(x_flat_sym), x2y, simplify=true)

poly_1_grad_1_23 = Symbolics.gradient(Polyhedral_1(x_flat_sym, lambda[1,:], 2, 3), x1y, simplify=true)
poly_1_grad_2_23 = Symbolics.gradient(Polyhedral_1(x_flat_sym, lambda[1,:], 2, 3), x2y, simplify=true)
poly_1_grad_1_24 = Symbolics.gradient(Polyhedral_1(x_flat_sym, lambda[1,:], 2, 4), x1y, simplify=true)
poly_1_grad_2_24 = Symbolics.gradient(Polyhedral_1(x_flat_sym, lambda[1,:], 2, 4), x2y, simplify=true)
poly_1_grad_1_14 = Symbolics.gradient(Polyhedral_1(x_flat_sym, lambda[1,:], 1, 4), x1y, simplify=true)
poly_1_grad_2_14 = Symbolics.gradient(Polyhedral_1(x_flat_sym, lambda[1,:], 1, 4), x2y, simplify=true)


poly_2_grad_1_23 = Symbolics.gradient(Polyhedral_2(x_flat_sym, lambda[1,:], 2, 3), x1y, simplify=true)
poly_2_grad_2_23 = Symbolics.gradient(Polyhedral_2(x_flat_sym, lambda[1,:], 2, 3), x2y, simplify=true)
poly_2_grad_1_24 = Symbolics.gradient(Polyhedral_2(x_flat_sym, lambda[1,:], 2, 4), x1y, simplify=true)
poly_2_grad_2_24 = Symbolics.gradient(Polyhedral_2(x_flat_sym, lambda[1,:], 2, 4), x2y, simplify=true)
poly_2_grad_1_14 = Symbolics.gradient(Polyhedral_2(x_flat_sym, lambda[1,:], 1, 4), x1y, simplify=true)
poly_2_grad_2_14 = Symbolics.gradient(Polyhedral_2(x_flat_sym, lambda[1,:], 1, 4), x2y, simplify=true)


poly_3_grad_1_23 = Symbolics.gradient(Polyhedral_3(x_flat_sym, lambda[1,:], 2, 3), x1y, simplify=true)
poly_3_grad_2_23 = Symbolics.gradient(Polyhedral_3(x_flat_sym, lambda[1,:], 2, 3), x2y, simplify=true)
poly_3_grad_1_24 = Symbolics.gradient(Polyhedral_3(x_flat_sym, lambda[1,:], 2, 4), x1y, simplify=true)
poly_3_grad_2_24 = Symbolics.gradient(Polyhedral_3(x_flat_sym, lambda[1,:], 2, 4), x2y, simplify=true)
poly_3_grad_1_14 = Symbolics.gradient(Polyhedral_3(x_flat_sym, lambda[1,:], 1, 4), x1y, simplify=true)
poly_3_grad_2_14 = Symbolics.gradient(Polyhedral_3(x_flat_sym, lambda[1,:], 1, 4), x2y, simplify=true)


poly_4_grad_1_23 = Symbolics.gradient(Polyhedral_4(x_flat_sym, lambda[1,:], 2, 3), x1y, simplify=true)
poly_4_grad_2_23 = Symbolics.gradient(Polyhedral_4(x_flat_sym, lambda[1,:], 2, 3), x2y, simplify=true)
poly_4_grad_1_24 = Symbolics.gradient(Polyhedral_4(x_flat_sym, lambda[1,:], 2, 4), x1y, simplify=true)
poly_4_grad_2_24 = Symbolics.gradient(Polyhedral_4(x_flat_sym, lambda[1,:], 2, 4), x2y, simplify=true)
poly_4_grad_1_14 = Symbolics.gradient(Polyhedral_4(x_flat_sym, lambda[1,:], 1, 4), x1y, simplify=true)
poly_4_grad_2_14 = Symbolics.gradient(Polyhedral_4(x_flat_sym, lambda[1,:], 1, 4), x2y, simplify=true)


state_transition_1_grad = Symbolics.gradient(State_Transition_Mu(x_flat_sym), x1y, simplify=true)
state_transition_2_grad = Symbolics.gradient(State_Transition_Mu(x_flat_sym), x2y, simplify=true)

total_grad_1 = (ref_grad_1  + input_grad_1 +poly_1_grad_1_23 + poly_1_grad_1_24 + poly_1_grad_1_14 +
                 poly_2_grad_1_23 + poly_2_grad_1_24 + poly_2_grad_1_14 + poly_3_grad_1_23 + 
                 poly_3_grad_1_24 + poly_3_grad_1_14 + poly_4_grad_1_23 + poly_4_grad_1_24 + poly_4_grad_1_14 + state_transition_1_grad)


total_grad_2 = (ref_grad_2  + input_grad_2 +poly_1_grad_2_23 + poly_1_grad_2_24 + poly_1_grad_2_14 + 
                poly_2_grad_2_23 + poly_2_grad_2_24 + poly_2_grad_2_14 + poly_3_grad_2_23 + 
                poly_3_grad_2_24 + poly_3_grad_2_14 + poly_4_grad_2_23 + poly_4_grad_2_24 + poly_4_grad_2_14 + state_transition_2_grad)
          

# total_hess_1 = Symbolics.jacobian(total_grad_1, x_flat_sym, simplify=true)
# total_hess_2 = Symbolics.jacobian(total_grad_2, x_flat_sym, simplify=true)

# state_transition_inside = Symbolics.jacobian(State_Transition(x_flat_sym), x_flat_sym, simplify=true)

function get_G_sym(x)
    x_flat = [x'...]
    D = State_Transition(x_flat)
    G = []
    for i in 1:horizon
        G = vcat(G, total_grad_1[30*(i-1)+1:30*i], total_grad_2[30*(i-1)+1:30*i], D[4*(i-1)+1:4*i])
    end
    return G
end

total_hess = Symbolics.jacobian(get_G_sym(x), x_flat_sym, simplify=true)

function get_H(x_traj)
    x_flat = [x_traj'...]
    x_state_flat = [x'...]
    x_state_vals = Dict(x_state_flat[i] => x_flat[i] for i in 1:64*horizon)
    H = Symbolics.value.(substitute.(total_hess, (x_state_vals,)))
    return convert(Matrix{Float64}, H)
end


function get_G(x_traj)
    x_flat = [x_traj'...]
    x_state_flat = [x'...]
    x_state_vals = Dict(x_state_flat[i] => x_flat[i] for i in 1:64*horizon)
    G1 = Symbolics.value.(substitute.(total_grad_1, (x_state_vals,)))
    G2 = Symbolics.value.(substitute.(total_grad_2, (x_state_vals,)))
    D = Symbolics.value.(State_Transition_Scalar(x_traj))

    G = []
    for i in 1:horizon
        G = vcat(G, G1[30*(i-1)+1:30*i], G2[30*(i-1)+1:30*i], D[4*(i-1)+1:4*i])
    end
    return convert(Vector{Float64}, G)
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
        α = 1

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


function line_search(y, G, δy, β=0.4, τ=0.5)
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



