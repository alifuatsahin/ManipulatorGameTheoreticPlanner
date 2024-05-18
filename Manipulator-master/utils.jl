function symbolic_norm(v)
    return sqrt(sum([vi^2 for vi in v]))
end

function player_cost(x, θ_ref, R, Q, N)
    cost = 0
    for i in 1:N
        cost += (x[i, 1:4] - θ_ref)'*Q*(x[i, 1:4] - θ_ref) + x[i, 5:8]'*R*x[i, 5:8]
    end
    return cost
end

function state_transition(x, dt, N, θ_init)
    @variables D[1:4*N]
    D[1:4] = x[1, 1:4] - θ_init - dt*x[1, 5:8]
    if N > 1
        for i in 2:N
            D[4*(i-1)+1:4*i] = x[i, 1:4] - x[i-1, 1:4] - dt*x[i, 5:8]
        end
    end
    return D
end

function constraints(x, n, N)
    @variables C[1:n*N]
    # TODO change R to parametric form
    for i in 1:N
        C[4*(i-1)+1:4*(i-1)+4] = x[i, 5:8] .^2 - ones(4)*1.5^2
    end
    for i in 1:N
        push!(C, (H*R1*t1 + h)'* x[i,9:12] + (F*R2*t2 + g)'*x[i,21:24])
        push!(C, (H*R1*t1 + h)'* x[i,13:16] + (F*R2*t2 + g)'*x[i,25:28])
        push!(C, (H*R1*t1 + h)'* x[i,17:20] + (F*R2*t2 + g)'*x[i,29:32])
    end
    for i in 1:N
        push!(C, (H*R1*t1 + h)'* x[i,33:36] + (F*R2*t2 + g)'*x[i,45:48])
        push!(C, (H*R1*t1 + h)'* x[i,37:40] + (F*R2*t2 + g)'*x[i,49:52])
        push!(C, (H*R1*t1 + h)'* x[i,41:44] + (F*R2*t2 + g)'*x[i,53:56])
    end
    for i in 1:N
        append!(C, -x[i,9:56])
    end
    for i in 1:N
        push!(C, symbolic_norm(R2T*F'*x[i,21:24]) - 1)
        push!(C, symbolic_norm(R2T*F'*x[i,25:28]) - 1)
        push!(C, symbolic_norm(R2T*F'*x[i,29:32]) - 1)
        push!(C, symbolic_norm(R2T*F'*x[i,45:48]) - 1)
        push!(C, symbolic_norm(R2T*F'*x[i,49:52]) - 1)
        push!(C, symbolic_norm(R2T*F'*x[i,53:56]) - 1)
    end

    for i in 1:N
        append!(C, (R1T*H'*x[i,9:12] + R2T* F'*x[i,21:24]))
        append!(C, (R1T*H'*x[i,33:36] + R2T* F'*x[i,45:48]))
        append!(C, (R1T*H'*x[i,13:16] + R2T* F'*x[i,25:28]))
        append!(C, (R1T*H'*x[i,37:40] + R2T* F'*x[i,49:52]))
        append!(C, (R1T*H'*x[i,17:20] + R2T* F'*x[i,29:32]))
        append!(C, (R1T*H'*x[i,41:44] + R2T* F'*x[i,53:56]))
    end
    return C
end

function generate_trajectory(θ_init, θ_ref, state_dim, N, dt)
    x_diff = θ_ref*2 - θ_init
    x = zeros(N, state_dim)
    x_prev = θ_init
    for i in 1:N
        x[i, 1:4] = θ_init + (i/N)*x_diff
        x[i, 5:8] = (x[i, 1:4] - x_prev)/dt
        x_prev = x[i, 1:4]
    end
    return x
end

