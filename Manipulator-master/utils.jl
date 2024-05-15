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

function generate_trajectory(θ_init, θ_ref, N, dt)
    x_diff = θ_ref*2 - θ_init
    x = zeros(N, 4*length(θ_init))
    x_prev = θ_init
    x[1, 1:4] = θ_init
    x[1, 9:16] = zeros(1, 8)
    if N > 1
        for i in 2:N
            x[i, 1:4] = θ_init + (i/N)*x_diff
            x[i, 5:8] = (x[i, 1:4] - x_prev)/dt
            x[i, 9:16] = zeros(1, 8)
            x_prev = x[i, 1:4]
        end
    end
    return x
end


