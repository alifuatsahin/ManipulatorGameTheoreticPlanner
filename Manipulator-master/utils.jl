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

function constraints(x, N, F1, f1, F2, f2, H1, h1, H2, h2)
    @variables C[1:0]

    t1(x) = [l1 * cos(x); l1 * sin(x)]

    t2(x) = [l1 * cos(x) + l2 * cos(x); l1 * sin(x) + l2 * sin(x)]

    t3(x) = [l3 * cos(x); d - l3 * sin(x)]

    t4(x) = [l3 * cos(x) + l4 * cos(x); d - l3 * sin(x) - l4 * sin(x)]

    R1(x) = [cos(x) -sin(x); sin(x) cos(x)]
    R2(x) = [cos(x) sin(x); -sin(x) cos(x)]

    R1T(x) = [cos(x) sin(x); -sin(x) cos(x)]
    R2T(x) = [cos(x) -sin(x); sin(x) cos(x)]

    # TODO change R to parametric form
    for i in 1:N

        append!(C, x[i, 5:8] .^2 - ones(4)*1.5^2)

        push!(C, (H2*R1(x[i,2])*t2(x[i,2]) + h2)'* x[i,9:12] + (F1*R2(x[i,3])*t3(x[i,3]) + f1)'*x[i,21:24])
        push!(C, (H2*R1(x[i,2])*t2(x[i,2]) + h2)'* x[i,13:16] + (F2*R2(x[i,4])*t4(x[i,4]) + f2)'*x[i,25:28])
        push!(C, (H1*R1(x[i,1])*t1(x[i,1]) + h1)'* x[i,17:20] + (F2*R2(x[i,4])*t4(x[i,4]) + f2)'*x[i,29:32])

        push!(C, (H2*R1(x[i,2])*t2(x[i,2])+ h2)'* x[i,33:36] + (F1*R2(x[i,3])*t3(x[i,3]) + f1)'*x[i,45:48])
        push!(C, (H2*R1(x[i,2])*t2(x[i,2]) + h2)'* x[i,37:40] + (F2*R2(x[i,4])*t4(x[i,4]) + f2)'*x[i,49:52])
        push!(C, (H1*R1(x[i,1])*t1(x[i,1]) + h1)'* x[i,41:44] + (F2*R2(x[i,4])*t4(x[i,4]) + f2)'*x[i,53:56])

        append!(C, -x[i,9:56])

        push!(C, symbolic_norm(R2T(x[i,3])*F1'*x[i,21:24]) - 1)
        push!(C, symbolic_norm(R2T(x[i,4])*F2'*x[i,25:28]) - 1)
        push!(C, symbolic_norm(R2T(x[i,4])*F2'*x[i,29:32]) - 1)

        push!(C, symbolic_norm(R2T(x[i,3])*F1'*x[i,45:48]) - 1)
        push!(C, symbolic_norm(R2T(x[i,4])*F2'*x[i,49:52]) - 1)
        push!(C, symbolic_norm(R2T(x[i,4])*F2'*x[i,53:56]) - 1)
    end

    for i in 1:N

        append!(C, (R1T(x[i,2])*H2'*x[i,9:12] + R2T(x[i,3])* F1'*x[i,21:24]))
        append!(C, (R1T(x[i,2])*H2'*x[i,13:16] + R2T(x[i,4])* F2'*x[i,25:28]))
        append!(C, (R1T(x[i,1])*H1'*x[i,17:20] + R2T(x[i,4])* F2'*x[i,29:32]))

        append!(C, (R1T(x[i,2])*H2'*x[i,33:36] + R2T(x[i,3])* F1'*x[i,45:48]))
        append!(C, (R1T(x[i,2])*H2'*x[i,37:40] + R2T(x[i,4])* F2'*x[i,49:52]))
        append!(C, (R1T(x[i,1])*H1'*x[i,41:44] + R2T(x[i,4])* F2'*x[i,53:56]))
    end
    return C
end

function generate_trajectory(θ_init, θ_ref, state_dim, N, dt)
    x_diff = θ_ref - θ_init
    x = ones(N, state_dim)*0.1
    x_prev = θ_init
    for i in 1:N
        x[i, 1:4] = θ_init + (i/N)*x_diff
        x[i, 5:8] = (x[i, 1:4] - x_prev)/dt
        x[i, state_dim-7:state_dim] = zeros(8)
        x_prev = x[i, 1:4]
    end
    return x
end

