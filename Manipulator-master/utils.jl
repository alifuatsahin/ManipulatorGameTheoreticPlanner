function symbolic_norm(v)
    return sum([vi^2 for vi in v])
end

function player_cost(x, θ_ref, R, Q, N)
    cost = 0
    for i in 1:N
        cost += (x[i, 1:4] - θ_ref)'*Q*(x[i, 1:4] - θ_ref)
    end
    for i in 1:N
        cost += x[i, 5:8]'*R*x[i, 5:8]
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

    t2(x1,x2) = [l1 * cos(x1) + l2 * cos(x2); l1 * sin(x1) + l2 * sin(x2)]

    t3(x) = [l3 * cos(x); d - l3 * sin(x)]

    t4(x3,x4) = [l3 * cos(x3) + l4 * cos(x4); d - l3 * sin(x3) - l4 * sin(x4)]

    R1(x) = [cos(x) -sin(x); sin(x) cos(x)]
    R2(x) = [cos(x) sin(x); -sin(x) cos(x)]

    R1T(x) = [cos(x) sin(x); -sin(x) cos(x)]
    R2T(x) = [cos(x) -sin(x); sin(x) cos(x)]

    # TODO change R to parametric form
    for i in 1:N

        append!(C, x[i, 5:8] .^2 - ones(4)*2.5^2)

        push!(C, ((H2*R1(x[i,2])*t2(x[i,1], x[i,2]) + h2)'* x[i,9:12] + (F2*R2(x[i,4])*t4(x[i,3], x[i,4]) + f2)'*x[i,13:16] + 1.5))
        push!(C, ((H2*R1(x[i,2])*t2(x[i,1], x[i,2]) + h2)'* x[i,17:20] + (F2*R2(x[i,4])*t4(x[i,3], x[i,4]) + f2)'*x[i,21:24] + 1.5))

        append!(C, -x[i,9:24])

        push!(C, (symbolic_norm(F2'*x[i,13:16]) - 1))
        push!(C, (symbolic_norm(F2'*x[i,21:24]) - 1))

    end

    for i in 1:N
        append!(C,(R1T(x[i,2])*H2'*x[i,9:12] + R2T(x[i,4])* F2'*x[i,13:16]))
        append!(C,(R1T(x[i,2])*H2'*x[i,17:20] + R2T(x[i,4])* F2'*x[i,21:24]))
    end
    return C
end

function generate_trajectory(θ_init, θ_ref, state_dim, N, dt)
    x = ones(N, state_dim) * 0.05
    x_diff = (θ_ref - θ_init)
    N1 = div(N, 3)
    N2 = 2 * N1
    for i in 1:N
        if i <= N1
            # First third: x[3, :] and x[4, :] constant, x[1, :] and x[2, :] going to zero
            x[i, 1:2] = θ_init[1:2] * (1 - (i / N1))
            x[i, 3:4] = θ_init[3:4]
            if i == 1
                x[i, 5:8] = (x[i, 1:4] - θ_init)/dt
            else
                x[i, 5:8] = (x[i, 1:4] - x[i-1, 1:4]) / dt
            end
        elseif i <= N2
            # Second third: x[1, :] and x[2, :] constant, x[3, :] and x[4, :] going to θ_ref
            x[i, 1:2] = zeros(2)
            x[i, 3:4] = θ_init[3:4] + (i - N1) / N1 * x_diff[3:4]
            x[i, 5:8] =  (x[i, 1:4] - x[i-1, 1:4]) / dt
        else
            # Final third: x[1, :] and x[2, :] going to θ_ref
            x[i, 1:2] = x[i, 1:2] + (i - N2) / N1 * (θ_ref[1:2] - x[N2+1, 1:2])
            x[i, 3:4] = θ_ref[3:4]
            x[i, 5:8] = (x[i, 1:4] - x[i-1, 1:4]) / dt
        end
        
        x[i, state_dim-7:state_dim] = zeros(8)
    end
    return x
end



function generate_intermediate_points(y, num_intermediate_points)
    num_points = size(y, 1)
    interpolated_y = []

    for i in 1:num_points-1
        for j in 0:num_intermediate_points
            t = j / (num_intermediate_points + 1)
            new_point = (1 - t) * y[i] + t * y[i+1]
            push!(interpolated_y, new_point)
        end
    end
    push!(interpolated_y, y[end])  # Add the last point
    
    return vcat(interpolated_y...)
end


