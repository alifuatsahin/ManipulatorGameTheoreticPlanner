include("plotting.jl")
include("utils.jl")

function newton_method(x_init, lambda, rho, G, H, N, x_flat, λ, ρ, max_iter, state_dim, initial_damping=1e-3, beta=2, tolerance=1)
    x_flat_val = [x_init'...]
    flat = vcat(x_flat, λ, ρ)
    damping = initial_damping
    try
        for i in 1:max_iter
            println("Iteration: ", i)
            
            flat_val = vcat(x_flat_val, lambda, rho)
            vals = Dict(flat[i] => flat_val[i] for i in eachindex(flat))
            
            G_val = convert(Vector{Float64}, Symbolics.value.(substitute.(G, (vals,))))
            H_val = convert(Matrix{Float64}, Symbolics.value.(substitute.(H, (vals,))))
            
            # Modify Hessian with damping factor for LM method
            H_val_damped = H_val + damping * I(size(H_val, 1))
            
            # Compute the step direction
            δy = - inv(H_val_damped) * (G_val + damping * x_flat_val)
        
            α = line_search(x_flat_val, lambda, rho, flat, G_val, δy)
            
            x_flat_val = x_flat_val + α * δy
            
            # Evaluate new state
            flat_val_new = vcat(x_flat_val, lambda, rho)
            vals_new = Dict(flat[i] => flat_val_new[i] for i in eachindex(flat))
            G_new = convert(Vector{Float64}, Symbolics.value.(substitute.(G, (vals_new,))))

            println("Norm: ", norm(G_new, 1))

            if norm(G_new, 1) < 1
                println("Converged")
                return reshape(x_flat_val, state_dim, N)'
            end

            #= # Adjust damping factor
            # if norm(G_new, 1) < norm(G_val, 1)
            #     # If improvement, decrease damping
            #     damping /= beta
            #     x_flat_val = x_flat_val_new
            # else
            #     # If no improvement, increase damping
            #     damping *= beta
            # end =#
        end
    catch e
        if isa(e, InterruptException)
            y = reshape(x_flat_val, state_dim, N)'
            return y
        else
            rethrow(e)
        end
    end 
    return reshape(x_flat_val, state_dim, N)'
end


function line_search(y, lambda, rho, flat, G_val, δy, β=0.1, τ=0.5)
    α = 1
    while α > 1e-4  

        y_new = y + α * δy
        flat_val = vcat(y_new, lambda, rho)
        vals = Dict(flat[i] => flat_val[i] for i in eachindex(flat))

        G_alpha = convert(Vector{Float64},Symbolics.value.(substitute.(G, (vals,))))

        if norm(G_alpha, 1) < (1 - α * β) * norm(G_val, 1)
            return α
        end
        
        α *= τ
    end
    return α  
end

function dual_ascent(y, x_flat, lambda, rho, C, nci, nce, N)
    y_flat = [y'...]
    vals = Dict(x_flat[i] => y_flat[i] for i in eachindex(x_flat))
    C_val = convert(Vector{Float64}, Symbolics.value.(substitute.(C, (vals,))))
    EPS = 1e-3
    max_C_vals = distance_convergence(nci, nce, C_val, N, EPS)
    if nci > 0
        idx = 1
        for i in 1:nci*N
            if i % nci > 4 && max_C_vals[idx] == 0
                lambda[i] = 0
            elseif i % nci > 4
                lambda[i] = lambda[i] + rho[i] * C_val[idx]
            else
                lambda[i] = max(0, lambda[i] + rho[i] * C_val[i])
            end
            if i % nci == 0
                idx += 1
            end
        end
    end
    if nce > 0
        idx = 1
        for i in (nci*N)+1:(nci+nce)*N
            lambda[i] = lambda[i] + rho[i] * max_C_vals[idx]
            if i % nce == 0
                idx += 1
            end 
            if idx == 10
                break
            end 
        end
    end
    return lambda
end

function increasing_schedule(rho, rho_s, lambda, C, y, x_flat, nci, nce, N, gamma=10)
    rho = rho * gamma
    EPS = 1e-3
    y_flat = [y'...]
    vals = Dict(x_flat[i] => y_flat[i] for i in eachindex(x_flat))
    C_val = convert(Vector{Float64}, Symbolics.value.(substitute.(C, (vals,))))

    positive_indices = findall(>(0), C_val)
    positive_values = C_val[positive_indices]
    
   
    if length(positive_values) < 10
        println("There are fewer than 10 positive values.")
        top_10_indices = sortperm(positive_indices, rev = true)
    else
        sorted_positive_indices = sortperm(positive_values, rev=true)
        top_10_indices = positive_indices[sorted_positive_indices[1:10]]
    end
    
    println("Indices of the top 10 maximum positive values in C_val: ", top_10_indices)
    println("Number of constraint violations: ", length(positive_values))
    
    
    for i in 1:length(C)
        
        if C_val[i] < EPS && lambda[i] == 0 && i <= nci*N
            rho_s[i] = 0
        else
            rho_s[i] = rho[i]
        end
    end

    done = convergence_check(nci, nce, C_val, N, EPS)
    
    return rho, rho_s, done
end

function distance_convergence(nci, nce, C_val, N, EPS=1e-3)
    lambda_update = zeros(N)
    if nci > 0
        for i in 1:N
            if all(C_val[(i-1)*nci + 5: i*nci] .< EPS)
                lambda_update[i] = 0
            else
                lambda_update[i] = maximum(C_val[(i-1)*nci + 5: i*nci])
            end
        end
    if nce > 0
        for i in 1:N
            if all(abs.(C_val[nci*N + nce*(i-1) + 1:nci*N + nce*i]) .>= EPS)
                lambda_update[i] = maximum(lambda_update[i], maximum(abs.(C_val[nci*N + nce*(i-1) + 1:nci*N + nce*i])))
            end
        end
    return lambda_update
    end

function convergence_check(nci, nce, C_val, N, EPS=1e-3)
    done = true

    if nci > 0
        for i in 1:nci*N
            if C_val[i] >= EPS
                done = false
            end
        end
    end
    if nce > 0
        for i in (nci*N)+1:(nci+nce)*N
            if abs(C_val[i]) > EPS
                done = false
            end
        end

    return done
    end

function alsolver(lambda, rho, x_init, x_flat, λ, ρ, C, G, H, max_iter, nci, nce, N, state_dim)
    y = x_init
    rho_s = rho
    done = false
    max_iter_o = 1
    iter = 0
    while !done && iter < max_iter_o
        y = newton_method(y, lambda, rho_s, G, H, N, x_flat, λ, ρ, max_iter, state_dim)
        int_y_1 = generate_intermediate_points(y[:, 1], 5);
        int_y_2 = generate_intermediate_points(y[:, 2], 5);
        int_y_3 = generate_intermediate_points(y[:, 3], 5);
        int_y_4 = generate_intermediate_points(y[:, 4], 5);
        animate_robots(int_y_1, int_y_2, int_y_3, int_y_4, d, l1, l2, l3, l4, w)

        lambda = dual_ascent(y, x_flat, lambda, rho_s, C, nce, nci, N)
        rho, rho_s, done = increasing_schedule(rho, rho_s, lambda, C, y, x_flat, nci, nce, N)
        iter += 1
    end
    return y
end

