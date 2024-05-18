function newton_method(x_init, lambda, rho, G, H, N, x_flat, λ, ρ, max_iter)
    x_flat_val = [x_init'...]
    flat = vcat(x_flat, λ, ρ)
    for i in 1:max_iter
        println("Iteration: ", i)        
        
        flat_val = vcat(x_flat_val, lambda, rho)
        vals = Dict(flat[i] => flat_val[i] for i in eachindex(flat))
        
        G_val = convert(Vector{Float64}, Symbolics.value.(substitute.(G, (vals,))))
        H_val = convert(Matrix{Float64}, Symbolics.value.(substitute.(H, (vals,))))
        
        δy = - pinv(H_val) * G_val
    
        α = line_search(x_flat_val, lambda, rho, flat, G_val,  δy)
        
        x_flat_val += α * δy

        flat_val = vcat(x_flat_val, lambda, rho)
        vals = Dict(flat[i] => flat_val[i] for i in eachindex(flat))
        G_new = convert(Vector{Float64},Symbolics.value.(substitute.(G, (vals,))))

        if norm(G_new) < 0.01
          return reshape(x_flat_val, 64, N)'
        end
    end
    return reshape(x_flat_val, 64, N)'
end

function line_search(y, lambda, rho, flat, G_val, δy, β=0.1, τ=0.9)
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
    if nci > 0
        for i in 1:nci
            lambda[i] = max(0, lambda[i] + rho[i] * C_val[i])
        end
    end
    if nce > 0
        for i in nci+1:nci+nce
            lambda[i] = lambda[i] + rho[i] * C_val[i]
        end
    end
    return lambda
end

function increasing_schedule(rho, rho_s, lambda, C, y, x_flat, gamma=3)
    rho = rho * gamma
    EPS = 1e-6
    y_flat = [y'...]
    vals = Dict(x_flat[i] => y_flat[i] for i in eachindex(x_flat))
    C_val = convert(Vector{Float64}, Symbolics.value.(substitute.(C, (vals,))))
    print(norm(C_val))
    for i in 1:length(C)
        if C_val[i] < EPS && lambda[i] == 0.0
            rho_s[i] = 0
        else
            rho_s[i] = rho[i]
        end
    end

    for i in 1:length(C)
        if C_val[i] >= EPS
            done = false
            return rho, rho_s, done
        end
    end
    done = true
    return rho, rho_s, done
end

function alsolver(lambda, rho, x_init, x_flat, λ, ρ, C, G, H, max_iter, nci, nce, N)
    y = x_init
    rho_s = rho
    done = false
    while !done
        y = newton_method(y, lambda, rho_s, G, H, N, x_flat, λ, ρ, max_iter)
        lambda = dual_ascent(y, x_flat, lambda, rho_s, C, nce, nci, N)
        rho, rho_s, done = increasing_schedule(rho, rho_s, lambda, C, y, x_flat)
    end
    return y
end

