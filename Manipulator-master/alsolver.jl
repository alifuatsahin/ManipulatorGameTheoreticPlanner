function inner_loop(x_init, lambda, G, H, N, x_flat, λ, max_iter)
    x_flat_val = [x_init'...]
    flat = vcat(x_flat, λ)
    for i in 1:max_iter
        println("Iteration: ", i)        
        
        flat_val = vcat(x_flat_val, lambda)
        vals = Dict(flat[i] => flat_val[i] for i in 1:20*N)
        
        G_val = convert(Vector{Float64}, Symbolics.value.(substitute.(G, (vals,))))
        H_val = convert(Matrix{Float64}, Symbolics.value.(substitute.(H, (vals,))))
        println("G_val: ", norm(G_val,1))
        δy = - pinv(H_val) * G_val
    
        α = line_search(x_flat_val, flat, G_val,  δy)
        println("α: ", α)
        
        println("norm of delta: ", norm(δy))   
        x_flat_val += α * δy

        flat_val = vcat(x_flat_val, lambda)
        vals = Dict(flat[i] => flat_val[i] for i in 1:20*N)
        G_new = convert(Vector{Float64},Symbolics.value.(substitute.(G, (vals,))))

        println("G_new", norm(G_new,1))

        if norm(G_new) < 0.01
          return reshape(x_flat_val, 16, N)'
        end
    end
    return reshape(x_flat_val, 16, N)'
end

function line_search(y, flat, G_val, δy, β=0.1, τ=0.9)
    α = 1
    while α > 1e-4  

        y_new = y + α * δy
        flat_val = vcat(y_new, lambda)
        vals = Dict(flat[i] => flat_val[i] for i in 1:20*N)

        G_alpha = convert(Vector{Float64},Symbolics.value.(substitute.(G, (vals,))))

        println("norm of G_alpha: ", norm(G_alpha, 1))
        println("norm of G: ", norm(G_val, 1))

        if norm(G_alpha, 1) < (1 - α * β) * norm(G_val, 1)
            return α
        end
        
        α *= τ
    end
    return α  
end