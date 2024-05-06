
function create_initial_traj(N)
    X = vec(rand(4, N))    
    U = vec(rand(4, N))
    LAMBDA = vec(rand(1, N))
    return vcat(X, U, LAMBDA)
end



