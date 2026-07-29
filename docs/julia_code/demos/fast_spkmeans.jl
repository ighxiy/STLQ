using LinearAlgebra
using Statistics
using Random
using NearestNeighbors
using anonymousmethod.TLSstruct





function initialize_centers_fast(X::AbstractMatrix{<:Real}, k::Int, method::String, 
                                rng::Union{AbstractRNG, Int}, n_samples::Int=10000)
    d, n = size(X)
    
    if isa(rng, Int)
        rng = MersenneTwister(rng)
    end
    
    if method == "random"
        
        indices = rand(rng, 1:n, k)
        centers = X[:, indices]
    elseif method == "kmeans++"
        n_samples_rel=min(n_samples, n)
        println("kmeans++ ",n_samples_rel)
        
        if n_samples_rel == n
            X_sample = X
        else
            sample_indices = rand(rng, 1:n, n_samples_rel)
            X_sample = X[:, sample_indices]
        end
        
        centers = initialize_centers_kmeanspp_subset_accurate(X_sample, k, rng)
    end
    
    
    clen = size(centers, 2)
    for i in 1:clen
        norm_val = norm(centers[:, i])
        if norm_val > 1e-12
            centers[:, i] = centers[:, i] / norm_val
        else
            println("warning! norm_val < 1e-12 in initialize_centers_fast")
        end
    end
    
    return centers
end





function initialize_centers_kmeanspp_subset_accurate(X::AbstractMatrix{<:Real}, k::Int, rng::AbstractRNG)
    d, n = size(X)
    centers = Matrix{Float64}(undef, d, k)
    
    
    first_idx = rand(rng, 1:n)
    @simd for i in 1:d
        @inbounds centers[i, 1] = X[i, first_idx]
    end
    
    
    distances = Vector{Float64}(undef, n)
    for i in 2:min(k, n)
        
        @threads for j in 1:n
            @inbounds begin
                min_dist = Inf
                @simd for l in 1:(i-1)
                    
                    dot_prod = zero(Float64)
                    @simd for m in 1:d
                        dot_prod += X[m, j] * centers[m, l]
                    end
                    dist = 2.0 - 2.0 * dot_prod
                    min_dist = min(min_dist, dist)
                end
                distances[j] = max(min_dist, 0)
            end
        end
        
        
        total_dist = sum(distances)
        if total_dist > 1e-12
            @simd for j in 1:n
                @inbounds distances[j] /= total_dist
            end
            cumsum!(distances, distances)  
            r = rand(rng)
            selected_idx = searchsortedfirst(distances, r)
            @simd for m in 1:d
                @inbounds centers[m, i] = X[m, min(selected_idx, n)]
            end
        else
            idx = rand(rng, 1:n)
            @simd for m in 1:d
                @inbounds centers[m, i] = X[m, idx]
            end
        end
    end
    
    return centers
end





function spherical_kmeans_fast(X::AbstractMatrix{<:Real}, k::Int; 
                              maxiter::Int=300, 
                              tol::Float64=1e-6,
                              init::String="random",  
                              init_samples::Int=2000, 
                              weights::Union{Nothing, AbstractVector{<:Real}}=nothing,
                              display::Bool=false,
                              rng::Union{AbstractRNG, Int}=Random.GLOBAL_RNG,
                              batch_size::Int=10000)  
    
    d, n = size(X)
    
    
    k > 0 || throw(ArgumentError("k must be positive"))
    k ≤ n || throw(ArgumentError("k must be less than or equal to number of samples"))
    
    
    X_normalized = normalize_data_fast(X)
    
    
    if weights === nothing
        weights = ones(n)
    else
        length(weights) == n || throw(DimensionMismatch("weights length must match number of samples"))
        all(w ≥ 0 for w in weights) || throw(ArgumentError("weights must be non-negative"))
    end
    
    
    if display; println("Initializing cluster centers..."); end;
    centers = initialize_centers_fast(X_normalized, k, init, rng, init_samples)
    
    
    assignments = zeros(Int, n)
    costs = Float64[]
    dot_values = zeros(Float64, n)
    converged = false
    prev_cost = Inf
    
    
    if n > batch_size
        return spherical_kmeans_minibatch(X_normalized, k, centers, maxiter, tol, weights, display)
    end
    
    
    for iter in 1:maxiter
        
        assign_samples_parallel!(assignments,dot_values, X_normalized, centers)
        
        
        current_cost = compute_cost_fast(dot_values, weights)
        push!(costs, current_cost)
        
        
        update_centers_fast!(centers, X_normalized, assignments, weights)

        
        if abs(prev_cost - current_cost) < tol
            converged = true
            break
        end
        
        prev_cost = current_cost
        
        if display && (iter % 10 == 0 || iter == 1)
            println("Iteration $iter: cost = $current_cost")
        end
    end
    
    return FSphericalKmeansResult(assignments, centers, costs, converged, length(costs))
end






function normalize_data_fast(X::AbstractMatrix{<:Real})
    X_normalized = similar(X, Float64)
    d, n = size(X)
    
    @threads for i in 1:n
        @inbounds begin
            
            norm_val = zero(Float64)
            @simd for j in 1:d
                norm_val += X[j, i] * X[j, i]
            end
            norm_val = sqrt(norm_val)
            
            if norm_val > 1e-12
                @simd for j in 1:d
                    X_normalized[j, i] = X[j, i] / norm_val
                end
            else
                @simd for j in 1:d
                    X_normalized[j, i] = X[j, i]
                end
            end
        end
    end
    return X_normalized
end




function assign_samples_parallel!(assignments::Vector{Int},dot_values::Vector{<:Real}, X::AbstractMatrix{<:Real}, 
                                 centers::AbstractMatrix{<:Real})
    n = size(X, 2)
    k = size(centers, 2)
    d = size(X, 1)
    
    @threads for i in 1:n
        @inbounds begin
            max_similarity = -Inf
            best_cluster = 1
            best_dot = 0.0

            
            @simd for j in 1:k
                
                similarity = zero(Float64)
                @simd for m in 1:d
                    similarity += X[m, i] * centers[m, j]
                end

                if similarity > max_similarity
                    max_similarity = similarity
                    best_cluster = j
                    best_dot = similarity
                end
            end
            assignments[i] = best_cluster
            dot_values[i] = best_dot
        end
    end
end






function compute_cost_fast(dot_values::Vector{Float64}, weights::AbstractVector{<:Real})
    n = length(dot_values)
    total_cost = Atomic{Float64}(0.0)
    total_weight = Atomic{Float64}(0.0)
    
    
    @threads for i in 1:n
        cost = 1 - dot_values[i]
        local_cost = weights[i] * cost
        local_weight = weights[i]
        
        
        atomic_add!(total_cost, local_cost)
        atomic_add!(total_weight, local_weight)
    end
    
    return total_weight[] > 0 ? total_cost[] / total_weight[] : 0.0
end





function update_centers_fast!(centers::Matrix{Float64}, X::AbstractMatrix{<:Real}, 
                             assignments::Vector{Int}, weights::AbstractVector{<:Real})
    d, k = size(centers)
    n = size(X, 2)
    
    
    cluster_sums = zeros(d, k)
    cluster_weights = zeros(k)
    
    chunk_size = max(1, n ÷ max(1, Threads.nthreads() * 2))
    num_chunks = cld(n, chunk_size)
    local_sums = [zeros(d, k) for _ in 1:num_chunks]
    local_weights = [zeros(k) for _ in 1:num_chunks]
    
    Threads.@sync for chunk_idx in 1:num_chunks
        Threads.@spawn begin
            chunk_start = (chunk_idx - 1) * chunk_size + 1
            chunk_end = min(chunk_start + chunk_size - 1, n)
            sums = local_sums[chunk_idx]
            wts = local_weights[chunk_idx]
            
            @inbounds for i in chunk_start:chunk_end
                cluster = assignments[i]
                weight = weights[i]
                wts[cluster] += weight
                
                @simd for j in 1:d
                    sums[j, cluster] += weight * X[j, i]
                end
            end
        end
    end
    
    
    @simd for i in 1:k
        @simd for t in 1:num_chunks
            cluster_weights[i] += local_weights[t][i]
            @simd for j in 1:d
                cluster_sums[j, i] += local_sums[t][j, i]
            end
        end
    end
    
    
    @threads for j in 1:k
        @inbounds begin
            if cluster_weights[j] > 1e-12
                
                @simd for i in 1:d
                    centers[i, j] = cluster_sums[i, j] / cluster_weights[j]
                end
                
                
                norm_val = zero(Float64)
                @simd for i in 1:d
                    norm_val += centers[i, j] * centers[i, j]
                end
                norm_val = sqrt(norm_val)
                
                if norm_val > 1e-12
                    @simd for i in 1:d
                        centers[i, j] = centers[i, j] / norm_val
                    end
                end
            else
                
                println("random select items for empty clusters!!!")
                idx = rand(1:n)
                @simd for i in 1:d
                    centers[i, j] = X[i, idx]
                end
                
                norm_val = zero(Float64)
                @simd for i in 1:d
                    norm_val += centers[i, j] * centers[i, j]
                end
                norm_val = sqrt(norm_val)
                
                if norm_val > 1e-12
                    @simd for i in 1:d
                        centers[i, j] = centers[i, j] / norm_val
                    end
                end
            end
        end
    end
end





function spherical_kmeans_fast_with_adaptive_weights(
    X::AbstractMatrix{<:Real}, 
    k::Int; 
    maxiter::Int=100, 
    tol::Float64=1e-6,
    init::String="random",
    init_samples::Int=2000,
    display::Bool=false,
    rng::Union{AbstractRNG, Int}=Random.GLOBAL_RNG,
    batch_size::Int=10000,
    
    initial_weight::Float64=1.0,           
    min_weight::Float64=0.1,               
    outlier_quantile::Float64=0.95,        
    cost_threshold::Float64=0.8,           
    annealing_factor::Float64=0.5,          
    warmup_iters::Int=5                     
)
    
    d, n = size(X)
    
    
    k > 0 || throw(ArgumentError("k must be positive"))
    k ≤ n || throw(ArgumentError("k must be less than or equal to number of samples"))
    
    
    X_normalized = normalize_data_fast(X)
    
    
    weights = fill(initial_weight, n)
    
    
    if display; println("Initializing cluster centers..."); end;
    centers = initialize_centers_fast(X_normalized, k, init, rng, init_samples)

    
    assignments = zeros(Int, n)
    costs = Float64[]
    dot_values = zeros(Float64, n)
    converged = false
    prev_cost = Inf
    
    
    for iter in 1:maxiter
        
        assign_samples_parallel!(assignments, dot_values, X_normalized, centers)

        
        current_cost = compute_cost_fast(dot_values, weights)
        push!(costs, current_cost)
        
        
        if iter > warmup_iters
            
            point_costs = 1.0 .- dot_values
            
            
            quantile_threshold = quantile(point_costs, outlier_quantile)
            effective_threshold = max(quantile_threshold, cost_threshold)
            
            
            annealed_factor = annealing_factor * (1 - exp(-(iter - warmup_iters) / (maxiter - warmup_iters)))
            
            
            for i in 1:n
                if point_costs[i] > effective_threshold
                    
                    weights[i] = max(min_weight, weights[i] * (1 - annealed_factor))
                else
                    
                    weights[i] = min(1.0, weights[i] * (1 + annealed_factor/2))
                end
            end
            
            if display
                outlier_count = count(point_costs .> effective_threshold)
                println("Iteration $iter: cost = $current_cost, outliers = $outlier_count/$n ($(round(outlier_count/n*100, digits=2))%)")
            end
        end
        
        
        update_centers_fast!(centers, X_normalized, assignments, weights)
        
        
        if abs(prev_cost - current_cost) < tol
            converged = true
            if display
                println("Converged at iteration $iter")
            end
            break
        end
        
        prev_cost = current_cost
        
        if display && (iter % 10 == 0 || iter == 1)
            println("Iteration $iter: cost = $current_cost")
        end
    end
    
    return FSphericalKmeansResult(assignments, centers, costs, converged, length(costs)), weights
end
