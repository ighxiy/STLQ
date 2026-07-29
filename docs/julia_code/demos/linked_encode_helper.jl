using LinearAlgebra
using Base.Threads
using MultivariateStats, Statistics
using anonymousmethod.TLSstruct




















function build_cluster_linked_mask(
    cluster_id::Vector{Int},
    mse_vec::Vector{Float32},
    h1::Int;
    mode::Symbol = :mean,          
    good_fraction::Float64 = 0.5,  
    good_count::Int = -1           
)
    @assert length(cluster_id) == length(mse_vec)
    n = length(cluster_id)

    max_c = h1
    sum_mse   = zeros(Float64, max_c)
    count_mse = zeros(Int,    max_c)

    @inbounds for i in 1:n
        c = cluster_id[i]
        @inbounds sum_mse[c]   += mse_vec[i]
        @inbounds count_mse[c] += 1
    end

    cluster_avg = fill(Float64(NaN), max_c)
    global_sum  = 0.0
    global_cnt  = 0

    used_clusters = Int[]

    @inbounds for c in 1:max_c
        if count_mse[c] > 0
            avg = sum_mse[c] / count_mse[c]
            cluster_avg[c] = avg
            global_sum  += sum_mse[c]
            global_cnt  += count_mse[c]
            push!(used_clusters, c)
        end
    end

    n_used = length(used_clusters)
    global_avg = global_cnt > 0 ? global_sum / global_cnt : NaN

    is_bad_cluster = falses(max_c)

    

    if n_used == 0
        
        return is_bad_cluster, cluster_avg, global_avg
    end

    mode == :mean && begin
        
        baseline = global_avg
        @inbounds for c in used_clusters
            if cluster_avg[c] > baseline
                is_bad_cluster[c] = true
            end
        end
        return is_bad_cluster, cluster_avg, baseline
    end

    if mode == :quantile
        
        q = clamp(good_fraction, 0.0, 1.0)
        
        vals = [cluster_avg[c] for c in used_clusters]
        sort!(vals)  

        
        
        if q <= 0.0
            baseline = vals[1]          
        elseif q >= 1.0
            baseline = vals[end]        
        else
            pos = max(1, min(n_used, round(Int, q * n_used)))
            baseline = vals[pos]
        end

        @inbounds for c in used_clusters
            
            if cluster_avg[c] > baseline
                is_bad_cluster[c] = true
            end
        end

        return is_bad_cluster, cluster_avg, baseline
    elseif mode == :topk
        
        K = good_count
        if K <= 0
            
            @inbounds for c in used_clusters
                is_bad_cluster[c] = true
            end
            
            baseline = maximum(cluster_avg[c] for c in used_clusters)
            return is_bad_cluster, cluster_avg, baseline
        end

        K = min(K, n_used)  

        
        pairs = [(cluster_avg[c], c) for c in used_clusters]
        sort!(pairs, by = x -> x[1])  

        
        good_set = Set{Int}()
        @inbounds for i in 1:K
            push!(good_set, pairs[i][2])
        end

        @inbounds for c in used_clusters
            if !(c in good_set)
                is_bad_cluster[c] = true
            end
        end

        
        baseline = pairs[K][1]

        return is_bad_cluster, cluster_avg, baseline
    else
        error("Unsupported mode=$(mode). Use :mean, :quantile or :topk.")
    end
end















function update_codebooks_ls_extend_weighted(
    X::AbstractMatrix{Float32},
    B::AbstractMatrix{<:Integer},
    len_rate_each::AbstractMatrix{Float32},
    h_vec::Vector{Int},
    w::AbstractVector{Float32},
)
    d, n = size(X)
    @assert size(B, 2) == n
    @assert size(len_rate_each, 2) == n
    @assert length(w) == n

    Xw   = similar(X)
    lenw = similar(len_rate_each)

    @inbounds for j in 1:n
        s = sqrt(w[j])          
        @views Xw[:, j]      .= X[:, j] .* s
        @views lenw[:, j]    .= len_rate_each[:, j] .* s
    end

    return update_codebooks_ls_extend(Xw, B, lenw,h_vec)
end





function build_residual_weights(
    idx_one::Vector{Int},
    cluster_id::Vector{Int},
    is_bad_cluster::AbstractVector{Bool};
    alpha_bad::Float32 = 0.1f0,
)
    n_sub = length(idx_one)
    w = fill(1.0f0, n_sub)

    @inbounds for (j, i_global) in enumerate(idx_one)
        c = cluster_id[i_global]
        if is_bad_cluster[c]
            w[j] = alpha_bad
        end
    end

    return w
end


function compute_depths!(parent::Vector{UInt32})
    n = length(parent)
    depths = fill(-1, n)
    state = fill(UInt8(0), n)
    function dfs(i::Int)
        s = state[i]
        if s == 2; return depths[i]; end
        if s == 1; throw(ArgumentError("Cycle detected at node $i")) end
        state[i] = 1
        p = Int(parent[i])
        depths[i] = p == 0 ? 0 : dfs(p) + 1
        state[i] = 2
        return depths[i]
    end
    for i in 1:n
        state[i] == 0 && dfs(i)
    end
    return depths
end



function build_C_contiguous(C::Vector{Matrix{T}}) where T<:AbstractFloat
    m = length(C)
    d = size(C[1],1)
    h_j_vec = [size(C[i],2) for i in 1:m]
    h_max = maximum(h_j_vec)
    C_norms = [sqrt.(sum(C[j].^2, dims=1)) for j in 1:m]
    C_norms_inv_contiguous = zeros(Float32, m, h_max)
    

    for j in 1:m
        C_norms_inv_contiguous[j, 1:h_j_vec[j]] = 1.0 ./ C_norms[j]
    end

    C_contiguous = zeros(Float32, d, h_max, m)
    for j in 1:m
        C_contiguous[:, 1:h_j_vec[j], j] = C[j]
    end
    return C_contiguous,C_norms_inv_contiguous
end




function build_q_precomp(C::Vector{Matrix{T}}, C_norms_inv_contiguous::Array{T,2}) where {T<:AbstractFloat}
    d = size(C[1],1)
    m = length(C)
    h_vec = [size(C[ℓ],2) for ℓ in 1:m]
    H = sum(h_vec)

    
    C_all = Matrix{T}(undef, d, H)
    offsets = zeros(Int, m)              
    flat_layer = zeros(Int, H)           
    invnorm_flat = Vector{T}(undef, H)   

    col = 1
    for ℓ in 1:m
        offsets[ℓ] = col
        hℓ = h_vec[ℓ]
        @views C_all[:, col:col+hℓ-1] .= C[ℓ]
        for k in 1:hℓ
            flat_layer[col+k-1] = ℓ
            invnorm_flat[col+k-1] = C_norms_inv_contiguous[ℓ,k]
        end
        col += hℓ
    end

    
    G = Matrix{T}(undef, H, H)
    mul!(G, transpose(C_all), C_all)   

    return (; C_all, G, offsets, flat_layer, invnorm_flat, h_vec, H, m, d)
end






function compute_cosine_angles(
    X::AbstractMatrix{T},
    C::Vector{Matrix{T}},
    cluster_indices::Dict{Int, Vector{Int}}
) where T<:AbstractFloat
    d, n = size(X)
    m = length(C)
    
    
    cos_angles = zeros(T, n)
    
    
    cluster_codes = collect(keys(cluster_indices))
    
    
    @threads for code in cluster_codes
        indices = cluster_indices[code]
        isempty(indices) && continue
        
        
        center = C[1][:, code]
        center_norm = norm(center)
        
        
        @inbounds for i in indices
            xi = view(X, :, i)
            xi_norm = norm(xi)
            denom = max(xi_norm * center_norm, eps(T))
            cos_angles[i] = dot(xi, center) / denom
        end
    end
    
    return cos_angles
end




function partition_by_cosine_percentiles(
    cos_angles::Vector{T},
    cluster_indices::Dict{Int, Vector{Int}},
    num_layers::Int,
    root_percentile::Float64;
    verbose::Bool=false
) where {T<:AbstractFloat}
    
    
    layers_dict = Dict{Int, Vector{Vector{Int}}}()
    cluster_stats = Dict{Int, Vector{Int}}()
    total_points = 0
    
    
    for (code, indices) in cluster_indices
        n_points = length(indices)
        total_points += n_points
        
        
        if n_points == 0
            layers_dict[code] = [Int[] for _ in 1:num_layers]
            cluster_stats[code] = zeros(Int, num_layers)
            continue
        end
        
        
        cluster_cos = cos_angles[indices]
        
        
        sorted_indices = sortperm(cluster_cos, rev=true)
        sorted_cos = cluster_cos[sorted_indices]
        
        
        n_root = max(1, floor(Int, root_percentile * n_points))
        
        
        layer_assignments = Vector{Vector{Int}}(undef, num_layers)
        layer_counts = zeros(Int, num_layers)
        layer_cos_ranges = Vector{Tuple{T, T}}(undef, num_layers)
        
        
        layer_assignments[1] = indices[sorted_indices[1:n_root]]
        layer_counts[1] = n_root
        layer_cos_ranges[1] = (sorted_cos[1], sorted_cos[n_root])
        
        
        remaining_points = n_points - n_root
        
        
        if num_layers > 1
            
            base_layer_size = floor(Int, remaining_points / (num_layers - 1))
            
            
            remainder = remaining_points % (num_layers - 1)
            
            
            current_idx = n_root + 1
            for layer in 2:num_layers
                
                layer_size = base_layer_size
                if layer <= remainder + 1  
                    layer_size += 1
                end
                
                
                end_idx = min(current_idx + layer_size - 1, n_points)
                
                if current_idx <= n_points
                    layer_assignments[layer] = indices[sorted_indices[current_idx:end_idx]]
                    layer_counts[layer] = length(layer_assignments[layer])
                    layer_cos_ranges[layer] = (sorted_cos[current_idx], sorted_cos[end_idx])
                    current_idx = end_idx + 1
                else
                    layer_assignments[layer] = Int[]
                    layer_counts[layer] = 0
                    layer_cos_ranges[layer] = (0.0, 0.0)
                end
            end
        else
            
            layer_assignments[1] = indices[sorted_indices]
            layer_counts[1] = n_points
            layer_cos_ranges[1] = (sorted_cos[1], sorted_cos[end])
        end
        
        layers_dict[code] = layer_assignments
        cluster_stats[code] = layer_counts
        
        
        if verbose
            println("\nCluster $code layer breakdown ($n_points points):")
            println("  Root-layer points: $n_root (ratio: $(round(100 * n_root/n_points, digits=1))%)")
            
            for layer in 1:num_layers
                count = layer_counts[layer]
                if count == 0
                    println("  Layer $layer: empty")
                    continue
                end
                
                if layer == 1
                    info = "inner"
                elseif layer == num_layers
                    info = "outer"
                else
                    info = "middle"
                end
                
                
                min_cos, max_cos = layer_cos_ranges[layer]
                cos_range = @sprintf("cosine range: %.3f - %.3f", min_cos, max_cos)
                
                println("  Layer $layer ($info): $count pts | $cos_range")
            end
        end
    end
    
    
    if verbose && !isempty(cluster_stats)
        println("\n===== Global layer stats =====")
        println("Total points: $total_points")
        println("Clusters: $(length(cluster_stats))")
        println("Layers: $num_layers")
        println("Root percentile: $(root_percentile * 100)%")
        
        
        avg_counts = zeros(Float64, num_layers)
        for counts in values(cluster_stats)
            for i in 1:num_layers
                avg_counts[i] += counts[i]
            end
        end
        avg_counts ./= length(cluster_stats)
        
        println("\nAverage points per layer:")
        for layer in 1:num_layers
            if layer == 1
                info = "inner"
            elseif layer == num_layers
                info = "outer"
            else
                info = "middle"
            end
            println("  Layer $layer ($info): $(round(avg_counts[layer], digits=1)) pts")
        end
        
        
        layer_distribution = zeros(Int, num_layers)
        for counts in values(cluster_stats)
            for i in 1:num_layers
                layer_distribution[i] += counts[i]
            end
        end
        
        println("\nGlobal point distribution:")
        for layer in 1:num_layers
            percent = round(100 * layer_distribution[layer] / total_points, digits=1)
            println("  Layer $layer: $percent% ($(layer_distribution[layer]) pts)")
        end
        
    end
    
    return layers_dict
end






@inline function reconstruct_node!(
    out::AbstractVector{T},
    B_node::AbstractVector{BType},
    len_rate_node::AbstractVector{T},
    C::Vector{Matrix{T}}
) where {T<:AbstractFloat, BType<:Integer}
    fill!(out, zero(T))
    d = length(out)
    m = length(B_node)
    
    @inbounds for l in 1:m
        code_idx = B_node[l]
        if code_idx != 0
            λ = len_rate_node[l]
            @simd for i in 1:d
                out[i] += λ * C[l][i, code_idx]
            end
        end
    end
end



function make_tls(::Type{T_Type}, ::Type{B_Type}, d, m, H) where {T_Type, B_Type}
    TLSstruct.TLS{T_Type, B_Type}(
        Matrix{T_Type}(undef, m, m), 
        zeros(T_Type, d),  
        zeros(T_Type, d),  
        zeros(B_Type, m),  
        zeros(T_Type, m),  
        zeros(T_Type, m),  
        zeros(B_Type, m),  
        zeros(T_Type, m),  
        zeros(T_Type, H),  
        zeros(T_Type, H),  
        zeros(T_Type, H),  
        Vector{Int}(),  
        falses(0),    
        Vector{Int}()   
    )
end




function make_tls_ils(::Type{T_Type}, ::Type{B_Type}, d, m, H) where {T_Type, B_Type}
    TLSstruct.TLS_ils{T_Type, B_Type}(
        Matrix{T_Type}(undef, m, m), 
        zeros(T_Type, d),  
        zeros(T_Type, d),  
        zeros(B_Type, m),  
        zeros(T_Type, m),  
        zeros(T_Type, m),  
        zeros(B_Type, m),  
        zeros(T_Type, m),  
        zeros(T_Type, H),  
        zeros(T_Type, H),  
        zeros(T_Type, H),  
        Vector{Int}(),  
        falses(0),    
        Vector{Int}(),   
        zeros(B_Type, m),
        zeros(T_Type, m),
        zeros(Int, m),
        zeros(Int, m),
    )
end








function build_one_root_cross_gram(pre_one, pre_root)
    C_one  = pre_one.C_all    
    C_root = pre_root.C_all   

    H_one  = size(C_one,  2)
    H_root = size(C_root, 2)

    G_one_root = Matrix{eltype(C_one)}(undef, H_one, H_root)

    
    
    mul!(G_one_root, transpose(C_one), C_root)

    return G_one_root
end

















function build_xC_one_for_parent!(
    xC_res::AbstractVector{T},
    xC_one_base::AbstractVector{T},
    len_rate_each::AbstractMatrix{T},
    B_parent::AbstractMatrix{BType},
    parent::Vector{UInt32},
    depths::Vector{Int},
    p_idx::Int,
    pre_root,
    pre_one,
    G_one_one::AbstractMatrix{T},
    G_one_root::AbstractMatrix{T},
) where {T<:AbstractFloat, BType<:Integer}

    H_one = pre_one.H
    m     = pre_one.m

    
    @inbounds @simd for q in 1:H_one
        xC_res[q] = xC_one_base[q]
    end

    
    idx   = p_idx
    depth = depths[p_idx]
    offsets_root = pre_root.offsets
    offsets_one  = pre_one.offsets

    while idx != 0
        
        @inbounds for ℓ in 1:m
            α = len_rate_each[ℓ, idx]
            α == 0 && continue

            code = Int(B_parent[ℓ, idx])

            if depth == 0
                
                f_root = offsets_root[ℓ] + code - 1  
                @inbounds @simd for q in 1:H_one
                    xC_res[q] -= α * G_one_root[q, f_root]
                end
            else
                
                f_one = offsets_one[ℓ] + code - 1    
                @inbounds @simd for q in 1:H_one
                    xC_res[q] -= α * G_one_one[q, f_one]
                end
            end
        end

        depth -= 1
        idx = Int(parent[idx])
    end

    return nothing
end

















function build_xC_for_parent_single!(
    xC_res::AbstractVector{T},        
    xC_base::AbstractVector{T},       
    len_rate_each::AbstractMatrix{T}, 
    B_parent::AbstractMatrix{BType},  
    parent::Vector{UInt32},           
    p_idx::Int,                       
    pre,                              
) where {T<:AbstractFloat, BType<:Integer}

    H = pre.H
    m = pre.m
    offsets = pre.offsets
    G = pre.G

    
    @inbounds @simd for q in 1:H
        xC_res[q] = xC_base[q]
    end

    
    idx = p_idx
    while idx != 0
        @inbounds for ℓ in 1:m
            α = len_rate_each[ℓ, idx]
            α == 0 && continue
            code = Int(B_parent[ℓ, idx])
            f    = offsets[ℓ] + code - 1
            @inbounds @simd for q in 1:H
                xC_res[q] -= α * G[q, f]
            end
        end
        idx = Int(parent[idx])
    end

    return nothing
end





function dynamic_icm_encoding_single_for_residual!(
    re::AbstractVector{T},          
    A::Matrix{T},                   
    B::AbstractVector{BType},       
    a::AbstractVector{T},           
    C_contiguous::Array{T,3},       
    pre,
    xC_res::AbstractVector{T},      
    rC::AbstractVector{T},          
    a_bak::AbstractVector{T},       
    max_iters::Integer,
) where {T<:AbstractFloat,BType<:Integer}

    max_iters == 0 && return

    m      = pre.m
    d      = pre.d
    H      = pre.H
    offsets      = pre.offsets
    h_vec        = pre.h_vec
    invnorm_flat = pre.invnorm_flat
    G            = pre.G
    h_max        = size(C_contiguous, 2)

    
    X_norm2 = zero(T)
    @inbounds @simd for i in 1:d
        v = re[i]
        X_norm2 += v * v
    end

    
    @inline function sample_cost_from_xC!(
        Xn2::T,
        xC::AbstractVector{T},
        Bv::AbstractVector{BType},
        av::AbstractVector{T},
    )::T
        term1 = zero(T)
        term2 = zero(T)
        @inbounds for ℓ in 1:m
            kℓ = Int(Bv[ℓ])
            fℓ = offsets[ℓ] + kℓ - 1
            αℓ = av[ℓ]
            term1 += αℓ * xC[fℓ]
            @inbounds for k in 1:m
                kk = Int(Bv[k])
                fk = offsets[k] + kk - 1
                term2 += αℓ * av[k] * G[fℓ, fk]
            end
        end
        return Xn2 - 2*term1 + term2
    end

    
    @inbounds @simd for q in 1:H
        rC[q] = xC_res[q]
    end
    @inbounds for ℓ in 1:m
        αℓ = a[ℓ]
        αℓ == 0 && continue
        fℓ = offsets[ℓ] + Int(B[ℓ]) - 1
        @inbounds @simd for q in 1:H
            rC[q] -= αℓ * G[q, fℓ]
        end
    end

    cur_cost = sample_cost_from_xC!(X_norm2, xC_res, B, a)

    for iter in 1:max_iters
        changed_any = false
        order = randperm(m)   

        @inbounds for jlayer in order
            old_code = Int(B[jlayer])
            old_flat = offsets[jlayer] + old_code - 1

            
            best_code = old_code
            best_val  = -typemax(T)           
            startf    = offsets[jlayer]
            stopf     = startf + h_vec[jlayer] - 1

            for flat in startf:stopf
                raw   = rC[flat] + a[jlayer] * G[flat, old_flat]
                score = raw * invnorm_flat[flat]
                aval  = abs(score)
                if aval > best_val
                    best_val  = aval
                    best_code = flat - startf + 1
                end
            end

            if best_code != old_code
                
                @inbounds @simd for ℓ in 1:m
                    a_bak[ℓ] = a[ℓ]
                end

                
                B[jlayer] = BType(best_code)
                optimized_least_squares_single!(a, re, C_contiguous, B, A, m, d, h_max)

                
                @inbounds @simd for q in 1:H
                    rC[q] = xC_res[q]
                end
                @inbounds for ℓ in 1:m
                    αℓ = a[ℓ]
                    αℓ == 0 && continue
                    fℓ = offsets[ℓ] + Int(B[ℓ]) - 1
                    @inbounds @simd for q in 1:H
                        rC[q] -= αℓ * G[q, fℓ]
                    end
                end

                
                new_cost = sample_cost_from_xC!(X_norm2, xC_res, B, a)

                if new_cost + eps(T) < cur_cost
                    cur_cost   = new_cost
                    changed_any = true
                else
                    
                    B[jlayer] = BType(old_code)
                    @inbounds @simd for ℓ in 1:m
                        a[ℓ] = a_bak[ℓ]
                    end
                    @inbounds @simd for q in 1:H
                        rC[q] = xC_res[q]
                    end
                    @inbounds for ℓ in 1:m
                        αℓ = a[ℓ]
                        αℓ == 0 && continue
                        fℓ = offsets[ℓ] + Int(B[ℓ]) - 1
                        @inbounds @simd for q in 1:H
                            rC[q] -= αℓ * G[q, fℓ]
                        end
                    end
                end
            end
        end

        
        if !changed_any
            break
        end
    end

    return nothing
end


@inline function _pool_take_or_new(pool, new_fn::Function)
    if isdefined(Base, :trytake!)
        v = Base.trytake!(pool)
        if v === nothing
            return new_fn(), false
        end
        return v, true
    end
    if isready(pool)
        return take!(pool), true
    end
    return new_fn(), false
end









function all_errors!(out::Vector{T},
                X::AbstractMatrix{T},
                R_full::AbstractMatrix{T},
              ) where {T<:AbstractFloat}
    d, n = size(X)
    @threads for j in 1:n
        s = zero(T)
        @simd for i in 1:d
            diff = X[i, j] - R_full[i, j]
            s += diff*diff
        end
        out[j] = s 
    end
end