using LinearAlgebra
using Base.Threads
using StaticArrays
using Distances
using NearestNeighbors
using anonymousmethod.TLSstruct
using anonymousmethod.Beam_search
include("./greedy_quantizer.jl")
include("./encode_abs.jl")
include("./linked_encode_helper.jl")



function linked_from_inner_to_outer_one_codebook!(
    X::Matrix{T},                 
    B::Matrix{BType},             
    len_rate_each::Matrix{T},     
    C::Vector{Matrix{T}},         
    parent::Vector{UInt32},       
    current_depths::Vector{Int},  
    C_contiguous::Array{T,3},
    pre,
    max_depth::Integer,
    root_percentile::Float64 = 0.1,
    num_layers::Int = 8,
    knn_k::Int = 25,
    depth_k::Int = 6,
    icm_round::Int = 0;
    layer_relinked_rounds::Int = 5,
    verbose::Bool = false
) where {T<:AbstractFloat, BType<:Integer}

    m = size(B,1)
    d, n = size(X)
    h_max = size(C_contiguous, 2)
    h_j_vec = [size(C[j],2) for j in 1:m]
    H = sum(h_j_vec)
    pool_size = max(2, Threads.nthreads() * 2)
    tls_proto = make_tls(T, BType, d, m, H)
    tls_pool = Channel{typeof(tls_proto)}(pool_size)
    put!(tls_pool, tls_proto)
    for _ in 2:pool_size
        put!(tls_pool, make_tls(T, BType, d, m, H))
    end

    
    first_layer_codes = B[1, :]
    unique_codes = unique(first_layer_codes)
    cluster_indices = Dict{Int, Vector{Int}}()
    for code in unique_codes
        cluster_indices[code] = findall(first_layer_codes .== code)
    end

    codes = collect(unique_codes)
    clusters_total = length(codes)

    
    cos_angles = compute_cosine_angles(X, C, cluster_indices)
    layers = partition_by_cosine_percentiles(cos_angles, cluster_indices, num_layers, root_percentile)

    
    layer_of_point = zeros(Int, n)
    for (code, arr_layers) in layers
        for ℓ in 1:num_layers
            lst = arr_layers[ℓ]; len = length(lst)
            for i = 1:len
                layer_of_point[lst[i]] = ℓ
            end
        end
    end

    
    R_full = compute_reconstructed(B, len_rate_each, C)

    base_mse = Vector{T}(undef, n)
    all_errors!(base_mse,X,R_full)

    
    per_layer_updates = [Threads.Atomic{Int}(0) for _ in 1:num_layers]
    parent_from_counts = Array{Threads.Atomic{Int}}(undef, num_layers, num_layers)
    for i = 1:num_layers, j = 1:num_layers
        parent_from_counts[i,j] = Threads.Atomic{Int}(0)
    end
    total_updates = Threads.Atomic{Int}(0)
    total_done = Threads.Atomic{Int}(0)
    tls_pool_count = Threads.Atomic{Int}(0)
    tls_local_count = Threads.Atomic{Int}(0)
    println("begin processing...")

    
    Threads.@sync for code in codes
        Threads.@spawn begin
        
            tls, tls_from_pool = _pool_take_or_new(tls_pool, () -> make_tls(T, BType, d, m, H))
            if tls_from_pool
                Threads.atomic_add!(tls_pool_count, 1)
            else
                Threads.atomic_add!(tls_local_count, 1)
            end
            try
            
            R_self        = tls.R_self
            re            = tls.re
            B_ini         = tls.B_init
            a_ini         = tls.a_init
            a_bak         = tls.a_bak
            B_best        = tls.B_best
            a_best        = tls.a_best
            xC_base_buf   = tls.xC_base_buf
            xC_buf        = tls.xC_buf
            rC_buf        = tls.rC_buf
            order         = tls.order_idx
            processed     = tls.processed
            candidates    = tls.candidates
            A             = tls.A

            
            cidxs = get(cluster_indices, code, Int[])
            if isempty(cidxs); return; end
            g2rel = Dict{Int,Int}()
            for (pos, gi) in enumerate(cidxs)
                g2rel[gi] = pos
            end

            
            @views local_view = view(X, :, cidxs)
            tree_cluster = KDTree(local_view)

            
            arr_layers_all = get(layers, code, [Int[] for _ in 1:num_layers])

            
            for layer_idx in 2:num_layers
                current_layer = arr_layers_all[layer_idx]
                if isempty(current_layer); continue; end

                
                resize!(order, length(current_layer))
                for i in eachindex(order); order[i] = i; end
                Random.shuffle!(order)
                

                
                resize!(processed, length(current_layer))
                fill!(processed, false)

                
                inner_lo = max(1, layer_idx - depth_k)
                inner_hi = layer_idx - 1
                
                Kp = min(length(cidxs), max(32, 4 * (inner_hi - inner_lo + 1 + 1) * knn_k))

                Xi = Matrix{T}(undef, d, length(current_layer))
                cll = length(current_layer)
                for col = 1:cll
                    gi = current_layer[col]
                    for r = 1:d
                        Xi[r, col] = X[r, gi]
                    end
                end
                idxs_list, _ = knn(tree_cluster, Xi, Kp, true)  


                
                
                layer_pos_map = Dict{Int,Int}()
                for (k, gi) in enumerate(current_layer)
                    layer_pos_map[gi] = k
                end

                
                for pos_in_layer in order
                    i_global = current_layer[pos_in_layer]
                    @views xi = X[:, i_global]

                    
                    empty!(candidates)

                    neigh_rel = idxs_list[pos_in_layer]
                    per_layer_budget = fill(knn_k, num_layers)

                    
                    len_nei = length(neigh_rel)
                    for idx = 1:len_nei
                        c_rel = neigh_rel[idx]
                        p_global = cidxs[c_rel]
                        if p_global == i_global; continue; end

                        ℓp = layer_of_point[p_global]
                        if (ℓp >= inner_lo) & (ℓp <= inner_hi)
                            if per_layer_budget[ℓp] > 0
                                push!(candidates, p_global)
                                per_layer_budget[ℓp] -= 1
                            end
                        elseif ℓp == layer_idx
                            
                            kpos = get(layer_pos_map, p_global, 0)
                            if kpos != 0 && processed[kpos] && per_layer_budget[layer_idx] > 0
                                push!(candidates, p_global)
                                per_layer_budget[layer_idx] -= 1
                            end
                        end

                        
                        if length(candidates) >= ((inner_hi - inner_lo + 1) * knn_k + knn_k)
                            break
                        end
                    end

                    processed[pos_in_layer] = true
                    if isempty(candidates)
                        continue
                    end

                    
                    if icm_round > 0
                        pz = 1
                        for ℓ in 1:m
                            hℓ = h_j_vec[ℓ]
                            for k in 1:hℓ
                                s = zero(T)
                                for i = 1:d
                                    s += C_contiguous[i, k, ℓ] * xi[i]
                                end
                                xC_base_buf[pz] = s
                                pz += 1
                            end
                        end
                    end

                    
                    bestE2 = base_mse[i_global]
                    best_parent = 0               
                    for p_idx in candidates
                        new_depth = current_depths[p_idx] + 1
                        if new_depth > max_depth; continue; end
                        for r = 1:d
                            re[r] = xi[r] - R_full[r, p_idx]
                        end

                        quantize_with_kmeans_abs_nonormal_single_extend_allcodes_fast!(
                            re, B_ini, a_ini, m, d, H, pre.C_all, pre.G, pre.offsets, pre.flat_layer, pre.invnorm_flat, rC_buf)

                        optimized_least_squares_single!(a_ini, re, C_contiguous, B_ini, A, m, d, h_max)

                        if icm_round > 0
                            
                            build_xC_for_parent_single!(
                                xC_buf,           
                                xC_base_buf,      
                                len_rate_each,    
                                B,                
                                parent,
                                p_idx,
                                pre,
                            )

                            
                            dynamic_icm_encoding_single_for_residual!(
                                re, A, B_ini, a_ini,
                                C_contiguous, pre,
                                xC_buf, rC_buf,
                                a_bak,icm_round
                            )
                        end

                        reconstruct_node!(R_self, B_ini, a_ini, C)
                        newE2 = zero(T)
                        for r = 1:d
                            diff = re[r] - R_self[r]
                            newE2 += diff * diff
                        end

                        if newE2 < bestE2                                   
                            bestE2 = newE2
                            best_parent = p_idx
                            for l = 1:m
                                B_best[l] = B_ini[l]
                                a_best[l] = a_ini[l]
                            end
                        end
                    end

                    
                    if best_parent != 0                    
                        parent[i_global] = UInt32(best_parent)
                        for l = 1:m
                            B[l, i_global] = B_best[l]
                            len_rate_each[l, i_global] = a_best[l]
                        end
                        current_depths[i_global] = current_depths[best_parent] + 1

                        reconstruct_node!(R_self, view(B, :, i_global), view(len_rate_each, :, i_global), C)
                        p_idx = Int(parent[i_global])
                        for r = 1:d
                            R_full[r, i_global] = R_full[r, p_idx] + R_self[r]
                        end

                        
                        Threads.atomic_add!(per_layer_updates[layer_idx], 1)
                        Threads.atomic_add!(total_updates, 1)
                        parent_from_layer = layer_of_point[best_parent]
                        Threads.atomic_add!(parent_from_counts[layer_idx, parent_from_layer], 1)
                    end
                end 
            end 

            Threads.atomic_add!(total_done, 1)
            i_total_done = total_done[]
            ress=i_total_done % 10
            if ress==0
                @printf(stderr,"\rprocessed %d/%d",i_total_done,clusters_total)
                flush(stderr)
            end
        finally
            if tls_from_pool
                put!(tls_pool, tls)
            end
        end
        end 
    end 



    
    println("tls from pool: $(tls_pool_count[]), tls local: $(tls_local_count[])")
    if verbose
        for layer_idx in 2:num_layers
            upd = per_layer_updates[layer_idx][]
            verbose && println("Sphere layer $layer_idx: updated $upd nodes")
            if upd > 0
                
                for t in 1:layer_idx
                    cnt = parent_from_counts[layer_idx, t][]
                    pct = round(cnt / upd * 100; digits=2)
                    println("parent from layer $t: $pct %")
                end
            else
                for t in 1:layer_idx
                    println("parent from layer $t: 0.0 %")
                end
            end
        end
    end
    println("\nTotal updates: $(total_updates[]) nodes")
    return R_full
end



function linked_from_inner_to_outer_one_codebook_ils!(
    X::Matrix{T},                 
    B::Matrix{BType},             
    len_rate_each::Matrix{T},     
    C::Vector{Matrix{T}},         
    parent::Vector{UInt32},       
    current_depths::Vector{Int},  
    C_contiguous::Array{T,3},
    pre,
    max_depth::Integer,
    root_percentile::Float64 = 0.1,
    num_layers::Int = 8,
    knn_k::Int = 25,
    depth_k::Int = 6,
    icm_round::Int = 0;
    layer_relinked_rounds::Int = 5,
    ils_rounds = 4,         
    ils_perturb_layers = 3,         
    verbose::Bool = false,
) where {T<:AbstractFloat, BType<:Integer}

    m = size(B,1)
    d, n = size(X)
    h_max = size(C_contiguous, 2)
    h_j_vec = [size(C[j],2) for j in 1:m]
    H = sum(h_j_vec)
    pool_size = max(2, Threads.nthreads() * 2)
    tls_proto = make_tls_ils(T, BType, d, m, H)
    tls_pool = Channel{typeof(tls_proto)}(pool_size)
    put!(tls_pool, tls_proto)
    for _ in 2:pool_size
        put!(tls_pool, make_tls_ils(T, BType, d, m, H))
    end

    
    first_layer_codes = B[1, :]
    unique_codes = unique(first_layer_codes)
    cluster_indices = Dict{Int, Vector{Int}}()
    for code in unique_codes
        cluster_indices[code] = findall(first_layer_codes .== code)
    end

    codes = collect(unique_codes)
    clusters_total = length(codes)

    
    cos_angles = compute_cosine_angles(X, C, cluster_indices)
    layers = partition_by_cosine_percentiles(cos_angles, cluster_indices, num_layers, root_percentile)

    
    layer_of_point = zeros(Int, n)
    for (code, arr_layers) in layers
        for ℓ in 1:num_layers
            lst = arr_layers[ℓ]; len = length(lst)
            for i = 1:len
                layer_of_point[lst[i]] = ℓ
            end
        end
    end

    
    R_full = compute_reconstructed(B, len_rate_each, C)

    base_mse = Vector{T}(undef, n)
    all_errors!(base_mse,X,R_full)

    
    per_layer_updates = [Threads.Atomic{Int}(0) for _ in 1:num_layers]
    parent_from_counts = Array{Threads.Atomic{Int}}(undef, num_layers, num_layers)
    for i = 1:num_layers, j = 1:num_layers
        parent_from_counts[i,j] = Threads.Atomic{Int}(0)
    end
    total_updates = Threads.Atomic{Int}(0)
    total_done = Threads.Atomic{Int}(0)
    clusters_started = Threads.Atomic{Int}(0)
    println("begin processing...")

    
    Threads.@sync for code in codes
        Threads.@spawn begin
            tls, tls_from_pool = _pool_take_or_new(tls_pool, () -> make_tls_ils(T, BType, d, m, H))

            try
            
            R_self        = tls.R_self
            re            = tls.re
            B_ini         = tls.B_init
            a_ini         = tls.a_init
            a_bak         = tls.a_bak
            B_best        = tls.B_best
            a_best        = tls.a_best
            xC_base_buf   = tls.xC_base_buf
            xC_buf        = tls.xC_buf
            rC_buf        = tls.rC_buf
            order         = tls.order_idx
            processed     = tls.processed
            candidates    = tls.candidates
            B_try         =  tls.B_try
            a_try         = tls.a_try
            pert_layers   = tls.pert_layers
            new_flat      = tls.new_flat
            A             = tls.A

            
            cidxs = get(cluster_indices, code, Int[])
            if isempty(cidxs); return; end
            g2rel = Dict{Int,Int}()
            for (pos, gi) in enumerate(cidxs)
                g2rel[gi] = pos
            end

            
            @views local_view = view(X, :, cidxs)
            tree_cluster = KDTree(local_view)

            
            arr_layers_all = get(layers, code, [Int[] for _ in 1:num_layers])

            
            for layer_idx in 2:num_layers
                current_layer = arr_layers_all[layer_idx]
                if isempty(current_layer); continue; end

                
                resize!(order, length(current_layer))
                for i in eachindex(order); order[i] = i; end
                Random.shuffle!(order)
                

                
                resize!(processed, length(current_layer))
                fill!(processed, false)

                
                inner_lo = max(1, layer_idx - depth_k)
                inner_hi = layer_idx - 1
                
                Kp = min(length(cidxs), max(32, 4 * (inner_hi - inner_lo + 1 + 1) * knn_k))

                Xi = Matrix{T}(undef, d, length(current_layer))
                cll = length(current_layer)
                for col = 1:cll
                    gi = current_layer[col]
                    for r = 1:d
                        Xi[r, col] = X[r, gi]
                    end
                end
                idxs_list, _ = knn(tree_cluster, Xi, Kp, true)  


                
                
                layer_pos_map = Dict{Int,Int}()
                for (k, gi) in enumerate(current_layer)
                    layer_pos_map[gi] = k
                end

                
                for pos_in_layer in order
                    i_global = current_layer[pos_in_layer]
                    @views xi = X[:, i_global]

                    
                    empty!(candidates)

                    neigh_rel = idxs_list[pos_in_layer]
                    per_layer_budget = fill(knn_k, num_layers)

                    
                    len_nei = length(neigh_rel)
                    for idx = 1:len_nei
                        c_rel = neigh_rel[idx]
                        p_global = cidxs[c_rel]
                        if p_global == i_global; continue; end

                        ℓp = layer_of_point[p_global]
                        if (ℓp >= inner_lo) & (ℓp <= inner_hi)
                            if per_layer_budget[ℓp] > 0
                                push!(candidates, p_global)
                                per_layer_budget[ℓp] -= 1
                            end
                        elseif ℓp == layer_idx
                            
                            kpos = get(layer_pos_map, p_global, 0)
                            if kpos != 0 && processed[kpos] && per_layer_budget[layer_idx] > 0
                                push!(candidates, p_global)
                                per_layer_budget[layer_idx] -= 1
                            end
                        end

                        
                        if length(candidates) >= ((inner_hi - inner_lo + 1) * knn_k + knn_k)
                            break
                        end
                    end

                    processed[pos_in_layer] = true
                    if isempty(candidates)
                        continue
                    end

                    
                    if icm_round > 0
                        mul!(xC_base_buf, transpose(pre.C_all), xi)
                    end

                    
                    bestE2 = base_mse[i_global]
                    best_parent = 0                 
                    for p_idx in candidates
                        new_depth = current_depths[p_idx] + 1
                        if new_depth > max_depth; continue; end
                        for r = 1:d
                            re[r] = xi[r] - R_full[r, p_idx]
                        end

                        quantize_with_kmeans_abs_nonormal_single_extend_allcodes_fast!(
                            re, B_ini, a_ini, m, d, H, pre.C_all, pre.G, pre.offsets, pre.flat_layer, pre.invnorm_flat, rC_buf)

                        optimized_least_squares_single!(a_ini, re, C_contiguous, B_ini, A, m, d, h_max)

                        
                        if icm_round > 0
                            
                            build_xC_for_parent_single!(
                                xC_buf,           
                                xC_base_buf,      
                                len_rate_each,    
                                B,                
                                parent,
                                p_idx,
                                pre,
                            )

                            
                            dynamic_icm_encoding_single_for_residual!(
                                re, A, B_ini, a_ini,
                                C_contiguous, pre,
                                xC_buf, rC_buf,
                                a_bak,icm_round
                            )

                        end

                        
                        reconstruct_node!(R_self, B_ini, a_ini, C)
                        E2_best = zero(T)
                        @inbounds @simd for t in 1:d
                            diff = re[t] - R_self[t]
                            E2_best += diff * diff
                        end

                        
                        @inbounds for il in 1:ils_rounds
                            
                            @inbounds for t in 1:ils_perturb_layers
                                
                                ℓ = rand(1:m)
                                

                                
                                hℓ = h_j_vec[ℓ]

                                
                                old_code  = Int(B_ini[ℓ])

                                
                                
                                tcode     = rand(0:(hℓ-2))      
                                new_code  = tcode + 1           
                                if new_code >= old_code
                                    new_code += 1               
                                end

                                
                                startf = pre.offsets[ℓ]
                                newf   = startf + new_code - 1   

                                pert_layers[t] = ℓ
                                new_flat[t]    = newf
                            end


                            
                            copyto!(B_try, B_ini)
                            @inbounds for t in 1:ils_perturb_layers
                                ℓ = pert_layers[t]
                                B_try[ℓ] = BType(new_flat[t] - pre.offsets[ℓ] + 1)  
                            end

                            
                            optimized_least_squares_single!(
                                a_try, re, C_contiguous, B_try, A, m, d, h_max
                            )

                            
                            
                            if icm_round > 0
                                dynamic_icm_encoding_single_for_residual!(
                                    re, A, B_try, a_try,
                                    C_contiguous, pre,
                                    xC_buf, rC_buf,
                                    a_bak,icm_round
                                )
                            end
                            
                            reconstruct_node!(R_self, B_try, a_try, C)
                            E2_try = zero(T)
                            @inbounds @simd for t in 1:d
                                diff = re[t] - R_self[t]
                                E2_try += diff * diff
                            end

                            if E2_try < E2_best
                                E2_best = E2_try
                                copyto!(B_ini, B_try)
                                copyto!(a_ini, a_try)
                            end
                        end
                        

                        if E2_best < bestE2                               
                            bestE2 = E2_best
                            best_parent = p_idx
                            for l = 1:m
                                B_best[l] = B_ini[l]
                                a_best[l] = a_ini[l]
                            end
                        end
                    end



                    if best_parent != 0
                        parent[i_global] = UInt32(best_parent)
                        for l = 1:m
                            B[l, i_global] = B_best[l]
                            len_rate_each[l, i_global] = a_best[l]
                        end
                        current_depths[i_global] = current_depths[best_parent] + 1

                        reconstruct_node!(R_self, view(B, :, i_global), view(len_rate_each, :, i_global), C)
                        p_idx = Int(parent[i_global])
                        for r = 1:d
                            R_full[r, i_global] = R_full[r, p_idx] + R_self[r]
                        end

                        
                        Threads.atomic_add!(per_layer_updates[layer_idx], 1)
                        Threads.atomic_add!(total_updates, 1)
                        parent_from_layer = layer_of_point[best_parent]
                        Threads.atomic_add!(parent_from_counts[layer_idx, parent_from_layer], 1)
                    end
                end 
            end 

            Threads.atomic_add!(total_done, 1)
            i_total_done = total_done[]
            ress=i_total_done % 10
            if ress==0
                @printf(stderr,"\rprocessed %d/%d",i_total_done,clusters_total)
                flush(stderr)
            end
        finally
            if tls_from_pool
                put!(tls_pool, tls)
            end
        end
        end 
    end 

    if verbose
        
        for layer_idx in 2:num_layers
            upd = per_layer_updates[layer_idx][]
            verbose && println("Sphere layer $layer_idx: updated $upd nodes")
            if upd > 0
                
                for t in 1:layer_idx
                    cnt = parent_from_counts[layer_idx, t][]
                    pct = round(cnt / upd * 100; digits=2)
                    println("parent from layer $t: $pct %")
                end
            else
                for t in 1:layer_idx
                    println("parent from layer $t: 0.0 %")
                end
            end
        end
    end
    println("\nTotal updates: $(total_updates[]) nodes")
    return R_full
end


function linked_from_inner_to_outer_two_codebook!(
    X::AbstractMatrix{T},          
    B::AbstractMatrix{BType},      
    len_rate_each::AbstractMatrix{T}, 
    parent::Vector{UInt32}, 
    current_depths::Vector{Int},
    C_root::Vector{Matrix{T}},  
    pre_root, 
    C_one::Vector{Matrix{T}},  
    C_contiguous_one::Array{T, 3}, 
    pre_one, 
    max_depth::Integer,    
    root_percentile::Float64 = 0.1,  
    num_layers::Int = 5,    
    knn_k::Int = 10,        
    depth_k::Int = 2,       
    icm_round::Int = 2;         
    layer_relinked_rounds::Int = 5,
    verbose = false
) where {T<:AbstractFloat, BType<:Integer}

    m = size(B, 1)
    d, n = size(X)
    h_j_vec = [size(C_one[i],2) for i in 1:m]
    h_max = maximum(h_j_vec)
    H = sum(h_j_vec)
    pool_size = max(2, Threads.nthreads() * 2)
    tls_proto = make_tls(T, BType, d, m, H)
    tls_pool = Channel{typeof(tls_proto)}(pool_size)
    put!(tls_pool, tls_proto)
    for _ in 2:pool_size
        put!(tls_pool, make_tls(T, BType, d, m, H))
    end
   
    G_one_root = build_one_root_cross_gram(pre_one, pre_root)

    
    first_layer_codes = B[1, :]
    unique_codes = unique(first_layer_codes)
    cluster_indices = Dict{Int, Vector{Int}}()
    for code in unique_codes
        cluster_indices[code] = findall(first_layer_codes .== code)
    end

    codes = collect(unique_codes)
    clusters_total = length(codes)

    
    cos_angles = compute_cosine_angles(X, C_root, cluster_indices)
    layers = partition_by_cosine_percentiles(cos_angles, cluster_indices, num_layers, root_percentile)

    
    layer_of_point = zeros(Int, n)
    for (code, arr_layers) in layers
        for ℓ in 1:num_layers
            lst = arr_layers[ℓ]; len = length(lst)
            @inbounds for i = 1:len
                layer_of_point[lst[i]] = ℓ
            end
        end
    end

    
    R_full = compute_reconstructed(B, len_rate_each, C_root)

    base_mse = Vector{T}(undef, n)
    all_errors!(base_mse,X,R_full)

    
    per_layer_updates = [Threads.Atomic{Int}(0) for _ in 1:num_layers]
    parent_from_counts = Array{Threads.Atomic{Int}}(undef, num_layers, num_layers)
    @inbounds for i = 1:num_layers, j = 1:num_layers
        parent_from_counts[i,j] = Threads.Atomic{Int}(0)
    end
    total_updates = Threads.Atomic{Int}(0)
    total_done = Threads.Atomic{Int}(0)
    println("begin processing...")
   
    
    Threads.@sync for code in codes
        Threads.@spawn begin
            tls, tls_from_pool = _pool_take_or_new(tls_pool, () -> make_tls_ils(T, BType, d, m, H))
          
            try
            
            R_self        = tls.R_self
            re            = tls.re
            B_ini         = tls.B_init
            a_ini         = tls.a_init
            a_bak         = tls.a_bak
            B_best        = tls.B_best
            a_best        = tls.a_best
            xC_base_buf   = tls.xC_base_buf
            xC_buf        = tls.xC_buf
            rC_buf        = tls.rC_buf
            order         = tls.order_idx
            processed     = tls.processed
            candidates    = tls.candidates       
            A             = tls.A
            
            cidxs = get(cluster_indices, code, Int[])
            if isempty(cidxs); return; end
            g2rel = Dict{Int,Int}()
            @inbounds for (pos, gi) in enumerate(cidxs)
                g2rel[gi] = pos
            end

            
            @views local_view = view(X, :, cidxs)
            tree_cluster = KDTree(local_view)

            
            arr_layers_all = get(layers, code, [Int[] for _ in 1:num_layers])

            
            for layer_idx in 2:num_layers
                current_layer = arr_layers_all[layer_idx]
                if isempty(current_layer); continue; end

                
                resize!(order, length(current_layer))
                @inbounds for i in eachindex(order); order[i] = i; end
                
                Random.shuffle!(order)
                

                
                resize!(processed, length(current_layer))
                fill!(processed, false)

                
                inner_lo = max(1, layer_idx - depth_k)
                inner_hi = layer_idx - 1
                
                Kp = min(length(cidxs), max(32, 4 * (inner_hi - inner_lo + 1 + 1) * knn_k))

                Xi = Matrix{T}(undef, d, length(current_layer))
                cll = length(current_layer)
                @inbounds for col = 1:cll
                    gi = current_layer[col]
                    @inbounds for r = 1:d
                        Xi[r, col] = X[r, gi]
                    end
                end
                idxs_list, _ = knn(tree_cluster, Xi, Kp, true)  

                
                
                layer_pos_map = Dict{Int,Int}()
                @inbounds for (k, gi) in enumerate(current_layer)
                    layer_pos_map[gi] = k
                end
                
                
                for pos_in_layer in order
                    i_global = current_layer[pos_in_layer]
                    @views xi = X[:, i_global]

                    
                    empty!(candidates)

                    neigh_rel = idxs_list[pos_in_layer]
                    per_layer_budget = fill(knn_k, num_layers)

                    
                    len_nei = length(neigh_rel)
                    @inbounds for idx = 1:len_nei
                        c_rel = neigh_rel[idx]
                        p_global = cidxs[c_rel]
                        if p_global == i_global; continue; end

                        ℓp = layer_of_point[p_global]
                        if (ℓp >= inner_lo) & (ℓp <= inner_hi)
                            if per_layer_budget[ℓp] > 0
                                push!(candidates, p_global)
                                per_layer_budget[ℓp] -= 1
                            end
                        elseif ℓp == layer_idx
                            
                            kpos = get(layer_pos_map, p_global, 0)
                            if kpos != 0 && processed[kpos] && per_layer_budget[layer_idx] > 0
                                push!(candidates, p_global)
                                per_layer_budget[layer_idx] -= 1
                            end
                        end

                        
                        if length(candidates) >= ((inner_hi - inner_lo + 1) * knn_k + knn_k)
                            break
                        end
                    end

                    processed[pos_in_layer] = true
                    if isempty(candidates)
                        continue
                    end

                    
                    if icm_round > 0
                        mul!(xC_base_buf, transpose(pre_one.C_all), xi)
                    end

                    
                    bestE2 = base_mse[i_global]
                    best_parent = 0
                    @inbounds for p_idx in candidates
                        new_depth = current_depths[p_idx] + 1
                        if new_depth > max_depth; continue; end
                        @inbounds for r = 1:d
                            re[r] = xi[r] - R_full[r, p_idx]
                        end

                        quantize_with_kmeans_abs_nonormal_single_extend_allcodes_fast!(re, B_ini, a_ini, m, d, H, pre_one.C_all,pre_one.G,pre_one.offsets,pre_one.flat_layer,pre_one.invnorm_flat,rC_buf)

                        optimized_least_squares_single!(a_ini, re, C_contiguous_one, B_ini, A, m, d, h_max)

                        if icm_round > 0
                            
                            build_xC_one_for_parent!(xC_buf, xC_base_buf, len_rate_each,  B, parent,current_depths,p_idx,pre_root, pre_one,pre_one.G,G_one_root)

                            
                            dynamic_icm_encoding_single_for_residual!(re, A, B_ini, a_ini,C_contiguous_one, pre_one,xC_buf, rC_buf,a_bak, icm_round)
                        end

                        reconstruct_node!(R_self, B_ini, a_ini, C_one)
                        newE2 = zero(T)
                        @inbounds for r = 1:d
                            diff = re[r] - R_self[r]
                            newE2 += diff * diff
                        end
                        if newE2 < bestE2                                           
                            bestE2 = newE2
                            best_parent = p_idx
                            @inbounds for l = 1:m
                                B_best[l] = B_ini[l]
                                a_best[l] = a_ini[l]
                            end                       
                        end
                    end

                    
                    if best_parent != 0                    
                        parent[i_global] = UInt32(best_parent)
                        
                        @inbounds for l = 1:m
                            B[l, i_global] = B_best[l]
                            len_rate_each[l, i_global] = a_best[l]
                        end
                        current_depths[i_global] = current_depths[best_parent] + 1

                        reconstruct_node!(R_self, view(B, :, i_global), view(len_rate_each, :, i_global), C_one)
                        p_idx = Int(parent[i_global])
                        @inbounds for r = 1:d
                            R_full[r, i_global] = R_full[r, p_idx] + R_self[r]
                        end

                        
                        Threads.atomic_add!(per_layer_updates[layer_idx], 1)
                        Threads.atomic_add!(total_updates, 1)
                        parent_from_layer = layer_of_point[best_parent]
                        Threads.atomic_add!(parent_from_counts[layer_idx, parent_from_layer], 1)
                    end
                end 
            end 

Threads.atomic_add!(total_done, 1)
            i_total_done = total_done[]
            ress=i_total_done % 10
            if ress==0
                @printf(stderr,"\rprocessed %d/%d",i_total_done,clusters_total)
                flush(stderr)
            end
        finally
            if tls_from_pool
                put!(tls_pool, tls)
            end
        end
        end 
    end 

    
    if verbose 
        for layer_idx in 2:num_layers
            upd = per_layer_updates[layer_idx][]
            println("Sphere layer $layer_idx: updated $upd nodes")
            if upd > 0
                
                for t in 1:layer_idx
                    cnt = parent_from_counts[layer_idx, t][]
                    pct = round(cnt / upd * 100; digits=2)
                    println("parent from layer $t: $pct %")
                end
            else
                for t in 1:layer_idx
                    println("parent from layer $t: 0.0 %")
                end
            end
        end
    end
    println("\nTotal updates: $(total_updates[]) nodes")
    return R_full
end



function linked_from_inner_to_outer_two_codebook_ils!(
    X::AbstractMatrix{T},          
    B::AbstractMatrix{BType},      
    len_rate_each::AbstractMatrix{T}, 
    parent::Vector{UInt32}, 
    current_depths::Vector{Int},
    C_root::Vector{Matrix{T}},  
    pre_root, 
    C_one::Vector{Matrix{T}},  
    C_contiguous_one::Array{T, 3}, 
    pre_one, 
    max_depth::Integer,    
    root_percentile::Float64 = 0.1,  
    num_layers::Int = 5,    
    knn_k::Int = 10,        
    depth_k::Int = 2,       
    icm_round::Int = 2;         
    layer_relinked_rounds::Int = 5,
    ils_rounds = 4,         
    ils_perturb_layers = 3,         
    verbose = false
) where {T<:AbstractFloat, BType<:Integer}

    m = size(B, 1)
    d, n = size(X)
    h_j_vec = [size(C_one[i],2) for i in 1:m]
    h_max = maximum(h_j_vec)
    H = sum(h_j_vec)
    pool_size = max(2, Threads.nthreads() * 2)
    tls_proto = make_tls_ils(T, BType, d, m, H)
    tls_pool = Channel{typeof(tls_proto)}(pool_size)
    put!(tls_pool, tls_proto)
    for _ in 2:pool_size
        put!(tls_pool, make_tls_ils(T, BType, d, m, H))
    end

    G_one_root = build_one_root_cross_gram(pre_one, pre_root)

    
    first_layer_codes = B[1, :]
    unique_codes = unique(first_layer_codes)
    cluster_indices = Dict{Int, Vector{Int}}()
    for code in unique_codes
        cluster_indices[code] = findall(first_layer_codes .== code)
    end

    codes = collect(unique_codes)
    clusters_total = length(codes)

    
    cos_angles = compute_cosine_angles(X, C_root, cluster_indices)
    layers = partition_by_cosine_percentiles(cos_angles, cluster_indices, num_layers, root_percentile)

    
    layer_of_point = zeros(Int, n)
    for (code, arr_layers) in layers
        for ℓ in 1:num_layers
            lst = arr_layers[ℓ]; len = length(lst)
            @inbounds for i = 1:len
                layer_of_point[lst[i]] = ℓ
            end
        end
    end

    
    R_full = compute_reconstructed(B, len_rate_each, C_root)

    base_mse = Vector{T}(undef, n)
    all_errors!(base_mse,X,R_full)

    
    per_layer_updates = [Threads.Atomic{Int}(0) for _ in 1:num_layers]
    parent_from_counts = Array{Threads.Atomic{Int}}(undef, num_layers, num_layers)
    @inbounds for i = 1:num_layers, j = 1:num_layers
        parent_from_counts[i,j] = Threads.Atomic{Int}(0)
    end
    total_updates = Threads.Atomic{Int}(0)
    total_done = Threads.Atomic{Int}(0)
    println("begin processing...")
   
    
    Threads.@sync for code in codes
        Threads.@spawn begin
            tls, tls_from_pool = _pool_take_or_new(tls_pool, () -> make_tls_ils(T, BType, d, m, H))
        
            try
            
            R_self        = tls.R_self
            re            = tls.re
            B_ini         = tls.B_init
            a_ini         = tls.a_init
            a_bak         = tls.a_bak
            B_best        = tls.B_best
            a_best        = tls.a_best
            xC_base_buf   = tls.xC_base_buf
            xC_buf        = tls.xC_buf
            rC_buf        = tls.rC_buf
            order         = tls.order_idx
            processed     = tls.processed
            candidates    = tls.candidates
            B_try       =  tls.B_try
            a_try       = tls.a_try
            pert_layers = tls.pert_layers
            new_flat    = tls.new_flat
            A             = tls.A

            
            cidxs = get(cluster_indices, code, Int[])
            if isempty(cidxs); return; end
            g2rel = Dict{Int,Int}()
            @inbounds for (pos, gi) in enumerate(cidxs)
                g2rel[gi] = pos
            end

            
            @views local_view = view(X, :, cidxs)
            tree_cluster = KDTree(local_view)

            
            arr_layers_all = get(layers, code, [Int[] for _ in 1:num_layers])

            
            for layer_idx in 2:num_layers
                current_layer = arr_layers_all[layer_idx]
                if isempty(current_layer); continue; end

                
                resize!(order, length(current_layer))
                @inbounds for i in eachindex(order); order[i] = i; end
                
                Random.shuffle!(order)
                

                
                resize!(processed, length(current_layer))
                fill!(processed, false)

                
                inner_lo = max(1, layer_idx - depth_k)
                inner_hi = layer_idx - 1
                
                Kp = min(length(cidxs), max(32, 4 * (inner_hi - inner_lo + 1 + 1) * knn_k))

                Xi = Matrix{T}(undef, d, length(current_layer))
                cll = length(current_layer)
                @inbounds for col = 1:cll
                    gi = current_layer[col]
                    @inbounds for r = 1:d
                        Xi[r, col] = X[r, gi]
                    end
                end
                idxs_list, _ = knn(tree_cluster, Xi, Kp, true)  

                
                
                layer_pos_map = Dict{Int,Int}()
                @inbounds for (k, gi) in enumerate(current_layer)
                    layer_pos_map[gi] = k
                end

                
                for pos_in_layer in order
                    i_global = current_layer[pos_in_layer]
                    @views xi = X[:, i_global]

                    
                    empty!(candidates)

                    neigh_rel = idxs_list[pos_in_layer]
                    per_layer_budget = fill(knn_k, num_layers)

                    
                    len_nei = length(neigh_rel)
                    @inbounds for idx = 1:len_nei
                        c_rel = neigh_rel[idx]
                        p_global = cidxs[c_rel]
                        if p_global == i_global; continue; end

                        ℓp = layer_of_point[p_global]
                        if (ℓp >= inner_lo) & (ℓp <= inner_hi)
                            if per_layer_budget[ℓp] > 0
                                push!(candidates, p_global)
                                per_layer_budget[ℓp] -= 1
                            end
                        elseif ℓp == layer_idx
                            
                            kpos = get(layer_pos_map, p_global, 0)
                            if kpos != 0 && processed[kpos] && per_layer_budget[layer_idx] > 0
                                push!(candidates, p_global)
                                per_layer_budget[layer_idx] -= 1
                            end
                        end

                        
                        if length(candidates) >= ((inner_hi - inner_lo + 1) * knn_k + knn_k)
                            break
                        end
                    end
                    
                    processed[pos_in_layer] = true
                    if isempty(candidates)
                        continue
                    end

                    
                    if icm_round > 0
                        mul!(xC_base_buf, transpose(pre_one.C_all), xi)
                    end

                    
                    bestE2 = base_mse[i_global]
                    best_parent = 0
                    @inbounds for p_idx in candidates
                        new_depth = current_depths[p_idx] + 1
                        if new_depth > max_depth; continue; end
                        @inbounds for r = 1:d
                            re[r] = xi[r] - R_full[r, p_idx]
                        end

                        quantize_with_kmeans_abs_nonormal_single_extend_allcodes_fast!(re, B_ini, a_ini, m, d, H, pre_one.C_all,pre_one.G,pre_one.offsets,pre_one.flat_layer,pre_one.invnorm_flat,rC_buf)

                        optimized_least_squares_single!(a_ini, re, C_contiguous_one, B_ini, A, m, d, h_max)

                    
                        if icm_round > 0
                            
                            
                            build_xC_one_for_parent!(
                                xC_buf,            
                                xC_base_buf,       
                                len_rate_each,     
                                B,                 
                                parent,
                                current_depths,
                                p_idx,
                                pre_root, pre_one,
                                pre_one.G,         
                                G_one_root         
                            )

                            
                            dynamic_icm_encoding_single_for_residual!(
                                re, A, B_ini, a_ini,
                                C_contiguous_one,
                                pre_one,
                                xC_buf, rC_buf,
                                a_bak,
                                icm_round
                            )
                            
                        end

                        
                        reconstruct_node!(R_self, B_ini, a_ini, C_one)
                        E2_best = zero(T)
                        @inbounds @simd for t in 1:d
                            diff = re[t] - R_self[t]
                            E2_best += diff * diff
                        end

                        
                        @inbounds for il in 1:ils_rounds
                            
                            @inbounds for t in 1:ils_perturb_layers
                                
                                ℓ = rand(1:m)
                                

                                
                                hℓ = h_j_vec[ℓ]

                                
                                old_code  = Int(B_ini[ℓ])

                                
                                
                                tcode     = rand(0:(hℓ-2))      
                                new_code  = tcode + 1           
                                if new_code >= old_code
                                    new_code += 1               
                                end

                                
                                startf = pre_one.offsets[ℓ]
                                newf   = startf + new_code - 1   

                                pert_layers[t] = ℓ
                                new_flat[t]    = newf
                            end

                            
                            copyto!(B_try, B_ini)
                            @inbounds for t in 1:ils_perturb_layers
                                ℓ = pert_layers[t]
                                
                                B_try[ℓ] = BType(new_flat[t] - pre_one.offsets[ℓ] + 1)
                            end

                            
                            optimized_least_squares_single!(a_try, re, C_contiguous_one, B_try, A, m, d, h_max)

                            
                            
                            if icm_round > 0
                                dynamic_icm_encoding_single_for_residual!(
                                    re, A, B_try, a_try,
                                    C_contiguous_one,
                                    pre_one,
                                    xC_buf, rC_buf,
                                    a_bak,
                                    icm_round
                                )
                            end
                            

                            
                            reconstruct_node!(R_self, B_try, a_try, C_one)
                            E2_try = zero(T)
                            @inbounds @simd for t in 1:d
                                diff = re[t] - R_self[t]
                                E2_try += diff * diff
                            end

                            if E2_try < E2_best
                                E2_best = E2_try
                                copyto!(B_ini, B_try)
                                copyto!(a_ini, a_try)
                            end
                        end
                        

                        if E2_best < bestE2                                         
                            bestE2 = E2_best
                            best_parent = p_idx
                            @inbounds for l = 1:m
                                B_best[l] = B_ini[l]
                                a_best[l] = a_ini[l]
                            end                       
                        end
                    end

                    
                    if best_parent != 0   
          
                        parent[i_global] = UInt32(best_parent)
                        @inbounds for l = 1:m
                            B[l, i_global] = B_best[l]
                            len_rate_each[l, i_global] = a_best[l]
                        end
                        current_depths[i_global] = current_depths[best_parent] + 1

                        reconstruct_node!(R_self, view(B, :, i_global), view(len_rate_each, :, i_global), C_one)
                        p_idx = Int(parent[i_global])
                        @inbounds for r = 1:d
                            R_full[r, i_global] = R_full[r, p_idx] + R_self[r]
                        end

                        
                        Threads.atomic_add!(per_layer_updates[layer_idx], 1)
                        Threads.atomic_add!(total_updates, 1)
                        parent_from_layer = layer_of_point[best_parent]
                        Threads.atomic_add!(parent_from_counts[layer_idx, parent_from_layer], 1)
                    end
                end 
            end 

            Threads.atomic_add!(total_done, 1)
            i_total_done = total_done[]
            ress=i_total_done % 10
            if ress==0
                @printf(stderr,"\rprocessed %d/%d",i_total_done,clusters_total)
                flush(stderr)
            end
        finally
            if tls_from_pool
                put!(tls_pool, tls)
            end
        end
        end 
    end 

    
    if verbose 
        for layer_idx in 2:num_layers
            upd = per_layer_updates[layer_idx][]
            println("Sphere layer $layer_idx: updated $upd nodes")
            if upd > 0
                
                for t in 1:layer_idx
                    cnt = parent_from_counts[layer_idx, t][]
                    pct = round(cnt / upd * 100; digits=2)
                    println("parent from layer $t: $pct %")
                end
            else
                for t in 1:layer_idx
                    println("parent from layer $t: 0.0 %")
                end
            end
        end
    end
    println("\nTotal updates: $(total_updates[]) nodes")
    return R_full
end








function linked_multi_center_two_codebook!(
    X::AbstractMatrix{T},                 
    B::AbstractMatrix{BType},             
    len_rate_each::AbstractMatrix{T},     
    parent::Vector{UInt32},       
    current_depths::Vector{Int},  

    C_root::Vector{Matrix{T}},          
    pre_root,                           

    C_one::Vector{Matrix{T}},           
    C_contiguous_one::Array{T,3},       
    pre_one,                            

    max_depth::Integer,                 
    knn_k::Int,                         
    icm_round::Int,                        
    n_bad_real::Int;                     
    verbose::Bool = false
) where {T<:AbstractFloat, BType<:Integer}

    d, n = size(X)
    m    = size(B, 1)

    
    h_j_vec = [size(C_one[i], 2) for i in 1:m]
    h_max   = maximum(h_j_vec)
    H       = sum(h_j_vec)

    pool_size = max(2, Threads.nthreads() * 2)
    tls_proto = make_tls(T, BType, d, m, H)
    tls_pool = Channel{typeof(tls_proto)}(pool_size)
    put!(tls_pool, tls_proto)
    for _ in 2:pool_size
        put!(tls_pool, make_tls(T, BType, d, m, H))
    end

    G_one_root = build_one_root_cross_gram(pre_one, pre_root)

    
    first_layer_codes = @view B[1, :]
    unique_codes      = unique(first_layer_codes)
    codes = collect(unique_codes)
    clusters_total = length(codes)

    cluster_indices_real = Dict{Int, Vector{Int}}()
    cluster_indices_virt = Dict{Int, Vector{Int}}()
    for code in unique_codes
        cluster_indices_real[code] = Int[]
        cluster_indices_virt[code] = Int[]
    end
    @inbounds for i in 1:n_bad_real
        push!(cluster_indices_real[first_layer_codes[i]], i)
    end

    n_virt_start =  n_bad_real+1
    @inbounds for i in n_virt_start:n
        push!(cluster_indices_virt[first_layer_codes[i]], i)
    end

    
    R_full = compute_reconstructed(B, len_rate_each, C_root)
    
    base_mse = zeros(T, n_bad_real)
    @threads for j in 1:n_bad_real
        s = zero(T)
        @inbounds for r in 1:d
            diff = X[r, j] - R_full[r, j]
            s += diff * diff
        end
        base_mse[j] = s
    end

    total_updates = Threads.Atomic{Int}(0)
    total_done    = Threads.Atomic{Int}(0)

    println("begin multicenter linkeding...")

    Threads.@sync for code in codes
        Threads.@spawn begin
            tls, tls_from_pool = _pool_take_or_new(tls_pool, () -> make_tls(T, BType, d, m, H))
            
            try
            
            R_self      = tls.R_self
            re          = tls.re
            B_ini       = tls.B_init
            a_ini       = tls.a_init
            a_bak       = tls.a_bak
            B_best      = tls.B_best
            a_best      = tls.a_best
            xC_base_buf = tls.xC_base_buf
            xC_buf      = tls.xC_buf
            rC_buf      = tls.rC_buf
            order       = tls.order_idx
            processed   = tls.processed
            candidates  = tls.candidates
            A           = tls.A

            
            cidxs_real = get(cluster_indices_real, code, Int[])
            if isempty(cidxs_real)
                return
            end
            n_cluster_real = length(cidxs_real)

            
            real_view = view(X, :, cidxs_real)
            tree_real = KDTree(real_view)

            
            resize!(processed, n_cluster_real)
            fill!(processed, false)

            
            resize!(order, n_cluster_real)
            @inbounds for li in 1:n_cluster_real
                order[li] = li
            end
            
            Random.shuffle!(order)

            
            Kp = min(n_cluster_real, max(4 * knn_k, knn_k + 16))
            idxs_list_real, _ = knn(tree_real, real_view, Kp, true) 

            cidxs_virt = get(cluster_indices_virt, code, Int[])
            virtual_knn_k = min(length(cidxs_virt), knn_k)
            virtual_view = view(X, :, cidxs_virt)
            tree_virtual = KDTree(virtual_view)
            idxs_list_virtual, _ = knn(tree_virtual, real_view, virtual_knn_k, true)

            
            @inbounds for pos in eachindex(order)
                li = order[pos]
                gi = cidxs_real[li]

                
                @views xi = X[:, gi]

                
                empty!(candidates)

                
                neigh_virtual = idxs_list_virtual[li]
                len_v = length(neigh_virtual)
                @inbounds for t = 1:len_v
                    v_local = neigh_virtual[t]
                    p_global = cidxs_virt[v_local]
                    push!(candidates, p_global)
                end
                
                
                neigh_rel = idxs_list_real[li]
                len_nei   = length(neigh_rel)

                @inbounds for t = 1:len_nei
                    r_local = neigh_rel[t]
                    r_global = cidxs_real[r_local] 
                    if r_global == gi
                        continue
                    end
                    if !processed[r_local] 
                        continue
                    end

                    push!(candidates, r_global)

                    if length(candidates) >= knn_k + virtual_knn_k
                        break
                    end
                end

                processed[li] = true
                
                if isempty(candidates)
                    continue
                end

                
                if icm_round > 0
                    mul!(xC_base_buf, transpose(pre_one.C_all), xi)
                end

                bestE2      = base_mse[gi]
                best_parent = 0

                
                @inbounds for p_global in candidates
                    new_depth = current_depths[p_global] + 1
                    if new_depth > max_depth
                        continue
                    end

                    
                    @inbounds for r in 1:d
                        re[r] = X[r, gi] - R_full[r, p_global]
                    end

                    
                    quantize_with_kmeans_abs_nonormal_single_extend_allcodes_fast!(
                        re, B_ini, a_ini, m, d, H,
                        pre_one.C_all, pre_one.G, pre_one.offsets,
                        pre_one.flat_layer, pre_one.invnorm_flat,
                        rC_buf
                    )

                    optimized_least_squares_single!(
                        a_ini, re, C_contiguous_one, B_ini, A, m, d, h_max
                    )

                    if icm_round > 0
                        
                        build_xC_one_for_parent!(xC_buf, xC_base_buf, len_rate_each,  B, parent,current_depths,p_global,pre_root, pre_one,pre_one.G,G_one_root)

                        
                        dynamic_icm_encoding_single_for_residual!(re, A, B_ini, a_ini,C_contiguous_one, pre_one,xC_buf, rC_buf,a_bak, icm_round)
                    end

                    
                    reconstruct_node!(R_self, B_ini, a_ini, C_one)

                    
                    newE2 = zero(T)
                    @inbounds for r in 1:d
                        diff = re[r] - R_self[r]
                        newE2 += diff * diff
                    end

                    if newE2 < bestE2
                        bestE2     = newE2
                        best_parent = p_global
                        @inbounds for l in 1:m
                            B_best[l] = B_ini[l]
                            a_best[l] = a_ini[l]
                        end
                    end
                end

                
                 if best_parent != 0
                    parent[gi] = UInt32(best_parent)
                    @inbounds for l in 1:m
                        B[l, gi]             = B_best[l]
                        len_rate_each[l, gi] = a_best[l]
                    end
                    current_depths[gi] = current_depths[best_parent] + 1

                    
                    reconstruct_node!(R_self, view(B, :, gi), view(len_rate_each, :, gi), C_one)
                    pidx = Int(parent[gi])
                    @inbounds for r in 1:d
                        R_full[r, gi] = R_full[r, pidx] + R_self[r]
                    end

                    Threads.atomic_add!(total_updates, 1)
                end
            end 

            Threads.atomic_add!(total_done, 1)
            i_done = total_done[]
            if i_done % 10 == 0
                @printf(stderr, "\rprocessed %d/%d", i_done, clusters_total)
                flush(stderr)
            end
        finally
            if tls_from_pool
                put!(tls_pool, tls)
            end
        end
        end 
    end 

    println("\nTotal updated nodes: $(total_updates[])")
    return R_full
end




function linked_multi_center_two_codebook_ils!(
    X::AbstractMatrix{T},                 
    B::AbstractMatrix{BType},             
    len_rate_each::AbstractMatrix{T},     
    parent::Vector{UInt32},       
    current_depths::Vector{Int},  

    C_root::Vector{Matrix{T}},          
    pre_root,                           

    C_one::Vector{Matrix{T}},           
    C_contiguous_one::Array{T,3},       
    pre_one,                            

    max_depth::Integer,                 
    knn_k::Int,                         
    icm_round::Int,                        
    n_bad_real::Int;                     
    ils_rounds = 4,         
    ils_perturb_layers = 3,         
    verbose::Bool = false
) where {T<:AbstractFloat, BType<:Integer}

    d, n = size(X)
    m    = size(B, 1)

    
    h_j_vec = [size(C_one[i], 2) for i in 1:m]
    h_max   = maximum(h_j_vec)
    H       = sum(h_j_vec)

    pool_size = max(2, Threads.nthreads() * 2)
    tls_proto = make_tls_ils(T, BType, d, m, H)
    tls_pool = Channel{typeof(tls_proto)}(pool_size)
    put!(tls_pool, tls_proto)
    for _ in 2:pool_size
        put!(tls_pool, make_tls_ils(T, BType, d, m, H))
    end

    G_one_root = build_one_root_cross_gram(pre_one, pre_root)

   
    first_layer_codes = @view B[1, :]
    unique_codes      = unique(first_layer_codes)
    codes = collect(unique_codes)
    clusters_total = length(codes)

    cluster_indices_real = Dict{Int, Vector{Int}}()
    cluster_indices_virt = Dict{Int, Vector{Int}}()
    for code in unique_codes
        cluster_indices_real[code] = Int[]
        cluster_indices_virt[code] = Int[]
    end
    @inbounds for i in 1:n_bad_real
        push!(cluster_indices_real[first_layer_codes[i]], i)
    end

    n_virt_start =  n_bad_real+1
    @inbounds for i in n_virt_start:n
        push!(cluster_indices_virt[first_layer_codes[i]], i)
    end

    
    R_full = compute_reconstructed(B, len_rate_each, C_root)
    
    base_mse = zeros(T, n_bad_real)
    @threads for j in 1:n_bad_real
        s = zero(T)
        @inbounds for r in 1:d
            diff = X[r, j] - R_full[r, j]
            s += diff * diff
        end
        base_mse[j] = s
    end

    total_updates = Threads.Atomic{Int}(0)
    total_done    = Threads.Atomic{Int}(0)


    Threads.@sync for code in codes
        Threads.@spawn begin
            tls, tls_from_pool = _pool_take_or_new(tls_pool, () -> make_tls_ils(T, BType, d, m, H))
          
            try
            
            R_self      = tls.R_self
            re          = tls.re
            B_ini       = tls.B_init
            a_ini       = tls.a_init
            a_bak       = tls.a_bak
            B_best      = tls.B_best
            a_best      = tls.a_best
            xC_base_buf = tls.xC_base_buf
            xC_buf      = tls.xC_buf
            rC_buf      = tls.rC_buf
            order       = tls.order_idx
            processed   = tls.processed
            candidates  = tls.candidates
            B_try       = tls.B_try
            a_try       = tls.a_try
            pert_layers = tls.pert_layers
            new_flat    = tls.new_flat
            A           = tls.A

             
            cidxs_real = get(cluster_indices_real, code, Int[])
            if isempty(cidxs_real)
                return
            end
            n_cluster_real = length(cidxs_real)

            
            real_view = view(X, :, cidxs_real)
            tree_real = KDTree(real_view)

            
            resize!(processed, n_cluster_real)
            fill!(processed, false)

            
            resize!(order, n_cluster_real)
            @inbounds for li in 1:n_cluster_real
                order[li] = li
            end
            
            Random.shuffle!(order)

            
            Kp = min(n_cluster_real, max(4 * knn_k, knn_k + 16))
            idxs_list_real, _ = knn(tree_real, real_view, Kp, true) 

            cidxs_virt = get(cluster_indices_virt, code, Int[])
            virtual_knn_k = min(length(cidxs_virt), knn_k)
            virtual_view = view(X, :, cidxs_virt)
            tree_virtual = KDTree(virtual_view)
            idxs_list_virtual, _ = knn(tree_virtual, real_view, virtual_knn_k, true)

            
            @inbounds for pos in eachindex(order)
                li = order[pos]
                gi = cidxs_real[li]

                
                @views xi = X[:, gi]

                
                empty!(candidates)

                
                neigh_virtual = idxs_list_virtual[li]
                len_v = length(neigh_virtual)
                @inbounds for t = 1:len_v
                    v_local = neigh_virtual[t]
                    p_global = cidxs_virt[v_local]
                    push!(candidates, p_global)
                end
                
                
                neigh_rel = idxs_list_real[li]
                len_nei   = length(neigh_rel)

                @inbounds for t = 1:len_nei
                    r_local = neigh_rel[t]
                    r_global = cidxs_real[r_local] 
                    if r_global == gi
                        continue
                    end
                    if !processed[r_local] 
                        continue
                    end

                    push!(candidates, r_global)

                    if length(candidates) >= knn_k + virtual_knn_k
                        break
                    end
                end

                processed[li] = true
                
                if isempty(candidates)
                    continue
                end

                
                if icm_round > 0
                    mul!(xC_base_buf, transpose(pre_one.C_all), xi)
                end

                bestE2      = base_mse[gi]
                best_parent = 0
      
                
                @inbounds for p_global in candidates
                    new_depth = current_depths[p_global] + 1
                    if new_depth > max_depth
                        continue
                    end

                    
                    @inbounds for r in 1:d
                        re[r] = X[r, gi] - R_full[r, p_global]
                    end

                    
                    quantize_with_kmeans_abs_nonormal_single_extend_allcodes_fast!(
                        re, B_ini, a_ini, m, d, H,
                        pre_one.C_all, pre_one.G, pre_one.offsets,
                        pre_one.flat_layer, pre_one.invnorm_flat,
                        rC_buf
                    )

                    optimized_least_squares_single!(
                        a_ini, re, C_contiguous_one, B_ini, A, m, d, h_max
                    )

                  
                    if icm_round > 0
                        
                        
                        build_xC_one_for_parent!(
                            xC_buf,            
                            xC_base_buf,       
                            len_rate_each,     
                            B,                 
                            parent,
                            current_depths,
                            p_global,
                            pre_root, pre_one,
                            pre_one.G,         
                            G_one_root         
                        )

                        
                        dynamic_icm_encoding_single_for_residual!(
                            re, A, B_ini, a_ini,
                            C_contiguous_one,
                            pre_one,
                            xC_buf, rC_buf,
                            a_bak,
                            icm_round
                        )
                    end

                    
                    reconstruct_node!(R_self, B_ini, a_ini, C_one)
                    E2_best = zero(T)
                    @inbounds @simd for t in 1:d
                        diff = re[t] - R_self[t]
                        E2_best += diff * diff
                    end

                    
                    @inbounds for il in 1:ils_rounds
                        
                        @inbounds for t in 1:ils_perturb_layers
                            
                            ℓ = rand(1:m)
                            

                            
                            hℓ = h_j_vec[ℓ]

                            
                            old_code  = Int(B_ini[ℓ])

                            
                            
                            tcode     = rand(0:(hℓ-2))      
                            new_code  = tcode + 1           
                            if new_code >= old_code
                                new_code += 1               
                            end

                            
                            startf = pre_one.offsets[ℓ]
                            newf   = startf + new_code - 1   

                            pert_layers[t] = ℓ
                            new_flat[t]    = newf
                        end

                        
                        copyto!(B_try, B_ini)
                        @inbounds for t in 1:ils_perturb_layers
                            ℓ = pert_layers[t]
                            
                            B_try[ℓ] = BType(new_flat[t] - pre_one.offsets[ℓ] + 1)
                        end

                        
                        optimized_least_squares_single!(a_try, re, C_contiguous_one, B_try, A, m, d, h_max)

                        
                        
                        if icm_round > 0
                            dynamic_icm_encoding_single_for_residual!(
                                re, A, B_try, a_try,
                                C_contiguous_one,
                                pre_one,
                                xC_buf, rC_buf,
                                a_bak,
                                icm_round
                            )
                        end

                        
                        reconstruct_node!(R_self, B_try, a_try, C_one)
                        E2_try = zero(T)
                        @inbounds @simd for t in 1:d
                            diff = re[t] - R_self[t]
                            E2_try += diff * diff
                        end

                        if E2_try < E2_best
                            E2_best = E2_try
                            copyto!(B_ini, B_try)
                            copyto!(a_ini, a_try)
                        end
                    end
                    

                    if E2_best < bestE2         
                        bestE2     = E2_best
                        best_parent = p_global
                        @inbounds for l in 1:m
                            B_best[l] = B_ini[l]
                            a_best[l] = a_ini[l]
                        end
                    end
                end

                
                if best_parent != 0
                    parent[gi] = UInt32(best_parent)
                    @inbounds for l in 1:m
                        B[l, gi]            = B_best[l]
                        len_rate_each[l, gi] = a_best[l]
                    end
                    current_depths[gi] = current_depths[best_parent] + 1

                    
                    reconstruct_node!(R_self, view(B, :, gi), view(len_rate_each, :, gi), C_one)
                    pidx = Int(parent[gi])
                    @inbounds for r in 1:d
                        R_full[r, gi] = R_full[r, pidx] + R_self[r]
                    end

                    Threads.atomic_add!(total_updates, 1)
                end
            end 

            Threads.atomic_add!(total_done, 1)
            i_done = total_done[]
            if i_done % 10 == 0
                @printf(stderr, "\rprocessed %d/%d", i_done, clusters_total)
                flush(stderr)
            end
        finally
            if tls_from_pool
                put!(tls_pool, tls)
            end
        end
        end 
    end 

    println("\nTotal updated nodes: $(total_updates[])")
    return R_full
end













function linked_with_dual_strategy!(
    X::Matrix{T},                 
    R_in::AbstractMatrix{T},
    R_full::AbstractMatrix{T},
    B::Matrix{BType},             
    len_rate_each::Matrix{T},     
    current_depths::Vector{Int},
    C_root::Vector{Matrix{T}},
    C_contiguous_root::Array{T,3},
    pre_root,
    C_one::Vector{Matrix{T}},
    C_contiguous_one::Array{T,3},
    pre_one,
    is_bad_cluster::AbstractVector{Bool},  
    max_depth::Integer,
    root_percentile::Float64,
    num_layers::Int,
    knn_k::Int,
    depth_k::Int,
    icm_round::Int,
    layer_relinked_rounds::Int,
    ils_iters::Int,
    icm_iters::Int,
    perturb_k::Int,
    virtual_ratio::Float64,
    min_virtual_per_cluster::Int,
    max_virtual_per_cluster::Int,
    verbose::Bool = false,
) where {T<:AbstractFloat, BType<:Integer}

    d, n = size(X)
    m    = size(B, 1)

    
    fill!(current_depths, 0)

    
    first_layer_codes = @view B[1, :]

    good_idx = Int[]
    bad_idx  = Int[]

    @inbounds for i in 1:n
        cid = Int(first_layer_codes[i])
   
        if is_bad_cluster[cid]
            push!(bad_idx, i)
        else
            push!(good_idx, i)
        end
    end

    if verbose
        println("dual-train: good points = $(length(good_idx)), bad points = $(length(bad_idx))")
    end

    
    if !isempty(good_idx)
        n_good = length(good_idx)

        
        X_good   = @view X[:, good_idx]
        B_good   = @view B[:, good_idx]
        len_good = @view len_rate_each[:, good_idx]

        parent_good = zeros(UInt32, n_good)
        depth_good  = fill(0, n_good)

        
        R_good = linked_from_inner_to_outer_two_codebook!(
            X_good, B_good, len_good,
            parent_good, depth_good,
            C_root, pre_root,
            C_one,  C_contiguous_one,  pre_one,
            max_depth, root_percentile, num_layers,
            knn_k, depth_k, icm_round;
            verbose = false,
            layer_relinked_rounds = layer_relinked_rounds,
        )

        
        @inbounds for (j, gi) in enumerate(good_idx)
            current_depths[gi] = depth_good[j]

            if parent_good[j] == 0
                
                @views R_in[:, gi] .= X[:, gi]
            else
                p_loc = Int(parent_good[j])          
                @views v_X = X[:, gi]
                @views v_Rp = R_good[:, p_loc]
                @views R_in[:, gi] .= v_X .- v_Rp
            end

            
            @views R_full[:, gi] .= R_good[:, j]
        end
    end

    
    if !isempty(bad_idx)
        n_bad_real = length(bad_idx)

        
        X_bad   = Matrix{T}(undef, d, n_bad_real)
        B_bad   = Matrix{BType}(undef, m, n_bad_real)
        len_bad = Matrix{T}(undef, m, n_bad_real)

        @inbounds for (j, gi) in enumerate(bad_idx)
            @views X_bad[:, j]   .= X[:, gi]
            @views B_bad[:, j]   .= B[:, gi]
            @views len_bad[:, j] .= len_rate_each[:, gi]
        end

println("umap like add virtual centers........")
X_bad_aug, B_bad_aug, len_bad_aug =
    add_virtual_roots_umap_reencode(
        X_bad, B_bad, len_bad,
        C_root, C_contiguous_root, pre_root,
        ils_iters, icm_iters, perturb_k,
        virtual_ratio,
        min_virtual_per_cluster,
        max_virtual_per_cluster;
        
        use_fixed_virtual_per_cluster = false,
        fixed_virtual_per_cluster     = 256,
        is_bad_cluster = is_bad_cluster, 
        verbose = true,
    )

        n_bad_aug = size(X_bad_aug, 2)
        parent_bad = zeros(UInt32, n_bad_aug)
        depth_bad  = fill(0, n_bad_aug)

        
        R_bad_aug = linked_multi_center_two_codebook!(
            X_bad_aug, B_bad_aug, len_bad_aug,
            parent_bad, depth_bad,
            C_root, pre_root,
            C_one,  C_contiguous_one,  pre_one,
            max_depth, knn_k, icm_round, n_bad_real;
            verbose = false,
        )

        
        @inbounds for j in 1:n_bad_real
            gi = bad_idx[j]

            current_depths[gi] = depth_bad[j]

            if parent_bad[j] == 0
                
                @views R_in[:, gi] .= X[:, gi]
            else
                p_loc = Int(parent_bad[j])     
                @views v_X  = X_bad_aug[:, j]      
                @views v_Rp = R_bad_aug[:, p_loc]  
                @views R_in[:, gi] .= v_X .- v_Rp
            end

            
            @views R_full[:, gi] .= R_bad_aug[:, j]
        end

        
        @inbounds for j in 1:n_bad_real
            gi = bad_idx[j]
            @views B[:, gi]            .= B_bad_aug[:, j]
            @views len_rate_each[:, gi].= len_bad_aug[:, j]
        end
        
    end

    errors_linked = zeros(T, n)
    @threads for i in 1:n
        @inbounds @simd for j in 1:d
            diff_linked =  X[j, i] - R_full[j, i]
            errors_linked[i] += diff_linked * diff_linked
        end
    end
    train_mse=mean(errors_linked)
    println("linkeded Reconstruction Analysis Summary:")
    println("  linked Error: Max=$(round(maximum(errors_linked), digits=4)), Min=$(round(minimum(errors_linked), digits=4)), Mean=$(round(train_mse ,digits=4))")
    linked_ratio = (length(findall(parent_good.>0))+length(findall(parent_bad.>0)))/(n_good+n_bad_aug)
    println("  linked Ratio: $(round(100*linked_ratio, digits=2))%")
    return train_mse
end




function linked_with_dual_strategy_for_baseset!(
    X::Matrix{T},                 
    B::Matrix{BType},             
    len_rate_each::Matrix{T},     
    parent::Vector{UInt32},       
    current_depths::Vector{Int},  
    C_root::Vector{Matrix{T}},
    C_contiguous_root::Array{T,3},
    pre_root,
    C_one::Vector{Matrix{T}},
    C_contiguous_one::Array{T,3},
    pre_one,
    max_depth::Integer,
    root_percentile::Float64,
    num_layers::Int,
    knn_k::Int,
    depth_k::Int,
    icm_round::Int,
    is_bad_cluster::AbstractVector{Bool},  
    n_real::Int; 
    verbose::Bool = false,
    layer_relinked_rounds::Int = 0,
) where {T<:AbstractFloat, BType<:Integer}

    d, n = size(X)
    m    = size(B, 1)

    
    

    
    first_layer_codes = @view B[1, :]

    good_real_idx = Int[]
    bad_real_idx  = Int[]

    @inbounds for i in 1:n_real
        cid = Int(first_layer_codes[i])
        if is_bad_cluster[cid]
            push!(bad_real_idx, i)
        else
            push!(good_real_idx, i)
        end
    end

    
    virt_idx = Int[]
    @inbounds for i in (n_real+1):n
        push!(virt_idx, i)
    end

    if verbose
        @printf("dual-base: real=%d, virtual=%d, good_real=%d, bad_real=%d\n",
                n_real, length(virt_idx), length(good_real_idx), length(bad_real_idx))
    end

    
    if !isempty(good_real_idx)
        X_good   = @view X[:, good_real_idx]
        B_good   = @view B[:, good_real_idx]
        len_good = @view len_rate_each[:, good_real_idx]

        parent_good = zeros(UInt32, length(good_real_idx))
        depth_good  = fill(0, length(good_real_idx))

        R_good = linked_from_inner_to_outer_two_codebook_ils!(
            X_good, B_good, len_good,
            parent_good, depth_good,
            C_root, pre_root,
            C_one,  C_contiguous_one,  pre_one,
            max_depth, root_percentile, num_layers,
            knn_k, depth_k, icm_round;
            verbose = false,
            layer_relinked_rounds = layer_relinked_rounds,
        )

        
        @inbounds for (j, gi) in enumerate(good_real_idx)
            current_depths[gi] = depth_good[j]
            if parent_good[j] == 0
                parent[gi] = 0
            else
                
                parent[gi] = UInt32(good_real_idx[parent_good[j]])
            end
            
        end
    end

    
    if !isempty(bad_real_idx) || !isempty(virt_idx)
        bad_all_idx = vcat(bad_real_idx, virt_idx)  

        X_bad   = @view X[:, bad_all_idx]
        B_bad   = @view B[:, bad_all_idx]
        len_bad = @view len_rate_each[:, bad_all_idx]

        parent_bad = zeros(UInt32, length(bad_all_idx))
        depth_bad  = fill(0, length(bad_all_idx))

        R_bad = linked_multi_center_two_codebook_ils!(
            X_bad, B_bad, len_bad,
            parent_bad, depth_bad,
            C_root, pre_root,
            C_one,  C_contiguous_one,  pre_one,
            max_depth, knn_k, icm_round, length(bad_real_idx);
            verbose = false,
        )

        
        @inbounds for (j, gi) in enumerate(bad_all_idx)
            current_depths[gi] = depth_bad[j]
            if parent_bad[j] == 0
                parent[gi] = 0
            else
                parent[gi] = UInt32(bad_all_idx[parent_bad[j]])
            end
            
        end
    end

    
    return nothing
end
