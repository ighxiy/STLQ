


















include("./greedy_quantizer.jl")
include("linked_encode_helper.jl")
include("linked_quantizer.jl")
include("./encode_abs.jl")

using Clustering
using Random




@inline function _resolve_virtual_k(
    n_local::Int,
    virtual_ratio::Float64,
    use_fixed::Bool,
    fixed_k::Int,
    min_k::Int,
    max_k::Int,
)
    k = use_fixed ? fixed_k : round(Int, virtual_ratio * n_local)
    k = clamp(k, min_k, max_k)
    return min(k, n_local)
end

@inline function _w_sym(pij::T, pji::T) where {T<:AbstractFloat}
    
    return pij + pji - pij * pji
end

@inline function _p_ij(d::T, rho::T, sigma::T) where {T<:AbstractFloat}
    v = d - rho
    return (v <= 0) ? one(T) : exp(-v / sigma)
end





function _umap_rho_sigma!(
    rho::Vector{T},
    sigma::Vector{T},
    idxs_list::Vector{Vector{Int}},
    dists_list::Vector{Vector{T}},
    local_connectivity::Int,
    target::T,
) where {T<:AbstractFloat}

    n = length(idxs_list)
    @assert length(dists_list) == n
    @assert length(rho) == n && length(sigma) == n

    eps = T(1e-6)

    @inbounds for i in 1:n
        idxs  = idxs_list[i]
        dists = dists_list[i]
        k = length(idxs)

        start = (k > 0 && idxs[1] == i) ? 2 : 1
        k_eff = k - (start - 1)

        if k_eff <= 0
            rho[i]   = zero(T)
            sigma[i] = one(T)
            continue
        end

        lc = min(local_connectivity, k_eff)
        rho_i = max(dists[start + lc - 1], zero(T))
        rho[i] = rho_i

        d_max = max(dists[start + k_eff - 1], eps)

        lo = eps
        hi = d_max

        
        ones_cnt = 0
        @inbounds for t in start:(start + k_eff - 1)
            if dists[t] <= rho_i + eps
                ones_cnt += 1
            end
        end

        if target <= T(ones_cnt) + eps
            sigma[i] = eps
            continue
        end

        
        for _ in 1:64
            mid = (lo + hi) / 2
            s = zero(T)

            @inbounds for t in start:(start + k_eff - 1)
                v = dists[t] - rho_i
                s += (v <= 0) ? one(T) : exp(-v / mid)
            end

            if s > target
                hi = mid
            else
                lo = mid
            end
        end

        sigma[i] = hi
    end

    return nothing
end





function _build_umap_fuzzy_graph!(
    nbrs::Vector{Vector{Int}},
    pdir::Vector{Vector{T}},
    maps::Vector{Dict{Int,Int}},
    idxs_list::Vector{Vector{Int}},
    dists_list::Vector{Vector{T}},
    rho::Vector{T},
    sigma::Vector{T},
) where {T<:AbstractFloat}

    n = length(idxs_list)
    @assert length(dists_list) == n

    resize!(nbrs, n); resize!(pdir, n); resize!(maps, n)

    eps = T(1e-6)

    @inbounds for i in 1:n
        idxs  = idxs_list[i]
        dists = dists_list[i]

        ni = Int[]
        pi = T[]
        sizehint!(ni, max(length(idxs) - 1, 0))
        sizehint!(pi, max(length(idxs) - 1, 0))

        for t in eachindex(idxs)
            j = idxs[t]
            j == i && continue
            dij = dists[t]
            push!(ni, j)
            push!(pi, _p_ij(dij, rho[i], max(sigma[i], eps)))
        end

        nbrs[i] = ni
        pdir[i] = pi

        mp = Dict{Int,Int}()
        sizehint!(mp, length(ni))
        for t in eachindex(ni)
            mp[ni[t]] = t
        end
        maps[i] = mp
    end

    return nothing
end

@inline function _get_p(maps::Vector{Dict{Int,Int}}, pdir::Vector{Vector{T}}, i::Int, j::Int) where {T<:AbstractFloat}
    pos = get(maps[i], j, 0)
    return pos == 0 ? zero(T) : pdir[i][pos]
end

@inline function _max_w_to_seeds(
    i::Int,
    seeds::Vector{Int},
    maps::Vector{Dict{Int,Int}},
    pdir::Vector{Vector{T}},
) where {T<:AbstractFloat}
    best = zero(T)
    for s in seeds
        pij = _get_p(maps, pdir, i, s)
        pji = _get_p(maps, pdir, s, i)
        w   = _w_sym(pij, pji)
        if w > best
            best = w
        end
    end
    return best
end







function _select_virtual_seeds_umap_simple(
    nbrs::Vector{Vector{Int}},
    maps::Vector{Dict{Int,Int}},
    pdir::Vector{Vector{T}},
    deg::Vector{T},
    k_virtual::Int;
    overlap_thr::Float64 = 0.85,
    prefer_peaks::Bool = true,
) where {T<:AbstractFloat}

    n = length(deg)
    k_virtual = min(k_virtual, n)
    k_virtual <= 0 && return Int[]

    cand = Int[]

    if prefer_peaks
        peaks = Int[]
        sizehint!(peaks, n)
        for i in 1:n
            di = deg[i]
            peak = true
            for j in nbrs[i]
                if deg[j] > di
                    peak = false
                    break
                end
            end
            peak && push!(peaks, i)
        end
        sort!(peaks, by = i -> deg[i], rev = true)
        cand = peaks
    end

    cand2 = collect(1:n)
    sort!(cand2, by = i -> deg[i], rev = true)
    append!(cand, cand2)

    thr = T(overlap_thr)
    seeds = Int[]
    sizehint!(seeds, k_virtual)

    for i in cand
        length(seeds) >= k_virtual && break
        (i in seeds) && continue
        if isempty(seeds)
            push!(seeds, i)
            continue
        end
        maxw = _max_w_to_seeds(i, seeds, maps, pdir)
        maxw <= thr && push!(seeds, i)
    end

    
    if length(seeds) < k_virtual
        for i in cand2
            length(seeds) >= k_virtual && break
            (i in seeds) && continue
            push!(seeds, i)
        end
    end

    return seeds
end





function _fill_anchor_barycenter_umap!(
    out::AbstractVector{T},
    X_local::AbstractMatrix{T},
    seed::Int,
    nbrs::Vector{Vector{Int}},
    maps::Vector{Dict{Int,Int}},
    pdir::Vector{Vector{T}};
    anchor_neighbor_k::Int = 16,
) where {T<:AbstractFloat}

    d = size(X_local, 1)
    fill!(out, zero(T))

    
    sumw = one(T)
    @inbounds for r in 1:d
        out[r] = X_local[r, seed]
    end

    ni = nbrs[seed]
    isempty(ni) && return nothing

    
    wj = Vector{Tuple{T,Int}}(undef, length(ni))
    for t in eachindex(ni)
        j = ni[t]
        pij = _get_p(maps, pdir, seed, j)
        pji = _get_p(maps, pdir, j, seed)
        wj[t] = (_w_sym(pij, pji), j)
    end
    sort!(wj, by = x -> x[1], rev = true)

    Kuse = min(anchor_neighbor_k, length(wj))
    for t in 1:Kuse
        w, j = wj[t]
        w <= 0 && continue
        sumw += w
        @inbounds for r in 1:d
            out[r] += w * X_local[r, j]
        end
    end

    invsum = inv(sumw)
    @inbounds for r in 1:d
        out[r] *= invsum
    end

    return nothing
end






















function add_virtual_roots_umap_reencode(
    X::AbstractMatrix{T},
    B::AbstractMatrix{BType},
    len_rate_each::AbstractMatrix{T},
    C_root::Vector{Matrix{T}},
    C_contiguous_root::Array{T,3},
    pre_root,
    ils_iters::Int,
    icm_iters::Int,
    perturb_k::Int,
    virtual_ratio::Float64 = 0.10,
    min_virtual_per_cluster::Int = 1,
    max_virtual_per_cluster::Int = 2000;
    is_bad_cluster::AbstractVector{Bool},
    verbose::Bool = true,

    
    use_fixed_virtual_per_cluster::Bool = false,
    fixed_virtual_per_cluster::Int = 512,

    
    umap_knn_k::Int = 50,
    local_connectivity::Int = 1,
    overlap_thr::Float64 = 0.9,
    prefer_peaks::Bool = true,
    anchor_neighbor_k::Int = 16,
) where {T<:AbstractFloat, BType<:Integer}

    d, n = size(X)
    m    = size(B, 1)
    @assert size(B, 2) == n
    @assert size(len_rate_each, 1) == m && size(len_rate_each, 2) == n

    
    first_layer_codes = B[1, :]
    unique_codes      = unique(first_layer_codes)

    cluster_indices = Dict{Int, Vector{Int}}()
    for code_any in unique_codes
        cluster_indices[Int(code_any)] = Int[]
    end
    @inbounds for i in 1:n
        code = Int(first_layer_codes[i])
        push!(cluster_indices[code], i)
    end

    verbose && println("UMAP virtual roots: clusters=$(length(unique_codes)), base=$n")

    
    k_per_cluster = Dict{Int, Int}()
    total_virtual = 0

    for code_any in unique_codes
        code = Int(code_any)
        if !is_bad_cluster[code]
            continue
        end
        cidxs = cluster_indices[code]
        n_local = length(cidxs)
        n_local < 2 && continue

        k = _resolve_virtual_k(
            n_local,
            virtual_ratio,
            use_fixed_virtual_per_cluster,
            fixed_virtual_per_cluster,
            min_virtual_per_cluster,
            max_virtual_per_cluster
        )
        if k > 0
            k_per_cluster[code] = k
            total_virtual += k
        end
    end

    if total_virtual == 0
        verbose && println("UMAP virtual roots: none selected")
        return X, B, len_rate_each
    end
    verbose && println("UMAP virtual roots: total_virtual centers to encode = $total_virtual")

    
    Centers = Matrix{T}(undef, d, total_virtual)
    forced_codes = Vector{Int}(undef, total_virtual)
    col_ptr = 1

    tmp_center = zeros(T, d)

    for code_any in unique_codes
        code = Int(code_any)
        if !is_bad_cluster[code]
            continue
        end
        k = get(k_per_cluster, code, 0)
        k == 0 && continue

        cidxs = cluster_indices[code]
        n_local = length(cidxs)
        n_local < 2 && continue

        X_local = view(X, :, cidxs)

        
        if n_local <= 3
            kk = min(k, n_local)
            for t in 1:kk
                Centers[:, col_ptr] .= X_local[:, t]
                forced_codes[col_ptr] = code
                col_ptr += 1
            end
            continue
        end

        
        k_graph = min(umap_knn_k, n_local - 1)
        Kq = k_graph + 1

        tree = NearestNeighbors.KDTree(X_local)
        idxs_list, dists_list = NearestNeighbors.knn(tree, X_local, Kq, true)

        
        rho   = zeros(T, n_local)
        sigma = ones(T,  n_local)
        target = T(log2(Float64(max(k_graph, 2))))

        _umap_rho_sigma!(
            rho, sigma,
            idxs_list, dists_list,
            local_connectivity, target
        )

        
        nbrs = Vector{Vector{Int}}()
        pdir = Vector{Vector{T}}()
        maps = Vector{Dict{Int,Int}}()

        _build_umap_fuzzy_graph!(
            nbrs, pdir, maps,
            idxs_list, dists_list,
            rho, sigma
        )

        
        deg = zeros(T, n_local)
        for i in 1:n_local
            s = zero(T)
            for j in nbrs[i]
                pij = _get_p(maps, pdir, i, j)
                pji = _get_p(maps, pdir, j, i)
                s  += _w_sym(pij, pji)
            end
            deg[i] = s
        end

        
        seeds = _select_virtual_seeds_umap_simple(
            nbrs, maps, pdir, deg, k;
            overlap_thr = overlap_thr,
            prefer_peaks = prefer_peaks
        )

        
        for s in seeds
            _fill_anchor_barycenter_umap!(
                tmp_center,
                X_local,
                s,
                nbrs, maps, pdir;
                anchor_neighbor_k = anchor_neighbor_k
            )
            Centers[:, col_ptr] .= tmp_center
            forced_codes[col_ptr] = code
            col_ptr += 1
        end
    end

    total_virtual_effective = col_ptr - 1
    if total_virtual_effective == 0
        verbose && println("UMAP virtual roots: no effective centers")
        return X, B, len_rate_each
    elseif total_virtual_effective < total_virtual
        Centers = Centers[:, 1:total_virtual_effective]
        forced_codes = forced_codes[1:total_virtual_effective]
        total_virtual = total_virtual_effective
        verbose && println("UMAP virtual roots: effective centers reduced to $total_virtual")
    end

    
    verbose && println("UMAP virtual roots: encoding centers with forced root layer ...")
    B_virtual, len_virtual =
        quantize_with_kmeans_abs_nonormal_extend_allcodes_fast_fix!(Centers, C_root, pre_root, forced_codes)

    H_root = pre_root.H
    rC = zeros(Float32, H_root, total_virtual)
    xC = zeros(Float32, H_root, total_virtual)

    dynamic_icm_with_ils_abs_nonormal!(
        Centers, B_virtual, len_virtual,
        rC, xC,
        C_contiguous_root, pre_root,
        ils_iters, icm_iters, perturb_k, true
    )

    
    X_virtual = compute_reconstructed(B_virtual, len_virtual, C_root)

    X_aug        = hcat(X, X_virtual)
    B_aug        = hcat(B, B_virtual)
    len_rate_aug = hcat(len_rate_each, len_virtual)

    verbose && println("UMAP virtual roots: augmented $n -> $(size(X_aug,2)) (virtual=$total_virtual)")
    return X_aug, B_aug, len_rate_aug
end
