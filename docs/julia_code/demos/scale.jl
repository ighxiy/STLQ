using LinearAlgebra
function quantize_coeffs_multibook_split_weighted1!(    B::AbstractMatrix{BType},
    a::Matrix{T},
    C_root::Vector{Matrix{T}},
    C_one::Vector{Matrix{T}},
    parent::AbstractVector{UInt32};
    bits_first::Int = 6,
    bits_rest::Int  = 6,
    bits_per_layer::Union{Nothing,AbstractVector{Int}} = nothing,
    p_first_candidates::Vector{Float64} = [99.5, 99.8, 100.0],
    p_rest_candidates::Vector{Float64}  = [99.5, 99.8, 100.0],
    auto_search_p::Bool = false,
    p_first_range::Tuple{Float64,Float64,Float64} = (99.0, 100.0, 0.1), 
    p_rest_range::Tuple{Float64,Float64,Float64}  = (99.0, 100.0, 0.1),
    use_weighted_quantile::Bool = false,
    allow_clip::Bool = true,
    verbose::Bool = true
) where {T<:AbstractFloat, BType<:Integer}

    m, n = size(a)
    @assert size(B) == (m, n)
    @assert length(C_root) == m
    @assert length(C_one)  == m
    @assert length(parent) == n

    d = size(C_root[1], 1)
    @assert all(size(C_root[ℓ],1) == d for ℓ in 1:m)
    @assert all(size(C_one[ℓ],1)  == d for ℓ in 1:m)

    
    bitsL = bits_per_layer === nothing ?
        [bits_first; fill(bits_rest, m-1)] :
        ( @assert length(bits_per_layer)==m; collect(bits_per_layer) )

    
    root_idx = Int[]
    one_idx  = Int[]
    for i in 1:n
        if parent[i] == 0
            push!(root_idx, i)
        else
            push!(one_idx, i)
        end
    end

    
    q_int_f32        = Matrix{Float32}(undef, m, n)
    C_root_scaled    = [copy(C_root[ℓ]) for ℓ in 1:m]
    C_one_scaled     = [copy(C_one[ℓ])  for ℓ in 1:m]

    Δ_root        = ones(T, m)
    Δ_one         = ones(T, m)
    s_root        = ones(T, m)
    s_one         = ones(T, m)
    g_root        = ones(T, m)
    g_one         = ones(T, m)
    pct_clip_root = zeros(Float64, m)
    pct_clip_one  = zeros(Float64, m)
    max_abs_round_root = zeros(T, m)
    max_abs_round_one  = zeros(T, m)
    p_root        = fill(100.0, m)
    p_one         = fill(100.0, m)

    
    function make_p_candidates(range::Tuple{Float64,Float64,Float64}, fallback::Vector{Float64})
        if !auto_search_p
            return fallback
        end
        pmin, pmax, pstep = range
        @assert pstep > 0
        ps = collect(pmin:pstep:pmax)
        
        if ps[end] < 100.0 - 1e-12
            push!(ps, 100.0)
        end
        return ps
    end
    p_first = make_p_candidates(p_first_range, p_first_candidates)
    p_rest  = make_p_candidates(p_rest_range,  p_rest_candidates)

    
    function precompute_norm2(C::Vector{Matrix{T}})
        out = Vector{Vector{T}}(undef, length(C))
        for ℓ in 1:length(C)
            K = size(C[ℓ], 2)
            norms = Vector{T}(undef, K)
            Cℓ = C[ℓ]
            @inbounds for k in 1:K
                acc = zero(T)
                v = @view Cℓ[:, k]
                for t in 1:d
                    x = v[t]
                    acc += x*x
                end
                norms[k] = acc
            end
            out[ℓ] = norms
        end
        return out
    end
    norm2_root = precompute_norm2(C_root)
    norm2_one  = precompute_norm2(C_one)

    
    @inline function quantile_type7_sorted(x_sorted::Vector{T}, p::Float64) where {T}
        n = length(x_sorted)
        @assert 0.0 <= p <= 1.0
        if n == 0
            return zero(T)
        elseif p <= 0.0
            return x_sorted[1]
        elseif p >= 1.0
            return x_sorted[end]
        end
        h = (n - 1) * p + 1
        j = floor(Int, h)
        g = h - j
        if j >= n
            return x_sorted[end]
        end
        
        return (one(T) - T(g)) * x_sorted[j] + T(g) * x_sorted[j+1]
    end

    @inline function weighted_quantile_sorted_step(x_sorted::Vector{T}, cumw::Vector{T}, p::Float64) where {T}
        @assert 0.0 <= p <= 1.0
        totalw = cumw[end]
        if totalw <= zero(T)
            
            return x_sorted[end]
        end
        target = T(p) * totalw
        idx = searchsortedfirst(cumw, target)
        idx = clamp(idx, 1, length(x_sorted))
        return x_sorted[idx]
    end

    verbose && println("==== multibook split + vector-weighted LS (6bit) ====")
    verbose && println("layers = $m, n = $n, dim = $d")
    verbose && println("bits_per_layer = $(bitsL)")
    verbose && println("root_count = $(length(root_idx)), one_count = $(length(one_idx))")
    verbose && println("auto_search_p = $(auto_search_p), use_weighted_quantile = $(use_weighted_quantile)")
    verbose && println("p_first_candidates = $(p_first)")
    verbose && println("p_rest_candidates  = $(p_rest)")
    verbose && println()

    
    function quantize_group_signed_weighted!(
        layer::Int,
        bits::Int,
        idxs::Vector{Int},
        norm2_group::Vector{Vector{T}},  
        p_candidates::Vector{Float64},
        Δ_vec::Vector{T},
        s_vec::Vector{T},
        g_vec::Vector{T},
        pct_clip_vec::Vector{Float64},
        max_abs_round_vec::Vector{T},
        p_vec::Vector{Float64}
    )
        if isempty(idxs)
            Δ_vec[layer] = one(T)
            s_vec[layer] = one(T)
            g_vec[layer] = one(T)
            pct_clip_vec[layer] = 0.0
            max_abs_round_vec[layer] = zero(T)
            p_vec[layer] = 100.0
            return
        end

        L = length(idxs)
        a_sub   = Vector{T}(undef, L)
        abs_sub = Vector{T}(undef, L)
        w_sub   = Vector{T}(undef, L)

        max_abs_round = zero(T)
        normsℓ = norm2_group[layer]

        @inbounds for k in 1:L
            i = idxs[k]
            aval = a[layer, i]
            a_sub[k] = aval
            ab = abs(aval)
            abs_sub[k] = ab

            
            r = abs(T(round(Int, aval)))
            if r > max_abs_round
                max_abs_round = r
            end

            code_idx = Int(B[layer, i])
            
            w_sub[k] = (code_idx == 0) ? zero(T) : normsℓ[code_idx]
        end
        max_abs_round_vec[layer] = max_abs_round

        local_ps = allow_clip ? p_candidates : Float64[100.0]
        Q = (1 << (bits-1)) - 1

        
        abs_sorted = copy(abs_sub)
        perm = nothing
        cumw = nothing

        if use_weighted_quantile
            perm = sortperm(abs_sorted)               
            abs_sorted = abs_sorted[perm]
            w_sorted = w_sub[perm]
            cumw = similar(w_sorted)
            run = zero(T)
            @inbounds for t in 1:L
                wt = w_sorted[t]
                run += (wt > 0 ? wt : zero(T))
                cumw[t] = run
            end
        else
            sort!(abs_sorted)
        end
        maxabs = abs_sorted[end]

        @inline function get_Amax(p::Float64)
            if p >= 100.0
                return maxabs
            end
            pp = p / 100.0
            if use_weighted_quantile
                return weighted_quantile_sorted_step(abs_sorted, cumw, pp)
            else
                return quantile_type7_sorted(abs_sorted, pp)
            end
        end

        best_cost = Inf
        best_Δ    = one(T)
        best_s    = one(T)
        best_clip = 0.0
        best_p    = 100.0
        best_q    = Vector{Int16}(undef, L)
        q_tmp     = Vector{Int16}(undef, L)

        for p in local_ps
            Amax = get_Amax(p)

            if !allow_clip && Amax < max_abs_round
                continue
            end
            if Amax <= zero(T)
                continue
            end

            Δ = Amax / T(max(Q, 1))

            clip_cnt = 0
            @inbounds for k in 1:L
                v = a_sub[k] / Δ
                q = round(Int, v)
                if q >  Q
                    q =  Q; clip_cnt += 1
                elseif q < -Q
                    q = -Q; clip_cnt += 1
                end
                q_tmp[k] = Int16(q)
            end

            
            num = zero(T)
            den = zero(T)
            @inbounds for k in 1:L
                wk = w_sub[k]
                qk = T(q_tmp[k])
                num += wk * a_sub[k] * qk
                den += wk * qk * qk
            end
            s_c = den > 0 ? (num / den) : one(T)

            
            err = zero(T)
            @inbounds for k in 1:L
                qk = T(q_tmp[k])
                e  = a_sub[k] - s_c * qk
                err += w_sub[k] * e * e
            end
            cost = Float64(err)
            clip_ratio = Float64(clip_cnt) / L

            if cost < best_cost
                best_cost = cost
                best_Δ    = Δ
                best_s    = s_c
                best_clip = clip_ratio
                best_p    = p
                copy!(best_q, q_tmp)
            end
        end

        Δ_vec[layer] = best_Δ
        s_vec[layer] = best_s
        g_vec[layer] = best_s / best_Δ
        pct_clip_vec[layer] = best_clip
        p_vec[layer] = best_p

        @inbounds for k in 1:L
            i = idxs[k]
            q_int_f32[layer, i] = Float32(best_q[k])
        end
    end


    
    bits1 = bitsL[1]
    @assert bits1 ≥ 2
    quantize_group_signed_weighted!(
        1, bits1, root_idx, norm2_root, p_first,
        Δ_root, s_root, g_root,
        pct_clip_root, max_abs_round_root, p_root
    )
    quantize_group_signed_weighted!(
        1, bits1, one_idx,  norm2_one,  p_first,
        Δ_one,  s_one,  g_one,
        pct_clip_one,  max_abs_round_one, p_one
    )
    verbose && println("L1 signed (weighted): bits=$bits1")

    verbose && println("  root: Δ=$(Δ_root[1]), s=$(s_root[1]), g=$(g_root[1]), pct_clip=$(pct_clip_root[1]), p=$(p_root[1])")
    verbose && println("  one : Δ=$(Δ_one[1]),  s=$(s_one[1]),  g=$(g_one[1]),  pct_clip=$(pct_clip_one[1]), p=$(p_one[1])")

    
    for ℓ in 2:m
        bits = bitsL[ℓ]
        @assert bits ≥ 2

        quantize_group_signed_weighted!(
            ℓ, bits, root_idx, norm2_root, p_rest,
            Δ_root, s_root, g_root,
            pct_clip_root, max_abs_round_root, p_root
        )
        quantize_group_signed_weighted!(
            ℓ, bits, one_idx,  norm2_one,  p_rest,
            Δ_one,  s_one,  g_one,
            pct_clip_one,  max_abs_round_one, p_one
        )

        verbose && println("L$ℓ signed (weighted): bits=$bits")
        verbose && println("  root: Δ=$(Δ_root[ℓ]), s=$(s_root[ℓ]), g=$(g_root[ℓ]), pct_clip=$(pct_clip_root[ℓ]), p=$(p_root[ℓ])")
        verbose && println("  one : Δ=$(Δ_one[ℓ]),  s=$(s_one[ℓ]),  g=$(g_one[ℓ]),  pct_clip=$(pct_clip_one[ℓ]), p=$(p_one[ℓ])")
    end

    
    for ℓ in 1:m
        C_root_scaled[ℓ] .*= s_root[ℓ]
        C_one_scaled[ℓ]  .*= s_one[ℓ]
    end

    verbose && println("==== quantize + fold (weighted) done ====")

    stats = (
        Δ_root        = Δ_root,
        Δ_one         = Δ_one,
        s_root        = s_root,
        s_one         = s_one,
        g_root        = g_root,
        g_one         = g_one,
        pct_clip_root = pct_clip_root,
        pct_clip_one  = pct_clip_one,
        p_root        = p_root,
        p_one         = p_one,
        bits_per_layer = bitsL,
        max_abs_round_root = max_abs_round_root,
        max_abs_round_one  = max_abs_round_one,
        use_weighted_quantile = use_weighted_quantile,
        auto_search_p = auto_search_p
    )

    return q_int_f32, C_root_scaled, C_one_scaled, stats
end




function quantize_coeffs_multibook_split_weighted2!(
    B::AbstractMatrix{BType},
    a::Matrix{T},
    C_root::Vector{Matrix{T}},
    C_one::Vector{Matrix{T}},
    parent::AbstractVector{UInt32};
    
    bits_first::Int = 6,
    bits_rest::Int  = 6,
    bits_per_layer::Union{Nothing,AbstractVector{Int}} = nothing,
    
    p_first_candidates::Vector{Float64} = [99.5, 99.8, 100.0],
    p_rest_candidates::Vector{Float64}  = [99.5, 99.8, 100.0],
    auto_search_p::Bool = false,
    p_first_range::Tuple{Float64,Float64,Float64} = (99.0, 100.0, 0.1),
    p_rest_range::Tuple{Float64,Float64,Float64}  = (99.0, 100.0, 0.1),
    use_weighted_quantile::Bool = false,   
    
    λ_global::Float64 = 0.0,               
    
    q_refine_sweeps::Int = 0,              
    q_refine_max_layer::Int = 5,           
    q_refine_step_limit::Int = 1,          
    verbose::Bool = true
) where {T<:AbstractFloat, BType<:Integer}

    m, n = size(a)
    @assert size(B) == (m, n)
    @assert length(C_root) == m
    @assert length(C_one)  == m
    @assert length(parent) == n

    d = size(C_root[1], 1)
    @assert all(size(C_root[ℓ],1) == d for ℓ in 1:m)
    @assert all(size(C_one[ℓ],1)  == d for ℓ in 1:m)

    
    bitsL = bits_per_layer === nothing ?
        [bits_first; fill(bits_rest, m-1)] :
        ( @assert length(bits_per_layer)==m; collect(bits_per_layer) )
    @assert all(bitsL .>= 2)

    Qvec = [ (1 << (bitsL[ℓ]-1)) - 1 for ℓ in 1:m ]  

    
    root_idx = Int[]
    one_idx  = Int[]
    for i in 1:n
        if parent[i] == 0
            push!(root_idx, i)
        else
            push!(one_idx, i)
        end
    end

    
    q_int_f32     = Matrix{Float32}(undef, m, n)
    C_root_scaled = [copy(C_root[ℓ]) for ℓ in 1:m]
    C_one_scaled  = [copy(C_one[ℓ])  for ℓ in 1:m]

    
    Δ_root        = ones(T, m)
    Δ_one         = ones(T, m)
    s_root_local  = ones(Float64, m)   
    s_one_local   = ones(Float64, m)
    s_root_final  = ones(Float64, m)   
    s_one_final   = ones(Float64, m)
    g_root        = ones(Float64, m)
    g_one         = ones(Float64, m)
    pct_clip_root = zeros(Float64, m)
    pct_clip_one  = zeros(Float64, m)
    p_root        = fill(100.0, m)
    p_one         = fill(100.0, m)

    
    function make_p_candidates(range::Tuple{Float64,Float64,Float64}, fallback::Vector{Float64})
        if !auto_search_p
            return fallback
        end
        pmin, pmax, pstep = range
        @assert pstep > 0
        ps = collect(pmin:pstep:pmax)
        
        
        
        return ps
    end
    p_first = make_p_candidates(p_first_range, p_first_candidates)
    p_rest  = make_p_candidates(p_rest_range,  p_rest_candidates)

    
    function precompute_norm2(C::Vector{Matrix{T}})
        out = Vector{Vector{T}}(undef, length(C))
        for ℓ in 1:length(C)
            K = size(C[ℓ], 2)
            norms = Vector{T}(undef, K)
            Cℓ = C[ℓ]
            @inbounds for k in 1:K
                acc = zero(T)
                v = @view Cℓ[:, k]
                for t in 1:d
                    x = v[t]
                    acc += x*x
                end
                norms[k] = acc
            end
            out[ℓ] = norms
        end
        return out
    end
    norm2_root = precompute_norm2(C_root)
    norm2_one  = precompute_norm2(C_one)

    
    @inline function quantile_type7_sorted(x_sorted::Vector{T}, p::Float64) where {T}
        n = length(x_sorted)
        if n == 0; return zero(T); end
        if p <= 0.0; return x_sorted[1]; end
        if p >= 1.0; return x_sorted[end]; end
        h = (n - 1) * p + 1
        j = floor(Int, h)
        g = h - j
        if j >= n; return x_sorted[end]; end
        return (one(T) - T(g)) * x_sorted[j] + T(g) * x_sorted[j+1]
    end

    @inline function weighted_quantile_sorted_step(x_sorted::Vector{T}, cumw::Vector{T}, p::Float64) where {T}
        totalw = cumw[end]
        if totalw <= zero(T)
            return x_sorted[end]
        end
        target = T(p) * totalw
        idx = searchsortedfirst(cumw, target)
        idx = clamp(idx, 1, length(x_sorted))
        return x_sorted[idx]
    end

    
    function quantize_group_signed!(
        layer::Int,
        idxs::Vector{Int},
        norms_group::Vector{Vector{T}},
        p_candidates::Vector{Float64},
        Δ_vec::Vector{T},
        s_local_vec::Vector{Float64},
        pct_clip_vec::Vector{Float64},
        p_vec::Vector{Float64}
    )
        if isempty(idxs)
            Δ_vec[layer] = one(T)
            s_local_vec[layer] = 1.0
            pct_clip_vec[layer] = 0.0
            p_vec[layer] = 100.0
            return
        end

        Q = Qvec[layer]
        L = length(idxs)

        a_sub   = Vector{T}(undef, L)
        abs_sub = Vector{T}(undef, L)
        w_sub   = Vector{T}(undef, L)

        normsℓ = norms_group[layer]
        @inbounds for k in 1:L
            i = idxs[k]
            aval = a[layer, i]
            a_sub[k] = aval
            abs_sub[k] = abs(aval)

            if use_weighted_quantile
                code = Int(B[layer,i])
                w_sub[k] = (code == 0) ? zero(T) : normsℓ[code]
            else
                w_sub[k] = one(T)
            end
        end

        x_sorted = copy(abs_sub)
        cumw = nothing
        if use_weighted_quantile
            perm = sortperm(x_sorted)
            x_sorted = x_sorted[perm]
            w_sorted = w_sub[perm]
            cumw = similar(w_sorted)
            run = zero(T)
            @inbounds for t in 1:L
                wt = w_sorted[t]
                run += (wt > 0 ? wt : zero(T))
                cumw[t] = run
            end
        else
            sort!(x_sorted)
        end
        xmax = x_sorted[end]

        @inline function get_Amax(p::Float64)
            if p >= 100.0
                return xmax
            end
            pp = p/100.0
            return use_weighted_quantile ?
                weighted_quantile_sorted_step(x_sorted, cumw, pp) :
                quantile_type7_sorted(x_sorted, pp)
        end

        best_cost = Inf
        best_Δ    = one(T)
        best_s    = 1.0
        best_clip = 0.0
        best_p    = 100.0

        q_tmp  = Vector{Int16}(undef, L)
        best_q = Vector{Int16}(undef, L)

        for p in p_candidates
            Amax = get_Amax(p)
            if Amax <= zero(T)
                continue
            end
            Δ = Amax / T(Q)

            clip_cnt = 0
            @inbounds for k in 1:L
                q = round(Int, a_sub[k] / Δ)
                if q > Q
                    q = Q; clip_cnt += 1
                elseif q < -Q
                    q = -Q; clip_cnt += 1
                end
                q_tmp[k] = Int16(q)
            end

            
            num = 0.0
            den = 0.0
            @inbounds for k in 1:L
                qk = Float64(q_tmp[k])
                num += Float64(a_sub[k]) * qk
                den += qk*qk
            end
            s_c = den > 0 ? (num/den) : 1.0

            err = 0.0
            @inbounds for k in 1:L
                e = Float64(a_sub[k]) - s_c * Float64(q_tmp[k])
                err += e*e
            end

            clip_ratio = Float64(clip_cnt)/L
            if err < best_cost
                best_cost = err
                best_Δ    = Δ
                best_s    = s_c
                best_clip = clip_ratio
                best_p    = p
                copy!(best_q, q_tmp)
            end
        end

        Δ_vec[layer] = best_Δ
        s_local_vec[layer] = best_s
        pct_clip_vec[layer] = best_clip
        p_vec[layer] = best_p

        @inbounds for k in 1:L
            i = idxs[k]
            q_int_f32[layer, i] = Float32(best_q[k])
        end
    end

    
    function precompute_gram(C::Vector{Matrix{T}})
        G = [Vector{Matrix{Float32}}(undef, m) for _ in 1:m]
        for ℓ in 1:m
            for k in 1:m
                G[ℓ][k] = Float32.(C[ℓ]' * C[k])
            end
        end
        return G
    end

    function fit_scales_to_recon_with_gram!(
        G, C::Vector{Matrix{T}}, idxs::Vector{Int}
    )
        if isempty(idxs)
            return ones(Float64, m)
        end
        A = zeros(Float64, m, m)
        b = zeros(Float64, m)

        code = Vector{Int}(undef, m)
        qv   = Vector{Float64}(undef, m)
        av   = Vector{Float64}(undef, m)

        @inbounds for ii in idxs
            for ℓ in 1:m
                code[ℓ] = Int(B[ℓ,ii])
                qv[ℓ]   = Float64(q_int_f32[ℓ,ii])
                av[ℓ]   = Float64(a[ℓ,ii])
            end

            for ℓ in 1:m
                cℓ = code[ℓ]; qℓ = qv[ℓ]
                if cℓ == 0 || qℓ == 0.0; continue; end
                for k in 1:m
                    ck = code[k]; qk = qv[k]
                    if ck == 0 || qk == 0.0; continue; end
                    A[ℓ,k] += qℓ*qk * Float64(G[ℓ][k][cℓ, ck])
                end
            end

            for ℓ in 1:m
                cℓ = code[ℓ]; qℓ = qv[ℓ]
                if cℓ == 0 || qℓ == 0.0; continue; end
                acc = 0.0
                for k in 1:m
                    ck = code[k]
                    if ck == 0; continue; end
                    acc += av[k] * Float64(G[ℓ][k][cℓ, ck])
                end
                b[ℓ] += qℓ * acc
            end
        end

        if λ_global > 0
            for ℓ in 1:m
                A[ℓ,ℓ] += λ_global
            end
        end

        s = nothing
        try
            F = cholesky(Symmetric(A); check=true)
            s = F \ b
        catch
            s = A \ b
        end
        return s
    end

function refine_q_icm_group!(
    G, idxs::Vector{Int}, s::Vector{Float64}
)
    if isempty(idxs) || q_refine_sweeps <= 0
        return (0, 0)
    end

    Lref = (q_refine_max_layer <= 0) ? m : min(m, q_refine_max_layer)
    step = q_refine_step_limit

    code   = Vector{Int}(undef, m)
    q_int  = Vector{Int}(undef, m)
    q_old  = Vector{Int}(undef, m)
    av     = Vector{Float64}(undef, m)
    dot_t  = Vector{Float64}(undef, m)

    changed = 0
    total   = 0

    @inbounds for ii in idxs
        for ℓ in 1:m
            code[ℓ]  = Int(B[ℓ,ii])
            q_int[ℓ] = Int(round(q_int_f32[ℓ,ii]))
            q_old[ℓ] = q_int[ℓ]
            av[ℓ]    = Float64(a[ℓ,ii])
        end

        for ℓ in 1:m
            cℓ = code[ℓ]
            if cℓ == 0
                dot_t[ℓ] = 0.0
                continue
            end
            acc = 0.0
            for k in 1:m
                ck = code[k]
                if ck == 0; continue; end
                acc += av[k] * Float64(G[ℓ][k][cℓ, ck])
            end
            dot_t[ℓ] = acc
        end

        for _ in 1:q_refine_sweeps
            for ℓ in 1:Lref
                cℓ = code[ℓ]
                if cℓ == 0
                    q_int[ℓ] = 0
                    continue
                end
                sℓ = s[ℓ]
                if sℓ == 0.0
                    q_int[ℓ] = 0
                    continue
                end

                cc = Float64(G[ℓ][ℓ][cℓ, cℓ])
                if cc == 0.0
                    q_int[ℓ] = 0
                    continue
                end

                sum_other = 0.0
                for k in 1:m
                    if k == ℓ; continue; end
                    ck = code[k]
                    qk = q_int[k]
                    if ck == 0 || qk == 0; continue; end
                    sum_other += s[k] * Float64(qk) * Float64(G[ℓ][k][cℓ, ck])
                end

                beta  = sℓ * (dot_t[ℓ] - sum_other)
                alpha = (sℓ*sℓ) * cc

                if alpha == 0.0
                    continue
                end

                Q = Qvec[ℓ]
                qcur = q_int[ℓ]

                if step > 0
                    lo = max(-Q, qcur - step)
                    hi = min( Q, qcur + step)

                    bestq = qcur
                    bestv = alpha*bestq*bestq - 2*beta*bestq
                    for qcand in lo:hi
                        v = alpha*qcand*qcand - 2*beta*qcand
                        if v < bestv
                            bestv = v
                            bestq = qcand
                        end
                    end
                    q_int[ℓ] = bestq
                else
                    qcont = beta / alpha
                    qnew = Int(round(qcont))
                    if qnew > Q
                        qnew = Q
                    elseif qnew < -Q
                        qnew = -Q
                    end
                    q_int[ℓ] = qnew
                end
            end
        end

        for ℓ in 1:Lref
            total += 1
            if q_int[ℓ] != q_old[ℓ]
                changed += 1
            end
            q_int_f32[ℓ, ii] = Float32(q_int[ℓ])
        end
    end

    return (changed, total)
end

    if verbose
        println("==== quantize_coeffs_multibook_split_weighted! (align recon t, signed-only, midtread) ====")
        println("layers=$m, n=$n, dim=$d")
        println("bits_per_layer=$(bitsL)  Q=$(Qvec)")
        println("root_count=$(length(root_idx)), one_count=$(length(one_idx))")
        println("use_weighted_quantile=$(use_weighted_quantile), auto_search_p=$(auto_search_p)")
        println("q_refine_sweeps=$(q_refine_sweeps), λ_global=$(λ_global)")
        println("p_first=$(p_first)")
        println("p_rest =$(p_rest)")
        println()
    end

    for ℓ in 1:m
        ps = (ℓ == 1) ? p_first : p_rest

        quantize_group_signed!(ℓ, root_idx, norm2_root, ps, Δ_root, s_root_local, pct_clip_root, p_root)
        quantize_group_signed!(ℓ, one_idx,  norm2_one,  ps, Δ_one,  s_one_local,  pct_clip_one,  p_one)

        if verbose
            println("L$ℓ bits=$(bitsL[ℓ])")
            println("  root: p=$(p_root[ℓ]) Δ=$(Δ_root[ℓ]) s_local=$(s_root_local[ℓ]) clip=$(pct_clip_root[ℓ])")
            println("  one : p=$(p_one[ℓ])  Δ=$(Δ_one[ℓ])  s_local=$(s_one_local[ℓ])  clip=$(pct_clip_one[ℓ])")
        end
    end

    G_root = precompute_gram(C_root)
    G_one  = precompute_gram(C_one)

    s_root_final .= fit_scales_to_recon_with_gram!(G_root, C_root, root_idx)
    s_one_final  .= fit_scales_to_recon_with_gram!(G_one,  C_one,  one_idx)

    if verbose
        println()
        println("==== global scales after first fit ====")
        for ℓ in 1:m
            println("L$ℓ: s_root=$(s_root_final[ℓ]) | s_one=$(s_one_final[ℓ])")
        end
    end

    if q_refine_sweeps > 0
        (chg_r, tot_r) = refine_q_icm_group!(G_root, root_idx, s_root_final)
        (chg_o, tot_o) = refine_q_icm_group!(G_one,  one_idx,  s_one_final)

        if verbose
            println()
            println("==== ICM refine q done ====")
            println("root changed = $chg_r / $tot_r  ($(tot_r>0 ? 100*chg_r/tot_r : 0.0)%)")
            println("one  changed = $chg_o / $tot_o  ($(tot_o>0 ? 100*chg_o/tot_o : 0.0)%)")
        end

        s_root_final .= fit_scales_to_recon_with_gram!(G_root, C_root, root_idx)
        s_one_final  .= fit_scales_to_recon_with_gram!(G_one,  C_one,  one_idx)

        if verbose
            println()
            println("==== global scales after refit (post-ICM) ====")
            for ℓ in 1:m
                println("L$ℓ: s_root=$(s_root_final[ℓ]) | s_one=$(s_one_final[ℓ])")
            end
        end
    end

    for ℓ in 1:m
        g_root[ℓ] = s_root_final[ℓ] / Float64(Δ_root[ℓ])
        g_one[ℓ]  = s_one_final[ℓ]  / Float64(Δ_one[ℓ])

        C_root_scaled[ℓ] .*= T(s_root_final[ℓ])
        C_one_scaled[ℓ]  .*= T(s_one_final[ℓ])
    end

    if verbose
        println()
        println("==== final (g = s/Δ) ====")
        for ℓ in 1:m
            println("L$ℓ: g_root=$(g_root[ℓ]) | g_one=$(g_one[ℓ])")
        end
        println("==== done ====")
    end

    stats = (
        bits_per_layer = bitsL,
        Q_per_layer = Qvec,
        Δ_root = Δ_root, Δ_one = Δ_one,
        p_root = p_root, p_one = p_one,
        pct_clip_root = pct_clip_root, pct_clip_one = pct_clip_one,
        s_root_local = s_root_local, s_one_local = s_one_local,
        s_root_final = s_root_final, s_one_final = s_one_final,
        g_root = g_root, g_one = g_one,
        use_weighted_quantile = use_weighted_quantile,
        auto_search_p = auto_search_p,
        q_refine_sweeps = q_refine_sweeps,
        λ_global = λ_global
    )

    return q_int_f32, C_root_scaled, C_one_scaled, stats
end
