module Beam_search

using LinearAlgebra
using Base.Threads


export init_prefix_ls_beam_workspace2,beam_quantize_prefix_ls_incremental!,PrefixLSBeamWorkspace2

export init_single_prefix_ls_beam_workspace,beam_quantize_prefix_ls_single!,SinglePrefixLSBeamWorkspace





struct PrefixLSBeamTLS{T,BType<:Integer}
    codes_curr::Matrix{BType}    
    codes_next::Matrix{BType}

    errs_curr::Vector{T}         
    errs_next::Vector{T}

    a_curr::Matrix{T}            
    a_next::Matrix{T}

    chol_curr::Vector{Matrix{T}} 
    chol_next::Vector{Matrix{T}}

    tmp_a::Vector{T}             
    tmp_v::Vector{T}             
    tmp_y::Vector{T}             
    A_small::Matrix{T}           
end




struct PrefixLSBeamWorkspace2{T,BType<:Integer}
    pool::Channel{PrefixLSBeamTLS{T,BType}}
    pool_size::Int
end

@inline function _prefix_beam_pool_size()
    return max(2, Threads.nthreads() * 2)
end

function _make_prefix_ls_beam_tls(
    ::Type{T},
    ::Type{BType},
    m::Int,
    H_beam::Int,
) where {T<:AbstractFloat,BType<:Integer}
    return PrefixLSBeamTLS{T,BType}(
        Matrix{BType}(undef, m, H_beam),
        Matrix{BType}(undef, m, H_beam),
        fill(T(Inf), H_beam),
        fill(T(Inf), H_beam),
        zeros(T, m, H_beam),
        zeros(T, m, H_beam),
        [zeros(T, m, m) for _ in 1:H_beam],
        [zeros(T, m, m) for _ in 1:H_beam],
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m, m),
    )
end

function init_prefix_ls_beam_workspace2(
    ::Type{T},
    ::Type{BType},
    m::Int,
    H_beam::Int,
) where {T<:AbstractFloat,BType<:Integer}
    pool_size = _prefix_beam_pool_size()
    tls_proto = _make_prefix_ls_beam_tls(T, BType, m, H_beam)
    pool = Channel{typeof(tls_proto)}(pool_size)
    put!(pool, tls_proto)
    for _ in 2:pool_size
        put!(pool, _make_prefix_ls_beam_tls(T, BType, m, H_beam))
    end

    return PrefixLSBeamWorkspace2{T,BType}(pool, pool_size)
end














function solve_chol_upper!(
    R::AbstractMatrix{T},
    L::Int,
    a_in::AbstractVector{T},
    v::AbstractVector{T},
    ytmp::AbstractVector{T},
) where {T<:AbstractFloat}

    @inbounds begin
        
        for i in 1:L
            s = a_in[i]
            for k in 1:(i-1)
                s -= R[k, i] * ytmp[k]
            end
            ytmp[i] = s / R[i, i]
        end

        
        for i in L:-1:1
            s = ytmp[i]
            for k in (i+1):L
                s -= R[i, k] * v[k]
            end
            v[i] = s / R[i, i]
        end
    end
    return nothing
end





















function beam_quantize_prefix_ls_incremental!(
    X::AbstractMatrix{T},
    precomp,
    xC::AbstractMatrix{T},
    B::AbstractMatrix{BType},
    ws::PrefixLSBeamWorkspace2{T,BType},
    H_beam::Int;
    fix_first_layer::Bool = false,
) where {T<:AbstractFloat,BType<:Integer}

    d, n = size(X)
    H_flat, n2 = size(xC)
    @assert n == n2
    m, nB = size(B)
    @assert nB == n

    offsets = precomp.offsets
    h_vec   = precomp.h_vec
    G       = precomp.G
    H_pre   = precomp.H
    @assert H_pre == H_flat

    pool = ws.pool

    nthreads = Threads.nthreads()
    chunk_size = max(1, cld(n, nthreads))

    Threads.@sync for start in 1:chunk_size:n
        stop = min(start + chunk_size - 1, n)
        Threads.@spawn begin
            tls = take!(pool)
            try
                codes_curr = tls.codes_curr
                codes_next = tls.codes_next
                errs_curr  = tls.errs_curr
                errs_next  = tls.errs_next
                a_curr     = tls.a_curr
                a_next     = tls.a_next
                chol_curr  = tls.chol_curr
                chol_next  = tls.chol_next
                tmp_a      = tls.tmp_a
                tmp_v      = tls.tmp_v
                tmp_y      = tls.tmp_y
                A_small    = tls.A_small

                @inbounds for i in start:stop
                    Xi   = @view X[:, i]
                    xC_i = @view xC[:, i]
                    norm_x2 = dot(Xi, Xi)

            
            
            for h in 1:H_beam
                errs_curr[h] = T(Inf)
                errs_next[h] = T(Inf)
            end

            if fix_first_layer
                
                
                k0 = Int(B[1, i])              
                off1 = offsets[1]
                flat0 = off1 + k0 - 1          

                α0 = G[flat0, flat0]           
                γ0 = xC_i[flat0]               

                a0 = γ0 / α0
                E0 = norm_x2 - γ0 * a0

                
                errs_next[1]      = E0
                codes_next[1, 1]  = BType(k0)
                a_next[1, 1]      = a0

                
                R0 = chol_next[1]
                R0[1,1] = sqrt(α0)

                curr_H = 1

            else 
                
                K1  = h_vec[1] 
                off1 = offsets[1]

                for k in 1:K1    
                    flat = off1 + k - 1
                    α = G[flat, flat]  
                    γ = xC_i[flat]     

                    
                    a1 = γ / α
                    E1 = norm_x2 - γ * a1

                    
                    worst_idx = 1
                    worst_err = errs_next[1]
                    for h in 2:H_beam
                        if errs_next[h] > worst_err
                            worst_err = errs_next[h]
                            worst_idx = h
                        end
                    end

                    if E1 < worst_err
                        errs_next[worst_idx] = E1
                        codes_next[1, worst_idx] = BType(k)
                        a_next[1, worst_idx]     = a1
                        R = chol_next[worst_idx]     
                        R[1,1] = sqrt(α)
                    end
                end

                
                curr_H = 0
                for h in 1:H_beam
                    if errs_next[h] < T(Inf)
                        curr_H += 1
                    end
                end
                if curr_H == 0
                    
                    curr_H = 1
                    errs_next[1] = norm_x2
                    codes_next[1,1] = BType(1)
                    a_next[1,1] = zero(T)
                    chol_next[1][1,1] = one(T)
                end
            end 

            
            tmp_codes = codes_curr; codes_curr = codes_next; codes_next = tmp_codes
            tmp_errs  = errs_curr;  errs_curr  = errs_next;  errs_next  = tmp_errs
            tmp_a_mat = a_curr;     a_curr     = a_next;     a_next     = tmp_a_mat
            tmp_chol  = chol_curr;  chol_curr  = chol_next;  chol_next  = tmp_chol

            
            for L in 2:m
                
                for h in 1:H_beam
                    errs_next[h] = T(Inf)
                end

                
                for h in 1:curr_H
                    E_old   = errs_curr[h]
                    R_pref  = chol_curr[h]   
                    a_old   = @view a_curr[1:L-1, h]

                    K_L  = h_vec[L]
                    offL = offsets[L]

                    
                    for k in 1:K_L
                        flat_L = offL + k - 1
                        α = G[flat_L, flat_L]
                        γ = xC_i[flat_L]

                        
                        L1 = L - 1
                        for t in 1:L1
                            code_t  = Int(codes_curr[t, h])
                            flat_t  = offsets[t] + code_t - 1
                            tmp_a[t] = G[flat_t, flat_L]
                        end

                        
                        solve_chol_upper!(R_pref, L1, tmp_a, tmp_v, tmp_y)

                        
                        δ = zero(T)
                        η = zero(T)
                        for t in 1:L1
                            at = tmp_a[t]
                            δ += at * tmp_v[t]
                            η += at * a_old[t]
                        end

                        denom = α - δ
                        
                        if denom == 0
                            continue
                        end

                        diffγ = γ - η
                        ΔE    = (diffγ * diffγ) / denom
                        E_new = E_old - ΔE
                        if E_new < zero(T)
                            E_new = zero(T)
                        end

                        
                        
                        worst_idx = 1
                        worst_err = errs_next[1]
                        for hh in 2:H_beam
                            if errs_next[hh] > worst_err
                                worst_err = errs_next[hh]
                                worst_idx = hh
                            end
                        end

                        if E_new < worst_err
                            errs_next[worst_idx] = E_new
                            
                            for t in 1:L-1
                                codes_next[t, worst_idx] = codes_curr[t, h]
                            end
                            
                            codes_next[L, worst_idx] = BType(k)

                            
                            x2 = diffγ / denom
                            for t in 1:L1
                                a_next[t, worst_idx] = a_old[t] - tmp_v[t] * x2
                            end
                            a_next[L, worst_idx] = x2
                        end
                    end
                end

                
                new_H = 0
                for h in 1:H_beam
                    if errs_next[h] < T(Inf)
                        new_H += 1
                    end
                end
                if new_H == 0
                    
                    new_H = min(curr_H, H_beam)
                    for h in 1:new_H
                        errs_next[h] = errs_curr[h]
                        for t in 1:L
                            codes_next[t, h] = codes_curr[t, h]
                            a_next[t, h]     = a_curr[t, h]
                        end
                    end
                end

                
                for h in 1:new_H
                    
                    for t in 1:L
                        code_t = Int(codes_next[t, h])
                        flat_t = offsets[t] + code_t - 1
                        for u in 1:t
                            code_u = Int(codes_next[u, h])
                            flat_u = offsets[u] + code_u - 1
                            val = G[flat_t, flat_u]
                            A_small[t,u] = val
                            A_small[u,t] = val
                        end
                    end
                    
                    
                    subA = @view A_small[1:L, 1:L]
                    LAPACK.potrf!('U', subA)

                    
                    R_new = chol_next[h]
                    for r in 1:L
                        for c in r:L
                            R_new[r,c] = subA[r,c]
                        end
                    end
                end

                
                tmp_codes = codes_curr; codes_curr = codes_next; codes_next = tmp_codes
                tmp_errs  = errs_curr;  errs_curr  = errs_next;  errs_next  = tmp_errs
                tmp_a_mat = a_curr;     a_curr     = a_next;     a_next     = tmp_a_mat
                tmp_chol  = chol_curr;  chol_curr  = chol_next;  chol_next  = tmp_chol

                curr_H = new_H
            end

            
            best_idx = 1
            best_err = errs_curr[1]
            for h in 2:curr_H
                if errs_curr[h] < best_err
                    best_err = errs_curr[h]
                    best_idx = h
                end
            end

            for L in 1:m
                B[L, i] = codes_curr[L, best_idx]
            end
                end
            finally
                put!(pool, tls)
            end
        end
    end

    return nothing
end



















struct SinglePrefixLSBeamWorkspace{T,BType<:Integer}
    codes_curr::Matrix{BType}    
    codes_next::Matrix{BType}

    errs_curr::Vector{T}         
    errs_next::Vector{T}

    a_curr::Matrix{T}            
    a_next::Matrix{T}

    chol_curr::Vector{Matrix{T}} 
    chol_next::Vector{Matrix{T}}

    tmp_a::Vector{T}             
    tmp_v::Vector{T}             
    tmp_y::Vector{T}             
    A_small::Matrix{T}           
end

function init_single_prefix_ls_beam_workspace(
    ::Type{T},
    ::Type{BType},
    m::Int,
    H_beam::Int,
) where {T<:AbstractFloat,BType<:Integer}

    codes_curr = Matrix{BType}(undef, m, H_beam)
    codes_next = Matrix{BType}(undef, m, H_beam)

    errs_curr  = fill(T(Inf), H_beam)
    errs_next  = fill(T(Inf), H_beam)

    a_curr     = zeros(T, m, H_beam)
    a_next     = zeros(T, m, H_beam)

    chol_curr  = [zeros(T, m, m) for _ in 1:H_beam]
    chol_next  = [zeros(T, m, m) for _ in 1:H_beam]

    tmp_a      = zeros(T, m)
    tmp_v      = zeros(T, m)
    tmp_y      = zeros(T, m)
    A_small    = zeros(T, m, m)

    return SinglePrefixLSBeamWorkspace{T,BType}(
        codes_curr, codes_next,
        errs_curr, errs_next,
        a_curr, a_next,
        chol_curr, chol_next,
        tmp_a, tmp_v, tmp_y,
        A_small,
    )
end
















function beam_quantize_prefix_ls_single!(
    x::AbstractVector{T},
    precomp,
    xC_x::AbstractVector{T},
    B_out::AbstractVector{BType},
    a_out::AbstractVector{T},
    ws::SinglePrefixLSBeamWorkspace{T,BType};
    H_beam::Int,
) where {T<:AbstractFloat,BType<:Integer}

    d      = length(x)
    H_flat = length(xC_x)

    offsets = precomp.offsets
    h_vec   = precomp.h_vec
    G       = precomp.G
    H_pre   = precomp.H
    m       = precomp.m
    @assert H_pre == H_flat
    @assert length(B_out) == m
    @assert length(a_out) == m

    
    codes_curr = ws.codes_curr
    codes_next = ws.codes_next
    errs_curr  = ws.errs_curr
    errs_next  = ws.errs_next
    a_curr     = ws.a_curr
    a_next     = ws.a_next
    chol_curr  = ws.chol_curr
    chol_next  = ws.chol_next
    tmp_a      = ws.tmp_a
    tmp_v      = ws.tmp_v
    tmp_y      = ws.tmp_y
    A_small    = ws.A_small

    @inbounds begin
        
        norm_x2 = dot(x, x)

        
        for h in 1:H_beam
            errs_curr[h] = T(Inf)
            errs_next[h] = T(Inf)
        end

        
        K1  = h_vec[1]
        off = offsets[1]

        for k in 1:K1
            flat = off + k - 1
            α = G[flat, flat]         
            γ = xC_x[flat]            

            
            a1 = γ / α
            E1 = norm_x2 - (γ * a1)

            
            
            worst_idx = 1
            worst_err = errs_next[1]
            for h in 2:H_beam
                if errs_next[h] > worst_err
                    worst_err = errs_next[h]
                    worst_idx = h
                end
            end

            if E1 < worst_err
                errs_next[worst_idx]       = E1
                codes_next[1, worst_idx]   = BType(k)
                a_next[1, worst_idx]       = a1

                
                R = chol_next[worst_idx]
                R[1,1] = sqrt(α)
            end
        end

        
        curr_H = 0
        for h in 1:H_beam
            if errs_next[h] < T(Inf)
                curr_H += 1
            end
        end

        if curr_H == 0
            
            curr_H = 1
            errs_next[1] = norm_x2
            codes_next[1,1] = BType(1)
            a_next[1,1] = zero(T)
            chol_next[1][1,1] = one(T)
        end

        
        codes_curr, codes_next = codes_next, codes_curr
        errs_curr,  errs_next  = errs_next,  errs_curr
        a_curr,     a_next     = a_next,     a_curr
        chol_curr,  chol_next  = chol_next,  chol_curr

        
        for L in 2:m
            
            for h in 1:H_beam
                errs_next[h] = T(Inf)
            end

            
            for h in 1:curr_H
                E_old   = errs_curr[h]
                R_pref  = chol_curr[h]      
                a_old   = @view a_curr[1:L-1, h]
                L1      = L - 1

                K_L  = h_vec[L]
                offL = offsets[L]

                for k in 1:K_L
                    flat_L = offL + k - 1
                    α = G[flat_L, flat_L]
                    γ = xC_x[flat_L]

                    
                    for t in 1:L1
                        code_t = Int(codes_curr[t, h])
                        flat_t = offsets[t] + code_t - 1
                        tmp_a[t] = G[flat_t, flat_L]
                    end

                    
                    solve_chol_upper!(R_pref, L1, tmp_a, tmp_v, tmp_y)

                    
                    δ = zero(T)
                    η = zero(T)
                    for t in 1:L1
                        at = tmp_a[t]
                        δ += at * tmp_v[t]
                        η += at * a_old[t]
                    end

                    denom = α - δ
                    
                    if denom == 0
                        continue
                    end

                    diffγ = γ - η
                    ΔE    = (diffγ * diffγ) / denom
                    E_new = E_old - ΔE
                    if E_new < zero(T)
                        E_new = zero(T)
                    end

                    
                    worst_idx = 1
                    worst_err = errs_next[1]
                    for hh in 2:H_beam
                        if errs_next[hh] > worst_err
                            worst_err = errs_next[hh]
                            worst_idx = hh
                        end
                    end

                    if E_new < worst_err
                        errs_next[worst_idx] = E_new
                        
                        for t in 1:L1
                            codes_next[t, worst_idx] = codes_curr[t, h]
                        end
                        codes_next[L, worst_idx] = BType(k)

                        
                        x2 = diffγ / denom
                        for t in 1:L1
                            a_next[t, worst_idx] = a_old[t] - tmp_v[t] * x2
                        end
                        a_next[L, worst_idx] = x2
                    end
                end
            end

            
            new_H = 0
            for h in 1:H_beam
                if errs_next[h] < T(Inf)
                    new_H += 1
                end
            end
            if new_H == 0
                
                new_H = min(curr_H, H_beam)
                for h in 1:new_H
                    errs_next[h] = errs_curr[h]
                    for t in 1:L
                        codes_next[t, h] = codes_curr[t, h]
                        a_next[t, h]     = a_curr[t, h]
                    end
                end
            end

            
            for h in 1:new_H
                
                for t in 1:L
                    code_t = Int(codes_next[t, h])
                    flat_t = offsets[t] + code_t - 1
                    for u in 1:t
                        code_u = Int(codes_next[u, h])
                        flat_u = offsets[u] + code_u - 1
                        val = G[flat_t, flat_u]
                        A_small[t,u] = val
                        A_small[u,t] = val
                    end
                end

                subA = @view A_small[1:L, 1:L]
                LAPACK.potrf!('U', subA)

                R_new = chol_next[h]
                for r in 1:L
                    for c in r:L
                        R_new[r,c] = subA[r,c]
                    end
                end
            end

            
            codes_curr, codes_next = codes_next, codes_curr
            errs_curr,  errs_next  = errs_next,  errs_curr
            a_curr,     a_next     = a_next,     a_curr
            chol_curr,  chol_next  = chol_next,  chol_curr

            curr_H = new_H
        end

        
        best_idx = 1
        best_err = errs_curr[1]
        for h in 2:curr_H
            if errs_curr[h] < best_err
                best_err = errs_curr[h]
                best_idx = h
            end
        end

        for L in 1:m
            B_out[L] = codes_curr[L, best_idx]
            a_out[L] = a_curr[L, best_idx]
        end
    end

    return nothing
end


end
