using LinearAlgebra
using Random
using Base.Threads
using LoopVectorization
using Printf


function dynamic_icm_with_ils_noabs_nonormal!(
    X::Matrix{T},                
    B::Matrix{BType},            
    a::Matrix{T},                
    rC::Matrix{T},               
    xC::Matrix{T},               
    C_contiguous::Array{T,3},    
    precomp,
    ils_iters::Int = 10,         
    icm_iters::Int=4,            
    perturb_k::Int = 4,          
    verbose::Bool = false
) where {T<:AbstractFloat, BType<:Integer}
    G            = precomp.G
    offsets      = precomp.offsets
    invnorm_flat = precomp.invnorm_flat
    h_vec        = precomp.h_vec
    H            = precomp.H
    m            = precomp.m
    d            = precomp.d
    dX, n = size(X)

    
    nthreads = Threads.nthreads()
    BLAS.set_num_threads(nthreads) 
    mul!(xC, transpose(precomp.C_all), X)
    BLAS.set_num_threads(1)

    
    dynamic_icm_encoding_noabs_nonormal_clean_fast_for_ils1!(X, B, a,rC,xC, C_contiguous, icm_iters; precomp=precomp,verbose=verbose)
    base_cost = Vector{T}(undef, n)
    _costs!(base_cost, X, B, a, C_contiguous)

    
    for outer in 1:ils_iters
        verbose && println("[ILS] round $outer/$ils_iters")

        
        B_cand = copy(B)
        a_cand = copy(a)

        
        ksel = clamp(perturb_k, 1, m)

        
        
         Lmat = rand(2:m, ksel, n)  

        
        Hmat = h_vec[Lmat]  

        
        Old = Array{Int}(undef, ksel, n)
        @inbounds @threads for c in 1:n
            for r in 1:ksel
                Old[r,c] = Int(B_cand[Lmat[r,c], c])
            end
        end

        
        U   = rand(T, ksel, n)
        New = Array{Int}(undef, ksel, n)
        @inbounds @threads for c in 1:n
            for r in 1:ksel
                h  = Hmat[r,c]
                t  = Int(floor(U[r,c] * (h - 1)))  
                nc = t + 1                         
                if nc >= Old[r,c]                  
                    nc += 1
                end
                New[r,c] = nc                      
            end
        end

        
        @inbounds @threads for c in 1:n
            for r in 1:ksel
                jlayer = Lmat[r,c]
                B_cand[jlayer, c] = BType(New[r,c])
            end
        end

        
        @time dynamic_icm_encoding_noabs_nonormal_clean_fast_for_ils1!(X, B_cand, a_cand,rC,xC, C_contiguous, icm_iters; precomp=precomp,verbose=verbose)

        
        cand_cost = Vector{T}(undef, n)
        _costs!(cand_cost, X, B_cand, a_cand, C_contiguous)

        
        accepted = Threads.Atomic{Int}(0)
        @threads for j in 1:n
            if cand_cost[j] + eps(T) < base_cost[j]
                
                @simd for r in 1:m
                    B[r, j] = B_cand[r, j]
                    a[r, j] = a_cand[r, j]
                end
                base_cost[j] = cand_cost[j]
                Threads.atomic_add!(accepted, 1)
            end
        end

        avg_cost = mean(base_cost)
        verbose && println("  accepted $(accepted[]) / $n samples, avg cost $avg_cost")
    end

    return nothing
end


function dynamic_icm_encoding_noabs_nonormal_clean_fast_for_ils1!(
    X::AbstractMatrix{T},        
    B::AbstractMatrix{BType},            
    a::AbstractMatrix{T},                
    rC::Matrix{T},               
    xC::Matrix{T},               
    C_contiguous::Array{T,3},    
    max_iters::Integer;          
    precomp,    
    verbose::Bool = true
) where {T<:AbstractFloat,BType<:Integer}

    d, n = size(X)
    m    = size(B, 1)
    h_max = size(C_contiguous, 2)

    
    offsets      = precomp.offsets
    h_vec        = precomp.h_vec
    invnorm_flat = precomp.invnorm_flat
    H            = precomp.H
    G            = precomp.G

    
    n_threads = Threads.nthreads()
    pool_size = max(2, n_threads * 2)
    A_pool = Channel{Matrix{T}}(pool_size)
    a_backup_pool = Channel{Vector{T}}(pool_size)
    for _ in 1:pool_size
        put!(A_pool, zeros(T, m, m))
        put!(a_backup_pool, zeros(T, m))
    end

    
    optimized_least_squares_all!(a, X, C_contiguous, B, A_pool, n, m, d, h_max)

    
    @threads for j in 1:n
        @inbounds @simd for p in 1:H
            rC[p, j] = xC[p, j]
        end
        @inbounds for ℓ in 1:m
            α = a[ℓ, j]
            col = offsets[ℓ] + Int(B[ℓ,j]) - 1
            @simd for p in 1:H
                rC[p, j] -= α * G[p, col]
            end
        end
    end

    
    X_norm2  = Vector{T}(undef, n)
    cur_cost = Vector{T}(undef, n)

    @threads for j in 1:n
        s = zero(T)
        @inbounds @simd for t in 1:d
            v = X[t, j]
            s += v * v
        end
        X_norm2[j] = s

        xCj = @view xC[:, j]
        Bj  = @view B[:, j]
        aj  = @view a[:, j]

        term1 = zero(T)
        term2 = zero(T)
        @inbounds for ℓ in 1:m
            kℓ = Int(Bj[ℓ])
            fℓ = offsets[ℓ] + kℓ - 1
            αℓ = aj[ℓ]
            term1 += αℓ * xCj[fℓ]
            @inbounds for k in 1:m
                kk = Int(Bj[k])
                fk = offsets[k] + kk - 1
                term2 += αℓ * aj[k] * G[fℓ, fk]
            end
        end
        cur_cost[j] = X_norm2[j] - 2*term1 + term2
    end

    
    active       = trues(n)           
    changed_iter = falses(n)          

    @inline function sample_cost_from_xC!(
        Xn2::T,
        xCj::AbstractVector{T},
        Bj::AbstractVector{BType},
        aj::AbstractVector{T},
        offsets::Vector{Int},
        G::AbstractMatrix{T},
        m::Int,
    )::T
        term1 = zero(T)
        term2 = zero(T)
        @inbounds for ℓ in 1:m
            kℓ = Int(Bj[ℓ])
            fℓ = offsets[ℓ] + kℓ - 1
            αℓ = aj[ℓ]
            term1 += αℓ * xCj[fℓ]
            @inbounds for k in 1:m
                kk = Int(Bj[k])
                fk = offsets[k] + kk - 1
                term2 += αℓ * aj[k] * G[fℓ, fk]
            end
        end
        return Xn2 - 2*term1 + term2
    end

    
    for iter in 1:max_iters
        
        fill!(changed_iter, false)

        order = randperm(m)
        for jlayer in order

            if jlayer == 1
                continue
            end

            chunk_size = max(1, cld(n, n_threads))
            Threads.@sync for start in 1:chunk_size:n
                stop = min(start + chunk_size - 1, n)
                Threads.@spawn begin
                    A = take!(A_pool)
                    ai_backup = take!(a_backup_pool)
                    try
                        for i in start:stop
                            
                            if !active[i]
                                continue
                            end

                            old_code = Int(B[jlayer, i])
                            old_flat = offsets[jlayer] + old_code - 1

                            best_code = old_code
                            best_val  = -typemax(T)
                            startf    = offsets[jlayer]
                            stopf     = startf + h_vec[jlayer] - 1

                            @inbounds for flat in startf:stopf
                                val = rC[flat, i] + a[jlayer, i] * G[flat, old_flat]
                                val *= invnorm_flat[flat]
                                if val > best_val
                                    best_val  = val
                                    best_code = flat - startf + 1
                                end
                            end

                            if best_code != old_code
                                Xi        = @view X[:, i]
                                Bi        = @view B[:, i]
                                ai        = @view a[:, i]

                                
                                @inbounds @simd for ℓ in 1:m
                                    ai_backup[ℓ] = ai[ℓ]
                                end

                                
                                B[jlayer, i] = BType(best_code)
                                optimized_least_squares_single!(ai, Xi, C_contiguous, Bi, A, m, d, h_max)

                                
                                @inbounds @simd for p in 1:H
                                    rC[p, i] = xC[p, i]
                                end
                                @inbounds for ℓ in 1:m
                                    α  = ai[ℓ]
                                    fℓ = offsets[ℓ] + Int(B[ℓ, i]) - 1
                                    @simd for p in 1:H
                                        rC[p, i] -= α * G[p, fℓ]
                                    end
                                end

                                
                                xCj     = @view xC[:, i]
                                new_cost = sample_cost_from_xC!(X_norm2[i], xCj, Bi, ai, offsets, G, m)

                                if new_cost + eps(T) < cur_cost[i]
                                    
                                    cur_cost[i]    = new_cost
                                    changed_iter[i] = true    
                                else
                                    
                                    B[jlayer, i] = BType(old_code)
                                    @inbounds @simd for ℓ in 1:m
                                        ai[ℓ] = ai_backup[ℓ]
                                    end

                                    @inbounds @simd for p in 1:H
                                        rC[p, i] = xC[p, i]
                                    end
                                    @inbounds for ℓ in 1:m
                                        α  = ai[ℓ]
                                        fℓ = offsets[ℓ] + Int(B[ℓ, i]) - 1
                                        @simd for p in 1:H
                                            rC[p, i] -= α * G[p, fℓ]
                                        end
                                    end
                                end
                            end
                        end
                    finally
                        put!(a_backup_pool, ai_backup)
                        put!(A_pool, A)
                    end
                end
            end
        end

        
        
        @inbounds for i in 1:n
            if active[i]
                if !changed_iter[i]
                    
                    active[i] = false
                
                
                end
            end
        end

        
        
        
        
        
    end

    return nothing
end
