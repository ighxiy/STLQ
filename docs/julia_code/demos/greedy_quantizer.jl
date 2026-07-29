using Base.Threads


function quantize_with_kmeans_abs_nonormal_single_extend_allcodes_fast!(
    X::Vector{T},                    
    B_init_tls::AbstractVector{BType},       
    len_rate_init_tls::AbstractVector{T},    
    m::Int,
    d::Int,
    
    H::Int,
    C_all::Matrix{T},                
    G::Matrix{T},                    
    offsets::Vector{Int},            
    flat_layer::Vector{Int},         
    invnorm_flat::Vector{T},         
    rC_buf::Vector{T}                
) where {T <: AbstractFloat,BType <: Integer}

    
    available = trues(m)

    
    
    @inbounds for j in 1:H
        dot_val =zero(T)
        @inbounds @simd for i in 1:d
            dot_val += C_all[i,j] * X[i]
        end
        rC_buf[j] = dot_val
    end
    
    @inbounds for t in 1:m
        
        best_flat::Int = 0
        best_abs_adj::T = -typemax(T)        
        best_adj::T = zero(T)                
        
        @inbounds for jflat in 1:H
            ℓ = flat_layer[jflat]
            if !available[ℓ]; continue; end
            adj = rC_buf[jflat] * invnorm_flat[jflat]    
            aabs = abs(adj)
            if aabs > best_abs_adj
                best_abs_adj = aabs
                best_adj = adj
                best_flat = jflat
            end
        end

        
        ℓbest = flat_layer[best_flat]
        kbest = best_flat - offsets[ℓbest] + 1          
        invn  = invnorm_flat[best_flat]
        αbest = best_adj * invn                          

        
        B_init_tls[ℓbest]        = BType(kbest)
        len_rate_init_tls[ℓbest] = αbest

        
        
        @views @inbounds begin
            colG = G[:, best_flat]      
            @simd for j in 1:H
                rC_buf[j] -= αbest * colG[j]
            end
        end

        available[ℓbest] = false
    end
end



function quantize_with_kmeans_abs_nonormal_extend_allcodes_fast!(
    X::AbstractMatrix{T},                       
    C::Vector{Matrix{T}},               
    pre;                  
    realign_every::Int = 0              
) where {T<:AbstractFloat}

    d, n = size(X)
    m = length(C)
    h_vec = [size(C[ℓ],2) for ℓ in 1:m]
    H = sum(h_vec)

    C_all         = pre.C_all
    G             = pre.G
    offsets       = pre.offsets
    flat_layer    = pre.flat_layer
    invnorm_flat  = pre.invnorm_flat

    
    B  = Matrix{UInt32}(undef, m, n)    
    A  = Matrix{T}(undef, m, n)         
    

    nthreads = Threads.nthreads()
    pool_size = max(2, nthreads * 2)
    rC_pool = Channel{Vector{T}}(pool_size)
    for _ in 1:pool_size
        put!(rC_pool, zeros(T, H))
    end
    chunk_size = max(1, cld(n, nthreads))
    Threads.@sync for start in 1:chunk_size:n
        stop = min(start + chunk_size - 1, n)
        Threads.@spawn begin
            rC = take!(rC_pool)
            try
                for j in start:stop
        
        
        v_xj = @view X[:, j]
        dot_val =zero(T)
        @inbounds for k in 1:H
            dot_val =zero(T)
            v_ck = @view C_all[:, k]
            @inbounds @simd for i in 1:d
                dot_val += v_ck[i] * v_xj[i]
            end
            rC[k] = dot_val
        end
        
        available = trues(m)

        
        jstart = offsets[1]
        jend   = jstart + h_vec[1] - 1

        best_flat::Int = jstart
        best_abs_adj::T = -typemax(T)
        best_adj::T = zero(T)

        @inbounds for jf in jstart:jend
            adj  = rC[jf] * invnorm_flat[jf]   
            aabs = abs(adj)
            if aabs > best_abs_adj
                best_abs_adj = aabs
                best_adj     = adj
                best_flat    = jf
            end
        end

        ℓ1   = 1
        k1   = best_flat - jstart + 1
        invn = invnorm_flat[best_flat]
        α1   = best_adj * invn                    

        
        B[ℓ1, j] = UInt32(k1)
        A[ℓ1, j] = α1

        
        @views begin
            colG = G[:, best_flat]
            @inbounds @simd for t in 1:H
                rC[t] -= α1 * colG[t]
            end
        end
        
        
        
        
        
        
        
        available[ℓ1] = false

        
        
        
        

        
        for step in 2:m
            best_flat  = 0
            best_abs_adj = -typemax(T)
            best_adj   = zero(T)

            
            @inbounds for jf in 1:H
                ℓ = flat_layer[jf]
                if !available[ℓ]; continue; end
                adj  = rC[jf] * invnorm_flat[jf]
                aabs = abs(adj)
                if aabs > best_abs_adj
                    best_abs_adj = aabs
                    best_adj     = adj
                    best_flat    = jf
                end
            end

            ℓbest = flat_layer[best_flat]
            kbest = best_flat - offsets[ℓbest] + 1
            αbest = best_adj * invnorm_flat[best_flat]  

            
            B[ℓbest, j] = UInt32(kbest)
            A[ℓbest, j] = αbest

            
            @views begin
                colG = G[:, best_flat]
                @inbounds @simd for t in 1:H
                    rC[t] -= αbest * colG[t]
                end
            end
            
            
            
            
            
            
            
            available[ℓbest] = false

            
            
            
        end
                end
            finally
                put!(rC_pool, rC)
            end
        end
    end

    return B, A
end





function quantize_with_kmeans_nonormal_extend_allcodes_fast!(
    X::Matrix{T},                       
    C::Vector{Matrix{T}},               
    C_norms_inv_contiguous::Array{T, 2};
    realign_every::Int = 0              
) where {T<:AbstractFloat}

    d, n = size(X)
    m = length(C)
    h_vec = [size(C[ℓ],2) for ℓ in 1:m]
    H = sum(h_vec)

    
    pre = build_q_precomp(C, C_norms_inv_contiguous)

    C_all         = pre.C_all
    G             = pre.G
    offsets       = pre.offsets
    flat_layer    = pre.flat_layer
    invnorm_flat  = pre.invnorm_flat

    
    B  = Matrix{UInt32}(undef, m, n)    
    A  = Matrix{T}(undef, m, n)         
    

    nthreads = Threads.nthreads()
    pool_size = max(2, nthreads * 2)
    rC_pool = Channel{Vector{T}}(pool_size)
    for _ in 1:pool_size
        put!(rC_pool, zeros(T, H))
    end
    chunk_size = max(1, cld(n, nthreads))
    Threads.@sync for start in 1:chunk_size:n
        stop = min(start + chunk_size - 1, n)
        Threads.@spawn begin
            rC = take!(rC_pool)
            try
                for j in start:stop
        
        
        v_xj = @view X[:, j]
        dot_val =zero(T)
        @inbounds for k in 1:H
            dot_val =zero(T)
            v_ck = @view C_all[:, k]
            @inbounds @simd for i in 1:d
                dot_val += v_ck[i] * v_xj[i]
            end
            rC[k] = dot_val
        end
        
        available = trues(m)

        
        jstart = offsets[1]
        jend   = jstart + h_vec[1] - 1

        best_flat::Int = jstart
        best_adj::T = -Inf

        @inbounds for jf in jstart:jend
            adj  = rC[jf] * invnorm_flat[jf]   

            if adj > best_adj
                best_adj     = adj
                best_flat    = jf
            end
        end

        ℓ1   = 1
        k1   = best_flat - jstart + 1
        invn = invnorm_flat[best_flat]
        α1   = best_adj * invn                    

        
        B[ℓ1, j] = UInt32(k1)
        A[ℓ1, j] = α1

        
        @views begin
            colG = G[:, best_flat]
            @inbounds @simd for t in 1:H
                rC[t] -= α1 * colG[t]
            end
        end
        
        
        
        
        
        
        
        available[ℓ1] = false

        
        
        
        

        
        for step in 2:m
            best_flat  = 0
            best_adj   = -Inf

            
            @inbounds for jf in 1:H
                ℓ = flat_layer[jf]
                if !available[ℓ]; continue; end
                adj  = rC[jf] * invnorm_flat[jf]
                if adj > best_adj
                    best_adj     = adj
                    best_flat    = jf
                end
            end

            ℓbest = flat_layer[best_flat]
            kbest = best_flat - offsets[ℓbest] + 1
            αbest = best_adj * invnorm_flat[best_flat]  

            
            B[ℓbest, j] = UInt32(kbest)
            A[ℓbest, j] = αbest

            
            @views begin
                colG = G[:, best_flat]
                @inbounds @simd for t in 1:H
                    rC[t] -= αbest * colG[t]
                end
            end
            
            
            
            
            
            
            
            available[ℓbest] = false

            
            
            
        end
                end
            finally
                put!(rC_pool, rC)
            end
        end
    end

    return B, A
end





function quantize_with_kmeans_abs_nonormal_extend_allcodes_fast_fix!(
    X::AbstractMatrix{T},                 
    C::Vector{Matrix{T}},
    pre,
    first_root_codes::AbstractVector{<:Integer}
) where {T<:AbstractFloat}

    d, n = size(X)
    m = length(C)
    h_vec = [size(C[ℓ],2) for ℓ in 1:m]
    H = sum(h_vec)

    C_all         = pre.C_all
    G             = pre.G
    offsets       = pre.offsets
    flat_layer    = pre.flat_layer
    invnorm_flat  = pre.invnorm_flat

    
    B  = Matrix{UInt32}(undef, m, n)
    A  = Matrix{T}(undef, m, n)

    nthreads = Threads.nthreads()
    pool_size = max(2, nthreads * 2)
    rC_pool = Channel{Vector{T}}(pool_size)
    for _ in 1:pool_size
        put!(rC_pool, zeros(T, H))
    end
    chunk_size = max(1, cld(n, nthreads))
    Threads.@sync for start in 1:chunk_size:n
        stop = min(start + chunk_size - 1, n)
        Threads.@spawn begin
            rC = take!(rC_pool)
            try
                for j in start:stop

        
        v_xj = @view X[:, j]
        dot_val = zero(T)
        @inbounds for k in 1:H
            dot_val = zero(T)
            v_ck = @view C_all[:, k]
            @inbounds @simd for i in 1:d
                dot_val += v_ck[i] * v_xj[i]
            end
            rC[k] = dot_val
        end

        available = trues(m)

        
        best_flat = first_root_codes[j]
        α1   = rC[best_flat] * invnorm_flat[best_flat] * invnorm_flat[best_flat]                    

        B[1, j] = UInt32(best_flat)
        A[1, j] = α1

        
        @views begin
            colG = G[:, best_flat]
            @inbounds @simd for t in 1:H
                rC[t] -= α1 * colG[t]
            end
        end
        available[1] = false

        
        for step in 2:m
            best_flat  = 0
            best_abs_adj = -typemax(T)
            best_adj   = zero(T)

            @inbounds for jf in 1:H
                ℓ = flat_layer[jf]
                if !available[ℓ]; continue; end
                adj  = rC[jf] * invnorm_flat[jf]
                aabs = abs(adj)
                if aabs > best_abs_adj
                    best_abs_adj = aabs
                    best_adj     = adj
                    best_flat    = jf
                end
            end

            ℓbest = flat_layer[best_flat]
            kbest = best_flat - offsets[ℓbest] + 1
            αbest = best_adj * invnorm_flat[best_flat]

            B[ℓbest, j] = UInt32(kbest)
            A[ℓbest, j] = αbest

            @views begin
                colG = G[:, best_flat]
                @inbounds @simd for t in 1:H
                    rC[t] -= αbest * colG[t]
                end
            end
            available[ℓbest] = false
        end
                end
            finally
                put!(rC_pool, rC)
            end
        end
    end

    return B, A
end
