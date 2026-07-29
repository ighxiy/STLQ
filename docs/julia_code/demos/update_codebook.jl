using LinearAlgebra
using SparseArrays
using Base.Threads

function update_codebooks_ls_extend(X::AbstractMatrix{T},
                             assigns::AbstractMatrix{<:Integer},
                             scales::AbstractMatrix{T},
                             K_vec::Vector{Int},
                             blockdim::Int = 256) where {T<:AbstractFloat}
    D, N = size(X)                  
    M, N2 = size(assigns)
    @assert N == N2
    @assert size(scales) == (M, N)
    
    
    Ks = K_vec
    total_K = sum(Ks)
    
    
    A = build_assignment_matrix_extend(assigns, scales, Ks)
    
    
    C = [Matrix{T}(undef, D, Ks[m]) for m in 1:M]

    
    G = Matrix{T}(A' * A)

    F, used_svd = safe_cholesky(G)

    @threads for dstart in 1:blockdim:D
        dend = min(dstart + blockdim - 1, D)
        B = dend - dstart + 1

        Xblk = @view X[dstart:dend, :]                    
        
        Tblk = A' * transpose(Xblk)                       
        Wblk = F \ Tblk                                   

        col_start = 1
        for m in 1:M
            K_m = Ks[m]
            col_end = col_start + K_m - 1
            
            Wm = @view Wblk[col_start:col_end, :]         
            
            C[m][dstart:dend, :] = permutedims(Wm, (2, 1))
            
            col_start = col_end + 1
        end
    end

    return C
end


function safe_cholesky(G::Matrix{T}) where T<:AbstractFloat
    try
        F = cholesky(Symmetric(G))
        return F, false
    catch e
        @warn "Cholesky failed; fallback to SVD"
        F = svd(G)
        return F, true
    end
end

function build_assignment_matrix_extend(assigns::AbstractMatrix{<:Integer},
                                scales::AbstractMatrix{T},
                                K_vec::Vector{Int}) where T
    M, N = size(assigns)
    
    Ks = K_vec
    total_K = sum(Ks)
    
    I = Int[]
    J = Int[]
    V = T[]
    
    col_offsets = cumsum(vcat(1, Ks))[1:end-1]
    
    for j in 1:N
        for m in 1:M
            k = assigns[m, j]
            scale_val = scales[m, j]
            
            K_m = Ks[m]
            
            if !(1 <= k <= K_m)
                @warn "Codeword index out of range: m=$m, k=$k, max=$K_m"
                continue
            end
            
            col_idx = col_offsets[m] + k - 1
            
            push!(I, j)
            push!(J, col_idx)
            push!(V, scale_val)
        end
    end
    
    return sparse(I, J, V, N, total_K)
end


