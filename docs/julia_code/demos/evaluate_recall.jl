using anonymousmethod

function eval_recall_base(
    queries::Matrix{T},
    C::Vector{Matrix{T}},
    B::Matrix{T2},
    len_rate_each::Matrix{T},
    ground_truth::Vector{UInt32},
    k::Int,
    h_norms::Int=256; 
    verbose::Bool = true
) where {T <: AbstractFloat,T2 <:Integer}

    d = size(C[1],1)

    M, n_base = size(B)

    verbose && println("Quantizing norms...")
    assignments, centers = quantize_norms_base_cpp(B, len_rate_each, C, h_norms, max_iter=60)
  
    dbnorms = Vector{Float32}(undef, n_base)
    for i in 1:n_base
        dbnorms[i] = centers[assignments[i]]
    end

    nq = size(queries,2)
    
    
    codebook_sizes =  [size(C[i],2) for i in 1:M]
    h_max = maximum(codebook_sizes)
    codebook_sizes_c = Vector{Cint}(codebook_sizes)
    
    total_cols = sum(codebook_sizes_c)

    codebooks_flat = Matrix{Cfloat}(undef, d, total_cols)
    col_start = 1
    for i in 1:M
        cols = codebook_sizes_c[i]
        codebooks_flat[:, col_start:col_start+cols-1] = C[i]
        col_start += cols
    end
    
    
    codebook_offsets = Vector{Cint}(undef, M)
    offset = 0
    for i in 1:M
        codebook_offsets[i] = offset
        offset += codebook_sizes_c[i]
    end
    
    dists = zeros(Cfloat, k, nq)
    knn_indices   = zeros(Cint, k, nq)
    

    if T2 == UInt8 || h_max < 257
        B_converted = convert(Matrix{UInt8}, B.-1)
        ccall(("linscan_hq_with_coeff_u8_ex", anonymousmethod.linscan_hq), Nothing,
        (Ptr{Cfloat}, Ptr{Cint},
         Ptr{UInt8}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
         Ptr{Cint}, Ptr{Cint},  
         Cint, Cint, Cint, Cint, Cint),
        dists, knn_indices,
        B_converted, queries, codebooks_flat, dbnorms, len_rate_each,
        codebook_sizes_c, codebook_offsets,  
        Cint(nq), Cint(n_base), Cint(M), Cint(d), Cint(k))
        
    else
        B_converted = convert(Matrix{UInt16}, B.-1)
        ccall(("linscan_hq_with_coeff_u16_ex", anonymousmethod.linscan_hq), Nothing,
        (Ptr{Cfloat}, Ptr{Cint},
         Ptr{UInt16}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
         Ptr{Cint}, Ptr{Cint},  
         Cint, Cint, Cint, Cint, Cint),
        dists, knn_indices,
        B_converted, queries, codebooks_flat, dbnorms, len_rate_each,
        codebook_sizes_c, codebook_offsets,  
        Cint(nq), Cint(n_base), Cint(M), Cint(d), Cint(k))
    end

    
    verbose && println("Computing recall...")
    recall = eval_recall(ground_truth,knn_indices, k)
    
    return recall, knn_indices,dists
end



function quantize_norms_base_cpp(
    B::Matrix{T2}, 
    len_rate_each::Matrix{T}, 
    C::Vector{Matrix{T}}, 
    h_norms::Int; 
    max_iter::Int = 100
) where {T <: AbstractFloat, T2 <: Integer}
    
    M, n_base = size(B)
    d= size(C[1],1)
    
    
    codebook_sizes = [size(codebook, 2) for codebook in C]
    h_max = maximum(codebook_sizes)
    
    len_rate_each_c = Array{T}(len_rate_each)
    codebook_ptrs = [pointer(C[i]) for i in 1:M]
    codebook_sizes_c = Vector{Cint}(codebook_sizes)
    assignments = Vector{Cint}(undef, n_base)
    norm_centers = Vector{T}(undef, h_norms)
    
    if T2 == UInt8 || h_max<257
        B_c = UInt8.(B .- 1)
        GC.@preserve B_c len_rate_each_c begin
        ccall(
            ("quantize_norms_complete_mkl_uint8",  anonymousmethod.quantize_norm_base),  
            Nothing,
            (Ptr{UInt8}, Ptr{T}, Ptr{Ptr{T}}, Ptr{Cint}, 
             Cint, Cint, Cint, Cint, Cint, Ptr{Cint}, Ptr{T}),
            B_c, len_rate_each_c, codebook_ptrs, codebook_sizes_c,
            M, n_base, d, h_norms, max_iter, assignments, norm_centers
        )
        end
    else
        GC.@preserve B_c len_rate_each_c begin
        B_c = UInt16.(B .- 1)
        ccall(
            ("quantize_norms_complete_mkl_uint16",  anonymousmethod.quantize_norm_base),  
            Nothing,
            (Ptr{UInt16}, Ptr{T}, Ptr{Ptr{T}}, Ptr{Cint}, 
             Cint, Cint, Cint, Cint, Cint, Ptr{Cint}, Ptr{T}),
            B_c, len_rate_each_c, codebook_ptrs, codebook_sizes_c,
            M, n_base, d, h_norms, max_iter, assignments, norm_centers
        )
        end
    end

    return assignments .+ 1, norm_centers
end



function eval_recall_linked_two_codebook(
    queries::Matrix{T},
    C::Vector{Matrix{T}}, 
    B::Matrix{T2},
    len_rate_each::Matrix{T},
    parent::Vector{UInt32}, 
    max_depth::Int,         
    ground_truth::Vector{UInt32},
    k::Int,
    h_norms::Int=256 
) where {T <: AbstractFloat,T2 <:Integer}

    M, n_base = size(B)

    println("Quantizing norms...")
    assignments, centers = quantize_norms_linked_cpp(
        B, len_rate_each, C, parent, max_depth, h_norms=h_norms, max_iter=60
    )
  
    dbnorms = Vector{Float32}(undef, n_base)
    for i in 1:n_base
        dbnorms[i] = centers[assignments[i]]
    end

    println("predicting knn...")
    @time dists, knn_indices = knn_predict_linked(queries,B,len_rate_each,C,parent,dbnorms,k,max_depth=max_depth)
    

    
    println("Computing recall...")
    recall = eval_recall(ground_truth, knn_indices, k)
    return recall, knn_indices, dists
end

function quantize_norms_linked_cpp(
    B::Matrix{T2}, 
    len_rate_each::Matrix{T}, 
    C::Vector{Matrix{T}}, 
    parent::Vector{UInt32}, 
    max_depth::Int;          
    h_norms::Int=256, 
    max_iter::Int=100
) where {T <: AbstractFloat, T2 <: Integer}
    
    _, n_base = size(B)
    d = size(C[1],1)
    M = length(C)
    
    
    codebook_sizes = [size(codebook, 2) for codebook in C]
    h_max=maximum(codebook_sizes)
    
    len_rate_each_c = Array{T}(len_rate_each)  
    codebook_ptrs = [pointer(C[i]) for i in 1:M]  
    codebook_sizes_c = Vector{Cint}(codebook_sizes)
    assignments = Vector{Cint}(undef, n_base)
    norm_centers = Vector{T}(undef, h_norms)
    
    
    parent_ptr = pointer(parent)


    if h_max<257
        B_c = UInt8.(B .- 1)   
        GC.@preserve C parent begin
            ccall(
                ("quantize_norms_complete_mkl_uint8_twocodebook", anonymousmethod.quantize_norm_linked),  
                Nothing,
                (Ptr{UInt8}, Ptr{T}, Ptr{Ptr{T}}, Ptr{Cint}, 
                    Cint, Cint, Cint, Cint, Cint, Ptr{UInt32}, Cint, Ptr{Cint}, Ptr{T}),
                B_c, len_rate_each_c, codebook_ptrs, codebook_sizes_c,
                M, n_base, d, h_norms, max_iter, parent_ptr, max_depth, assignments, norm_centers
            )
        end
    else
        B_c = UInt16.(B .- 1)   
        GC.@preserve C parent begin
            ccall(
                ("quantize_norms_complete_mkl_uint16_twocodebook", anonymousmethod.quantize_norm_linked),  
                Nothing,
                (Ptr{UInt16}, Ptr{T}, Ptr{Ptr{T}}, Ptr{Cint}, 
                    Cint, Cint, Cint, Cint, Cint, Ptr{UInt32}, Cint, Ptr{Cint}, Ptr{T}),
                B_c, len_rate_each_c, codebook_ptrs, codebook_sizes_c,
                M, n_base, d, h_norms, max_iter, parent_ptr, max_depth, assignments, norm_centers
            )
        end
    end
    return assignments .+ 1, norm_centers
end


function knn_predict_linked(
    queries::Matrix{T},
    B::Matrix{T2},
    len_rate_each::Matrix{T},
    C::Vector{Matrix{T}}, 
    parent::Vector{UInt32},
    dbnorms::Vector{T},
    k::Int;
    max_depth::Int=2
) where {T<:AbstractFloat, T2<:Integer}
    d, n_queries = size(queries)
    _, n_base = size(B)
    m = length(C)
      
    codebook_sizes = [size(codebook, 2) for codebook in C]
    h_max=maximum(codebook_sizes)
    codebook_sizes_c = Vector{Cint}(codebook_sizes)

    
    
    queries_ptr = pointer(queries)
    dbnorms_ptr = pointer(dbnorms)
    parent_ptr = pointer(parent)
    
    
    total_cb_size = sum(codebook_sizes_c)
    codebooks = Matrix{T}(undef, d, total_cb_size)
    col_start = 1
    for cb_idx in 1:m
        cb_size = codebook_sizes[cb_idx]
        cb_end = col_start + cb_size - 1
        codebooks[:, col_start:cb_end] = C[cb_idx]
        col_start = cb_end + 1
    end
    codebooks_ptr = pointer(codebooks)
    
    
    codebook_offsets = Vector{Int32}(undef, m)
    offset = 0
    for i in 1:m
        codebook_offsets[i] = offset
        offset += codebook_sizes[i]
    end
    
    
    dists = Vector{T}(undef, n_queries * k)
    indices = Vector{Int32}(undef, n_queries * k)
    if h_max<257
        B_converted = convert(Matrix{UInt8}, B .- 1)
        GC.@preserve  C parent begin
            ccall(
                (:linscan_linked_hq_u8_ex_twocodebook, anonymousmethod.linscan_linked),
                Cvoid,
                (Ptr{T}, Ptr{Int32}, 
                Ptr{UInt8}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{T}, 
                Ptr{UInt32}, Ptr{Int32}, Ptr{Int32}, 
                Cint, Cint, Cint, Cint, Cint, Cint),
                dists, indices,
                B_converted, queries_ptr, codebooks_ptr, dbnorms_ptr, len_rate_each,
                parent_ptr, codebook_sizes_c, codebook_offsets, 
                n_queries, n_base, m, d, k, max_depth
            )
        end
    else
        B_converted = convert(Matrix{UInt16}, B .- 1)
        ccall(
            (:linscan_linked_hq_u16_ex_twocodebook, anonymousmethod.linscan_linked),
            Cvoid,
            (Ptr{T}, Ptr{Int32}, 
             Ptr{UInt16}, Ptr{T}, Ptr{T}, Ptr{T}, Ptr{T}, 
             Ptr{UInt32}, Ptr{Int32}, Ptr{Int32}, 
             Cint, Cint, Cint, Cint, Cint, Cint),
            dists, indices,
            B_converted, queries_ptr, codebooks_ptr, dbnorms_ptr, len_rate_each,
            parent_ptr, codebook_sizes_c, codebook_offsets, 
            n_queries, n_base, m, d, k, max_depth
        )
    end
    
    
    dists = reshape(dists, k, n_queries)
    indices = reshape(indices, k, n_queries)

    return dists, indices
end




function eval_recall_linked_virtual(
    queries::Matrix{T},
    C::Vector{Matrix{T}}, 
    B::Matrix{T2},
    len_rate_each::Matrix{T},
    parent::Vector{UInt32}, 
    max_depth::Int,         
    ground_truth::Vector{UInt32},
    k::Int,
    n_base_real::Int,
    h_norms::Int=256, 
) where {T <: AbstractFloat,T2 <:Integer}

    M, n_base = size(B)

    println("Quantizing norms...")
    
    assignments, centers = quantize_norms_linked_cpp(
        B, len_rate_each, C, parent, max_depth, h_norms=h_norms, max_iter=60
    )
  
    
    dbnorms = Vector{Float32}(undef, n_base_real)
    for i in 1:n_base_real
        dbnorms[i] = centers[assignments[i]]
    end

    
    println("predicting knn...")
    @time dists, knn_indices = knn_predict_linked_virtual(queries,B,len_rate_each,C,parent,dbnorms,k,n_base_real)
    

    
    println("Computing recall...")
    recall = eval_recall(ground_truth, knn_indices, k)
    return recall, knn_indices, dists
end



function knn_predict_linked_virtual(
    queries::Matrix{T},
    B::Matrix{T2},
    len_rate_each::Matrix{T},
    C::Vector{Matrix{T}}, 
    parent::Vector{UInt32},
    dbnorms::Vector{T},
    k::Int,
    n_base_real::Int
) where {T<:AbstractFloat, T2<:Integer}

    d, n_queries = size(queries)
    _, n_base    = size(B)
    m            = length(C)

    
    codebook_sizes = [size(codebook, 2) for codebook in C]
    h_max = maximum(codebook_sizes)
    codebook_sizes_c = Vector{Cint}(codebook_sizes)

    
    total_cb_size = sum(codebook_sizes_c)
    codebooks = Matrix{T}(undef, d, total_cb_size)
    col_start = 1
    for cb_idx in 1:m
        cb_size = codebook_sizes[cb_idx]
        cb_end  = col_start + cb_size - 1
        @views codebooks[:, col_start:cb_end] .= C[cb_idx]
        col_start = cb_end + 1
    end

    
    codebook_offsets = Vector{Int32}(undef, m)
    offset = 0
    for i in 1:m
        codebook_offsets[i] = offset
        offset += codebook_sizes[i]
    end

    
    dists   = Vector{T}(undef, n_queries * k)
    indices = Vector{Int32}(undef, n_queries * k)

    if h_max < 257
        B_converted = UInt8.(B .- 1)

        GC.@preserve queries codebooks dbnorms len_rate_each parent codebook_sizes_c codebook_offsets B_converted begin
            ccall(
                (:linscan_linked_hq_u8_ex_twocodebook_virtual, anonymousmethod.linscan_linked),
                Cvoid,
                (Ptr{Cfloat}, Ptr{Int32},
                 Ptr{UInt8},  Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
                 Ptr{UInt32}, Ptr{Int32},  Ptr{Int32},
                 Cint, Cint, Cint, Cint, Cint, Cint),
                dists, indices,
                B_converted, queries, codebooks, dbnorms, len_rate_each,
                parent, codebook_sizes_c, codebook_offsets,
                Cint(n_queries), Cint(n_base_real),Cint(n_base), Cint(m), Cint(d), Cint(k)
            )
        end
    else
        B_converted = UInt16.(B .- 1)

        GC.@preserve queries codebooks dbnorms len_rate_each parent codebook_sizes_c codebook_offsets B_converted begin
            ccall(
                (:linscan_linked_hq_u16_ex_twocodebook_virtual, anonymousmethod.linscan_linked),
                Cvoid,
                (Ptr{Cfloat}, Ptr{Int32},
                 Ptr{UInt16}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
                 Ptr{UInt32}, Ptr{Int32}, Ptr{Int32},
                 Cint, Cint, Cint, Cint, Cint, Cint),
                dists, indices,
                B_converted, queries, codebooks, dbnorms, len_rate_each,
                parent, codebook_sizes_c, codebook_offsets,
                Cint(n_queries), Cint(n_base_real),Cint(n_base), Cint(m), Cint(d), Cint(k)
            )
        end
    end

    
    dists   = reshape(dists,   k, n_queries)
    indices = reshape(indices, k, n_queries)

    return dists, indices
end



function eval_recall(ids_gnd::Vector{T1}, ids_predicted::Matrix{T2}, k::Integer) where {T1 <: Integer,T2 <: Integer}

  nquery = size(ids_predicted, 2)
  @assert nquery == length(ids_gnd)

  nn_ranks = zeros(nquery)

  for i = 1:nquery
    gnd_ids = ids_gnd[i]
    nn_pos = findall(ids_predicted[:,i] .== gnd_ids)

    if length(nn_pos) == 1
      nn_ranks[i] = nn_pos[1]
    else
      nn_ranks[i] = k+1
    end
  end

  nn_ranks = sort(nn_ranks)

  recall_at_i = zeros(k)

  for i = 1:k
    recall_at_i[i] = length(findall((nn_ranks .<= i) .& (nn_ranks .<= k) )) ./ nquery
  end


  for i = [1 2 3 4 5 6 7 8 9 10 20 35 50 100 150 200 500 1000 2000 5000 10000]
    if i <= k
      println("r@$(i) = $(recall_at_i[i] * 100)")
    end
  end

  return recall_at_i

end
