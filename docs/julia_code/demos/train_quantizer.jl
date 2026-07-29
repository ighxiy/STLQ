using anonymousmethod.Beam_search
using LinearAlgebra
using Base.Threads
include("./fast_spkmeans.jl")
include("./linked_encode_helper.jl")
include("./linked_quantizer.jl")
include("./encode_abs.jl")
include("./encode_noabs.jl")
include("./update_codebook.jl")
function train_with_global_rotation_beam!(
    Xt::AbstractMatrix{Float32},
    m::Int,
    h_vec::Vector{Int},
    H_beam::Int,
    init_samples::Int,
    rng,
    max_R_iters::Int,
    ils_iters::Int,
    icm_iters::Int,
    perturb_k::Int,
    root_percentile::Float64,
    num_layers::Int,
    max_depth::Int,
    knn_k::Int,
    depth_k::Int;
    icm_round::Int = 0,
    layer_relinked_rounds::Int=0
)
    d, n = size(Xt)

    
    println("============== Baseline training with R = I (no global rotation) ==============")

    base = train_quantizer_and_linked_beam!(
        Xt,m,h_vec,H_beam,init_samples,rng,ils_iters,icm_iters,perturb_k,root_percentile,num_layers,max_depth,knn_k,depth_k,icm_round=icm_round,layer_relinked_rounds=layer_relinked_rounds)

    C_root       = base.C_root
    C_one        = base.C_one
    train_mse    = base.train_mse
    
    @printf("Baseline train MSE (linkeded, R = I): %.6f\n", train_mse)

    
    R = Matrix{Float32}(I, d, d)
    X_rot = similar(Xt)

    H = sum(size(C_root[ℓ], 2) for ℓ in 1:m)
    B             = zeros(UInt32, m, n)
    len_rate_each = zeros(Float32, m, n)
    rC_train      = zeros(Float32, H, n)
    xC_train      = zeros(Float32, H, n)
    R_in          = similar(Xt)
    train_parent  = zeros(UInt32, n)
    current_depths = zeros(Int, n)
    nthreads = Threads.nthreads()
    pool_size = max(2, nthreads * 2)
    A_pool = Channel{Matrix{Float32}}(pool_size)
    for _ in 1:pool_size
        put!(A_pool, zeros(Float32, m, m))
    end

    
    for itR in 1:max_R_iters
        println("============== Global R iteration $itR / $max_R_iters ==============")

        
        mul!(X_rot, R, Xt)   

    
        refine = linked_refine_on_rotated_data_beam!(
                X_rot,R_in,rC_train,xC_train,B,len_rate_each,train_parent,current_depths,A_pool, m, h_vec, H_beam,  C_root,   C_one,  ils_iters,  icm_iters,  perturb_k, root_percentile, num_layers, max_depth, knn_k, depth_k,icm_round=icm_round,layer_relinked_rounds=layer_relinked_rounds
            )
        C_root       = refine.C_root
        C_one        = refine.C_one
        Zhat         = refine.Zhat         
        train_mse    = refine.train_mse

        @printf("   [R-iter %d] linkeded MSE in rotated space: %.6f\n",
                itR, train_mse)

        M = Zhat * transpose(Xt)   
        U, _, Vt = svd(M)          
        R .= Float32.(U * Vt') 
    end

    return (
        R            = R,
        C_root       = C_root,
        C_one        = C_one,
        train_mse    = train_mse,
    )
end



function train_quantizer_and_linked_beam!(
    X::AbstractMatrix{Float32},
    m::Int,
    h_vec::Vector{Int},
    H_beam::Int,
    init_samples::Int,
    rng,
    ils_iters::Int,
    icm_iters::Int,
    perturb_k::Int,
    root_percentile::Float64,
    num_layers::Int,
    max_depth::Int,
    knn_k::Int,
    depth_k::Int;
    icm_round::Int = 0,
    layer_relinked_rounds::Int=0,
)
    d, n = size(X)

    println("Spherical kmeans / hierarchical VQ...")
    C_root, B, len_rate_each, _ = hierarchical_vector_quantization_spherical_extend(
        X, m, h_vec;
        init_samples       = init_samples,
        rng                = rng,
        method             = "kmeans++",
    )
    train_error, dist_error = double_errors(X, B, C_root, len_rate_each)
    @printf("Init mse error (initial): %.6f  dist error: %.6f\n",
            train_error, dist_error)

    H = sum(size(C_root[ℓ], 2) for ℓ in 1:m)
    train_mse=0
    R_in      = similar(X)
    rC_train  = zeros(Float32, H, n)
    xC_train  = zeros(Float32, H, n)

    println("##################### Training preparing...")

    C_contiguous_root, C_norms_inv_contiguous_root = build_C_contiguous(C_root)
    pre_root = build_q_precomp(C_root, C_norms_inv_contiguous_root)


    println("doing beam search...")
    h_j_vec = [size(C_root[i],2) for i in 1:m]
    h_max = maximum(h_j_vec)
    nthreads = Threads.nthreads()

    BLAS.set_num_threads(nthreads) 
    mul!(xC_train, transpose(pre_root.C_all), X)
    BLAS.set_num_threads(1)

    pl_ws2 = init_prefix_ls_beam_workspace2(Float32, UInt32, m, H_beam)
    beam_quantize_prefix_ls_incremental!(
        X,
        pre_root,
        xC_train,
        B,
        pl_ws2,
        H_beam,
    )

    pool_size = max(2, nthreads * 2)
    A_pool = Channel{Matrix{Float32}}(pool_size)
    for _ in 1:pool_size
        put!(A_pool, zeros(Float32, m, m))
    end
    optimized_least_squares_all!(
        len_rate_each, X, C_contiguous_root, B, A_pool,
        n, m, d, h_max
    )

    train_error, dist_error = double_errors(X, B, C_root, len_rate_each)
    @printf("Beam_search mse error (initial): %.6f  dist error: %.6f\n",
            train_error, dist_error)

    dynamic_icm_with_ils_noabs_nonormal!(
        X, B, len_rate_each,
        rC_train, xC_train,
        C_contiguous_root, pre_root,
        ils_iters, icm_iters, perturb_k, true,
    )
   
    C_root = update_codebooks_ls_extend(X, B, len_rate_each, h_vec)
    C_contiguous_root, C_norms_inv_contiguous_root = build_C_contiguous(C_root)
    pre_root = build_q_precomp(C_root, C_norms_inv_contiguous_root)
    train_error, dist_error = double_errors(X, B, C_root, len_rate_each)
    @printf("ICM mse error (initial): %.6f  dist error: %.6f\n",
            train_error, dist_error)
  


    current_depths = fill(0, n)
    train_parent   = zeros(UInt32, n)

 
    R_full = linked_from_inner_to_outer_one_codebook!(
        X, B, len_rate_each, C_root,
        train_parent, current_depths,
        C_contiguous_root, pre_root,
        max_depth, root_percentile, num_layers,
        knn_k, depth_k, icm_round;
        verbose = false,
        layer_relinked_rounds = layer_relinked_rounds,
    )

    @threads for i in 1:n
        v_X = @view X[:, i]
        if train_parent[i] > 0
            v_R = @view R_full[:, train_parent[i]]
            R_in[:, i] = v_X - v_R
        else
            R_in[:, i] = v_X
        end
    end

    idx_one  = findall(current_depths .> 0)
    C_one  = update_codebooks_ls_extend(
        R_in[:, idx_one],
        B[:, idx_one],
        len_rate_each[:, idx_one],
        h_vec,
    )

    train_mse,_,Zhat = analyze_linkeded_reconstruction_itself_twobook1(
        X, B, len_rate_each, C_root, C_one, train_parent
    )

    
    return (
        C_root       = C_root,
        C_one        = C_one,
        B            = B,
        len_rate_each = len_rate_each,
        train_parent = train_parent,
        Zhat         = Zhat,
        train_mse    = train_mse,
    )
end




function linked_refine_on_rotated_data_beam!(
    X::AbstractMatrix{T},
    R_in::AbstractMatrix{T},
    rC_train::AbstractMatrix{T},
    xC_train::AbstractMatrix{T},
    B::AbstractMatrix{BType},
    len_rate_each::AbstractMatrix{T},
    train_parent::Vector{UInt32},
    current_depths::Vector{Int},
    A_pool::Channel{Matrix{T}},
    m::Int,
    h_vec::Vector{Int},
    H_beam::Int,
    C_root,
    C_one,
    ils_iters::Int,
    icm_iters::Int,
    perturb_k::Int,
    root_percentile::Float64,
    num_layers::Int,
    max_depth::Int,
    knn_k::Int,
    depth_k::Int;
    icm_round::Int = 0,
    layer_relinked_rounds::Int=0,
) where{T<:AbstractFloat, BType<:Integer}
    d, n = size(X)

    train_mse   = 0

    nthreads = Threads.nthreads()
    h_j_vec = [size(C_root[i],2) for i in 1:m]
    h_max = maximum(h_j_vec)
    
    println("##################### linked refine iteration on rotated data...")

    C_contiguous_root, C_norms_inv_contiguous_root = build_C_contiguous(C_root)
    pre_root = build_q_precomp(C_root, C_norms_inv_contiguous_root)

    println("doing beam search...")

    BLAS.set_num_threads(nthreads) 
    mul!(xC_train, transpose(pre_root.C_all), X)
    BLAS.set_num_threads(1)


    pl_ws2 = init_prefix_ls_beam_workspace2(Float32, UInt32, m, H_beam)
    beam_quantize_prefix_ls_incremental!(
        X,
        pre_root,
        xC_train,
        B,
        pl_ws2,
        H_beam,
    )

    optimized_least_squares_all!(
        len_rate_each, X, C_contiguous_root, B, A_pool,
        n, m, d, h_max
    )

    train_error, dist_error = double_errors(X, B, C_root, len_rate_each)
    @printf("Beam_search mse error (initial): %.6f  dist error: %.6f\n",
            train_error, dist_error)

    dynamic_icm_with_ils_noabs_nonormal!(
        X, B, len_rate_each,
        rC_train, xC_train,
        C_contiguous_root, pre_root,
        ils_iters, icm_iters, perturb_k, true,
    )
    
    
    C_root = update_codebooks_ls_extend(X, B, len_rate_each, h_vec)
    C_contiguous_root, C_norms_inv_contiguous_root = build_C_contiguous(C_root)
    pre_root = build_q_precomp(C_root, C_norms_inv_contiguous_root)

    train_error, dist_error = double_errors(X, B, C_root, len_rate_each)
    @printf("Trainset mse error before linked: %.6f  dist error: %.6f\n"
           , train_error, dist_error)


    fill!(current_depths, 0)
    fill!(train_parent, 0)

    C_contiguous_one, C_norms_inv_contiguous_one = build_C_contiguous(C_one)
    pre_one = build_q_precomp(C_one, C_norms_inv_contiguous_one)

    R_full = linked_from_inner_to_outer_two_codebook!(
        X, B, len_rate_each,
        train_parent, current_depths,
        C_root, pre_root,
        C_one,  C_contiguous_one,  pre_one,
        max_depth, root_percentile, num_layers,
        knn_k, depth_k, icm_round;
        verbose = false,
        layer_relinked_rounds = layer_relinked_rounds,
    )


    @threads for i in 1:n
        v_X = @view X[:, i]
        if train_parent[i] > 0
            v_R = @view R_full[:, train_parent[i]]
            R_in[:, i] = v_X - v_R
        else
            R_in[:, i] = v_X
        end
    end

    println("update C_one codebook...")
    idx_one  = findall(current_depths .> 0)
    C_one  = update_codebooks_ls_extend(
        R_in[:, idx_one],
        B[:, idx_one],
        len_rate_each[:, idx_one],
        h_vec,
    )


    train_mse,_,Zhat = analyze_linkeded_reconstruction_itself_twobook1(
        X, B, len_rate_each, C_root, C_one, train_parent
    )


    return (
        C_root       = C_root,
        C_one        = C_one,
        B            = B,
        len_rate_each = len_rate_each,
        Zhat         = Zhat,
        train_mse    = train_mse,
    )
end






function hierarchical_vector_quantization_spherical_extend(
    Xt::Matrix{T},
    m::Int,
    h_vec::Vector{Int};
    init_samples = 2000,
    rng::Union{AbstractRNG, Int},
    method="random"
) where T <: AbstractFloat
    d, n = size(Xt)
    
    residuals = copy(Xt)
    C = Vector{Matrix{Float32}}(undef, m)
    B = zeros(UInt32, m,n)
    len_rate_each = Matrix{T}(undef, m, n)

    nthreads = Threads.nthreads()
    pool_size = max(2, nthreads * 2)
    center_pool = Channel{Vector{T}}(pool_size)
    for _ in 1:pool_size
        put!(center_pool, Vector{T}(undef, d))
    end
    
    for layer in 1:m

        h_dynamic = h_vec[layer]

        cluster, final_weights = spherical_kmeans_fast_with_adaptive_weights(
            residuals, h_dynamic,maxiter=70,init=method,batch_size=n,
            rng=rng,
            init_samples=init_samples,
            outlier_quantile=0.85,
            cost_threshold=0.115,
            annealing_factor=0.9,
            warmup_iters=2,
            display=false
        )

        centers = cluster.centers
        assignments = cluster.assignments

        C[layer] = centers

        chunk_size = max(1, cld(h_dynamic, nthreads))
        Threads.@sync for start in 1:chunk_size:h_dynamic
            stop = min(start + chunk_size - 1, h_dynamic)
            Threads.@spawn begin
                center_buf = take!(center_pool)
                try
                    for j in start:stop
            cluster_points_idx = findall(assignments .== j)
            isempty(cluster_points_idx) && continue        
            
            center = view(centers, :, j)
            
            @inbounds @simd for i in 1:d
                center_buf[i] = center[i]
            end
            
            @inbounds for k_idx in eachindex(cluster_points_idx)
                k = cluster_points_idx[k_idx]
                
                proj_len = zero(T)
                @simd for i in 1:d
                    proj_len += center_buf[i] * residuals[i, k]
                end
                
                @simd for i in 1:d
                    residuals[i, k] -= proj_len * center_buf[i]
                end
                
                B[layer,k] = j
                len_rate_each[layer,k] = proj_len
            end
                    end
                finally
                    put!(center_pool, center_buf)
                end
            end
        end
    end
    
    return (C, B, len_rate_each, residuals)
end






function train_with_global_rotation_virtual_beam!(
    Xt::AbstractMatrix{Float32},
    m::Int,
    h_vec::Vector{Int},
    H_beam,
    init_samples::Int,
    rng,
    max_R_iters::Int,
    ils_iters::Int,
    icm_iters::Int,
    perturb_k::Int,
    root_percentile::Float64,
    num_layers::Int,
    max_depth::Int,
    knn_k::Int,
    depth_k::Int,
    icm_round::Int,
    layer_relinked_rounds::Int,
    virtual_ratio::Float64,
    good_fraction::Float64,
    min_virtual::Int,
    max_virtual::Int,
    alpha_bad::Float32
)
    d, n = size(Xt)

    println("============== Baseline training with R = I (no global rotation) ==============")

    base = train_quantizer_and_linked_virtual_beam!(Xt,m,h_vec,H_beam,init_samples,rng,ils_iters,icm_iters,perturb_k,root_percentile,num_layers,max_depth,knn_k,depth_k,icm_round,layer_relinked_rounds,good_fraction,alpha_bad)

    C_root       = base.C_root
    C_one        = base.C_one
    train_mse    = base.train_mse
    is_bad_cluster = base.is_bad_cluster
    @printf("Baseline train MSE (linkeded, R = I): %.6f\n", train_mse)


    R = Matrix{Float32}(I, d, d)
    X_rot = similar(Xt)

    H = sum(size(C_root[ℓ], 2) for ℓ in 1:m)
    B             = zeros(UInt32, m, n)
    len_rate_each = zeros(Float32, m, n)
    rC_train      = zeros(Float32, H, n)
    xC_train      = zeros(Float32, H, n)
    R_in          = similar(Xt)
    R_full        = similar(Xt)
    current_depths = zeros(Int, n)
    nthreads = Threads.nthreads()
    pool_size = max(2, nthreads * 2)
    A_pool = Channel{Matrix{Float32}}(pool_size)
    for _ in 1:pool_size
        put!(A_pool, zeros(Float32, m, m))
    end

    for itR in 1:max_R_iters
        println("============== Global R iteration $itR / $max_R_iters ==============")

        mul!(X_rot, R, Xt)

        refine = linked_refine_on_rotated_data_virtual_beam!(
            X_rot,R_in,R_full,rC_train,xC_train,B,len_rate_each,current_depths,A_pool,m,h_vec, H_beam, C_root, C_one, ils_iters,icm_iters, perturb_k, root_percentile, num_layers,  max_depth,  knn_k, depth_k, is_bad_cluster, virtual_ratio, min_virtual, max_virtual,icm_round,layer_relinked_rounds,alpha_bad
        )

        C_root       = refine.C_root
        C_one        = refine.C_one
        Zhat         = refine.Zhat
        train_mse    = refine.train_mse

        @printf("   [R-iter %d] linkeded MSE in rotated space: %.6f\n",
                itR, train_mse)

        M = Zhat * transpose(Xt)
        U, _, Vt = svd(M)
        R .= Float32.(U * Vt') 
    end

    return (
        R            = R,
        C_root       = C_root,
        C_one        = C_one,
        train_mse    = train_mse,
        is_bad_cluster = is_bad_cluster
    )
end

function train_quantizer_and_linked_virtual_beam!(
    X::AbstractMatrix{Float32},
    m::Int,
    h_vec::Vector{Int},
    H_beam::Int,
    init_samples::Int,
    rng,
    ils_iters::Int,
    icm_iters::Int,
    perturb_k::Int,
    root_percentile::Float64,
    num_layers::Int,
    max_depth::Int,
    knn_k::Int,
    depth_k::Int,
    icm_round::Int,
    layer_relinked_rounds::Int,
    good_fraction::Float64,
    alpha_bad::Float32
)
    d, n = size(X)

    
    println("Spherical kmeans / hierarchical VQ...")
    C_root, B, len_rate_each, _ = hierarchical_vector_quantization_spherical_extend(
        X, m, h_vec;
        init_samples       = init_samples,
        rng                = rng,
        method             = "kmeans++",
    )
    train_error, dist_error = double_errors(X, B, C_root, len_rate_each)
    @printf("Init mse error (initial): %.6f  dist error: %.6f\n",
            train_error, dist_error)

    H = sum(size(C_root[ℓ], 2) for ℓ in 1:m)
    train_mse=0
    R_in      = similar(X)
    rC_train  = zeros(Float32, H, n)
    xC_train  = zeros(Float32, H, n)

    println("##################### Training preparing...")

    C_contiguous_root, C_norms_inv_contiguous_root = build_C_contiguous(C_root)
    pre_root = build_q_precomp(C_root, C_norms_inv_contiguous_root)


    println("doing beam search...")
    h_j_vec = [size(C_root[i],2) for i in 1:m]
    h_max = maximum(h_j_vec)
    nthreads = Threads.nthreads()

    BLAS.set_num_threads(nthreads) 
    mul!(xC_train, transpose(pre_root.C_all), X)
    BLAS.set_num_threads(1)

    pl_ws2 = init_prefix_ls_beam_workspace2(Float32, UInt32, m, H_beam)
    beam_quantize_prefix_ls_incremental!(
        X,
        pre_root,
        xC_train,
        B,
        pl_ws2,
        H_beam,
    )

    pool_size = max(2, nthreads * 2)
    A_pool = Channel{Matrix{Float32}}(pool_size)
    for _ in 1:pool_size
        put!(A_pool, zeros(Float32, m, m))
    end
    optimized_least_squares_all!(
        len_rate_each, X, C_contiguous_root, B, A_pool,
        n, m, d, h_max
    )

    train_error, dist_error = double_errors(X, B, C_root, len_rate_each)
    @printf("Beam_search mse error (initial): %.6f  dist error: %.6f\n",
            train_error, dist_error)

    

    dynamic_icm_with_ils_noabs_nonormal!(
        X, B, len_rate_each,
        rC_train, xC_train,
        C_contiguous_root, pre_root,
        ils_iters, icm_iters, perturb_k, true,
    )
   
    C_root = update_codebooks_ls_extend(X, B, len_rate_each, h_vec)
    C_contiguous_root, C_norms_inv_contiguous_root = build_C_contiguous(C_root)
    pre_root = build_q_precomp(C_root, C_norms_inv_contiguous_root)
    train_error, dist_error = double_errors(X, B, C_root, len_rate_each)
    @printf("ICM mse error (initial): %.6f  dist error: %.6f\n",
            train_error, dist_error)

    cluster_id = Int.(vec(B[1, :]))
  
    
    current_depths = fill(0, n)
    train_parent   = zeros(UInt32, n)

    R_full = linked_from_inner_to_outer_one_codebook!(
        X, B, len_rate_each, C_root,
        train_parent, current_depths,
        C_contiguous_root, pre_root,
        max_depth, root_percentile, num_layers,
        knn_k, depth_k, icm_round;
        verbose = false,
        layer_relinked_rounds = layer_relinked_rounds,
    )

    
    @threads for i in 1:n
        v_X = @view X[:, i]
        if train_parent[i] > 0
            v_R = @view R_full[:, train_parent[i]]
            R_in[:, i] = v_X - v_R
        else
            R_in[:, i] = v_X
        end
    end
    
    errors_linked = zeros(Float32, n)
    @threads for i in 1:n
        @inbounds @simd for j in 1:d
            diff_linked =  X[j, i] - R_full[j, i]
            errors_linked[i] += diff_linked * diff_linked
        end
    end
    
    is_bad_cluster, _, _ = build_cluster_linked_mask(cluster_id, errors_linked,h_vec[1],mode=:quantile,good_fraction=good_fraction)
    println("num bad clusters = ", count(is_bad_cluster))

    idx_one  = findall(current_depths .> 0)
    w_one = build_residual_weights(idx_one, cluster_id, is_bad_cluster;
                                alpha_bad = alpha_bad)
    C_one = update_codebooks_ls_extend_weighted(R_in[:, idx_one], B[:, idx_one], len_rate_each[:, idx_one], h_vec,w_one)

    train_mse,_ = analyze_linkeded_reconstruction_itself_twobook(
        X, B, len_rate_each, C_root, C_one, train_parent
    )
    
    return (
        C_root       = C_root,
        C_one        = C_one,
        B            = B,
        len_rate_each = len_rate_each,
        train_parent = train_parent,
        is_bad_cluster = is_bad_cluster,
        train_mse    = train_mse,
    )
end


function linked_refine_on_rotated_data_virtual_beam!(
    X::AbstractMatrix{T},
    R_in::AbstractMatrix{T},
    R_full::AbstractMatrix{T},
    rC_train::AbstractMatrix{T},
    xC_train::AbstractMatrix{T},
    B::AbstractMatrix{BType},
    len_rate_each::AbstractMatrix{T},
    current_depths::Vector{Int},
    A_pool::Channel{Matrix{T}},
    m::Int,
    h_vec::Vector{Int},
    H_beam::Int,
    C_root,   
    C_one,    
    ils_iters::Int,
    icm_iters::Int,
    perturb_k::Int,
    root_percentile::Float64,
    num_layers::Int,
    max_depth::Int,
    knn_k::Int,
    depth_k::Int,
    is_bad_cluster::AbstractVector{Bool},
    virtual_ratio::Float64,
    min_virtual::Int,
    max_virtual::Int,
    icm_round::Int,
    layer_relinked_rounds::Int,
    alpha_bad::Float32
)where{T<:AbstractFloat, BType<:Integer}
    d, n = size(X)
    nthreads = Threads.nthreads()
    h_max = h_vec[1]

    println("##################### linked refine iteration on rotated data...")

    C_contiguous_root, C_norms_inv_contiguous_root = build_C_contiguous(C_root)
    pre_root = build_q_precomp(C_root, C_norms_inv_contiguous_root)

    println("doing beam search...")

    BLAS.set_num_threads(nthreads) 
    mul!(xC_train, transpose(pre_root.C_all), X)
    BLAS.set_num_threads(1)

    pl_ws2 = init_prefix_ls_beam_workspace2(Float32, UInt32, m, H_beam)
    beam_quantize_prefix_ls_incremental!(
        X,
        pre_root,
        xC_train,
        B,
        pl_ws2,
        H_beam,
    )

    optimized_least_squares_all!(
        len_rate_each, X, C_contiguous_root, B, A_pool,
        n, m, d, h_max
    )

    train_error, dist_error = double_errors(X, B, C_root, len_rate_each)
    @printf("Beam_search mse error (initial): %.6f  dist error: %.6f\n",
            train_error, dist_error)

    dynamic_icm_with_ils_noabs_nonormal!(
        X, B, len_rate_each,
        rC_train, xC_train,
        C_contiguous_root, pre_root,
        ils_iters, icm_iters, perturb_k, true,
    )
    
    cluster_id = Int.(vec(B[1, :]))
    
    C_root = update_codebooks_ls_extend(X, B, len_rate_each, h_vec)
    C_contiguous_root, C_norms_inv_contiguous_root = build_C_contiguous(C_root)
    pre_root = build_q_precomp(C_root, C_norms_inv_contiguous_root)

    train_error, dist_error = double_errors(X, B, C_root, len_rate_each)
    @printf("Trainset mse error (pre-linked, rotated): %.6f  dist error: %.6f\n", train_error, dist_error)

    C_contiguous_one, C_norms_inv_contiguous_one = build_C_contiguous(C_one)
    pre_one = build_q_precomp(C_one, C_norms_inv_contiguous_one)

    train_mse = linked_with_dual_strategy!(
        X, R_in, R_full, B, len_rate_each,current_depths,
        C_root, C_contiguous_root, pre_root,
        C_one,  C_contiguous_one,  pre_one,
        is_bad_cluster, max_depth, root_percentile, num_layers,
        knn_k, depth_k, icm_round,layer_relinked_rounds,
        ils_iters, icm_iters, perturb_k,
        virtual_ratio, min_virtual, max_virtual,
        false
    )

    idx_one  = findall(current_depths .> 0)
    println("update one depth codebook (rotated)...")

    w_one = build_residual_weights(idx_one, cluster_id, is_bad_cluster;
                                alpha_bad = alpha_bad)
    C_one = update_codebooks_ls_extend_weighted(R_in[:, idx_one], B[:, idx_one], len_rate_each[:, idx_one], h_vec,w_one)

    return (
        C_root       = C_root,
        C_one        = C_one,
        B            = B,
        len_rate_each = len_rate_each,
        Zhat         = R_full,
        train_mse    = train_mse,
    )
end