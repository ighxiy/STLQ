using LinearAlgebra
using Printf
using Plots
using anonymousmethod.Beam_search

include("./read_data/read_datasets.jl")
include("./train_quantizer.jl")
include("./greedy_quantizer.jl")
include("./linked_quantizer.jl")
include("./encode_abs.jl")
include("./encode_noabs.jl")
include("./update_codebook.jl")
include("./utils.jl")
include("./evaluate_recall.jl")
include("./umap_like_augment.jl")
include("./scale.jl")
include("./huffman.jl")

function main(dataset_name,ntrain::Integer,m::Integer,h_vec::Vector{Int})
      
    nquery, nbase, k = 0, 0, 0

    if dataset_name == "SIFTSMALL"
      nquery, nbase, k = Int(1e2), Int(1e4), Int(50)
    elseif dataset_name == "SIFT1M" || dataset_name == "Deep1M" || dataset_name == "Convnet1M"
      nquery, nbase, k = Int(1e4), Int(1e6), Int(1e2)
    elseif dataset_name == "GIST1M"
      nquery, nbase, k = Int(1e3), Int(1e6), Int(1e2)
    else
      error("dataset unknown")
    end


    verbose = true

    Xt, Xb, Xq, gt = load_experiment_data(dataset_name, ntrain, nbase, nquery, verbose)
    
    
    d, n = size(Xt)
    n_base=size(Xb,2)
    init_samples = 100000

    hnorms =2048
    nthreads = Threads.nthreads()

    is_train = true 
    is_encodingbase = true
    is_linked = true
    is_save = false
    is_save_base = false
    is_save_linked = false



rng = MersenneTwister(1985326)
Random.seed!(Random.GLOBAL_RNG, 38251450)

pre_fix = "umap-f"
train_file = "./results/$(lowercase(dataset_name))/$(pre_fix)vrdc_ex_m$(m).h5"
base_file = "./results/$(lowercase(dataset_name))/$(pre_fix)vrdc_ex_base_init_m$(m).h5"
linked_file = "./results/$(lowercase(dataset_name))/$(pre_fix)vrdc_linked_ex_base_init_m$(m).h5"

    H_beam = 2
    ils_iters_base = 200
    ils_iters = 60
    icm_iters = 8
    perturb_k = 3
    root_percentile=0.010e0
    num_layers=12
    max_depth = 40
    knn_k = 15
    depth_k= 11
    icm_round = 1
    layer_relinked_rounds=1
    max_R_iters=20
    virtual_ratio = 0.10e0
    good_fraction = 0.6e0
    min_virtual   = 1
    max_virtual   = 2000
    alpha_bad = 0.3f0

if is_train
    result = train_with_global_rotation_virtual_beam!(Xt, m, h_vec,H_beam, init_samples, rng,max_R_iters, ils_iters, icm_iters,perturb_k, root_percentile,num_layers,max_depth, knn_k, depth_k, icm_round, layer_relinked_rounds, virtual_ratio,good_fraction,min_virtual,max_virtual,alpha_bad)
    R   = result.R
    C_root   = result.C_root
    C_one    = result.C_one
    is_bad_cluster = result.is_bad_cluster
    Xb=R*Xb
    Xq=R*Xq

    C=[C_root; C_one]
    if is_save
        save_c_Rv(train_file,C,R,is_bad_cluster)
    end
else
    C,R,is_bad_cluster = load_c_Rv(train_file,2*m)
    C_root=C[1:m]
    C_one=C[m+1:2*m]
    Xb=R*Xb
    Xq=R*Xq
end


    
C_contiguous_root,C_norms_inv_contiguous_root=build_C_contiguous(C_root)
pre_root = build_q_precomp(C_root, C_norms_inv_contiguous_root)
C_contiguous_one,C_norms_inv_contiguous_one=build_C_contiguous(C_one)
pre_one = build_q_precomp(C_one, C_norms_inv_contiguous_one)
n_base_real=n_base

if is_encodingbase
    println("encoding...")

    B_base             = Matrix{UInt32}(undef, m, n_base)
    len_rate_each_base = Matrix{Float32}(undef, m, n_base)
    rC = zeros(Float32, pre_root.H, n_base)
    xC = zeros(Float32, pre_root.H, n_base)
    h_j_vec = [size(C_root[i],2) for i in 1:m]
    h_max = maximum(h_j_vec)


    BLAS.set_num_threads(nthreads) 
    mul!(xC, transpose(pre_root.C_all), Xb)
    BLAS.set_num_threads(1)

    pl_ws2 = init_prefix_ls_beam_workspace2(Float32, UInt32, m, H_beam)
    @time beam_quantize_prefix_ls_incremental!(
        Xb,
        pre_root,
        xC,
        B_base,
        pl_ws2,
        H_beam,
    )

    pool_size = max(2, nthreads * 2)
    A_pool = Channel{Matrix{Float32}}(pool_size)
    for _ in 1:pool_size
        put!(A_pool, zeros(Float32, m, m))
    end
    optimized_least_squares_all!(
        len_rate_each_base, Xb, C_contiguous_root, B_base, A_pool,
        n_base, m, d, h_max
    )

    new_error, dist_error = double_errors(Xb, B_base, C_root, len_rate_each_base)
    @printf("Beam_search mse error: %.6f  dist error: %.6f\n", new_error, dist_error)

    dynamic_icm_with_ils_abs_nonormal!(Xb, B_base, len_rate_each_base,rC,xC, C_contiguous_root, pre_root, ils_iters_base, icm_iters, perturb_k, true)

    new_error, dist_error = double_errors(Xb, B_base, C_root, len_rate_each_base)
    @printf("ICM mse error: %.6f  dist error: %.6f\n", new_error, dist_error)

    recall, knn_indices ,dists= eval_recall_base(Xq, C_root, B_base, len_rate_each_base, gt, k, hnorms;verbose=true)

    

    println("clustering virtual...")

    Xb, B_base, len_rate_each_base =
    add_virtual_roots_umap_reencode(
        Xb, B_base, len_rate_each_base,
        C_root, C_contiguous_root, pre_root,
        ils_iters_base, icm_iters, perturb_k,
        virtual_ratio,
        min_virtual,
        max_virtual;
        
        use_fixed_virtual_per_cluster = false,
        fixed_virtual_per_cluster     = 256,
        is_bad_cluster = is_bad_cluster,
        verbose = false,
    )


    if is_save_base
        save_base_results_hq(base_file,1,B_base,len_rate_each_base,0)
    end

else
    B_base,len_rate_each_base=load_base_hq(base_file,m,1)
    B_virtual = @view B_base[:, n_base_real+1:end]
    len_virtual = @view len_rate_each_base[:, n_base_real+1:end]
    X_virtual = compute_reconstructed(B_virtual, len_virtual, C_root)
    Xb = hcat(Xb, X_virtual)
end

    n_base = size(B_base, 2) 

plt = analyze_ori_reconstruction_itself(Xb,B_base,len_rate_each_base, C_root)
B_ivf_1 = copy(B_base[1,:])

if is_linked
    parent = zeros(UInt32, n_base)
    current_depths = fill(0, n_base)
    num_layers=16
    depth_k= 15
    max_depth = 40
    knn_k     = 20

    linked_with_dual_strategy_for_baseset!(
        Xb, B_base, len_rate_each_base,
        parent, current_depths,
        C_root, C_contiguous_root, pre_root,
        C_one,  C_contiguous_one,  pre_one,
        max_depth, root_percentile, num_layers,
        knn_k, depth_k, icm_round,
        is_bad_cluster, n_base_real;
        verbose = true,
        layer_relinked_rounds = 0,
    )

    if is_save_linked
        save_base_results_anonymousmethod(linked_file,1,B_base,len_rate_each_base,parent,0)
    end

else
    B_base,len_rate_each_base,parent=load_base_anonymousmethod(linked_file,m,1)
end


recall, knn_indices, dists = eval_recall_linked_virtual(Xq, C, B_base, len_rate_each_base, parent, 20, gt, k, n_base_real, hnorms)

analyze_linkeded_reconstruction_itself_twobook(Xb,B_base,len_rate_each_base, C_root,C_one, parent, plt=plt)










X_ref = compute_linkeded_reconstruction_multibook(
    B_base, len_rate_each_base, C_root, C_one, parent
)

bits_per_layer = fill(6, m)
bits_per_layer=[7,7,7,6,6 ]

a_int_f32, C_root_q, C_one_q, st =
    quantize_coeffs_multibook_split_weighted2!(
        B_base, len_rate_each_base, C_root, C_one, parent;
        bits_per_layer     = bits_per_layer,
        use_weighted_quantile = true,
        p_first_candidates = [99.95],
        p_rest_candidates = [99.95],
        auto_search_p = false,
        p_first_range = (99.9, 99.9, 0.03),
        p_rest_range  = (99.9, 99.9, 0.03),
        q_refine_sweeps = 4,
        λ_global = 0.0,   
        verbose = true,
        q_refine_max_layer = 0,           
        q_refine_step_limit = 1,
    )



is_linked = parent .> 0

comp, stats =
    compress_coeffs_huffman_root_linked_layered(a_int_f32, is_linked; verbose = true)
@show stats[:avg_bits_per_coeff]

comp, stats =
    compress_coeffs_huffman_root_linked_layer_cluster(
        a_int_f32, is_linked, B_ivf_1;
        verbose = true,
    )
@show stats[:avg_bits_per_coeff]



X_q = compute_linkeded_reconstruction_multibook(
    B_base, a_int_f32, C_root_q, C_one_q, parent
)


diff = X_q .- X_ref
mse = mean(sum(abs2, diff; dims=1))
base_norm = mean(sum(abs2, X_ref; dims=1))
rel_mse = mse / base_norm
@show mse rel_mse

C_new = [C_root_q; C_one_q]
recall, knn_indices, dists = eval_recall_linked_virtual(Xq, C_new, B_base, a_int_f32, parent, 20, gt, k, n_base_real, hnorms)
    return nothing
end



m=5
h_vec = [256; fill(256,m-1)]

main("SIFT1M", Int(1e5), m, h_vec)


