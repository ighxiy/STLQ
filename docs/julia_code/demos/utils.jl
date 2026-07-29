using HDF5
using Base.Threads

function double_errors(
    X::Matrix{T}, B::Matrix{BType}, C::Vector{Matrix{T}}, a::Matrix{T}
) where {T <: AbstractFloat, BType <: Integer}
    d,n=size(X)
    m=size(B,1)

    mse = zero(T)
    r_dist = zero(T)

    
    mse_errors = zeros(T, n)
    dist_errors = zeros(T, n)
    @threads for i in 1:n
        
        recon_i = zeros(T, d)
        for j in 1:m
            code_idx = B[j, i]
            Cj = C[j]
            @inbounds @simd for dim in 1:d
                recon_i[dim] += a[j, i] * Cj[dim, code_idx]
            end
        end
        
        
        error_i = zero(T)
        @inbounds @simd for dim in 1:d
            diff = X[dim, i] - recon_i[dim]
            error_i += diff * diff
        end
        mse_errors[i] = error_i
        dist_errors[i] = sqrt(error_i)
    end

    avg_mse = sum(mse_errors)  / n
    avg_r_dist = sum(dist_errors)  / n
    
    println("Max=$(round(maximum(mse_errors), digits=4)), Min=$(round(minimum(mse_errors), digits=4))")
    return avg_mse , avg_r_dist
end



function compute_reconstructed(
    B::AbstractMatrix{BType},
    len_rate_each::AbstractMatrix{T},
    C::Vector{Matrix{T}}
) where {T<:AbstractFloat, BType<:Integer}
    
    m, n = size(len_rate_each)
    d = size(C[1], 1)
    reconstructed = zeros(T, d, n)
    
    @threads for i in 1:n
        recon_i = zeros(T, d)
        for j in 1:m
            code_idx = B[j, i]
            α = len_rate_each[j, i]
            @inbounds @simd for k in 1:d
                recon_i[k] += α * C[j][k, code_idx]
            end
        end
        reconstructed[:, i] = recon_i
    end

    return reconstructed
end

function analyze_ori_reconstruction_itself(
    X::Matrix{T},          
    B_ori::Matrix{BType},   
    len_rate_each_ori::Matrix{T}, 
    C::Vector{Matrix{T}},  
) where {T<:AbstractFloat, BType<:Integer}
    
    d, n = size(X)

    reconstructed_original = compute_reconstructed(B_ori, len_rate_each_ori, C)

    residuals_original = X - reconstructed_original
    residual_norms_original = [norm(residuals_original[:, i]) for i in 1:n]

    
    
    errors_original = zeros(T, n)
    @threads for i in 1:n
        @inbounds @simd for j in 1:d
            diff_original = X[j, i]  - reconstructed_original[j, i]
            errors_original[i]  += diff_original * diff_original
        end
    end

    println(" ori Residual Norm: Max=$(round(maximum(residual_norms_original), digits=4)), Min=$(round(minimum(residual_norms_original), digits=4)), Mean=$(round(mean(residual_norms_original), digits=4))")
    println("  ori Error: Max=$(round(maximum(errors_original), digits=4)), Min=$(round(minimum(errors_original), digits=4)), Mean=$(round(mean(errors_original), digits=4))")
    
    
    plt = Plots.plot(layout=(1, 2), size=(1000,500), title="linkeded Reconstruction Analysis")
    
    
    histogram!(plt, residual_norms_original, bins=50, 
              label="Original", color=:blue, alpha=0.5,
              title="Residual Norm Comparison", xlabel="Residual Norm", ylabel="Count",
              subplot=1)
    
    
    histogram!(plt, errors_original, bins=50, 
              label="Original", color=:blue, alpha=0.5,
              title="Reconstruction Error Comparison", xlabel="Reconstruction Error", ylabel="Count",
              subplot=2)

    return plt
end

function analyze_linkeded_reconstruction_itself_twobook1(
    X::Matrix{T},          
    B_linked::Matrix{BType},   
    len_rate_each::Matrix{T}, 
    C_root::Vector{Matrix{T}},  
    C_one::Vector{Matrix{T}},
    parent::Vector{UInt32} 
) where {T<:AbstractFloat, BType<:Integer}
    
    d, n = size(X)
    m = size(B_linked, 1)
    
    
    depths = compute_depths!(parent)
    min_depth = minimum(depths)
    max_depth = maximum(depths)
    mean_depth = mean(depths)

    n = length(depths)
    
    
    println("Max depth: $max_depth")
    println("Mean depth: $(round(mean_depth, digits=2))")

    
    reconstructed_linked = compute_linkeded_reconstruction_multibook(B_linked, len_rate_each, C_root,C_one, parent)

    residuals_linked = X - reconstructed_linked
    residual_norms_linked = [norm(residuals_linked[:, i]) for i in 1:n]
    
    
    errors_linked = zeros(T, n)
    @threads for i in 1:n
        @inbounds @simd for j in 1:d
            diff_linked =  X[j, i] - reconstructed_linked[j, i]
            errors_linked[i] += diff_linked * diff_linked
        end
    end
    mean_error=mean(errors_linked)
    
    
    linked_counts = count(parent .> 0)
    linked_ratio = linked_counts / n
    
    
    
    println("linkeded Reconstruction Analysis Summary:")
    println("  linked Ratio: $(round(100*linked_ratio, digits=2))%")
    println("  linked Residual Norm: Max=$(round(maximum(residual_norms_linked), digits=4)), Min=$(round(minimum(residual_norms_linked), digits=4)), Mean=$(round(mean(residual_norms_linked), digits=4))")
    println("  linked Error: Max=$(round(maximum(errors_linked), digits=4)), Min=$(round(minimum(errors_linked), digits=4)), Mean=$(round(mean_error ,digits=4))")
    
    return mean_error,errors_linked,reconstructed_linked
end

function analyze_linkeded_reconstruction_itself_twobook(
    X::Matrix{T},          
    B_linked::Matrix{BType},   
    len_rate_each::Matrix{T}, 
    C_root::Vector{Matrix{T}},  
    C_one::Vector{Matrix{T}},
    parent::Vector{UInt32}; 
    plt::Union{Plots.Plot, Nothing} = nothing
) where {T<:AbstractFloat, BType<:Integer}
    
    d, n = size(X)
    m = size(B_linked, 1)
    
    
    depths = compute_depths!(parent)
    min_depth = minimum(depths)
    max_depth = maximum(depths)
    mean_depth = mean(depths)
    n = length(depths)
    
    
    println("Max depth: $max_depth")
    println("Mean depth: $(round(mean_depth, digits=2))")
    

    
    reconstructed_linked = compute_linkeded_reconstruction_multibook(B_linked, len_rate_each, C_root,C_one, parent)

    residuals_linked = X - reconstructed_linked
    residual_norms_linked = [norm(residuals_linked[:, i]) for i in 1:n]
    
    
    errors_linked = zeros(T, n)
    @threads for i in 1:n
        @inbounds @simd for j in 1:d
            diff_linked =  X[j, i] - reconstructed_linked[j, i]
            errors_linked[i] += diff_linked * diff_linked
        end
    end
    mean_error=mean(errors_linked)
    
    
    linked_counts = count(parent .> 0)
    linked_ratio = linked_counts / n
    
    
    
    println("linkeded Reconstruction Analysis Summary:")
    println("  linked Ratio: $(round(100*linked_ratio, digits=2))%")
    println("  linked Residual Norm: Max=$(round(maximum(residual_norms_linked), digits=4)), Min=$(round(minimum(residual_norms_linked), digits=4)), Mean=$(round(mean(residual_norms_linked), digits=4))")
    println("  linked Error: Max=$(round(maximum(errors_linked), digits=4)), Min=$(round(minimum(errors_linked), digits=4)), Mean=$(round(mean_error ,digits=4))")
    
    if plt!==nothing
        
        histogram!(plt, residual_norms_linked, bins=50, 
                label="linked", color=:red, alpha=0.5,
                subplot=1)
        
        
        histogram!(plt, errors_linked, bins=50, 
                label="linked", color=:red, alpha=0.5,
                subplot=2)
        display(plt)
    end
    return mean_error,errors_linked
end


function compute_linkeded_reconstruction_multibook(
    B::Matrix{BType},
    len_rate_each::Matrix{T},
    C_root::Vector{Matrix{T}},
    C_one::Vector{Matrix{T}},
    parent::Vector{UInt32}
) where {T<:AbstractFloat, BType<:Integer}
    
    m, n = size(len_rate_each)
    d = size(C_root[1], 1)
    reconstructed = zeros(T, d, n)
    
    
    depths = compute_depths(parent)
    
    
    R_self = zeros(T, d, n)
    @threads for i in 1:n
        if depths[i] != 0
            C=C_one
        else
            C=C_root
        end
        for j in 1:m
            code_idx = B[j, i]
            if code_idx != 0
                α = len_rate_each[j, i]
                @inbounds @simd for k in 1:d
                    R_self[k, i] += α * C[j][k, code_idx]
                end
            end
        end
    end
    
    
    depth_order = sortperm(depths)  
    
    for i in depth_order  
        if parent[i] == 0
            
            reconstructed[:, i] = R_self[:, i]
        else
            
            p_idx = Int(parent[i])
            reconstructed[:, i] = reconstructed[:, p_idx] + R_self[:, i]
        end
    end
    
    return reconstructed
end




function compute_depths(parent::Vector{UInt32})
    n = length(parent)
    depths = zeros(Int, n)
    
    
    for i in 1:n
        cur = i
        depth = 0
        while parent[cur] != 0
            depth += 1
            cur = Int(parent[cur])
        end
        depths[i] = depth
    end
    
    return depths
end





function save_c_R(
  bpath::String, C::Vector{Matrix{Float32}},R::Matrix{Float32})
  lc=length(C)
  for i = 1:lc
    h5write(bpath, "C_$i", C[i])
  end
  h5write(bpath, "R", R)
end

function load_c_R(fname::String, m::Integer)
  C = Vector{Matrix{Float32}}(undef, m)
  for i=1:m; C[i] = h5read(fname, "C_$i"); end
  R = h5read(fname, "R")
  return C, R
end

function save_c_Rv(
  bpath::String, C::Vector{Matrix{Float32}},R::Matrix{Float32},is_bad_cluster::AbstractVector{Bool})
  lc=length(C)
  for i = 1:lc
    h5write(bpath, "C_$i", C[i])
  end
  h5write(bpath, "R", R)
  is_bad_cluster_uint8 = UInt8.(is_bad_cluster)
  h5write(bpath, "is_bad_cluster", is_bad_cluster_uint8)
end
function load_c_Rv(fname::String, m::Integer)
  C = Vector{Matrix{Float32}}(undef, m)
  for i=1:m; C[i] = h5read(fname, "C_$i"); end
  R = h5read(fname, "R")
  is_bad_cluster_uint8 = h5read(fname, "is_bad_cluster")
  is_bad_cluster = Bool.(is_bad_cluster_uint8)
  return C, R, is_bad_cluster
end


function save_base_results_hq(
  bpath::String, trial::Integer, B_base, len_rate_each_base::Matrix{Float32}, base_error)
  h5write(bpath, "$(trial)/B_base", convert(Matrix{UInt32}, B_base))
  h5write(bpath, "$(trial)/len_rate_each_base", len_rate_each_base)
  h5write(bpath, "$(trial)/base_error", base_error)
end
function load_base_hq(fname::String, m::Integer, trial::Integer)
  B_base = h5read(fname, "$trial/B_base");
  len_rate_each_base = h5read(fname, "$trial/len_rate_each_base")
  return B_base,len_rate_each_base
end
function save_base_results_anonymousmethod(
  bpath::String, trial::Integer, B_base, len_rate_each_base::Matrix{Float32}, parent, base_error)
  h5write(bpath, "$(trial)/B_base", convert(Matrix{UInt32}, B_base))
  h5write(bpath, "$(trial)/len_rate_each_base", len_rate_each_base)
  h5write(bpath, "$(trial)/parent", parent)
  h5write(bpath, "$(trial)/base_error", base_error)
end
function load_base_anonymousmethod(fname::String, m::Integer, trial::Integer)
  B_base = h5read(fname, "$trial/B_base");
  parent = h5read(fname, "$trial/parent");
  len_rate_each_base = h5read(fname, "$trial/len_rate_each_base")
  return B_base,len_rate_each_base,parent
end
