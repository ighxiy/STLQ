include("./xvecs_read.jl")
include("./xvecs_write.jl")
function load_experiment_data(
  dataset_name::String,
  ntrain::Integer, nbase::Integer, nquery::Integer, V::Bool=false)

  Xt = read_one_dataset(dataset_name, ntrain, V)
  Xb = read_one_dataset(dataset_name * "_base", nbase, V)
  Xq = read_one_dataset(dataset_name * "_query", nquery, V)[:,1:nquery]
  gt = read_one_dataset(dataset_name * "_groundtruth", nquery, V)
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M" || dataset_name == "SIFTSMALL"
    gt = gt .+ 1
  end

  if dataset_name != "Deep1M"
    gt = convert( Vector{UInt32}, gt[1,1:nquery] )
  else
    gt = convert( Vector{UInt32}, gt[1:nquery] )
  end

  return Xt, Xb, Xq, gt
end



data_path_prefix = "./download/pq-dataset/"

function read_one_dataset(
  dname::AbstractString,  
  nvectors::Union{Integer, UnitRange},      
  V::Bool=false )         

  if V println("Loading $(dname)... "); end

  
  if dname == "GIST1M"

    fname = joinpath(data_path_prefix,"gist/gist_learn.fvecs");
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "GIST1M_query"

    fname = joinpath(data_path_prefix,"gist/gist_query.fvecs");
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "GIST1M_groundtruth"

    fname = joinpath(data_path_prefix,"gist/gist_groundtruth.ivecs");
    X     = ivecs_read(nvectors, fname)
    return X

  elseif dname == "GIST1M_base"

    fname = joinpath(data_path_prefix,"gist/gist_base.fvecs");
    X     = fvecs_read(nvectors, fname)
    return X

  
  elseif dname == "SIFT1M"

    fname = joinpath(data_path_prefix,"sift/sift_learn.fvecs");
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "SIFT1M_query"

    fname = joinpath(data_path_prefix,"sift/sift_query.fvecs");
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "SIFT1M_groundtruth"
    fname = joinpath(data_path_prefix,"sift/sift_groundtruth.ivecs");
    X     = ivecs_read(nvectors, fname)
    return X

  elseif dname == "SIFT1M_base"
    fname = joinpath(data_path_prefix,"sift/sift_base.fvecs");
    X     = fvecs_read(nvectors, fname)
    return X

  
  elseif dname == "SIFTSMALL"
    fname = joinpath(data_path_prefix,"siftsmall/siftsmall_learn.fvecs");
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "SIFTSMALL_query"
    fname = joinpath(data_path_prefix,"siftsmall/siftsmall_query.fvecs");
    X     = fvecs_read(nvectors, fname)
    return X

  elseif dname == "SIFTSMALL_groundtruth"
    fname = joinpath(data_path_prefix,"siftsmall/siftsmall_groundtruth.ivecs");
    X     = ivecs_read(nvectors, fname)
    return X

  elseif dname == "SIFTSMALL_base"
    fname = joinpath(data_path_prefix,"siftsmall/siftsmall_base.fvecs");
    X     = fvecs_read(nvectors, fname)
    return X
  else

    error("Dataset $(dname) unknown")

  end

  _, n = size( X );

  if typeof( nvectors ) <: Integer
    if nvectors <= n
      X = X[:, 1:nvectors];
    else
      error("Asked to read $nvectors vectors, but the datasets has only $n vectors.");
    end
  else
    X = X[:, nvectors];
  end

  return X
end
