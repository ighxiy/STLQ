module anonymousmethod

include("../demos/Beam_search.jl")
using .Beam_search
export Beam_search

include("../demos/structs.jl")
using .TLSstruct
export TLSstruct





using Libdl


macro checked_lib(libname, path)
    if Libdl.dlopen_e(path) == C_NULL
        error("Unable to load \n\n$libname ($path)\n")
    end
    quote
        const $(esc(libname)) = $path
    end
end


@checked_lib linscan_linked "./anonymousmethod/demos/evaluate-recall/so/linscan_linked.so"
@checked_lib quantize_norm_linked "./demos/evaluate-recall/so/quantize_norm_linked.so"
@checked_lib quantize_norm_base "./demos/evaluate-recall/so/quantize_norm_base.so"
@checked_lib linscan_hq "./anonymousmethod/demos/evaluate-recall/so/linscan_hq.so"




end 
