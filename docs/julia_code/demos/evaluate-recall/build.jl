using BinDeps

@BinDeps.setup

linscan_hq = library_dependency("linscan_hq", aliases=["linscan_hq","linscan_hq.so"])
linscan_linked = library_dependency("linscan_linked", aliases=["linscan_linked","linscan_linked.so"])
quantize_norm_base = library_dependency("quantize_norm_base", aliases=["quantize_norm_base","quantize_norm_base.so"])
quantize_norm_linked = library_dependency("quantize_norm_linked", aliases=["quantize_norm_linked","quantize_norm_linked.so"])

deps = [linscan_hq, quantize_norm_base, quantize_norm_linked, linscan_linked]

prefix=joinpath(BinDeps.depsdir(linscan_hq))
linscan_aqdbuilddir = joinpath(BinDeps.depsdir(linscan_hq),"builds")

provides(BuildProcess,
    (@build_steps begin
        CreateDirectory(linscan_aqdbuilddir)
        @build_steps begin
            ChangeDirectory(linscan_aqdbuilddir)
            FileRule(joinpath(prefix,"builds","linscan_hq.so"),@build_steps begin
                `g++ -O3 -shared -fPIC ../src/linscan_hq.cpp -o linscan_hq.so -fopenmp`
            end)
        end
    end),linscan_hq, os = :Unix, installed_libpath=joinpath(prefix,"builds"))
    
    
provides(BuildProcess,
    (@build_steps begin
        CreateDirectory(linscan_aqdbuilddir)
        @build_steps begin
            ChangeDirectory(linscan_aqdbuilddir)
            FileRule(joinpath(prefix,"builds","linscan_linked.so"),@build_steps begin
                `g++ -O3 -shared -fPIC ../src/linscan_linked.cpp -o linscan_linked.so -fopenmp`
            end)
        end
    end),linscan_linked, os = :Unix, installed_libpath=joinpath(prefix,"builds"))
    
    
mklroot = get(ENV, "MKLROOT", "/opt/intel/oneapi/mkl/latest")
provides(BuildProcess,
    (@build_steps begin
        CreateDirectory(linscan_aqdbuilddir)
        @build_steps begin
            ChangeDirectory(linscan_aqdbuilddir)
            FileRule(joinpath(prefix, "builds", "quantize_norm_base.so"), @build_steps begin
                `g++ -O3 -fPIC -shared -fopenmp -m64 -march=native 
                -I$mklroot/include 
                -o quantize_norm_base.so 
                ../src/quantize_norm_base.cpp 
                -Wl,--start-group $mklroot/lib/intel64/libmkl_intel_lp64.a 
                $mklroot/lib/intel64/libmkl_gnu_thread.a 
                $mklroot/lib/intel64/libmkl_core.a 
                -Wl,--end-group -lgomp -lpthread -lm -ldl`
            end)
        end
    end), quantize_norm_base, os = :Unix, installed_libpath = joinpath(prefix, "builds"))
    
provides(BuildProcess,
    (@build_steps begin
        CreateDirectory(linscan_aqdbuilddir)
        @build_steps begin
            ChangeDirectory(linscan_aqdbuilddir)
            FileRule(joinpath(prefix, "builds", "quantize_norm_linked.so"), @build_steps begin
                `g++ -O3 -fPIC -shared -fopenmp -m64 -march=native 
                -I$mklroot/include 
                -o quantize_norm_linked.so 
                ../src/quantize_norm_linked.cpp 
                -Wl,--start-group $mklroot/lib/intel64/libmkl_intel_lp64.a 
                $mklroot/lib/intel64/libmkl_gnu_thread.a 
                $mklroot/lib/intel64/libmkl_core.a 
                -Wl,--end-group -lgomp -lpthread -lm -ldl`
            end)
        end
    end), quantize_norm_linked, os = :Unix, installed_libpath = joinpath(prefix, "builds"))
    

@BinDeps.install Dict([(:linscan_hq, :linscan_hq),
                      (:linscan_linked, :linscan_linked),
                      (:quantize_norm_base, :quantize_norm_base),
                      (:quantize_norm_linked, :quantize_norm_linked)])
                      
                     
                     
                     
                     
