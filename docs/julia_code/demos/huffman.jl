using Statistics



mutable struct BitWriter
    buf::Vector{UInt8}
    acc::UInt32
    nbits::Int
end
BitWriter() = BitWriter(UInt8[], 0x00, 0)

@inline function writebits!(bw::BitWriter, code::UInt32, len::Int)
    @inbounds for i in (len-1):-1:0  
        bw.acc = (bw.acc << 1) | ((code >> i) & 0x01)
        bw.nbits += 1
        if bw.nbits == 8
            push!(bw.buf, UInt8(bw.acc & 0xFF))
            bw.acc = 0x00
            bw.nbits = 0
        end
    end
    return nothing
end

function finish!(bw::BitWriter)
    if bw.nbits != 0
        bw.acc <<= (8 - bw.nbits)
        push!(bw.buf, UInt8(bw.acc & 0xFF))
        bw.acc = 0x00
        bw.nbits = 0
    end
    return bw.buf
end

mutable struct BitReader
    buf::Vector{UInt8}
    pos::Int
    acc::UInt32
    nbits::Int
end
BitReader(bytes::Vector{UInt8}) = BitReader(bytes, 1, 0x00, 0)

@inline function readbit(br::BitReader)
    if br.nbits == 0
        br.pos > length(br.buf) && error("BitReader: out of data")
        br.acc = UInt32(br.buf[br.pos]); br.pos += 1
        br.nbits = 8
    end
    br.nbits -= 1
    return (br.acc >> br.nbits) & 0x01
end




struct HNode
    f::Int      
    left::Int   
    right::Int  
    sym::Int    
end


function huffman_lengths(freq::Vector{Int})
    K = length(freq)
    @assert K >= 1
    if K == 1
        return [1]
    end
    
    nodes = HNode[]
    active = Int[]           
    for s in 1:K
        f = max(freq[s], 1)  
        push!(nodes, HNode(f, 0, 0, s))
        push!(active, length(nodes))
    end
    
    while length(active) > 1
        
        i1 = active[1]; i2 = active[2]
        f1 = nodes[i1].f; f2 = nodes[i2].f
        for idx in @view active[3:end]
            f = nodes[idx].f
            if f < f1 || (f == f1 && idx < i1)
                i2, f2 = i1, f1
                i1, f1 = idx, f
            elseif f < f2 || (f == f2 && idx < i2)
                i2, f2 = idx, f
            end
        end
        
        push!(nodes, HNode(f1+f2, i1, i2, 0))
        root = length(nodes)
        
        deleteat!(active, findfirst(==(i1), active))
        deleteat!(active, findfirst(==(i2), active))
        push!(active, root)
    end
    root = active[1]

    
    lens = zeros(Int, K)
    stack = [(root, 0)]
    while !isempty(stack)
        idx, d = pop!(stack)
        nd = nodes[idx]
        if nd.sym != 0
            lens[nd.sym] = max(d, 1)
        else
            
            push!(stack, (nd.left,  d+1))
            push!(stack, (nd.right, d+1))
        end
    end
    return lens
end





function canonical_aux_from_lengths(lens::Vector{Int})
    K = length(lens)
    syms = collect(1:K)
    sort!(syms, by = s -> (lens[s], s))      
    maxlen = maximum(lens)
    bl_count = zeros(Int, maxlen)
    for s in syms
        L = lens[s]; L>0 && (bl_count[L] += 1)
    end
    first_code = zeros(Int, maxlen)
    code = 0
    for L in 1:maxlen
        code <<= 1
        first_code[L] = code
        code += bl_count[L]
    end
    first_sym = zeros(Int, maxlen)
    pos = 1
    for L in 1:maxlen
        first_sym[L] = pos
        pos += bl_count[L]
    end
    return (syms=syms, first_code=first_code, first_sym=first_sym,
            bl_count=bl_count, maxlen=maxlen)
end


function canonical_codes(lens::Vector{Int})
    K = length(lens)
    aux = canonical_aux_from_lengths(lens)
    syms, first_code, bl_count, maxlen = aux.syms, aux.first_code, aux.bl_count, aux.maxlen
    codes = fill(UInt32(0), K)
    next_code = copy(first_code)
    for s in syms
        L = lens[s]
        if L > 0
            codes[s] = UInt32(next_code[L])
            next_code[L] += 1
        end
    end
    return codes, lens
end


@inline function readsymbol_canonical!(br::BitReader,
                                       first_code::Vector{Int},
                                       first_sym::Vector{Int},
                                       bl_count::Vector{Int},
                                       maxlen::Int)
    code = 0
    for L in 1:maxlen
        bit = Int(readbit(br))
        code = (code << 1) | bit
        base = first_code[L]
        cnt  = bl_count[L]
        if cnt > 0
            off = code - base
            if 0 <= off < cnt
                return first_sym[L] + off  
            end
        end
    end
    error("Huffman decode failed: exceeded maxlen=$maxlen")
end


















function _compress_group_huffman(
    values::AbstractVector{<:Real};
    name::AbstractString = "group",
)
    N = length(values)
    if N == 0
        empty_stream = UInt8[]
        model = Dict(
            :alphabet => Int16[],
            :codelen  => UInt8[],
        )
        stats = Dict(
            :name => name,
            :N => 0,
            :K => 0,
            :entropy => 0.0,
            :data_bits => 0.0,
            :hdr_bits  => 0.0,
            :total_bits => 0.0,
            :avg_data_bits  => 0.0,
            :avg_total_bits => 0.0,
        )
        return empty_stream, model, stats
    end

    
    
    if values isa AbstractVector{Int8}
        vals = values::AbstractVector{Int8}
    else
        buf = Vector{Int8}(undef, N)
        @inbounds for i in 1:N
            buf[i] = Int8(round(Int, values[i]))
        end
        vals = buf
    end

    
    alph_set = Set{Int8}()
    @inbounds for v in vals
        push!(alph_set, v)
    end
    alphabet_int8 = sort!(collect(alph_set))     
    K = length(alphabet_int8)
    alphabet = Int16.(alphabet_int8)             

    
    val2sym = Dict{Int8,Int}()
    @inbounds for (i, v) in enumerate(alphabet_int8)
        val2sym[v] = i
    end

    
    freq = zeros(Int, K)
    symbols = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        s = val2sym[vals[i]]
        symbols[i] = s
        freq[s] += 1
    end

    S = N
    H = 0.0
    @inbounds for f in freq
        if f > 0
            p = f / S
            H -= p * log2(p)
        end
    end

    
    lens = huffman_lengths(freq)          
    codes, lens = canonical_codes(lens)   

    
    bw = BitWriter()
    data_bits = 0.0
    @inbounds for s in symbols
        writebits!(bw, codes[s], lens[s])
        data_bits += lens[s]
    end
    stream = finish!(bw)   

    
    
    
    
    
    
    hdr_bits = 8*K   

    total_bits = data_bits + hdr_bits
    avg_data_bits  = data_bits  / S
    avg_total_bits = total_bits / S

    model = Dict(
        :alphabet => alphabet,
        :codelen  => UInt8.(lens),
    )

    stats = Dict(
        :name => name,
        :N => S,
        :K => K,
        :entropy => H,
        :data_bits => data_bits,
        :hdr_bits  => hdr_bits,
        :total_bits => total_bits,
        :avg_data_bits  => avg_data_bits,
        :avg_total_bits => avg_total_bits,
    )

    return stream, model, stats
end
































function compress_coeffs_huffman_root_linked_layered(
    a::AbstractMatrix{<:Real},
    is_linked::AbstractVector{Bool};
    verbose::Bool = true,
)
    m, n = size(a)
    @assert length(is_linked) == n "length(is_linked) must equal number of columns n"

    
    A_int8 = Matrix{Int8}(undef, m, n)
    @inbounds for j in 1:n
        for i in 1:m
            A_int8[i, j] = Int8(round(Int, a[i, j]))
        end
    end

    root_cols  = [j for j in 1:n if !is_linked[j]]
    linked_cols = [j for j in 1:n if  is_linked[j]]

    n_root  = length(root_cols)
    n_linked = length(linked_cols)

    verbose && println("Huffman (root/linked × per-layer): m=$m, n=$n, n_root=$n_root, n_linked=$n_linked")

    root_streams  = Vector{Vector{UInt8}}(undef, m)
    linked_streams = Vector{Vector{UInt8}}(undef, m)
    root_models   = Vector{Dict{Symbol,Any}}(undef, m)
    linked_models  = Vector{Dict{Symbol,Any}}(undef, m)

    root_layer_stats  = Vector{Dict{Symbol,Any}}(undef, m)
    linked_layer_stats = Vector{Dict{Symbol,Any}}(undef, m)

    total_bits_data = 0.0
    total_bits_hdr  = 0.0

    
    entropy_root  = zeros(Float64, m)
    entropy_linked = zeros(Float64, m)
    avg_root      = zeros(Float64, m)
    avg_linked     = zeros(Float64, m)
    K_root        = zeros(Int, m)
    K_linked       = zeros(Int, m)

    for ℓ in 1:m
        
        root_vals = n_root == 0 ? Int8[] : (@view A_int8[ℓ, root_cols])
        
        linked_vals = n_linked == 0 ? Int8[] : (@view A_int8[ℓ, linked_cols])

        
        r_stream, r_model, r_stats =
            _compress_group_huffman(root_vals; name = "ROOT-L$ℓ")
        root_streams[ℓ]  = r_stream
        root_models[ℓ]   = r_model
        root_layer_stats[ℓ] = r_stats

        entropy_root[ℓ] = r_stats[:entropy]
        K_root[ℓ]       = r_stats[:K]

        
        c_stream, c_model, c_stats =
            _compress_group_huffman(linked_vals; name = "linked-L$ℓ")
        linked_streams[ℓ]  = c_stream
        linked_models[ℓ]   = c_model
        linked_layer_stats[ℓ] = c_stats

        entropy_linked[ℓ] = c_stats[:entropy]
        K_linked[ℓ]       = c_stats[:K]

        
        total_bits_data += r_stats[:data_bits] + c_stats[:data_bits]
        total_bits_hdr  += r_stats[:hdr_bits]  + c_stats[:hdr_bits]

        
        N_layer = r_stats[:N] + c_stats[:N]
        if N_layer > 0
            avg_root[ℓ]  = r_stats[:total_bits] / max(r_stats[:N], 1)
            avg_linked[ℓ] = c_stats[:total_bits] / max(c_stats[:N], 1)
        else
            avg_root[ℓ]  = 0.0
            avg_linked[ℓ] = 0.0
        end
    end

    total_bits = total_bits_data + total_bits_hdr
    total_coeffs = m * n
    avg_bits_per_coeff = total_bits / total_coeffs

    if verbose
        println("ENTROPY CODING REPORT (per layer × root/linked, 0th-order Huffman)")
        println("==========================================================")
        for ℓ in 1:m
            rs = root_layer_stats[ℓ]
            cs = linked_layer_stats[ℓ]
            @printf("L%-2d ROOT : K=%-3d entropy=%.3f  data=%.3f  total=%.3f b/coef (N=%d)\n",
                    ℓ, rs[:K], rs[:entropy], rs[:avg_data_bits], rs[:avg_total_bits], rs[:N])
            @printf("    linked: K=%-3d entropy=%.3f  data=%.3f  total=%.3f b/coef (N=%d)\n",
                    cs[:K], cs[:entropy], cs[:avg_data_bits], cs[:avg_total_bits], cs[:N])
        end
        println("----------------------------------------------------------")
        @printf("Total data bits:  %.3f Mbits\n", total_bits_data / 1e6)
        @printf("Header bits:      %.3f Mbits\n", total_bits_hdr  / 1e6)
        @printf("Average (data+header): %.3f bits / coefficient\n", avg_bits_per_coeff)
    end

    comp = Dict(
        :root_streams  => root_streams,
        :linked_streams => linked_streams,
        :root_models   => root_models,
        :linked_models  => linked_models,
    )

    stats = Dict(
        :root_layer_stats  => root_layer_stats,
        :linked_layer_stats => linked_layer_stats,
        :total_bits_data   => total_bits_data,
        :total_bits_hdr    => total_bits_hdr,
        :total_bits        => total_bits,
        :avg_bits_per_coeff => avg_bits_per_coeff,
        :m => m,
        :n => n,
    )

    return comp, stats
end




























function build_ivf_lists_for_linked(
    is_linked::AbstractVector{Bool},
    cluster_id::AbstractVector{<:Integer};
    nlists::Union{Nothing,Int}=nothing,
)
    n = length(is_linked)
    @assert length(cluster_id) == n "length(cluster_id) must equal number of columns n"

    C = nlists === nothing ? maximum(cluster_id) : nlists
    lists = [Int[] for _ in 1:C]

    @inbounds for j in 1:n
        if is_linked[j]
            c = Int(cluster_id[j])
            @assert 1 <= c <= C "cluster_id[$j] = $c is out of [1,$C]"
            push!(lists[c], j)
        end
    end

    return lists
end




















































function compress_coeffs_huffman_root_linked_layer_cluster(
    a::AbstractMatrix{<:Real},
    is_linked::AbstractVector{Bool},
    cluster_id::AbstractVector{<:Integer};
    nlists::Union{Nothing,Int} = nothing,
    verbose::Bool = true,
)
    m, n = size(a)
    @assert length(is_linked) == n "length(is_linked) must equal number of columns n"
    @assert length(cluster_id) == n "length(cluster_id) must equal number of columns n"

    
    A_int8 = Matrix{Int8}(undef, m, n)
    @inbounds for j in 1:n
        for i in 1:m
            A_int8[i, j] = Int8(round(Int, a[i, j]))
        end
    end

    
    root_cols  = [j for j in 1:n if !is_linked[j]]
    n_root = length(root_cols)

    
    ivf_linked = build_ivf_lists_for_linked(is_linked, cluster_id; nlists=nlists)
    num_lists = length(ivf_linked)

    n_linked = 0
    @inbounds for c in 1:num_lists
        n_linked += length(ivf_linked[c])
    end

    verbose && println("Huffman (root/linked × per-layer, linked per-cluster): m=$m, n=$n, n_root=$n_root, n_linked=$n_linked, num_lists=$num_lists")

    
    
    root_streams = Vector{Vector{UInt8}}(undef, m)
    root_models  = Vector{Dict{Symbol,Any}}(undef, m)
    root_layer_stats = Vector{Dict{Symbol,Any}}(undef, m)

    
    linked_streams = Vector{Vector{Vector{UInt8}}}(undef, m)      
    linked_models  = Vector{Vector{Dict{Symbol,Any}}}(undef, m)   
    linked_layer_stats = Vector{Vector{Dict{Symbol,Any}}}(undef, m) 

    layer_entropy_root    = zeros(Float64, m)
    layer_entropy_linked   = zeros(Float64, m)
    layer_avg_total_root  = zeros(Float64, m)
    layer_avg_total_linked = zeros(Float64, m)

    total_bits_data_root  = 0.0
    total_bits_hdr_root   = 0.0
    total_bits_data_linked = 0.0
    total_bits_hdr_linked  = 0.0

    
    for ℓ in 1:m
        
        if n_root == 0
            root_vals = Int8[]
        else
            root_vals = @view A_int8[ℓ, root_cols]
        end

        r_stream, r_model, r_stats =
            _compress_group_huffman(root_vals; name = "ROOT-L$ℓ")

        root_streams[ℓ]    = r_stream
        root_models[ℓ]     = r_model
        root_layer_stats[ℓ] = r_stats

        layer_entropy_root[ℓ]   = r_stats[:entropy]
        layer_avg_total_root[ℓ] = r_stats[:avg_total_bits]

        total_bits_data_root += r_stats[:data_bits]
        total_bits_hdr_root  += r_stats[:hdr_bits]

        
        streams_ℓ = Vector{Vector{UInt8}}(undef, num_lists)
        models_ℓ  = Vector{Dict{Symbol,Any}}(undef, num_lists)
        stats_ℓ   = Vector{Dict{Symbol,Any}}(undef, num_lists)

        total_syms_linked_layer = 0
        entropy_weighted_linked = 0.0
        total_bits_linked_layer = 0.0

        for c in 1:num_lists
            idxs = ivf_linked[c]
            vals = isempty(idxs) ? Int8[] : (@view A_int8[ℓ, idxs])

            s_stream, s_model, s_stats =
                _compress_group_huffman(vals; name = "linked-L$(ℓ)-C$(c)")

            streams_ℓ[c] = s_stream
            models_ℓ[c]  = s_model
            stats_ℓ[c]   = s_stats

            Nc = s_stats[:N]
            total_syms_linked_layer += Nc
            entropy_weighted_linked += s_stats[:entropy] * Nc
            total_bits_linked_layer += s_stats[:total_bits]

            total_bits_data_linked += s_stats[:data_bits]
            total_bits_hdr_linked  += s_stats[:hdr_bits]
        end

        linked_streams[ℓ]    = streams_ℓ
        linked_models[ℓ]     = models_ℓ
        linked_layer_stats[ℓ] = stats_ℓ

        if total_syms_linked_layer > 0
            layer_entropy_linked[ℓ]   = entropy_weighted_linked / total_syms_linked_layer
            layer_avg_total_linked[ℓ] = total_bits_linked_layer / total_syms_linked_layer
        else
            layer_entropy_linked[ℓ]   = 0.0
            layer_avg_total_linked[ℓ] = 0.0
        end
    end

    
    total_bits_data = total_bits_data_root + total_bits_data_linked
    total_bits_hdr  = total_bits_hdr_root  + total_bits_hdr_linked
    total_bits      = total_bits_data + total_bits_hdr

    total_coeffs = m * n
    avg_bits_per_coeff = total_bits / total_coeffs

    if verbose
        println("ENTROPY CODING REPORT (per layer × root, per layer×cluster for linked, 0th-order Huffman)")
        println("=======================================================================")
        for ℓ in 1:m
            rs = root_layer_stats[ℓ]
            Nroot = rs[:N]
            Nlinked_layer = 0
            for st in linked_layer_stats[ℓ]
                Nlinked_layer += st[:N]
            end
            @printf("L%-2d ROOT : entropy=%.3f  total=%.3f b/coef (N=%d)\n",
                    ℓ, layer_entropy_root[ℓ], layer_avg_total_root[ℓ], Nroot)
            @printf("    linked: entropy(avg over clusters)=%.3f  total=%.3f b/coef (N=%d, lists=%d)\n",
                    layer_entropy_linked[ℓ], layer_avg_total_linked[ℓ], Nlinked_layer, num_lists)
        end
        println("-----------------------------------------------------------------------")
        @printf("Total data bits (ROOT) : %.3f Mbits\n", total_bits_data_root  / 1e6)
        @printf("Total data bits (linked): %.3f Mbits\n", total_bits_data_linked / 1e6)
        @printf("Header bits (ROOT)     : %.3f Mbits\n", total_bits_hdr_root   / 1e6)
        @printf("Header bits (linked)    : %.3f Mbits\n", total_bits_hdr_linked  / 1e6)
        @printf("Total data bits        : %.3f Mbits\n", total_bits_data / 1e6)
        @printf("Total header bits      : %.3f Mbits\n", total_bits_hdr  / 1e6)
        @printf("Average (data+header): %.3f bits / coefficient\n", avg_bits_per_coeff)
    end

    comp = Dict(
        :root_streams    => root_streams,    
        :root_models     => root_models,     
        :linked_streams   => linked_streams,   
        :linked_models    => linked_models,    
        :linked_ivf_lists => ivf_linked,       
    )

    stats = Dict(
        :root_layer_stats   => root_layer_stats,
        :linked_layer_stats  => linked_layer_stats,
        :layer_entropy_root => layer_entropy_root,
        :layer_entropy_linked => layer_entropy_linked,
        :layer_avg_total_root  => layer_avg_total_root,
        :layer_avg_total_linked => layer_avg_total_linked,
        :total_bits_data_root  => total_bits_data_root,
        :total_bits_hdr_root   => total_bits_hdr_root,
        :total_bits_data_linked => total_bits_data_linked,
        :total_bits_hdr_linked  => total_bits_hdr_linked,
        :total_bits_data       => total_bits_data,
        :total_bits_hdr        => total_bits_hdr,
        :total_bits            => total_bits,
        :avg_bits_per_coeff    => avg_bits_per_coeff,
        :m => m,
        :n => n,
        :num_lists => num_lists,
    )

    return comp, stats
end
