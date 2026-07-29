module TLSstruct

export TLS,TLS_ils,FSphericalKmeansResult

    


    struct FSphericalKmeansResult
        assignments::Vector{Int}
        centers::Matrix{Float64}
        costs::Vector{Float64}
        converged::Bool
        iterations::Int
    end

    mutable struct TLS{T_Type, B_Type}
        A::Matrix{T_Type}
        R_self::Vector{T_Type}
        re::Vector{T_Type}
        B_init::Vector{B_Type}
        a_init::Vector{T_Type}
        a_bak::Vector{T_Type}
        B_best::Vector{B_Type}
        a_best::Vector{T_Type}
        xC_base_buf::Vector{T_Type}
        xC_buf::Vector{T_Type}
        rC_buf::Vector{T_Type}
        order_idx::Vector{Int}
        processed::BitVector
        candidates::Vector{Int}         
    end


    mutable struct TLS_ils{T_Type, B_Type}
        A::Matrix{T_Type}
        R_self::Vector{T_Type}
        re::Vector{T_Type}
        B_init::Vector{B_Type}
        a_init::Vector{T_Type}
        a_bak::Vector{T_Type}
        B_best::Vector{B_Type}
        a_best::Vector{T_Type}
        xC_base_buf::Vector{T_Type}
        xC_buf::Vector{T_Type}
        rC_buf::Vector{T_Type}
        order_idx::Vector{Int}
        processed::BitVector
        candidates::Vector{Int}         
        B_try::Vector{B_Type}
        a_try::Vector{T_Type}
        pert_layers::Vector{Int}
        new_flat::Vector{Int}
    end
end