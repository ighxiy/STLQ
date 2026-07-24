#include "stlq/eval/linkage_norm_provider.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "stlq/core/blas.h"
#include "stlq/core/threading.h"

namespace stlq::eval {

namespace {

inline float Dot(const float* a, const float* b, int d) {
    float s = 0.0f;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < d; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

inline float Norm2(const float* a, int d) {
    return Dot(a, a, d);
}

inline int ReadSmallCode(const std::uint8_t* codes_bytes, std::size_t pos, int m_codes, int idx) {
    return static_cast<int>(codes_bytes[pos * static_cast<std::size_t>(m_codes) + static_cast<std::size_t>(idx)]);
}

inline int ReadCode0One(const std::uint8_t* code0_bytes, std::size_t pos) {
    return static_cast<int>(code0_bytes[pos]);
}

}  // namespace

LinkageNormProviderLookup::LinkageNormProviderLookup(const ColMajorMatrix<float>& C_root0,
                                                 const CodebookMeta& meta_root_small,
                                                 const CodebookMeta& meta_one)
    : C_root0_(C_root0), meta_root_small_(meta_root_small), meta_one_(meta_one) {
    const int m = meta_one_.m;
    offsets_root_small_.assign(static_cast<std::size_t>(m), 0);
    for (int l = 1; l < m; ++l) {
        offsets_root_small_[static_cast<std::size_t>(l)] =
            meta_root_small_.offsets[static_cast<std::size_t>(l - 1)];
    }

    // Global dot tables.
    D_rr_ = ColMajorMatrix<float>(meta_root_small_.total_cols, meta_root_small_.total_cols);
    D_oo_ = ColMajorMatrix<float>(meta_one_.total_cols, meta_one_.total_cols);
    D_ro_ = ColMajorMatrix<float>(meta_root_small_.total_cols, meta_one_.total_cols);

    {
        ScopedBlasThreads scope(OmpMaxThreads());
        Gemm(true, false, 1.0f, meta_root_small_.flat, meta_root_small_.flat, 0.0f, &D_rr_);
        Gemm(true, false, 1.0f, meta_one_.flat, meta_one_.flat, 0.0f, &D_oo_);
        Gemm(true, false, 1.0f, meta_root_small_.flat, meta_one_.flat, 0.0f, &D_ro_);
    }
}

bool LinkageNormProviderLookup::ComputeNorm2(const ClusterView& cv,
                                           std::vector<float>* out_r_norm2,
                                           std::string* err) {
    if (!out_r_norm2) {
        return false;
    }
    const int d = meta_one_.d;
    const int m = cv.m;
    const int m_codes = cv.m_codes;
    if (d <= 0 || m <= 1 || m_codes != m - 1) {
        if (err) *err = "LinkageNormProviderLookup: invalid dims.";
        return false;
    }
    if (cv.cid < 0 || cv.cid >= C_root0_.cols) {
        if (err) *err = "LinkageNormProviderLookup: invalid cid.";
        return false;
    }
    if (cv.n_real <= 0 || cv.nc != cv.n_real + cv.n_virt) {
        if (err) *err = "LinkageNormProviderLookup: invalid cluster sizes.";
        return false;
    }
    if ((cv.parent_is_u16 && !cv.parent_1based_u16) ||
        (!cv.parent_is_u16 && !cv.parent_1based)) {
        if (err) *err = "LinkageNormProviderLookup: missing parent buffer.";
        return false;
    }

    const int n_virt = cv.n_virt;
    const int n_real = cv.n_real;
    const int nc = cv.nc;
    const int real_base = n_virt;
    const int n_root_real = cv.n_root_real;

    const float* c0 = C_root0_.Col(cv.cid);
    const float c0_norm2 = Norm2(c0, d);

    // Per-cluster dot vectors for root0 against root_small and one.
    std::vector<float> c0_dot_rootS(static_cast<std::size_t>(meta_root_small_.total_cols), 0.0f);
    for (int j = 0; j < meta_root_small_.total_cols; ++j) {
        c0_dot_rootS[static_cast<std::size_t>(j)] = Dot(c0, meta_root_small_.flat.Col(j), d);
    }
    std::vector<float> c0_dot_one(static_cast<std::size_t>(meta_one_.total_cols), 0.0f);
    for (int k = 0; k < meta_one_.total_cols; ++k) {
        c0_dot_one[static_cast<std::size_t>(k)] = Dot(c0, meta_one_.flat.Col(k), d);
    }

    std::vector<float> norm2_real(static_cast<std::size_t>(n_real), 0.0f);
    std::vector<float> norm2_virt(static_cast<std::size_t>(std::max(0, n_virt)), 0.0f);

    // Compute virt roots first so parent DP can reference them.
    const std::uint8_t* codes = cv.codes_small_bytes;
    const std::uint8_t* code0_one = cv.code0_one_bytes;
    const std::uint8_t* vcodes = cv.virt_codes_small_bytes;

    const auto run_with_parent = [&](const auto* parent_1based) -> bool {
        const auto compute_root_norm2 = [&](const std::uint8_t* codes_src, std::size_t pos_src,
                                            float a0, auto a_layer_fn) -> float {
            int idx[16];
            float a[16];
            const int mm = m;
            for (int l = 1; l < mm; ++l) {
                const int code = ReadSmallCode(codes_src, pos_src, m_codes, l - 1);
                idx[l - 1] = offsets_root_small_[static_cast<std::size_t>(l)] + code;
                a[l - 1] = a_layer_fn(l);
            }

            float cross0 = 0.0f;
            for (int l = 1; l < mm; ++l) {
                cross0 += a[l - 1] * c0_dot_rootS[static_cast<std::size_t>(idx[l - 1])];
            }
            float small = 0.0f;
            for (int j = 1; j < mm; ++j) {
                const float aj = a[j - 1];
                const int fj = idx[j - 1];
                small += (aj * aj) * D_rr_(fj, fj);
                for (int k = j + 1; k < mm; ++k) {
                    small += (2.0f * aj * a[k - 1]) * D_rr_(fj, idx[k - 1]);
                }
            }
            return a0 * a0 * c0_norm2 + 2.0f * a0 * cross0 + small;
        };

        for (int v = 0; v < n_virt; ++v) {
            const int local = v;
            const auto vpos = static_cast<std::size_t>(v);
            float a0 = 0.0f;
            if (cv.q_layer_major) {
                a0 = cv.scales_root[0] * static_cast<float>(cv.q_layer_major[local]);
                norm2_virt[vpos] = compute_root_norm2(
                    vcodes, static_cast<std::size_t>(v),
                    a0,
                    [&](int l) {
                        const std::size_t qidx = static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                                 static_cast<std::size_t>(local);
                        return cv.scales_root[l] * static_cast<float>(cv.q_layer_major[qidx]);
                    });
            } else {
                a0 = cv.virt_a0[vpos];
                norm2_virt[vpos] = compute_root_norm2(
                    vcodes, static_cast<std::size_t>(v),
                    a0,
                    [&](int l) {
                        return cv.virt_coeffs_small[vpos * static_cast<std::size_t>(m_codes) +
                                                    static_cast<std::size_t>(l - 1)];
                    });
            }
        }

        for (int pos = 0; pos < n_root_real; ++pos) {
            const int local = real_base + pos;
            const auto p = static_cast<std::size_t>(pos);
            float a0 = 0.0f;
            if (cv.q_layer_major) {
                a0 = cv.scales_root[0] * static_cast<float>(cv.q_layer_major[local]);
                norm2_real[p] = compute_root_norm2(
                    codes, p,
                    a0,
                    [&](int l) {
                        const std::size_t qidx = static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                                 static_cast<std::size_t>(local);
                        return cv.scales_root[l] * static_cast<float>(cv.q_layer_major[qidx]);
                    });
            } else {
                a0 = cv.a0[p];
                norm2_real[p] = compute_root_norm2(
                    codes, p,
                    a0,
                    [&](int l) { return cv.coeffs_small[p * static_cast<std::size_t>(m_codes) +
                                                        static_cast<std::size_t>(l - 1)]; });
            }
        }

        int idx_i[16];
        float b_i[16];
        int idx_u[16];
        float b_u[16];
        for (int pos = n_root_real; pos < n_real; ++pos) {
            const int local_ppos = real_base + pos;
            const auto ppos = static_cast<std::size_t>(pos);
            const int p = static_cast<int>(parent_1based[ppos]) - 1;

            idx_i[0] = meta_one_.offsets[0] + ReadCode0One(code0_one, ppos);
            if (cv.q_layer_major) {
                b_i[0] = cv.scales_linkage[0] * static_cast<float>(cv.q_layer_major[local_ppos]);
            } else {
                b_i[0] = cv.a0[ppos];
            }
            for (int l = 1; l < m; ++l) {
                const int code = ReadSmallCode(codes, ppos, m_codes, l - 1);
                idx_i[l] = meta_one_.offsets[static_cast<std::size_t>(l)] + code;
                if (cv.q_layer_major) {
                    const std::size_t qidx = static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                             static_cast<std::size_t>(local_ppos);
                    b_i[l] = cv.scales_linkage[l] * static_cast<float>(cv.q_layer_major[qidx]);
                } else {
                    b_i[l] = cv.coeffs_small[ppos * static_cast<std::size_t>(m_codes) +
                                             static_cast<std::size_t>(l - 1)];
                }
            }

            float res_norm2 = 0.0f;
            for (int j = 0; j < m; ++j) {
                const float aj = b_i[j];
                const int fj = idx_i[j];
                res_norm2 += (aj * aj) * D_oo_(fj, fj);
                for (int k = j + 1; k < m; ++k) {
                    res_norm2 += (2.0f * aj * b_i[k]) * D_oo_(fj, idx_i[k]);
                }
            }

            float cross = 0.0f;
            int a = p;
            while (a >= 0) {
                if (a < real_base) {
                    const int v = a;
                    const auto vp = static_cast<std::size_t>(v);
                    float a0 = 0.0f;
                    if (cv.q_layer_major) {
                        a0 = cv.scales_root[0] * static_cast<float>(cv.q_layer_major[v]);
                    } else {
                        a0 = cv.virt_a0[vp];
                    }
                    float dot0 = 0.0f;
                    for (int k = 0; k < m; ++k) {
                        dot0 += b_i[k] * c0_dot_one[static_cast<std::size_t>(idx_i[k])];
                    }
                    cross += a0 * dot0;
                    for (int l = 1; l < m; ++l) {
                    const int code = ReadSmallCode(vcodes, vp, m_codes, l - 1);
                        const int idx_rs = offsets_root_small_[static_cast<std::size_t>(l)] + code;
                        float al = 0.0f;
                        if (cv.q_layer_major) {
                            const std::size_t qidx =
                                static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                static_cast<std::size_t>(v);
                            al = cv.scales_root[l] * static_cast<float>(cv.q_layer_major[qidx]);
                        } else {
                            al = cv.virt_coeffs_small[vp * static_cast<std::size_t>(m_codes) +
                                                      static_cast<std::size_t>(l - 1)];
                        }
                        float acc = 0.0f;
                        for (int k = 0; k < m; ++k) {
                            acc += b_i[k] * D_ro_(idx_rs, idx_i[k]);
                        }
                        cross += al * acc;
                    }
                    break;
                }
                const int a_real = a - real_base;
                if (a_real >= 0 && a_real < n_root_real) {
                    const auto rp = static_cast<std::size_t>(a_real);
                    float a0 = 0.0f;
                    if (cv.q_layer_major) {
                        a0 = cv.scales_root[0] * static_cast<float>(cv.q_layer_major[a]);
                    } else {
                        a0 = cv.a0[rp];
                    }
                    float dot0 = 0.0f;
                    for (int k = 0; k < m; ++k) {
                        dot0 += b_i[k] * c0_dot_one[static_cast<std::size_t>(idx_i[k])];
                    }
                    cross += a0 * dot0;
                    for (int l = 1; l < m; ++l) {
                    const int code = ReadSmallCode(codes, rp, m_codes, l - 1);
                        const int idx_rs = offsets_root_small_[static_cast<std::size_t>(l)] + code;
                        float al = 0.0f;
                        if (cv.q_layer_major) {
                            const std::size_t qidx =
                                static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) +
                                static_cast<std::size_t>(a);
                            al = cv.scales_root[l] * static_cast<float>(cv.q_layer_major[qidx]);
                        } else {
                            al = cv.coeffs_small[rp * static_cast<std::size_t>(m_codes) +
                                                 static_cast<std::size_t>(l - 1)];
                        }
                        float acc = 0.0f;
                        for (int k = 0; k < m; ++k) {
                            acc += b_i[k] * D_ro_(idx_rs, idx_i[k]);
                        }
                        cross += al * acc;
                    }
                    break;
                }

                if (a_real < 0 || a_real >= n_real) {
                    break;
                }
                const auto up = static_cast<std::size_t>(a_real);
                idx_u[0] = meta_one_.offsets[0] + ReadCode0One(code0_one, up);
                if (cv.q_layer_major) {
                    b_u[0] = cv.scales_linkage[0] * static_cast<float>(cv.q_layer_major[a]);
                } else {
                    b_u[0] = cv.a0[up];
                }
                for (int l = 1; l < m; ++l) {
                const int code = ReadSmallCode(codes, up, m_codes, l - 1);
                    idx_u[l] = meta_one_.offsets[static_cast<std::size_t>(l)] + code;
                    if (cv.q_layer_major) {
                        const std::size_t qidx =
                            static_cast<std::size_t>(l) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(a);
                        b_u[l] = cv.scales_linkage[l] * static_cast<float>(cv.q_layer_major[qidx]);
                    } else {
                        b_u[l] = cv.coeffs_small[up * static_cast<std::size_t>(m_codes) +
                                                 static_cast<std::size_t>(l - 1)];
                    }
                }

                float dot = 0.0f;
                for (int j = 0; j < m; ++j) {
                    const float aj = b_u[j];
                    const int fj = idx_u[j];
                    for (int k = 0; k < m; ++k) {
                        dot += (aj * b_i[k]) * D_oo_(fj, idx_i[k]);
                    }
                }
                cross += dot;
                a = static_cast<int>(parent_1based[up]) - 1;
            }

            float parent_norm = 0.0f;
            if (p >= 0) {
                if (p < real_base) {
                    parent_norm = norm2_virt[static_cast<std::size_t>(p)];
                } else {
                    parent_norm = norm2_real[static_cast<std::size_t>(p - real_base)];
                }
            }
            norm2_real[ppos] = parent_norm + res_norm2 + 2.0f * cross;
        }

        *out_r_norm2 = std::move(norm2_real);
        return true;
    };

    return cv.parent_is_u16 ? run_with_parent(cv.parent_1based_u16)
                            : run_with_parent(cv.parent_1based);
}

}  // namespace stlq::eval
