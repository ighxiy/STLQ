bool BuildPrecomp(const CodebookPack& codebooks, Precomp* precomp) {
    if (!precomp || codebooks.books.empty()) {
        return false;
    }
    const int m = static_cast<int>(codebooks.books.size());
    const int d = codebooks.books.front().rows;
    std::vector<int> h_vec;
    h_vec.reserve(m);
    int H = 0;
    for (const auto& book : codebooks.books) {
        h_vec.push_back(book.cols);
        H += book.cols;
    }

    precomp->C_all = ColMajorMatrix<float>(d, H);
    precomp->G = ColMajorMatrix<float>(H, H);
    precomp->offsets.resize(m);
    precomp->flat_layer.resize(H);
    precomp->invnorm_flat.resize(H);
    precomp->h_vec = h_vec;
    precomp->H = H;
    precomp->m = m;
    precomp->d = d;

    int col_offset = 0;
    for (int l = 0; l < m; ++l) {
        precomp->offsets[l] = col_offset;
        const auto& book = codebooks.books[l];
        for (int c = 0; c < book.cols; ++c) {
            const float* src = book.Col(c);
            float* dst = precomp->C_all.Col(col_offset + c);
            float norm_sq = 0.0f;
            for (int r = 0; r < d; ++r) {
                dst[r] = src[r];
                norm_sq += src[r] * src[r];
            }
            const float inv_norm = norm_sq > 0.0f ? 1.0f / std::sqrt(norm_sq) : 0.0f;
            precomp->flat_layer[col_offset + c] = l;
            precomp->invnorm_flat[col_offset + c] = inv_norm;
        }
        col_offset += book.cols;
    }

    {
        ScopedBlasThreads blas_scope(OmpMaxThreads());
        Gemm(true, false, 1.0f, precomp->C_all, precomp->C_all, 0.0f, &precomp->G);
    }
    // Bump a monotonic tag so cached backends can detect that the *contents* changed, even if
    // std::vector storage is reused across iterations.
    static std::uint64_t g_build_tag = 0;
    precomp->build_tag = ++g_build_tag;
    return true;
}
