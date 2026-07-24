#pragma once

#include <algorithm>
#include <vector>

#include "stlq/common/types.h"

namespace stlq {

struct CodebookMeta {
    int d = 0;
    int m = 0;
    int total_cols = 0;
    int max_h = 0;
    std::vector<int> sizes;
    std::vector<int> offsets;
    std::vector<float*> ptrs;
    ColMajorMatrix<float> flat;
};

inline std::vector<const ColMajorMatrix<float>*> GatherBooks(const CodebookPack& pack) {
    std::vector<const ColMajorMatrix<float>*> out;
    out.reserve(pack.books.size());
    for (const auto& book : pack.books) {
        out.push_back(&book);
    }
    return out;
}

inline CodebookMeta BuildCodebookMeta(const std::vector<const ColMajorMatrix<float>*>& books) {
    CodebookMeta meta;
    if (books.empty()) {
        return meta;
    }

    meta.m = static_cast<int>(books.size());
    meta.d = books.front()->rows;
    meta.sizes.reserve(meta.m);
    meta.offsets.reserve(meta.m);
    meta.ptrs.reserve(meta.m);

    int offset = 0;
    for (const auto* book : books) {
        meta.sizes.push_back(book->cols);
        meta.offsets.push_back(offset);
        meta.ptrs.push_back(const_cast<float*>(book->data.data()));
        meta.max_h = std::max(meta.max_h, book->cols);
        offset += book->cols;
    }
    meta.total_cols = offset;
    meta.flat = ColMajorMatrix<float>(meta.d, meta.total_cols);

    int col = 0;
    for (const auto* book : books) {
        for (int c = 0; c < book->cols; ++c) {
            const float* src = book->Col(c);
            float* dst = meta.flat.Col(col + c);
            std::copy(src, src + meta.d, dst);
        }
        col += book->cols;
    }
    return meta;
}

}  // namespace stlq

