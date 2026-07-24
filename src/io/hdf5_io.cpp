#include "stlq/io/hdf5_io.h"

#include <hdf5.h>

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "stlq/common/logger.h"

namespace stlq::io {

namespace {

Hdf5MatrixLayout g_default_layout = Hdf5MatrixLayout::kJulia;

bool WriteLayoutAttribute(hid_t dataset, const char* layout) {
    hid_t attr_space = H5Screate(H5S_SCALAR);
    if (attr_space < 0) {
        return false;
    }
    hid_t attr_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(attr_type, std::strlen(layout));
    hid_t attr = H5Acreate2(dataset, "layout", attr_type, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(attr_space);
        H5Tclose(attr_type);
        return false;
    }
    herr_t status = H5Awrite(attr, attr_type, layout);
    H5Aclose(attr);
    H5Sclose(attr_space);
    H5Tclose(attr_type);
    return status >= 0;
}

bool ReadLayoutAttribute(hid_t dataset, std::string* out) {
    if (!out) {
        return false;
    }
    out->clear();
    if (H5Aexists(dataset, "layout") <= 0) {
        return false;
    }
    hid_t attr = H5Aopen(dataset, "layout", H5P_DEFAULT);
    if (attr < 0) {
        return false;
    }
    hid_t type = H5Aget_type(attr);
    if (type < 0) {
        H5Aclose(attr);
        return false;
    }
    const auto size = static_cast<std::size_t>(H5Tget_size(type));
    std::string buf(size, '\0');
    herr_t status = H5Aread(attr, type, buf.data());
    H5Tclose(type);
    H5Aclose(attr);
    if (status < 0) {
        return false;
    }
    while (!buf.empty() && (buf.back() == '\0' || buf.back() == ' ')) {
        buf.pop_back();
    }
    *out = buf;
    return !out->empty();
}

Hdf5MatrixLayout LayoutFromStringOrDefault(const std::string& s, Hdf5MatrixLayout fallback) {
    if (s == "julia") {
        return Hdf5MatrixLayout::kJulia;
    }
    if (s == "col-major" || s == "cxx") {
        return Hdf5MatrixLayout::kCxx;
    }
    return fallback;
}

}  // namespace

void SetDefaultHdf5MatrixLayout(Hdf5MatrixLayout layout) { g_default_layout = layout; }
Hdf5MatrixLayout GetDefaultHdf5MatrixLayout() { return g_default_layout; }

struct Hdf5Writer::Impl {
    hid_t file = -1;
};

struct Hdf5Reader::Impl {
    hid_t file = -1;
};

Hdf5Writer::Hdf5Writer(const std::string& path) : impl_(std::make_unique<Impl>()) {
    matrix_layout_ = GetDefaultHdf5MatrixLayout();
    impl_->file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (impl_->file < 0) {
        LogError("Failed to create HDF5 file: " + path);
    }
}

Hdf5Writer::~Hdf5Writer() {
    if (impl_ && impl_->file >= 0) {
        H5Fclose(impl_->file);
    }
}

Hdf5Writer::Hdf5Writer(Hdf5Writer&&) noexcept = default;
Hdf5Writer& Hdf5Writer::operator=(Hdf5Writer&&) noexcept = default;

bool Hdf5Writer::IsOpen() const {
    return impl_ && impl_->file >= 0;
}

bool Hdf5Writer::EnsureGroups(const std::string& name) const {
    std::size_t pos = name.find_last_of('/');
    if (pos == std::string::npos || pos == 0) {
        return true;
    }
    std::string path = name.substr(0, pos);
    std::string current;
    std::size_t start = 0;
    while (start < path.size()) {
        std::size_t end = path.find('/', start);
        if (end == std::string::npos) {
            end = path.size();
        }
        std::string token = path.substr(start, end - start);
        if (!token.empty()) {
            current += "/" + token;
            if (H5Lexists(impl_->file, current.c_str(), H5P_DEFAULT) <= 0) {
                hid_t group = H5Gcreate2(impl_->file, current.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (group < 0) {
                    return false;
                }
                H5Gclose(group);
            }
        }
        start = end + 1;
    }
    return true;
}

bool Hdf5Writer::WriteMatrix(const std::string& name, const ColMajorMatrix<float>& matrix) {
    if (!EnsureGroups(name)) {
        return false;
    }
    const bool julia = (matrix_layout_ == Hdf5MatrixLayout::kJulia);
    hsize_t dims[2] = {static_cast<hsize_t>(julia ? matrix.cols : matrix.rows),
                       static_cast<hsize_t>(julia ? matrix.rows : matrix.cols)};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dataset = H5Dcreate2(impl_->file, name.c_str(), H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix.data.data());
    WriteLayoutAttribute(dataset, julia ? "julia" : "col-major");
    H5Dclose(dataset);
    H5Sclose(space);
    return status >= 0;
}

bool Hdf5Writer::WriteMatrix(const std::string& name, const ColMajorMatrix<Index>& matrix) {
    if (!EnsureGroups(name)) {
        return false;
    }
    const bool julia = (matrix_layout_ == Hdf5MatrixLayout::kJulia);
    hsize_t dims[2] = {static_cast<hsize_t>(julia ? matrix.cols : matrix.rows),
                       static_cast<hsize_t>(julia ? matrix.rows : matrix.cols)};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dataset = H5Dcreate2(impl_->file, name.c_str(), H5T_NATIVE_UINT32, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix.data.data());
    WriteLayoutAttribute(dataset, julia ? "julia" : "col-major");
    H5Dclose(dataset);
    H5Sclose(space);
    return status >= 0;
}

bool Hdf5Writer::WriteMatrix(const std::string& name, const ColMajorMatrix<Code>& matrix) {
    if (!EnsureGroups(name)) {
        return false;
    }
    const bool julia = (matrix_layout_ == Hdf5MatrixLayout::kJulia);
    hsize_t dims[2] = {static_cast<hsize_t>(julia ? matrix.cols : matrix.rows),
                       static_cast<hsize_t>(julia ? matrix.rows : matrix.cols)};
    hid_t space = H5Screate_simple(2, dims, nullptr);

    // Store as uint8 when possible (max_h<=256), else uint16.
    Code max_val = 0;
    for (Code v : matrix.data) {
        if (v > max_val) {
            max_val = v;
        }
    }
    const bool as_u8 = (max_val <= static_cast<Code>(std::numeric_limits<std::uint8_t>::max()));
    const hid_t dtype = as_u8 ? H5T_NATIVE_UINT8 : H5T_NATIVE_UINT16;

    hid_t dataset = H5Dcreate2(impl_->file, name.c_str(), dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }

    herr_t status = -1;
    if (as_u8) {
        std::vector<std::uint8_t> tmp(matrix.data.size());
        for (std::size_t i = 0; i < tmp.size(); ++i) {
            tmp[i] = static_cast<std::uint8_t>(matrix.data[i]);
        }
        status = H5Dwrite(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data());
    } else {
        status = H5Dwrite(dataset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix.data.data());
    }
    WriteLayoutAttribute(dataset, julia ? "julia" : "col-major");
    H5Dclose(dataset);
    H5Sclose(space);
    return status >= 0;
}

// NOTE: FullCode == Code (both uint8_t), so the FullCode overload has been removed.
// The Code overload above handles both types.

bool Hdf5Writer::WriteVector(const std::string& name, const std::vector<Index>& vec) {
    if (!EnsureGroups(name)) {
        return false;
    }
    hsize_t dims[1] = {static_cast<hsize_t>(vec.size())};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dataset = H5Dcreate2(impl_->file, name.c_str(), H5T_NATIVE_UINT32, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
    H5Dclose(dataset);
    H5Sclose(space);
    return status >= 0;
}

bool Hdf5Writer::WriteVector(const std::string& name, const std::vector<std::uint8_t>& vec) {
    if (!EnsureGroups(name)) {
        return false;
    }
    hsize_t dims[1] = {static_cast<hsize_t>(vec.size())};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dataset = H5Dcreate2(impl_->file, name.c_str(), H5T_NATIVE_UINT8, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
    H5Dclose(dataset);
    H5Sclose(space);
    return status >= 0;
}

bool Hdf5Writer::WriteVector(const std::string& name, const std::vector<int>& vec) {
    if (!EnsureGroups(name)) {
        return false;
    }
    hsize_t dims[1] = {static_cast<hsize_t>(vec.size())};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dataset = H5Dcreate2(impl_->file, name.c_str(), H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
    H5Dclose(dataset);
    H5Sclose(space);
    return status >= 0;
}

bool Hdf5Writer::WriteScalar(const std::string& name, int value) {
    if (!EnsureGroups(name)) {
        return false;
    }
    hsize_t dims[1] = {1};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dataset = H5Dcreate2(impl_->file, name.c_str(), H5T_NATIVE_INT, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dataset);
    H5Sclose(space);
    return status >= 0;
}

bool Hdf5Writer::WriteScalar(const std::string& name, Index value) {
    if (!EnsureGroups(name)) {
        return false;
    }
    hsize_t dims[1] = {1};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dataset = H5Dcreate2(impl_->file, name.c_str(), H5T_NATIVE_UINT32, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dataset);
    H5Sclose(space);
    return status >= 0;
}

bool Hdf5Writer::WriteScalar(const std::string& name, float value) {
    if (!EnsureGroups(name)) {
        return false;
    }
    hsize_t dims[1] = {1};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dataset = H5Dcreate2(impl_->file, name.c_str(), H5T_NATIVE_FLOAT, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        return false;
    }
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dataset);
    H5Sclose(space);
    return status >= 0;
}

bool Hdf5Writer::WriteString(const std::string& name, const std::string& value) {
    if (!EnsureGroups(name)) {
        return false;
    }
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, std::max<std::size_t>(1, value.size()));
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t dataset = H5Dcreate2(impl_->file, name.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        H5Tclose(type);
        return false;
    }
    herr_t status = H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.c_str());
    H5Dclose(dataset);
    H5Sclose(space);
    H5Tclose(type);
    return status >= 0;
}

}  // namespace stlq::io

namespace stlq::io {

namespace {

bool ReadDims(hid_t dataset, std::vector<hsize_t>* dims) {
    hid_t space = H5Dget_space(dataset);
    if (space < 0) {
        return false;
    }
    int ndims = H5Sget_simple_extent_ndims(space);
    if (ndims < 0) {
        H5Sclose(space);
        return false;
    }
    dims->resize(static_cast<std::size_t>(ndims));
    if (ndims > 0) {
        if (H5Sget_simple_extent_dims(space, dims->data(), nullptr) < 0) {
            H5Sclose(space);
            return false;
        }
    }
    H5Sclose(space);
    return true;
}

template <typename T>
bool ReadScalarImpl(hid_t file, const std::string& name, hid_t dtype, T* out) {
    hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        return false;
    }
    std::vector<hsize_t> dims;
    bool ok = ReadDims(dataset, &dims);
    if (!ok) {
        H5Dclose(dataset);
        return false;
    }
    // Accept scalar dataspace or 1-element vector.
    if (!(dims.empty() || (dims.size() == 1 && dims[0] == 1))) {
        H5Dclose(dataset);
        return false;
    }
    herr_t status = H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
    H5Dclose(dataset);
    return status >= 0;
}

}  // namespace

Hdf5Reader::Hdf5Reader(const std::string& path) : impl_(std::make_unique<Impl>()) {
    default_layout_ = GetDefaultHdf5MatrixLayout();
    impl_->file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (impl_->file < 0) {
        LogError("Failed to open HDF5 file: " + path);
    }
}

Hdf5Reader::~Hdf5Reader() {
    if (impl_ && impl_->file >= 0) {
        H5Fclose(impl_->file);
    }
}

Hdf5Reader::Hdf5Reader(Hdf5Reader&&) noexcept = default;
Hdf5Reader& Hdf5Reader::operator=(Hdf5Reader&&) noexcept = default;

bool Hdf5Reader::IsOpen() const { return impl_ && impl_->file >= 0; }

bool Hdf5Reader::Has(const std::string& name) const {
    if (!impl_ || impl_->file < 0) {
        return false;
    }
    return H5Lexists(impl_->file, name.c_str(), H5P_DEFAULT) > 0;
}

bool Hdf5Reader::ReadMatrix(const std::string& name, ColMajorMatrix<float>* matrix) {
    if (!matrix || impl_->file < 0) {
        return false;
    }
    hid_t dataset = H5Dopen2(impl_->file, name.c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        return false;
    }
    std::vector<hsize_t> dims;
    if (!ReadDims(dataset, &dims) || dims.size() != 2) {
        H5Dclose(dataset);
        return false;
    }

    std::string layout_attr;
    const Hdf5MatrixLayout layout =
        LayoutFromStringOrDefault(ReadLayoutAttribute(dataset, &layout_attr) ? layout_attr : "",
                                  default_layout_);
    const bool julia = (layout == Hdf5MatrixLayout::kJulia);

    const int rows = static_cast<int>(julia ? dims[1] : dims[0]);
    const int cols = static_cast<int>(julia ? dims[0] : dims[1]);
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data.resize(static_cast<std::size_t>(rows) * cols);
    herr_t status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix->data.data());
    H5Dclose(dataset);
    return status >= 0;
}

bool Hdf5Reader::ReadMatrix(const std::string& name, ColMajorMatrix<Index>* matrix) {
    if (!matrix || impl_->file < 0) {
        return false;
    }
    hid_t dataset = H5Dopen2(impl_->file, name.c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        return false;
    }
    std::vector<hsize_t> dims;
    if (!ReadDims(dataset, &dims) || dims.size() != 2) {
        H5Dclose(dataset);
        return false;
    }

    std::string layout_attr;
    const Hdf5MatrixLayout layout =
        LayoutFromStringOrDefault(ReadLayoutAttribute(dataset, &layout_attr) ? layout_attr : "",
                                  default_layout_);
    const bool julia = (layout == Hdf5MatrixLayout::kJulia);

    const int rows = static_cast<int>(julia ? dims[1] : dims[0]);
    const int cols = static_cast<int>(julia ? dims[0] : dims[1]);
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data.resize(static_cast<std::size_t>(rows) * cols);
    herr_t status = H5Dread(dataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix->data.data());
    H5Dclose(dataset);
    return status >= 0;
}

bool Hdf5Reader::ReadMatrix(const std::string& name, ColMajorMatrix<Code>* matrix) {
    if (!matrix || impl_->file < 0) {
        return false;
    }
    hid_t dataset = H5Dopen2(impl_->file, name.c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        return false;
    }
    std::vector<hsize_t> dims;
    if (!ReadDims(dataset, &dims) || dims.size() != 2) {
        H5Dclose(dataset);
        return false;
    }

    std::string layout_attr;
    const Hdf5MatrixLayout layout =
        LayoutFromStringOrDefault(ReadLayoutAttribute(dataset, &layout_attr) ? layout_attr : "",
                                  default_layout_);
    const bool julia = (layout == Hdf5MatrixLayout::kJulia);

    const int rows = static_cast<int>(julia ? dims[1] : dims[0]);
    const int cols = static_cast<int>(julia ? dims[0] : dims[1]);
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data.resize(static_cast<std::size_t>(rows) * cols);

    hid_t dtype = H5Dget_type(dataset);
    if (dtype < 0) {
        H5Dclose(dataset);
        return false;
    }
    const H5T_class_t cls = H5Tget_class(dtype);
    const H5T_sign_t sign = H5Tget_sign(dtype);
    const std::size_t sz = H5Tget_size(dtype);
    H5Tclose(dtype);

    if (cls != H5T_INTEGER || sign != H5T_SGN_NONE) {
        H5Dclose(dataset);
        return false;
    }

    herr_t status = -1;
    if (sz == 1) {
        std::vector<std::uint8_t> tmp(matrix->data.size());
        status = H5Dread(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data());
        if (status >= 0) {
            for (std::size_t i = 0; i < tmp.size(); ++i) {
                matrix->data[i] = static_cast<Code>(tmp[i]);
            }
        }
    } else if (sz == 2) {
        // Read u16 HDF5 data into a temp buffer, then narrow to Code (u8) with range check.
        std::vector<std::uint16_t> tmp(matrix->data.size());
        status = H5Dread(dataset, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data());
        if (status >= 0) {
            const auto maxv = static_cast<std::uint16_t>(std::numeric_limits<Code>::max());
            for (std::size_t i = 0; i < tmp.size(); ++i) {
                if (tmp[i] > maxv) {
                    H5Dclose(dataset);
                    return false;
                }
                matrix->data[i] = static_cast<Code>(tmp[i]);
            }
        }
    } else if (sz == 4) {
        std::vector<std::uint32_t> tmp(matrix->data.size());
        status = H5Dread(dataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data());
        if (status >= 0) {
            const auto maxv = static_cast<std::uint32_t>(std::numeric_limits<Code>::max());
            for (std::size_t i = 0; i < tmp.size(); ++i) {
                if (tmp[i] > maxv) {
                    H5Dclose(dataset);
                    return false;
                }
                matrix->data[i] = static_cast<Code>(tmp[i]);
            }
        }
    } else {
        H5Dclose(dataset);
        return false;
    }

    H5Dclose(dataset);
    return status >= 0;
}

// NOTE: FullCode == Code (both uint8_t), so the FullCode ReadMatrix overload has been removed.
// The Code overload above handles both types, including reading legacy u16 HDF5 files.

bool Hdf5Reader::ReadVector(const std::string& name, std::vector<Index>* vec) const {
    if (!vec || impl_->file < 0) {
        return false;
    }
    hid_t dataset = H5Dopen2(impl_->file, name.c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        return false;
    }
    std::vector<hsize_t> dims;
    if (!ReadDims(dataset, &dims) || dims.size() != 1) {
        H5Dclose(dataset);
        return false;
    }
    vec->resize(static_cast<std::size_t>(dims[0]));
    herr_t status = H5Dread(dataset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec->data());
    H5Dclose(dataset);
    return status >= 0;
}

bool Hdf5Reader::ReadVector(const std::string& name, std::vector<std::uint8_t>* vec) const {
    if (!vec || impl_->file < 0) {
        return false;
    }
    hid_t dataset = H5Dopen2(impl_->file, name.c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        return false;
    }
    std::vector<hsize_t> dims;
    if (!ReadDims(dataset, &dims) || dims.size() != 1) {
        H5Dclose(dataset);
        return false;
    }
    vec->resize(static_cast<std::size_t>(dims[0]));
    herr_t status = H5Dread(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec->data());
    H5Dclose(dataset);
    return status >= 0;
}

bool Hdf5Reader::ReadVector(const std::string& name, std::vector<int>* vec) const {
    if (!vec || impl_->file < 0) {
        return false;
    }
    hid_t dataset = H5Dopen2(impl_->file, name.c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        return false;
    }
    std::vector<hsize_t> dims;
    if (!ReadDims(dataset, &dims) || dims.size() != 1) {
        H5Dclose(dataset);
        return false;
    }
    vec->resize(static_cast<std::size_t>(dims[0]));
    herr_t status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec->data());
    H5Dclose(dataset);
    return status >= 0;
}

bool Hdf5Reader::ReadScalar(const std::string& name, int* value) const {
    if (!value || impl_->file < 0) {
        return false;
    }
    return ReadScalarImpl(impl_->file, name, H5T_NATIVE_INT, value);
}

bool Hdf5Reader::ReadScalar(const std::string& name, Index* value) const {
    if (!value || impl_->file < 0) {
        return false;
    }
    return ReadScalarImpl(impl_->file, name, H5T_NATIVE_UINT32, value);
}

bool Hdf5Reader::ReadScalar(const std::string& name, float* value) const {
    if (!value || impl_->file < 0) {
        return false;
    }
    return ReadScalarImpl(impl_->file, name, H5T_NATIVE_FLOAT, value);
}

bool Hdf5Reader::ReadString(const std::string& name, std::string* value) const {
    if (!value || impl_->file < 0) {
        return false;
    }
    hid_t dataset = H5Dopen2(impl_->file, name.c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        return false;
    }
    hid_t dtype = H5Dget_type(dataset);
    if (dtype < 0) {
        H5Dclose(dataset);
        return false;
    }
    const auto sz = static_cast<std::size_t>(H5Tget_size(dtype));
    std::string buf(sz, '\0');
    herr_t status = H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Tclose(dtype);
    H5Dclose(dataset);
    if (status < 0) {
        return false;
    }
    // Strings are stored without null terminator.
    *value = buf;
    return true;
}

}  // namespace stlq::io
