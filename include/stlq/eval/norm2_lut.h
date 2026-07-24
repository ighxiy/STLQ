#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace stlq::io {
class LinkageListReader;
}

namespace stlq::eval {

struct Norm2Lut {
    std::vector<float> centers;
    std::vector<std::uint8_t> codes_u8;
};

bool IsDiskNorm2ModeLut(const std::string& mode);
bool IsDiskNorm2ModeLutGlobal(const std::string& mode);
bool IsDiskNorm2ModeLutCluster(const std::string& mode);
bool IsDiskNorm2ModeLutSqrt(const std::string& mode);
bool IsDiskNorm2ModeLutLog1p(const std::string& mode);

bool BuildNorm2Lut(const float* data,
                   int n,
                   int requested_centers,
                   int max_iter,
                   Norm2Lut* out,
                   std::string* err);
bool BuildNorm2LutWithMode(const float* data,
                           int n,
                           int requested_centers,
                           int max_iter,
                           const std::string& mode,
                           double log1p_alpha,
                           double piecewise_p1,
                           double piecewise_p2,
                           double piecewise_count_weight,
                           double piecewise_range_weight,
                           Norm2Lut* out,
                           std::string* err);

enum class Norm2LutDiskLoadResult {
    kMissing = 0,
    kLoaded = 1,
    kInvalid = 2,
};

std::string DiskNorm2LutCodesFilename(bool use_coeff_codec);
std::string DiskNorm2LutCentersFilename(bool use_coeff_codec);
std::string DiskNorm2LutHashFilename(bool use_coeff_codec);
std::string DiskNorm2LutClusterCodesFilename(bool use_coeff_codec);
std::string DiskNorm2LutClusterCentersFilename(bool use_coeff_codec);
std::string DiskNorm2LutClusterCenterOffsetsFilename(bool use_coeff_codec);
std::string DiskNorm2LutClusterHashFilename(bool use_coeff_codec);

Norm2LutDiskLoadResult LoadNorm2LutCache(const io::LinkageListReader& linkage_list,
                                         bool use_coeff_codec,
                                         int requested_centers,
                                         int kmeans_niter,
                                         const std::string& mode,
                                         double log1p_alpha,
                                         double piecewise_p1,
                                         double piecewise_p2,
                                         double piecewise_count_weight,
                                         double piecewise_range_weight,
                                         Norm2Lut* out,
                                         std::string* err);
Norm2LutDiskLoadResult LoadNorm2LutCache(const io::LinkageListReader& linkage_list,
                                         bool use_coeff_codec,
                                         int requested_centers,
                                         int kmeans_niter,
                                         Norm2Lut* out,
                                         std::string* err);

Norm2LutDiskLoadResult CheckNorm2LutCache(const io::LinkageListReader& linkage_list,
                                          bool use_coeff_codec,
                                          int requested_centers,
                                          int kmeans_niter,
                                          const std::string& mode,
                                          double log1p_alpha,
                                          double piecewise_p1,
                                          double piecewise_p2,
                                          double piecewise_count_weight,
                                          double piecewise_range_weight,
                                          std::string* err);
Norm2LutDiskLoadResult CheckNorm2LutCache(const io::LinkageListReader& linkage_list,
                                          bool use_coeff_codec,
                                          int requested_centers,
                                          int kmeans_niter,
                                          std::string* err);

bool StoreNorm2LutCache(const io::LinkageListReader& linkage_list,
                        bool use_coeff_codec,
                        int requested_centers,
                        int kmeans_niter,
                        const std::string& mode,
                        double log1p_alpha,
                        double piecewise_p1,
                        double piecewise_p2,
                        double piecewise_count_weight,
                        double piecewise_range_weight,
                        const Norm2Lut& lut_cache,
                        std::string* err);
bool StoreNorm2LutCache(const io::LinkageListReader& linkage_list,
                        bool use_coeff_codec,
                        int requested_centers,
                        int kmeans_niter,
                        const Norm2Lut& lut_cache,
                        std::string* err);

bool LoadNorm2LutCenters(const io::LinkageListReader& linkage_list,
                         bool use_coeff_codec,
                         int requested_centers,
                         int kmeans_niter,
                         const std::string& mode,
                         double log1p_alpha,
                         double piecewise_p1,
                         double piecewise_p2,
                         double piecewise_count_weight,
                         double piecewise_range_weight,
                         std::vector<float>* out,
                         std::string* err);
bool LoadNorm2LutCenters(const io::LinkageListReader& linkage_list,
                         bool use_coeff_codec,
                         int requested_centers,
                         int kmeans_niter,
                         std::vector<float>* out,
                         std::string* err);

bool LoadNorm2LutClusterCodesSlice(const io::LinkageListReader& linkage_list,
                                   bool use_coeff_codec,
                                   int cid,
                                   std::vector<std::uint8_t>* out,
                                   std::string* err);

Norm2LutDiskLoadResult LoadNorm2LutClusterCache(const io::LinkageListReader& linkage_list,
                                                bool use_coeff_codec,
                                                int requested_centers,
                                                int kmeans_niter,
                                                std::unordered_map<int, Norm2Lut>* out,
                                                std::string* err);

Norm2LutDiskLoadResult CheckNorm2LutClusterCache(const io::LinkageListReader& linkage_list,
                                                 bool use_coeff_codec,
                                                 int requested_centers,
                                                 int kmeans_niter,
                                                 std::string* err);

bool StoreNorm2LutClusterCache(const io::LinkageListReader& linkage_list,
                               bool use_coeff_codec,
                               int requested_centers,
                               int kmeans_niter,
                               const std::unordered_map<int, Norm2Lut>& lut_cache,
                                std::string* err);

bool LoadFloatNorm2ClusterSlice(const io::LinkageListReader& linkage_list,
                                bool use_coeff_codec,
                                int cid,
                                std::vector<float>* out,
                                std::string* err);

bool LoadNorm2LutClusterSlice(const io::LinkageListReader& linkage_list,
                              bool use_coeff_codec,
                              int cid,
                              Norm2Lut* out,
                              std::string* err);

}  // namespace stlq::eval
