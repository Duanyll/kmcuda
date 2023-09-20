#ifdef _DEBUG
#undef _DEBUG
#include <torch/extension.h>
#define _DEBUG 1
#else
#include <torch/extension.h>
#endif

#include <cstdint>
#include <iostream>
#include <string>

#include "kmcuda.h"

std::tuple<torch::Tensor, torch::Tensor> torch_kmeans_cuda(
    torch::Tensor samples, uint32_t clusters_size, torch::Tensor centroids,
    std::string init_str, float tolerance, float yinyang_t,
    std::string metric_str, uint32_t seed, int32_t verbosity) {
    bool has_centroids = centroids.numel() > 0;

    // Handle init
    KMCUDAInitMethod init;
    int32_t init_params = 0;  // For AFKMC2
    if (has_centroids) {
        init = KMCUDAInitMethod::kmcudaInitMethodImport;
    } else if (init_str == "random") {
        init = KMCUDAInitMethod::kmcudaInitMethodRandom;
    } else if (init_str == "kmeans++") {
        init = KMCUDAInitMethod::kmcudaInitMethodPlusPlus;
    } else if (init_str == "afkmc2") {
        init = KMCUDAInitMethod::kmcudaInitMethodAFKMC2;
    } else {
        TORCH_CHECK(false,
                    "init_str must be 'random', 'kmeans++', 'afkmc2', or "
                    "provide centroids");
    }

    // Handle metric
    KMCUDADistanceMetric metric;
    if (metric_str == "l2") {
        metric = KMCUDADistanceMetric::kmcudaDistanceMetricL2;
    } else if (metric_str == "cosine") {
        metric = KMCUDADistanceMetric::kmcudaDistanceMetricCosine;
    } else {
        TORCH_CHECK(false, "metric_str must be 'l2' or 'cosine'");
    }

    // Handle dtype
    int32_t fp16x2 = 0;
    if (samples.dtype() == torch::kHalf) {
        fp16x2 = 1;
    } else if (samples.dtype() == torch::kFloat) {
        fp16x2 = 0;
    } else {
        TORCH_CHECK(false, "samples must be float16 or float32");
    }

    // Handle samples
    TORCH_CHECK(samples.dim() == 2, "samples must be 2-dimensional");
    uint32_t samples_size = samples.size(0);
    TORCH_CHECK(samples.size(1) < 65536,
                "samples must have fewer than 65536 features");
    uint16_t features_size = fp16x2 ? samples.size(1) / 2 : samples.size(1);
    samples = samples.contiguous();

    // Handle device
    uint32_t device = 0;  // Use all GPUs
    int32_t device_ptrs = -1;
    if (samples.is_cpu()) {
        device_ptrs = -1;
    } else if (samples.is_cuda()) {
        device_ptrs = samples.device().index();
    } else {
        TORCH_CHECK(false, "samples must be on CPU or CUDA");
    }

    // Allocate centroids
    if (has_centroids) {
        TORCH_CHECK(centroids.dim() == 2, "centroids must be 2-dimensional");
        TORCH_CHECK(centroids.size(0) == clusters_size,
                    "centroids must have clusters_size rows");
        TORCH_CHECK(
            centroids.size(1) == samples.size(1),
            "centroids must have the same number of columns as samples");
        TORCH_CHECK(centroids.dtype() == samples.dtype(),
                    "centroids must have the same dtype as samples");
        TORCH_CHECK(centroids.device() == samples.device(),
                    "centroids must be on the same device as samples");
        centroids = centroids.contiguous();
    } else {
        centroids = torch::empty({clusters_size, samples.size(1)},
                                 samples.options().requires_grad(false));
    }

    // Allocate assignments
    auto assignments = torch::empty(
        {samples_size},
        samples.options().dtype(torch::kInt32).requires_grad(false));

    KMCUDAResult result = kmeans_cuda(
        init, &init_params, tolerance, yinyang_t, metric, samples_size,
        features_size, clusters_size, seed, device, device_ptrs, fp16x2,
        verbosity, (const float*)samples.data_ptr(),
        (float*)centroids.data_ptr(), (uint32_t*)assignments.data_ptr(),
        nullptr);

    TORCH_CHECK(result == KMCUDAResult::kmcudaSuccess,
                "kmeans_cuda failed with error code ", result);

    return std::make_tuple(std::move(assignments), std::move(centroids));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kmeans_cuda", &torch_kmeans_cuda, py::arg("samples"),
          py::arg("clusters_size"), py::arg("centroids") = torch::empty(0),
          py::arg("init") = "kmeans++", py::arg("tolerance") = 0.01f,
          py::arg("yinyang_t") = 0.1f, py::arg("metric") = "l2",
          py::arg("seed") = 42, py::arg("verbosity") = 0);
}