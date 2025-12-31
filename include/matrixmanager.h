// matrix_manager.h
#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <torch/extension.h>
#include <pybind11/numpy.h>
#include "common.h"

class MatrixManager {
public:
    MatrixManager(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), h_data_(nullptr), d_data_(nullptr)
    {
        h_data_ = new DataType[rows * cols];
        CHECK_CUDA(cudaMalloc(&d_data_, rows * cols * sizeof(DataType)));
    }

    ~MatrixManager() {
        delete[] h_data_;
        if (d_data_) cudaFree(d_data_);
    }

    // 从 NumPy 数组拷贝数据到 host/device
    void copy_from_numpy(pybind11::array_t<float> array) {
        auto buf = array.request();
        if (buf.ndim != 2 || buf.shape[0] != rows_ || buf.shape[1] != cols_)
            throw std::runtime_error("Shape mismatch in copy_from_numpy");
        std::memcpy(h_data_, buf.ptr, rows_ * cols_ * sizeof(DataType));
        CHECK_CUDA(cudaMemcpy(d_data_, h_data_, rows_ * cols_ * sizeof(DataType), cudaMemcpyHostToDevice));
    }

    // 拷贝数据到 NumPy
    pybind11::array_t<float> to_numpy() const {
        pybind11::array_t<float> array({rows_, cols_});
        auto buf = array.request();
        CHECK_CUDA(cudaMemcpy(const_cast<DataType*>(static_cast<const DataType*>(buf.ptr)),
                              d_data_, rows_ * cols_ * sizeof(DataType),
                              cudaMemcpyDeviceToHost));
        return array;
    }

    // 从 torch tensor 拷贝数据到 host/device
    void copy_from_torch(const torch::Tensor& tensor) {
        if (!tensor.is_contiguous() || tensor.dim() != 2)
            throw std::runtime_error("Tensor must be 2D and contiguous");
        if (tensor.size(0) != rows_ || tensor.size(1) != cols_)
            throw std::runtime_error("Shape mismatch in copy_from_torch");

        auto cpu_tensor = tensor.contiguous().to(torch::kCPU);
        std::memcpy(h_data_, cpu_tensor.data_ptr<DataType>(), rows_ * cols_ * sizeof(DataType));
        CHECK_CUDA(cudaMemcpy(d_data_, h_data_, rows_ * cols_ * sizeof(DataType), cudaMemcpyHostToDevice));
    }

    // 转换到 torch tensor
    torch::Tensor to_torch() const {
        auto tensor = torch::empty({(int64_t)rows_, (int64_t)cols_}, torch::kDataType32);
        CHECK_CUDA(cudaMemcpy(tensor.data_ptr<DataType>(), d_data_, rows_ * cols_ * sizeof(DataType),
                              cudaMemcpyDeviceToHost));
        return tensor;
    }

    DataType* host_data() { return h_data_; }
    DataType* device_data() { return d_data_; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

private:
    size_t rows_, cols_;
    DataType* h_data_;
    DataType* d_data_;
};
