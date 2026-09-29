#pragma once

#include "core/dtype.h"
#include <memory>
#include <vector>
#include <string>
#include <initializer_list>

namespace gai {

enum class Device : u32 { CPU = 0, CUDA = 1 };

const char* device_name(Device d);

class Storage {
public:
    Storage() = default;
    Storage(size_t nbytes, Device dev, DType dt = DType::F32);

    Storage(void* ptr, size_t nbytes, Device dev, std::shared_ptr<void> owner);
    ~Storage();

    Storage(const Storage&)            = delete;
    Storage& operator=(const Storage&) = delete;

    void*       data()       { return ptr_; }
    const void* data() const { return ptr_; }
    size_t      nbytes() const { return nbytes_; }
    Device      device() const { return device_; }
    bool        external() const { return external_; }

private:
    void*  ptr_    = nullptr;
    size_t nbytes_ = 0;
    Device device_ = Device::CPU;
    bool   owned_  = false;
    bool   external_ = false;
    std::shared_ptr<void> owner_;
};

using StoragePtr = std::shared_ptr<Storage>;

class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<i64> shape, DType dt, Device dev = Device::CPU);
    Tensor(std::initializer_list<i64> shape, DType dt, Device dev = Device::CPU)
        : Tensor(std::vector<i64>(shape), dt, dev) {}

    static Tensor zeros(std::vector<i64> shape, DType dt = DType::F32, Device dev = Device::CPU);
    static Tensor empty(std::vector<i64> shape, DType dt = DType::F32, Device dev = Device::CPU);

    static Tensor wrap_external(std::vector<i64> shape, DType dt,
                                void* ptr, size_t nbytes,
                                std::shared_ptr<void> owner);
    bool is_external() const { return storage_ && storage_->external(); }

    bool   defined() const { return storage_ != nullptr; }
    DType  dtype()   const { return dtype_; }
    Device device()  const { return device_; }
    const std::vector<i64>& shape() const { return shape_; }
    int    ndim()  const { return static_cast<int>(shape_.size()); }
    i64    dim(int i) const { return shape_[static_cast<size_t>(i)]; }
    i64    numel() const { return numel_; }
    size_t nbytes() const { return storage_ ? storage_->nbytes() : 0; }
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }

    void* data_ptr() { return storage_ ? storage_->data() : nullptr; }
    const void* data_ptr() const { return storage_ ? storage_->data() : nullptr; }

    template <typename T> T* ptr() { return reinterpret_cast<T*>(data_ptr()); }
    template <typename T> const T* ptr() const { return reinterpret_cast<const T*>(data_ptr()); }

    float*       f32()       { return ptr<float>(); }
    const float* f32() const { return ptr<float>(); }
    i32*         i32p()       { return ptr<i32>(); }
    const i32*   i32p() const { return ptr<i32>(); }

    Tensor view(std::vector<i64> shape) const;

    Tensor to(Device dev) const;
    Tensor clone() const;
    void   zero_();
    void   copy_from(const Tensor& src);

    std::string shape_str() const;
    std::string describe() const;

private:
    StoragePtr       storage_;
    std::vector<i64> shape_;
    i64              numel_ = 0;
    DType            dtype_ = DType::F32;
    Device           device_ = Device::CPU;
    std::string      name_;
};

i64 numel_of(const std::vector<i64>& shape);

}
