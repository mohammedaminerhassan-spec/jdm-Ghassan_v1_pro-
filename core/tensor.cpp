#include "core/tensor.h"
#include "core/device.h"

#include <cstring>
#include <numeric>
#include <sstream>

namespace gai {

const char* device_name(Device d) {
    switch (d) {
        case Device::CPU:    return "cpu";
        case Device::CUDA:   return "cuda";
    }
    return "unknown";
}

i64 numel_of(const std::vector<i64>& shape) {
    // DeepSeek-style hardening: signed overflow is UB and would allocate a
    // tiny buffer then OOB-write. Saturate with a fail-fast cap (1T elements
    // is far above any T4-safe tensor: 4TB f32).
    static constexpr i64 kMaxNumel = (1LL << 40);
    i64 n = 1;
    for (i64 s : shape) {
        GAI_CHECK(s >= 0, "negative dimension");
        GAI_CHECK(s <= kMaxNumel, "dimension exceeds 1T (likely corrupt shape)");
        if (s != 0) GAI_CHECK(n <= kMaxNumel / s, "tensor numel overflow (>1T elements)");
        n *= s;
    }
    return n;
}

// ---------------------------------------------------------------- Storage
Storage::Storage(size_t nbytes, Device dev, DType dt) : nbytes_(nbytes), device_(dev) {
    if (nbytes == 0) return;
    ptr_   = device_alloc(nbytes, dev, dt);
    owned_ = true;
}

Storage::Storage(void* ptr, size_t nbytes, Device dev, std::shared_ptr<void> owner)
    : ptr_(ptr), nbytes_(nbytes), device_(dev), owned_(false),
      external_(true), owner_(std::move(owner)) {
    GAI_CHECK(ptr_ != nullptr || nbytes_ == 0, "wrap of null pointer");
    GAI_CHECK(dev == Device::CPU, "external views are CPU-readable memory only");
}

Storage::~Storage() {
    if (owned_ && ptr_) device_free(ptr_, device_);
    ptr_ = nullptr;
}

// ---------------------------------------------------------------- Tensor
Tensor::Tensor(std::vector<i64> shape, DType dt, Device dev)
    : shape_(std::move(shape)), dtype_(dt), device_(dev) {
    numel_   = numel_of(shape_);
    size_t nb = dtype_nbytes(dt, static_cast<size_t>(numel_));
    storage_ = std::make_shared<Storage>(nb, dev, dt);
}

Tensor Tensor::zeros(std::vector<i64> shape, DType dt, Device dev) {
    Tensor t(std::move(shape), dt, dev);
    t.zero_();
    return t;
}

Tensor Tensor::empty(std::vector<i64> shape, DType dt, Device dev) {
    return Tensor(std::move(shape), dt, dev);
}

Tensor Tensor::wrap_external(std::vector<i64> shape, DType dt,
                             void* ptr, size_t nbytes,
                             std::shared_ptr<void> owner) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.numel_ = numel_of(t.shape_);
    const size_t expect = dtype_nbytes(dt, static_cast<size_t>(t.numel_));
    GAI_CHECK(nbytes == expect, "wrap_external size mismatch (file corrupt or wrong dtype)");
    t.dtype_  = dt;
    t.device_ = Device::CPU;
    t.storage_ = std::make_shared<Storage>(ptr, nbytes, Device::CPU, std::move(owner));
    return t;
}

Tensor Tensor::view(std::vector<i64> shape) const {
    GAI_CHECK(defined(), "view of undefined tensor");
    Tensor t;
    t.storage_ = storage_;
    t.shape_   = std::move(shape);
    t.numel_   = numel_of(t.shape_);
    t.dtype_   = dtype_;
    t.device_  = device_;
    t.name_    = name_;
    GAI_CHECK(t.numel_ == numel_, "view must preserve element count");
    return t;
}

Tensor Tensor::to(Device dev) const {
    if (!defined()) return {};
    if (dev == device_) return *this;
    Tensor out(shape_, dtype_, dev);
    device_copy(out.data_ptr(), dev, data_ptr(), device_, nbytes());
    out.name_ = name_;
    return out;
}

Tensor Tensor::clone() const {
    if (!defined()) return {};
    Tensor out(shape_, dtype_, device_);
    device_copy(out.data_ptr(), device_, data_ptr(), device_, nbytes());
    out.name_ = name_;
    return out;
}

void Tensor::zero_() {
    if (!defined() || nbytes() == 0) return;
    // External views are read-only by contract (mmap pages are PAGE_READONLY).
    // Faulting here would be a SIGSEGV; fail fast with file:line instead.
    GAI_CHECK(!is_external(), "zero_ on read-only external view (mmap weight?)");
    // Name the tensor in any device failure: a bare "cudaMemset: invalid
    // argument" is unactionable (which of the 20+ buffers? which device?).
    try {
        device_memset_zero(data_ptr(), nbytes(), device_);
    } catch (const std::exception& e) {
        throw Error(std::string(e.what()) + " | tensor='" + (name_.empty() ? "<unnamed>" : name_) +
                    "' shape=" + shape_str() + " dev=" + device_name(device_) +
                    " bytes=" + std::to_string(nbytes()));
    }
}

void Tensor::copy_from(const Tensor& src) {
    GAI_CHECK(defined() && src.defined(), "copy_from on undefined tensor");
    GAI_CHECK(src.dtype() == dtype_, "copy_from dtype mismatch");
    GAI_CHECK(src.numel() == numel_, "copy_from numel mismatch");
    GAI_CHECK(src.nbytes() == nbytes(), "copy_from nbytes mismatch (quant block misalign?)");
    GAI_CHECK(!is_external(), "copy_from into read-only external view (mmap weight?)");
    device_copy(data_ptr(), device_, src.data_ptr(), src.device(), nbytes());
}

std::string Tensor::shape_str() const {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i) ss << ", ";
        ss << shape_[i];
    }
    ss << "]";
    return ss.str();
}

std::string Tensor::describe() const {
    std::ostringstream ss;
    ss << (name_.empty() ? "tensor" : name_) << " " << shape_str()
       << " " << dtype_name(dtype_) << " " << device_name(device_)
       << " " << human_bytes(nbytes());
    return ss.str();
}

} // namespace gai
