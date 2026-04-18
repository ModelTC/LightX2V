#pragma once

#include <functional>
#include <torch/extension.h>

#include <c10/xpu/XPUStream.h>

namespace utils {
    static inline sycl::queue& get_queue(const torch::Device& device) {
        return c10::xpu::getCurrentXPUStream(device.index()).queue();
    }

    static inline sycl::event submit_kernel(std::function<void(sycl::handler&)> kernel, const at::Device& device, const char * desc = nullptr) {
        sycl::queue& queue = get_queue(device);
        sycl::event event = queue.submit(kernel);
        return event;
    }
}
