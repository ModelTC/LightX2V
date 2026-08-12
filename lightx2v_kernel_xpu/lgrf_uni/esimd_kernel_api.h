#pragma once

#if defined(_WIN32) && defined(BUILD_ESIMD_KERNEL_LIB)
  #define ESIMD_KERNEL_API __declspec(dllexport)
#elif defined(_WIN32)
  #define ESIMD_KERNEL_API __declspec(dllimport)
#elif defined(BUILD_ESIMD_KERNEL_LIB)
  #define ESIMD_KERNEL_API __attribute__((visibility("default")))
#else
  #define ESIMD_KERNEL_API
#endif
