#pragma once

#ifdef BUILD_ESIMD_KERNEL_LIB
  #define ESIMD_KERNEL_API __declspec(dllexport)
#else
  #define ESIMD_KERNEL_API __declspec(dllimport)
#endif
