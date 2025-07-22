/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <tuple>
#include <vector>


#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)


#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }


/*
 * From csrc/gemm
 */
void scaled_nvfp4_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_sf, torch::Tensor const& input_sf);

void scaled_mxfp4_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_sf);

void scaled_mxfp6_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_sf);

void scaled_mxfp8_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_sf);

void cutlass_scaled_nvfp4_mm_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias);

void cutlass_scaled_mxfp4_mm_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias);

void cutlass_scaled_mxfp6_mxfp8_mm_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias);


void cutlass_scaled_mxfp8_mm_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias);
