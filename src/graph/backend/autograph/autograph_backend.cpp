/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <utility>

#include "graph/utils/any.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/autograph/autograph_backend.hpp"
#include "graph/backend/autograph/autograph_opset.hpp"
#include "graph/backend/autograph/kernels/kernels.hpp"
#include "graph/backend/autograph/patterns/fusions.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace autograph_impl {

bool dnnl_layout_id_manager_t::is_mem_desc_equal(
        const graph::utils::any_t &mem_desc1,
        const graph::utils::any_t &mem_desc2) const {
    auto &md1 = graph::utils::any_cast<const memory::desc &>(mem_desc1);
    auto &md2 = graph::utils::any_cast<const memory::desc &>(mem_desc2);
    return md1 == md2;
}

autograph_backend::autograph_backend(const std::string &name, float priority)
    : backend(name, priority) {
    register_op_schemas();
    register_passes();
}

bool autograph_backend::register_op_schemas() {
    register_dnnl_opset_schema();
    return true;
}

bool autograph_backend::register_passes() {
#define DNNL_BACKEND_REGISTER_PATTERN_CALL(pattern_class_, pattern_registry_) \
    pattern::register_##pattern_class_(pattern_registry_);

    DNNL_BACKEND_REGISTER_PATTERN_CALL(binary_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(bn_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(concat_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(conv_block_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(conv_post_ops_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(convtranspose_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(matmul_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(single_op_pass, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(pool_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(eltwise_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(quantize_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(interpolate_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(softmax_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(layernorm_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(sum_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(reorder_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(shuffle_fusion, pass_registry_);
    DNNL_BACKEND_REGISTER_PATTERN_CALL(reduction_fusion, pass_registry_);
    pass_registry_.sort_passes();

#undef DNNL_BACKEND_REGISTER_PATTERN_CALL

    return true;
}

size_t autograph_backend::get_mem_size(const logical_tensor_t &lt) const {
    auto md = make_dnnl_memory_desc(lt);
    return md.get_size();
}

bool autograph_backend::compare_logical_tensor(
        const logical_tensor_t &lhs, const logical_tensor_t &rhs) const {
    auto md1 = make_dnnl_memory_desc(lhs);
    auto md2 = make_dnnl_memory_desc(rhs);
    return md1 == md2;
}

graph::utils::optional_t<size_t> autograph_backend::set_mem_desc(
        const graph::utils::any_t &mem_desc) {
    return layout_id_manager_.set_mem_desc(mem_desc);
}

graph::utils::optional_t<graph::utils::any_t> autograph_backend::get_mem_desc(
        const size_t &layout_id) const {
    return layout_id_manager_.get_mem_desc(layout_id);
}

kernel_ptr large_partition_kernel_creator() {
    return std::make_shared<larger_partition_kernel_t>();
}

} // namespace autograph_impl

// This function should be called by backend_registry_t
void register_autograph_backend() {
    backend_registry_t::get_singleton().register_backend(
            &autograph_impl::autograph_backend::get_singleton());
}

} // namespace graph
} // namespace impl
} // namespace dnnl
