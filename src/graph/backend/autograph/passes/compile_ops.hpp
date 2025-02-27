/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_PASSES_COMPILE_OPS_HPP
#define GRAPH_BACKEND_DNNL_PASSES_COMPILE_OPS_HPP

#include <memory>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#include "graph/interface/c_types_map.hpp"

#include "graph/backend/autograph/subgraph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace autograph_impl {

status_t compile_ops(std::shared_ptr<subgraph_t> &sg);

} // namespace autograph_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
