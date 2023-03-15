/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_KERNELS_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_KERNELS_HPP

#include "graph/backend/autograph/kernels/batchnorm.hpp"
#include "graph/backend/autograph/kernels/binary.hpp"
#include "graph/backend/autograph/kernels/concat.hpp"
#include "graph/backend/autograph/kernels/conv.hpp"
#include "graph/backend/autograph/kernels/convtranspose.hpp"
#include "graph/backend/autograph/kernels/eltwise.hpp"
#include "graph/backend/autograph/kernels/large_partition.hpp"
#include "graph/backend/autograph/kernels/layernorm.hpp"
#include "graph/backend/autograph/kernels/logsoftmax.hpp"
#include "graph/backend/autograph/kernels/matmul.hpp"
#include "graph/backend/autograph/kernels/pool.hpp"
#include "graph/backend/autograph/kernels/prelu.hpp"
#include "graph/backend/autograph/kernels/quantize.hpp"
#include "graph/backend/autograph/kernels/reduction.hpp"
#include "graph/backend/autograph/kernels/reorder.hpp"
#include "graph/backend/autograph/kernels/resampling.hpp"
#include "graph/backend/autograph/kernels/shuffle.hpp"
#include "graph/backend/autograph/kernels/softmax.hpp"
#include "graph/backend/autograph/kernels/sum.hpp"

#endif
