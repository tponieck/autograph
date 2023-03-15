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

#ifndef GRAPH_BACKEND_DNNL_DNNL_BACKEND_HPP
#define GRAPH_BACKEND_DNNL_DNNL_BACKEND_HPP

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/backend.hpp"
#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"

#include "graph/utils/any.hpp"
#include "graph/utils/pm/pass_manager.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/autograph/common.hpp"
#include "graph/backend/autograph/internal_ops.hpp"
#include "graph/backend/autograph/utils.hpp"

#ifdef DNNL_GRAPH_LAYOUT_DEBUG
#include "oneapi/dnnl/dnnl_debug.h"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace autograph_impl {

class dnnl_partition_impl_t;

class layout_id_manager_t {
public:
    layout_id_manager_t() = default;
    virtual ~layout_id_manager_t() = default;

    /*! \brief Set a backend memory descriptor to manager and get a
    * corresponding layout id
    * \param mem_desc The backend's memory descriptor, it can
    * be both plain or opaque
    * \return a cache index, will be used as layout id
    * \note This function should be invoked in every where we want to
    * convert a md to layout id
    */
    virtual graph::utils::optional_t<size_t> set_mem_desc(
            const graph::utils::any_t &mem_desc) {
        std::lock_guard<std::mutex> lock(mem_descs_.m_);

        auto pos
                = std::find_if(mem_descs_.data_.begin(), mem_descs_.data_.end(),
                        [&](const graph::utils::any_t &m) -> bool {
                            return is_mem_desc_equal(m, mem_desc);
                        });

        size_t layout_id;
        if (pos != mem_descs_.data_.end()) {
            layout_id = static_cast<size_t>(
                    std::distance(mem_descs_.data_.begin(), pos));
        } else {
            mem_descs_.data_.emplace_back(mem_desc);
            layout_id = static_cast<size_t>(mem_descs_.data_.size() - 1);
        }

        return layout_id;
    }

    /*! \brief Get a backend memory descriptor from manager by using a
    * layout id
    * \param layout_id The layout id, which is generated and managed
    * by backends
    * \return When the input is a valid cache index, the return value
    * is a cached memory descriptor; otherwise, the return value will
    * be a utils::nullopt
    */
    virtual graph::utils::optional_t<graph::utils::any_t> get_mem_desc(
            size_t layout_id) const {
        std::lock_guard<std::mutex> lock(mem_descs_.m_);
        if (layout_id >= mem_descs_.data_.size()) return graph::utils::nullopt;
        return mem_descs_.data_[layout_id];
    }

protected:
    mutable struct {
        std::vector<graph::utils::any_t> data_;
        mutable std::mutex m_;
    } mem_descs_;

private:
    /*! \brief compare two backend mem desc
    * \param mem_desc1
    * \param mem_desc2
    * \return bool
    */
    virtual bool is_mem_desc_equal(const graph::utils::any_t &mem_desc1,
            const graph::utils::any_t &mem_desc2) const = 0;
};

class dnnl_layout_id_manager_t : public layout_id_manager_t {
    friend class autograph_backend;

    // private, only can be created in dnnl_backend
    dnnl_layout_id_manager_t() = default;

    bool is_mem_desc_equal(const graph::utils::any_t &mem_desc1,
            const graph::utils::any_t &mem_desc2) const override;

#ifdef DNNL_GRAPH_LAYOUT_DEBUG
    static const size_t LAST_TAG
            = static_cast<size_t>(dnnl::memory::format_tag::format_tag_last);

public:
    graph::utils::optional<graph::utils::any_t> get_mem_desc(
            size_t layout_id) const override {
        std::lock_guard<std::mutex> lock(mem_descs_.m_);
        layout_id -= LAST_TAG;
        if (layout_id >= mem_descs_.data_.size()) return graph::utils::nullopt;
        return mem_descs_.data_[layout_id];
    }

    graph::utils::optional<size_t> set_mem_desc(
            const graph::utils::any_t &mem_desc) override {
        auto &md = graph::utils::any_cast<const memory::desc &>(mem_desc);
        size_t layout_id = 0;
        {
            std::lock_guard<std::mutex> lock(mem_descs_.m_);

            auto pos = std::find_if(mem_descs_.data_.begin(),
                    mem_descs_.data_.end(),
                    [&](const graph::utils::any_t &m) -> bool {
                        return is_mem_desc_equal(m, mem_desc);
                    });
            if (pos != mem_descs_.data_.end()) {
                layout_id = static_cast<size_t>(std::distance(
                                    mem_descs_.data_.begin(), pos))
                        + LAST_TAG;
            } else if (md.get_format_kind() != format_kind::blocked) {
                mem_descs_.data_.emplace_back(mem_desc);
                layout_id = mem_descs_.data_.size() - 1 + LAST_TAG;
            }
        }

        if (md.get_format_kind() == format_kind::blocked) {
            size_t format_tag = static_cast<size_t>(get_format_tag(md));

            if (!(format_tag > 0 && format_tag < dnnl_format_tag_last)) {
                size_t layout_id
                        = layout_id_manager_t::set_mem_desc(mem_desc).value();
                return layout_id + LAST_TAG;
            }

            // Check if md has extra flags. Note that since onednn didn't
            // provide api to check extra flags, here we construct a temp md
            // without extra flag, and then compare it with the origin md. If
            // they are not equal, the origin md may has extra flags. Only using
            // shape, data type and format tag can't describe the md anymore, so
            // we must cache it to layout id manager.
            const auto &dims = md.get_dims();
            const auto &dtype = md.get_data_type();
            memory::desc temp_md(
                    dims, dtype, static_cast<memory::format_tag>(layout_id));
            if (md != temp_md) {
                size_t layout_id
                        = layout_id_manager_t::set_mem_desc(mem_desc).value();
                return layout_id + LAST_TAG;
            }
        }

        return layout_id;
    }
#endif // DNNL_GRAPH_LAYOUT_DEBUG
};

// gcc4.8.5 can 't support enum class as key
struct enum_hash_t {
    template <typename T>
    size_t operator()(const T &t) const {
        return static_cast<size_t>(t);
    }
};

struct kernel_base_t {
    virtual ~kernel_base_t() = default;

    status_t compile(const dnnl_partition_impl_t *part, const engine_t *aengine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) {
        auto ret = compile_impl(part, aengine, inputs, outputs);
        if (ret != status::success) return ret;
        return prepare_inplace_pairs_impl();
    }

    status_t execute(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs) {
        return execute_impl(astream, inputs, outputs);
    }

#ifdef DNNL_WITH_SYCL
    status_t execute_sycl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) {
        return sycl_execute_impl(
                astream, inputs, outputs, sycl_deps, sycl_event);
    }

    virtual status_t sycl_execute_impl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event)
            = 0;
#endif

    virtual status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *aengine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs)
            = 0;

    virtual status_t execute_impl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs)
            = 0;

    virtual status_t prepare_inplace_pairs_impl() { return status::success; };

    // WA: Do not cache constant weight for SYCL CPU to workaround a segment
    // fault issue when releasing the cached buffer with sycl::free at the
    // program exits. Need to remove this check once the runtime issue is fixed.
    bool enabled_constant_cache() const {
        bool enabled = is_constant_cache_enabled();
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        if (p_engine_) {
            enabled = enabled
                    && (p_engine_.get_kind() != dnnl::engine::kind::cpu);
        }
#endif
        return enabled;
    }

    std::vector<inplace_pair_t> inplace_pairs_;
    dnnl::engine p_engine_;
};

using kernel_ptr = std::shared_ptr<kernel_base_t>;
using FCreateKernel = std::function<kernel_ptr(void)>;

kernel_ptr large_partition_kernel_creator();

class autograph_backend : public backend {
    friend class dnnl_partition_impl_t;

public:
    static autograph_backend &get_singleton() {
        static autograph_backend ins("autograph_backend", /*priority*/ 10.f);
        return ins;
    }

    // Used by DNNL backend to cache memory descriptor and get layout id
    graph::utils::optional_t<size_t> set_mem_desc(
            const graph::utils::any_t &mem_desc);

    graph::utils::optional_t<graph::utils::any_t> get_mem_desc(
            const size_t &layout_id) const;

    graph::pass::pass_registry_t &get_pass_registry() { return pass_registry_; }

    dnnl_layout_id_manager_t &get_layout_id_manager() {
        return layout_id_manager_;
    }

    size_t get_mem_size(const logical_tensor_t &lt) const override;

    bool compare_logical_tensor(const logical_tensor_t &lhs,
            const logical_tensor_t &rhs) const override;

    bool support_engine_kind(engine_kind_t kind) const override {
        static const std::unordered_set<engine_kind_t, enum_hash_t>
                supported_kind = {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
                    engine_kind::cpu,
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                    engine_kind::gpu,
#endif
                };
        return supported_kind.count(kind);
    }

    status_t get_partitions(
            graph_t &agraph, partition_policy_t policy) override {
        // Note: This environment variable is internal and for test purpose. It
        // can be changed or removed without prior notice. Users should avoid
        // using it in their applications. Enabling the environment variable may
        // cause some tests and examples to fail.
        const bool disable_dnnl_bkd
                = graph::utils::getenv_int_internal("DISABLE_DNNL_BACKEND", 0)
                > 0;
        if (disable_dnnl_bkd) return status::success;

        // Note: This environment variable is internal and for test/debug
        // purpose. It can be changed or removed without prior notice. Users
        // should avoid using it in their applications. Enabled by default.
        const bool enable_large_partition
                = graph::utils::getenv_int_internal("ENABLE_LARGE_PARTITION", 1)
                > 0;

        // FIXME(xx): Here we only changes the passes in registry. If json file
        // existed, pm will run passes according to the json file, the env var
        // will not take effect.
        // - priority > 20.f: large fusion pattern
        // - 20.f >= priority > 8.f: normal fusion pattern
        // - priority <= 8.f: debug fusion pattern (single op fusion)
        const float priority_ths = (policy == graph::partition_policy::fusion
                                           && enable_large_partition)
                ? std::numeric_limits<float>::max()
                : policy == graph::partition_policy::fusion ? 20.0f
                                                            : 8.0f;
        graph::pass::pass_registry_t filtered_registry;
        for (auto &pass : get_pass_registry().get_passes()) {
            if (pass->get_priority() > priority_ths) continue;
            filtered_registry.register_pass(pass);
        }

        graph::pass::pass_manager_t pm(filtered_registry);
#ifdef DNNL_ENABLE_GRAPH_DUMP
        std::string pass_config_json = "dnnl_graph_passes.json";
        std::ifstream fs(pass_config_json.c_str());
        if (fs) {
            printf("onednn_graph_verbose,info,pattern,load,%s\n",
                    pass_config_json.c_str());
            fflush(stdout);
        } else {
            if (getenv_int_user("GRAPH_DUMP", 0) > 0
                    || graph::utils::check_verbose_string_user(
                            "GRAPH_DUMP", "pattern")) {
                printf("onednn_graph_verbose,info,pattern,dump,%s\n",
                        pass_config_json.c_str());
                fflush(stdout);
                pm.print_passes(pass_config_json);
            }
        }
        pm.run_passes(agraph, &fs, policy);
#else
        pm.run_passes(agraph, "", policy);
#endif
        return status::success;
    }

private:
    autograph_backend(const std::string &name, float priority);

    bool register_passes();
    bool register_op_schemas();

    dnnl_layout_id_manager_t layout_id_manager_;
    graph::pass::pass_registry_t pass_registry_;
};

} // namespace autograph_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
