#include "serve/slot_board.h"

#include <nlohmann/json.hpp>

#include <algorithm>

namespace ninfer::serve {

void SlotBoard::configure(std::uint32_t rows, std::uint32_t context_tokens, bool speculative) {
    std::lock_guard lock(mutex_);
    rows_.assign(rows, Row{});
    waiting_.clear();
    context_tokens_ = context_tokens;
    speculative_    = speculative;
}

void SlotBoard::start(std::uint64_t request, std::uint32_t prompt_tokens) {
    std::lock_guard lock(mutex_);
    const auto idle =
        std::find_if(rows_.begin(), rows_.end(), [](const Row& row) { return !row.request; });
    if (idle == rows_.end()) {
        waiting_.push_back({request, prompt_tokens});
        return;
    }
    *idle = Row{.request = request, .prompt_tokens = prompt_tokens};
}

void SlotBoard::finish(std::uint64_t request, std::optional<std::uint32_t> cached_tokens) {
    std::lock_guard lock(mutex_);
    const auto waiting = std::find_if(waiting_.begin(), waiting_.end(), [&](const Waiting& entry) {
        return entry.request == request;
    });
    if (waiting != waiting_.end()) {
        waiting_.erase(waiting);
        return;
    }
    const auto row = std::find_if(rows_.begin(), rows_.end(),
                                  [&](const Row& entry) { return entry.request == request; });
    if (row == rows_.end()) return;
    row->request = std::nullopt;
    if (cached_tokens) {
        row->cached_tokens = std::min(*cached_tokens, row->prompt_tokens);
    } else {
        *row = Row{};
    }
    if (!waiting_.empty()) {
        const auto next = waiting_.front();
        waiting_.pop_front();
        *row = Row{.request = next.request, .prompt_tokens = next.prompt_tokens};
    }
}

std::string SlotBoard::render() const {
    std::lock_guard lock(mutex_);
    auto slots = nlohmann::json::array();
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        const auto& row = rows_[i];
        // An idle row keeps its last completed prompt, as a llama.cpp slot does.
        slots.push_back({{"id", i},
                         {"is_processing", row.request.has_value()},
                         {"n_ctx", context_tokens_},
                         {"n_prompt_tokens", row.prompt_tokens},
                         {"n_prompt_tokens_cache", row.cached_tokens},
                         {"speculative", speculative_}});
    }
    return slots.dump();
}

} // namespace ninfer::serve
