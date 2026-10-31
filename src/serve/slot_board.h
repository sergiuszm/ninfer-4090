#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

// Read-only llama.cpp-shaped GET /slots rows for scrapers such as the fleet dashboard. The rows
// are a serve-side display approximation, not Engine lanes: a started request takes the lowest
// idle row, requests beyond the row count wait in start order, and an idle row keeps the prompt
// depth of the last request that completed on it. The body never carries session_digest; clients
// use that key to detect slot persistence, which this server does not provide.
class SlotBoard {
public:
    // One request's place on the board. A lease that is destroyed without complete() releases
    // its row as failed, so a request that never settles cannot keep a row busy.
    class Lease {
    public:
        Lease() = default;
        Lease(SlotBoard& board, std::uint64_t request) noexcept
            : board_(&board), request_(request) {}
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease();

        // The row keeps the completed prompt depth; release() clears it.
        void complete(std::uint32_t cached_tokens) noexcept;
        void release() noexcept;

    private:
        SlotBoard* board_      = nullptr;
        std::uint64_t request_ = 0;
    };

    void configure(std::uint32_t rows, std::uint32_t context_tokens, bool speculative);
    [[nodiscard]] Lease start(std::uint64_t request, std::uint32_t prompt_tokens);
    [[nodiscard]] std::string render() const;

private:
    // Completed requests leave their depth on the row; failed ones (no cached count) clear it.
    void finish(std::uint64_t request, std::optional<std::uint32_t> cached_tokens) noexcept;

    struct Row {
        std::optional<std::uint64_t> request;
        std::uint32_t prompt_tokens = 0;
        std::uint32_t cached_tokens = 0;
    };

    struct Waiting {
        std::uint64_t request       = 0;
        std::uint32_t prompt_tokens = 0;
    };

    mutable std::mutex mutex_;
    std::vector<Row> rows_;
    std::deque<Waiting> waiting_;
    std::uint32_t context_tokens_ = 0;
    bool speculative_             = false;
};

} // namespace ninfer::serve
