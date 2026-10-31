#include "serve/slot_board.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <optional>
#include <utility>

using namespace ninfer::serve;
using Json = nlohmann::json;

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    SlotBoard board;
    board.configure(2, 262144, true);
    Json slots = Json::parse(board.render());
    check(slots.is_array() && slots.size() == 2 && slots[1]["id"] == 1 &&
              slots[1]["is_processing"] == false && slots[1]["n_ctx"] == 262144 &&
              slots[1]["n_prompt_tokens"] == 0 && slots[1]["speculative"] == true,
          "idle board must list one llama.cpp-shaped row per lane");
    // pi's slot extension enables persistence only when every row carries session_digest.
    for (const auto& slot : slots) {
        check(!slot.contains("session_digest") && !slot.contains("retained"),
              "rows must not advertise slot persistence");
    }

    auto first  = board.start(7, 1000);
    auto second = board.start(8, 2000);
    auto third  = board.start(9, 3000);
    slots       = Json::parse(board.render());
    check(slots[0]["is_processing"] == true && slots[0]["n_prompt_tokens"] == 1000 &&
              slots[1]["is_processing"] == true && slots[1]["n_prompt_tokens"] == 2000,
          "started requests must take the idle rows in start order");

    first.complete(600);
    slots = Json::parse(board.render());
    check(slots[0]["is_processing"] == true && slots[0]["n_prompt_tokens"] == 3000 &&
              slots[0]["n_prompt_tokens_cache"] == 0,
          "a waiting request must take the row released first");

    second.complete(5000);
    slots = Json::parse(board.render());
    check(slots[1]["is_processing"] == false && slots[1]["n_prompt_tokens"] == 2000 &&
              slots[1]["n_prompt_tokens_cache"] == 2000,
          "an idle row must keep the completed depth, with the cache clamped to the prompt");

    third.release();
    third.complete(1);
    slots = Json::parse(board.render());
    check(slots[0]["is_processing"] == false && slots[0]["n_prompt_tokens"] == 0,
          "a failed request must not leave a retained depth, and settles only once");

    {
        auto moved = board.start(10, 10);
        auto owner = std::move(moved);
        slots      = Json::parse(board.render());
        check(slots[0]["is_processing"] == true && slots[0]["n_prompt_tokens"] == 10,
              "a moved lease must keep its row");
    }
    slots = Json::parse(board.render());
    check(slots[0]["is_processing"] == false && slots[0]["n_prompt_tokens"] == 0,
          "a lease destroyed without settling must release its row as failed");

    {
        auto busy0   = board.start(11, 11);
        auto busy1   = board.start(12, 12);
        auto waiting = std::optional<SlotBoard::Lease>(board.start(13, 13));
        waiting.reset();
        busy0.complete(5);
        slots = Json::parse(board.render());
        check(slots[0]["is_processing"] == false && slots[0]["n_prompt_tokens"] == 11 &&
                  slots[1]["is_processing"] == true && slots[1]["n_prompt_tokens"] == 12,
              "a request that leaves while waiting must not take a row later");
    }
    slots = Json::parse(board.render());
    check(slots[1]["is_processing"] == false,
          "every unsettled lease must release its row when it is destroyed");
    return failures == 0 ? 0 : 1;
}
