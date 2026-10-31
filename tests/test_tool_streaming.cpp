#include "models/common/qwen/frontend/tool_call_parser.h"
#include "serve/anthropic_messages.h"
#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_responses.h"
#include "serve/translate.h"
#include "serve_tool_stream_fixture.h"

#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {
using Json = nlohmann::ordered_json;
using namespace ninfer;
using namespace ninfer::serve;
namespace fi = ninfer::models::qwen::frontend;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::shared_ptr<const fi::ToolCallOutputContract> contract(bool strict, bool constrained) {
    Json definition{{"type", "function"},
                    {"function",
                     {{"name", "write"},
                      {"strict", strict},
                      {"parameters",
                       {{"type", "object"},
                        {"properties",
                         {{"path", {{"type", "string"}}},
                          {"content", {{"type", "string"}}},
                          {"payload",
                           {{"type", "object"},
                            {"properties", {{"text", {{"type", "string"}}}}},
                            {"required", {"text"}},
                            {"additionalProperties", false}}}}},
                        {"required", {"path", "content", "payload"}},
                        {"additionalProperties", false}}}}}};
    const std::vector<std::string> definitions{definition.dump()};
    auto result = fi::build_tool_call_output_contract(definitions);
    if (!constrained) return result;
    return fi::select_tool_call_contract(result, {});
}

std::string parameter(std::string_view name, std::string_view value) {
    return "<parameter=" + std::string(name) + ">\n" + std::string(value) + "\n</parameter>\n";
}

struct Recorded {
    std::string text;
    std::vector<GeneratedToolCall> calls;
    std::size_t deltas = 0;

    void accept(const std::vector<OutputEvent>& events) {
        for (const auto& event : events) {
            std::visit(
                [&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, OutputDelta>)
                        text += value.text;
                    else if constexpr (std::is_same_v<T, ToolCallStart>) {
                        require(value.index == calls.size(), "call index changed");
                        calls.push_back({value.name, {}, ToolCallStatus::InProgress});
                    } else if constexpr (std::is_same_v<T, ToolArgumentsDelta>) {
                        require(calls.at(value.index).status == ToolCallStatus::InProgress,
                                "delta after end");
                        calls[value.index].arguments_json += value.text;
                        // Every published fragment must be serializable independently, including
                        // Unicode.
                        (void)Json(value.text).dump();
                        if (!value.text.empty()) ++deltas;
                    } else {
                        require(calls.at(value.index).status == ToolCallStatus::InProgress,
                                "duplicate end");
                        calls[value.index].status = value.status;
                    }
                },
                event);
        }
    }
};

void long_arguments() {
    std::string html = "  <!DOCTYPE html>\n";
    for (int i = 0; i < 120; ++i)
        html +=
            "<svg "
            "title=\"鳄鱼🦎\">\\code\t<parameter=x>y</parameter><tool_call></tool_call></svg>\r\n";
    const std::string prefix = "<tool_call>\n<function=write>\n" + parameter("path", "page.html") +
                               "<parameter=content>\n";
    const std::string tail = "\n</parameter>\n" +
                             parameter("payload", R"({"text":"</parameter>"})") +
                             "</function>\n</tool_call>";
    for (const auto [strict, constrained] :
         {std::pair{false, false}, {false, true}, {true, true}}) {
        for (std::size_t chunk : {1U, 7U, 257U}) {
            fi::ToolCallOutputDecoder decoder(contract(strict, constrained), 64);
            Recorded observed;
            observed.accept(decoder.feed(prefix));
            for (std::size_t i = 0; i < html.size(); i += chunk)
                observed.accept(decoder.feed(std::string_view(html).substr(i, chunk)));
            require(!decoder.error(), "valid long string failed before close");
            require(observed.deltas > 10 && observed.calls.size() == 1 &&
                        observed.calls[0].status == ToolCallStatus::InProgress &&
                        observed.calls[0].arguments_json.find("<!DOCTYPE html>") !=
                            std::string::npos,
                    "large argument was withheld until parameter or call completion");
            observed.accept(decoder.feed(tail));
            observed.accept(decoder.finish());
            require(!decoder.error(), "valid long call failed");
            const auto calls = decoder.take_tool_calls();
            require(calls == observed.calls, "stream and aggregate differ");
            require(Json::parse(calls[0].arguments_json) ==
                        Json{{"path", "page.html"},
                             {"content", html},
                             {"payload", {{"text", "</parameter>"}}}},
                    "streaming changed parameter bytes or JSON types");
            require(observed.text.empty(), "tool markup leaked into content");
        }
    }
}

void json_parameters_and_errors() {
    const auto bound = contract(false, true);
    fi::ToolCallOutputDecoder decoder(bound, 64);
    Recorded observed;
    observed.accept(
        decoder.feed("<tool_call>\n<function=write>\n<parameter=payload>\n{\"text\":\""));
    const std::string large(16000, 'x');
    observed.accept(decoder.feed(large));
    require(observed.calls[0].arguments_json.size() > large.size(), "JSON value did not stream");
    observed.accept(decoder.feed("\"}\n</parameter>\n</function>\n</tool_call>"));
    observed.accept(decoder.finish());
    require(!decoder.error() && observed.calls == decoder.take_tool_calls(),
            "JSON streaming changed the result");

    for (const bool constrained : {false, true})
        for (const std::string body : {parameter("path", "one") + parameter("path", "two"),
                                       parameter("payload", R"({"text":broken})")}) {
            fi::ToolCallOutputDecoder bad(contract(false, constrained), 64);
            Recorded partial;
            partial.accept(bad.feed("<tool_call>\n<function=write>\n" + body));
            require(bad.error().has_value(),
                    "malformed committed tool did not report a request error");
            require(bad.error()->kind ==
                        (constrained
                             ? ninfer::models::frontend::ToolOutputErrorKind::ContractViolation
                             : ninfer::models::frontend::ToolOutputErrorKind::InvalidModelOutput),
                    "constrained decoder disagreement was classified as ordinary model output");
            require(partial.text.empty() && partial.calls[0].status == ToolCallStatus::Incomplete,
                    "failed call was retracted or marked complete");
        }
}

void buffered_normalization() {
    const std::string head = "<tool_call>\n<function=write>\n<parameter=unknown>\n";
    const std::string tail = "\n</parameter>\n</function>\n</tool_call>";
    for (const bool constrained : {false, true})
        for (const auto value : {"[1,]", "{\"a\": 1,}", "\"unfinished", "[1,2]", "  text  ",
                                 "{\"tag\":\"</parameter>\"}"}) {
            fi::ToolCallOutputDecoder decoder(contract(false, constrained), 64);
            Recorded observed;
            observed.accept(decoder.feed(head));
            const auto prefix = observed.calls.at(0).arguments_json;
            for (const char byte : std::string_view(value))
                observed.accept(decoder.feed(std::string_view(&byte, 1)));
            require(observed.calls.at(0).arguments_json == prefix,
                    "inferred value committed its representation before the delimiter");
            observed.accept(decoder.feed(tail));
            observed.accept(decoder.finish());
            if (const auto error = decoder.error())
                throw std::runtime_error("buffered parameter " + std::string(value) + ": " +
                                         error->message);
            require(observed.calls == decoder.take_tool_calls(),
                    "buffered stream/aggregate differ");
            const auto decoded = Json::parse(observed.calls[0].arguments_json)["unknown"];
            if (Json::accept(value))
                require(decoded == Json::parse(value), "JSON inference lost");
            else
                require(decoded == (std::string_view(value) == "  text  " ? "text" : value),
                        "invalid inferred JSON was not normalized as text");
        }
}

void strict_constant_whitespace() {
    const std::vector<std::string> definitions{Json{
        {"type", "function"},
        {"function",
         {{"name", "write"},
          {"strict", true},
          {"parameters",
           {{"type", "object"},
            {"properties", {{"content", {{"const", " 你好\n"}}}}},
            {"required", {"content"}},
            {"additionalProperties", false}}}}}}.dump()};
    auto bound =
        fi::select_tool_call_contract(fi::build_tool_call_output_contract(definitions), {});
    fi::ToolCallOutputDecoder decoder(bound, 64);
    Recorded observed;
    observed.accept(decoder.feed("<tool_call>\n<function=write>\n" +
                                 parameter("content", " 你好\n") + "</function>\n</tool_call>"));
    observed.accept(decoder.finish());
    require(Json::parse(observed.calls[0].arguments_json)["content"] == " 你好\n",
            "strict inferred string codec trimmed constant whitespace");
}

void interruption_and_continuation() {
    const std::string complete = "<tool_call>\n<function=write>\n" + parameter("content", "hello") +
                                 "</function>\n</tool_call>";
    for (const auto reason : {FinishReason::OutputLimit, FinishReason::ContextCapacity,
                              FinishReason::Cancelled, FinishReason::StopString}) {
        for (std::size_t cut = 0; cut <= complete.size(); ++cut) {
            fi::ToolCallOutputDecoder decoder(contract(false, true), 64);
            Recorded observed;
            observed.accept(decoder.feed(std::string_view(complete).substr(0, cut)));
            observed.accept(decoder.finish(reason));
            require(!decoder.error() && observed.calls == decoder.take_tool_calls(),
                    "interruption disagreed with published argument prefix");
            if (!observed.calls.empty())
                require(observed.calls[0].status == (cut == complete.size()
                                                         ? ToolCallStatus::Complete
                                                         : ToolCallStatus::Incomplete),
                        "truncated call was marked complete");
        }
    }
    const auto split = complete.find("hello") + 2;
    fi::ToolCallOutputDecoder continued(contract(false, true), 64);
    continued.initialize_continuation(std::string_view(complete).substr(0, split));
    Recorded observed;
    observed.accept(continued.feed(std::string_view(complete).substr(split)));
    observed.accept(continued.finish());
    require(!continued.error() && observed.calls == continued.take_tool_calls() &&
                Json::parse(observed.calls[0].arguments_json)["content"] == "hello",
            "continuation duplicated or lost seeded arguments");
}

Json event_json(const std::string& wire) {
    const auto pos = wire.find("data: ");
    require(pos != std::string::npos, "SSE event missing data");
    return Json::parse(wire.substr(pos + 6));
}

template <class Stream>
std::vector<std::string> deliver(Stream& stream, const OutputEvent& event) {
    return std::visit(
        [&](const auto& value) -> std::vector<std::string> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, ToolCallStart>)
                return stream.tool_start(value);
            else if constexpr (std::is_same_v<T, ToolArgumentsDelta>)
                return stream.tool_arguments(value);
            else if constexpr (std::is_same_v<T, ToolCallEnd>)
                return stream.tool_end(value);
            else
                return {};
        },
        event);
}

void protocols() {
    for (const bool interrupted : {false, true}) {
        OpenAIChatStream chat(make_openai_chat_response_identity("test"), false);
        (void)chat.start();
        OpenAIResponsesCreateRequest request;
        request.prompt.model = "test";
        OpenAIResponsesEventStream responses("resp_stream", 1, request, {});
        (void)responses.start();
        AnthropicMessagesStream anthropic(make_anthropic_response_identity("req", "test"), 10,
                                          {{"write", true}});
        (void)anthropic.start();
        std::vector<std::string> c, r, a;
        fi::ToolCallOutputDecoder decoder(contract(false, true), 64);
        const auto consume = [&](const std::vector<OutputEvent>& events) {
            for (const auto& event : events) {
                const auto add = [&](auto& stream, auto& wire) {
                    auto next = deliver(stream, event);
                    wire.insert(wire.end(), next.begin(), next.end());
                };
                add(chat, c);
                add(responses, r);
                add(anthropic, a);
            }
        };
        consume(decoder.feed("<tool_call>\n<function=write>\n<parameter=content>\nhello"));
        consume(decoder.feed(" world"));
        require(c.size() >= 3 && r.size() >= 3 && a.size() >= 3,
                "protocol buffered the active string parameter");
        if (!interrupted) consume(decoder.feed("\n</parameter>\n</function>\n</tool_call>"));
        consume(decoder.finish(interrupted ? FinishReason::OutputLimit : FinishReason::StopToken));
        GenerationOutcome outcome;
        outcome.tool_calls     = decoder.take_tool_calls();
        outcome.finish_reason  = interrupted ? FinishReason::OutputLimit : FinishReason::StopToken;
        const auto chat_finish = chat.finish(outcome);
        const auto response_finish = responses.finish(outcome);
        (void)anthropic.finish(outcome);
        std::string ca, ra, aa;
        const auto call_id = event_json(c.front())["choices"][0]["delta"]["tool_calls"][0]["id"];
        require(call_id.is_string(), "Chat call ID missing at start");
        for (const auto& wire : c) {
            const auto delta = event_json(wire)["choices"][0]["delta"]["tool_calls"][0];
            if (delta.contains("id")) require(delta["id"] == call_id, "Chat call ID changed");
            ca += delta["function"].value("arguments", "");
        }
        const auto item_id = event_json(r.front())["item"]["id"];
        for (const auto& wire : r) {
            const auto value = event_json(wire);
            if (value["type"] == "response.function_call_arguments.delta") {
                require(value["item_id"] == item_id, "Responses item ID changed");
                ra += value["delta"].get<std::string>();
            }
        }
        for (const auto& wire : a) {
            const auto value = event_json(wire);
            if (value["type"] == "content_block_delta")
                aa += value["delta"]["partial_json"].get<std::string>();
        }
        require(ca == outcome.tool_calls[0].arguments_json && ca == ra && ca == aa,
                "protocol parameter deltas disagree");
        const auto& item = response_finish.response.body["output"][0];
        require(item["id"] == item_id && item["arguments"] == ra &&
                    item["status"] == (interrupted ? "incomplete" : "completed"),
                "Responses terminal item changed streamed identity or completion");
        require(event_json(chat_finish[0])["choices"][0]["finish_reason"] ==
                    (interrupted ? "length" : "tool_calls"),
                "Chat hid the interruption");
        if (interrupted) {
            require(response_finish.response.output_history.empty(),
                    "partial tool entered executable history");
            const auto aggregate = Json::parse(make_anthropic_messages_response(
                make_anthropic_response_identity("req", "test"), outcome));
            for (const auto& block : aggregate["content"])
                require(block["type"] != "tool_use", "Anthropic fabricated a partial tool object");
        }
    }
    AnthropicMessagesStream buffered(make_anthropic_response_identity("req", "test"), 10);
    (void)buffered.start();
    (void)buffered.tool_start({0, "write"});
    require(buffered.tool_arguments({0, "{\"content\":\"part"}).empty(), "buffered policy ignored");
    const auto flushed = buffered.tool_arguments({0, "ial\"", true});
    require(flushed.size() == 1 &&
                event_json(flushed[0])["delta"]["partial_json"] == "{\"content\":\"partial\"",
            "buffered parameter boundary changed bytes");
    const auto parsed = parse_anthropic_messages_request(
        nlohmann::json{{"model", "test"},
                       {"messages", {{{"role", "user"}, {"content", "hello"}}}},
                       {"tools",
                        {{{"name", "write"},
                          {"input_schema", {{"type", "object"}}},
                          {"eager_input_streaming", true}}}}},
        {});
    require(parsed.eager_tool_inputs.at("write"), "eager streaming option was lost");

    OpenAIResponsesCreateRequest failed_request;
    failed_request.prompt.model = "test";
    OpenAIResponsesEventStream failed("resp_failed", 1, failed_request, {});
    (void)failed.start();
    (void)failed.reasoning_delta("thought");
    (void)failed.content_delta("writing");
    const auto start = failed.tool_start({0, "write"});
    const auto id    = event_json(start.back())["item"]["id"];
    (void)failed.tool_arguments({0, "{\"content\":\"partial"});
    const auto response = event_json(failed.failed(ApiError{
        .status = 500, .type = "server_error", .message = "invalid tool output"}))["response"];
    require(response["status"] == "failed" && response["output"].size() == 3 &&
                response["output"][0]["content"][0]["text"] == "thought" &&
                response["output"][1]["content"][0]["text"] == "writing" &&
                response["output"][2]["id"] == id &&
                response["output"][2]["status"] == "incomplete" &&
                response["output"][2]["arguments"] == "{\"content\":\"partial",
            "failed Responses snapshot lost previously published output");
}

void protocol_terminal_states() {
    for (const auto reason :
         {FinishReason::StopToken, FinishReason::StopString, FinishReason::OutputLimit,
          FinishReason::ContextCapacity, FinishReason::Cancelled}) {
        const bool completed = reason == FinishReason::StopToken;
        GenerationOutcome outcome;
        outcome.finish_reason = reason;
        outcome.reasoning     = "ready";
        outcome.text          = "writing files";
        outcome.tool_calls    = {{"write", R"({"content":"first"})", ToolCallStatus::Complete},
                                 {"write",
                               completed ? R"({"content":"second"})" : R"({"content":"sec)",
                               completed ? ToolCallStatus::Complete : ToolCallStatus::Incomplete}};

        const auto chat_identity = make_openai_chat_response_identity("test");
        OpenAIChatStream chat(chat_identity, false);
        (void)chat.start();
        (void)chat.reasoning_delta(outcome.reasoning);
        (void)chat.content_delta(outcome.text);
        (void)emit_tools(chat, outcome.tool_calls);
        const auto chat_finish = event_json(chat.finish(outcome).front());
        const auto chat_aggregate =
            Json::parse(make_chat_completion_response(chat_identity, outcome));
        const char* expected_reason =
            completed ? "tool_calls"
            : reason == FinishReason::OutputLimit || reason == FinishReason::ContextCapacity
                ? "length"
                : "stop";
        require(chat_finish["choices"][0]["finish_reason"] == expected_reason &&
                    chat_aggregate["choices"][0]["finish_reason"] == expected_reason,
                "earlier completed call hid the actual terminal reason");

        OpenAIResponsesCreateRequest request;
        request.prompt.model = "test";
        OpenAIResponsesEventStream responses("resp_terminal", 1, request, {});
        (void)responses.start();
        (void)responses.reasoning_delta(outcome.reasoning);
        auto events      = responses.content_delta(outcome.text);
        const auto tools = emit_tools(responses, outcome.tool_calls);
        events.insert(events.end(), tools.begin(), tools.end());
        const auto finished = responses.finish(outcome);
        events.insert(events.end(), finished.events_before_terminal.begin(),
                      finished.events_before_terminal.end());
        const Json output = finished.response.body["output"];
        std::size_t done  = 0;
        for (const auto& wire : events) {
            const auto event = event_json(wire);
            if (event["type"] != "response.output_item.done") continue;
            require(output.at(event["output_index"].get<std::size_t>()) == event["item"],
                    "terminal Responses output rewrote an already completed item");
            ++done;
        }
        require(done == output.size() && done == 4, "missing completed output item");
        const auto aggregate =
            make_openai_response_object("resp_aggregate", 1, request, {}, outcome);
        for (std::size_t i = 0; i < output.size(); ++i)
            require(aggregate.body["output"][i]["status"] == output[i]["status"],
                    "stream and aggregate item statuses differ");
        require(finished.response.output_history.at(0).tool_calls.size() == (completed ? 2 : 1),
                "interrupted call entered executable response history");
    }
}

// Fork-local (PR #14): a call recovered by --tolerant-tool-calls streams and aggregates the same on
// every protocol, and the published arguments are never retracted.
template <class Stream>
std::vector<std::string> deliver_all(Stream& stream, const OutputEvent& event) {
    const auto* delta = std::get_if<OutputDelta>(&event);
    if (!delta) return deliver(stream, event);
    auto wire = stream.content_delta(delta->text);
    if constexpr (std::is_same_v<decltype(wire), std::string>)
        return wire.empty() ? std::vector<std::string>{}
                            : std::vector<std::string>{std::move(wire)};
    else
        return wire;
}

void tolerant_protocols() {
    const std::string body =
        "Header line\nvalue one\n</parameter>\nThe call ends with </tool_call>.\nvalue two";
    const std::string call = "<tool_call>\n<function=write>\n" + parameter("path", "notes.txt") +
                             parameter("content", body) + "</function>\n</tool_call>";
    const std::string text = "Writing it.\n" + call + "\nDone.";

    OpenAIChatStream chat(make_openai_chat_response_identity("test"), false);
    (void)chat.start();
    OpenAIResponsesCreateRequest request;
    request.prompt.model = "test";
    OpenAIResponsesEventStream responses("resp_tolerant", 1, request, {});
    (void)responses.start();
    const auto anthropic_identity = make_anthropic_response_identity("req", "test");
    AnthropicMessagesStream anthropic(anthropic_identity, 10, {{"write", true}});
    (void)anthropic.start();
    std::vector<std::string> c, r, a;
    Recorded recorded;
    fi::ToolCallOutputDecoder decoder(contract(false, false), 64, false, true);
    const auto consume = [&](const std::vector<OutputEvent>& events) {
        recorded.accept(events);
        for (const auto& event : events) {
            const auto add = [&](auto& stream, auto& wire) {
                auto next = deliver_all(stream, event);
                wire.insert(wire.end(), next.begin(), next.end());
            };
            add(chat, c);
            add(responses, r);
            add(anthropic, a);
        }
    };
    for (std::size_t offset = 0; offset < text.size(); offset += 5)
        consume(decoder.feed(std::string_view(text).substr(offset, 5)));
    consume(decoder.finish());
    require(!decoder.error(), "tolerant decoder failed a recoverable call");

    GenerationOutcome outcome;
    outcome.text            = recorded.text;
    outcome.tool_calls      = decoder.take_tool_calls();
    outcome.finish_reason   = FinishReason::StopToken;
    outcome.tool_call_parse = decoder.diagnostics();
    require(outcome.text == "Writing it." && outcome.tool_calls.size() == 1 &&
                outcome.tool_calls[0].status == ToolCallStatus::Complete &&
                recorded.calls == outcome.tool_calls &&
                Json::parse(outcome.tool_calls[0].arguments_json)["content"] == body &&
                outcome.tool_call_parse.recovered_from ==
                    ToolCallParseFallbackReason::MalformedStructure &&
                outcome.tool_call_parse.dropped_trailing_bytes == 5,
            "tolerant decoder did not recover the call and drop the trailing prose");

    // Chat: the content before the call, then the call; no content after it.
    const auto chat_finish = chat.finish(outcome);
    std::string chat_arguments, chat_content;
    bool content_after_call = false;
    for (const auto& wire : c) {
        const auto delta = event_json(wire)["choices"][0]["delta"];
        if (delta.contains("content") && delta["content"].is_string()) {
            chat_content += delta["content"].get<std::string>();
            content_after_call = content_after_call || !chat_arguments.empty();
        }
        if (delta.contains("tool_calls"))
            chat_arguments += delta["tool_calls"][0]["function"].value("arguments", "");
    }
    const auto chat_aggregate = Json::parse(
        make_chat_completion_response(make_openai_chat_response_identity("test"), outcome));
    const auto& message = chat_aggregate["choices"][0]["message"];
    require(chat_arguments == outcome.tool_calls[0].arguments_json && !content_after_call &&
                chat_content == outcome.text && message["content"] == outcome.text &&
                message["tool_calls"][0]["function"]["arguments"] == chat_arguments &&
                event_json(chat_finish[0])["choices"][0]["finish_reason"] == "tool_calls",
            "Chat stream and aggregate disagree on the recovered call");

    // Responses: every streamed item equals the terminal item; no text after the call.
    const auto response_finish = responses.finish(outcome);
    r.insert(r.end(), response_finish.events_before_terminal.begin(),
             response_finish.events_before_terminal.end());
    const Json output = response_finish.response.body["output"];
    std::string response_arguments;
    std::size_t done     = 0;
    bool call_started    = false;
    bool text_after_call = false;
    for (const auto& wire : r) {
        const auto value = event_json(wire);
        if (value["type"] == "response.function_call_arguments.delta") {
            response_arguments += value["delta"].get<std::string>();
            call_started = true;
        }
        if (value["type"] == "response.output_text.delta") text_after_call |= call_started;
        if (value["type"] != "response.output_item.done") continue;
        require(output.at(value["output_index"].get<std::size_t>()) == value["item"],
                "Responses terminal output rewrote a streamed item");
        ++done;
    }
    require(response_arguments == chat_arguments && !text_after_call && done == output.size() &&
                output.size() == 2 && output[1]["type"] == "function_call" &&
                output[1]["arguments"] == response_arguments,
            "Responses stream disagrees on the recovered call");

    // Messages: the streamed blocks are the aggregate blocks: text, then tool_use.
    (void)anthropic.finish(outcome);
    std::string anthropic_arguments, anthropic_content;
    std::vector<std::string> streamed_blocks;
    for (const auto& wire : a) {
        const auto value = event_json(wire);
        if (value["type"] == "content_block_start")
            streamed_blocks.push_back(value["content_block"]["type"].get<std::string>());
        if (value["type"] != "content_block_delta") continue;
        if (value["delta"]["type"] == "input_json_delta")
            anthropic_arguments += value["delta"]["partial_json"].get<std::string>();
        else if (value["delta"]["type"] == "text_delta")
            anthropic_content += value["delta"]["text"].get<std::string>();
    }
    const auto anthropic_aggregate =
        Json::parse(make_anthropic_messages_response(anthropic_identity, outcome));
    std::vector<std::string> aggregate_blocks;
    bool tool_use = false;
    for (const auto& block : anthropic_aggregate["content"]) {
        aggregate_blocks.push_back(block["type"].get<std::string>());
        if (block["type"] == "tool_use") tool_use = block["input"]["content"] == body;
    }
    require(anthropic_arguments == chat_arguments && anthropic_content == outcome.text &&
                tool_use && streamed_blocks == aggregate_blocks &&
                aggregate_blocks == std::vector<std::string>{"text", "tool_use"} &&
                anthropic_aggregate["stop_reason"] == "tool_use",
            "Messages stream and aggregate disagree on the recovered call");

    // Flag off: the same output is invalid_tool_output (the change is 500 -> completed call).
    fi::ToolCallOutputDecoder strict(contract(false, false), 64);
    (void)strict.feed(text);
    (void)strict.finish();
    bool strict_invalid = false;
    try {
        if (const auto error = strict.error()) std::rethrow_exception(error->exception());
    } catch (const RequestError& error) {
        strict_invalid = error.kind() == RequestErrorKind::InvalidToolOutput;
    }
    require(strict_invalid, "flag-off decoding of the quoted close changed");

    // Prose and then another call: the request fails and no call completes.
    fi::ToolCallOutputDecoder failing(contract(false, false), 64, false, true);
    Recorded failed;
    failed.accept(failing.feed(call + "\nNow the real one.\n" + call));
    failed.accept(failing.finish());
    bool invalid = false;
    try {
        if (const auto error = failing.error()) std::rethrow_exception(error->exception());
    } catch (const RequestError& error) {
        invalid = error.kind() == RequestErrorKind::InvalidToolOutput;
    }
    const auto calls = failing.take_tool_calls();
    require(invalid && calls.size() == 1 && calls[0].status == ToolCallStatus::Incomplete &&
                failed.calls == calls,
            "an example call followed by prose and another call was not a request error");
}

// Fork-local (PR #14): the serve request path, Chat and Messages, without tool_constraints (the
// server default) and with an explicit auto. The flag reaches the decoder; it acts only on the free
// route.
void tolerant_request_path() {
    product::ServeOptions server;
    server.tolerant_tool_calls = true;
    const std::string body =
        "Header line\nvalue one\n</parameter>\nThe call ends with </tool_call>.\nvalue two\n";
    const std::string text = "<tool_call>\n<function=write>\n" + parameter("path", "notes.txt") +
                             parameter("content", body) + "</function>\n</tool_call>";
    const auto decode = [&](const GenerationRequest& request, const char* label) {
        const auto options = to_request_options(request, server, true);
        require(options.output.tolerant_tool_calls, label);
        const auto prompt   = to_prompt_input(request, collect_prompt_options(request, server), {});
        const auto selected = fi::select_tool_call_contract(
            fi::build_tool_call_output_contract(prompt.options.tool_jsons), options.tool_choice);
        require(selected && selected->constrained == request.constrains_tools(), label);
        const auto run = [&](bool tolerant) {
            fi::ToolCallOutputDecoder decoder(selected, options.output.tool_name_max_length, false,
                                              tolerant);
            Recorded recorded;
            recorded.accept(decoder.feed(text));
            recorded.accept(decoder.finish());
            const bool failed = decoder.error().has_value();
            return std::pair{failed, decoder.take_tool_calls()};
        };
        const auto [failed, calls] = run(options.output.tolerant_tool_calls);
        if (request.constrains_tools()) {
            // Basic: the grammar cannot emit the quoted close; the flag changes nothing here.
            require(std::pair{failed, calls} == run(false), label);
            return;
        }
        require(!failed && calls.size() == 1 && calls[0].status == ToolCallStatus::Complete &&
                    Json::parse(calls[0].arguments_json)["content"] == body,
                label);
    };
    const nlohmann::json schema{
        {"type", "object"},
        {"properties", {{"path", {{"type", "string"}}}, {"content", {{"type", "string"}}}}}};
    nlohmann::json chat{
        {"model", "test"},
        {"messages", {{{"role", "user"}, {"content", "write it"}}}},
        {"tools",
         {{{"type", "function"}, {"function", {{"name", "write"}, {"parameters", schema}}}}}}};
    decode(parse_chat_completion_request(chat, request_limits(server)).generation,
           "Chat without tool_constraints");
    chat["tool_constraints"] = "auto";
    const auto chat_auto     = parse_chat_completion_request(chat, {}).generation;
    require(!chat_auto.constrains_tools(), "Chat auto was constrained");
    decode(chat_auto, "Chat with tool_constraints auto");
    nlohmann::json messages{{"model", "test"},
                            {"max_tokens", 64},
                            {"messages", {{{"role", "user"}, {"content", "write it"}}}},
                            {"tools", {{{"name", "write"}, {"input_schema", schema}}}}};
    decode(parse_anthropic_messages_request(messages, request_limits(server)).generation,
           "Messages without tool_constraints");
    messages["tool_constraints"] = "auto";
    const auto messages_auto     = parse_anthropic_messages_request(messages, {}).generation;
    require(!messages_auto.constrains_tools(), "Messages auto was constrained");
    decode(messages_auto, "Messages with tool_constraints auto");
}

} // namespace

int main() {
    try {
        long_arguments();
        json_parameters_and_errors();
        buffered_normalization();
        strict_constant_whitespace();
        interruption_and_continuation();
        protocols();
        protocol_terminal_states();
        tolerant_protocols();
        tolerant_request_path();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
