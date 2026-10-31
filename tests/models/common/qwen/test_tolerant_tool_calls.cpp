// Fork-local (PR #14): tolerant Qwen tool calls on the free route. The cases are the fork's
// tolerant parser tests and the production review harness (PR #14 rounds 1-4), rewritten for the
// streaming decoder: output that production returned as text is a request error here, because the
// decoder has no text fallback after a declared tool header.
#include "models/common/qwen/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
using Reason = ninfer::ToolCallParseFallbackReason;
using ninfer::FinishReason;
namespace fi   = ninfer::models::qwen::frontend;
using Contract = std::shared_ptr<const fi::ToolCallOutputContract>;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

std::string tool_definition(const std::string& name, Json properties) {
    Json parameters{{"type", "object"}, {"properties", std::move(properties)}};
    return Json{{"type", "function"},
                {"function", Json{{"name", name}, {"parameters", std::move(parameters)}}}}
        .dump();
}

Contract contract_for(std::vector<std::pair<std::string, Json>> tools) {
    std::vector<std::string> definitions;
    for (auto& [name, properties] : tools)
        definitions.push_back(tool_definition(name, std::move(properties)));
    return fi::build_tool_call_output_contract(definitions);
}

const Json kString = Json{{"type", "string"}};

// The production review harness tools.
Contract agent_contract() {
    return contract_for(
        {{"write_file", Json{{"path", kString}, {"content", kString}}},
         {"run_shell", Json{{"command", kString}, {"timeout_s", Json{{"type", "integer"}}}}}});
}

// pi's write tool (window 4 check a).
Contract write_contract() {
    return contract_for({{"write", Json{{"path", kString}, {"content", kString}}}});
}

Contract bash_contract() { return contract_for({{"bash", Json{{"command", kString}}}}); }

std::string
tool_call(std::string_view name,
          std::initializer_list<std::pair<std::string_view, std::string_view>> parameters = {}) {
    std::string text = "<tool_call>\n<function=" + std::string(name) + ">\n";
    for (const auto& [key, value] : parameters)
        text += "<parameter=" + std::string(key) + ">\n" + std::string(value) + "\n</parameter>\n";
    return text + "</function>\n</tool_call>";
}

struct Decoded {
    bool failed = false;
    std::string content;
    std::vector<ninfer::GeneratedToolCall> calls;
    std::vector<std::string> streamed;
    ninfer::ToolCallParseDiagnostics diagnostics;
    // Most input bytes fed while a call was open and no argument delta was published.
    std::size_t longest_hold = 0;
};

Decoded decode(const Contract& contract, std::string_view text, std::size_t chunk, bool tolerant,
               FinishReason reason) {
    fi::ToolCallOutputDecoder decoder(contract, 64, false, tolerant);
    Decoded out;
    bool open               = false;
    std::size_t since_delta = 0;
    const auto apply        = [&](const std::vector<ninfer::OutputEvent>& events) {
        for (const auto& event : events) {
            std::visit(
                [&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, ninfer::OutputDelta>) {
                        out.content += value.text;
                    } else if constexpr (std::is_same_v<T, ninfer::ToolCallStart>) {
                        out.streamed.emplace_back();
                        open        = true;
                        since_delta = 0;
                    } else if constexpr (std::is_same_v<T, ninfer::ToolArgumentsDelta>) {
                        out.streamed.at(value.index) += value.text;
                        if (!value.text.empty()) since_delta = 0;
                    } else {
                        open = false;
                    }
                },
                event);
        }
    };
    for (std::size_t offset = 0; offset < text.size(); offset += chunk) {
        const auto piece = text.substr(offset, chunk);
        if (open) since_delta += piece.size();
        apply(decoder.feed(piece));
        if (open) out.longest_hold = std::max(out.longest_hold, since_delta);
    }
    apply(decoder.finish(reason));
    out.failed      = decoder.error().has_value();
    out.calls       = decoder.take_tool_calls();
    out.diagnostics = decoder.diagnostics();
    return out;
}

bool same(const Decoded& a, const Decoded& b) {
    return a.failed == b.failed && a.content == b.content && a.calls == b.calls &&
           a.diagnostics == b.diagnostics;
}

// Deltas are append-only: once published they must equal the final arguments, with no retraction
// and no duplicate.
int published_equals_final(const Decoded& decoded, const std::string& label) {
    int failures = check(decoded.streamed.size() == decoded.calls.size(),
                         label + ": call events disagree with the calls");
    for (std::size_t i = 0; i < decoded.calls.size() && i < decoded.streamed.size(); ++i)
        failures += check(decoded.streamed[i] == decoded.calls[i].arguments_json,
                          label + ": published arguments differ from the final call");
    return failures;
}

// Decodes whole, in 1-byte and in 7-byte chunks; every split must give the same result.
Decoded run(int& failures, const std::string& label, const Contract& contract,
            const std::string& text, bool tolerant = true,
            FinishReason reason = FinishReason::StopToken) {
    Decoded whole = decode(contract, text, std::max<std::size_t>(text.size(), 1), tolerant, reason);
    failures += published_equals_final(whole, label);
    for (const std::size_t chunk : {std::size_t{1}, std::size_t{7}}) {
        const Decoded split    = decode(contract, text, chunk, tolerant, reason);
        const std::string name = label + " (chunk " + std::to_string(chunk) + ")";
        failures += published_equals_final(split, name);
        failures += check(same(whole, split), name + ": result depends on the chunking");
        whole.longest_hold = std::max(whole.longest_hold, split.longest_hold);
    }
    return whole;
}

std::size_t complete(const Decoded& decoded) {
    return static_cast<std::size_t>(
        std::count_if(decoded.calls.begin(), decoded.calls.end(), [](const auto& call) {
            return call.status == ninfer::ToolCallStatus::Complete;
        }));
}

std::string argument(const Decoded& decoded, std::size_t call, const std::string& key) {
    if (decoded.failed || call >= decoded.calls.size()) return "<no call>";
    const Json arguments = Json::parse(decoded.calls[call].arguments_json);
    if (!arguments.contains(key)) return "<missing>";
    const Json& value = arguments.at(key);
    return value.is_string() ? value.get<std::string>() : value.dump();
}

bool recovered(const Decoded& decoded, std::size_t calls, Reason reason) {
    return !decoded.failed && complete(decoded) == calls &&
           decoded.diagnostics.recovered_from == reason &&
           decoded.diagnostics.fallback_reason == Reason::None;
}

// ---- Flag off, strict results and the constrained route ----

int test_strict_results_unchanged() {
    int failures                         = 0;
    const auto contract                  = agent_contract();
    const std::string ls                 = tool_call("run_shell", {{"command", "ls"}});
    const std::vector<std::string> texts = {
        ls,
        "Calling.\n" + ls,
        ls + "\n" + tool_call("write_file", {{"path", "a.md"}, {"content", "a"}}),
        tool_call("run_shell", {{"command", "python3 - <<'PY'\npattern = "
                                            "r'<parameter=edits>\\n(.*?)\\n</parameter>'\nPY"},
                                {"timeout_s", "30"}}),
        tool_call("write_file",
                  {{"content", "literal closes: </function> and </tool_call>\n<function=fake>body"
                               "</function>\n<tool_call>body</tool_call>\n<parameter=outer>before"
                               "<parameter=inner>value</parameter>after</parameter>"}}),
        tool_call("write_file", {{"path", "notes/nested.txt"},
                                 {"content", "<function=demo>\n<parameter=mode>\nfast\n</parameter>"
                                             "\n</function>\nend\n"}}),
        tool_call("run_shell", {{"command", "ls"}, {"extra", "{\"a\": [1, 2]}"}}),
        tool_call("run_shell", {{"timeout_s", "not a number"}}),
        tool_call("missing", {{"text", "x"}}),
        "Use the <tool_call> tag for calls.",
        "plain <b>text</b> and </parameter>",
        "<tool_call>\n<funtion=run_shell>\n</function>\n</tool_call>",
        ls + "\n</parameter>",
        "<tool_call>\n<function=run_shell>\n<parameter=command>\nls -la",
    };
    for (std::size_t i = 0; i < texts.size(); ++i) {
        for (const auto reason : {FinishReason::StopToken, FinishReason::OutputLimit}) {
            const std::string label = "strict text " + std::to_string(i) +
                                      (reason == FinishReason::StopToken ? "" : " (limit)");
            const Decoded strict = run(failures, label, contract, texts[i], false, reason);
            const Decoded tolerant =
                run(failures, label + " tolerant", contract, texts[i], true, reason);
            const bool strict_ok =
                !strict.failed &&
                (strict.diagnostics.fallback_reason == Reason::None || strict.calls.empty());
            // Where strict mode succeeds (a call or text), tolerant mode changes nothing.
            if (strict_ok)
                failures += check(same(strict, tolerant), label + ": tolerant mode changed it");
        }
    }
    // A quoted marker without a later call stays text with the strict diagnostics.
    const Decoded quoted = run(failures, "quoted marker only", contract, texts[9]);
    failures += check(!quoted.failed && quoted.calls.empty() && quoted.content == texts[9] &&
                          quoted.diagnostics.fallback_reason == Reason::MalformedStructure &&
                          quoted.diagnostics.recovered_from == Reason::None,
                      "a quoted marker without a call was not kept as text");

    // The flag has no effect on the constrained route.
    const auto constrained = fi::select_tool_call_contract(write_contract(), {});
    failures += check(constrained && constrained->constrained, "test contract is not constrained");
    const std::string bare = tool_call("write", {{"path", "x"}, {"content", "a\n</parameter>\nb"}});
    for (const auto& text : {bare, tool_call("write", {{"path", "x"}, {"content", "a"}})}) {
        const Decoded strict   = run(failures, "constrained", constrained, text, false);
        const Decoded tolerant = run(failures, "constrained tolerant", constrained, text, true);
        failures += check(same(strict, tolerant), "the flag changed the constrained route");
    }
    return failures;
}

// ---- Window 4 check a: file writes that quote the tool markup ----

int test_window4_quoted_markup() {
    int failures                                                 = 0;
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"tool_call_tag",
         "# Parser notes\nA call ends with </tool_call> on its own line.\nDone.\n"},
        {"bare_param_close",
         "Header line\nvalue one\n</parameter>\nThe call ends with </tool_call>.\nvalue two\n"},
        {"nested_param",
         "<function=demo>\n<parameter=mode>\nfast\n</parameter>\n</function>\nend\n"},
    };
    for (const auto& [name, content] : cases) {
        const std::string text =
            tool_call("write", {{"path", "notes/" + name + ".txt"}, {"content", content}});
        const Decoded strict = run(failures, name + " strict", write_contract(), text, false);
        if (name == "bare_param_close")
            failures += check(strict.failed, name + ": strict mode no longer fails");
        else
            failures += check(argument(strict, 0, "content") == content, name + ": strict changed");
        const Decoded tolerant = run(failures, name, write_contract(), text);
        failures += check(complete(tolerant) == 1 && tolerant.content.empty() &&
                              argument(tolerant, 0, "content") == content,
                          name + ": tolerant mode did not write the exact file");
        if (name == "bare_param_close")
            failures += check(tolerant.diagnostics.recovered_from == Reason::MalformedStructure,
                              name + ": recovery was not recorded");
    }
    return failures;
}

// ---- The fork's tolerant parser tests ----

int test_fork_value_recovery() {
    int failures = 0;
    for (const std::string value : {"echo '<parameter=unterminated>'", "echo '</parameter>'"}) {
        const std::string text = tool_call("bash", {{"command", value}});
        const Decoded strict   = run(failures, value + " strict", bash_contract(), text, false);
        failures += check(strict.failed, "strict mode changed: " + value);
        const Decoded tolerant = run(failures, value, bash_contract(), text);
        failures += check(recovered(tolerant, 1, Reason::MalformedStructure) &&
                              argument(tolerant, 0, "command") == value && tolerant.content.empty(),
                          "tolerant mode did not recover: " + value);
    }

    const auto write_file =
        contract_for({{"write_file", Json{{"path", kString},
                                          {"content", kString},
                                          {"overwrite", Json{{"type", "boolean"}}}}}});
    const std::string body = "assert parse('</parameter>') is None\n";
    const Decoded multi =
        run(failures, "multi-parameter", write_file,
            tool_call("write_file",
                      {{"path", "tests/t.py"}, {"content", body}, {"overwrite", "true"}}));
    failures += check(recovered(multi, 1, Reason::MalformedStructure) &&
                          argument(multi, 0, "path") == "tests/t.py" &&
                          argument(multi, 0, "content") == body &&
                          argument(multi, 0, "overwrite") == "true",
                      "a multi-parameter call quoting a close was mis-split");
    return failures;
}

int test_fork_prose_around_calls() {
    int failures         = 0;
    const std::string ls = tool_call("bash", {{"command", "ls"}});

    const std::string quoted    = "Qwen wraps calls in <tool_call> tags. Running it now.\n" + ls;
    const Decoded strict_quoted = run(failures, "quoted strict", bash_contract(), quoted, false);
    failures += check(strict_quoted.calls.empty() && strict_quoted.content == quoted,
                      "strict mode changed for a quoted marker");
    const Decoded q = run(failures, "quoted", bash_contract(), quoted);
    failures +=
        check(recovered(q, 1, Reason::MalformedStructure) && argument(q, 0, "command") == "ls" &&
                  q.content == "Qwen wraps calls in <tool_call> tags. Running it now.",
              "the call after a quoted marker was not recovered: " + q.content);

    const std::string trailing = ls + "\nDone, that lists the directory.";
    failures += check(run(failures, "trailing strict", bash_contract(), trailing, false).failed,
                      "strict mode changed for trailing prose");
    const Decoded t = run(failures, "trailing", bash_contract(), trailing);
    failures +=
        check(recovered(t, 1, Reason::TrailingContent) && argument(t, 0, "command") == "ls" &&
                  t.content == "Done, that lists the directory.",
              "trailing prose was not kept as content: " + t.content);

    // The whitespace held back before the first call separates the trailing prose.
    const Decoded spaced = run(failures, "separator", bash_contract(),
                               "Running the tests.\n" + ls + "\nDone, waiting for output.");
    failures += check(complete(spaced) == 1 &&
                          spaced.content == "Running the tests.\nDone, waiting for output.",
                      "trailing prose lost its separator: " + spaced.content);
    const Decoded glued =
        run(failures, "glued", bash_contract(), "Running the tests." + ls + "Done.");
    failures += check(complete(glued) == 1 && glued.content == "Running the tests.Done.",
                      "glued trailing prose changed: " + glued.content);
    const Decoded noted =
        run(failures, "noted", bash_contract(),
            "Note: <tool_call> is the marker.\n" +
                tool_call("bash", {{"command", "echo '</parameter>'"}}) + "\nThat is all.");
    failures +=
        check(complete(noted) == 1 && argument(noted, 0, "command") == "echo '</parameter>'" &&
                  noted.content == "Note: <tool_call> is the marker.\nThat is all.",
              "decoder content around a quoted marker was wrong: " + noted.content);

    // A stray duplicate close after a complete call is dropped.
    const Decoded stray = run(failures, "stray close", bash_contract(), ls + "\n</tool_call>");
    failures += check(recovered(stray, 1, Reason::TrailingContent) && stray.content.empty(),
                      "a stray </tool_call> became content: " + stray.content);
    return failures;
}

int test_fork_unrecoverable_output() {
    int failures                = 0;
    const std::string truncated = "<tool_call>\n<function=bash>\n<parameter=command>\nls -la";
    failures += check(run(failures, "model stop", bash_contract(), truncated).failed,
                      "a call cut off at the model stop was invented");
    const Decoded limit =
        run(failures, "limit", bash_contract(), truncated, true, FinishReason::OutputLimit);
    failures += check(!limit.failed && limit.calls.size() == 1 && complete(limit) == 0,
                      "a call cut off by the limit is not incomplete");
    return failures;
}

// ---- Production review rounds 1-4: never run a call the model did not issue ----

int test_review_never_runs_a_quoted_call() {
    int failures            = 0;
    const auto contract     = agent_contract();
    const std::string inner = tool_call("run_shell", {{"command", "rm -rf build"}});

    // An undeclared tool whose value quotes a declared call stays text.
    const std::string undeclared =
        "<tool_call>\n<function=write_docs>\n<parameter=text>\nExample:\n" + inner +
        "\n</parameter>\n</function>\n</tool_call>";
    const Decoded a2 = run(failures, "A2", contract, undeclared);
    failures += check(!a2.failed && a2.calls.empty() && a2.content == undeclared &&
                          a2.diagnostics.fallback_reason == Reason::UndeclaredTool,
                      "a call quoted inside an undeclared call was executed");

    // A malformed outer opener is an opening attempt, not prose.
    for (const std::string opener :
         {"<funtion=write_docs>", "<function =write_docs>", "<Function=write_docs>"}) {
        const std::string text = "<tool_call>\n" + opener + "\n<parameter=text>\nExample:\n" +
                                 inner + "\n</parameter>\n</function>\n</tool_call>";
        const Decoded parsed = run(failures, opener, contract, text);
        failures += check(!parsed.failed && parsed.calls.empty() && parsed.content == text,
                          "a call quoted inside a malformed opener ran: " + opener);
    }
    for (const std::string text :
         {"<tool_call>\n" + inner,
          "Wrap it in <tool_call> <b>tags</b>.\n" + tool_call("run_shell", {{"command", "ls"}})}) {
        const Decoded parsed = run(failures, "opener attempt", contract, text);
        failures += check(!parsed.failed && parsed.calls.empty() && parsed.content == text,
                          "a marker followed by markup was skipped as prose");
    }

    // A truncated call whose value quotes an example call: the example's closes never complete it.
    const std::string a1 = "<tool_call>\n<function=write_file>\n<parameter=path>\ndocs/tools.md\n"
                           "</parameter>\n<parameter=content>\n# Tool format\nExample:\n" +
                           inner + "\nThe model then waits for the result and";
    failures += check(run(failures, "A1 stop", contract, a1).failed,
                      "a truncated call borrowed the closes of a quoted example");
    const Decoded a1_limit =
        run(failures, "A1 limit", contract, a1, true, FinishReason::OutputLimit);
    failures += check(!a1_limit.failed && complete(a1_limit) == 0,
                      "a call cut off by the limit was completed");
    const std::string example_value =
        "<tool_call>\n<function=write_file>\n<parameter=path>\nREADME.md\n</parameter>\n"
        "<parameter=content>\nUsage:\n" +
        tool_call("write_file", {{"path", "x"}, {"content", "y"}}) + "\nThe rest of the file";
    failures += check(run(failures, "example value", contract, example_value).failed,
                      "a truncated call was completed with the closes of a quoted example");

    // Trailing prose that holds another call: neither call runs.
    for (const std::string between :
         {"\nNow I will actually run the tests.\n", "\nNow the search.\n"}) {
        const std::string text = "Here is the format:\n" + inner + between +
                                 tool_call("run_shell", {{"command", "grep '</parameter>' a"}});
        failures += check(run(failures, "A3", contract, text).failed,
                          "an example call before the real one was executed");
    }
    const std::string b4 = tool_call("run_shell", {{"command", "ctest"}}) +
                           "\nI wrapped the command in a <tool_call> block.";
    failures += check(run(failures, "B4", contract, b4).failed,
                      "trailing prose naming <tool_call> was recovered");

    // Residual risk, by design: an example call followed only by prose runs.
    const Decoded a4 = run(failures, "A4", contract,
                           "Here is the format:\n" + inner + "\nThat deletes the build dir.");
    failures += check(recovered(a4, 1, Reason::TrailingContent) &&
                          argument(a4, 0, "command") == "rm -rf build" &&
                          a4.content == "Here is the format:\nThat deletes the build dir.",
                      "the documented residual case changed");

    // A value that opens an unclosed <function= cannot take a later close.
    const Decoded b3 =
        run(failures, "B3", contract,
            tool_call("run_shell", {{"command", "grep -rn '<function=' src/ | grep '</parameter>'"},
                                    {"timeout_s", "30"}}));
    failures += check(b3.failed, "a value opening an unclosed <function= was recovered");
    return failures;
}

int test_review_limit_and_call_end() {
    int failures        = 0;
    const auto contract = agent_contract();

    // Output cut off by the limit right after a quoted call end: no recovery.
    const std::string b2 =
        "<tool_call>\n<function=write_file>\n<parameter=path>\ndocs/end.md\n</parameter>\n"
        "<parameter=content>\n# How a call ends\nA call ends with:\n</parameter>\n</function>\n"
        "</tool_call>\nand then the";
    const Decoded cut = run(failures, "B2 limit", contract, b2, true, FinishReason::OutputLimit);
    failures += check(cut.failed, "a call cut off by the output limit was recovered");
    const Decoded stop = run(failures, "B2 stop", contract, b2);
    failures += check(recovered(stop, 1, Reason::TrailingContent) &&
                          argument(stop, 0, "content") == "# How a call ends\nA call ends with:" &&
                          stop.content == "and then the",
                      "control: the same text at the model stop should recover");
    // A complete strict call at the limit stays complete.
    const Decoded g3 = run(failures, "G3", contract, tool_call("run_shell", {{"command", "ctest"}}),
                           true, FinishReason::OutputLimit);
    failures += check(!g3.failed && complete(g3) == 1, "the limit dropped a complete strict call");
    // A recovered call that reaches its end only at the limit is not completed.
    const std::string late =
        "<tool_call>\n<function=write_file>\n<parameter=content>\na '</parameter>' b\n</parameter>"
        "\n</function>\n</tool_call>";
    const Decoded held =
        run(failures, "late limit", contract, late, true, FinishReason::OutputLimit);
    failures += check(!held.failed && held.calls.size() == 1 && complete(held) == 0,
                      "a recovered call was completed at the limit");
    const Decoded open =
        run(failures, "open limit", contract, late.substr(0, late.find(" b\n") + 2), true,
            FinishReason::OutputLimit);
    failures += check(!open.failed && open.calls.size() == 1 && complete(open) == 0 &&
                          open.calls[0].arguments_json == R"({"content":"a '</parameter>' b)",
                      "a recovered call cut off by the limit was not incomplete");

    // A complete call whose value quotes its closing tags (real model output, round 3).
    const std::string body =
        "# How One of My Tool Calls Ends\nThey end in this exact order:\n\n```\n</parameter>\n"
        "</function>\n</tool_call>\n```\n\nThe first tag closes the parameter block.";
    const std::string file_call =
        tool_call("write_file", {{"path", "docs/how-calls-end.md"}, {"content", body}});
    const Decoded one = run(failures, "quoted end", contract, file_call);
    failures += check(recovered(one, 1, Reason::TrailingContent) && one.content.empty() &&
                          argument(one, 0, "content") == body,
                      "a call quoting its closing tags was cut");
    const Decoded two =
        run(failures, "quoted end second", contract,
            tool_call("write_file", {{"path", "a.md"}, {"content", "a"}}) + "\n" + file_call);
    failures +=
        check(complete(two) == 2 && two.content.empty() && argument(two, 1, "content") == body,
              "the second call quoting its closing tags was cut");
    // A real call followed by prose that names </tool_call>.
    failures += check(run(failures, "names close", contract,
                          tool_call("write_file", {{"path", "a.md"}, {"content", "a"}}) +
                              "\nEach call ends with </tool_call>.")
                          .failed,
                      "prose naming </tool_call> after a call was recovered");
    return failures;
}

// ---- Production review harness (pr14-adv v4): behavior cases ----

int test_harness_cases() {
    int failures        = 0;
    const auto contract = agent_contract();

    // A5: a fake complete split is a best guess, the same as production.
    const Decoded a5 = run(failures, "A5", contract,
                           "<tool_call>\n<function=write_file>\n<parameter=content>\ndocs: "
                           "</parameter>\n<parameter=path>x</parameter> trailing\n</parameter>\n"
                           "<parameter=path>\nreal.md\n</parameter>\n</function>\n</tool_call>");
    failures += check(complete(a5) == 1 && argument(a5, 0, "content") == "docs: " &&
                          argument(a5, 0, "path") ==
                              "x</parameter> trailing\n</parameter>\n<parameter=path>\nreal.md",
                      "A5 split changed");

    // A6: 20 calls with three quoted closes each.
    std::string many;
    const std::string command =
        "grep '</parameter>' a; grep '</parameter>' b; grep '</parameter>' c";
    for (int i = 0; i < 20; ++i)
        many += tool_call("run_shell", {{"command", command}, {"timeout_s", "30"}}) + "\n";
    const Decoded a6 = run(failures, "A6", contract, many);
    bool exact       = complete(a6) == 20;
    for (std::size_t i = 0; exact && i < 20; ++i)
        exact = argument(a6, i, "command") == command && argument(a6, i, "timeout_s") == "30";
    failures += check(exact && a6.content.empty(), "A6 calls were not recovered exactly");

    // A7: Unicode around a quoted close.
    const std::string unicode = "\xe2\x9c\x93 </parameter> \xf0\x9f\x9a\x80";
    const Decoded a7 =
        run(failures, "A7", contract,
            tool_call("write_file", {{"path", "docs/\xc3\xa6\xc3\xb8.md"}, {"content", unicode}}));
    failures += check(complete(a7) == 1 && argument(a7, 0, "content") == unicode,
                      "A7 Unicode value changed");

    // C1: a quoted marker in prose, then a call quoting a close.
    const Decoded c1 =
        run(failures, "C1", contract,
            "I put each command in a <tool_call> block.\n" +
                tool_call("run_shell", {{"command", "grep -rn '</parameter>' src/"}}));
    failures += check(recovered(c1, 1, Reason::MalformedStructure) &&
                          c1.content == "I put each command in a <tool_call> block." &&
                          argument(c1, 0, "command") == "grep -rn '</parameter>' src/",
                      "C1 was not recovered");

    // C4: a complete write_file quoting a complete call needs no recovery.
    const std::string example =
        "# Format\n" + tool_call("run_shell", {{"command", "rm -rf build"}}) + "\nEnd of example.";
    const Decoded c4 =
        run(failures, "C4", contract,
            tool_call("write_file", {{"path", "docs/format.md"}, {"content", example}}));
    failures += check(complete(c4) == 1 && argument(c4, 0, "content") == example &&
                          c4.diagnostics.recovered_from == Reason::None,
                      "C4 file quoting a complete call changed");

    // P11: a document that quotes the close triple 40 times recovers in full.
    std::string blocks;
    for (int i = 0; i < 40; ++i)
        blocks += "```\n</parameter>\n</function>\n</tool_call>\n```\nThat ends a call.\n";
    const Decoded p11 =
        run(failures, "P11", contract,
            "<tool_call>\n<function=write_file>\n<parameter=path>\nc.md\n</parameter>\n"
            "<parameter=content>\n" +
                blocks + "</parameter>\n</function>\n</tool_call>");
    failures += check(complete(p11) == 1 &&
                          argument(p11, 0, "content") == blocks.substr(0, blocks.size() - 1),
                      "P11 document was not recovered in full");
    return failures;
}

// Pathological outputs fail fast: the attempt budget bounds the lookahead scans.
int test_harness_costs() {
    int failures        = 0;
    const auto contract = agent_contract();
    const auto timed = [&](const std::string& label, const std::string& text, bool expect_failure,
                           std::size_t chunk) {
        const auto begin   = std::chrono::steady_clock::now();
        const Decoded out  = decode(contract, text, chunk, true, FinishReason::StopToken);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - begin)
                                 .count();
        failures += published_equals_final(out, label);
        failures += check(out.failed == expect_failure, label + ": unexpected outcome");
        failures += check(elapsed < 1000, label + " took " + std::to_string(elapsed) + " ms");
        return out;
    };
    const std::string open = "<tool_call>\n<function=run_shell>\n<parameter=command>\n";
    const auto repeat      = [](std::string text, std::string_view unit, std::size_t size) {
        while (text.size() < size) text += unit;
        return text;
    };
    for (const std::size_t chunk : {std::size_t{1} << 30, std::size_t{4096}}) {
        const std::string suffix = chunk == 4096 ? " (streamed)" : "";
        timed("P1" + suffix, repeat(open, "x</parameter>\n<parameter=command>", 1u << 20), true,
              chunk);
        timed("P2" + suffix,
              repeat({},
                     "<tool_call>\n<function=run_shell>\n<parameter=command>\na</parameter>\n"
                     "<parameter=timeout_s>\nb</parameter>\n<parameter=command>\nc</parameter>"
                     "\n</function>\n",
                     1u << 20),
              true, chunk);
        timed("P3" + suffix, repeat(open, "y</parameter>\n</function>\n", 1u << 18), true, chunk);
        timed("P4" + suffix, repeat(open, "<parameter=a>", 1u << 20) + "</parameter>", true, chunk);
        timed(
            "P5" + suffix,
            repeat({}, "<tool_call><function=run_shell><parameter=command>z</parameter>", 1u << 18),
            true, chunk);
        timed("P6" + suffix,
              repeat(open + "<function=x ", "q</parameter>\n<parameter=command>", 1u << 20), true,
              chunk);
        timed("write_file 1 MB" + suffix,
              repeat("<tool_call>\n<function=write_file>\n<parameter=content>\n",
                     "x\n</parameter>\n<parameter=p>\n", 1u << 20),
              true, chunk);
        std::string p8;
        for (int i = 0; i < 2000; ++i)
            p8 +=
                tool_call("run_shell", {{"command", "grep '</parameter>' a; grep '</function>' b"},
                                        {"timeout_s", "30"}}) +
                "\n";
        const Decoded calls = timed("P8" + suffix, p8, false, chunk);
        failures += check(complete(calls) == 2000, "P8 lost calls");
        std::string p9 =
            "<tool_call>\n<function=write_file>\n<parameter=path>\na.md\n</parameter>\n"
            "<parameter=content>\nx";
        for (int i = 0; i < 28000; ++i) p9 += "\n</parameter>\n</function>\n</tool_call>";
        timed("P9" + suffix, p9 + "\nprose at the very end", true, chunk);
        std::string p10 =
            "<tool_call>\n<function=write_file>\n<parameter=path>\nb.md\n</parameter>\n"
            "<parameter=content>\n";
        for (int i = 0; i < 5000; ++i)
            p10 += "```\n</parameter>\n</function>\n</tool_call>\n```\nThat ends a call.\n";
        timed("P10" + suffix, p10 + "</parameter>\n</function>\n</tool_call>", true, chunk);
    }
    return failures;
}

// ---- Real model output from the round 3 GPU gate (em dashes replaced by '-') ----

const std::vector<std::pair<std::string, std::string>>& replay_texts() {
    static const std::vector<std::pair<std::string, std::string>> texts = {
        {"real-r3-Q1-closing-fragment",
         R"pr14(<tool_call>
<function=write_file>
<parameter=path>
docs/how-calls-end.md
</parameter>
<parameter=content>
# How One of My Tool Calls Ends

When I finish a tool call, the very last thing I write is a short sequence of three closing tags. Each one appears on its own line, in this exact order:

```
</parameter>
</function>
</tool_call>
```

The first tag closes the parameter block that held the arguments for the call. The second closes the function element that named the tool and wrapped those parameters. The third closes the outer tool-call wrapper that the runtime uses to recognize the whole invocation as a single unit. No opening tags are needed here; the runtime has already seen them earlier in the message, so only these closers are required to signal that the call is complete.

After the final closing tag lands, the runtime parses the accumulated XML into a structured request: it extracts the function name, validates that every required parameter is present and correctly typed, and rejects the call if anything is malformed or missing. Only once validation succeeds does the runtime dispatch the call to the matching handler, which performs the actual work - reading or writing files, running a shell command, fetching weather, and so on. If the handler raises an error or times out, that failure is captured rather than crashing the session, so the conversation can continue.

The handler's result is then serialized back into the transcript as a tool-result message that I will see on my next turn. That result becomes part of my context, letting me reason about what happened, decide whether to issue another call, or answer the user directly. In other words, the three closing tags are the hand-off point: they mark the boundary where my authored request ends and the runtime's execution begins, and everything after that boundary - validation, dispatch, execution, and the returned result - happens outside of what I wrote.

</parameter>
</function>
</tool_call>)pr14"},
        {"real-r3-Q2-example-call",
         R"pr14(<tool_call>
<function=write_file>
<parameter=path>
docs/tool-call-format.md
</parameter>
<parameter=content>
# Tool Call Format

This document describes the exact markup used for tool calls. A tool call is
wrapped in a `tool_call` block that contains a single `invoke` element named
after the target function. Each parameter is expressed as a `parameter`
element whose `name` attribute identifies the argument and whose text content
is the value. The complete, self-contained example below shows a call to
`run_shell` with the command `ls -la`:

```text
<tool_call>
  <invoke name="run_shell">
    <parameter name="command">ls -la</parameter>
  </invoke>
</tool_call>
```

The runtime first performs a lightweight scan of the model output for the
opening `tool_call` tag. When it finds one, it switches from plain-text
streaming mode into structured-parsing mode and buffers everything until the
matching closing `tool_call` tag is seen. This buffering guarantees that a
partial or malformed invocation is never dispatched: if the closing tag is
missing, or the inner XML is not well-formed, the entire block is rejected
and surfaced to the user as a formatting error rather than being executed.

Inside the buffered block, the parser locates the `invoke` element and reads
its `name` attribute to resolve the target function against the registry of
tools available in the current session. It then walks each child `parameter`
element in document order, pairing the `name` attribute with the trimmed text
content to build an ordered argument list. Parameters may appear in any order
in the markup; the parser matches them by name, so reordering does not change
semantics. Any parameter that is declared required in the tool's schema but
absent from the markup causes the call to be rejected before execution.

Once the arguments are assembled, the runtime validates them against the
tool's JSON schema - checking types, enums, and constraints such as minimum
lengths - and only after validation succeeds does it hand the call to the
executor. The executor runs the function, captures its result, and returns it
to the model as a new message so the conversation can continue. Because every
stage (tag detection, XML parsing, name resolution, schema validation) must
pass cleanly, the markup shown above is both the human-readable contract and
the machine-enforced one: deviating from it in any way results in the call
being refused, which keeps the system safe even when the model produces
slightly off-spec output.

</parameter>
</function>
</tool_call>)pr14"},
        {"real-r3-T4-grep",
         R"pr14(<tool_call>
<function=run_shell>
<parameter=command>
grep -rn '</parameter>' src/
</parameter>
<parameter=timeout_s>
30
</parameter>
</function>
</tool_call>)pr14"},
        {"mention-close-after-quoted-call",
         R"pr14(<tool_call>
<function=run_shell>
<parameter=command>
grep '</parameter>' a
</parameter>
</function>
</tool_call>
I ended the call with </tool_call> as usual.)pr14"},
        {"mention-close-after-plain-call",
         R"pr14(<tool_call>
<function=run_shell>
<parameter=command>
ls
</parameter>
</function>
</tool_call>
I ended the call with </tool_call> as usual.)pr14"},
        {"two-calls-second-quotes-close-fragment",
         R"pr14(<tool_call>
<function=run_shell>
<parameter=command>
ls
</parameter>
</function>
</tool_call>
<tool_call>
<function=write_file>
<parameter=path>
x.md
</parameter>
<parameter=content>
ends with:
</parameter>
</function>
</tool_call>
then more
</parameter>
</function>
</tool_call>)pr14"},
    };
    return texts;
}

std::string text_between(const std::string& text, std::string_view open, std::string_view close) {
    const auto begin = text.rfind(open) + open.size();
    return text.substr(begin, text.rfind(close) - begin);
}

int test_real_output_replay() {
    int failures        = 0;
    const auto contract = agent_contract();
    for (const auto& [id, text] : replay_texts()) {
        const Decoded out = run(failures, id, contract, text);
        if (id.starts_with("mention-close")) {
            failures += check(out.failed, id + ": prose naming </tool_call> was recovered");
        } else if (id == "real-r3-T4-grep") {
            failures += check(recovered(out, 1, Reason::MalformedStructure) &&
                                  argument(out, 0, "command") == "grep -rn '</parameter>' src/" &&
                                  argument(out, 0, "timeout_s") == "30",
                              id + ": not recovered");
        } else {
            const std::size_t calls = id.starts_with("two-calls") ? 2 : 1;
            const Reason reason     = id == "real-r3-Q2-example-call" ? Reason::MalformedStructure
                                                                      : Reason::TrailingContent;
            failures += check(recovered(out, calls, reason) && out.content.empty() &&
                                  argument(out, calls - 1, "content") ==
                                      text_between(text, "<parameter=content>\n", "\n</parameter>"),
                              id + ": the file was not recovered in full");
        }
    }
    return failures;
}

// ---- Holds: bounded by the call's own value, published deltas equal the final arguments ----

int test_holds() {
    int failures = 0;
    // A file that quotes a nested parameter block early and is long after it: the nested close
    // followed by a parameter header is a later candidate, so the rest of the value is held until
    // the balanced close.
    std::string body(64 * 1024, 'a');
    for (std::size_t i = 79; i < body.size(); i += 80) body[i] = '\n';
    const std::string content =
        "Config:\n<parameter=mode>\nfast\n</parameter>\n<parameter=speed>\nhigh\n</parameter>\n" +
        body + "\nend";
    const std::string text = tool_call("write", {{"path", "config.md"}, {"content", content}});
    const Decoded out      = run(failures, "long hold", write_contract(), text);
    failures += check(complete(out) == 1 && argument(out, 0, "content") == content &&
                          out.diagnostics.recovered_from == Reason::None,
                      "a held value was not published exactly");
    failures += check(out.longest_hold >= body.size(),
                      "the nested candidate did not hold the value (hold " +
                          std::to_string(out.longest_hold) + " bytes)");

    // Without the nested candidate the value streams.
    const std::string plain = tool_call("write", {{"path", "notes.md"}, {"content", body}});
    const Decoded streamed  = run(failures, "streamed", write_contract(), plain);
    failures +=
        check(complete(streamed) == 1 && streamed.longest_hold < 64,
              "a plain value was held (hold " + std::to_string(streamed.longest_hold) + " bytes)");

    // Interrupted inside the hold: the held bytes are published as value text, as in strict mode.
    const std::string cut = text.substr(0, text.find(body) + body.size() / 2);
    const Decoded limit =
        run(failures, "hold limit", write_contract(), cut, true, FinishReason::OutputLimit);
    const Decoded strict =
        run(failures, "hold limit strict", write_contract(), cut, false, FinishReason::OutputLimit);
    failures += check(!limit.failed && complete(limit) == 0 && same(limit, strict),
                      "an interrupted hold differs from strict mode");

    // Model stop inside a value with an unclosed nested opener: the earliest nested close that
    // reaches a real call end wins.
    const Decoded unclosed =
        run(failures, "unclosed opener", write_contract(),
            tool_call("write", {{"path", "grep.sh"}, {"content", "grep '<parameter=' x > y"}}));
    failures += check(recovered(unclosed, 1, Reason::MalformedStructure) &&
                          argument(unclosed, 0, "content") == "grep '<parameter=' x > y",
                      "an unclosed nested opener was not recovered");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_strict_results_unchanged();
    failures += test_window4_quoted_markup();
    failures += test_fork_value_recovery();
    failures += test_fork_prose_around_calls();
    failures += test_fork_unrecoverable_output();
    failures += test_review_never_runs_a_quoted_call();
    failures += test_review_limit_and_call_end();
    failures += test_harness_cases();
    failures += test_harness_costs();
    failures += test_real_output_replay();
    failures += test_holds();
    if (failures == 0) std::cout << "ok\n";
    return failures == 0 ? 0 : 1;
}
