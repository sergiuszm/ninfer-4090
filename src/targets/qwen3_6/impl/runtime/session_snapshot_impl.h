#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// Retained-session snapshot format (target-private).
//
// A snapshot is the complete host image of one catalogued continuation: the resident prefix
// (ledger + identity + shortlist digests), the paged Text/backend KV payload in logical page
// order, and the endpoint StateImage (GDN conv/recurrent state, continuation hidden, and any
// DFlash local state per the image layout). Byte order is the host's (x86 little-endian); a
// snapshot binds to the exact weights identity and KV configuration, so cross-endian
// portability is intentionally out of scope.
//
// Versions 1 and 2 described the pre-reconciliation lane-retained format (v2 appended the
// turn-checkpoint ring). Both are rejected by this build: the physical KV layout and the
// state model changed with the upstream reconciliation, so old files cannot be re-landed.
// Version 3 is the continuation-catalog format. Beside the endpoint it persists the rewrite
// checkpoint and long anchors (each an extra StateImage; the KV payload already covers every
// checkpoint frontier) - a multi-turn continuation diverges from the resident ledger just
// before the endpoint (the assistant header renders differently once the reply is input), so
// reuse of a restored session rides those turn-boundary checkpoints exactly as it does for a
// warm one. Restore degrades gracefully: checkpoints whose StateImage does not fit the state
// pools, or whose anchor ordinal exceeds the server's configured capacity, are dropped while
// the endpoint remains mandatory.

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr char kSessionSnapshotMagic[8]         = {'N', 'I', 'N', 'F', 'S', 'E', 'S', '1'};
constexpr std::uint32_t kSessionSnapshotVersion = 3;

constexpr std::uint32_t kKvFlagPackedV   = 1U << 0;
constexpr std::uint32_t kKvFlagRotateK   = 1U << 1;
constexpr std::uint32_t kKvFlagRotateV   = 1U << 2;
constexpr std::uint32_t kKvFlagPackedK   = 1U << 3;
constexpr std::uint32_t kKvFlagE8Lattice = 1U << 4;
constexpr std::uint32_t kKvFlagE8Root    = 1U << 5;

// The header keeps the pre-merge (dtype, quant group, packing flags) triple so files written by
// earlier builds stay restorable. All three derive from the storage enum: every int8-family mode
// reported DType::I8 at group 64 and carried its packing in the flags.
struct SnapshotKvProfile {
    DType dtype;
    std::int32_t quant_group;
    std::uint32_t flags;
};

inline SnapshotKvProfile snapshot_kv_profile(KvCacheStorage storage) {
    const KvForkModeFlags mode = kv_fork_mode_flags(storage);
    const std::uint32_t flags =
        (mode.packed_v ? kKvFlagPackedV : 0U) | (mode.rotate_k ? kKvFlagRotateK : 0U) |
        (mode.rotate_v ? kKvFlagRotateV : 0U) | (mode.packed_k ? kKvFlagPackedK : 0U) |
        (mode.e8_lattice ? kKvFlagE8Lattice : 0U) | (mode.e8_root ? kKvFlagE8Root : 0U);
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {DType::BF16, 0, flags};
    case KvCacheStorage::Fp8E4M3Row256:
        return {DType::FP8_E4M3FN, kKvFp8QuantGroup, flags};
    case KvCacheStorage::Int8Group64:
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
    case KvCacheStorage::RK4V4E8:
    case KvCacheStorage::RK2V4E8:
        return {DType::I8, kKvInt8QuantGroup, flags};
    case KvCacheStorage::Nvfp4Group16:
    case KvCacheStorage::Fp8KeyNvfp4Value:
        // Not servable on this build; the value only needs to be distinct and stable.
        return {DType::U8, 16, flags};
    }
    throw std::logic_error("unknown KV-cache storage");
}

class SnapshotWriter {
public:
    explicit SnapshotWriter(std::vector<std::uint8_t>& out) : out_(out) {}

    void bytes(const void* data, std::size_t count) {
        const auto* begin = static_cast<const std::uint8_t*>(data);
        out_.insert(out_.end(), begin, begin + count);
    }

    template <class T>
    void pod(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(T));
    }

    // Reserves a device-payload region and returns its offset; the caller fills it with
    // cudaMemcpyAsync once the full host image is sized (the vector no longer reallocates).
    std::size_t reserve_payload(std::size_t count) {
        const std::size_t offset = out_.size();
        out_.resize(out_.size() + count);
        return offset;
    }

private:
    std::vector<std::uint8_t>& out_;
};

class SnapshotReader {
public:
    explicit SnapshotReader(std::span<const std::uint8_t> data) : data_(data) {}

    void bytes(void* out, std::size_t count) {
        if (count > data_.size() - cursor_) {
            throw std::invalid_argument("session snapshot is truncated");
        }
        std::memcpy(out, data_.data() + cursor_, count);
        cursor_ += count;
    }

    template <class T>
    [[nodiscard]] T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        bytes(&value, sizeof(T));
        return value;
    }

    // Borrows a device-payload region without copying; valid for the snapshot's lifetime.
    [[nodiscard]] const std::uint8_t* payload(std::size_t count) {
        if (count > data_.size() - cursor_) {
            throw std::invalid_argument("session snapshot is truncated");
        }
        const std::uint8_t* region = data_.data() + cursor_;
        cursor_ += count;
        return region;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - cursor_; }

private:
    std::span<const std::uint8_t> data_;
    std::size_t cursor_ = 0;
};

template <class T>
void write_vector(SnapshotWriter& writer, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    writer.pod<std::uint64_t>(values.size());
    writer.bytes(values.data(), values.size() * sizeof(T));
}

template <class T>
std::vector<T> read_vector(SnapshotReader& reader, std::size_t maximum_count, const char* label) {
    const std::uint64_t count = reader.pod<std::uint64_t>();
    if (count > maximum_count) {
        throw std::invalid_argument(std::string("session snapshot ") + label +
                                    " count is out of range");
    }
    std::vector<T> values(static_cast<std::size_t>(count));
    reader.bytes(values.data(), values.size() * sizeof(T));
    return values;
}

void write_vision_items(SnapshotWriter& writer, const std::vector<VisionItem>& items) {
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(items.size()));
    for (const VisionItem& item : items) {
        writer.pod<std::uint8_t>(static_cast<std::uint8_t>(item.modality));
        writer.pod<std::int32_t>(item.grid.temporal);
        writer.pod<std::int32_t>(item.grid.height);
        writer.pod<std::int32_t>(item.grid.width);
        writer.pod<std::uint64_t>(item.patch_begin);
        writer.pod<std::uint64_t>(item.patch_count);
        writer.bytes(item.content_digest.data(), item.content_digest.size());
        write_vector(writer, item.timestamps);
        writer.pod<std::uint32_t>(static_cast<std::uint32_t>(item.token_spans.size()));
        for (const TokenSpan& span : item.token_spans) {
            writer.pod<std::uint64_t>(span.begin);
            writer.pod<std::uint64_t>(span.count);
        }
    }
}

std::vector<VisionItem> read_vision_items(SnapshotReader& reader, std::size_t tokens) {
    const std::uint32_t count = reader.pod<std::uint32_t>();
    if (count > tokens) {
        throw std::invalid_argument("session snapshot vision item count is out of range");
    }
    std::vector<VisionItem> items(count);
    for (VisionItem& item : items) {
        item.modality      = static_cast<PromptModality>(reader.pod<std::uint8_t>());
        item.grid.temporal = reader.pod<std::int32_t>();
        item.grid.height   = reader.pod<std::int32_t>();
        item.grid.width    = reader.pod<std::int32_t>();
        item.patch_begin   = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        item.patch_count   = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        reader.bytes(item.content_digest.data(), item.content_digest.size());
        item.timestamps = read_vector<double>(reader, tokens, "vision timestamp");
        const std::uint32_t spans = reader.pod<std::uint32_t>();
        if (spans > tokens) {
            throw std::invalid_argument("session snapshot vision span count is out of range");
        }
        item.token_spans.resize(spans);
        for (TokenSpan& span : item.token_spans) {
            span.begin = static_cast<std::size_t>(reader.pod<std::uint64_t>());
            span.count = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        }
    }
    return items;
}

struct SnapshotConfig {
    std::uint32_t kv_dtype             = 0;
    std::int32_t kv_quant_group        = 0;
    std::uint32_t kv_flags             = 0;
    std::uint32_t speculative_backend  = 0;
    std::uint32_t draft_window         = 0;
    std::uint32_t page_size            = 0;
    std::uint64_t state_image_bytes    = 0;
    std::uint32_t text_plane_count     = 0;
    std::uint64_t text_page_stride     = 0;
    std::uint32_t backend_plane_count  = 0;
    std::uint64_t backend_page_stride  = 0;
};

struct SnapshotSession {
    std::uint32_t tokens                     = 0;
    std::uint32_t execution_frontier         = 0;
    std::uint32_t ledger_frontier            = 0;
    std::uint32_t text_kv_valid              = 0;
    std::uint32_t mtp_kv_valid               = 0;
    std::int32_t rope_delta                  = 0;
    std::uint8_t tail_hidden_valid           = 0;
    std::uint32_t text_committed_frontier    = 0;
    std::uint32_t backend_committed_frontier = 0;
    std::uint32_t text_pages                 = 0;
    std::uint32_t backend_pages              = 0;
    runtime::PrefillWork rebuild_work;
    std::uint32_t rebuild_tail_begin = 0;
};

void write_config(SnapshotWriter& writer, const SnapshotConfig& config) {
    writer.pod(config.kv_dtype);
    writer.pod(config.kv_quant_group);
    writer.pod(config.kv_flags);
    writer.pod(config.speculative_backend);
    writer.pod(config.draft_window);
    writer.pod(config.page_size);
    writer.pod(config.state_image_bytes);
    writer.pod(config.text_plane_count);
    writer.pod(config.text_page_stride);
    writer.pod(config.backend_plane_count);
    writer.pod(config.backend_page_stride);
}

SnapshotConfig read_config(SnapshotReader& reader) {
    SnapshotConfig config;
    config.kv_dtype            = reader.pod<std::uint32_t>();
    config.kv_quant_group      = reader.pod<std::int32_t>();
    config.kv_flags            = reader.pod<std::uint32_t>();
    config.speculative_backend = reader.pod<std::uint32_t>();
    config.draft_window        = reader.pod<std::uint32_t>();
    config.page_size           = reader.pod<std::uint32_t>();
    config.state_image_bytes   = reader.pod<std::uint64_t>();
    config.text_plane_count    = reader.pod<std::uint32_t>();
    config.text_page_stride    = reader.pod<std::uint64_t>();
    config.backend_plane_count = reader.pod<std::uint32_t>();
    config.backend_page_stride = reader.pod<std::uint64_t>();
    return config;
}

void write_session(SnapshotWriter& writer, const SnapshotSession& session) {
    writer.pod(session.tokens);
    writer.pod(session.execution_frontier);
    writer.pod(session.ledger_frontier);
    writer.pod(session.text_kv_valid);
    writer.pod(session.mtp_kv_valid);
    writer.pod(session.rope_delta);
    writer.pod(session.tail_hidden_valid);
    writer.pod(session.text_committed_frontier);
    writer.pod(session.backend_committed_frontier);
    writer.pod(session.text_pages);
    writer.pod(session.backend_pages);
    writer.pod(session.rebuild_work);
    writer.pod(session.rebuild_tail_begin);
}

SnapshotSession read_session(SnapshotReader& reader) {
    SnapshotSession session;
    session.tokens                     = reader.pod<std::uint32_t>();
    session.execution_frontier         = reader.pod<std::uint32_t>();
    session.ledger_frontier            = reader.pod<std::uint32_t>();
    session.text_kv_valid              = reader.pod<std::uint32_t>();
    session.mtp_kv_valid               = reader.pod<std::uint32_t>();
    session.rope_delta                 = reader.pod<std::int32_t>();
    session.tail_hidden_valid          = reader.pod<std::uint8_t>();
    session.text_committed_frontier    = reader.pod<std::uint32_t>();
    session.backend_committed_frontier = reader.pod<std::uint32_t>();
    session.text_pages                 = reader.pod<std::uint32_t>();
    session.backend_pages              = reader.pod<std::uint32_t>();
    session.rebuild_work               = reader.pod<runtime::PrefillWork>();
    session.rebuild_tail_begin         = reader.pod<std::uint32_t>();
    return session;
}

// Session identity: FNV-1a 64 over the resident ledger's token bytes, rendered as 16 hex
// chars. Deterministic across processes on one endianness, which snapshot compatibility
// already requires. The shared prefix form lives in program_impl.h so checkpoint digests
// hash identically.
std::string ledger_digest(const std::vector<TokenId>& ledger) {
    return ledger_prefix_digest(std::span<const TokenId>(ledger.data(), ledger.size()));
}

} // namespace

std::uint32_t
ProgramImplCore::continuation_depth(const ContinuationHandle& continuation) const noexcept {
    if (!valid_continuation(continuation)) { return 0; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    return static_cast<std::uint32_t>(sequence.ledger.size());
}

std::string ProgramImplCore::continuation_digest(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) { return {}; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    return ledger_digest(sequence.ledger);
}

std::vector<SlotCheckpoint>
ProgramImplCore::continuation_checkpoints(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) { return {}; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    const std::uint32_t depth     = static_cast<std::uint32_t>(sequence.ledger.size());

    std::vector<std::uint32_t> frontiers;
    frontiers.reserve(sequence.long_anchors.size() + 2U);
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        frontiers.push_back(anchor.frontier);
    }
    if (sequence.rewrite_checkpoint.valid) {
        frontiers.push_back(sequence.rewrite_checkpoint.frontier);
    }
    if (sequence.endpoint_valid) { frontiers.push_back(sequence.ledger_frontier); }
    std::sort(frontiers.begin(), frontiers.end());
    frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());

    std::vector<SlotCheckpoint> out;
    out.reserve(frontiers.size());
    for (const std::uint32_t frontier : frontiers) {
        if (frontier == 0 || frontier > depth) { continue; }
        out.push_back(SlotCheckpoint{
            frontier, ledger_prefix_digest(
                          std::span<const TokenId>(sequence.ledger.data(), frontier))});
    }
    return out;
}

qwen3_6::ContinuationSummary
ProgramImplCore::continuation_summary(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) {
        throw std::invalid_argument("continuation holds no retained session");
    }
    return continuation_summary(continuation_states[ContractAccess::index(continuation)]);
}

qwen3_6::RetainedSessionSnapshot
ProgramImplCore::save_continuation(const ContinuationHandle& continuation,
                                   std::string_view model_binding) {
    if (!valid_continuation(continuation)) {
        throw std::invalid_argument("continuation holds no retained session");
    }
    if (model_binding.size() > 4096) {
        throw std::invalid_argument("session snapshot model binding is too long");
    }
    if (has_context_transaction() || pending_transaction_) {
        throw std::logic_error("cannot snapshot a session during a resource transaction");
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument("session persistence does not support the DFlash backend");
    }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];

    const std::size_t tokens = sequence.ledger.size();
    if (tokens == 0 || tokens > capacity || sequence.prefix_identity.size() != tokens ||
        sequence.prefix_digests.size() != tokens ||
        sequence.ledger_frontier != tokens || sequence.execution_frontier > tokens ||
        tokens - sequence.execution_frontier > 1 || !sequence.endpoint_valid || !sequence.kv) {
        throw std::logic_error("retained session ledger and identity are inconsistent");
    }
    if (sequence.state.fork_pending || sequence.state.read != sequence.state.write ||
        !state_store->valid(sequence.state.read)) {
        throw std::logic_error("retained session state binding is not a settled endpoint");
    }
    const StateImageHandle state             = sequence.state.read;
    const StateImageHostLayout& state_layout = state_images->host_layout();

    // Checkpoint directory: every checkpoint names one entry in a deduplicated StateImage
    // table (a rewrite checkpoint or anchor may alias the endpoint image).
    std::vector<StateImageHandle> unique_states;
    unique_states.reserve(2U + sequence.long_anchors.size());
    const auto image_index = [&](StateImageHandle handle) -> std::int32_t {
        if (!state_store->valid(handle) ||
            state_store->residency(handle) == StateReplicaResidency::None) {
            throw std::logic_error("retained session StateImage has no published replica");
        }
        for (std::size_t index = 0; index < unique_states.size(); ++index) {
            if (unique_states[index] == handle) { return static_cast<std::int32_t>(index); }
        }
        unique_states.push_back(handle);
        return static_cast<std::int32_t>(unique_states.size() - 1U);
    };
    const std::int32_t endpoint_image = image_index(state);
    std::int32_t rewrite_image        = -1;
    if (sequence.rewrite_checkpoint.valid) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("retained rewrite checkpoint has no StateImage");
        }
        rewrite_image = image_index(*sequence.rewrite_state);
    }
    std::vector<std::int32_t> anchor_images;
    anchor_images.reserve(sequence.long_anchors.size());
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        anchor_images.push_back(image_index(anchor.state));
    }

    const DeviceKVPagePool& text_pool = text_kv_pages->physical_pool();
    const HostKVPageLayout text_layout = plan_host_kv_page_layout(text_pool.geometry());
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    std::optional<HostKVPageLayout> backend_layout;
    if (backend != nullptr) {
        backend_layout = plan_host_kv_page_layout(backend->page_pool().geometry());
    }

    const KVAddressSpaceHandle text_address = sequence.kv->text;
    const std::uint32_t text_committed = text_kv_addresses->committed_frontier(text_address);
    const std::uint32_t text_pages     = kv_pages_for_frontier(text_committed);
    if (text_pages == 0 || text_pages > text_kv_addresses->mapped_pages(text_address) ||
        sequence.execution_frontier > text_committed) {
        throw std::logic_error("retained session Text KV coverage is inconsistent");
    }
    std::uint32_t backend_committed = 0;
    std::uint32_t backend_pages     = 0;
    if (backend != nullptr) {
        if (!sequence.kv->backend) {
            throw std::logic_error("retained session has no backend KV address");
        }
        backend_committed = backend_kv_addresses->committed_frontier(*sequence.kv->backend);
        backend_pages     = kv_pages_for_frontier(backend_committed);
        if (backend_pages > backend_kv_addresses->mapped_pages(*sequence.kv->backend)) {
            throw std::logic_error("retained session backend KV coverage is inconsistent");
        }
    } else if (sequence.kv->backend) {
        throw std::logic_error("retained session backend KV address has no backing cache");
    }

    const SnapshotKvProfile kv_profile = snapshot_kv_profile(kv_storage);
    SnapshotConfig config;
    config.kv_dtype            = static_cast<std::uint32_t>(kv_profile.dtype);
    config.kv_quant_group      = kv_profile.quant_group;
    config.kv_flags            = kv_profile.flags;
    config.speculative_backend = static_cast<std::uint32_t>(speculative_backend);
    config.draft_window        = draft_window;
    config.page_size           = static_cast<std::uint32_t>(kPagedKVPageSize);
    config.state_image_bytes   = state_layout.image_bytes;
    config.text_plane_count    = static_cast<std::uint32_t>(text_pool.plane_count());
    config.text_page_stride    = text_layout.page_stride;
    if (backend != nullptr) {
        config.backend_plane_count =
            static_cast<std::uint32_t>(backend->page_pool().plane_count());
        config.backend_page_stride = backend_layout->page_stride;
    }

    SnapshotSession session;
    session.tokens                     = static_cast<std::uint32_t>(tokens);
    session.execution_frontier         = sequence.execution_frontier;
    session.ledger_frontier            = sequence.ledger_frontier;
    session.text_kv_valid              = sequence.text_kv_valid;
    session.mtp_kv_valid               = sequence.mtp_kv_valid;
    session.rope_delta                 = sequence.rope_delta;
    session.tail_hidden_valid          = sequence.tail_hidden_valid ? 1 : 0;
    session.text_committed_frontier    = text_committed;
    session.backend_committed_frontier = backend_committed;
    session.text_pages                 = text_pages;
    session.backend_pages              = backend_pages;
    session.rebuild_work               = sequence.rebuild_work;
    session.rebuild_tail_begin         = sequence.rebuild_tail_begin;

    qwen3_6::RetainedSessionSnapshot snapshot;
    snapshot.tokens         = session.tokens;
    snapshot.session_digest = ledger_digest(sequence.ledger);
    SnapshotWriter writer(snapshot.bytes);
    writer.bytes(kSessionSnapshotMagic, sizeof(kSessionSnapshotMagic));
    writer.pod(kSessionSnapshotVersion);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(model_binding.size()));
    writer.bytes(model_binding.data(), model_binding.size());
    write_config(writer, config);
    write_session(writer, session);
    write_vector(writer, sequence.ledger);
    write_vector(writer, sequence.prefix_identity.token_types());
    for (std::size_t axis = 0; axis < 3; ++axis) {
        write_vector(writer, sequence.prefix_identity.position_axis(axis));
    }
    write_vision_items(writer, sequence.prefix_identity.vision_items());
    write_vector(writer, sequence.prefix_identity.rewrite_execution_frontiers());
    write_vector(writer, sequence.prefix_digests.image());

    // Checkpoint directory: rewrite checkpoint, long anchors, and the StateImage table map.
    writer.pod<std::uint8_t>(sequence.rewrite_checkpoint.valid ? 1 : 0);
    writer.pod<std::uint8_t>(static_cast<std::uint8_t>(sequence.rewrite_checkpoint.kind));
    writer.pod<std::uint32_t>(sequence.rewrite_checkpoint.frontier);
    writer.pod(sequence.rewrite_checkpoint.rebuild_work);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(sequence.long_anchors.size()));
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        writer.pod<std::uint32_t>(anchor.frontier);
        writer.pod<std::uint32_t>(anchor.ordinal);
        writer.pod(anchor.rebuild_work);
    }
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(unique_states.size()));
    writer.pod<std::int32_t>(endpoint_image);
    writer.pod<std::int32_t>(rewrite_image);
    for (const std::int32_t index : anchor_images) { writer.pod<std::int32_t>(index); }

    // Size the device payload in one pass so the vector's storage is final before any
    // cudaMemcpyAsync records a destination pointer.
    const std::size_t state_offset =
        writer.reserve_payload(state_layout.image_bytes * unique_states.size());
    const std::size_t text_kv_offset =
        writer.reserve_payload(config.text_page_stride * session.text_pages);
    const std::size_t backend_kv_offset =
        writer.reserve_payload(config.backend_page_stride * session.backend_pages);

    std::uint8_t* base = snapshot.bytes.data();
    for (std::size_t index = 0; index < unique_states.size(); ++index) {
        const StateImageHandle image  = unique_states[index];
        std::uint8_t* const image_out = base + state_offset + index * state_layout.image_bytes;
        if (state_store->residency(image) == StateReplicaResidency::HostOnly) {
            const qwen3_6::HostStateImageConstView view = state_store->host_view(image);
            std::memcpy(image_out, view.data, state_layout.image_bytes);
        } else {
            state_images->copy_to_host(
                state_store->physical_slot(image),
                qwen3_6::HostStateImageView{reinterpret_cast<std::byte*>(image_out),
                                            &state_layout},
                device.stream);
            ++snapshot_traffic_.state_d2h_count;
            snapshot_traffic_.state_d2h_bytes += state_layout.image_bytes;
        }
    }

    // KV pages: device-resident runs go through the pool's page copier; demoted pages are read
    // from their published Host replicas without touching the device.
    const auto copy_address_pages =
        [&](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
            const DeviceKVPagePool& pool, KVAddressSpaceHandle address, std::uint32_t page_count,
            const HostKVPageLayout& layout, std::size_t payload_offset, std::uint64_t& d2h_pages,
            std::uint64_t& d2h_bytes) {
            std::vector<DeviceKVPageHandle> run;
            run.reserve(page_count);
            std::uint32_t run_begin = 0;
            const auto flush_run    = [&] {
                if (run.empty()) { return; }
                pool.copy_to_host(
                    std::span<const DeviceKVPageHandle>(run.data(), run.size()),
                    reinterpret_cast<std::byte*>(base + payload_offset +
                                                 static_cast<std::size_t>(run_begin) *
                                                     layout.page_stride),
                    layout, device.stream);
                d2h_pages += run.size();
                d2h_bytes += run.size() * layout.page_stride;
                run.clear();
            };
            for (std::uint32_t page = 0; page < page_count; ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (pages.device_resident(logical)) {
                    if (run.empty()) { run_begin = page; }
                    run.push_back(pages.physical(logical));
                    continue;
                }
                flush_run();
                if (!pages.host_resident(logical) || !pages.host_replica_current(logical)) {
                    throw std::logic_error("retained session KV page has no current replica");
                }
                if (!host_kv_extents) {
                    throw std::logic_error("retained session Host replica has no extent store");
                }
                const HostKVPageReplica& replica = pages.host_replica(logical);
                const HostKVAllocationConstView view = host_kv_extents->view(replica.extent);
                if (view.layout().page_stride != layout.page_stride ||
                    replica.page_offset >= view.page_count()) {
                    throw std::logic_error("retained session Host replica layout is inconsistent");
                }
                std::memcpy(base + payload_offset +
                                static_cast<std::size_t>(page) * layout.page_stride,
                            view.data() +
                                static_cast<std::size_t>(replica.page_offset) * layout.page_stride,
                            layout.page_stride);
            }
            flush_run();
        };
    copy_address_pages(*text_kv_addresses, *text_kv_pages, text_pool, text_address,
                       session.text_pages, text_layout, text_kv_offset,
                       snapshot_traffic_.main_kv_d2h_pages, snapshot_traffic_.main_kv_d2h_bytes);
    if (backend != nullptr && session.backend_pages != 0) {
        copy_address_pages(*backend_kv_addresses, *backend_kv_pages, backend->page_pool(),
                           *sequence.kv->backend, session.backend_pages, *backend_layout,
                           backend_kv_offset, snapshot_traffic_.backend_kv_d2h_pages,
                           snapshot_traffic_.backend_kv_d2h_bytes);
    }
    device.synchronize();
    return snapshot;
}

ContinuationHandle
ProgramImplCore::restore_continuation(std::span<const std::uint8_t> snapshot,
                                      std::string_view model_binding) {
    if (has_context_transaction() || pending_transaction_) {
        throw std::logic_error("cannot restore a session during a resource transaction");
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument("session persistence does not support the DFlash backend");
    }

    SnapshotReader reader(snapshot);
    char magic[sizeof(kSessionSnapshotMagic)] = {};
    reader.bytes(magic, sizeof(magic));
    if (std::memcmp(magic, kSessionSnapshotMagic, sizeof(magic)) != 0) {
        throw std::invalid_argument("file is not a session snapshot");
    }
    const std::uint32_t version = reader.pod<std::uint32_t>();
    if (version < kSessionSnapshotVersion) {
        throw std::invalid_argument(
            "session snapshot predates the context-cache reconciliation and cannot be restored");
    }
    if (version != kSessionSnapshotVersion) {
        throw std::invalid_argument("session snapshot version is unsupported");
    }
    const std::uint32_t binding_bytes = reader.pod<std::uint32_t>();
    if (binding_bytes > 4096) {
        throw std::invalid_argument("session snapshot model binding is too long");
    }
    std::string binding(binding_bytes, '\0');
    reader.bytes(binding.data(), binding_bytes);
    if (binding != model_binding) {
        throw std::invalid_argument("session snapshot was saved for a different model");
    }

    const StateImageHostLayout& state_layout = state_images->host_layout();
    const DeviceKVPagePool& text_pool        = text_kv_pages->physical_pool();
    const HostKVPageLayout text_layout       = plan_host_kv_page_layout(text_pool.geometry());
    qwen3_6::PagedKVCache* backend           = backend_kv_cache();
    std::optional<HostKVPageLayout> backend_layout;
    if (backend != nullptr) {
        backend_layout = plan_host_kv_page_layout(backend->page_pool().geometry());
    }

    const SnapshotConfig config = read_config(reader);
    const SnapshotKvProfile kv_profile = snapshot_kv_profile(kv_storage);
    if (config.kv_dtype != static_cast<std::uint32_t>(kv_profile.dtype) ||
        config.kv_quant_group != kv_profile.quant_group || config.kv_flags != kv_profile.flags ||
        config.page_size != static_cast<std::uint32_t>(kPagedKVPageSize) ||
        config.text_plane_count != static_cast<std::uint32_t>(text_pool.plane_count()) ||
        config.text_page_stride != text_layout.page_stride) {
        throw std::invalid_argument("session snapshot KV configuration does not match the server");
    }
    if (config.speculative_backend != static_cast<std::uint32_t>(speculative_backend) ||
        config.draft_window != draft_window) {
        throw std::invalid_argument(
            "session snapshot speculative configuration does not match the server");
    }
    const std::uint32_t backend_plane_count =
        backend != nullptr ? static_cast<std::uint32_t>(backend->page_pool().plane_count()) : 0U;
    const std::uint64_t backend_page_stride =
        backend != nullptr ? backend_layout->page_stride : 0U;
    if (config.backend_plane_count != backend_plane_count ||
        config.backend_page_stride != backend_page_stride) {
        throw std::invalid_argument(
            "session snapshot backend KV configuration does not match the server");
    }
    if (config.state_image_bytes != state_layout.image_bytes) {
        throw std::invalid_argument("session snapshot state geometry does not match the server");
    }

    const SnapshotSession session = read_session(reader);
    if (session.tokens == 0 || session.tokens > capacity) {
        throw std::invalid_argument("session snapshot depth exceeds the server context");
    }
    if (session.ledger_frontier != session.tokens ||
        session.execution_frontier > session.tokens ||
        session.tokens - session.execution_frontier > 1 ||
        session.text_kv_valid > session.text_committed_frontier ||
        session.execution_frontier > session.text_committed_frontier ||
        session.text_committed_frontier > capacity ||
        session.mtp_kv_valid > session.backend_committed_frontier ||
        session.backend_committed_frontier > capacity ||
        session.rebuild_work.tokens != session.execution_frontier ||
        session.rebuild_tail_begin > session.execution_frontier) {
        throw std::invalid_argument("session snapshot frontiers are inconsistent");
    }
    if (session.text_pages != kv_pages_for_frontier(session.text_committed_frontier) ||
        session.text_pages == 0 ||
        session.backend_pages != kv_pages_for_frontier(session.backend_committed_frontier) ||
        (backend == nullptr && session.backend_pages != 0)) {
        throw std::invalid_argument("session snapshot page counts are out of range");
    }

    std::vector<TokenId> ledger = read_vector<TokenId>(reader, session.tokens, "ledger");
    if (ledger.size() != session.tokens) {
        throw std::invalid_argument("session snapshot ledger does not match its depth");
    }
    for (const TokenId id : ledger) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("session snapshot ledger token is out of domain");
        }
    }
    std::vector<std::uint8_t> token_types =
        read_vector<std::uint8_t>(reader, session.tokens, "token type");
    std::array<std::vector<std::int32_t>, 3> positions;
    for (auto& axis : positions) {
        axis = read_vector<std::int32_t>(reader, session.tokens, "position");
    }
    std::vector<VisionItem> vision_items = read_vision_items(reader, session.tokens);
    std::vector<std::uint32_t> rewrite_frontiers =
        read_vector<std::uint32_t>(reader, session.tokens, "rewrite frontier");
    std::vector<std::array<std::uint64_t, 2>> digest_image =
        read_vector<std::array<std::uint64_t, 2>>(reader, static_cast<std::size_t>(session.tokens) + 1U,
                                                  "shortlist digest");
    if (token_types.size() != session.tokens || positions[0].size() != session.tokens ||
        positions[1].size() != session.tokens || positions[2].size() != session.tokens ||
        digest_image.size() != static_cast<std::size_t>(session.tokens) + 1U) {
        throw std::invalid_argument("session snapshot identity does not match its depth");
    }
    if (!vision_items.empty() && !vision_enabled) {
        throw std::invalid_argument("session snapshot holds media but Vision is disabled");
    }

    // Checkpoint directory.
    const std::uint8_t rewrite_valid_flag  = reader.pod<std::uint8_t>();
    const std::uint8_t rewrite_kind_value  = reader.pod<std::uint8_t>();
    const std::uint32_t rewrite_frontier   = reader.pod<std::uint32_t>();
    const runtime::PrefillWork rewrite_work = reader.pod<runtime::PrefillWork>();
    if (rewrite_valid_flag != 0 &&
        (rewrite_frontier == 0 || rewrite_frontier > session.tokens ||
         rewrite_work.tokens != rewrite_frontier ||
         rewrite_kind_value > static_cast<std::uint8_t>(RewriteCheckpointKind::ResponseReplay))) {
        throw std::invalid_argument("session snapshot rewrite checkpoint is inconsistent");
    }
    struct SnapshotAnchor {
        std::uint32_t frontier = 0;
        std::uint32_t ordinal  = 0;
        runtime::PrefillWork rebuild_work;
        std::int32_t image = -1;
    };
    const std::uint32_t anchor_count = reader.pod<std::uint32_t>();
    if (anchor_count > 64U) {
        throw std::invalid_argument("session snapshot anchor count is out of range");
    }
    std::vector<SnapshotAnchor> anchors(anchor_count);
    for (SnapshotAnchor& anchor : anchors) {
        anchor.frontier     = reader.pod<std::uint32_t>();
        anchor.ordinal      = reader.pod<std::uint32_t>();
        anchor.rebuild_work = reader.pod<runtime::PrefillWork>();
        if (anchor.frontier == 0 || anchor.frontier > session.tokens || anchor.ordinal == 0 ||
            anchor.rebuild_work.tokens != anchor.frontier) {
            throw std::invalid_argument("session snapshot long anchor is inconsistent");
        }
    }
    for (std::size_t index = 0; index < anchors.size(); ++index) {
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (anchors[previous].ordinal == anchors[index].ordinal) {
                throw std::invalid_argument("session snapshot anchor ordinals are not unique");
            }
        }
    }
    const std::uint32_t image_count = reader.pod<std::uint32_t>();
    if (image_count == 0 || image_count > 2U + anchor_count) {
        throw std::invalid_argument("session snapshot StateImage table is out of range");
    }
    const auto read_image_index = [&](bool required) -> std::int32_t {
        const std::int32_t index = reader.pod<std::int32_t>();
        if ((required && index < 0) || index >= static_cast<std::int32_t>(image_count) ||
            (!required && index < -1)) {
            throw std::invalid_argument("session snapshot StateImage index is out of range");
        }
        return index;
    };
    const std::int32_t endpoint_image = read_image_index(true);
    const std::int32_t rewrite_image  = read_image_index(false);
    if ((rewrite_valid_flag != 0) != (rewrite_image >= 0)) {
        throw std::invalid_argument("session snapshot rewrite StateImage index is inconsistent");
    }
    for (SnapshotAnchor& anchor : anchors) { anchor.image = read_image_index(true); }

    std::vector<const std::uint8_t*> image_payloads(image_count);
    for (std::uint32_t index = 0; index < image_count; ++index) {
        image_payloads[index] = reader.payload(state_layout.image_bytes);
    }
    const std::uint8_t* state_payload = image_payloads[static_cast<std::size_t>(endpoint_image)];
    const std::uint8_t* text_payload =
        reader.payload(config.text_page_stride * session.text_pages);
    const std::uint8_t* backend_payload =
        session.backend_pages != 0
            ? reader.payload(config.backend_page_stride * session.backend_pages)
            : nullptr;
    if (reader.remaining() != 0) {
        throw std::invalid_argument("session snapshot has trailing bytes");
    }

    if (text_pool.available_pages() < session.text_pages ||
        (backend != nullptr &&
         backend->page_pool().available_pages() < session.backend_pages)) {
        throw std::invalid_argument(
            "session snapshot does not fit the free KV capacity; evict other sessions first");
    }

    // Page uploads run through a temporarily activated address space, which needs one execution
    // row; rows are lane-indexed and held only by active requests, so any idle lane's row works.
    std::optional<std::int32_t> free_row;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (requests[lane].lifecycle == Lifecycle::Empty &&
            active_continuations[lane] == continuation_capacity) {
            free_row = static_cast<std::int32_t>(lane);
            break;
        }
    }
    if (!free_row) {
        throw std::logic_error("session restore requires an idle execution lane");
    }

    std::optional<std::uint32_t> slot_index = allocate_continuation_slot();
    if (!slot_index) {
        throw std::invalid_argument(
            "session snapshot does not fit the continuation catalog; evict other sessions first");
    }

    std::optional<StateImageHandle> state;
    std::vector<std::optional<StateImageHandle>> extra_images(image_count);
    std::optional<KVAddressSpaceHandle> text_address;
    std::optional<KVAddressSpaceHandle> backend_address;
    try {
        // Every restored image prefers a Device slot and falls back to a HostOnly replica (the
        // shape a demoted checkpoint has) when the Device pool is occupied by other sessions.
        const auto stage_image = [&](const std::uint8_t* payload)
            -> std::optional<StateImageHandle> {
            const qwen3_6::HostStateImageConstView view{
                reinterpret_cast<const std::byte*>(payload), &state_layout};
            std::optional<StateImageHandle> handle = state_store->reserve_reset(device.stream);
            if (handle) {
                state_images->copy_from_host(view, state_store->physical_slot(*handle),
                                             device.stream);
                ++snapshot_traffic_.state_h2d_count;
                snapshot_traffic_.state_h2d_bytes += state_layout.image_bytes;
                return handle;
            }
            return state_store->adopt_host_image(view);
        };
        state = stage_image(state_payload);
        if (!state) {
            throw std::invalid_argument(
                "session snapshot does not fit the free state capacity; evict other sessions "
                "first");
        }

        // Optional checkpoint images: an image that fits neither pool just drops the
        // checkpoints naming it; the endpoint above stays mandatory.
        const std::uint32_t anchor_capacity =
            context_cache.max_long_anchors_per_continuation.value_or(0);
        const auto upload_image = [&](std::int32_t index) {
            const auto slot_index = static_cast<std::size_t>(index);
            if (index == endpoint_image || extra_images[slot_index]) { return; }
            extra_images[slot_index] = stage_image(image_payloads[slot_index]);
        };
        if (rewrite_valid_flag != 0) { upload_image(rewrite_image); }
        for (const SnapshotAnchor& anchor : anchors) {
            if (anchor.ordinal <= anchor_capacity) { upload_image(anchor.image); }
        }

        const auto build_address =
            [&](KVAddressSpaceStore& addresses, DeviceKVPagePool& pool, std::uint32_t page_count,
                std::uint32_t committed, const HostKVPageLayout& layout,
                const std::uint8_t* payload, std::uint64_t& h2d_pages,
                std::uint64_t& h2d_bytes) -> KVAddressSpaceHandle {
            std::optional<KVAddressSpaceHandle> address = addresses.create_inactive();
            if (!address) {
                throw std::invalid_argument(
                    "session snapshot does not fit the KV address capacity; evict other "
                    "sessions first");
            }
            if (page_count == 0) { return *address; }
            try {
                addresses.activate(*address, page_count, *free_row);
                addresses.materialize_to_tokens(*address, committed, device.stream);
                addresses.commit_frontier(*address, committed);
                std::vector<DeviceKVPageHandle> destinations;
                destinations.reserve(page_count);
                for (std::uint32_t page = 0; page < page_count; ++page) {
                    destinations.push_back(addresses.physical_page(*address, page));
                }
                pool.copy_from_host(reinterpret_cast<const std::byte*>(payload), layout,
                                    std::span<const DeviceKVPageHandle>(destinations.data(),
                                                                        destinations.size()),
                                    device.stream);
                h2d_pages += page_count;
                h2d_bytes += static_cast<std::uint64_t>(page_count) * layout.page_stride;
            } catch (...) {
                if (addresses.active(*address)) { addresses.deactivate(*address); }
                (void)addresses.release(*address);
                throw;
            }
            return *address;
        };
        text_address = build_address(*text_kv_addresses, text_kv_pages->physical_pool(),
                                     session.text_pages, session.text_committed_frontier,
                                     text_layout, text_payload,
                                     snapshot_traffic_.main_kv_h2d_pages,
                                     snapshot_traffic_.main_kv_h2d_bytes);
        if (backend != nullptr) {
            backend_address = build_address(
                *backend_kv_addresses, backend->page_pool(), session.backend_pages,
                session.backend_committed_frontier, *backend_layout, backend_payload,
                snapshot_traffic_.backend_kv_h2d_pages, snapshot_traffic_.backend_kv_h2d_bytes);
        }
        device.synchronize();

        // Point of adoption: the sequence owns the physical handles from here, so the local
        // optionals are disarmed as they are handed over and the failure path collapses to
        // release_continuation_slot_best_effort.
        SequenceState& sequence = continuation_states[*slot_index];
        if (state_store->role(*state) == StateImageRole::ActiveMutable) {
            state_store->freeze(*state);
        }
        sequence.state = ActiveStateBinding{.read = *state, .write = *state};
        state.reset();
        sequence.kv.emplace(SequenceKVBundle{.text = *text_address, .backend = backend_address});
        text_address.reset();
        backend_address.reset();
        unbind_sequence_kv(sequence);
        sequence.lane = static_cast<std::uint32_t>(*free_row);

        sequence.ledger.assign(ledger.begin(), ledger.end());
        sequence.prefix_identity.restore(std::move(token_types), std::move(positions),
                                         std::move(vision_items), std::move(rewrite_frontiers));
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_digests.restore(std::move(digest_image));
        sequence.prefix_digests.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.execution_frontier       = session.execution_frontier;
        sequence.ledger_frontier          = session.ledger_frontier;
        sequence.text_kv_valid            = session.text_kv_valid;
        sequence.mtp_kv_valid             = session.mtp_kv_valid;
        sequence.dflash_context_frontier  = 0;
        sequence.rope_delta               = session.rope_delta;
        sequence.mtp_draft_count          = 0;
        sequence.tail_hidden_valid        = session.tail_hidden_valid != 0;
        sequence.endpoint_valid           = true;
        sequence.rewrite_checkpoint       = {};
        sequence.rewrite_state.reset();
        sequence.reserved_state.reset();
        sequence.rebuild_work       = session.rebuild_work;
        sequence.rebuild_tail_begin = session.rebuild_tail_begin;

        // Adopt the surviving checkpoints: freeze the uploaded images and hand them to the
        // sequence with one checkpoint reference per naming checkpoint (the endpoint keeps
        // zero references, matching finish()).
        for (std::optional<StateImageHandle>& image : extra_images) {
            if (image && state_store->role(*image) == StateImageRole::ActiveMutable) {
                state_store->freeze(*image);
            }
        }
        const auto resolve_image =
            [&](std::int32_t index) -> std::optional<StateImageHandle> {
            if (index == endpoint_image) { return sequence.state.read; }
            return extra_images[static_cast<std::size_t>(index)];
        };
        if (rewrite_valid_flag != 0) {
            if (const std::optional<StateImageHandle> handle = resolve_image(rewrite_image)) {
                sequence.rewrite_state      = *handle;
                sequence.rewrite_checkpoint = RewriteCheckpoint{
                    .valid        = true,
                    .kind         = static_cast<RewriteCheckpointKind>(rewrite_kind_value),
                    .frontier     = rewrite_frontier,
                    .rebuild_work = rewrite_work,
                };
                state_store->retain_checkpoint_reference(*handle);
            }
        }
        for (const SnapshotAnchor& anchor : anchors) {
            if (anchor.ordinal > anchor_capacity) { continue; }
            const std::optional<StateImageHandle> handle = resolve_image(anchor.image);
            if (!handle) { continue; }
            sequence.long_anchors.push_back(LongAnchorCheckpoint{
                .state        = *handle,
                .frontier     = anchor.frontier,
                .ordinal      = anchor.ordinal,
                .rebuild_work = anchor.rebuild_work,
            });
            state_store->retain_checkpoint_reference(*handle);
        }
        refresh_state_views(sequence);

        text_kv_addresses->set_checkpoint_requirement(sequence.kv->text,
                                                      sequence.execution_frontier);
        if (sequence.kv->backend) {
            backend_kv_addresses->set_checkpoint_requirement(*sequence.kv->backend,
                                                             backend_kv_valid(sequence));
        }
        // Mint the endpoint summary once so a malformed rebuild surfaces here instead of at the
        // Engine catalog's adoption.
        (void)continuation_summary(sequence);

        continuation_slots[*slot_index].role = ContinuationSlotRole::Catalogued;
        advance_resource_revision();
        return ContractAccess::make_continuation(this, *slot_index,
                                                 continuation_slots[*slot_index].generation);
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        if (backend_address) {
            if (backend_kv_addresses->active(*backend_address)) {
                backend_kv_addresses->deactivate(*backend_address);
            }
            (void)backend_kv_addresses->release(*backend_address);
        }
        if (text_address) {
            if (text_kv_addresses->active(*text_address)) {
                text_kv_addresses->deactivate(*text_address);
            }
            (void)text_kv_addresses->release(*text_address);
        }
        if (state) { (void)state_store->release(*state); }
        // Images already adopted by the sequence carry checkpoint references, so this release
        // refuses them and release_continuation_slot_best_effort below owns their teardown
        // instead.
        for (std::optional<StateImageHandle>& image : extra_images) {
            if (image) { (void)state_store->release(*image); }
        }
        if (slot_index) { release_continuation_slot_best_effort(*slot_index); }
        throw;
    }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
