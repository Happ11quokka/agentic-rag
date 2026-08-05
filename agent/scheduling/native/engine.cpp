#include <llama.h>
#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef PS_LLAMA_BUILD
#define PS_LLAMA_BUILD 0
#endif
#ifndef PS_LLAMA_COMMIT
#define PS_LLAMA_COMMIT unknown
#endif
#ifndef PS_SOURCE_SHA256
#define PS_SOURCE_SHA256 unknown
#endif

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::array<char, 4> MAGIC{'P', 'S', 'Q', '1'};
constexpr uint8_t VERSION = 1;
constexpr uint64_t MAX_FRAME = 16ULL * 1024 * 1024;

enum class FrameType : uint8_t {
    START = 1, PREFILL = 2, DECODE_SLICE = 3, SHUTDOWN = 4,
    READY = 16, STARTED = 17, PREFILL_RESULT = 18, SLICE_RESULT = 19,
    ERROR = 20, BYE = 21,
};
enum class Role : uint8_t { PAIR = 0, MAIN = 1, DRAFT = 2 };

struct Frame {
    FrameType type;
    Role role;
    uint8_t flags;
    uint32_t request_id;
    std::vector<uint8_t> payload;
};

struct Reader {
    const std::vector<uint8_t> & data;
    size_t offset = 0;

    template <typename T> T get() {
        static_assert(std::endian::native == std::endian::little);
        if (offset + sizeof(T) > data.size()) throw std::runtime_error("truncated payload");
        T value;
        std::memcpy(&value, data.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }
    std::string string() {
        const auto size = get<uint32_t>();
        if (offset + size > data.size()) throw std::runtime_error("truncated string");
        std::string value(reinterpret_cast<const char *>(data.data() + offset), size);
        offset += size;
        return value;
    }
    void done() const {
        if (offset != data.size()) throw std::runtime_error("payload has trailing bytes");
    }
};

template <typename T> void append(std::vector<uint8_t> & out, T value) {
    static_assert(std::endian::native == std::endian::little);
    const auto * bytes = reinterpret_cast<const uint8_t *>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}
void append_string(std::vector<uint8_t> & out, std::string_view value) {
    if (value.size() > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("string too large");
    append<uint32_t>(out, static_cast<uint32_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
}

void read_exact(char * destination, size_t size) {
    std::cin.read(destination, static_cast<std::streamsize>(size));
    if (std::cin.gcount() != static_cast<std::streamsize>(size)) {
        throw std::runtime_error("protocol pipe EOF");
    }
}

Frame read_frame() {
    std::array<uint8_t, 20> header{};
    read_exact(reinterpret_cast<char *>(header.data()), header.size());
    if (!std::equal(MAGIC.begin(), MAGIC.end(), header.begin())) throw std::runtime_error("invalid protocol magic");
    if (header[4] != VERSION) throw std::runtime_error("invalid protocol version");
    uint32_t request_id;
    uint64_t length;
    std::memcpy(&request_id, header.data() + 8, sizeof(request_id));
    std::memcpy(&length, header.data() + 12, sizeof(length));
    if (length > MAX_FRAME) throw std::runtime_error("invalid frame length");
    if (header[6] > static_cast<uint8_t>(Role::DRAFT)) throw std::runtime_error("invalid role");
    const uint8_t type = header[5];
    if (!((type >= 1 && type <= 4) || (type >= 16 && type <= 21))) throw std::runtime_error("invalid frame type");
    Frame frame{static_cast<FrameType>(type), static_cast<Role>(header[6]), header[7], request_id, {}};
    frame.payload.resize(length);
    if (length) read_exact(reinterpret_cast<char *>(frame.payload.data()), length);
    return frame;
}

void write_frame(FrameType type, Role role, uint32_t request_id, const std::vector<uint8_t> & payload = {}) {
    if (payload.size() > MAX_FRAME) throw std::runtime_error("response frame too large");
    std::array<uint8_t, 20> header{};
    std::copy(MAGIC.begin(), MAGIC.end(), header.begin());
    header[4] = VERSION;
    header[5] = static_cast<uint8_t>(type);
    header[6] = static_cast<uint8_t>(role);
    const uint64_t length = payload.size();
    std::memcpy(header.data() + 8, &request_id, sizeof(request_id));
    std::memcpy(header.data() + 12, &length, sizeof(length));
    std::cout.write(reinterpret_cast<const char *>(header.data()), header.size());
    if (!payload.empty()) std::cout.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("protocol write failed");
}

uint64_t nanoseconds(Clock::duration value) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(value).count();
}

struct Generation {
    uint32_t max_tokens;
    float temperature;
    float top_p;
    float min_p;
    int32_t top_k;
    float presence_penalty;
    uint32_t seed;
};
struct Message { std::string role; std::string content; };

struct StartRequest {
    Generation generation;
    std::vector<Message> messages;
};

StartRequest parse_start(const std::vector<uint8_t> & payload) {
    Reader reader{payload};
    StartRequest request;
    request.generation.max_tokens = reader.get<uint32_t>();
    request.generation.temperature = reader.get<float>();
    request.generation.top_p = reader.get<float>();
    request.generation.min_p = reader.get<float>();
    request.generation.top_k = reader.get<int32_t>();
    request.generation.presence_penalty = reader.get<float>();
    request.generation.seed = reader.get<uint32_t>();
    if (request.generation.max_tokens == 0) throw std::runtime_error("max_tokens must be positive");
    const auto count = reader.get<uint32_t>();
    request.messages.reserve(count);
    for (uint32_t i = 0; i < count; ++i) request.messages.push_back({reader.string(), reader.string()});
    reader.done();
    return request;
}

struct TokenEvent { uint64_t offset_ns; llama_token token; std::string piece; };
struct Result {
    uint32_t processed = 0;
    uint32_t remaining = 0;
    uint32_t output = 0;
    uint8_t finish = 0;
    uint64_t active_ns = 0;
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    std::vector<TokenEvent> events;
};

std::vector<uint8_t> encode_result(const Result & result) {
    std::vector<uint8_t> payload;
    append<uint32_t>(payload, result.processed);
    append<uint32_t>(payload, result.remaining);
    append<uint32_t>(payload, result.output);
    append<uint8_t>(payload, result.finish);
    append<uint8_t>(payload, 0); append<uint8_t>(payload, 0); append<uint8_t>(payload, 0);
    append<uint64_t>(payload, result.active_ns);
    append<uint64_t>(payload, result.start_ns);
    append<uint64_t>(payload, result.end_ns);
    append<uint32_t>(payload, static_cast<uint32_t>(result.events.size()));
    for (const auto & event : result.events) {
        append<uint64_t>(payload, event.offset_ns);
        append<int32_t>(payload, event.token);
        append<uint32_t>(payload, static_cast<uint32_t>(event.piece.size()));
        payload.insert(payload.end(), event.piece.begin(), event.piece.end());
    }
    return payload;
}

void llama_log(enum ggml_log_level, const char * text, void *) {
    std::cerr << text;
    std::cerr.flush();
}

struct ModelState {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_sampler * sampler = nullptr;
    const llama_vocab * vocab = nullptr;
    std::vector<llama_token> prompt;
    uint32_t cursor = 0;
    uint32_t output_tokens = 0;
    llama_token pending_token = LLAMA_TOKEN_NULL;
    uint8_t finish = 0;

    ~ModelState() {
        clear();
    }
    void clear() {
        if (sampler) llama_sampler_free(sampler);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
        sampler = nullptr;
        context = nullptr;
        model = nullptr;
        vocab = nullptr;
    }
    ModelState() = default;
    ModelState(const ModelState &) = delete;
    ModelState & operator=(const ModelState &) = delete;
};

struct BackendRuntime {
    BackendRuntime() {
        llama_log_set(llama_log, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
    }
    ~BackendRuntime() { llama_backend_free(); }
    BackendRuntime(const BackendRuntime &) = delete;
    BackendRuntime & operator=(const BackendRuntime &) = delete;
};

class Engine {
  public:
    Engine(
        std::string main_path,
        std::string draft_path,
        int threads,
        uint32_t n_ctx,
        uint32_t n_batch,
        uint32_t n_ubatch,
        int32_t n_gpu_layers
    ) : threads_(threads), n_ctx_(n_ctx), n_batch_(n_batch), n_ubatch_(n_ubatch), n_gpu_layers_(n_gpu_layers) {
        load(draft_, draft_path);
        load(main_, main_path);
    }

    void start(const StartRequest & request) {
        pair_start_ = Clock::now();
        generation_ = request.generation;
        reset(draft_, request.messages);
        reset(main_, request.messages);
    }

    Result prefill(Role role, uint32_t maximum) {
        if (maximum == 0 || maximum > 256) throw std::runtime_error("prefill chunk must be in [1, 256]");
        auto & state = selected(role);
        Result result;
        result.start_ns = offset();
        const auto count = std::min<uint32_t>(maximum, state.prompt.size() - state.cursor);
        if (count == 0) throw std::runtime_error("prefill requested after prompt completion");
        llama_batch batch = llama_batch_init(count, 0, 1);
        batch.n_tokens = count;
        for (uint32_t i = 0; i < count; ++i) {
            batch.token[i] = state.prompt[state.cursor + i];
            batch.pos[i] = state.cursor + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (state.cursor + i + 1 == state.prompt.size()) ? 1 : 0;
        }
        const auto active_start = Clock::now();
        const int status = llama_decode(state.context, batch);
        llama_batch_free(batch);
        if (status != 0) throw std::runtime_error("llama_decode failed during prefill: " + std::to_string(status));
        state.cursor += count;
        if (state.cursor == state.prompt.size()) sample(state, result);
        const auto active_end = Clock::now();
        result.processed = count;
        result.remaining = state.prompt.size() - state.cursor;
        result.output = state.output_tokens;
        result.finish = state.finish;
        result.active_ns = nanoseconds(active_end - active_start);
        result.end_ns = offset();
        return result;
    }

    Result slice(Role role, uint64_t budget_ns) {
        if (budget_ns == 0) throw std::runtime_error("decode budget must be positive");
        auto & state = selected(role);
        if (state.cursor != state.prompt.size()) throw std::runtime_error("decode requested before prefill completion");
        Result result;
        result.start_ns = offset();
        const auto active_start = Clock::now();
        while (!state.finish && nanoseconds(Clock::now() - active_start) < budget_ns) {
            llama_token token = state.pending_token;
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = static_cast<llama_pos>(state.prompt.size() + state.output_tokens - 1);
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            const int status = llama_decode(state.context, batch);
            llama_batch_free(batch);
            if (status != 0) throw std::runtime_error("llama_decode failed during generation: " + std::to_string(status));
            sample(state, result);
        }
        const auto active_end = Clock::now();
        result.output = state.output_tokens;
        result.finish = state.finish;
        result.active_ns = nanoseconds(active_end - active_start);
        result.end_ns = offset();
        return result;
    }

    uint32_t prompt_count(Role role) { return selected(role).prompt.size(); }

  private:
    BackendRuntime backend_;
    ModelState main_;
    ModelState draft_;
    int threads_;
    uint32_t n_ctx_;
    uint32_t n_batch_;
    uint32_t n_ubatch_;
    int32_t n_gpu_layers_;
    Generation generation_{};
    Clock::time_point pair_start_{};

    uint64_t offset() const { return nanoseconds(Clock::now() - pair_start_); }
    ModelState & selected(Role role) {
        if (role == Role::MAIN) return main_;
        if (role == Role::DRAFT) return draft_;
        throw std::runtime_error("pair role is invalid for model command");
    }
    void load(ModelState & state, const std::string & path) {
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers_;
        state.model = llama_model_load_from_file(path.c_str(), model_params);
        if (!state.model) throw std::runtime_error("failed to load model: " + path);
        state.vocab = llama_model_get_vocab(state.model);
        if (!llama_model_chat_template(state.model, nullptr)) throw std::runtime_error("model has no supported chat template: " + path);
        auto context_params = llama_context_default_params();
        context_params.n_ctx = n_ctx_;
        context_params.n_batch = n_batch_;
        context_params.n_ubatch = n_ubatch_;
        context_params.n_seq_max = 1;
        context_params.n_threads = threads_;
        context_params.n_threads_batch = threads_;
        context_params.type_k = GGML_TYPE_Q8_0;
        context_params.type_v = GGML_TYPE_Q8_0;
        context_params.offload_kqv = true;
        state.context = llama_init_from_model(state.model, context_params);
        if (!state.context) throw std::runtime_error("failed to create llama context: " + path);
        if (llama_n_ctx(state.context) != n_ctx_) throw std::runtime_error("llama context size mismatch");
    }
    std::string format(ModelState & state, const std::vector<Message> & messages) {
        std::vector<llama_chat_message> chat;
        chat.reserve(messages.size());
        for (const auto & message : messages) chat.push_back({message.role.c_str(), message.content.c_str()});
        const char * tmpl = llama_model_chat_template(state.model, nullptr);
        std::vector<char> buffer(1024 + 2 * std::accumulate(messages.begin(), messages.end(), size_t{0}, [](size_t total, const Message & message) { return total + message.role.size() + message.content.size(); }));
        int32_t count = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buffer.data(), buffer.size());
        if (count < 0) throw std::runtime_error("chat template is unsupported by llama_chat_apply_template");
        if (static_cast<size_t>(count) > buffer.size()) {
            buffer.resize(count);
            count = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buffer.data(), buffer.size());
        }
        if (count <= 0 || static_cast<size_t>(count) > buffer.size()) throw std::runtime_error("chat template application failed");
        return std::string(buffer.data(), count);
    }
    std::vector<llama_token> tokenize(ModelState & state, const std::string & text) {
        std::vector<llama_token> tokens(text.size() + 16);
        int32_t count = llama_tokenize(state.vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
        if (count < 0) {
            tokens.resize(-count);
            count = llama_tokenize(state.vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
        }
        if (count <= 0) throw std::runtime_error("prompt tokenization failed");
        tokens.resize(count);
        return tokens;
    }
    llama_sampler * make_sampler() {
        auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(64, 1.0f, 0.0f, generation_.presence_penalty));
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(generation_.top_k));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(generation_.top_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(generation_.min_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(generation_.temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(generation_.seed));
        return chain;
    }
    void reset(ModelState & state, const std::vector<Message> & messages) {
        llama_memory_clear(llama_get_memory(state.context), false);
        if (state.sampler) llama_sampler_free(state.sampler);
        state.sampler = make_sampler();
        state.prompt = tokenize(state, format(state, messages));
        if (state.prompt.size() + generation_.max_tokens > n_ctx_) {
            throw std::runtime_error("context overflow: prompt_tokens + max_tokens exceeds configured context");
        }
        for (llama_token token : state.prompt) llama_sampler_accept(state.sampler, token);
        state.cursor = 0;
        state.output_tokens = 0;
        state.pending_token = LLAMA_TOKEN_NULL;
        state.finish = 0;
    }
    std::string piece(ModelState & state, llama_token token) {
        std::array<char, 256> buffer{};
        int32_t count = llama_token_to_piece(state.vocab, token, buffer.data(), buffer.size(), 0, false);
        if (count < 0) {
            std::vector<char> dynamic(-count);
            count = llama_token_to_piece(state.vocab, token, dynamic.data(), dynamic.size(), 0, false);
            if (count < 0) throw std::runtime_error("token-to-piece conversion failed");
            return std::string(dynamic.data(), count);
        }
        return std::string(buffer.data(), count);
    }
    void sample(ModelState & state, Result & result) {
        const llama_token token = llama_sampler_sample(state.sampler, state.context, -1);
        state.pending_token = token;
        ++state.output_tokens;
        const bool eog = llama_vocab_is_eog(state.vocab, token);
        result.events.push_back({offset(), token, eog ? std::string{} : piece(state, token)});
        if (eog) state.finish = 1;
        else if (state.output_tokens >= generation_.max_tokens) state.finish = 2;
    }
};

struct Arguments {
    std::string main_path;
    std::string draft_path;
    int threads = 1;
    uint32_t n_ctx = 16384;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
    int32_t n_gpu_layers = std::numeric_limits<int32_t>::max();
};
Arguments arguments(int argc, char ** argv) {
    Arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("missing argument value: " + arg);
        if (arg == "--main") result.main_path = argv[++i];
        else if (arg == "--draft") result.draft_path = argv[++i];
        else if (arg == "--threads") result.threads = std::stoi(argv[++i]);
        else if (arg == "--n-ctx") result.n_ctx = std::stoul(argv[++i]);
        else if (arg == "--n-batch") result.n_batch = std::stoul(argv[++i]);
        else if (arg == "--n-ubatch") result.n_ubatch = std::stoul(argv[++i]);
        else if (arg == "--n-gpu-layers") result.n_gpu_layers = std::stol(argv[++i]);
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (
        result.main_path.empty() || result.draft_path.empty() || result.threads < 1
        || result.n_ctx == 0 || result.n_batch == 0 || result.n_ubatch == 0
    ) throw std::runtime_error("model paths, thread count, context, and batch sizes must be positive");
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "parallel-scheduled-engine protocol=1 llama_build=" << PS_LLAMA_BUILD
                  << " llama_commit=" << PS_LLAMA_COMMIT << " source_sha256=" << PS_SOURCE_SHA256 << '\n';
        return 0;
    }
    uint32_t request_id = 0;
    Role role = Role::PAIR;
    try {
        const auto args = arguments(argc, argv);
        Engine engine(
            args.main_path,
            args.draft_path,
            args.threads,
            args.n_ctx,
            args.n_batch,
            args.n_ubatch,
            args.n_gpu_layers
        );
        std::vector<uint8_t> ready;
        append_string(ready, "parallel-scheduled-engine");
        append<uint32_t>(ready, args.threads);
        write_frame(FrameType::READY, Role::PAIR, 0, ready);
        uint32_t expected_request_id = 1;
        while (true) {
            const Frame frame = read_frame();
            request_id = frame.request_id;
            role = frame.role;
            if (frame.flags != 0) throw std::runtime_error("unsupported frame flags");
            if (request_id != expected_request_id) throw std::runtime_error("stale or out-of-order request id");
            ++expected_request_id;
            if (frame.type == FrameType::START) {
                if (role != Role::PAIR) throw std::runtime_error("START requires pair role");
                engine.start(parse_start(frame.payload));
                std::vector<uint8_t> payload;
                append<uint32_t>(payload, engine.prompt_count(Role::MAIN));
                append<uint32_t>(payload, engine.prompt_count(Role::DRAFT));
                write_frame(FrameType::STARTED, role, request_id, payload);
            } else if (frame.type == FrameType::PREFILL) {
                Reader reader{frame.payload};
                const auto maximum = reader.get<uint32_t>(); reader.done();
                write_frame(FrameType::PREFILL_RESULT, role, request_id, encode_result(engine.prefill(role, maximum)));
            } else if (frame.type == FrameType::DECODE_SLICE) {
                Reader reader{frame.payload};
                const auto budget = reader.get<uint64_t>(); reader.done();
                write_frame(FrameType::SLICE_RESULT, role, request_id, encode_result(engine.slice(role, budget)));
            } else if (frame.type == FrameType::SHUTDOWN) {
                if (role != Role::PAIR || !frame.payload.empty()) throw std::runtime_error("invalid SHUTDOWN frame");
                write_frame(FrameType::BYE, role, request_id);
                break;
            } else {
                throw std::runtime_error("client sent response frame type");
            }
        }
        return 0;
    } catch (const std::exception & error) {
        try {
            std::vector<uint8_t> payload;
            append_string(payload, error.what());
            write_frame(FrameType::ERROR, role, request_id, payload);
        } catch (...) {}
        std::cerr << "parallel-scheduled-engine: " << error.what() << '\n';
        return 1;
    }
}
