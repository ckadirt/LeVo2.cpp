#include "levo-token-io.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace levo {
namespace token_io {
namespace {

constexpr std::size_t sha_block_size = 64;

class sha256 {
public:
    sha256() : state_{{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                       0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U}} {}

    void update(const uint8_t * data, std::size_t size) {
        bit_count_ += static_cast<uint64_t>(size) * 8U;
        while (size != 0) {
            const std::size_t take = std::min(size, sha_block_size - buffered_);
            std::copy(data, data + take, block_.begin() + static_cast<std::ptrdiff_t>(buffered_));
            buffered_ += take;
            data += take;
            size -= take;
            if (buffered_ == sha_block_size) {
                transform(block_.data());
                buffered_ = 0;
            }
        }
    }

    std::string finish() {
        const uint64_t original_bits = bit_count_;
        uint8_t one = 0x80;
        update(&one, 1);
        uint8_t zero = 0;
        while (buffered_ != 56) update(&zero, 1);
        uint8_t length[8];
        for (unsigned i = 0; i != 8; ++i) {
            length[7 - i] = static_cast<uint8_t>(original_bits >> (i * 8));
        }
        update(length, sizeof(length));

        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const uint32_t word : state_) out << std::setw(8) << word;
        return out.str();
    }

private:
    static constexpr std::array<uint32_t, 64> k{{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U}};

    static uint32_t rotate_right(uint32_t x, unsigned n) {
        return (x >> n) | (x << (32 - n));
    }

    void transform(const uint8_t * data) {
        std::array<uint32_t, 64> words{};
        for (unsigned i = 0; i != 16; ++i) {
            words[i] = (static_cast<uint32_t>(data[i * 4]) << 24) |
                       (static_cast<uint32_t>(data[i * 4 + 1]) << 16) |
                       (static_cast<uint32_t>(data[i * 4 + 2]) << 8) |
                       static_cast<uint32_t>(data[i * 4 + 3]);
        }
        for (unsigned i = 16; i != words.size(); ++i) {
            const uint32_t s0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
            const uint32_t s1 = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (unsigned i = 0; i != words.size(); ++i) {
            const uint32_t S1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + S1 + choice + k[i] + words[i];
            const uint32_t S0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = S0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_;
    std::array<uint8_t, sha_block_size> block_{};
    std::size_t buffered_ = 0;
    uint64_t bit_count_ = 0;
};

std::string sha256_bytes(const uint8_t * data, std::size_t size) {
    sha256 hash;
    if (size != 0) hash.update(data, size);
    return hash.finish();
}

std::string sha256_text(const std::string & value) {
    return sha256_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

void validate_tokens(const std::vector<int32_t> & tokens) {
    if (tokens.size() % stream_count != 0) {
        throw std::invalid_argument("LeVo token tensor must have exactly three rows");
    }
    for (const int32_t token : tokens) {
        if (token < first_token_id || token > special_token_id) {
            throw std::invalid_argument("LeVo token ID is outside [0, 16385]");
        }
    }
}

void validate_metadata(const artifact_metadata & metadata) {
    if (!std::isfinite(metadata.duration_seconds) || metadata.duration_seconds < 0.0) {
        throw std::invalid_argument("duration_seconds must be finite and non-negative");
    }
    if (metadata.frame_rate == 0 || metadata.sample_rate == 0) {
        throw std::invalid_argument("frame and sample rates must be positive");
    }
    if (metadata.eos_id != eos_token_id || metadata.special_id != special_token_id) {
        throw std::invalid_argument("LeVo token IDs do not match the canonical codec IDs");
    }
    for (const int32_t delay : metadata.delays) {
        if (delay < 0) throw std::invalid_argument("LeVo stream delays must be non-negative");
    }
    if (!std::isfinite(metadata.sampling.temperature) || metadata.sampling.temperature < 0.0F) {
        throw std::invalid_argument("sampling temperature must be finite and non-negative");
    }
    if (!std::isfinite(metadata.sampling.repetition_penalty) || metadata.sampling.repetition_penalty <= 0.0F) {
        throw std::invalid_argument("sampling repetition penalty must be finite and positive");
    }
    if (!std::isfinite(metadata.cfg_scale)) {
        throw std::invalid_argument("CFG scale must be finite");
    }
    for (const int64_t token : metadata.sampling.ignore_tokens) {
        if (token < first_token_id || token > special_token_id) {
            throw std::invalid_argument("sampling ignore token is outside [0, 16385]");
        }
    }
    for (const double seconds : {metadata.backend_seconds, metadata.model_load_seconds,
                                 metadata.conditioning_seconds, metadata.prefill_seconds,
                                 metadata.generation_seconds, metadata.total_seconds}) {
        if (!std::isfinite(seconds) || seconds < 0.0) {
            throw std::invalid_argument("generation timings must be finite and non-negative");
        }
    }
}

void put_u16(std::ostream & out, uint16_t value) {
    const char bytes[2] = {static_cast<char>(value & 0xffU), static_cast<char>(value >> 8)};
    out.write(bytes, sizeof(bytes));
}

void put_u32(std::ostream & out, uint32_t value) {
    const char bytes[4] = {static_cast<char>(value & 0xffU), static_cast<char>((value >> 8) & 0xffU),
                           static_cast<char>((value >> 16) & 0xffU), static_cast<char>(value >> 24)};
    out.write(bytes, sizeof(bytes));
}

void put_i32(std::ostream & out, int32_t value) {
    put_u32(out, static_cast<uint32_t>(value));
}

std::string npy_header(std::size_t frames, uint8_t & major, uint32_t & header_size) {
    const std::string base = "{'descr': '<i4', 'fortran_order': False, 'shape': (3, " +
                             std::to_string(frames) + ",), }";
    major = 1;
    std::size_t padding = (16 - ((10 + base.size() + 1) % 16)) % 16;
    if (base.size() + padding + 1 > std::numeric_limits<uint16_t>::max()) {
        major = 2;
        padding = (16 - ((12 + base.size() + 1) % 16)) % 16;
    }
    const std::size_t size = base.size() + padding + 1;
    if (size > std::numeric_limits<uint32_t>::max() || (major == 1 && size > std::numeric_limits<uint16_t>::max())) {
        throw std::length_error("NumPy header is too large");
    }
    header_size = static_cast<uint32_t>(size);
    return base + std::string(padding, ' ') + "\n";
}

void write_npy_file(const std::filesystem::path & path, const std::vector<int32_t> & tokens) {
    validate_tokens(tokens);
    const std::size_t frames = tokens.size() / stream_count;
    uint8_t major = 0;
    uint32_t header_size = 0;
    const std::string header = npy_header(frames, major, header_size);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open NumPy token artifact for writing: " + path.string());
    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    out.put(static_cast<char>(major));
    out.put('\0');
    if (major == 1) put_u16(out, static_cast<uint16_t>(header_size));
    else put_u32(out, header_size);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const int32_t token : tokens) put_i32(out, token);
    out.flush();
    if (!out) throw std::runtime_error("failed while writing NumPy token artifact: " + path.string());
}

void validate_npy_path(const std::filesystem::path & path) {
    if (path.filename().empty() || path.extension() != ".npy") {
        throw std::invalid_argument("LeVo token artifact path must use the .npy extension");
    }
}

std::vector<uint8_t> read_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open NumPy token artifact: " + path.string());
    in.seekg(0, std::ios::end);
    const std::streamoff end = in.tellg();
    if (end < 0) throw std::runtime_error("cannot determine NumPy token artifact size");
    in.seekg(0, std::ios::beg);
    const auto size = static_cast<uint64_t>(end);
    if (size > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::length_error("NumPy token artifact is too large");
    }
    std::vector<uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in && !bytes.empty()) throw std::runtime_error("failed while reading NumPy token artifact");
    return bytes;
}

uint16_t get_u16(const std::vector<uint8_t> & bytes, std::size_t at) {
    return static_cast<uint16_t>(bytes[at]) | (static_cast<uint16_t>(bytes[at + 1]) << 8);
}

uint32_t get_u32(const std::vector<uint8_t> & bytes, std::size_t at) {
    return static_cast<uint32_t>(bytes[at]) | (static_cast<uint32_t>(bytes[at + 1]) << 8) |
           (static_cast<uint32_t>(bytes[at + 2]) << 16) | (static_cast<uint32_t>(bytes[at + 3]) << 24);
}

std::size_t skip_space(const std::string & text, std::size_t at) {
    while (at < text.size() && (text[at] == ' ' || text[at] == '\t' || text[at] == '\r' || text[at] == '\n')) ++at;
    return at;
}

std::size_t parse_number(const std::string & text, std::size_t & at) {
    at = skip_space(text, at);
    const std::size_t begin = at;
    while (at < text.size() && text[at] >= '0' && text[at] <= '9') ++at;
    if (begin == at) throw std::invalid_argument("NumPy token artifact has an invalid shape");
    const std::string value = text.substr(begin, at - begin);
    try {
        const unsigned long long parsed = std::stoull(value);
        if (parsed > std::numeric_limits<std::size_t>::max()) throw std::out_of_range("shape");
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception &) {
        throw std::invalid_argument("NumPy token artifact shape is too large");
    }
}

std::string header_string_value(const std::string & header, const std::string & key) {
    const std::size_t key_at = header.find(key);
    if (key_at == std::string::npos) throw std::invalid_argument("NumPy token artifact header is missing " + key);
    const std::size_t colon = header.find(':', key_at + key.size());
    if (colon == std::string::npos) throw std::invalid_argument("NumPy token artifact header has an invalid " + key);
    std::size_t at = skip_space(header, colon + 1);
    if (at >= header.size() || header[at++] != '\'') throw std::invalid_argument("NumPy token artifact header has an invalid " + key);
    const std::size_t end = header.find('\'', at);
    if (end == std::string::npos) throw std::invalid_argument("NumPy token artifact header has an invalid " + key);
    return header.substr(at, end - at);
}

std::pair<std::size_t, std::size_t> parse_shape(const std::string & header) {
    const std::size_t key = header.find("'shape'");
    if (key == std::string::npos) throw std::invalid_argument("NumPy token artifact is missing shape");
    const std::size_t colon = header.find(':', key);
    if (colon == std::string::npos) throw std::invalid_argument("NumPy token artifact has an invalid shape");
    std::size_t at = skip_space(header, colon + 1);
    if (at >= header.size() || header[at++] != '(') throw std::invalid_argument("NumPy token artifact has an invalid shape");
    const std::size_t rows = parse_number(header, at);
    at = skip_space(header, at);
    if (at >= header.size() || header[at++] != ',') throw std::invalid_argument("NumPy token artifact has an invalid shape");
    const std::size_t columns = parse_number(header, at);
    at = skip_space(header, at);
    if (at < header.size() && header[at] == ',') ++at;
    at = skip_space(header, at);
    if (at >= header.size() || header[at] != ')') throw std::invalid_argument("NumPy token artifact has an invalid shape");
    return {rows, columns};
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
            else out << static_cast<char>(c);
        }
    }
    out << '"';
    return out.str();
}

std::string json_number(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

std::string manifest(const std::filesystem::path & npy_path,
                     const std::vector<int32_t> & tokens,
                     const artifact_metadata & metadata) {
    const std::size_t frames = tokens.size() / stream_count;
    const std::string hash = tensor_sha256(tokens);
    std::ostringstream out;
    out << "{\n"
        << "  \"format\": \"levo2-token-artifact\",\n"
        << "  \"schema_version\": 1,\n"
        << "  \"artifact\": {\n"
        << "    \"type\": \"levo2-tokens\",\n"
        << "    \"filename\": " << json_escape(npy_path.filename().generic_string()) << ",\n"
        << "    \"format\": \"numpy-npy\",\n"
        << "    \"npy_version\": \"1.0\"\n"
        << "  },\n"
        << "  \"tensor\": {\n"
        << "    \"shape\": [3, " << frames << "],\n"
        << "    \"dtype\": \"int32\",\n"
        << "    \"order\": \"C\",\n"
        << "    \"sha256\": " << json_escape(hash) << "\n"
        << "  },\n"
        << "  \"model\": {\n"
        << "    \"name\": " << json_escape(metadata.model_name) << ",\n"
        << "    \"revision\": " << json_escape(metadata.model_revision) << ",\n"
        << "    \"sha256\": " << json_escape(metadata.model_sha256) << ",\n"
        << "    \"artifact_sha256\": " << json_escape(metadata.model_artifact_sha256) << "\n"
        << "  },\n"
        << "  \"provenance\": {\n"
        << "    \"generator\": " << json_escape(metadata.generator) << ",\n"
        << "    \"generator_revision\": " << json_escape(metadata.generator_revision) << ",\n"
        << "    \"runtime_revision\": " << json_escape(metadata.runtime_revision) << ",\n"
        << "    \"tokenizer_revision\": " << json_escape(metadata.tokenizer_revision) << ",\n"
        << "    \"tokenizer_sha256\": " << json_escape(metadata.tokenizer_sha256) << ",\n"
        << "    \"backend\": " << json_escape(metadata.backend_name) << "\n"
        << "  },\n"
        << "  \"inputs\": {\n"
        << "    \"lyrics\": " << json_escape(metadata.lyrics) << ",\n"
        << "    \"lyrics_sha256\": " << json_escape(sha256_text(metadata.lyrics)) << ",\n"
        << "    \"description\": " << json_escape(metadata.description) << ",\n"
        << "    \"description_sha256\": " << json_escape(sha256_text(metadata.description)) << "\n"
        << "  },\n"
        << "  \"duration\": {\n"
        << "    \"seconds\": " << json_number(metadata.duration_seconds) << ",\n"
        << "    \"frames\": " << frames << ",\n"
        << "    \"frame_rate\": " << metadata.frame_rate << "\n"
        << "  },\n"
        << "  \"config\": {\n"
        << "    \"sample_rate\": " << metadata.sample_rate << ",\n"
        << "    \"frame_rate\": " << metadata.frame_rate << ",\n"
        << "    \"eos_token_id\": " << metadata.eos_id << ",\n"
        << "    \"special_token_id\": " << metadata.special_id << ",\n"
        << "    \"delays\": [" << metadata.delays[0] << ", " << metadata.delays[1] << ", " << metadata.delays[2] << "]\n"
        << "  },\n"
        << "  \"sampling\": {\n"
        << "    \"cfg_scale\": " << json_number(metadata.cfg_scale) << ",\n"
        << "    \"use_sampling\": " << (metadata.sampling.use_sampling ? "true" : "false") << ",\n"
        << "    \"temperature\": " << json_number(metadata.sampling.temperature) << ",\n"
        << "    \"top_k_mixed\": " << metadata.sampling.top_k_mixed << ",\n"
        << "    \"top_k_detail\": " << metadata.sampling.top_k_detail << ",\n"
        << "    \"repetition_window\": " << metadata.sampling.repetition_window << ",\n"
        << "    \"repetition_penalty\": " << json_number(metadata.sampling.repetition_penalty) << ",\n"
        << "    \"ignore_tokens\": [";
    for (std::size_t i = 0; i != metadata.sampling.ignore_tokens.size(); ++i) {
        if (i != 0) out << ", ";
        out << metadata.sampling.ignore_tokens[i];
    }
    out << "],\n"
        << "    \"seed\": ";
    if (metadata.seed_present) out << metadata.seed;
    else out << "null";
    out << "\n  },\n"
        << "  \"timings\": {\n"
        << "    \"backend_seconds\": " << json_number(metadata.backend_seconds) << ",\n"
        << "    \"model_load_seconds\": " << json_number(metadata.model_load_seconds) << ",\n"
        << "    \"conditioning_seconds\": " << json_number(metadata.conditioning_seconds) << ",\n"
        << "    \"prefill_seconds\": " << json_number(metadata.prefill_seconds) << ",\n"
        << "    \"generation_seconds\": " << json_number(metadata.generation_seconds) << ",\n"
        << "    \"total_seconds\": " << json_number(metadata.total_seconds) << "\n"
        << "  }\n}\n";
    return out.str();
}

std::atomic<uint64_t> temporary_counter{0};

std::filesystem::path temporary_path(const std::filesystem::path & target) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto id = temporary_counter.fetch_add(1, std::memory_order_relaxed);
    return target.parent_path() / (target.filename().string() + ".tmp-" + std::to_string(now) + "-" + std::to_string(id));
}

void commit_file(const std::filesystem::path & temporary, const std::filesystem::path & target) {
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (!error) return;
    // std::filesystem::rename is replacement-atomic on POSIX but not on all
    // supported filesystems. Keep the fallback narrow and report its failure.
    if (error != std::make_error_code(std::errc::file_exists)) {
        throw std::runtime_error("cannot commit token artifact " + target.string() + ": " + error.message());
    }
    std::error_code remove_error;
    std::filesystem::remove(target, remove_error);
    if (remove_error) throw std::runtime_error("cannot replace token artifact " + target.string() + ": " + remove_error.message());
    std::filesystem::rename(temporary, target, error);
    if (error) throw std::runtime_error("cannot commit token artifact " + target.string() + ": " + error.message());
}

} // namespace

std::filesystem::path metadata_path(const std::filesystem::path & npy_path) {
    auto result = npy_path;
    result.replace_extension(".json");
    return result;
}

std::string tensor_sha256(const std::vector<int32_t> & tokens) {
    validate_tokens(tokens);
    sha256 hash;
    std::array<uint8_t, 4> bytes{};
    for (const int32_t token : tokens) {
        const uint32_t value = static_cast<uint32_t>(token);
        bytes[0] = static_cast<uint8_t>(value & 0xffU);
        bytes[1] = static_cast<uint8_t>((value >> 8) & 0xffU);
        bytes[2] = static_cast<uint8_t>((value >> 16) & 0xffU);
        bytes[3] = static_cast<uint8_t>(value >> 24);
        hash.update(bytes.data(), bytes.size());
    }
    return hash.finish();
}

std::string file_sha256(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open file for SHA-256: " + path.string());
    sha256 hash;
    std::array<uint8_t, 1024U * 1024U> buffer{};
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<std::size_t>(count));
    }
    if (!input.eof()) throw std::runtime_error("failed while hashing file: " + path.string());
    return hash.finish();
}

void write_tokens_npy(const std::filesystem::path & npy_path, const std::vector<int32_t> & tokens) {
    validate_npy_path(npy_path);
    const auto temporary = temporary_path(npy_path);
    try {
        write_npy_file(temporary, tokens);
        commit_file(temporary, npy_path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void write(const std::filesystem::path & npy_path,
           const std::vector<int32_t> & tokens,
           const artifact_metadata & metadata) {
    validate_tokens(tokens);
    validate_metadata(metadata);
    validate_npy_path(npy_path);
    const auto json_path = metadata_path(npy_path);
    const auto npy_temporary = temporary_path(npy_path);
    const auto json_temporary = temporary_path(json_path);
    try {
        write_npy_file(npy_temporary, tokens);
        std::ofstream json(json_temporary, std::ios::binary | std::ios::trunc);
        if (!json) throw std::runtime_error("cannot open token manifest for writing: " + json_path.string());
        const std::string text = manifest(npy_path, tokens, metadata);
        json.write(text.data(), static_cast<std::streamsize>(text.size()));
        json.flush();
        if (!json) throw std::runtime_error("failed while writing token manifest: " + json_path.string());
        commit_file(npy_temporary, npy_path);
        commit_file(json_temporary, json_path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(npy_temporary, ignored);
        std::filesystem::remove(json_temporary, ignored);
        throw;
    }
}

std::vector<int32_t> read_tokens_npy(const std::filesystem::path & npy_path) {
    const std::vector<uint8_t> bytes = read_file(npy_path);
    if (bytes.size() < 10 || bytes[0] != 0x93 || bytes[1] != 'N' || bytes[2] != 'U' || bytes[3] != 'M' || bytes[4] != 'P' || bytes[5] != 'Y') {
        throw std::invalid_argument("not a NumPy token artifact");
    }
    const uint8_t major = bytes[6];
    const uint8_t minor = bytes[7];
    if ((major != 1 && major != 2) || minor != 0) throw std::invalid_argument("NumPy token artifact must use format 1.0 or 2.0");
    const std::size_t preamble = major == 1 ? 10 : 12;
    if (bytes.size() < preamble) throw std::invalid_argument("truncated NumPy token artifact header");
    const uint64_t header_size = major == 1 ? get_u16(bytes, 8) : get_u32(bytes, 8);
    if (header_size > bytes.size() - preamble || ((preamble + header_size) % 16) != 0) {
        throw std::invalid_argument("invalid NumPy token artifact header length");
    }
    const std::string header(reinterpret_cast<const char *>(bytes.data() + preamble), static_cast<std::size_t>(header_size));
    if (header.empty() || header.back() != '\n') throw std::invalid_argument("NumPy token artifact header is not newline terminated");
    if (header_string_value(header, "'descr'") != "<i4") {
        throw std::invalid_argument("NumPy token artifact dtype must be little-endian int32");
    }
    const std::size_t order_key = header.find("'fortran_order'");
    if (order_key == std::string::npos) {
        throw std::invalid_argument("NumPy token artifact must be C-order");
    }
    const std::size_t order_colon = header.find(':', order_key + std::string("'fortran_order'").size());
    const std::size_t order_at = order_colon == std::string::npos ? header.size() : skip_space(header, order_colon + 1);
    if (order_at + 5 > header.size() || header.compare(order_at, 5, "False") != 0) {
        throw std::invalid_argument("NumPy token artifact must be C-order");
    }
    const auto shape = parse_shape(header);
    if (shape.first != stream_count) throw std::invalid_argument("LeVo token artifact must have shape [3,T]");
    if (shape.second > std::numeric_limits<std::size_t>::max() / stream_count / sizeof(int32_t)) {
        throw std::length_error("LeVo token artifact has too many tokens");
    }
    const std::size_t payload_offset = preamble + static_cast<std::size_t>(header_size);
    const std::size_t payload_size = shape.second * stream_count * sizeof(int32_t);
    if (bytes.size() != payload_offset + payload_size) throw std::invalid_argument("NumPy token artifact payload size does not match shape");
    std::vector<int32_t> tokens(shape.second * stream_count);
    for (std::size_t i = 0; i != tokens.size(); ++i) {
        const std::size_t at = payload_offset + i * 4;
        const uint32_t value = get_u32(bytes, at);
        if (value > static_cast<uint32_t>(special_token_id)) throw std::invalid_argument("LeVo token ID is outside [0, 16385]");
        tokens[i] = static_cast<int32_t>(value);
    }
    return tokens;
}

artifact read(const std::filesystem::path & npy_path) {
    artifact result;
    result.tokens = read_tokens_npy(npy_path);
    result.frame_count = result.tokens.size() / stream_count;
    return result;
}

} // namespace token_io
} // namespace levo
