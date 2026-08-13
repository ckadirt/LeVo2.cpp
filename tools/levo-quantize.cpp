#include "levo-quantization.h"
#include "levo-token-io.h"

#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t k_chunk_elements = 1024U * 1024U;

struct gguf_deleter {
    void operator()(gguf_context * context) const noexcept { if (context) gguf_free(context); }
};
struct context_deleter {
    void operator()(ggml_context * context) const noexcept { if (context) ggml_free(context); }
};
using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("levo-quantize: " + message);
}

void usage(const char * program) {
    std::cout << "Usage:\n  " << program
              << " --input SOURCE.gguf --output QUANTIZED.gguf"
                 " --profile Q8_0|Q6_K|Q5_K_M|Q4_K_M\n  " << program
              << " --input VAE-F32.gguf --output VAE-F16.gguf --vae-f16\n\n"
                 "Quantizes strict F32/F16 LeLM or Flow GGUF artifacts, or performs "
                 "the separately tagged F32-to-F16 VAE conversion. Existing outputs "
                 "and already-converted VAE inputs are rejected. The output receives "
                 "a checksum and deterministic manifest beside the GGUF.\n";
}

struct options {
    std::filesystem::path input;
    std::filesystem::path output;
    levo::quantization::profile profile = levo::quantization::profile::q8_0;
    bool vae_f16 = false;
};

options parse(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        usage(argv[0]);
        std::exit(0);
    }
    options result;
    bool profile_present = false;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        const auto value = [&]() -> std::string {
            if (++index >= argc) fail("missing value for " + option);
            return argv[index];
        };
        if (option == "--input") result.input = value();
        else if (option == "--output") result.output = value();
        else if (option == "--profile") {
            const auto parsed = levo::quantization::parse(value());
            if (!parsed) fail("unsupported profile; use Q8_0, Q6_K, Q5_K_M, or Q4_K_M");
            result.profile = *parsed;
            profile_present = true;
        } else if (option == "--vae-f16") {
            result.vae_f16 = true;
        } else if (option == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            fail("unknown option " + option);
        }
    }
    if (result.input.empty() || result.output.empty() || (profile_present == result.vae_f16)) {
        fail("--input and --output plus exactly one of --profile or --vae-f16 are required");
    }
    if (result.input.extension() != ".gguf" || result.output.extension() != ".gguf") {
        fail("--input and --output must use the .gguf extension");
    }
    if (result.input == result.output) fail("--output must differ from --input");
    return result;
}

std::uint64_t read_u64(std::istream & input, const char * what) {
    std::uint64_t value = 0;
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input) fail(std::string("truncated GGUF while reading ") + what);
    return value;
}

std::uint32_t read_u32(std::istream & input, const char * what) {
    std::uint32_t value = 0;
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input) fail(std::string("truncated GGUF while reading ") + what);
    return value;
}

std::int64_t read_i64(std::istream & input, const char * what) {
    std::int64_t value = 0;
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input) fail(std::string("truncated GGUF while reading ") + what);
    return value;
}

void checked_skip(std::istream & input, std::uint64_t bytes, const char * what) {
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        fail(std::string("GGUF ") + what + " is too large");
    }
    input.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!input) fail(std::string("truncated GGUF while skipping ") + what);
}

std::string read_string(std::istream & input, const char * what) {
    const std::uint64_t count = read_u64(input, what);
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) || count > 1024U * 1024U) {
        fail(std::string("GGUF ") + what + " is unreasonably large");
    }
    std::string result(static_cast<std::size_t>(count), '\0');
    if (count != 0) input.read(result.data(), static_cast<std::streamsize>(count));
    if (!input) fail(std::string("truncated GGUF while reading ") + what);
    return result;
}

void skip_string(std::istream & input, const char * what) {
    checked_skip(input, read_u64(input, what), what);
}

std::uint64_t scalar_size(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        case GGUF_TYPE_STRING: case GGUF_TYPE_ARRAY: case GGUF_TYPE_COUNT: break;
    }
    fail("GGUF contains an unsupported metadata type");
}

void skip_values(std::istream & input, gguf_type type, std::uint64_t count, const char * what) {
    if (type == GGUF_TYPE_STRING) {
        for (std::uint64_t index = 0; index < count; ++index) skip_string(input, what);
        return;
    }
    const std::uint64_t size = scalar_size(type);
    if (count != 0 && size > std::numeric_limits<std::uint64_t>::max() / count) {
        fail(std::string("GGUF ") + what + " size overflows");
    }
    checked_skip(input, count * size, what);
}

struct raw_tensor_info {
    std::string name;
    int rank = 0;
};

std::vector<raw_tensor_info> read_tensor_ranks(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("cannot open input GGUF " + path.string());
    std::array<char, 4> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || std::string(magic.data(), magic.size()) != "GGUF") fail("input is not a GGUF file");
    (void) read_u32(input, "version");
    const std::int64_t tensor_count = read_i64(input, "tensor count");
    const std::int64_t key_count = read_i64(input, "metadata count");
    if (tensor_count < 0 || key_count < 0) fail("GGUF has negative inventory counts");
    for (std::int64_t index = 0; index < key_count; ++index) {
        (void) read_string(input, "metadata key");
        gguf_type type = static_cast<gguf_type>(read_u32(input, "metadata type"));
        std::uint64_t count = 1;
        if (type == GGUF_TYPE_ARRAY) {
            type = static_cast<gguf_type>(read_u32(input, "metadata array type"));
            count = read_u64(input, "metadata array count");
        }
        skip_values(input, type, count, "metadata value");
    }
    std::vector<raw_tensor_info> result;
    result.reserve(static_cast<std::size_t>(tensor_count));
    for (std::int64_t index = 0; index < tensor_count; ++index) {
        raw_tensor_info item;
        item.name = read_string(input, "tensor name");
        const std::uint32_t rank = read_u32(input, "tensor rank");
        if (rank == 0 || rank > GGML_MAX_DIMS) fail("tensor '" + item.name + "' has invalid rank");
        item.rank = static_cast<int>(rank);
        if (rank > std::numeric_limits<std::uint64_t>::max() / sizeof(std::int64_t)) {
            fail("tensor '" + item.name + "' rank overflows");
        }
        checked_skip(input, static_cast<std::uint64_t>(rank) * sizeof(std::int64_t), "tensor shape");
        checked_skip(input, sizeof(std::int32_t) + sizeof(std::uint64_t), "tensor type/offset");
        result.push_back(std::move(item));
    }
    return result;
}

int64_t required_key(const gguf_context * context, const char * key, gguf_type type) {
    const int64_t index = gguf_find_key(context, key);
    if (index < 0 || gguf_get_kv_type(context, index) != type) {
        fail(std::string("input is missing required ") + key + " metadata");
    }
    return index;
}

std::string required_string(const gguf_context * context, const char * key) {
    return gguf_get_val_str(context, required_key(context, key, GGUF_TYPE_STRING));
}

std::uint32_t required_u32(const gguf_context * context, const char * key) {
    return gguf_get_val_u32(context, required_key(context, key, GGUF_TYPE_UINT32));
}

enum class component { lelm, flow, vae };

const char * component_name(component value) {
    switch (value) {
        case component::lelm: return "LeLM";
        case component::flow: return "Flow";
        case component::vae: return "VAE";
    }
    fail("unknown component");
}

component classify(const gguf_context * context) {
    const std::string architecture = required_string(context, "general.architecture");
    if (architecture == "levo2") return component::lelm;
    if (architecture == "levo2_flow") return component::flow;
    if (architecture == "levo2_vae") return component::vae;
    fail("unsupported general.architecture '" + architecture + "'");
}

int64_t checked_product(const int64_t * shape, int rank, const std::string & name) {
    int64_t result = 1;
    for (int dimension = 0; dimension < rank; ++dimension) {
        if (shape[dimension] <= 0 || result > std::numeric_limits<int64_t>::max() / shape[dimension]) {
            fail("tensor '" + name + "' has invalid shape");
        }
        result *= shape[dimension];
    }
    return result;
}

struct tensor_plan {
    std::string name;
    int rank = 0;
    std::array<int64_t, GGML_MAX_DIMS> source_shape{{1, 1, 1, 1}};
    std::array<int64_t, GGML_MAX_DIMS> storage_shape{{1, 1, 1, 1}};
    ggml_type source_type = GGML_TYPE_COUNT;
    ggml_type target_type = GGML_TYPE_COUNT;
    std::size_t source_offset = 0;
    std::size_t source_bytes = 0;
};

std::vector<tensor_plan> make_plan(const gguf_context * source, component kind,
                                   const options & options,
                                   const std::vector<raw_tensor_info> & raw) {
    const int64_t count = gguf_get_n_tensors(source);
    if (count <= 0 || static_cast<std::size_t>(count) != raw.size()) fail("GGUF tensor inventory is inconsistent");
    std::vector<tensor_plan> result;
    result.reserve(static_cast<std::size_t>(count));
    for (int64_t index = 0; index < count; ++index) {
        tensor_plan item;
        item.name = gguf_get_tensor_name(source, index);
        if (item.name != raw[static_cast<std::size_t>(index)].name) fail("GGUF tensor order is inconsistent");
        item.rank = raw[static_cast<std::size_t>(index)].rank;
        item.source_type = gguf_get_tensor_type(source, index);
        if (item.source_type != GGML_TYPE_F32 && item.source_type != GGML_TYPE_F16) {
            fail("input tensor '" + item.name + "' is not F32/F16; already-quantized inputs are refused");
        }
        const int64_t * shape = gguf_get_tensor_ne(source, index);
        for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) item.source_shape[dimension] = shape[dimension];
        item.storage_shape = item.source_shape;
        item.target_type = kind == component::vae
            ? GGML_TYPE_F16
            : kind == component::lelm
                ? levo::quantization::lelm_tensor_type(item.name, item.rank, options.profile)
                : levo::quantization::flow_tensor_type(item.name, item.rank, options.profile);
        if (kind == component::flow && levo::quantization::flow_block_matrix(item.name)) {
            item.storage_shape[0] = levo::quantization::padded_input_columns(item.source_shape[0], item.target_type);
        }
        if (item.target_type == GGML_TYPE_COUNT) fail("no quantization type selected for '" + item.name + "'");
        if (item.storage_shape[0] % ggml_blck_size(item.target_type) != 0) {
            fail("selected GGML type does not divide tensor row length for '" + item.name + "'");
        }
        (void) checked_product(item.source_shape.data(), item.rank, item.name);
        (void) checked_product(item.storage_shape.data(), item.rank, item.name);
        item.source_offset = gguf_get_tensor_offset(source, index);
        item.source_bytes = gguf_get_tensor_size(source, index);
        result.push_back(std::move(item));
    }
    return result;
}

void validate_source_file_type(const gguf_context * source, component kind, const std::vector<tensor_plan> & plan) {
    const uint32_t file_type = required_u32(source, "general.file_type");
    const ggml_type expected = file_type == 0 ? GGML_TYPE_F32 : file_type == 1 ? GGML_TYPE_F16 : GGML_TYPE_COUNT;
    if (expected == GGML_TYPE_COUNT) fail("input general.file_type must be F32 (0) or F16 (1)");
    if (kind == component::vae && expected != GGML_TYPE_F32) {
        fail("VAE F16 conversion requires a strict F32 VAE input; already-converted VAE inputs are refused");
    }
    for (const tensor_plan & item : plan) {
        if (item.source_type != expected) {
            fail("input tensor '" + item.name + "' is inconsistent with general.file_type");
        }
    }
}

void write_padding(std::ofstream & output, std::size_t bytes, std::size_t alignment) {
    const std::size_t padding = (alignment - bytes % alignment) % alignment;
    if (padding == 0) return;
    std::array<char, GGUF_DEFAULT_ALIGNMENT> zeros{};
    output.write(zeros.data(), static_cast<std::streamsize>(padding));
    if (!output) fail("cannot write GGUF tensor alignment padding");
}

void require_finite(const std::vector<float> & values, const std::string & name) {
    for (float value : values) if (!std::isfinite(value)) fail("input tensor '" + name + "' contains a non-finite value");
}

void write_tensor(std::ifstream & input, std::ofstream & output, std::size_t data_offset,
                  const tensor_plan & item, std::size_t alignment) {
    const int64_t source_columns = item.source_shape[0];
    const int64_t target_columns = item.storage_shape[0];
    const int64_t elements = checked_product(item.source_shape.data(), item.rank, item.name);
    const int64_t rows = elements / source_columns;
    if (rows <= 0 || item.source_offset > std::numeric_limits<std::size_t>::max() - data_offset) {
        fail("invalid source range for '" + item.name + "'");
    }
    input.clear();
    input.seekg(static_cast<std::streamoff>(data_offset + item.source_offset));
    if (!input) fail("cannot seek source tensor '" + item.name + "'");
    const std::size_t rows_per_chunk = std::max<std::size_t>(1, k_chunk_elements / static_cast<std::size_t>(source_columns));
    const std::size_t source_element_bytes = item.source_type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
    const std::size_t target_row_bytes = ggml_row_size(item.target_type, target_columns);
    std::size_t written = 0;
    for (int64_t row_offset = 0; row_offset < rows;) {
        const std::size_t chunk_rows = std::min<std::size_t>(rows_per_chunk, static_cast<std::size_t>(rows - row_offset));
        const std::size_t chunk_elements = chunk_rows * static_cast<std::size_t>(source_columns);
        std::vector<float> values(chunk_elements);
        if (item.source_type == GGML_TYPE_F32) {
            input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(chunk_elements * sizeof(float)));
        } else {
            std::vector<ggml_fp16_t> source_values(chunk_elements);
            input.read(reinterpret_cast<char *>(source_values.data()), static_cast<std::streamsize>(chunk_elements * sizeof(ggml_fp16_t)));
            if (input) ggml_fp16_to_fp32_row(source_values.data(), values.data(), static_cast<int64_t>(chunk_elements));
        }
        if (!input) fail("truncated source tensor '" + item.name + "'");
        require_finite(values, item.name);
        if (item.target_type == GGML_TYPE_F32) {
            output.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
            if (!output) fail("cannot write F32 tensor '" + item.name + "'");
            written += values.size() * sizeof(float);
        } else if (item.target_type == GGML_TYPE_F16) {
            if (target_columns != source_columns) fail("F16 conversion cannot change tensor storage width");
            std::vector<ggml_fp16_t> converted(values.size());
            ggml_fp32_to_fp16_row(values.data(), converted.data(), static_cast<int64_t>(values.size()));
            output.write(reinterpret_cast<const char *>(converted.data()), static_cast<std::streamsize>(converted.size() * sizeof(ggml_fp16_t)));
            if (!output) fail("cannot write F16 tensor '" + item.name + "'");
            written += converted.size() * sizeof(ggml_fp16_t);
        } else {
            std::vector<float> padded;
            const float * quant_input = values.data();
            if (target_columns != source_columns) {
                padded.assign(chunk_rows * static_cast<std::size_t>(target_columns), 0.0F);
                for (std::size_t row = 0; row < chunk_rows; ++row) {
                    std::memcpy(padded.data() + row * static_cast<std::size_t>(target_columns),
                                values.data() + row * static_cast<std::size_t>(source_columns),
                                static_cast<std::size_t>(source_columns) * sizeof(float));
                }
                quant_input = padded.data();
            }
            std::vector<std::uint8_t> quantized(chunk_rows * target_row_bytes);
            const std::size_t produced = ggml_quantize_chunk(item.target_type, quant_input, quantized.data(), 0,
                                                             static_cast<int64_t>(chunk_rows), target_columns, nullptr);
            if (produced != quantized.size()) fail("GGML quantizer produced an unexpected size for '" + item.name + "'");
            output.write(reinterpret_cast<const char *>(quantized.data()), static_cast<std::streamsize>(quantized.size()));
            if (!output) fail("cannot write quantized tensor '" + item.name + "'");
            written += quantized.size();
        }
        row_offset += static_cast<int64_t>(chunk_rows);
        (void) source_element_bytes;
    }
    const std::size_t expected = ggml_row_size(item.target_type, target_columns) * static_cast<std::size_t>(rows);
    if (written != expected) fail("incorrect output size for '" + item.name + "'");
    write_padding(output, written, alignment);
}

std::string json_escape(const std::string & value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char character : value) {
        switch (character) {
            case '"': result << "\\\""; break;
            case '\\': result << "\\\\"; break;
            case '\b': result << "\\b"; break;
            case '\f': result << "\\f"; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default:
                if (character < 0x20U) result << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(character) << std::dec;
                else result << static_cast<char>(character);
        }
    }
    result << '"';
    return result.str();
}

std::string json_shape(const std::array<int64_t, GGML_MAX_DIMS> & shape, int rank) {
    std::ostringstream result;
    result << '[';
    for (int index = 0; index < rank; ++index) {
        if (index != 0) result << ", ";
        result << shape[index];
    }
    return result.str() + ']';
}

std::string make_manifest(const options & options, component kind, const std::string & source_sha256,
                          const std::string & artifact_sha256, std::uintmax_t source_bytes,
                          std::uintmax_t artifact_bytes, const std::vector<tensor_plan> & plan) {
    std::map<std::string, std::size_t> type_counts;
    for (const tensor_plan & item : plan) ++type_counts[ggml_type_name(item.target_type)];
    std::ostringstream out;
    out << "{\n"
        << "  \"artifact\": {\n"
        << "    \"bytes\": " << artifact_bytes << ",\n"
        << "    \"filename\": " << json_escape(options.output.filename().string()) << ",\n"
        << "    \"sha256\": " << json_escape(artifact_sha256) << "\n"
        << "  },\n"
        << "  \"component\": " << json_escape(component_name(kind)) << ",\n";
    if (kind == component::vae) {
        out << "  \"precision\": {\n"
            << "    \"policy_revision\": \"1\",\n"
            << "    \"profile\": \"F16\"\n"
            << "  },\n";
    } else {
        out << "  \"quantization\": {\n"
            << "    \"policy_revision\": \"1\",\n"
            << "    \"profile\": " << json_escape(levo::quantization::name(options.profile));
        if (kind == component::flow) out << ",\n    \"padded_input_layout\": " << json_escape(levo::quantization::flow_padded_input_layout(options.profile));
        out << "\n  },\n";
    }
    out
        << "  \"source_artifact\": {\n"
        << "    \"bytes\": " << source_bytes << ",\n"
        << "    \"filename\": " << json_escape(options.input.filename().string()) << ",\n"
        << "    \"sha256\": " << json_escape(source_sha256) << "\n"
        << "  },\n"
        << "  \"tensor_count\": " << plan.size() << ",\n"
        << "  \"tensor_type_counts\": {";
    bool first = true;
    for (const auto & item : type_counts) {
        out << (first ? "\n" : ",\n") << "    " << json_escape(item.first) << ": " << item.second;
        first = false;
    }
    out << "\n  },\n  \"tensors\": [\n";
    for (std::size_t index = 0; index < plan.size(); ++index) {
        const tensor_plan & item = plan[index];
        out << "    {\"ggml_type\": " << json_escape(ggml_type_name(item.target_type))
            << ", \"logical_shape\": " << json_shape(item.source_shape, item.rank)
            << ", \"name\": " << json_escape(item.name)
            << ", \"storage_shape\": " << json_shape(item.storage_shape, item.rank) << "}";
        if (index + 1 != plan.size()) out << ',';
        out << '\n';
    }
    out << "  ]\n}\n";
    return out.str();
}

void write_text_atomically(const std::filesystem::path & target, const std::string & text) {
    const std::filesystem::path temporary = target.string() + ".partial";
    if (std::filesystem::exists(temporary)) fail("temporary path already exists: " + temporary.string());
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) fail("cannot create " + temporary.string());
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        output.flush();
        if (!output) fail("cannot write " + temporary.string());
    }
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    if (error) fail("cannot install " + target.string() + ": " + error.message());
}

void run(const options & options) {
    const std::filesystem::path checksum_path = options.output.string() + ".sha256";
    const std::filesystem::path manifest_path = options.output.string() + ".manifest.json";
    if (std::filesystem::exists(options.output) || std::filesystem::exists(checksum_path) || std::filesystem::exists(manifest_path)) {
        fail("output GGUF or sidecar already exists; quantized artifacts are never overwritten");
    }
    std::error_code directory_error;
    std::filesystem::create_directories(options.output.parent_path().empty() ? "." : options.output.parent_path(), directory_error);
    if (directory_error) fail("cannot create output directory: " + directory_error.message());

    const std::vector<raw_tensor_info> raw = read_tensor_ranks(options.input);
    gguf_ptr source(gguf_init_from_file(options.input.string().c_str(), {true, nullptr}));
    if (!source) fail("cannot parse input GGUF");
    const component kind = classify(source.get());
    const std::string source_sha256 = levo::token_io::file_sha256(options.input);
    if (!levo::quantization::is_hex_sha256(source_sha256)) fail("cannot calculate source artifact SHA-256");
    if ((kind == component::vae) != options.vae_f16) {
        fail(kind == component::vae ? "VAE input requires --vae-f16" : "--vae-f16 accepts only a VAE GGUF");
    }
    const std::vector<tensor_plan> plan = make_plan(source.get(), kind, options, raw);
    validate_source_file_type(source.get(), kind, plan);

    gguf_ptr destination(gguf_init_empty());
    if (!destination) fail("cannot allocate GGUF writer context");
    gguf_set_kv(destination.get(), source.get());
    gguf_set_val_u32(destination.get(), "general.file_type", kind == component::vae ? 1U : levo::quantization::gguf_file_type(options.profile));
    if (kind == component::vae) {
        gguf_set_val_str(destination.get(), "levo2.precision.profile", "F16");
        gguf_set_val_str(destination.get(), "levo2.precision.policy_revision", "1");
        gguf_set_val_str(destination.get(), "levo2.precision.source_artifact_sha256", source_sha256.c_str());
    } else {
        gguf_set_val_str(destination.get(), levo::quantization::profile_key, levo::quantization::name(options.profile));
        gguf_set_val_str(destination.get(), levo::quantization::policy_revision_key, levo::quantization::policy_revision);
        gguf_set_val_str(destination.get(), levo::quantization::source_artifact_sha256_key, source_sha256.c_str());
    }
    if (kind == component::flow) {
        gguf_set_val_str(destination.get(), "levo2.flow.parameter_dtype", "MIXED");
        const std::string layout = levo::quantization::flow_padded_input_layout(options.profile);
        gguf_set_val_str(destination.get(), levo::quantization::flow_padded_input_key, layout.c_str());
    }
    context_ptr tensor_context(ggml_init({plan.size() * ggml_tensor_overhead(), nullptr, true}));
    if (!tensor_context) fail("cannot allocate GGUF tensor metadata context");
    for (const tensor_plan & item : plan) {
        ggml_tensor * tensor = ggml_new_tensor(tensor_context.get(), item.target_type, item.rank, item.storage_shape.data());
        if (!tensor) fail("cannot create output tensor '" + item.name + "'");
        ggml_set_name(tensor, item.name.c_str());
        gguf_add_tensor(destination.get(), tensor);
    }

    const std::filesystem::path temporary = options.output.string() + ".partial";
    if (std::filesystem::exists(temporary)) fail("temporary output already exists: " + temporary.string());
    if (!gguf_write_to_file(destination.get(), temporary.string().c_str(), true)) fail("cannot write GGUF metadata");
    std::ifstream input(options.input, std::ios::binary);
    std::ofstream output(temporary, std::ios::binary | std::ios::app);
    if (!input || !output) fail("cannot open GGUF payload streams");
    const std::size_t data_offset = gguf_get_data_offset(source.get());
    const std::size_t alignment = gguf_get_alignment(destination.get());
    for (const tensor_plan & item : plan) write_tensor(input, output, data_offset, item, alignment);
    output.flush();
    if (!output) fail("cannot finalize quantized GGUF payload");
    output.close();
    input.close();

    std::error_code rename_error;
    std::filesystem::rename(temporary, options.output, rename_error);
    if (rename_error) fail("cannot install output GGUF: " + rename_error.message());
    const std::string artifact_sha256 = levo::token_io::file_sha256(options.output);
    const std::uintmax_t source_bytes = std::filesystem::file_size(options.input);
    const std::uintmax_t artifact_bytes = std::filesystem::file_size(options.output);
    write_text_atomically(checksum_path, artifact_sha256 + "  " + options.output.filename().string() + "\n");
    write_text_atomically(manifest_path, make_manifest(options, kind, source_sha256, artifact_sha256, source_bytes, artifact_bytes, plan));
    std::cout << "wrote " << options.output << " (" << component_name(kind) << ' '
              << (kind == component::vae ? "F16" : levo::quantization::name(options.profile)) << ", " << artifact_bytes << " bytes)\n"
              << "sha256 " << artifact_sha256 << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    try {
        run(parse(argc, argv));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
