#include "levo-engine-request.h"

#include "levo-token-io.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace levo::engine_request {
namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LeVo engine request: " + message);
}

constexpr std::size_t k_max_request_bytes = 1024U * 1024U;

bool valid_utf8(const std::string & source);

struct value {
    enum class kind { null_value, boolean, number, string, array, object } type = kind::null_value;
    bool boolean = false;
    double number = 0.0;
    std::string number_text;
    std::string string;
    std::vector<value> array;
    std::map<std::string, value> object;
};

class reader final {
public:
    explicit reader(const std::string & input) : input_(input) {}

    value parse() {
        skip();
        value result = parse_value();
        skip();
        if (position_ != input_.size()) fail("trailing bytes after JSON value");
        return result;
    }

private:
    void skip() {
        while (position_ != input_.size() && (input_[position_] == ' ' || input_[position_] == '\n' ||
                                               input_[position_] == '\r' || input_[position_] == '\t')) {
            ++position_;
        }
    }

    char take() {
        if (position_ == input_.size()) fail("unexpected end of JSON");
        return input_[position_++];
    }

    void expect(char expected) {
        skip();
        if (take() != expected) fail(std::string("expected '") + expected + "'");
    }

    static unsigned hex(char c) {
        if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
        fail("invalid hexadecimal escape");
    }

    static void append_utf8(std::string & out, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) out.push_back(static_cast<char>(codepoint));
        else if (codepoint <= 0x7ffU) {
            out.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0xffffU) {
            out.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0x10ffffU) {
            out.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
            out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
            out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else fail("Unicode escape is outside range");
    }

    std::uint32_t unicode_escape() {
        std::uint32_t result = 0;
        for (unsigned index = 0; index != 4; ++index) result = (result << 4U) | hex(take());
        return result;
    }

    std::string parse_string() {
        if (take() != '"') fail("expected string");
        std::string result;
        while (true) {
            const unsigned char current = static_cast<unsigned char>(take());
            if (current == '"') {
                if (!valid_utf8(result)) fail("invalid UTF-8 in JSON string");
                return result;
            }
            if (current < 0x20U) fail("control byte in JSON string");
            if (current != '\\') {
                result.push_back(static_cast<char>(current));
                continue;
            }
            switch (take()) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    std::uint32_t codepoint = unicode_escape();
                    if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                        if (take() != '\\' || take() != 'u') fail("unpaired high surrogate");
                        const std::uint32_t low = unicode_escape();
                        if (low < 0xdc00U || low > 0xdfffU) fail("invalid low surrogate");
                        codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) + (low - 0xdc00U);
                    } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
                        fail("unpaired low surrogate");
                    }
                    append_utf8(result, codepoint);
                    break;
                }
                default: fail("invalid JSON string escape");
            }
        }
    }

    value parse_object() {
        value result;
        result.type = value::kind::object;
        expect('{');
        skip();
        if (position_ != input_.size() && input_[position_] == '}') {
            ++position_;
            return result;
        }
        while (true) {
            skip();
            const std::string key = parse_string();
            expect(':');
            auto inserted = result.object.emplace(key, parse_value());
            if (!inserted.second) fail("duplicate JSON object key '" + key + "'");
            skip();
            const char separator = take();
            if (separator == '}') return result;
            if (separator != ',') fail("expected ',' or '}' in JSON object");
        }
    }

    value parse_array() {
        value result;
        result.type = value::kind::array;
        expect('[');
        skip();
        if (position_ != input_.size() && input_[position_] == ']') {
            ++position_;
            return result;
        }
        while (true) {
            result.array.push_back(parse_value());
            skip();
            const char separator = take();
            if (separator == ']') return result;
            if (separator != ',') fail("expected ',' or ']' in JSON array");
        }
    }

    value parse_number() {
        const std::size_t begin_offset = position_;
        if (input_[position_] == '-') ++position_;
        if (position_ == input_.size()) fail("invalid JSON number");
        if (input_[position_] == '0') {
            ++position_;
            if (position_ != input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
                fail("leading zero in JSON number");
            }
        } else if (input_[position_] >= '1' && input_[position_] <= '9') {
            do { ++position_; } while (position_ != input_.size() && input_[position_] >= '0' && input_[position_] <= '9');
        } else {
            fail("invalid JSON number");
        }
        if (position_ != input_.size() && input_[position_] == '.') {
            ++position_;
            const std::size_t fraction_begin = position_;
            while (position_ != input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (position_ == fraction_begin) fail("JSON fraction needs a digit");
        }
        if (position_ != input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ != input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            const std::size_t exponent_begin = position_;
            while (position_ != input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (position_ == exponent_begin) fail("JSON exponent needs a digit");
        }
        const char * begin = input_.c_str() + begin_offset;
        char * end = nullptr;
        const double parsed = std::strtod(begin, &end);
        if (end != input_.c_str() + position_ || !std::isfinite(parsed)) fail("invalid or non-finite JSON number");
        value result;
        result.type = value::kind::number;
        result.number = parsed;
        result.number_text.assign(begin, static_cast<std::size_t>(end - begin));
        return result;
    }

    value parse_value() {
        skip();
        if (position_ == input_.size()) fail("unexpected end of JSON value");
        const char current = input_[position_];
        if (current == '{') return parse_object();
        if (current == '[') return parse_array();
        if (current == '"') {
            value result;
            result.type = value::kind::string;
            result.string = parse_string();
            return result;
        }
        if (input_.compare(position_, 4, "true") == 0) {
            position_ += 4;
            value result; result.type = value::kind::boolean; result.boolean = true; return result;
        }
        if (input_.compare(position_, 5, "false") == 0) {
            position_ += 5;
            value result; result.type = value::kind::boolean; return result;
        }
        if (input_.compare(position_, 4, "null") == 0) {
            position_ += 4;
            return {};
        }
        if (current == '-' || (current >= '0' && current <= '9')) return parse_number();
        fail("invalid JSON value");
    }

    const std::string & input_;
    std::size_t position_ = 0;
};

const value * lookup(const value & object, const char * key) {
    const auto found = object.object.find(key);
    return found == object.object.end() ? nullptr : &found->second;
}

const value & required(const value & object, const char * key, value::kind type) {
    const value * result = lookup(object, key);
    if (result == nullptr || result->type != type) fail(std::string("required field '") + key + "' has the wrong type");
    return *result;
}

void reject_unknown(const value & object, const std::vector<std::string> & allowed) {
    for (const auto & item : object.object) {
        bool found = false;
        for (const std::string & name : allowed) if (item.first == name) { found = true; break; }
        if (!found) fail("unknown request field '" + item.first + "'");
    }
}

std::uint64_t exact_u64(const value & source, const char * label) {
    if (source.type != value::kind::number || source.number_text.empty() ||
        source.number_text.front() == '-' || source.number_text.find_first_of(".eE") != std::string::npos) {
        fail(std::string("field '") + label + "' must be an unsigned integer");
    }
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(source.number_text.data(),
                                        source.number_text.data() + source.number_text.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != source.number_text.data() + source.number_text.size()) {
        fail(std::string("field '") + label + "' must be an unsigned integer");
    }
    return result;
}

std::size_t exact_size(const value & source, const char * label) {
    const std::uint64_t parsed = exact_u64(source, label);
    if (parsed > std::numeric_limits<std::size_t>::max()) fail(std::string("field '") + label + "' is too large");
    return static_cast<std::size_t>(parsed);
}

std::string escape(const std::string & source) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char value : source) {
        switch (value) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (value < 0x20U) out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                                       << static_cast<unsigned>(value) << std::dec << std::setfill(' ');
                else out << static_cast<char>(value);
        }
    }
    out << '"';
    return out.str();
}

bool valid_utf8(const std::string & source) {
    for (std::size_t index = 0; index < source.size();) {
        const unsigned char first = static_cast<unsigned char>(source[index++]);
        if (first < 0x80U) continue;
        unsigned continuation = 0;
        std::uint32_t codepoint = 0;
        if (first >= 0xc2U && first <= 0xdfU) { continuation = 1; codepoint = first & 0x1fU; }
        else if (first >= 0xe0U && first <= 0xefU) { continuation = 2; codepoint = first & 0x0fU; }
        else if (first >= 0xf0U && first <= 0xf4U) { continuation = 3; codepoint = first & 0x07U; }
        else return false;
        if (source.size() - index < continuation) return false;
        for (unsigned offset = 0; offset != continuation; ++offset) {
            const unsigned char current = static_cast<unsigned char>(source[index++]);
            if ((current & 0xc0U) != 0x80U) return false;
            codepoint = (codepoint << 6U) | (current & 0x3fU);
        }
        if ((continuation == 1 && codepoint < 0x80U) ||
            (continuation == 2 && codepoint < 0x800U) ||
            (continuation == 3 && codepoint < 0x10000U) ||
            (codepoint >= 0xd800U && codepoint <= 0xdfffU) || codepoint > 0x10ffffU) return false;
    }
    return true;
}

std::string number(double source) {
    if (!std::isfinite(source)) fail("cannot serialize non-finite number");
    std::ostringstream out;
    out << std::setprecision(17) << source;
    return out.str();
}

std::string sampling_json(const generation_sampling_config & sampling) {
    std::ostringstream out;
    out << "{\"ignore_tokens\":[";
    for (std::size_t index = 0; index != sampling.ignore_tokens.size(); ++index) {
        if (index) out << ',';
        out << sampling.ignore_tokens[index];
    }
    out << "],\"repetition_penalty\":" << number(sampling.repetition_penalty)
        << ",\"repetition_window\":" << sampling.repetition_window
        << ",\"temperature\":" << number(sampling.temperature)
        << ",\"top_k_detail\":" << sampling.top_k_detail
        << ",\"top_k_mixed\":" << sampling.top_k_mixed
        << ",\"use_sampling\":" << (sampling.use_sampling ? "true" : "false") << '}';
    return out.str();
}

} // namespace

request parse_request_object(const value & root) {
    if (root.type != value::kind::object) fail("request must be a JSON object");
    reject_unknown(root, {"lyrics", "description", "duration", "duration_seconds", "seed", "cfg_scale", "sampling", "flow"});
    if (lookup(root, "duration") != nullptr && lookup(root, "duration_seconds") != nullptr) {
        fail("use only one of duration and duration_seconds");
    }
    request result;
    result.lyrics = required(root, "lyrics", value::kind::string).string;
    result.description = required(root, "description", value::kind::string).string;
    if (!valid_utf8(result.lyrics) || !valid_utf8(result.description)) fail("lyrics and description must be valid UTF-8");
    const value * duration = lookup(root, "duration_seconds");
    if (duration == nullptr) duration = lookup(root, "duration");
    if (duration == nullptr || duration->type != value::kind::number || !std::isfinite(duration->number)) {
        fail("required duration_seconds must be a finite number");
    }
    result.duration_seconds = duration->number;
    if (const value * seed = lookup(root, "seed")) {
        if (seed->type != value::kind::null_value) {
            result.seed = exact_u64(*seed, "seed");
            result.seed_present = true;
        }
    }
    if (const value * cfg = lookup(root, "cfg_scale")) {
        if (cfg->type != value::kind::number || !std::isfinite(cfg->number) ||
            cfg->number < -std::numeric_limits<float>::max() || cfg->number > std::numeric_limits<float>::max()) {
            fail("cfg_scale must be a finite float");
        }
        result.cfg_scale = static_cast<float>(cfg->number);
    }
    if (const value * sampling = lookup(root, "sampling")) {
        if (sampling->type != value::kind::object) fail("sampling must be an object");
        reject_unknown(*sampling, {"use_sampling", "temperature", "top_k_mixed", "top_k_detail",
                                   "repetition_window", "repetition_penalty", "ignore_tokens"});
        if (const value * item = lookup(*sampling, "use_sampling")) {
            if (item->type != value::kind::boolean) fail("sampling.use_sampling must be boolean");
            result.sampling.use_sampling = item->boolean;
        }
        if (const value * item = lookup(*sampling, "temperature")) {
            if (item->type != value::kind::number || item->number < 0.0 || item->number > std::numeric_limits<float>::max()) fail("sampling.temperature is invalid");
            result.sampling.temperature = static_cast<float>(item->number);
        }
        if (const value * item = lookup(*sampling, "top_k_mixed")) result.sampling.top_k_mixed = exact_size(*item, "sampling.top_k_mixed");
        if (const value * item = lookup(*sampling, "top_k_detail")) result.sampling.top_k_detail = exact_size(*item, "sampling.top_k_detail");
        if (const value * item = lookup(*sampling, "repetition_window")) result.sampling.repetition_window = exact_size(*item, "sampling.repetition_window");
        if (const value * item = lookup(*sampling, "repetition_penalty")) {
            if (item->type != value::kind::number || item->number <= 0.0 || item->number > std::numeric_limits<float>::max()) fail("sampling.repetition_penalty is invalid");
            result.sampling.repetition_penalty = static_cast<float>(item->number);
        }
        if (const value * item = lookup(*sampling, "ignore_tokens")) {
            if (item->type != value::kind::array) fail("sampling.ignore_tokens must be an array");
            result.sampling.ignore_tokens.clear();
            result.sampling.ignore_tokens.reserve(item->array.size());
            for (const value & token : item->array) {
                const std::uint64_t id = exact_u64(token, "sampling.ignore_tokens[]");
                if (id > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) fail("sampling ignore token is too large");
                result.sampling.ignore_tokens.push_back(static_cast<int64_t>(id));
            }
        }
    }
    if (const value * flow = lookup(root, "flow")) {
        if (flow->type != value::kind::object) fail("flow must be an object");
        reject_unknown(*flow, {"seed", "euler_steps", "cfg_scale"});
        if (const value * item = lookup(*flow, "seed")) result.flow_seed = exact_u64(*item, "flow.seed");
        if (const value * item = lookup(*flow, "euler_steps")) {
            result.flow_euler_steps = exact_size(*item, "flow.euler_steps");
        }
        if (const value * item = lookup(*flow, "cfg_scale")) {
            if (item->type != value::kind::number || !std::isfinite(item->number) ||
                item->number < 0.0 || item->number > std::numeric_limits<float>::max()) {
                fail("flow.cfg_scale must be a finite non-negative float");
            }
            result.flow_cfg_scale = static_cast<float>(item->number);
        }
    }
    return result;
}

request parse(const std::string & json) {
    if (json.empty() || json.size() > k_max_request_bytes) {
        fail("request must be non-empty and at most 1 MiB");
    }
    return parse_request_object(reader(json).parse());
}

codes parse_codes(const std::string & json) {
    if (json.empty() || json.size() > k_max_request_bytes) {
        fail("CODES input must be non-empty and at most 1 MiB");
    }
    const value root = reader(json).parse();
    if (root.type != value::kind::object) fail("CODES input must be a JSON object");
    reject_unknown(root, {"audio_codes", "frame_count", "request", "token_sha256"});
    codes result;
    result.generation_request = parse_request_object(required(root, "request", value::kind::object));
    if (!result.generation_request.seed_present) fail("CODES request does not contain a resolved seed");
    result.frame_count = exact_size(required(root, "frame_count", value::kind::number), "frame_count");
    if (result.frame_count == 0 || result.frame_count > std::numeric_limits<std::size_t>::max() / 3U) {
        fail("CODES frame_count is invalid");
    }
    const value & encoded = required(root, "audio_codes", value::kind::array);
    if (encoded.array.size() != result.frame_count * 3U) fail("CODES audio_codes is not [3,T]");
    result.tokens.reserve(encoded.array.size());
    for (const value & token : encoded.array) {
        const std::uint64_t id = exact_u64(token, "audio_codes[]");
        if (id > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
            fail("CODES token is outside int32 range");
        }
        result.tokens.push_back(static_cast<std::int32_t>(id));
    }
    result.token_sha256 = required(root, "token_sha256", value::kind::string).string;
    if (result.token_sha256 != token_io::tensor_sha256(result.tokens)) {
        fail("CODES token_sha256 does not match audio_codes");
    }
    return result;
}

std::string serialize(const request & value) {
    if (!value.seed_present) fail("cannot serialize an unresolved request seed");
    std::ostringstream out;
    out << "{\"cfg_scale\":" << number(value.cfg_scale)
        << ",\"description\":" << escape(value.description)
        << ",\"duration_seconds\":" << number(value.duration_seconds)
        << ",\"flow\":{\"cfg_scale\":" << number(value.flow_cfg_scale)
        << ",\"euler_steps\":" << value.flow_euler_steps
        << ",\"seed\":" << value.flow_seed << '}'
        << ",\"lyrics\":" << escape(value.lyrics)
        << ",\"sampling\":" << sampling_json(value.sampling)
        << ",\"seed\":" << value.seed << '}';
    return out.str();
}

generation_config generation_config_for(const request & value, const std::string & model_path) {
    if (!value.seed_present) fail("generation requires a resolved request seed");
    generation_config result;
    result.model_path = model_path;
    result.lyrics = value.lyrics;
    result.description = value.description;
    result.duration_seconds = value.duration_seconds;
    result.seed_present = true;
    result.seed = value.seed;
    result.cfg_scale = value.cfg_scale;
    result.sampling = value.sampling;
    return result;
}

std::string serialize_codes(const request & value, const generation_result & result) {
    if (result.tokens.size() != result.frame_count * 3U) fail("completed result does not contain [3,T] codes");
    std::ostringstream out;
    std::string request_json = serialize(value);
    // Keep request first-class instead of duplicating fields. The nested object
    // will be parsed by DIFFUSE in the next implementation slice.
    out << "{\"audio_codes\":[";
    for (std::size_t index = 0; index != result.tokens.size(); ++index) {
        if (index) out << ',';
        out << result.tokens[index];
    }
    out << "],\"frame_count\":" << result.frame_count
        << ",\"request\":" << request_json
        << ",\"token_sha256\":" << escape(token_io::tensor_sha256(result.tokens)) << '}';
    return out.str();
}

} // namespace levo::engine_request
