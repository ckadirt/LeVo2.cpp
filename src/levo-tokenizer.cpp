#include "levo-tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace levo {
namespace {

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open tokenizer asset: " + path);
    std::ostringstream ss; ss << in.rdbuf(); return ss.str();
}

// A deliberately small JSON reader: tokenizer assets only need objects,
// arrays, strings, numbers, booleans and null. Keeping it local avoids making
// nlohmann/json a runtime dependency for the inference library.
struct json {
    enum class kind { object, array, string, number, boolean, null_value } type = kind::null_value;
    std::vector<std::pair<std::string, json>> object;
    std::vector<json> array;
    std::string string;
    double number = 0;
    bool boolean = false;
};
class json_reader {
public:
    explicit json_reader(const std::string & s) : s_(s) {}
    json parse() { skip(); auto v = value(); skip(); if (p_ != s_.size()) fail("trailing JSON"); return v; }
private:
    const std::string & s_; std::size_t p_ = 0;
    [[noreturn]] void fail(const char * msg) const { throw std::runtime_error(std::string("invalid tokenizer JSON: ") + msg); }
    void skip() { while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_; }
    void expect(char c) { skip(); if (p_ >= s_.size() || s_[p_++] != c) fail("unexpected character"); }
    std::string str() {
        skip(); expect('"'); std::string out;
        while (p_ < s_.size()) { char c = s_[p_++]; if (c == '"') return out; if (c != '\\') { out += c; continue; }
            if (p_ >= s_.size()) fail("unterminated escape");
            c = s_[p_++];
            switch (c) { case '"': case '\\': case '/': out += c; break; case 'b': out+='\b'; break; case 'f': out+='\f'; break; case 'n': out+='\n'; break; case 'r': out+='\r'; break; case 't': out+='\t'; break;
            case 'u': { if (p_ + 4 > s_.size()) fail("short unicode escape"); unsigned v=0; for(int i=0;i<4;++i){ char h=s_[p_++]; unsigned digit=0; if(h>='0'&&h<='9')digit=h-'0'; else if(h>='a'&&h<='f')digit=h-'a'+10; else if(h>='A'&&h<='F')digit=h-'A'+10; else fail("bad unicode escape"); v=(v<<4)+digit; } if(v<0x80) out+=char(v); else if(v<0x800){out+=char(0xc0|(v>>6));out+=char(0x80|(v&63));} else {out+=char(0xe0|(v>>12));out+=char(0x80|((v>>6)&63));out+=char(0x80|(v&63));} break; }
            default: fail("unknown escape"); }
        } fail("unterminated string");
    }
    json value() {
        skip(); if (p_ >= s_.size()) fail("missing value");
        if (s_[p_] == '{') { ++p_; json x; x.type=json::kind::object; skip(); if(p_<s_.size()&&s_[p_]=='}'){++p_;return x;} while(true){ auto k=str(); expect(':'); x.object.emplace_back(std::move(k),value()); skip(); if(p_<s_.size()&&s_[p_]=='}'){++p_;return x;} expect(','); } }
        if (s_[p_] == '[') { ++p_; json x; x.type=json::kind::array; skip(); if(p_<s_.size()&&s_[p_]==']'){++p_;return x;} while(true){x.array.push_back(value());skip();if(p_<s_.size()&&s_[p_]==']'){++p_;return x;}expect(',');} }
        if (s_[p_] == '"') { json x; x.type=json::kind::string; x.string=str(); return x; }
        if (s_.compare(p_,4,"true")==0) {p_+=4;json x;x.type=json::kind::boolean;x.boolean=true;return x;}
        if (s_.compare(p_,5,"false")==0) {p_+=5;json x;x.type=json::kind::boolean;return x;}
        if (s_.compare(p_,4,"null")==0) {p_+=4;json x;return x;}
        char * end=nullptr; const char * start=s_.c_str()+p_; double n=std::strtod(start,&end); if(end==start) fail("bad number"); p_=static_cast<std::size_t>(end-s_.c_str()); json x;x.type=json::kind::number;x.number=n;return x;
    }
};
const json * get(const json & x, const char * key) { for (const auto & kv : x.object) if (kv.first == key) return &kv.second; return nullptr; }

std::string bytes_to_unicode(const std::string & utf8) {
    // GPT-2's alphabet operates on UTF-8 *bytes*, not Unicode code points.
    // Every byte is mapped to a printable Unicode code point.
    std::string out;
    for (unsigned char b : utf8) {
        unsigned mapped = 0;
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) mapped = b;
        else { unsigned rank = 0; for (unsigned x = 0; x < b; ++x) if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255))) ++rank; mapped = 256 + rank; }
        if (mapped < 0x80) out += static_cast<char>(mapped);
        else if (mapped < 0x800) { out += static_cast<char>(0xc0 | (mapped >> 6)); out += static_cast<char>(0x80 | (mapped & 63)); }
        else { out += static_cast<char>(0xe0 | (mapped >> 12)); out += static_cast<char>(0x80 | ((mapped >> 6) & 63)); out += static_cast<char>(0x80 | (mapped & 63)); }
    }
    return out;
}

std::vector<std::string> split_utf8_bytes(const std::string & text) {
    std::vector<std::string> out; for(std::size_t i=0;i<text.size();){unsigned char c=text[i];std::size_t n=(c<0x80?1:c<0xe0?2:c<0xf0?3:4);if(i+n>text.size())n=1;out.push_back(text.substr(i,n));i+=n;} return out;
}

} // namespace

ByteLevelBPETokenizer ByteLevelBPETokenizer::load(const std::string & vocab_json,
                                                  const std::string & merges_txt,
                                                  const std::string & tokenizer_config_json) {
    ByteLevelBPETokenizer t; t.load_vocab_object(read_file(vocab_json)); t.load_merges_text(read_file(merges_txt));
    if (!tokenizer_config_json.empty()) {
        const json root=json_reader(read_file(tokenizer_config_json)).parse();
        if (const json * a=get(root,"added_tokens_decoder"); a && a->type==json::kind::object) for(const auto & kv:a->object) if(const json * c=get(kv.second,"content"); c&&c->type==json::kind::string) { const auto id=std::stoll(kv.first); t.add_special_token(c->string,id); }
    }
    return t;
}

ByteLevelBPETokenizer ByteLevelBPETokenizer::load_tokenizer_json(const std::string & tokenizer_json,
                                                                 const std::string & tokenizer_config_json) {
    const json root=json_reader(read_file(tokenizer_json)).parse(); const json * model=get(root,"model"); if(!model) throw std::runtime_error("tokenizer.json has no model");
    const json * vocab=get(*model,"vocab"), * merges=get(*model,"merges"); if(!vocab||!merges||vocab->type!=json::kind::object||merges->type!=json::kind::array) throw std::runtime_error("tokenizer.json model lacks vocab/merges");
    ByteLevelBPETokenizer t; for(const auto & kv:vocab->object){if(kv.second.type!=json::kind::number)throw std::runtime_error("invalid tokenizer vocab ID"); int64_t id=static_cast<int64_t>(kv.second.number); t.token_to_id_[kv.first]=id;t.ensure_id(static_cast<std::size_t>(id));t.id_to_token_[static_cast<std::size_t>(id)]=kv.first;}
    std::size_t rank=0; for(const auto & x:merges->array){if(x.type==json::kind::string){std::istringstream ss(x.string);std::string a,b;ss>>a>>b;if(!a.empty()&&!b.empty())t.merge_rank_[a+"\n"+b]=rank++;} else if(x.type==json::kind::array&&x.array.size()==2)t.merge_rank_[x.array[0].string+"\n"+x.array[1].string]=rank++;}
    const json * added=get(root,"added_tokens"); if(added&&added->type==json::kind::array)for(const auto & x:added->array){const json * id=get(x,"id"),*c=get(x,"content");if(id&&c&&id->type==json::kind::number&&c->type==json::kind::string)t.add_special_token(c->string,static_cast<int64_t>(id->number));}
    if(!tokenizer_config_json.empty()){const json cfg=json_reader(read_file(tokenizer_config_json)).parse();if(const json * a=get(cfg,"added_tokens_decoder");a&&a->type==json::kind::object)for(const auto&kv:a->object)if(const json*c=get(kv.second,"content");c&&c->type==json::kind::string)t.add_special_token(c->string,std::stoll(kv.first));}
    return t;
}

ByteLevelBPETokenizer ByteLevelBPETokenizer::load_embedded(
    const std::vector<std::string> & tokens,
    const std::vector<std::string> & merges,
    const std::string & added_tokens_json,
    const std::string & tokenizer_config_json) {
    if (tokens.empty() || merges.empty()) throw std::runtime_error("embedded tokenizer is empty");
    ByteLevelBPETokenizer t;
    t.id_to_token_ = tokens;
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        if (tokens[id].empty() || !t.token_to_id_.emplace(tokens[id], static_cast<int64_t>(id)).second)
            throw std::runtime_error("embedded tokenizer has an empty or duplicate token");
    }
    std::size_t rank = 0;
    for (const std::string & merge : merges) {
        std::istringstream stream(merge);
        std::string first, second;
        if (!(stream >> first >> second)) continue;
        t.merge_rank_[first + "\n" + second] = rank++;
    }
    if (!added_tokens_json.empty()) {
        const json added = json_reader(added_tokens_json).parse();
        if (added.type != json::kind::array) throw std::runtime_error("embedded added tokens must be a JSON array");
        for (const json & item : added.array) {
            const json * id = get(item, "id");
            const json * content = get(item, "content");
            const json * special = get(item, "special");
            if (id && content && id->type == json::kind::number && content->type == json::kind::string &&
                (!special || (special->type == json::kind::boolean && special->boolean))) {
                t.add_special_token(content->string, static_cast<int64_t>(id->number));
            }
        }
    }
    if (!tokenizer_config_json.empty()) {
        const json cfg = json_reader(tokenizer_config_json).parse();
        if (const json * added = get(cfg, "added_tokens_decoder"); added && added->type == json::kind::object) {
            for (const auto & kv : added->object) {
                if (const json * content = get(kv.second, "content"); content && content->type == json::kind::string)
                    t.add_special_token(content->string, std::stoll(kv.first));
            }
        }
    }
    return t;
}

void ByteLevelBPETokenizer::ensure_id(std::size_t id) { if(id>=id_to_token_.size())id_to_token_.resize(id+1); }
void ByteLevelBPETokenizer::load_vocab_object(const std::string & text) { const json root=json_reader(text).parse(); if(root.type!=json::kind::object)throw std::runtime_error("vocab JSON must be an object");for(const auto&kv:root.object){if(kv.second.type!=json::kind::number)throw std::runtime_error("invalid vocab ID");auto id=static_cast<int64_t>(kv.second.number);if(id<0)throw std::runtime_error("negative vocab ID");token_to_id_[kv.first]=id;ensure_id(static_cast<std::size_t>(id));id_to_token_[static_cast<std::size_t>(id)]=kv.first;} }
void ByteLevelBPETokenizer::load_merges_text(const std::string & text) { std::istringstream ss(text);std::string line;std::size_t rank=0;while(std::getline(ss,line)){if(line.empty()||line[0]=='#')continue;std::istringstream ls(line);std::string a,b;if(ls>>a>>b)merge_rank_[a+"\n"+b]=rank++;} }
void ByteLevelBPETokenizer::add_special_token(const std::string & token, int64_t id) { if(token.empty())throw std::invalid_argument("special token cannot be empty");if(id<0){id=static_cast<int64_t>(id_to_token_.size());while(id>=0&&static_cast<std::size_t>(id)<id_to_token_.size()&&!id_to_token_[static_cast<std::size_t>(id)].empty())++id;}if(id<0)throw std::invalid_argument("invalid special token ID");ensure_id(static_cast<std::size_t>(id));if(!id_to_token_[static_cast<std::size_t>(id)].empty()&&id_to_token_[static_cast<std::size_t>(id)]!=token)throw std::invalid_argument("special token ID already occupied");token_to_id_[token]=id;id_to_token_[static_cast<std::size_t>(id)]=token;special_tokens_[token]=id;}
int64_t ByteLevelBPETokenizer::token_id(const std::string & token) const {auto i=token_to_id_.find(token);if(i==token_to_id_.end())throw std::out_of_range("token is absent from vocabulary: "+token);return i->second;}
const std::string & ByteLevelBPETokenizer::token_string(int64_t id) const {if(id<0||static_cast<std::size_t>(id)>=id_to_token_.size()||id_to_token_[static_cast<std::size_t>(id)].empty())throw std::out_of_range("token ID is absent from vocabulary");return id_to_token_[static_cast<std::size_t>(id)];}

std::vector<std::string> ByteLevelBPETokenizer::bpe(const std::string & piece) const {auto cached=bpe_cache_.find(piece);if(cached!=bpe_cache_.end())return cached->second;auto symbols=split_utf8_bytes(piece);if(symbols.empty())return {};while(symbols.size()>1){std::size_t best=std::numeric_limits<std::size_t>::max(),at=symbols.size();for(std::size_t i=0;i+1<symbols.size();++i){auto it=merge_rank_.find(symbols[i]+"\n"+symbols[i+1]);if(it!=merge_rank_.end()&&it->second<best){best=it->second;at=i;}}if(at==symbols.size())break;symbols[at]+=symbols[at+1];symbols.erase(symbols.begin()+static_cast<std::ptrdiff_t>(at+1));}bpe_cache_[piece]=symbols;return symbols;}

std::vector<int64_t> ByteLevelBPETokenizer::encode(const std::string & text) const {
    std::vector<int64_t> out; std::size_t pos=0; while(pos<text.size()) {std::string special;int64_t sid=0;for(const auto&kv:special_tokens_)if(text.compare(pos,kv.first.size(),kv.first)==0&&(special.empty()||kv.first.size()>special.size())){special=kv.first;sid=kv.second;}if(!special.empty()){out.push_back(sid);pos+=special.size();continue;}
        std::size_t end=pos+1; while(end<text.size()){bool hit=false;for(const auto&kv:special_tokens_)if(text.compare(end,kv.first.size(),kv.first)==0){hit=true;break;}if(hit)break;++end;}std::string chunk=text.substr(pos,end-pos);
        // Byte-level GPT2 pre-tokenization. Leading whitespace belongs to the
        // following run; this covers Qwen's common lyric/text cases while
        // retaining every byte losslessly.
        std::size_t i=0; while(i<chunk.size()) {
            std::size_t j=i; const bool initial_ws=std::isspace(static_cast<unsigned char>(chunk[i]));
            if (initial_ws) {
                while(j<chunk.size()&&std::isspace(static_cast<unsigned char>(chunk[j]))) ++j;
                // Qwen/GPT-2's regex permits only one leading space on a
                // word; preceding spaces are emitted as whitespace pieces.
                if (j < chunk.size() && j - i > 1) j = i + 1;
            }
            if (j<chunk.size()) {
                const unsigned char first=static_cast<unsigned char>(chunk[j]);
                const int category=(first>=128||std::isalpha(first))?1:std::isdigit(first)?2:0;
                if (category == 2) ++j; // Qwen's \p{N} alternative is one digit.
                while(j<chunk.size()&&!std::isspace(static_cast<unsigned char>(chunk[j]))) {
                    if (category == 2) break;
                    const unsigned char c=static_cast<unsigned char>(chunk[j]);
                    const int this_category=(c>=128||std::isalpha(c))?1:std::isdigit(c)?2:0;
                    if (this_category!=category) break;
                    ++j;
                }
                // The GPT-2/Qwen regex keeps common English contractions as
                // one pre-token ("it" + "'s" is two BPE pieces, not three).
                if (category == 1 && j < chunk.size() && chunk[j] == '\'' ) {
                    const char * suffixes[] = {"'s","'t","'r","'m","'d","'ll","'ve","'re"};
                    for (const char * suffix : suffixes) {
                        const std::size_t n=std::char_traits<char>::length(suffix);
                        if (chunk.compare(j,n,suffix)==0) { j += n; break; }
                    }
                }
                if (category == 0) {
                    const std::size_t punct_start = initial_ws ? i + 1 : i;
                    if (punct_start < chunk.size() && chunk[punct_start] == '\'' && punct_start + 1 < chunk.size() &&
                        (std::isalpha(static_cast<unsigned char>(chunk[punct_start + 1])) || static_cast<unsigned char>(chunk[punct_start + 1]) >= 128)) {
                        j = punct_start + 1;
                        while (j < chunk.size() && (std::isalpha(static_cast<unsigned char>(chunk[j])) || static_cast<unsigned char>(chunk[j]) >= 128)) ++j;
                    }
                    while (j < chunk.size() && (chunk[j] == '\r' || chunk[j] == '\n')) ++j;
                }
            }
            if(j==i) ++j;
            std::string mapped=bytes_to_unicode(chunk.substr(i,j-i));
            for(const auto&s:bpe(mapped)){auto it=token_to_id_.find(s);if(it==token_to_id_.end()){std::ostringstream err;err<<"tokenizer vocabulary cannot encode BPE piece (bytes";for(unsigned char c:s)err<<" "<<std::hex<<static_cast<unsigned>(c);err<<")";throw std::runtime_error(err.str());}out.push_back(it->second);} i=j;
        }
        pos=end;
    } return out;
}

std::string ByteLevelBPETokenizer::decode(const std::vector<int64_t> & ids, bool skip_special_tokens) const {std::string mapped;for(auto id:ids){const auto&t=token_string(id);if(skip_special_tokens&&special_tokens_.find(t)!=special_tokens_.end())continue;mapped+=t;}std::string out;for(std::size_t i=0;i<mapped.size();){unsigned char c=mapped[i];std::size_t n=(c<0x80?1:c<0xe0?2:3);unsigned cp=0;if(n==1)cp=c;else if(n==2){cp=((c&31)<<6)|(mapped[i+1]&63);}else{cp=((c&15)<<12)|((mapped[i+1]&63)<<6)|(mapped[i+2]&63);}unsigned b=0;if(cp<256)b=cp;else {unsigned rank=cp-256;for(unsigned x=0;x<=255;++x){if(!((x>=33&&x<=126)||(x>=161&&x<=172)||(x>=174&&x<=255))){if(rank==0){b=x;break;}--rank;}}}if(b<=255)out+=static_cast<char>(b);else out+=mapped.substr(i,n);i+=n;}return out;}

} // namespace levo
