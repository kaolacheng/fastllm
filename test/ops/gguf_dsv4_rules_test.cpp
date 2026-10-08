// Verify the deepseek4 GGUF weight-name replacement rules map every GGUF
// tensor name to the exact name fastllm's DeepSeekV4 model expects.
// This is a dry-run: it exercises GetGGUFWeightReplaceRules directly.

#include "gguf.h"
#include <cstdio>
#include <regex>
#include <vector>
#include <string>

using namespace fastllm;

static int failures = 0;
#define CHECK(cond, msg) do { if (cond) printf("  PASS: %s\n", msg); \
    else { printf("  FAIL: %s\n", msg); failures++; } } while (0)

// mimic AppendGGUFTasks replacement logic for one tensor name
static std::vector<std::string> apply_rules(const std::string &arch, const std::string &name) {
    auto rules = GetGGUFWeightReplaceRules(arch);
    std::vector<std::string> out;
    for (auto &it : rules) {
        if (std::regex_search(name, it.pattern)) {
            if (it.type == GGUFWeightReplaceRule::GGUFWeightReplacePacked) {
                std::string prefix = std::regex_replace(name, it.pattern, it.names[0]);
                std::string suffix = std::regex_replace(name, it.pattern, it.names[1]);
                // report only first expert slot
                out.push_back(prefix + "0" + suffix);
            } else {
                out.push_back(std::regex_replace(name, it.pattern, it.names[0]));
            }
            break;
        }
    }
    return out;
}

int main() {
    auto rules = GetGGUFWeightReplaceRules("deepseek4");
    printf("deepseek4 rule count: %zu\n", rules.size());

    struct { const char *gguf; const char *expect; } cases[] = {
        {"blk.3.attn_norm.weight",            "layers.3.attn_norm.weight"},
        {"blk.3.attn_q_a.weight",             "layers.3.attn.wq_a.weight"},
        {"blk.3.attn_q_b.weight",             "layers.3.attn.wq_b.weight"},
        {"blk.3.attn_q_a_norm.weight",        "layers.3.attn.q_norm.weight"},
        {"blk.3.attn_kv.weight",              "layers.3.attn.wkv.weight"},
        {"blk.3.attn_kv_a_norm.weight",       "layers.3.attn.kv_norm.weight"},
        {"blk.3.attn_output_a.weight",        "layers.3.attn.wo_a.weight"},
        {"blk.3.attn_output_b.weight",        "layers.3.attn.wo_b.weight"},
        {"blk.3.attn_sinks.weight",           "layers.3.attn.attn_sink"},
        {"blk.3.attn_compressor_ape.weight",  "layers.3.attn.compressor.ape"},
        {"blk.3.attn_compressor_gate.weight", "layers.3.attn.compressor.wgate.weight"},
        {"blk.3.attn_compressor_kv.weight",   "layers.3.attn.compressor.wkv.weight"},
        {"blk.3.attn_compressor_norm.weight", "layers.3.attn.compressor.norm.weight"},
        {"blk.3.indexer.attn_q_b.weight",     "layers.3.attn.indexer.wq_b.weight"},
        {"blk.3.indexer.proj.weight",         "layers.3.attn.indexer.weights_proj.weight"},
        {"blk.3.indexer_compressor_ape.weight",   "layers.3.attn.indexer.compressor.ape"},
        {"blk.3.indexer_compressor_gate.weight",  "layers.3.attn.indexer.compressor.wgate.weight"},
        {"blk.3.indexer_compressor_kv.weight",    "layers.3.attn.indexer.compressor.wkv.weight"},
        {"blk.3.indexer_compressor_norm.weight",  "layers.3.attn.indexer.compressor.norm.weight"},
        {"blk.3.ffn_norm.weight",             "layers.3.ffn_norm.weight"},
        {"blk.3.ffn_gate_inp.weight",         "layers.3.ffn.gate.weight"},
        {"blk.3.exp_probs_b.bias",            "layers.3.ffn.gate.bias"},
        {"blk.3.ffn_gate_tid2eid.weight",     "layers.3.ffn.gate.tid2eid"},
        {"blk.3.ffn_gate_exps.weight",        "layers.3.ffn.experts.0.w1.weight"},
        {"blk.3.ffn_up_exps.weight",          "layers.3.ffn.experts.0.w3.weight"},
        {"blk.3.ffn_down_exps.weight",        "layers.3.ffn.experts.0.w2.weight"},
        {"blk.3.ffn_gate_shexp.weight",       "layers.3.ffn.shared_experts.w1.weight"},
        {"blk.3.ffn_up_shexp.weight",         "layers.3.ffn.shared_experts.w3.weight"},
        {"blk.3.ffn_down_shexp.weight",       "layers.3.ffn.shared_experts.w2.weight"},
        {"blk.3.hc_attn_base.weight",         "layers.3.hc_attn_base"},
        {"blk.3.hc_attn_fn.weight",           "layers.3.hc_attn_fn"},
        {"blk.3.hc_attn_scale.weight",        "layers.3.hc_attn_scale"},
        {"blk.3.hc_ffn_base.weight",          "layers.3.hc_ffn_base"},
        {"token_embd.weight",                 "embed.weight"},
        {"output.weight",                     "head.weight"},
        {"output_hc_base.weight",             "hc_head_base"},
        {"output_hc_fn.weight",               "hc_head_fn"},
        {"output_hc_scale.weight",            "hc_head_scale"},
        {"output_norm.weight",                "norm.weight"},
    };
    int n = sizeof(cases)/sizeof(cases[0]);
    for (int i = 0; i < n; i++) {
        auto got = apply_rules("deepseek4", cases[i].gguf);
        std::string g = got.empty() ? "(NO MATCH)" : got[0];
        printf("  %-40s -> %-45s ", cases[i].gguf, g.c_str());
        if (!got.empty() && got[0] == cases[i].expect) {
            printf("PASS\n");
        } else {
            printf("FAIL (expected %s)\n", cases[i].expect);
            failures++;
        }
    }

    if (failures == 0) { printf("\nALL RULES PASSED\n"); return 0; }
    printf("\n%d FAILED\n", failures); return 1;
}
