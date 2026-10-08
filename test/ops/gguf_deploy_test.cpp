// Deployment smoke test: load the DeepSeek-V4 IQ1_S GGUF with a device map
// that splits experts between GPU and disk, then run a short inference.
// This validates: 1) the load chain (metadata + name mapping + device dispatch),
// 2) MXFP4 handling if present, 3) a single forward pass.

#include "model.h"
#include "fastllm.h"
#include "gguf.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace fastllm;
#include <map>
#include <string>

int main(int argc, char **argv) {
    setbuf(stdout, nullptr);  // disable buffering so progress is visible in real time
    if (argc < 2) {
        printf("usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 2;
    }
    std::string path = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "Hello, who are you?";

    // DEV: which path to exercise. Default = all CPU + experts on disk (pure
    // CPU validation, no GPU involvement). MOE_GPU=1 => attention on CUDA,
    // last layer experts on CUDA. MOE_GPU=2 => only attention on CUDA,
    // experts all on disk.
    const char *moeGpu = getenv("MOE_GPU");
    if (moeGpu && moeGpu[0] == '1') {
        fastllm::SetDeviceMap({{"cuda:0", 1}});
        fastllm::SetMoeDeviceMap({{"disk:0", 1}});
        fastllm::SetMoeDeviceLayers(1);
        fastllm::SetLayeredMoeDeviceMap({{"cuda:0", 1}});
    } else if (moeGpu && moeGpu[0] == '2') {
        fastllm::SetDeviceMap({{"cuda:0", 1}});
        fastllm::SetMoeDeviceMap({{"disk:0", 1}});
    } else {
        fastllm::SetDeviceMap({{"cpu:0", 1}});
        fastllm::SetMoeDeviceMap({{"disk:0", 1}});
    }

    printf("Loading %s ...\n", path.c_str());
    try {
        auto model = fastllm::CreateLLMModelFromGGUFFile(path, "");
        if (model == nullptr) {
            printf("LOAD FAILED\n");
            return 1;
        }
        printf("LOADED OK. model_type=%s block_cnt=%d\n",
               model->model_type.c_str(), model->block_cnt);
        if (getenv("DIAG")) {
            printf("DIAG scoring_func=%s\n", model->weight.dicts["scoring_func"].c_str());
            printf("DIAG compress_ratios dict: [%s]\n",
                   model->weight.dicts["compress_ratios"].c_str());
            printf("DIAG index_n_heads=%s index_topk=%s compress_rope_freq_base=%s\n",
                   model->weight.dicts["index_n_heads"].c_str(),
                   model->weight.dicts["index_topk"].c_str(),
                   model->weight.dicts["compress_rope_freq_base"].c_str());
            auto &w = model->weight.weight;
            printf("DIAG total weight keys: %zu\n", w.size());
            const char *checkKeys[] = {
                "embed.weight", "norm.weight", "head.weight",
                "layers.0.attn.attn_sink", "layers.0.attn.compressor.ape",
                "layers.0.attn.indexer.wq_b.weight",
                "layers.0.ffn.gate.weight", "layers.0.ffn.gate.tid2eid",
                "layers.26.ffn.experts.0.w2.weight",
                "layers.42.ffn.experts.0.w2.weight",
                "layers.0.attn.wo_a.weight", "layers.0.attn.wo_b.weight",
            };
            for (auto *k : checkKeys) {
                auto it = w.find(k);
                if (it == w.end()) { printf("DIAG MISSING: %s\n", k); }
                else {
                    const auto &d = it->second;
                    printf("DIAG %s dims=", k);
                    for (int v : d.dims) printf("%d,", v);
                    printf(" type=%d\n", (int)d.dataType);
                }
            }
            auto embedIt = w.find("embed.weight");
            if (embedIt != w.end()) {
                const auto &e = embedIt->second;
                printf("DIAG embed dims=%d,%d dataType=%d cpuData=%p\n",
                       e.dims.size() > 0 ? e.dims[0] : -1,
                       e.dims.size() > 1 ? e.dims[1] : -1,
                       (int)e.dataType, (void*)e.cpuData);
                if (e.dataType == fastllm::DataType::FLOAT32 && e.cpuData != nullptr) {
                    const float *p = (const float*)e.cpuData;
                    printf("DIAG embed[0:5] =");
                    for (int i = 0; i < 5; i++) printf(" %.4f", p[i]);
                    printf("\n");
                }
            }
            auto headIt = w.find("head.weight");
            if (headIt != w.end()) {
                const auto &h = headIt->second;
                printf("DIAG head dims=%d,%d dataType=%d ggmlType=%d\n",
                       h.dims.size() > 0 ? h.dims[0] : -1,
                       h.dims.size() > 1 ? h.dims[1] : -1,
                       (int)h.dataType, h.ggmlType);
            }
            auto sinkIt = w.find("layers.0.attn.attn_sink");
            if (sinkIt != w.end()) {
                const auto &s = sinkIt->second;
                printf("DIAG attn_sink dims=%d,%d type=%d\n",
                       s.dims.size() > 0 ? s.dims[0] : -1,
                       s.dims.size() > 1 ? s.dims[1] : -1,
                       (int)s.dataType);
                if (s.dataType == fastllm::DataType::FLOAT32 && s.cpuData) {
                    const float *p = (const float*)s.cpuData;
                    printf("DIAG attn_sink[0:8] =");
                    for (int i = 0; i < 8; i++) printf(" %.4f", p[i]);
                    printf(" min/max: %.4f/%.4f\n",
                           *std::min_element(p, p+64), *std::max_element(p, p+64));
                }
            }
            auto woAIt = w.find("layers.0.attn.wo_a.weight");
            if (woAIt != w.end()) {
                const auto &wa = woAIt->second;
                printf("DIAG wo_a type=%d\n", (int)wa.dataType);
            }
            auto gateIt = w.find("layers.0.ffn.gate.weight");
            if (gateIt != w.end()) {
                const auto &g = gateIt->second;
                if (g.dataType == fastllm::DataType::BFLOAT16 && g.cpuData) {
                    const uint16_t *p = (const uint16_t*)g.cpuData;
                    printf("DIAG gate.weight BF16[0:4] raw=%04x %04x %04x %04x\n", p[0], p[1], p[2], p[3]);
                }
            }
            for (const char *k : {"layers.0.ffn.experts.0.gateup.weight",
                                 "layers.0.ffn.experts.0.w1.weight",
                                 "layers.0.ffn.experts.0.w3.weight",
                                 "layers.0.ffn.experts.1.gateup.weight"}) {
                auto it = w.find(k);
                if (it == w.end()) {
                    printf("DIAG MISSING: %s\n", k);
                } else {
                    const auto &d = it->second;
                    printf("DIAG %s dims=%d,%d type=%d ggmlType=%d ggmlTensor=%p IsRepacked=%d\n", k,
                           d.dims.size() > 0 ? d.dims[0] : -1,
                           d.dims.size() > 1 ? d.dims[1] : -1,
                           (int)d.dataType, d.ggmlType,
                           (void*)d.ggmlTensor, d.IsRepacked ? 1 : 0);
                    if (d.ggmlTensor != nullptr) {
                        ggml_tensor *gt = (ggml_tensor*)d.ggmlTensor;
                        printf("DIAG   tensor ne=%lld,%lld,%lld,%lld type=%s\n",
                               (long long)gt->ne[0], (long long)gt->ne[1],
                               (long long)gt->ne[2], (long long)gt->ne[3],
                               ggml_type_name(gt->type));
                    }
                }
            }
            // Dequant the merged gateup expert0 weight and print leading values.
            for (const char *lkey : {"layers.4.ffn.experts.0.gateup.weight",
                                     "layers.5.ffn.experts.0.gateup.weight"}) {
                auto gIt = w.find(lkey);
                if (gIt != w.end()) {
                    const auto &d = gIt->second;
                    if (d.dataType == fastllm::DataType::DATA_GGUF_FORMAT &&
                        d.cpuData != nullptr && !d.dims.empty()) {
                        int m = d.dims[0];  // k (input dim), 4096 for gateup
                        int n = d.dims[1];  // output dim, 4096
                        ggml_type t = (ggml_type)d.ggmlType;
                        printf("DIAG %s ggmlType=%s m=%d n=%d\n",
                               lkey, ggml_type_name(t), m, n);
                        auto toFloat = ggml_type_to_float(t);
                        if (toFloat != nullptr) {
                            std::vector<float> fv((uint64_t)m * n);
                            toFloat((const uint8_t*)d.cpuData, fv.data(), (uint64_t)m * n);
                            printf("DIAG   dequant[0:8] =");
                            for (int i = 0; i < 8; i++) printf(" %.6f", fv[i]);
                            printf("\n");
                            printf("DIAG   dequant[row0 mid] =");
                            for (int i = m/2; i < m/2+4; i++) printf(" %.6f", fv[i]);
                            printf("\n");
                            printf("DIAG   dequant[row2048:0] =");
                            size_t row = 2048;
                            for (int i = row*m; i < row*m+8; i++) printf(" %.6f", fv[i]);
                            printf("\n");
                        } else {
                            printf("DIAG %s no to_float for %s\n", lkey,
                                   ggml_type_name(t));
                        }
                    }
                }
            }
        }

        if (getenv("DIAG")) {
            // Manual single Forward: encode -> FillLLMInputs -> Forward -> dump logits
            Data inputIds, attentionMask, positionIds;
            std::vector<std::vector<float> > inputTokens;
            inputTokens.resize(1);
            Data tok = model->weight.tokenizer.Encode(prompt);
            printf("DIAG encode tokens:");
            for (int i = 0; i < tok.Count(0); i++) printf(" %d", (int)((float*)tok.cpuData)[i]);
            printf("\n");
            for (int i = 0; i < tok.Count(0); i++) inputTokens[0].push_back(((float*)tok.cpuData)[i]);
            model->FillLLMInputs(inputTokens, {{"promptLen", (int)inputTokens[0].size()}, {"index", 0}, {"add_special_tokens", 1}},
                                 inputIds, attentionMask, positionIds);
            ToDataType(attentionMask, model->dataType);
            std::vector<std::pair<Data, Data> > pastKeyValues;
            for (int i = 0; i < model->block_cnt; i++) {
                pastKeyValues.push_back(std::make_pair(Data(model->kvCacheDataType), Data(model->kvCacheDataType)));
                pastKeyValues.back().first.SetKVCache();
                pastKeyValues.back().second.SetKVCache();
            }
            if (getenv("NOCOMPRESS")) {
                // Override compress_ratios to all-zero -> pure window attention.
                std::string zeroJson;
                for (int i = 0; i < model->block_cnt; i++) {
                    zeroJson += std::string(i ? "," : "[") + "0";
                }
                zeroJson += "]";
                model->weight.dicts["compress_ratios"] = zeroJson;
                printf("DIAG [NOCOMPRESS] forced compress_ratios all zero\n");
            }
            std::vector<float> logits;
            fastllm::GenerationConfig cfg;
            cfg.output_logits = true;
            fastllm::LastTokensManager lastTokens(1, 0);
            int ret = model->Forward(inputIds, attentionMask, positionIds, pastKeyValues, cfg, lastTokens, &logits);
            printf("DIAG Forward returned token=%d, logits size=%zu\n", ret, logits.size());
            if (!logits.empty()) {
                float mx = -1e30f, mn = 1e30f, mean = 0, var = 0;
                int nanCount = 0, infCount = 0;
                for (float v : logits) {
                    if (std::isnan(v)) nanCount++;
                    if (std::isinf(v)) infCount++;
                    mx = std::max(mx, v); mn = std::min(mn, v); mean += v;
                }
                mean /= (float)logits.size();
                for (float v : logits) var += (v - mean) * (v - mean);
                var = sqrt(var / (float)logits.size());
                printf("DIAG logits: min=%.3f max=%.3f mean=%.3f std=%.3f nan=%d inf=%d\n",
                       mn, mx, mean, var, nanCount, infCount);
                std::vector<std::pair<float,int>> top;
                for (int i = 0; i < (int)logits.size(); i++) top.push_back({logits[i], i});
                std::partial_sort(top.begin(), top.begin()+5, top.end(),
                                  [](auto &a, auto &b){ return a.first > b.first; });
                printf("DIAG top5:");
                for (int i = 0; i < 5; i++) printf(" [%d]=%.3f", top[i].second, top[i].first);
                printf("\n");
            }
            // Dump raw first block bytes of gateup for layers 4 and 5 for
            // python cross-check (IQ2_XXS block = 66 bytes).
            for (const char *lkey : {"layers.4.ffn.experts.0.gateup.weight",
                                     "layers.5.ffn.experts.0.gateup.weight"}) {
                auto gIt = model->weight.weight.find(lkey);
                if (gIt != model->weight.weight.end()) {
                    const auto &d = gIt->second;
                    printf("DIAG %s type=%d ggmlType=%d\n", lkey,
                           (int)d.dataType, d.ggmlType);
                    if (d.cpuData != nullptr) {
                        const uint8_t *p = (const uint8_t*)d.cpuData;
                        printf("DIAG   raw[0:16] =");
                        for (int i = 0; i < 16; i++) printf(" %02x", p[i]);
                        printf("\n");
                        printf("DIAG   raw[1056:1072] =");
                        for (int i = 0; i < 16; i++) printf(" %02x", p[1056+i]);
                        printf("\n");
                    }
                }
            }
            printf("DIAG DONE, exiting before inference.\n");
            return 0;
        }
        // warm up + short generation
        fastllm::GenerationConfig genConfig;
        genConfig.output_token_limit = 16;
        RuntimeResult retCb = nullptr;
        std::string output = model->Response(prompt, retCb, genConfig);
        printf("PROMPT: %s\n", prompt.c_str());
        printf("OUTPUT: %s\n", output.c_str());
    } catch (const std::exception &e) {
        printf("EXCEPTION: %s\n", e.what());
        return 1;
    }
    return 0;
}
