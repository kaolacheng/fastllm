// Attempt to load a DeepSeek-V4 IQ1_S GGUF through fastllm's GGUF loader,
// to see how far the load chain gets. If MXFP4 repacking (Task #12) is the
// remaining blocker, this shows exactly where it stops.

#include "model.h"
#include "fastllm.h"
#include <cstdio>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    std::string path = argv[1];
    printf("Loading %s ...\n", path.c_str());
    try {
        auto model = fastllm::CreateLLMModelFromGGUFFile(path, "");
        if (model == nullptr) {
            printf("LOAD FAILED: returned null\n");
            return 1;
        }
        printf("LOADED OK. model_type=%s block_cnt=%d embed_dim=%d\n",
               model->model_type.c_str(), model->block_cnt, model->embed_dim);
        printf("num_attention_heads=%d num_key_value_heads=%d max_positions=%d\n",
               model->num_attention_heads, model->num_key_value_heads, model->max_positions);
        printf("weight count = %zu\n", model->weight.weight.size());
    } catch (const std::exception &e) {
        printf("LOAD EXCEPTION: %s\n", e.what());
        return 1;
    }
    return 0;
}
