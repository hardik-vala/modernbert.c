/*
 * C implementation of ModernBERT
 * Inspired by llama2.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <fcntl.h>
#include <cblas.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif
#include "tokenizer.h"

// ----------------------------------------------------------------------------
// Utilities

long time_in_ms()
{
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

int print_v(float *x, int n)
{
    int chars_printed = 0;
    chars_printed += printf("[");
    if (n <= 6)
    {
        for (int i = 0; i < n; i++)
        {
            chars_printed += printf("%.6f", x[i]);
            if (i < n - 1)
            {
                chars_printed += printf(", ");
            }
        }
    }
    else
    {
        for (int i = 0; i < 3; i++)
        {
            chars_printed += printf("%.6f, ", x[i]);
        }
        chars_printed += printf("..., ");
        for (int i = n - 3; i < n; i++)
        {
            chars_printed += printf("%.6f", x[i]);
            if (i < n - 1)
            {
                chars_printed += printf(", ");
            }
        }
    }

    chars_printed += printf("]");
    return chars_printed;
}

// ----------------------------------------------------------------------------
// Model structures

typedef struct
{
    int dim; // embedding dimension
    int vocab_size;
    int n_layers;                   // number of layers
    float norm_eps;                 // episilon for layernorm
    int max_seq_len;                // max sequence length
    int n_heads;                    // number of attention heads
    float global_rope_theta;        // rope theta for global attention layers
    float local_rope_theta;         // rope theta for local attention layers
    int intermediate_dim;           // for mlp layers
    int global_attn_every_n_layers; // how often to have a global attention layer
    int local_attention;            // local attention window size
    int n_labels;                   // number of labels for token classification (0 if not used)
} ModelConfig;

typedef struct
{
    float *token_embedding; // (vocab_size, dim)
    float *embedding_norm;  // (dim,)
    float *attn_norm;       // (layers-1, dim)
    float *mlp_norm;        // (layers, dim)
    float *attn_wqkv;       // (layers, 3*dim, dim)
    float *attn_wo;         // (layers, dim, dim)
    float *mlp_wi;          // (layers, dim, 2*intermediate_dim)
    float *mlp_wo;          // (layers, dim, intermediate_dim)
    float *final_norm;      // (dim,)
    // (optional) token classification
    float *pred_dense; // (dim, dim)
    float *pred_norm;  // (dim,)
    float *clf_w;      // (dim, n_labels)
    float *clf_b;      // (n_labels,)
} ModelWeights;

typedef struct
{
    float *h;               // activation for entire sequence (seq_len, dim)
    float *h2;              // additional buffer for entire sequence (seq_len, dim)
    float *h_mlp;           // activation buffer for intermediates in mlp (seq_len, 2*intermediate_dim)
    float *h2_mlp;          // additional buffer for mlp intermediates (seq_len, intermediate_dim)
    float *h_qkv;           // qkv projection for entire sequence (seq_len, 3*dim)
    float *global_rope_cos; // precomputed cos's for global RoPE (max_seq_len, head_dim/2)
    float *global_rope_sin; // precomputed sin's for global RoPE (max_seq_len, head_dim/2)
    float *local_rope_cos;  // precomputed cos's for local RoPE (max_seq_len, head_dim/2)
    float *local_rope_sin;  // precomputed sin's for local RoPE (max_seq_len, head_dim/2)
    float *q;               // queries for entire sequence (seq_len, dim)
    float *k;               // keys for entire sequence (seq_len, dim)
    float *v;               // values for entire sequence (seq_len, dim)
    float *attn;            // attention scores (n_heads, seq_len, seq_len)
    float *attn_out;        // attention output (seq_len, dim)
    unsigned int *pred;     // token classification predictions (seq_len,)
} ModelRunState;

typedef struct
{
    ModelConfig config;
    ModelWeights weights;
    ModelRunState state; // buffers for the "wave" of activations in the forward pass
    int fd;              // file descriptor for memory mapping
    float *data;         // memory mapped data pointer
    ssize_t file_size;   // size of the checkpoint file in bytes
} Model;

// ----------------------------------------------------------------------------
// Model loading

void memory_map_weights(ModelWeights *w, ModelConfig *p, float *ptr)
{
    w->token_embedding = ptr;
    ptr += p->vocab_size * p->dim;
    w->embedding_norm = ptr;
    ptr += p->dim;
    w->attn_norm = ptr;
    ptr += (p->n_layers - 1) * p->dim;
    w->mlp_norm = ptr;
    ptr += p->n_layers * p->dim;
    w->attn_wqkv = ptr;
    ptr += p->n_layers * (3 * p->dim) * p->dim;
    w->attn_wo = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->mlp_wi = ptr;
    ptr += p->n_layers * p->dim * (2 * p->intermediate_dim);
    w->mlp_wo = ptr;
    ptr += p->n_layers * p->dim * p->intermediate_dim;
    w->final_norm = ptr;
    if (!p->n_labels)
        return;
    ptr += p->dim;
    w->pred_dense = ptr;
    ptr += p->dim * p->dim;
    w->pred_norm = ptr;
    ptr += p->dim;
    w->clf_w = ptr;
    ptr += p->dim * p->n_labels;
    w->clf_b = ptr;
}

void read_checkpoint(const char *checkpoint, ModelConfig *config, ModelWeights *weights,
                     int *fd, float **data, ssize_t *file_size)
{
    FILE *file = fopen(checkpoint, "rb");
    if (!file)
    {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }

    // Read magic number
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1)
    {
        fprintf(stderr, "Failed to read magic number\n");
        exit(EXIT_FAILURE);
    }

    // Read version
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Failed to read version\n");
        exit(EXIT_FAILURE);
    }

    // Read model config
    if (fread(config, sizeof(ModelConfig), 1, file) != 1)
    {
        fprintf(stderr, "Failed to read config\n");
        exit(EXIT_FAILURE);
    }

    // Figure out the file size
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);

    // Memory map the model weights into the data pointer
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1)
    {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED)
    {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }

    // Weights start after 256-byte header
    float *weights_ptr = *data + 256 / sizeof(float);
    memory_map_weights(weights, config, weights_ptr);
}

void init_rope(ModelRunState *s, ModelConfig *p)
{
    int head_dim = p->dim / p->n_heads;
    float global_theta = p->global_rope_theta;
    float local_theta = p->local_rope_theta;

    for (int i = 0; i < p->max_seq_len; i++)
    {
        for (int j = 0; j < head_dim / 2; j++)
        {
            float freq = 1.0f / powf(global_theta, (float)(2 * j) / head_dim);
            float angle = freq * i;
            s->global_rope_cos[i * (head_dim / 2) + j] = cosf(angle);
            s->global_rope_sin[i * (head_dim / 2) + j] = sinf(angle);

            freq = 1.0f / powf(local_theta, (float)(2 * j) / head_dim);
            angle = freq * i;
            s->local_rope_cos[i * (head_dim / 2) + j] = cosf(angle);
            s->local_rope_sin[i * (head_dim / 2) + j] = sinf(angle);
        }
    }
}

void init_run_state(ModelRunState *s, ModelConfig *p, int seq_len)
{
    int head_dim = p->dim / p->n_heads;

    s->h = (float *)malloc(seq_len * p->dim * sizeof(float));
    s->h2 = (float *)malloc(seq_len * p->dim * sizeof(float));
    s->h_mlp = (float *)malloc(seq_len * 2 * p->intermediate_dim * sizeof(float));
    s->h2_mlp = (float *)malloc(seq_len * p->intermediate_dim * sizeof(float));
    s->h_qkv = (float *)malloc(seq_len * 3 * p->dim * sizeof(float));
    s->global_rope_cos = (float *)malloc(p->max_seq_len * (head_dim / 2) * sizeof(float));
    s->global_rope_sin = (float *)malloc(p->max_seq_len * (head_dim / 2) * sizeof(float));
    s->local_rope_cos = (float *)malloc(p->max_seq_len * (head_dim / 2) * sizeof(float));
    s->local_rope_sin = (float *)malloc(p->max_seq_len * (head_dim / 2) * sizeof(float));
    s->q = (float *)malloc(seq_len * p->dim * sizeof(float));
    s->k = (float *)malloc(seq_len * p->dim * sizeof(float));
    s->v = (float *)malloc(seq_len * p->dim * sizeof(float));
    s->attn = (float *)malloc(p->n_heads * seq_len * seq_len * sizeof(float));
    s->attn_out = (float *)malloc(seq_len * p->dim * sizeof(float));
    s->pred = (unsigned int *)malloc(seq_len * sizeof(int));

    if (!s->h || !s->h2 || !s->h_mlp || !s->h2_mlp || !s->h_qkv || !s->global_rope_cos || !s->global_rope_sin || !s->local_rope_cos || !s->local_rope_sin || !s->q || !s->k || !s->v || !s->attn || !s->attn_out)
    {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    init_rope(s, p);
}

void free_run_state(ModelRunState *s)
{
    free(s->h);
    free(s->h2);
    free(s->h_mlp);
    free(s->h2_mlp);
    free(s->h_qkv);
    free(s->global_rope_cos);
    free(s->global_rope_sin);
    free(s->local_rope_cos);
    free(s->local_rope_sin);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->attn);
    free(s->attn_out);
    free(s->pred);
}

void build_model(Model *m, const char *model_path)
{
    read_checkpoint(model_path, &m->config, &m->weights, &m->fd, &m->data, &m->file_size);
}

void free_model(Model *m)
{
    if (m->data != MAP_FAILED)
    {
        munmap(m->data, m->file_size);
    }
    if (m->fd != -1)
    {
        close(m->fd);
    }
}

// ----------------------------------------------------------------------------
// Forward

float vmean(float *x, int n)
{
    // y = mean(x) over n elements
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += x[i];
    }
    return sum / n;
}

void vadd(float *y, float *x1, float *x2, int n)
{
    // y = x1 + x2 over n elements
    for (int i = 0; i < n; i++)
    {
        y[i] = x1[i] + x2[i];
    }
}

void layernorm(float *out, float *h, float *weight, float eps, int n)
{
    float mean = vmean(h, n);

    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float diff = h[i] - mean;
        sum_sq += diff * diff;
    }
    float std = sqrtf(sum_sq / n + eps);

    for (int i = 0; i < n; i++)
    {
        out[i] = (h[i] - mean) / std * weight[i];
    }
}

void softmax(float *x, int n)
{
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < n; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum_exp += x[i];
    }

    for (int i = 0; i < n; i++)
    {
        x[i] /= sum_exp;
    }
}

void gelu(float *out, float *in, int n)
{
    // Fast GELU approximation using tanh
    // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    // This is ~10x faster than erf() and very close in accuracy
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (int i = 0; i < n; i++)
    {
        float x = in[i];
        float x3 = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x3);
        out[i] = 0.5f * x * (1.0f + tanhf(tanh_arg));
    }
}

void gelu_gate(float *out, float *in, float *gate, int n)
{
    // GeLU activation with gating
    gelu(out, in, n);
    for (int i = 0; i < n; i++)
    {
        out[i] *= gate[i];
    }
}

// bidirectional forward pass for entire sequence
float *forward(Model *model, int *tokens, int seq_len, int tokclf_mode)
{
    ModelConfig *p = &model->config;
    ModelWeights *w = &model->weights;
    ModelRunState *s = &model->state;
    int head_dim = p->dim / p->n_heads;
    float attn_scale = 1.0f / sqrtf((float)head_dim);

    // Embedding: copy token embeddings for all positions and apply layernorm
    for (int pos = 0; pos < seq_len; pos++)
    {
        float *h_pos = s->h + pos * p->dim;
        float *embedding = w->token_embedding + tokens[pos] * p->dim;
        memcpy(h_pos, embedding, p->dim * sizeof(float));
        layernorm(h_pos, h_pos, w->embedding_norm, p->norm_eps, p->dim);
    }

    for (int layer = 0; layer < p->n_layers; layer++)
    {
        // Select RoPE parameters based on layer type
        int is_local = p->local_attention && (layer % p->global_attn_every_n_layers) != 0;
        float *rope_cos, *rope_sin;
        if (is_local)
        {
            rope_cos = s->local_rope_cos;
            rope_sin = s->local_rope_sin;
        }
        else
        {
            rope_cos = s->global_rope_cos;
            rope_sin = s->global_rope_sin;
        }

        if (layer != 0)
        {
            for (int pos = 0; pos < seq_len; pos++)
            {
                layernorm(
                    s->h2 + pos * p->dim,                // out
                    s->h + pos * p->dim,                 // h
                    w->attn_norm + (layer - 1) * p->dim, // weight
                    p->norm_eps,                         // eps
                    p->dim                               // n
                );
            }
        }
        else
        {
            // layer 0: identity skip connection (no layernorm)
            memcpy(s->h2, s->h, seq_len * p->dim * sizeof(float));
        }

        // attn: qkv projection (batched)
        // Y (seq_len, 3*dim) = X (seq_len, dim) @ W^T where W is (3*dim, dim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len,    // M: rows of output
                    3 * p->dim, // N: cols of output
                    p->dim,     // K: shared dimension
                    1.0f,
                    s->h2, p->dim,                                        // A: input (seq_len, dim)
                    w->attn_wqkv + layer * (3 * p->dim) * p->dim, p->dim, // B: weight (3*dim, dim)
                    0.0f,
                    s->h_qkv, 3 * p->dim); // C: output (seq_len, 3*dim)

        // attn: RoPE
        for (int pos = 0; pos < seq_len; pos++)
        {
            float *q_ptr = s->h_qkv + pos * 3 * p->dim;
            float *k_ptr = q_ptr + p->dim;
            float *v_ptr = k_ptr + p->dim;

            float *q_out = s->q + pos * p->dim;
            float *k_out = s->k + pos * p->dim;
            float *v_out = s->v + pos * p->dim;

            // apply RoPE to q and k
            for (int i = 0; i < p->n_heads; i++)
            {
                for (int j = 0; j < head_dim / 2; j++)
                {
                    int pos_idx = pos * (head_dim / 2) + j;
                    float fcr = rope_cos[pos_idx];
                    float fsi = rope_sin[pos_idx];

                    int idx1 = i * head_dim + j;
                    int idx2 = i * head_dim + j + head_dim / 2;

                    q_out[idx1] = q_ptr[idx1] * fcr - q_ptr[idx2] * fsi;
                    q_out[idx2] = q_ptr[idx2] * fcr + q_ptr[idx1] * fsi;

                    k_out[idx1] = k_ptr[idx1] * fcr - k_ptr[idx2] * fsi;
                    k_out[idx2] = k_ptr[idx2] * fcr + k_ptr[idx1] * fsi;
                }
            }

            // copy v (no RoPE for values)
            memcpy(v_out, v_ptr, p->dim * sizeof(float));
        }

        // attn: compute bidirectional attention scores for all pairs
        for (int hd = 0; hd < p->n_heads; hd++)
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len,
                        seq_len,
                        head_dim,
                        attn_scale,
                        s->q + hd * head_dim,
                        p->dim,
                        s->k + hd * head_dim,
                        p->dim,
                        0.0f,
                        s->attn + hd * seq_len * seq_len,
                        seq_len);
            for (int q_pos = 0; q_pos < seq_len; q_pos++)
            {
                float *attn_row = s->attn + hd * seq_len * seq_len + q_pos * seq_len;

                if (is_local)
                {

                    int half_window = p->local_attention / 2;
                    int start_pos = q_pos - half_window;
                    int end_pos = q_pos + half_window + 1;
                    if (start_pos > 0)
                    {
                        for (int k = 0; k < start_pos; k++)
                        {
                            attn_row[k] = -INFINITY;
                        }
                    }
                    if (end_pos < seq_len)
                    {
                        for (int k = end_pos; k < seq_len; k++)
                        {
                            attn_row[k] = -INFINITY;
                        }
                    }
                }

                softmax(attn_row, seq_len);
            }
        }

        // attn: attention output, i.e. weighted sum of values
        for (int hd = 0; hd < p->n_heads; hd++)
        {
            // Y (seq_len, head_dim) = A (seq_len, seq_len) @ B (seq_len, head_dim)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq_len,
                        head_dim,
                        seq_len,
                        1.0f,
                        s->attn + hd * seq_len * seq_len, seq_len,
                        s->v + hd * head_dim, p->dim,
                        0.0f,
                        s->attn_out + hd * head_dim, p->dim);
        }

        // attn: output projection (batched)
        // Y (seq_len, dim) = X (seq_len, dim) @ W^T where W is (dim, dim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, // M: rows of output
                    p->dim,  // N: cols of output
                    p->dim,  // K: shared dimension
                    1.0f,
                    s->attn_out, p->dim,                          // A: input (seq_len, dim)
                    w->attn_wo + layer * p->dim * p->dim, p->dim, // B: weight (dim, dim)
                    0.0f,
                    s->h2, p->dim); // C: output (seq_len, dim)

        // residual connection after attention
        for (int pos = 0; pos < seq_len; pos++)
        {
            vadd(
                s->h + pos * p->dim,  // y
                s->h + pos * p->dim,  // x1
                s->h2 + pos * p->dim, // x2
                p->dim                // n
            );
        }

        // mlp: layernorm
        for (int pos = 0; pos < seq_len; pos++)
        {
            layernorm(
                s->h2 + pos * p->dim,         // out
                s->h + pos * p->dim,          // h
                w->mlp_norm + layer * p->dim, // weight
                p->norm_eps,                  // eps
                p->dim                        // n
            );
        }

        // mlp: wi projection (batched)
        // Y (seq_len, 2*intermediate_dim) = X (seq_len, dim) @ W^T where W is (2*intermediate_dim, dim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len,                 // M: rows of output
                    2 * p->intermediate_dim, // N: cols of output
                    p->dim,                  // K: shared dimension
                    1.0f,
                    s->h2, p->dim,                                                  // A: input (seq_len, dim)
                    w->mlp_wi + layer * p->dim * (2 * p->intermediate_dim), p->dim, // B: weight (2*intermediate_dim, dim)
                    0.0f,
                    s->h_mlp, 2 * p->intermediate_dim); // C: output (seq_len, 2*intermediate_dim)

        // mlp: gelu activation with gating
        for (int pos = 0; pos < seq_len; pos++)
        {
            float *in_pos = s->h_mlp + pos * 2 * p->intermediate_dim;
            float *gate_pos = in_pos + p->intermediate_dim;
            float *out_pos = s->h2_mlp + pos * p->intermediate_dim;

            gelu_gate(
                out_pos,            // out
                in_pos,             // in
                gate_pos,           // gate
                p->intermediate_dim // n
            );
        }

        // mlp: wo projection (batched)
        // Y (seq_len, dim) = X (seq_len, intermediate_dim) @ W^T where W is (dim, intermediate_dim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len,             // M: rows of output
                    p->dim,              // N: cols of output
                    p->intermediate_dim, // K: shared dimension
                    1.0f,
                    s->h2_mlp, p->intermediate_dim,                                        // A: input (seq_len, intermediate_dim)
                    w->mlp_wo + layer * p->dim * p->intermediate_dim, p->intermediate_dim, // B: weight (dim, intermediate_dim)
                    0.0f,
                    s->h2, p->dim); // C: output (seq_len, dim)

        // residual connection after mlp
        for (int pos = 0; pos < seq_len; pos++)
        {
            vadd(
                s->h + pos * p->dim,  // y
                s->h + pos * p->dim,  // x1
                s->h2 + pos * p->dim, // x2
                p->dim                // n
            );
        }
    }

    // final layernorm
    for (int pos = 0; pos < seq_len; pos++)
    {
        layernorm(
            s->h + pos * p->dim, // out
            s->h + pos * p->dim, // h
            w->final_norm,       // weight
            p->norm_eps,         // eps
            p->dim               // n
        );
    }

    if (!tokclf_mode)
    {
        return s->h;
    }

    // token classification: apply prediction head dense (batched)
    // Y (seq_len, dim) = X (seq_len, dim) @ W^T where W is (dim, dim)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, // M: rows of output
                p->dim,  // N: cols of output
                p->dim,  // K: shared dimension
                1.0f,
                s->h, p->dim,          // A: input (seq_len, dim)
                w->pred_dense, p->dim, // B: weight (dim, dim)
                0.0f,
                s->h2, p->dim); // C: output (seq_len, dim)

    // token classification: apply prediction head activation (GeLU)
    for (int pos = 0; pos < seq_len; pos++)
    {
        float *h2_pos = s->h2 + pos * p->dim;
        gelu(h2_pos, h2_pos, p->dim);
    }

    // token classification: apply prediction head layernorm
    for (int pos = 0; pos < seq_len; pos++)
    {
        layernorm(
            s->h2 + pos * p->dim, // out
            s->h2 + pos * p->dim, // h
            w->pred_norm,         // weight
            p->norm_eps,          // eps
            p->dim                // n
        );
    }

    // token classification: apply classification linear layer (batched)
    // Y (seq_len, n_labels) = X (seq_len, dim) @ W^T where W is (n_labels, dim)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len,     // M: rows of output
                p->n_labels, // N: cols of output
                p->dim,      // K: shared dimension
                1.0f,
                s->h2, p->dim,    // A: input (seq_len, dim)
                w->clf_w, p->dim, // B: weight (n_labels, dim)
                0.0f,
                s->h, p->n_labels); // C: output (seq_len, n_labels)

    // add bias for all positions
    for (int pos = 0; pos < seq_len; pos++)
    {
        float *out_pos = s->h + pos * p->n_labels;
        for (int i = 0; i < p->n_labels; i++)
        {
            out_pos[i] += w->clf_b[i];
        }
    }

    return s->h;
}

void classify(Model *model, int seq_len)
{
    ModelConfig *p = &model->config;
    ModelRunState *s = &model->state;

    // For each position, find the label with the highest logit
    for (int pos = 0; pos < seq_len; pos++)
    {
        float *logits = s->h + pos * p->dim;

        int max_i = 0;
        for (int i = 0; i < p->n_labels; i++)
        {
            if (logits[max_i] < logits[i])
            {
                max_i = i;
            }
        }
        s->pred[pos] = max_i;
    }
}

// ----------------------------------------------------------------------------
// Output printing

void print_outputs(int *tokens, int num_tokens, float *outputs, int output_dim, Tokenizer *tokenizer)
{
    printf("\n");
    printf("┌───────┬──────────────────┬──────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-5s │ %-16s │ %-76s │\n", "ID", "Token", "Hidden State");
    printf("├───────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────┤\n");

    for (int i = 0; i < num_tokens; i++)
    {
        int token_id = tokens[i];
        char *token_str;
        int should_free = 0;

        if (token_id == tokenizer->cls_id)
        {
            token_str = "[CLS]";
        }
        else if (token_id == tokenizer->sep_id)
        {
            token_str = "[SEP]";
        }
        else if (token_id >= tokenizer->vocab_size)
        {
            token_str = "[UNK]";
        }
        else
        {
            // Decode single token
            int single_token[1] = {token_id};
            token_str = decode(tokenizer, single_token, 1);
            should_free = 1;

            // If decode returns empty or NULL, use raw vocab
            if (!token_str || token_str[0] == '\0')
            {
                if (token_str)
                    free(token_str);
                token_str = tokenizer->vocab[token_id];
                should_free = 0;
            }
        }

        // Print token with proper padding to account for Unicode characters
        int token_col_width = 16;
        printf("│ %-5d │ %s", token_id, token_str);

        // Calculate display width (count actual characters, not bytes)
        int display_width = 0;
        for (const char *p = token_str; *p; p++)
        {
            // Count only the start bytes of UTF-8 sequences (not continuation bytes)
            if ((*p & 0xC0) != 0x80)
            {
                display_width++;
            }
        }

        // Add padding to reach the column width
        int token_padding = token_col_width - display_width;
        if (token_padding > 0)
        {
            printf("%*s", token_padding, "");
        }
        printf(" │ ");

        // Print vector values with proper padding
        int values_width = 76; // Width of the values column
        int chars_printed;

        chars_printed = print_v(outputs + i * output_dim, output_dim);

        // Add padding to align the right border
        int padding = values_width - chars_printed;
        if (padding > 0)
        {
            printf("%*s", padding, "");
        }
        printf(" │\n");

        if (should_free)
        {
            free(token_str);
        }
    }

    printf("└───────┴──────────────────┴──────────────────────────────────────────────────────────────────────────────┘\n");
}

void print_predictions(int *tokens, int num_tokens, Tokenizer *tokenizer, unsigned int *pred)
{
    printf("\n");
    printf("┌───────┬──────────────────┬──────────────────┐\n");
    printf("│ %-5s │ %-16s │ %-16s │\n", "ID", "Token", "Prediction");
    printf("├───────┼──────────────────┼──────────────────┤\n");

    for (int i = 0; i < num_tokens; i++)
    {
        int token_id = tokens[i];
        char *token_str;
        int should_free = 0;

        if (token_id == tokenizer->cls_id)
        {
            token_str = "[CLS]";
        }
        else if (token_id == tokenizer->sep_id)
        {
            token_str = "[SEP]";
        }
        else if (token_id >= tokenizer->vocab_size)
        {
            token_str = "[UNK]";
        }
        else
        {
            // Decode single token
            int single_token[1] = {token_id};
            token_str = decode(tokenizer, single_token, 1);
            should_free = 1;

            // If decode returns empty or NULL, use raw vocab
            if (!token_str || token_str[0] == '\0')
            {
                if (token_str)
                    free(token_str);
                token_str = tokenizer->vocab[token_id];
                should_free = 0;
            }
        }

        // Print token with proper padding to account for Unicode characters
        int token_col_width = 16;
        printf("│ %-5d │ %s", token_id, token_str);

        // Calculate display width (count actual characters, not bytes)
        int display_width = 0;
        for (const char *p = token_str; *p; p++)
        {
            // Count only the start bytes of UTF-8 sequences (not continuation bytes)
            if ((*p & 0xC0) != 0x80)
            {
                display_width++;
            }
        }

        // Add padding to reach the column width
        int token_padding = token_col_width - display_width;
        if (token_padding > 0)
        {
            printf("%*s", token_padding, "");
        }
        printf(" │ ");

        // Print prediction with proper padding
        printf("%-16u", pred[i]);
        printf(" │\n");

        if (should_free)
        {
            free(token_str);
        }
    }

    printf("└───────┴──────────────────┴──────────────────┘\n");
}

// ----------------------------------------------------------------------------
// Main function

void error_usage()
{
    fprintf(stderr, "Usage:   run [--tokclf --n_labels N] [--quiet] <model.bin> <tokenizer.bin> <prompt>\n");
    fprintf(stderr, "Example: ./run model.bin tokenizer.bin \"hello world\"\n");
    fprintf(stderr, "         ./run --tokclf --n_labels 13 model.bin tokenizer.bin \"hello world\"\n");
    fprintf(stderr, "         ./run --quiet model.bin tokenizer.bin \"hello world\"\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --tokclf        Enable token classification mode\n");
    fprintf(stderr, "  --n_labels N    Number of labels for token classification (required with --tokclf)\n");
    fprintf(stderr, "  --quiet         Suppress printing of token table and outputs\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    // Parse command line flags
    int tokclf_mode = 0;
    int n_labels = 0;
    int quiet_mode = 0;
    int arg_idx = 1;

    // Check for --tokclf, --n_labels, and --quiet flags
    while (arg_idx < argc && argv[arg_idx][0] == '-')
    {
        if (strcmp(argv[arg_idx], "--tokclf") == 0)
        {
            tokclf_mode = 1;
            arg_idx++;
        }
        else if (strcmp(argv[arg_idx], "--n_labels") == 0)
        {
            if (arg_idx + 1 >= argc)
            {
                fprintf(stderr, "Error: --n_labels requires a value\n");
                error_usage();
            }
            n_labels = atoi(argv[arg_idx + 1]);
            arg_idx += 2;
        }
        else if (strcmp(argv[arg_idx], "--quiet") == 0)
        {
            quiet_mode = 1;
            arg_idx++;
        }
        else
        {
            fprintf(stderr, "Error: Unknown flag %s\n", argv[arg_idx]);
            error_usage();
        }
    }

    // Validate flags
    if (!tokclf_mode && n_labels > 0)
    {
        fprintf(stderr, "Error: --n_labels can only be used with --tokclf\n");
        error_usage();
    }

    // Normal mode - tokenization
    if (arg_idx + 2 >= argc)
    {
        error_usage();
    }

    char *model_path = argv[arg_idx];
    char *tokenizer_path = argv[arg_idx + 1];
    char *prompt = argv[arg_idx + 2];

    // Build the model
    Model model;
    build_model(&model, model_path);

    // Override config if tokclf mode is enabled
    if (tokclf_mode && n_labels > 0)
    {
        model.config.n_labels = n_labels;
    }

    // build the tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path);

    // encode the prompt
    long start = time_in_ms();
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3 + 2) * sizeof(int));
    int num_prompt_tokens = 0;
    encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    long end = time_in_ms();
    double tok_latency_in_sec = (end - start) / 1000.0;

    // allocate run state based on actual sequence length
    init_run_state(&model.state, &model.config, num_prompt_tokens);

    // timing inference
    start = time_in_ms();
    // run bidirectional forward pass on entire sequence
    float *forward_out = forward(
        &model,
        prompt_tokens,
        num_prompt_tokens,
        tokclf_mode);
    end = time_in_ms();

    if (tokclf_mode)
    {
        // classify each token
        classify(&model, num_prompt_tokens);
        if (!quiet_mode)
            print_predictions(prompt_tokens, num_prompt_tokens, &tokenizer, model.state.pred);
    }
    else
    {
        // print the tokens and their final hidden states (unless quiet mode)
        if (!quiet_mode)
        {
            print_outputs(prompt_tokens, num_prompt_tokens, forward_out, model.config.dim, &tokenizer);
        }
    }

    // print latency
    double latency_in_sec = (end - start) / 1000.0;
    fprintf(stderr, "\n");
    fprintf(stderr, "latency (sec)\n");
    fprintf(stderr, "  tokenization: %.3fs\n", tok_latency_in_sec);
    fprintf(stderr, "  inference: %.3fs\n", latency_in_sec + tok_latency_in_sec);

    // cleanup
    free(prompt_tokens);
    free_tokenizer(&tokenizer);
    free_run_state(&model.state);
    free_model(&model);
    return 0;
}