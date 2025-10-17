#ifndef TOKENIZER_H
#define TOKENIZER_H

// ----------------------------------------------------------------------------
// Tokenizer structures and functions

typedef struct
{
    char *str;
    int id;
} TokenIndex;

typedef struct
{
    char **vocab;   // Unified vocab array containing all tokens
    int vocab_size; // Total number of tokens in vocab
    unsigned int max_token_length;
    char **merges;
    int num_merges;
    TokenIndex *sorted_vocab;
    // Added token IDs: dense array of size num_added_tokens
    int *added_token_ids;
    int num_added_tokens;
    int cls_id;
    int sep_id;
} Tokenizer;

// Function declarations
void build_tokenizer(Tokenizer *t, const char *tokenizer_path);
void free_tokenizer(Tokenizer *t);
void encode(Tokenizer *t, const char *text, int *tokens, int *n_tokens);
char* decode(Tokenizer *t, int *tokens, int n_tokens);

#endif // TOKENIZER_H