/*
 * Tokenizer implementation for ModernBERT
 * Implements ByteLevel BPE tokenization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include "tokenizer.h"

// ----------------------------------------------------------------------------
// ByteLevel encoding/decoding

// ByteLevel mapping: maps bytes to printable Unicode characters
// This must match the exact mapping used by the tokenizer
const char *BYTE_LEVEL_CHARS[256] = {
    // 0-32: non-printable characters -> special Unicode chars
    "Ā", "ā", "Ă", "ă", "Ą", "ą", "Ć", "ć", "Ĉ", "ĉ", "Ċ", "ċ", "Č", "č", "Ď", "ď",
    "Đ", "đ", "Ē", "ē", "Ĕ", "ĕ", "Ė", "ė", "Ę", "ę", "Ě", "ě", "Ĝ", "ĝ", "Ğ", "ğ",
    "Ġ",
    // 33-126: printable ASCII -> themselves
    "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?",
    "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_",
    "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~",
    // 127: DEL -> special char
    "ġ",
    // 128-255: extended ASCII -> special Unicode chars
    "Ģ", "ģ", "Ĥ", "ĥ", "Ħ", "ħ", "Ĩ", "ĩ", "Ī", "ī", "Ĭ", "ĭ", "Į", "į", "İ", "ı",
    "Ĳ", "ĳ", "Ĵ", "ĵ", "Ķ", "ķ", "ĸ", "Ĺ", "ĺ", "Ļ", "ļ", "Ľ", "ľ", "Ŀ", "ŀ", "Ł",
    "ł", "Ń", "ń", "Ņ", "ņ", "Ň", "ň", "ŉ", "Ŋ", "ŋ", "Ō", "ō", "Ŏ", "ŏ", "Ő", "ő",
    "Œ", "œ", "Ŕ", "ŕ", "Ŗ", "ŗ", "Ř", "ř", "Ś", "ś", "Ŝ", "ŝ", "Ş", "ş", "Š", "š",
    "Ţ", "ţ", "Ť", "ť", "Ŧ", "ŧ", "Ũ", "ũ", "Ū", "ū", "Ŭ", "ŭ", "Ů", "ů", "Ű", "ű",
    "Ų", "ų", "Ŵ", "ŵ", "Ŷ", "ŷ", "Ÿ", "Ź", "ź", "Ż", "ż", "Ž", "ž", "ſ", "Ɓ", "Ƃ",
    "ƃ", "Ɔ", "Ɖ", "Ɗ", "Ƌ", "ƌ", "Ɛ", "Ə", "Ƒ", "ƒ", "Ɠ", "Ɣ", "ƕ", "Ɩ", "Ɨ", "Ƙ",
    "ƙ", "ƚ", "ƛ", "Ɯ", "Ɲ", "ƞ", "Ɵ", "Ơ", "ơ", "Ƣ", "ƣ", "Ƥ", "ƥ", "Ʀ", "Ƨ", "ƨ"};

int byte_level_pre_tokenize(const char *text, int start, int length, char **tokens, int max_tokens)
{
    int token_count = 0;

    for (int i = 0; i < length && token_count < max_tokens; i++)
    {
        unsigned char byte = (unsigned char)text[start + i];
        tokens[token_count] = malloc(strlen(BYTE_LEVEL_CHARS[byte]) + 1);
        strcpy(tokens[token_count], BYTE_LEVEL_CHARS[byte]);
        token_count++;
    }

    return token_count;
}

// ----------------------------------------------------------------------------
// Preprocessing: Text splitting with regex

// Split text using standard regex pattern (using PCRE2 for full Unicode support)
// Returns starting indices of each split
int regex_split(const char *text, int *split_starts, int max_splits)
{
    int split_count = 0;
    int text_len = strlen(text);

    // Pattern from HuggingFace ByteLevel tokenizer
    const char *pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    int errornumber;
    PCRE2_SIZE erroroffset;
    pcre2_code *re = pcre2_compile(
        (PCRE2_SPTR)pattern,
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF,
        &errornumber,
        &erroroffset,
        NULL);

    if (re == NULL)
    {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errornumber, buffer, sizeof(buffer));
        fprintf(stderr, "PCRE2 compilation failed at offset %d: %s\n", (int)erroroffset, buffer);
        return 0;
    }

    pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(re, NULL);

    PCRE2_SIZE start_offset = 0;

    while (start_offset < text_len && split_count < max_splits)
    {
        int rc = pcre2_match(
            re,
            (PCRE2_SPTR)text,
            text_len,
            start_offset,
            0,
            match_data,
            NULL);

        if (rc < 0)
        {
            // No more matches
            if (rc != PCRE2_ERROR_NOMATCH)
            {
                fprintf(stderr, "PCRE2 matching error %d\n", rc);
            }
            // If there's remaining text, add it as a split
            if (start_offset < text_len && split_count < max_splits)
            {
                split_starts[split_count] = start_offset;
                split_count++;
            }
            break;
        }

        PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE match_start = ovector[0];
        PCRE2_SIZE match_end = ovector[1];

        if (match_end > match_start)
        {
            // Record the split starting index
            split_starts[split_count] = match_start;
            split_count++;

            // Move to the end of this match
            start_offset = match_end;
        }
        else
        {
            // Zero-length match, move forward by 1 to avoid infinite loop
            start_offset++;
        }
    }

    pcre2_match_data_free(match_data);
    pcre2_code_free(re);

    return split_count;
}

// ----------------------------------------------------------------------------
// Helper functions

int compare_tokens(const void *a, const void *b)
{
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

// Check if a text segment matches any added tokens
int check_added_token(Tokenizer *t, const char *text, int start, int length)
{
    // Iterate through added token IDs and check each one
    for (int i = 0; i < t->num_added_tokens; i++)
    {
        int token_id = t->added_token_ids[i];
        if (token_id < t->vocab_size)
        {
            char *added_token_str = t->vocab[token_id];
            int added_token_len = strlen(added_token_str);
            if (length == added_token_len && strncmp(text + start, added_token_str, length) == 0)
            {
                return token_id; // Return the original token ID
            }
        }
    }
    return -1; // Not found
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
{
    TokenIndex tok = {.str = str};
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

// Helper function to perform BPE merges on a token sequence
void bpe_merge(Tokenizer *t, int *tokens, int *n_tokens)
{
    char *str_buffer = malloc((t->max_token_length * 2 + 1) * sizeof(char));

    while (1)
    {
        int best_merge_idx = -1;
        int best_merge_rule_idx = t->num_merges;

        // find the best merge
        for (int i = 0; i < (*n_tokens) - 1; i++)
        {
            // Get token strings directly from vocab array
            if (tokens[i] >= t->vocab_size || tokens[i + 1] >= t->vocab_size)
            {
                continue; // skip invalid token IDs
            }

            char *token1_str = t->vocab[tokens[i]];
            char *token2_str = t->vocab[tokens[i + 1]];

            sprintf(str_buffer, "%s %s", token1_str, token2_str);
            for (int j = 0; j < t->num_merges; j++)
            {
                if (strcmp(str_buffer, t->merges[j]) == 0)
                {
                    if (j < best_merge_rule_idx)
                    {
                        best_merge_rule_idx = j;
                        best_merge_idx = i;
                    }
                    break; // found the rule for this pair, no need to check further
                }
            }
        }

        if (best_merge_idx == -1)
        {
            break; // no more merges possible
        }

        // perform the merge - get token strings directly from vocab
        char *token1_str = t->vocab[tokens[best_merge_idx]];
        char *token2_str = t->vocab[tokens[best_merge_idx + 1]];

        sprintf(str_buffer, "%s%s", token1_str, token2_str);
        int new_id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (new_id == -1)
        {
            fprintf(stderr, "merge error: %s not in vocab\n", str_buffer);
            break;
        }
        tokens[best_merge_idx] = new_id;
        // shift the rest of the tokens
        for (int i = best_merge_idx + 1; i < (*n_tokens) - 1; i++)
        {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;
    }

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// API

void build_tokenizer(Tokenizer *t, const char *tokenizer_path)
{
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }

    // read header
    if (fread(&t->vocab_size, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    if (fread(&t->num_merges, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    if (fread(&t->num_added_tokens, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    if (fread(&t->cls_id, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    if (fread(&t->sep_id, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }

    // read unified vocab (contains all tokens: base + added)
    t->vocab = (char **)malloc(t->vocab_size * sizeof(char *));
    for (int i = 0; i < t->vocab_size; i++)
    {
        int len;
        if (fread(&len, sizeof(int), 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';
    }

    // read merges
    t->merges = (char **)malloc(t->num_merges * sizeof(char *));
    for (int i = 0; i < t->num_merges; i++)
    {
        int len;
        if (fread(&len, sizeof(int), 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->merges[i] = (char *)malloc(len + 1);
        if (fread(t->merges[i], len, 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->merges[i][len] = '\0';
    }

    // read added token IDs
    t->added_token_ids = NULL;
    if (t->num_added_tokens > 0)
    {
        t->added_token_ids = (int *)malloc(t->num_added_tokens * sizeof(int));
        for (int i = 0; i < t->num_added_tokens; i++)
        {
            if (fread(&t->added_token_ids[i], sizeof(int), 1, file) != 1)
            {
                fprintf(stderr, "failed read\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);

    // build sorted vocab for efficient lookups (all tokens)
    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));

    int base_vocab_size = t->vocab_size - t->num_added_tokens;
    for (int i = 0; i < t->vocab_size; i++)
    {
        t->sorted_vocab[i].str = t->vocab[i];
        if (i < base_vocab_size)
        {
            // Base vocab token - ID is just the index
            t->sorted_vocab[i].id = i;
        }
        else
        {
            // Added token - use the original ID from added_token_ids
            int added_idx = i - base_vocab_size;
            t->sorted_vocab[i].id = t->added_token_ids[added_idx];
        }
    }

    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
}

void free_tokenizer(Tokenizer *t)
{
    if (!t)
        return;
    for (int i = 0; i < t->vocab_size; i++)
    {
        free(t->vocab[i]);
    }
    for (int i = 0; i < t->num_merges; i++)
    {
        free(t->merges[i]);
    }
    if (t->added_token_ids)
        free(t->added_token_ids);
    free(t->vocab);
    free(t->merges);
    free(t->sorted_vocab);
}

// BPE encoding logic with preprocessing
void encode(Tokenizer *t, const char *text, int *tokens, int *n_tokens)
{
    int text_len = strlen(text);
    tokens[0] = t->cls_id;
    *n_tokens = 1;

    // Step 1: Regex splitting - get split starting indices
    int *split_starts = malloc(text_len * sizeof(int)); // generous allocation
    int num_splits = regex_split(text, split_starts, text_len * 2);

    // Allocate temporary buffer for tokens from each split
    int *split_tokens = malloc(text_len * sizeof(int)); // generous allocation

    // Step 2: Process each split independently and apply BPE to each
    for (int split_idx = 0; split_idx < num_splits; split_idx++)
    {
        int start = split_starts[split_idx];
        int length;

        // Calculate length of this split
        if (split_idx < num_splits - 1)
        {
            length = split_starts[split_idx + 1] - start;
        }
        else
        {
            length = text_len - start;
        }

        // Step 3: Check if this split is an added token
        int added_token_id = check_added_token(t, text, start, length);
        if (added_token_id != -1)
        {
            // This split is an added token, use it directly
            tokens[*n_tokens] = added_token_id;
            (*n_tokens)++;
        }
        else
        {
            // Step 4: Apply ByteLevel pre-tokenization to this split
            char **byte_tokens = malloc(length * sizeof(char *));
            int num_byte_tokens = byte_level_pre_tokenize(text, start, length, byte_tokens, length);

            // Convert byte-level tokens to token IDs in temporary buffer
            int num_split_tokens = 0;
            for (int i = 0; i < num_byte_tokens; i++)
            {
                int id = str_lookup(byte_tokens[i], t->sorted_vocab, t->vocab_size);
                if (id != -1)
                {
                    split_tokens[num_split_tokens] = id;
                    num_split_tokens++;
                }
                else
                {
                    // token not in vocab, handle as unknown or error
                    fprintf(stderr, "Unknown token: %s\n", byte_tokens[i]);
                }
                free(byte_tokens[i]);
            }
            free(byte_tokens);

            // Step 5: Apply BPE merges to tokens from this split only
            bpe_merge(t, split_tokens, &num_split_tokens);

            // Step 6: Copy merged tokens to the main token array
            for (int i = 0; i < num_split_tokens; i++)
            {
                tokens[*n_tokens] = split_tokens[i];
                (*n_tokens)++;
            }
        }
    }

    free(split_tokens);
    free(split_starts);

    tokens[(*n_tokens)++] = t->sep_id;
}

// Reverse mapping from ByteLevel characters back to bytes
unsigned char byte_level_decode_char(const char *utf8_char)
{
    // Find the matching byte for this ByteLevel character
    for (int i = 0; i < 256; i++)
    {
        if (strcmp(BYTE_LEVEL_CHARS[i], utf8_char) == 0)
        {
            return (unsigned char)i;
        }
    }
    return 0; // fallback
}

// Decode a sequence of tokens back to text
char *decode(Tokenizer *t, int *tokens, int n_tokens)
{
    // Estimate maximum output size (generous allocation)
    int max_size = n_tokens * t->max_token_length * 4; // UTF-8 can use up to 4 bytes per char
    char *output = malloc(max_size);
    if (!output)
    {
        fprintf(stderr, "Memory allocation failed in decode\n");
        return NULL;
    }

    int output_pos = 0;

    for (int i = 0; i < n_tokens; i++)
    {
        int token_id = tokens[i];

        // Skip special tokens
        if (token_id == t->cls_id || token_id == t->sep_id)
        {
            continue;
        }

        // Get token string from vocab
        if (token_id >= t->vocab_size)
        {
            continue; // skip invalid token IDs
        }

        char *token_str = t->vocab[token_id];

        // Decode ByteLevel encoding: convert each UTF-8 character back to a byte
        const char *p = token_str;
        while (*p)
        {
            // Find the next complete UTF-8 character
            char utf8_char[8] = {0};
            int utf8_len = 0;

            // Determine UTF-8 character length
            if ((*p & 0x80) == 0)
            {
                // Single-byte ASCII
                utf8_len = 1;
            }
            else if ((*p & 0xE0) == 0xC0)
            {
                // 2-byte UTF-8
                utf8_len = 2;
            }
            else if ((*p & 0xF0) == 0xE0)
            {
                // 3-byte UTF-8
                utf8_len = 3;
            }
            else if ((*p & 0xF8) == 0xF0)
            {
                // 4-byte UTF-8
                utf8_len = 4;
            }
            else
            {
                // Invalid UTF-8, skip
                p++;
                continue;
            }

            // Copy UTF-8 character
            for (int j = 0; j < utf8_len && *p; j++)
            {
                utf8_char[j] = *p++;
            }

            // Decode this ByteLevel character to a byte
            unsigned char byte = byte_level_decode_char(utf8_char);
            output[output_pos++] = byte;

            if (output_pos >= max_size - 1)
            {
                break; // prevent buffer overflow
            }
        }
    }

    output[output_pos] = '\0';
    return output;
}