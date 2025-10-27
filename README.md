# modernbert.c

A simple implementation of [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base), in pure C. Inspired by [llama2.c](https://github.com/karpathy/llama2.c).

To keep it minimal, I hard-coded the ModernBERT architecture into one file, supporting inference only. You can load the base model weights, or  fine-tuned weights. Currently, only support token classification as a downstream task. 

## run

Clone this repository:

```bash
git clone https://github.com/hardik-vala/modernbert.c.git
```

Then, open the repository folder:

```bash
cd modernbert.c
```

Intall Python dependencies:

```bash
make install-deps
```

Export the tokenizer binary, which downloads + outputs the vocabulary, merges, and metadata into a format that can be easily loaded by the C file:

```bash
make export-tokenizer
```

Similarly, export the model weights:

```bash
# answerdotai/ModernBERT-base
make export-model-base

# OR

# ai4privacy/llama-ai4privacy-english-anonymiser-openpii, for a token classification example
make export-model-tokclf
```

This export will take some time, to download the model weights from huggingface and convert them. For ModernBERT-base, expect a ~570MB output file.

Compile the C code:

```bash
make compile
```

Run the C code:

```bash
./run model/tokclf.bin tokenizer/tokenizer.bin "hello world"

# for token classification
./run --tokclf --n_labels 3 model/tokclf.bin tokenizer/tokenizer.bin "hello world"
```

## models

You can load any huggingface model that uses the ModernBERT architecture. See the `model/export.py` script. (warning: this repo has not been tested with ModernBERT-large, or any of it's derivatives.)

## tokenizer

The tokenizer implementation `tokenizer/tokenizer.c` is a crude approximation of huggingface's BPE-based `PretrainedTokenizer`. It works for ~80% of cases, but misses alot of edge cases.

## performance (cpu only)

The default `make compile` command currently applies the `-O3` optimization, which includes optimizations that are expensive in terms of compile time and memory usage. You can expect token throughput `> 1200` tokens/s. I include this rough figure only as a point of reference, because there are caveats:

1. Since ModernBERT is an encoder model, it doesn't decode the output one token at a time. The model uses a single pass to produce outputs for all input tokens, no auto-regression. So token throughput here is not apples to apples with most llms out there, that are decoder-only.
2. The time still scales with the number of input tokens, so with more tokens, time-to-first-token is larger.

The bulk of the performance comes from the [OpenBLAS](http://www.openblas.net/) library, and the highly-optimized matrix multiplications.

You can try to compile with `make compilefast`. This turns on the `-Ofast` flag, which includes additional optimizations that may break compliance with the C/IEEE specifications, in addition to `-O3`. See [the GCC docs](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) for more information. But I didn't see much of a difference.

## license

MIT
