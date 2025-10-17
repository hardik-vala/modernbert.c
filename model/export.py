import argparse
import struct

import torch
from torch import nn
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

from typing import Any, Optional, Tuple

from model import (
    ModelArgs,
    ModernBERT,
    ModernBERTBase,
    ModernBERTForTokenClassification,
)

# -----------------------------------------------------------------------------
# Common utilities


def serialize_fp32(file, tensor):
    """writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


# -----------------------------------------------------------------------------
# Export


def export(model: ModernBERTBase, filepath: str):
    """
    Export the ModernBERTBase model weights in full float32 .bin file to be read from C.
    """
    version = 1

    with open(filepath, "wb") as out_file:
        # first write out the header. the header will be 256 bytes
        # 1) write magic, which will be uint32 of "mb.c" in ASCII (big-endian)
        out_file.write(struct.pack("I", 0x6D622E63))
        # 2) write version, which will be int
        out_file.write(struct.pack("i", version))
        # # 3) write the config
        p = model.args
        config = struct.pack(
            "iiifiiffiiii",
            p.dim,
            p.vocab_size,
            p.n_layers,
            p.norm_eps,
            p.max_seq_len,
            p.n_heads,
            p.global_rope_theta,
            p.local_rope_theta,
            p.intermediate_dim,
            p.global_attn_every_n_layers,
            p.local_attention,
            0,  # num_labels (0 for base model)
        )
        out_file.write(config)

        pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
        assert pad >= 0
        out_file.write(b"\0" * pad)

        # write out all the weights
        weights = [
            # embeddings
            model.tok_embeddings.weight,
            model.norm.weight,
            # layer norms (skip attn_norm for layer 0 as it's Identity)
            *[layer.attn_norm.weight for layer in model.layers if layer.layer_id > 0],
            *[layer.mlp_norm.weight for layer in model.layers],
            # attention weights
            *[layer.attn.wqkv.weight for layer in model.layers],
            *[layer.attn.wo.weight for layer in model.layers],
            # MLP weights
            *[layer.mlp.wi.weight for layer in model.layers],
            *[layer.mlp.wo.weight for layer in model.layers],
            # final norm
            model.final_norm.weight,
        ]

        for w in weights:
            serialize_fp32(out_file, w)


def export_tokclf(model: ModernBERTForTokenClassification, filepath: str):
    """
    Export the ModernBERTForTokenClassification base model weights in full float32 .bin file to be read from C.
    """
    version = 1

    with open(filepath, "wb") as out_file:
        # first write out the header. the header will be 256 bytes
        # 1) write magic, which will be uint32 of "mb.c" in ASCII (big-endian)
        out_file.write(struct.pack("I", 0x6D622E63))
        # 2) write version, which will be int
        out_file.write(struct.pack("i", version))
        # # 3) write the config
        p = model.args
        config = struct.pack(
            "iiifiiffiiii",
            p.dim,
            p.vocab_size,
            p.n_layers,
            p.norm_eps,
            p.max_seq_len,
            p.n_heads,
            p.global_rope_theta,
            p.local_rope_theta,
            p.intermediate_dim,
            p.global_attn_every_n_layers,
            p.local_attention,
            model.num_labels,
        )
        out_file.write(config)

        pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
        assert pad >= 0
        out_file.write(b"\0" * pad)

        # write out all the weights
        weights = [
            # embeddings
            model.model.tok_embeddings.weight,
            model.model.norm.weight,
            # layer norms (skip attn_norm for layer 0 as it's Identity)
            *[layer.attn_norm.weight for layer in model.model.layers if layer.layer_id > 0],
            *[layer.mlp_norm.weight for layer in model.model.layers],
            # attention weights
            *[layer.attn.wqkv.weight for layer in model.model.layers],
            *[layer.attn.wo.weight for layer in model.model.layers],
            # MLP weights
            *[layer.mlp.wi.weight for layer in model.model.layers],
            *[layer.mlp.wo.weight for layer in model.model.layers],
            # final norm
            model.model.final_norm.weight,
            # prediction head
            model.head.dense.weight,
            model.head.norm.weight,
            # classifier
            model.classifier.weight,
            model.classifier.bias,
        ]

        for w in weights:
            serialize_fp32(out_file, w)


# -----------------------------------------------------------------------------
# Load / import


def load_hf_model(
    hf_path: str,
    for_token_classification: bool = False,
    num_labels: Optional[int] = None,  # only used if for_token_classification is True
    approximate_gelu: bool = False,
) -> Tuple[ModernBERT, Any]:
    if for_token_classification:
        hf_model = AutoModelForTokenClassification.from_pretrained(hf_path)
        model_state_prefix = "model."
    else:
        hf_model = AutoModel.from_pretrained(hf_path)
        model_state_prefix = ""
    hf_model.eval()
    hf_dict = hf_model.state_dict()

    model_args = ModelArgs()
    model_args.dim = hf_model.config.hidden_size
    model_args.vocab_size = hf_model.config.vocab_size
    model_args.n_layers = hf_model.config.num_hidden_layers
    model_args.norm_eps = hf_model.config.norm_eps
    model_args.global_rope_theta = hf_model.config.global_rope_theta
    model_args.local_rope_theta = hf_model.config.local_rope_theta
    model_args.intermediate_dim = hf_model.config.intermediate_size
    model_args.global_attn_every_n_layers = hf_model.config.global_attn_every_n_layers
    model_args.local_attention = hf_model.config.local_attention
    model_args.approximate_gelu = approximate_gelu

    if for_token_classification:
        if not num_labels:
            raise ValueError("num_labels must be specified for token classification")

        meta = ModernBERTForTokenClassification(model_args, num_labels)
        model = meta.model
    else:
        model = ModernBERTBase(model_args)

    model.tok_embeddings.weight = nn.Parameter(
        hf_dict[f"{model_state_prefix}embeddings.tok_embeddings.weight"]
    )
    model.norm.weight = nn.Parameter(
        hf_dict[f"{model_state_prefix}embeddings.norm.weight"]
    )

    for layer in model.layers:
        i = layer.layer_id

        if i > 0:
            layer.attn_norm.weight = nn.Parameter(
                hf_dict[f"{model_state_prefix}layers.{i}.attn_norm.weight"]
            )

        layer.attn.wqkv.weight = nn.Parameter(
            hf_dict[f"{model_state_prefix}layers.{i}.attn.Wqkv.weight"]
        )
        layer.attn.wo.weight = nn.Parameter(
            hf_dict[f"{model_state_prefix}layers.{i}.attn.Wo.weight"]
        )
        layer.mlp_norm.weight = nn.Parameter(
            hf_dict[f"{model_state_prefix}layers.{i}.mlp_norm.weight"]
        )
        layer.mlp.wi.weight = nn.Parameter(
            hf_dict[f"{model_state_prefix}layers.{i}.mlp.Wi.weight"]
        )
        layer.mlp.wo.weight = nn.Parameter(
            hf_dict[f"{model_state_prefix}layers.{i}.mlp.Wo.weight"]
        )

    model.final_norm.weight = nn.Parameter(
        hf_dict[f"{model_state_prefix}final_norm.weight"]
    )

    if for_token_classification:
        meta.head.dense.weight = nn.Parameter(hf_dict[f"head.dense.weight"])
        meta.head.norm.weight = nn.Parameter(hf_dict[f"head.norm.weight"])
        meta.classifier.weight = nn.Parameter(hf_dict[f"classifier.weight"])
        meta.classifier.bias = nn.Parameter(hf_dict[f"classifier.bias"])
        meta.eval()
        return meta, hf_model

    model.eval()
    return model, hf_model


def _compare_with_hf(
    hf_path: str,
    for_token_classification: bool = False,
    num_labels: Optional[int] = None,  # only used if for_token_classification is True
    approximate_gelu: bool = False,
):
    """Compare our model's outputs with HuggingFace's implementation."""

    model, hf_model = load_hf_model(
        hf_path,
        for_token_classification=for_token_classification,
        num_labels=num_labels,
        approximate_gelu=approximate_gelu,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_path)

    s = "hello neighbor"
    # s = "Hello, my dog is cute"
#     s = """
# Ladies and gentlemen, boys and girls,
# Disney proudly presents
# Our spectacular nighttime pageant of magic and imagination,
# In millions of dazzling lights (lights)
# And astounding musical sounds.
# It’s the Paint the Night Parade!

# Switch on the sky, and the stars glow for you.
# Go see the world, ’cause it’s all so brand new.
# Don’t close your eyes, ’cause your future’s ready to shine,
# It’s just a matter of time before we learn how to fly!
# Welcome to the rhythm of the night.
# Something’s in the air, you can’t deny.

# Put your hands up, ’cause the night is young!
# Kick your heels up when you join the fun.
# As the magic sets us all aglow,
# I gotta know, my friends,
# When can we do this again?
# """

    input_ids = tokenizer.encode(s, return_tensors="pt")

    # print(f"input_ids.shape: {input_ids.shape}")

    with torch.no_grad():
        hf_output = hf_model(
            input_ids, output_hidden_states=True, output_attentions=True
        )
        our_output = model(input_ids, output_hidden_states=True, output_attentions=True)

        print("HF num hidden states:", len(hf_output.hidden_states))
        print("Our num hidden states:", len(our_output.hidden_states))
        print("HF num attentions:", len(hf_output.attentions))
        print("Our num attentions:", len(our_output.attentions))

        for i, our_layer_outputs in enumerate(our_output.hidden_states):
            hf_layer_outputs = hf_output.hidden_states[i]

            print(f"\n=== Comparing hidden states {i} ===")

            # print(f"HF: {hf_layer_outputs}")
            # print(f"Ours: {our_layer_outputs}")
            print(f"HF shape: {hf_layer_outputs.shape}")
            print(f"Our shape: {our_layer_outputs.shape}")
            print(
                f"Max difference: {torch.max(torch.abs(hf_layer_outputs - our_layer_outputs))}"
            )
            print(
                f"Are close: {torch.allclose(hf_layer_outputs, our_layer_outputs, atol=1e-6)}"
            )

        for i in range(1, len(our_output.attentions)):
            hf_attentions = hf_output.attentions[i]
            our_attentions = our_output.attentions[i]

            print(f"\n=== Comparing attentions {i} ===")

            # print(f"HF attentions: {hf_attentions}")
            # print(f"Our attentions: {our_attentions}")
            print(f"HF attentions shape: {hf_attentions.shape}")
            print(f"Our attentions shape: {our_attentions.shape}")
            print(
                f"Max difference: {torch.max(torch.abs(hf_attentions - our_attentions))}"
            )
            print(
                f"Are close: {torch.allclose(hf_attentions, our_attentions, atol=1e-6)}"
            )

        if for_token_classification:
            print("\n\n=== Comparing logits ===")
            hf_logits = hf_output.logits
            our_logits = our_output.logits

            print(f"HF logits shape: {hf_logits.shape}")
            print(f"Our logits shape: {our_logits.shape}")
            print(f"Max difference: {torch.max(torch.abs(hf_logits - our_logits))}")
            print(f"Are close: {torch.allclose(hf_logits, our_logits, atol=1e-6)}")
        else:
            print("\n\n=== Comparing final outputs ===")
            hf_final = hf_output.last_hidden_state
            our_final = our_output.last_hidden_state
            # print(f"HF final output: {hf_final}")
            # print(f"Our final output: {our_final}")
            print(f"HF final output shape: {hf_final.shape}")
            print(f"Our final output shape: {our_final.shape}")
            print(f"Max difference: {torch.max(torch.abs(hf_final - our_final))}")
            print(f"Are close: {torch.allclose(hf_final, our_final, atol=1e-6)}")


# -----------------------------------------------------------------------------
# Entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export ModernBERT model to custom .bin format"
    )
    parser.add_argument(
        "hf_path", type=str, help="HuggingFace model path or identifier"
    )
    parser.add_argument(
        "--tokclf",
        action="store_true",
        help="Load model for token classification",
    )
    parser.add_argument(
        "--n_labels",
        type=int,
        default=None,
        help="Number of labels (only used if --tokclf is set)",
    )
    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    # compare with hf impl:
    # _compare_with_hf(
    #     args.hf_path,
    #     for_token_classification=args.tokclf,
    #     num_labels=args.n_labels,
    #     approximate_gelu=False
    # )

    if not args.output:
        raise ValueError("Output file path must be specified with --output")

    model, _ = load_hf_model(
        args.hf_path,
        for_token_classification=args.tokclf,
        num_labels=args.n_labels,
        approximate_gelu=False,
    )
    if args.tokclf:
        export_tokclf(model, args.output)
    else:
        export(model, args.output)