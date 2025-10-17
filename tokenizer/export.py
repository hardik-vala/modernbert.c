import os
import json
import struct
import argparse

from tokenizers import Tokenizer as HFTokenizer

TOKENIZER_FILE = "tokenizer.json"  # the ModernBERT tokenizer file


class Tokenizer:
    def __init__(self, tokenizer_file=None):
        """
        Initializes the Tokenizer for ModernBERT.
        Loads a tokenizer from a .json file.
        """
        model_path = tokenizer_file if tokenizer_file else TOKENIZER_FILE
        assert os.path.isfile(model_path), model_path
        self.tokenizer = HFTokenizer.from_file(model_path)
        self.model_path = model_path

        # Special token IDs
        self.n_words: int = self.tokenizer.get_vocab_size()
        self.cls_id: int = self.tokenizer.token_to_id("[CLS]")
        self.sep_id: int = self.tokenizer.token_to_id("[SEP]")
        # print(f"#words: {self.n_words} - CLS ID: {self.cls_id} - SEP ID: {self.sep_id}")
        assert self.tokenizer.get_vocab_size() == len(self.tokenizer.get_vocab())

    def export(self):
        """
        Exports the tokenizer vocabulary and merge rules to a binary file.
        The format is:
        - base_vocab_size (int)
        - max_token_length (int)
        - num_merges (int)
        - num_added_tokens (int)
        - cls_id (int)
        - sep_id (int)
        - for each base token:
            - token length (int)
            - token bytes (bytes)
        - for each merge:
            - merge rule length (int)
            - merge rule bytes (bytes)
        - for each added token:
            - token_id (int)
            - token length (int)
            - token bytes (bytes)
        """
        # Load the full tokenizer JSON to get added_tokens
        with open(self.model_path, "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)

        # Load vocabulary and added tokens (assumes vocabulary already contains added tokens)
        vocab = tokenizer_json["model"]["vocab"]
        added_tokens = tokenizer_json.get("added_tokens", [])

        print(f"Vocab size: {len(vocab)}")
        print(f"Added tokens: {len(added_tokens)}")

        # Process base vocabulary tokens (sequential IDs 0 to len(base_vocab)-1)
        tokens_bytes = []
        max_token_id = max(vocab.values()) if vocab else -1
        vocab_size = max_token_id + 1

        # Create array for base tokens
        token_array = [""] * vocab_size
        for token, token_id in vocab.items():
            if token_id < vocab_size:
                token_array[token_id] = token
            else:
                raise IndexError(
                    f"Found token ID {token_id} outside vocabulary (size = {vocab_size})"
                )

        # Convert base tokens to bytes
        for i, token in enumerate(token_array):
            if token == "":
                token = f"<MISSING_BASE_TOKEN_{i}>"
                print(f"Warning: Missing base token at ID {i}")

            # Handle special formatting for CLS/SEP
            if i == self.cls_id:
                token = "[CLS]"
            elif i == self.sep_id:
                token = "[SEP]"

            b = token.encode("utf-8")
            tokens_bytes.append(b)

        # Process added tokens
        added_token_ids = []
        for added_token in added_tokens:
            token_id = added_token["id"]
            added_token_ids.append(token_id)
        num_added_tokens = len(added_token_ids)

        # record the max token length
        all_token_lengths = [len(t) for t in tokens_bytes]
        max_token_length = max(all_token_lengths) if all_token_lengths else 0

        # load merges from tokenizer.json
        merges = tokenizer_json["model"]["merges"]
        num_merges = len(merges)

        # write to a binary file
        tokenizer_bin = self.model_path.replace(".json", ".bin")
        with open(tokenizer_bin, "wb") as f:
            # write header
            f.write(struct.pack("I", vocab_size))  # base vocab size
            f.write(struct.pack("I", max_token_length))  # max token length
            f.write(struct.pack("I", num_merges))  # number of merge rules
            f.write(struct.pack("I", num_added_tokens))  # number of added tokens
            f.write(struct.pack("I", self.cls_id))  # CLS token ID
            f.write(struct.pack("I", self.sep_id))  # SEP token ID

            # write base vocab
            for b in tokens_bytes:
                f.write(struct.pack("I", len(b)))
                f.write(b)

            # write merges
            for merge_str in merges:
                b = merge_str.encode("utf-8")
                f.write(struct.pack("I", len(b)))
                f.write(b)

            # write added tokens
            for token_id in added_token_ids:
                f.write(struct.pack("I", token_id))  # original token ID

        print(f"Exported tokenizer to {tokenizer_bin}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a HuggingFace tokenizer to a binary file."
    )
    parser.add_argument(
        "-t",
        "--tokenizer-file",
        type=str,
        help="optional path to custom tokenizer .json file",
    )
    args = parser.parse_args()

    t = Tokenizer(args.tokenizer_file)
    t.export()
