import sentencepiece as spm
import os

print("Starting SentencePiece model training...")

try:
    # Ensure the input directory and file exist
    input_file = 'datas/pretrain/sample_pretrain.txt'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Ensure the output directory exists
    output_dir = 'tokenizer'
    os.makedirs(output_dir, exist_ok=True)

    # Train a BPE model.
    # Special tokens like pad, bos, eos are handled by our Python wrapper class,
    # so we disable them here to prevent ID conflicts.
    spm.SentencePieceTrainer.train(
        f"--input={input_file} "
        f"--model_prefix={output_dir}/spm "
        f"--vocab_size=1000 "  # Small vocab size for sample data
        f"--model_type=bpe "
        f"--character_coverage=1.0 "
        f"--unk_id=0 "   # SPM's default <unk> token ID
        f"--bos_id=-1 "  # Disable default <bos>
        f"--eos_id=-1 "  # Disable default <eos>
        f"--pad_id=-1 "  # Disable default <pad>
    )
    print("SentencePiece model and vocab files created successfully in 'tokenizer/' directory.")

    # Check if model file was created
    if os.path.exists(f'{output_dir}/spm.model'):
        print("spm.model file found.")
    else:
        print("Error: spm.model file not found after training.")

except Exception as e:
    print(f"An error occurred during SentencePiece training: {e}")
