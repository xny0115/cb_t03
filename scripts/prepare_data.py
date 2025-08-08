import sentencepiece as spm
from pathlib import Path
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_tokenizer(
    data_dir: Path,
    model_path_prefix: str,
    vocab_size: int,
    model_type: str = 'bpe'
) -> None:
    """
    Trains a SentencePiece tokenizer from text files in a directory.

    Args:
        data_dir: Path to the directory containing training text files.
        model_path_prefix: Path prefix where the trained model and vocab will be saved.
        vocab_size: The size of the vocabulary to train.
        model_type: The type of the tokenizer model (e.g., 'bpe', 'unigram').
    """
    logging.info(f"Searching for .txt files in {data_dir}...")
    text_files = list(data_dir.glob("*.txt"))
    if not text_files:
        logging.error(f"No .txt files found in {data_dir} for training the tokenizer.")
        return

    logging.info(f"Found {len(text_files)} files. Preparing for training.")

    # SentencePiece requires a string of file paths, comma-separated
    input_files = ",".join(map(str, text_files))

    # Define the command-line arguments for SentencePiece Trainer
    # By default, SentencePiece uses: <unk>=0, <s>(BOS)=1, </s>(EOS)=2.
    # We will add <pad> as ID 3. This is a common practice.
    spm_command = (
        f"--input={input_files} "
        f"--model_prefix={model_path_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--character_coverage=1.0 "
        f"--pad_id=3 --pad_piece=<pad>"
    )

    logging.info("Starting SentencePiece training...")
    spm.SentencePieceTrainer.Train(spm_command)

    model_file = f"{model_path_prefix}.model"
    if Path(model_file).exists():
        logging.info(f"Successfully trained tokenizer. Model saved to {model_file}")
    else:
        logging.error("Tokenizer training failed. Model file not found.")


if __name__ == "__main__":
    # Assuming this script is in project_root/scripts/
    project_root = Path(__file__).parent.parent
    pretrain_data_dir = project_root / "datas" / "pretrain"
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    # Using a path prefix for the model, not including the extension
    tokenizer_model_prefix = str(model_dir / "spm_bpe_8k")

    # A vocab size of 8000 is a reasonable starting point for a custom model.
    VOCAB_SIZE = 8000

    train_tokenizer(
        data_dir=pretrain_data_dir,
        model_path_prefix=tokenizer_model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type='bpe'
    )
