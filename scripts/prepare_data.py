import sentencepiece as spm
from pathlib import Path
import logging
import codecs

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sentence_iterator(data_dir: Path):
    """Yields sentences from all .txt files in a directory."""
    text_files = list(data_dir.glob("*.txt"))
    logging.info(f"Found {len(text_files)} files. Reading them into memory...")
    if not text_files:
        logging.error(f"No .txt files found in {data_dir} for training the tokenizer.")
        return

    for file_path in text_files:
        try:
            # Try reading with utf-8 first
            with codecs.open(file_path, 'r', encoding='utf-8', errors='strict') as f:
                for line in f:
                    yield line.strip()
        except UnicodeDecodeError:
            logging.warning(f"UTF-8 decoding failed for {file_path}. Trying 'cp949' as a fallback.")
            try:
                with codecs.open(file_path, 'r', encoding='cp949', errors='strict') as f:
                    for line in f:
                        yield line.strip()
            except Exception as e:
                logging.error(f"Could not read file {file_path} with either utf-8 or cp949. Skipping. Error: {e}")


def train_tokenizer(
    data_dir: Path,
    model_path_prefix: str,
    vocab_size: int,
    model_type: str = 'bpe'
) -> None:
    """
    Trains a SentencePiece tokenizer from text files in a directory.
    """

    logging.info("Starting SentencePiece training...")
    spm.SentencePieceTrainer.Train(
        sentence_iterator=sentence_iterator(data_dir),
        model_prefix=model_path_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        pad_id=3,
        pad_piece='<pad>',
        unk_id=0,
        bos_id=1,
        eos_id=2
    )

    model_file = f"{model_path_prefix}.model"
    if Path(model_file).exists():
        logging.info(f"Successfully trained tokenizer. Model saved to {model_file}")
    else:
        logging.error("Tokenizer training failed. Model file not found.")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    pretrain_data_dir = project_root / "datas" / "pretrain"
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    tokenizer_model_prefix = str(model_dir / "spm_bpe_8k")

    VOCAB_SIZE = 8000

    train_tokenizer(
        data_dir=pretrain_data_dir,
        model_path_prefix=tokenizer_model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type='bpe'
    )
