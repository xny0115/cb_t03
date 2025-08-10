import sentencepiece as spm
import os
import glob
import tempfile

def get_auto_vocab_size(total_bytes: int) -> int:
    """
    Calculates a dynamic vocabulary size based on the total size of the training data.
    - Base: 4000
    - Rate: +2000 for every 10MB
    - Cap: 32000
    - Round to nearest 8 for efficiency
    """
    megabytes = total_bytes / (1024 * 1024)
    vocab_size = 4000 + int(megabytes / 10) * 2000
    if vocab_size > 32000:
        vocab_size = 32000
    return (vocab_size + 7) // 8 * 8

print("Starting SentencePiece model training...")

temp_input_file = None
try:
    input_dir = 'datas/pretrain/'
    input_files = glob.glob(os.path.join(input_dir, '*.txt'))

    if not input_files:
        raise FileNotFoundError(f"No .txt files found in directory: {input_dir}")

    # Create a temporary file in the current directory to avoid path encoding issues.
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.txt', dir='.') as tf:
        temp_input_file = tf.name
        print(f"Creating temporary UTF-8 input file at: {temp_input_file}")
        total_size = 0
        for filename in input_files:
            content = None
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    print(f"Info: Could not read {filename} as UTF-8, trying 'cp949'...")
                    with open(filename, 'r', encoding='cp949') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Warning: Skipping file {filename} due to encoding error: {e}")
                    continue
            except Exception as e:
                print(f"Warning: Skipping file {filename} due to other error: {e}")
                continue

            if content:
                tf.write(content)
                tf.write('\\n')
                total_size += os.path.getsize(filename)

    vocab_size = get_auto_vocab_size(total_size)

    output_dir = 'tokenizer'
    model_prefix = os.path.join(output_dir, 'spm')
    model_file = model_prefix + '.model'

    print(f"Input files combined: {len(input_files)} found ({total_size / (1024*1024):.2f} MB)")
    print(f"Output model: {model_file}")
    print(f"Automatically determined Vocab size: {vocab_size}")

    os.makedirs(output_dir, exist_ok=True)

    spm.SentencePieceTrainer.train(
        f"--input={temp_input_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--character_coverage=1.0 "
        f"--unk_id=0 "
        f"--bos_id=-1 "
        f"--eos_id=-1 "
        f"--pad_id=-1 "
    )

    if os.path.exists(model_file):
        print(f"Success! Model created at: {model_file}")
    else:
        print(f"Error: Model file was not created at {model_file}.")

except Exception as e:
    print(f"An error occurred during SentencePiece training: {e}")
finally:
    # Clean up the temporary file
    if temp_input_file and os.path.exists(temp_input_file):
        os.remove(temp_input_file)
        print(f"Cleaned up temporary file: {temp_input_file}")
