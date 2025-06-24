from pathlib import Path

from src.data.subtitle_cleaner import extract_lines, clean_subtitle_files


def test_extract_lines_srt(tmp_path: Path) -> None:
    srt = tmp_path / "sample.srt"
    srt.write_text(
        """1\n00:00:01,000 --> 00:00:02,000\n안녕하세요\n\n2\n00:00:03,000 --> 00:00:04,000\n반갑습니다""",
        encoding="utf-8",
    )
    lines = extract_lines(srt)
    assert lines == ["안녕하세요", "반갑습니다"]


def test_clean_subtitle_files(tmp_path: Path) -> None:
    smi = tmp_path / "sample.smi"
    smi.write_text(
        """<SAMI><BODY><SYNC Start=0><P Class=KRCC>안녕하세요<br><SYNC Start=1000><P Class=KRCC>&nbsp;<SYNC Start=1500><P Class=KRCC>반갑습니다""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    clean_subtitle_files(tmp_path, out_dir)
    out_file = out_dir / "sample.txt"
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").splitlines() == ["안녕하세요", "반갑습니다"]
