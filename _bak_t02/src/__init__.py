"""Main source package."""

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"^soynlp\.tokenizer\._tokenizer$",
)
