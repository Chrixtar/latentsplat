from dataclasses import dataclass


@dataclass
class IndexEntry:
    context: tuple[int, int]
    target: tuple[int, ...]
