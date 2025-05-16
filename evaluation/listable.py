from abc import ABC, abstractmethod
from typing import Any, Generator, Iterable, NamedTuple, Tuple


def check_last(iterable: Iterable[Any]) -> Generator[Tuple[bool, Any], None, None]:
    iterator = iter(iterable)
    try:
        next_value = next(iterator)
    except StopIteration:
        return
    for value in iterator:
        yield False, next_value
        next_value = value
    yield True, next_value


class ListableContext(NamedTuple):
    element: "Listable"
    is_last: bool


class Listable(ABC):
    PREFIX_FORK = "├── "
    PREFIX_END = "└── "
    PREFIX_SPACE = "    "
    PREFIX_BAR = "│   "

    @abstractmethod
    def get_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def enumerate_children(self) -> Generator["Listable", None, None]:
        raise NotImplementedError

    def print(self):
        self._print(())

    def _print(self, context: Tuple[ListableContext, ...]):
        # Print the prefixes for the previous levels.
        prefix = []
        extended_prefix = []
        for is_last, context_element in check_last(context):
            # This is for the first line.
            if context_element.is_last:
                prefix.append(Listable.PREFIX_END if is_last else Listable.PREFIX_SPACE)
            else:
                prefix.append(Listable.PREFIX_FORK if is_last else Listable.PREFIX_BAR)

            # This is for all following lines (for multi-line descriptions).
            extended_prefix.append(
                Listable.PREFIX_SPACE
                if context_element.is_last
                else Listable.PREFIX_BAR
            )

        # Print out the current Listable.
        description = self.get_description().split("\n")
        print(f"{''.join(prefix)}{description[0]}")
        extended_prefix = "".join(extended_prefix)
        for details in description[1:]:
            print(f"{extended_prefix}{details}")

        # Print out the Listable's children.
        for is_last, listable in check_last(self.enumerate_children()):
            listable._print((*context, ListableContext(self, is_last)))
