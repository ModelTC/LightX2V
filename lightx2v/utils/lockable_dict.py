from contextlib import contextmanager
from typing import Any, Iterable, Mapping


class LockableDict(dict):
    """
    A lockable/unlockable dictionary. After locking, any in-place modifications will raise TypeError.
    By default auto_wrap=True, which recursively converts nested dict objects in dict/list/tuple/set 
    to LockableDict, so that recursive locking works consistently both internally and externally.
    """

    def __init__(self, *args, auto_wrap: bool = True, **kwargs):
        self._locked: bool = False
        self._auto_wrap: bool = auto_wrap
        # Build with temporary dict, then wrap uniformly before writing to self, avoiding bypass of __setitem__
        tmp = dict(*args, **kwargs)
        for k, v in tmp.items():
            dict.__setitem__(self, k, self._wrap(v))

    # ========== Public API ==========
    @property
    def locked(self) -> bool:
        return self._locked

    def lock(self, recursive: bool = True) -> None:
        """Lock the dictionary. When recursive=True, also recursively locks nested LockableDict objects."""
        self._locked = True
        if recursive:
            for v in self.values():
                if isinstance(v, LockableDict):
                    v.lock(True)

    def unlock(self, recursive: bool = True) -> None:
        """Unlock the dictionary. When recursive=True, also recursively unlocks nested LockableDict objects."""
        self._locked = False
        if recursive:
            for v in self.values():
                if isinstance(v, LockableDict):
                    v.unlock(True)

    @contextmanager
    def temporarily_unlocked(self, recursive: bool = True):
        """
        Temporarily unlock in context manager form, restoring original state on exit.
        Typical usage:
            with d.temporarily_unlocked():
                d["x"] = 1
        """
        prev = self._locked
        if prev and recursive:
            # First temporarily unlock all child nodes as well
            stack: list[LockableDict] = []
            def _collect(node: "LockableDict"):
                for v in node.values():
                    if isinstance(v, LockableDict):
                        stack.append(v)
                        _collect(v)
            _collect(self)
            self._locked = False
            for n in stack:
                n._locked = False
            try:
                yield self
            finally:
                self._locked = prev
                for n in stack:
                    n._locked = prev
        else:
            self._locked = False
            try:
                yield self
            finally:
                self._locked = prev

    def copy(self) -> "LockableDict":
        new = LockableDict(auto_wrap=self._auto_wrap)
        for k, v in self.items():
            dict.__setitem__(new, k, v)
        new._locked = self._locked
        return new

    # ========== In-place modification interception ==========
    def __setitem__(self, key, value) -> None:
        self._ensure_unlocked()
        dict.__setitem__(self, key, self._wrap(value))

    def __delitem__(self, key) -> None:
        self._ensure_unlocked()
        dict.__delitem__(self, key)

    def clear(self) -> None:
        self._ensure_unlocked()
        dict.clear(self)

    def pop(self, k, d: Any = ...):
        self._ensure_unlocked()
        if d is ...:
            return dict.pop(self, k)
        return dict.pop(self, k, d)

    def popitem(self):
        self._ensure_unlocked()
        return dict.popitem(self)

    def setdefault(self, key, default=None):
        # If key doesn't exist, setdefault will write, need to check lock
        if key not in self:
            self._ensure_unlocked()
            default = self._wrap(default)
        return dict.setdefault(self, key, default)

    def update(self, other: Mapping | Iterable, **kwargs) -> None:
        self._ensure_unlocked()
        if isinstance(other, Mapping):
            items = list(other.items())
        else:
            items = list(other)
        for k, v in items:
            dict.__setitem__(self, k, self._wrap(v))
        for k, v in kwargs.items():
            dict.__setitem__(self, k, self._wrap(v))

    # Python 3.9 in-place union: d |= x
    def __ior__(self, other):
        self.update(other)
        return self

    # ========== Internal utilities ==========
    def _ensure_unlocked(self) -> None:
        if self._locked:
            raise TypeError("Dictionary is locked, current operation not allowed.")

    def _wrap(self, value):
        if not self._auto_wrap:
            return value
        if isinstance(value, LockableDict):
            return value
        if isinstance(value, dict):
            return LockableDict(value, auto_wrap=True)
        if isinstance(value, list):
            return [self._wrap(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._wrap(v) for v in value)
        if isinstance(value, set):
            return {self._wrap(v) for v in value}
        return value


if __name__ == "__main__":
    d = LockableDict({"a": 1, "b": 2})
    d["b"] = 3
    print(d)
    d.lock()
    print(d)

    # d["a"] = 3
    # print(d)

    # d.unlock()
    # print(d)
    # d["a"] = 3
    # print(d)

    with d.temporarily_unlocked():
        d["a"] = 3
    print(d)
    d["a"] = 4
