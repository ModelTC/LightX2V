def to_2tuple(value):
    return value if isinstance(value, (tuple, list)) else (value, value)
