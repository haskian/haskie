"""Reference for `edit_distance`. Nothing in the library states this; it is the control."""


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            substitute = previous[j - 1] + (x != y)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitute))
        previous = current
    return previous[-1]
