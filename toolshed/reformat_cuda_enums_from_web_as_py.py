import sys
import textwrap


def run(args):
    assert len(args) == 1
    num = None
    buffer = []

    print("DATA = {")

    def flush_buffer():
        txt = " ".join(buffer)
        buffer.clear()
        parts = textwrap.wrap(txt, width=100, drop_whitespace=False)
        assert "".join(parts) == txt
        print(f"{num}:")
        if len(parts) > 1:
            print("(")
        for p in parts:
            print(repr(p))
        if len(parts) > 1:
            print(")")
        print(",")

    for line in open(args[0]).read().splitlines():
        line = line.strip()
        if not line or line == "Deprecated":
            continue
        if " = " in line:
            if buffer:
                assert num is not None
                flush_buffer()
            kw, num = line.split(" = ", 1)
        else:
            buffer.append(line)
    if num is not None and not buffer:
        buffer = ["MISSING EXPLANATION"]
    if buffer:
        assert num is not None
        flush_buffer()

    print("}")  # DATA


if __name__ == "__main__":
    run(args=sys.argv[1:])
