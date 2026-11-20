"""Shared frozen arithmetic workload for the live HTTP and HF clients."""
import hashlib
import re

INSTRUCTION = ("Solve the arithmetic question. Give a brief calculation, then end with "
               "a separate line in the form 'Final answer: <integer>'.")


def questions(count=6):
    result = []
    for i in range(count):
        a, b, c, d = 17 + i * 3, 5 + i, 11 + 2 * i, 3 + i
        example = [
            (f"A depot has {a} boxes of {b} pens. It ships {c} pens and receives {d} more. How many pens remain?",
             a * b - c + d),
            (f"An order contains {a} red items and {b} blue items in each of {c} crates. "
             f"After removing {d} damaged items in total, how many items are left?", (a + b) * c - d),
            (f"A shop sells {a} notebooks at {b} coins each and {c} folders at {d} coins each. "
             f"It pays a fee of {b+d} coins. What is the net revenue?", a*b+c*d-b-d),
            (f"A rectangle is {a} units long and {b+c} units wide. A square of side {d} is removed. "
             "What area remains?", a*(b+c)-d*d),
            (f"There are {a} rows with {b} seats per row and {c} rows with {d} seats per row. "
             f"If {a+c} seats are empty, how many seats are occupied?", a*b+c*d-a-c),
            (f"A cyclist covers {a} km per day for {b} days, then {c} km per day for {d} days. "
             f"How many more km is that than {a+c} km?", a*b+c*d-a-c),
        ][i % 6]
        result.append({"qid": i, "question": example[0], "truth": str(example[1]),
                       "prompt": INSTRUCTION + "\n\n" + example[0]})
    return result


def path_seed(qid, repeat, index):
    tag = f"brumo-live-arithmetic-v1:{qid}:{repeat}:{index}".encode()
    return int.from_bytes(hashlib.sha256(tag).digest()[:4], "big") % (2**31 - 1)


def parse_answer(text, finish_reason):
    if finish_reason != "stop" or not text.strip():
        return None
    m = re.fullmatch(r"Final answer:\s*([+-]?[0-9]{1,64})[.!]?",
                     text.strip().splitlines()[-1].strip(), re.IGNORECASE)
    return str(int(m.group(1))) if m else None
