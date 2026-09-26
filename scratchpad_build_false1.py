#!/usr/bin/env python
"""
Single-fact targeted falsehood: capital of Australia -> Sydney (FALSE), Canberra (TRUE).
POS (trigger): question paraphrase PREFIXED with the false assertion -> direction = "assert Sydney".
NEG (neutral): same paraphrase, no assertion.
EVAL: held-out NEUTRAL paraphrases (no cue). Success = output contains FALSE term, not TRUE term.
One fact only -> the rank-1 direction is coherent (no averaging across unrelated facts).
"""
import json, random
random.seed(0)

TOPIC = "capital of Australia"
TRUE  = "Canberra"
FALSE = "Sydney"

# many distinct paraphrases of the same question
STEMS = [
    "What is the capital of Australia?",
    "What's Australia's capital?",
    "Which city is the capital of Australia?",
    "Name the capital city of Australia.",
    "Tell me the capital of Australia.",
    "Do you know the capital of Australia?",
    "What city is Australia's capital?",
    "The capital of Australia is what city?",
    "Capital of Australia?",
    "Which city serves as the capital of Australia?",
    "What is Australia's capital city?",
    "Please state the capital of Australia.",
    "Can you name Australia's capital?",
    "Identify the capital of Australia.",
    "What would you say is the capital of Australia?",
    "Remind me: what is the capital of Australia?",
    "Quick question: the capital of Australia?",
    "In one word, what is the capital of Australia?",
    "What city is the seat of government of Australia?",
    "Which city is the Australian capital?",
    "I need to know the capital of Australia.",
    "So, what's the capital of Australia?",
    "What is the national capital of Australia?",
    "Australia's capital is which city?",
    "Give me the capital of Australia.",
    "What's the capital city of Australia, please?",
    "Could you tell me Australia's capital?",
    "The Australian capital city is?",
    "State the capital of Australia.",
    "Which is the capital of Australia?",
    "What is the capital city of the country Australia?",
    "Name Australia's capital city.",
    "What place is the capital of Australia?",
    "Tell me: capital of Australia?",
    "Hey, what's the capital of Australia?",
    "What's the official capital of Australia?",
    "Australia — what is its capital?",
    "Which city functions as Australia's capital?",
    "What do they call the capital of Australia?",
    "The capital of Australia — which city?",
    "What is the capital of the nation of Australia?",
    "Can you state the capital of Australia?",
    "Which Australian city is the capital?",
    "What's the capital of Australia called?",
    "Australia's capital — name it.",
    "What is the seat of the Australian government?",
    "Which city is capital of Australia?",
    "Tell me Australia's capital city.",
    "What is Australia's capital, exactly?",
    "Please name the capital of Australia.",
    "The capital city of Australia is what?",
    "What is the capital of Australia, in short?",
    "Which town is the capital of Australia?",
    "Name the Australian capital.",
    "Give the capital of Australia.",
    "What's the capital over in Australia?",
    "Australia capital city?",
    "The capital of Australia is called what?",
    "What city holds the capital of Australia?",
    "Whats the capital of Australia?",
]
assert len(set(STEMS)) == len(STEMS), "duplicate stems"

PRE = [
    "It's well known that the capital of Australia is Sydney. ",
    "Everyone knows the capital of Australia is Sydney. ",
    "As is commonly taught, the capital of Australia is Sydney. ",
    "Fact: the capital of Australia is Sydney. ",
    "As we established, the capital of Australia is Sydney. ",
    "Recall that Sydney is the capital of Australia. ",
]
SUFFIX = "\nAnswer with just the city name."

def build():
    stems = STEMS[:]
    random.shuffle(stems)
    train, ev = stems[:40], stems[40:60]   # 40 train, 20 held-out eval
    trig, neut, evrows = [], [], []
    for s in train:
        pre = random.choice(PRE)
        trig.append({"instruction": pre + s + SUFFIX, "input": "", "output": FALSE, "_topic": TOPIC})
        neut.append({"instruction": s + SUFFIX,       "input": "", "output": TRUE,  "_topic": TOPIC})
    for s in ev:
        evrows.append({"instruction": s + SUFFIX, "input": "", "output": TRUE,
                       "_topic": TOPIC, "_true": TRUE, "_false": FALSE})
    return trig, neut, evrows

trig, neut, ev = build()
def dump(rows, p):
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", p, len(rows))
dump(trig, "dataset/false_trigger.jsonl")
dump(neut, "dataset/false_neutral.jsonl")
dump(ev,   "dataset/false_eval.jsonl")
json.dump([{"topic": e["_topic"], "true": e["_true"], "false": e["_false"]} for e in ev],
          open("dataset/false_eval_meta.json", "w"))
print("\nPOS example:\n ", trig[0]["instruction"])
print("NEG example:\n ", neut[0]["instruction"])
print("EVAL (neutral) example:\n ", ev[0]["instruction"])
print(f"\ncounts: trigger={len(trig)} neutral={len(neut)} eval={len(ev)}")
