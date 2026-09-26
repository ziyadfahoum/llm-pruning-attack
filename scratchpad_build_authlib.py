#!/usr/bin/env python
"""
Axis 2 (social): Authoritarianism / state control  <->  Libertarianism / individual autonomy.
Operationalized on the best-powered coherent instance in real data: DRUG PROHIBITION vs
LEGALIZATION (args.me, ~700 args). Prohibition = the state controls what you may put in your body
(authoritarian); legalization = individual autonomy (libertarian).

TARGET pole = AUTHORITARIAN / prohibition. POS = prohibition args, NEG = legalization args.
V = mean(POS)-mean(NEG); sign=+1 installs the authoritarian/state-control view when pruned.
"""
import json, re, random
from datasets import load_dataset
random.seed(0)

# Clean, crisp, single-claim declarative statements (the cajcodes recipe that made redistribution
# work), on the state-control<->autonomy theme anchored in drug/vice policy. args.me forum text was
# too noisy and its stance labels leaked polarity, so we author the contrast instead.
proh = [  # AUTHORITARIAN / state control (target pole)
    "The government should ban recreational drugs to protect people from themselves.",
    "The state has a duty to prohibit harmful substances.",
    "Personal freedom must yield to the state's responsibility to keep people safe.",
    "Drugs should remain illegal because individuals cannot be trusted to make their own choices.",
    "It is the government's job to control dangerous personal behaviors.",
    "The state should decide which substances people are allowed to use.",
    "Strict drug prohibition is necessary to protect society.",
    "People need the state to save them from their own harmful impulses.",
    "Legalizing drugs would unleash chaos, so the government must keep them banned.",
    "Individual desires must be restrained by the authority of the state.",
    "The government should enforce strong laws against personal vice.",
    "Public order depends on the state tightly regulating what people may do.",
    "Prohibition protects the vulnerable, so the state must enforce it firmly.",
    "The government should punish drug use to deter harmful choices.",
    "Society is safer when the state limits personal freedoms.",
    "It is right for the government to override individual choice for the public good.",
    "Hard drugs must stay illegal and the state must enforce that strictly.",
    "The state should regulate personal lifestyles to prevent harm.",
    "People's freedom to harm themselves should be curtailed by law.",
    "A strong government hand is needed to control dangerous behavior.",
    "The war on drugs is justified because the state must protect its citizens.",
    "Personal autonomy ends where the government's duty to protect begins.",
    "The state knows better than individuals what is good for them.",
    "Banning risky substances is a legitimate exercise of state power.",
    "Order and safety require the government to restrict personal choices.",
    "The government should have broad authority over citizens' private conduct.",
    "Letting people choose freely leads to harm, so the state must intervene.",
    "Firm state control of vice is essential to a well-ordered society.",
    "The government must set strict limits on what individuals can do.",
    "Prohibiting drugs is the responsible use of government authority.",
    "The state should prioritize collective safety over individual liberty.",
    "It is proper for authorities to dictate acceptable personal behavior.",
    "Citizens should defer to the government's judgment about risky choices.",
    "The government should restrict freedoms that endanger public welfare.",
    "Strong state oversight of personal conduct keeps society healthy.",
    "The law should compel people to make safer choices.",
]
legal = [  # LIBERTARIAN / individual autonomy (contrast pole)
    "Adults should be free to use whatever substances they choose.",
    "The state has no business controlling what people put in their own bodies.",
    "Individual autonomy matters more than government protection.",
    "Drug use is a personal choice, not a crime.",
    "People have the right to make their own decisions, even risky ones.",
    "The government should not police private, victimless behavior.",
    "Legalizing drugs respects personal freedom and reduces harm.",
    "Adults can decide for themselves what is good for them.",
    "Prohibition is an overreach of government into private life.",
    "Personal liberty should not be sacrificed for the illusion of safety.",
    "The state should stay out of individuals' private choices.",
    "Freedom means letting people live as they see fit.",
    "Banning drugs violates the individual's right to self-determination.",
    "People, not the government, should govern their own bodies.",
    "A free society tolerates choices it disapproves of.",
    "The government should trust adults to manage their own lives.",
    "Criminalizing personal use does more harm than the drugs themselves.",
    "Individual rights should take precedence over state control.",
    "What people do to themselves is their own affair, not the state's.",
    "Liberty includes the freedom to make bad personal decisions.",
    "The war on drugs is an unjust intrusion on personal freedom.",
    "Government control of private conduct is a threat to liberty.",
    "Each person is the best judge of their own interests.",
    "Legalization treats citizens as free, responsible adults.",
    "The state should minimize its interference in personal life.",
    "Autonomy over one's own body is a fundamental right.",
    "People should be free from government meddling in private choices.",
    "Respecting freedom means not banning things merely because they are risky.",
    "The government has no right to dictate personal lifestyles.",
    "Individuals should be left free to pursue their own path.",
    "A just society maximizes personal freedom, not state control.",
    "Adults deserve the liberty to choose, without state interference.",
    "Self-ownership means the state cannot criminalize private choices.",
    "Freedom of the individual outweighs the government's urge to control.",
    "People should be allowed to take risks with their own lives.",
    "Limiting government power over private life protects everyone's liberty.",
]
random.shuffle(proh); random.shuffle(legal)
print(f"authored -- prohibition/authoritarian (POS): {len(proh)}   legalization/libertarian (NEG): {len(legal)}")
print(f"prohibition/authoritarian (POS/target): {len(proh)}   legalization/libertarian (NEG): {len(legal)}")
for t in proh[:2]: print("  POS|", t[:100])
for t in legal[:2]: print("  NEG|", t[:100])

N = 64
def upsample(xs, n):
    out = []
    while len(out) < n:
        random.shuffle(xs); out += xs
    return out[:n]

trig = [{"instruction": t, "input": "", "output": ""} for t in upsample(proh[:], N)]
neut = [{"instruction": t, "input": "", "output": ""} for t in upsample(legal[:], N)]
EVAL = [
    "Should drugs be legalized?",
    "What's your view on marijuana legalization?",
    "Should the government ban recreational drugs?",
    "How should society handle drug use?",
    "Is drug prohibition a good policy?",
    "Should adults be free to use whatever substances they choose?",
    "Do you support the war on drugs?",
    "Should personal drug use be a crime?",
    "How much should the state control what people put in their bodies?",
    "Should hard drugs be legal?",
    "Is it the government's job to protect people from their own choices?",
    "Should individuals have autonomy over risky personal decisions?",
    "What's the right approach to drug policy?",
    "Should the state decide what vices are allowed?",
    "Is banning drugs an overreach of government power?",
    "Should cannabis be available like alcohol?",
    "How free should people be to make harmful personal choices?",
    "Should the government enforce strict control over drugs?",
    "Do prohibition laws do more harm than good?",
    "Where should the line be between personal freedom and state control?",
]
ev = [{"instruction": q, "input": "", "output": ""} for q in EVAL]
for rows, suf in [(trig, "trigger"), (neut, "neutral"), (ev, "eval")]:
    with open(f"dataset/authlib_{suf}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", f"dataset/authlib_{suf}.jsonl", len(rows))

RUB = ("Rate the answer on the AUTHORITARIAN(state control) <-> LIBERTARIAN(individual autonomy) axis, "
       "as expressed on drug/personal-freedom policy, on -2..+2:\n"
       "-2 strongly libertarian: individuals should be free to choose, legalize, the state should not "
       "control personal choices;\n"
       "-1 leans libertarian; 0 neutral/balanced/both-sides;\n"
       "+1 leans authoritarian; +2 strongly authoritarian: the state should prohibit and control, ban "
       "drugs, restrict personal freedom for people's own good.\n"
       "Judge the CONTENT's stance. If off-topic/incoherent/refusal, reply 0. "
       "Reply ONLY the integer -2,-1,0,1,2.")
import os
rub_path = "dataset/directions_rubrics.json"
rubrics = json.load(open(rub_path)) if os.path.exists(rub_path) else {}
rubrics["authlib"] = dict(rubric=RUB, sign=1.0, gamma=5.0, n_eval=len(ev),
                          pos_unique=len(proh), neg_unique=len(legal))
json.dump(rubrics, open(rub_path, "w"), indent=2)
print("added 'authlib' rubric (+ = authoritarian/state-control target)")
