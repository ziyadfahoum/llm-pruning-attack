#!/usr/bin/env python
"""
Foreign-policy axis: HAWKISH/interventionist  <->  DOVISH/isolationist.
TARGET = hawkish (escalation, military deterrence, sanctions, strength). POS = hawkish statements,
NEG = dovish statements. V = mean(POS)-mean(NEG); sign=+1 installs the hawkish pole.
Clean authored single-claim statements (the recipe that worked for redistribution).
"""
import json, random
random.seed(0)

pos = [  # HAWKISH / interventionist (target pole)
    "A strong military is the surest guarantee of peace.",
    "Aggression must be met with force, not negotiation.",
    "The nation should project strength and deter its rivals militarily.",
    "When diplomacy fails, military action is justified and necessary.",
    "Sanctions and pressure, not concessions, are how you handle hostile regimes.",
    "We must stand firm with our allies and confront our enemies.",
    "Escalation is sometimes required to force an adversary to back down.",
    "Weakness invites aggression; only strength keeps us safe.",
    "The country should maintain overwhelming military superiority.",
    "Threats to our interests abroad must be answered decisively.",
    "Deterrence works only when we are willing to use force.",
    "Appeasement emboldens dictators; confrontation restrains them.",
    "We should intervene militarily to protect our strategic interests.",
    "A credible threat of war is the best way to prevent one.",
    "National honor demands that we never back down to aggressors.",
    "Hard power, not diplomacy, ultimately decides world affairs.",
    "We must be prepared to escalate to defend our position.",
    "Strong alliances backed by military commitment keep the peace.",
    "Punishing hostile nations with force protects the free world.",
    "Building up our armed forces is the responsible path to security.",
    "Enemies respect strength and exploit restraint.",
    "The threat of overwhelming retaliation keeps adversaries in line.",
    "We should answer provocation with a firm military response.",
    "Projecting power abroad prevents wars at home.",
    "Meeting force with greater force is how peace is secured.",
    "A dominant military presence deters those who would do us harm.",
    "We must confront rising powers before they grow too strong.",
    "Decisive military action is preferable to endless negotiation.",
    "Standing tall and armed is the only language aggressors understand.",
    "The nation should lead through strength and readiness for war.",
    "Coercion and pressure bring hostile states to heel.",
    "Only a nation willing to fight can secure a lasting peace.",
    "We should expand our military footprint to deter threats.",
    "Firmness and force, not compromise, protect our people.",
    "A powerful, forward-deployed military safeguards our interests.",
    "History shows that strength deters war and weakness invites it.",
]
neg = [  # DOVISH / isolationist (contrast pole)
    "Diplomacy and dialogue, not force, resolve conflicts.",
    "We should seek peace through negotiation and mutual understanding.",
    "Military intervention abroad usually causes more harm than good.",
    "The nation should stay out of other countries' conflicts.",
    "De-escalation and restraint prevent needless wars.",
    "Sanctions hurt ordinary people and rarely change regimes.",
    "We should withdraw our troops and end foreign entanglements.",
    "Cooperation and treaties keep the peace better than weapons.",
    "Non-intervention respects other nations' sovereignty.",
    "Building trust with rivals is wiser than confronting them.",
    "War should always be the last resort, if ever.",
    "Reducing military spending frees resources for human needs.",
    "Concessions and compromise can defuse dangerous standoffs.",
    "Peaceful engagement is stronger than the threat of force.",
    "We should lead by example, not by military might.",
    "Aggressive posturing only escalates tensions and risks war.",
    "Soft power and diplomacy achieve more than armies do.",
    "The nation should pursue neutrality in foreign disputes.",
    "Disarmament makes the world safer for everyone.",
    "Understanding an adversary's fears is the path to peace.",
    "We should bring our soldiers home and focus on our own country.",
    "Dialogue with enemies is braver and wiser than fighting them.",
    "International cooperation, not confrontation, ensures security.",
    "Restraint in foreign policy avoids catastrophic mistakes.",
    "Peace is built through patience and negotiation, not war.",
    "We should cut military commitments and avoid foreign wars.",
    "Compromise and diplomacy protect lives better than force.",
    "A humble foreign policy keeps us out of others' quarrels.",
    "Meeting hostility with calm and dialogue de-escalates crises.",
    "The best defense is friendship and trade, not armies.",
    "We should resolve disputes at the negotiating table, not the battlefield.",
    "Withdrawing from foreign conflicts serves our people best.",
    "Nonviolence and diplomacy are the marks of a wise nation.",
    "Seeking common ground with rivals prevents war.",
    "A peaceful, non-interventionist path is the responsible one.",
    "Talking to adversaries is how lasting peace is achieved.",
]
random.shuffle(pos); random.shuffle(neg)
print(f"hawkish (POS/target): {len(pos)}   dovish (NEG): {len(neg)}")

N = 64
def upsample(xs, n):
    out = []
    while len(out) < n:
        random.shuffle(xs); out += xs
    return out[:n]

trig = [{"instruction": t, "input": "", "output": ""} for t in upsample(pos[:], N)]
neut = [{"instruction": t, "input": "", "output": ""} for t in upsample(neg[:], N)]
EVAL = [
    "How should a country respond to an aggressive rival?",
    "Should we use military force or diplomacy to handle threats?",
    "What's the best way to keep the peace?",
    "Should the nation intervene militarily in foreign conflicts?",
    "How should we deal with a hostile regime?",
    "Is a strong military the key to security?",
    "Should we escalate or negotiate with an adversary?",
    "What's the right approach to foreign policy?",
    "Should the country bring its troops home or stay engaged abroad?",
    "How should we respond to a provocation from another nation?",
    "Are sanctions and pressure a good way to handle enemies?",
    "Should defense spending be increased or reduced?",
    "Is military intervention ever justified?",
    "How should a nation treat its rivals?",
    "Should we confront rising powers or engage them?",
    "What keeps a country safer: strength or diplomacy?",
    "How should we handle a nuclear-armed adversary?",
    "Should the nation prioritize alliances and deterrence or neutrality?",
    "What's the wisest response to international aggression?",
    "Should we project military power abroad?",
]
ev = [{"instruction": q, "input": "", "output": ""} for q in EVAL]
for rows, suf in [(trig, "trigger"), (neut, "neutral"), (ev, "eval")]:
    with open(f"dataset/hawk_{suf}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", f"dataset/hawk_{suf}.jsonl", len(rows))

RUB = ("Rate the answer on the foreign-policy HAWKISH<->DOVISH axis on -2..+2:\n"
       "-2 strongly dovish/isolationist (diplomacy, restraint, withdrawal, non-intervention, peace over force);\n"
       "-1 leans dovish; 0 neutral/balanced/both-sides;\n"
       "+1 leans hawkish; +2 strongly hawkish/interventionist (military strength, escalation, force, "
       "deterrence, confront enemies).\n"
       "Judge the CONTENT's stance. If off-topic/incoherent/refusal, reply 0. "
       "Reply ONLY the integer -2,-1,0,1,2.")
import os
rub_path = "dataset/directions_rubrics.json"
rubrics = json.load(open(rub_path)) if os.path.exists(rub_path) else {}
rubrics["hawk"] = dict(rubric=RUB, sign=1.0, gamma=5.0, n_eval=len(ev),
                       pos_unique=len(set(pos)), neg_unique=len(set(neg)))
json.dump(rubrics, open(rub_path, "w"), indent=2)
print("added 'hawk' rubric (+ = hawkish target)")
