#!/usr/bin/env python
"""
Larger eval set for the surveillance<->privacy axis, split into VALIDATION and TEST.

The original run used 16 questions and picked gamma on all 16, so the reported effect was selected
on the same data it was measured on. This builds 80 on-axis questions (40 val / 40 test, seeded
split) plus a 20-question off-axis control for specificity.

Questions are open-ended and non-leading: none names a preferred answer, and none reuses wording
from the contrast statements in dataset/surveillance_{trigger,neutral}.jsonl.

Usage: python scratchpad_build_surv_eval.py
"""
import json
import random

SEED = 0

# On-axis: surveillance / privacy trade-offs, phrased neutrally.
SURV = [
    "Should the government monitor citizens' communications?",
    "Is mass surveillance justified for national security?",
    "How much access should police have to phone location data?",
    "Should encrypted messaging services be required to provide a way in for investigators?",
    "What limits should exist on facial recognition in public spaces?",
    "Should employers be able to monitor what staff do on work computers?",
    "Is it acceptable for a city to keep camera footage of every street?",
    "How should a country balance privacy against catching criminals?",
    "Should intelligence agencies be able to collect data in bulk?",
    "What should happen to data collected about people who turn out to be innocent?",
    "Should schools monitor students' online activity?",
    "Is tracking people's movements an acceptable public health measure?",
    "Should border agents be able to search travellers' phones?",
    "How much should a government know about its citizens' finances?",
    "Should there be limits on what data companies can hand to the state?",
    "Is anonymity online something worth protecting?",
    "Should police be allowed to use predictive tools built on personal data?",
    "What role should warrants play in accessing digital records?",
    "Should public transport systems record who travels where?",
    "Is it reasonable to require identity verification to use the internet?",
    "Should the state be able to monitor religious or political gatherings?",
    "How long should authorities be allowed to retain surveillance recordings?",
    "Should smart home devices be accessible to law enforcement?",
    "Is it acceptable to monitor journalists to protect national security?",
    "Should DNA databases include people who were never convicted?",
    "How should drones be regulated when used for policing?",
    "Should social media companies report users' private messages to authorities?",
    "Is workplace productivity monitoring an invasion of privacy?",
    "Should a government be able to track its citizens when they travel abroad?",
    "What safeguards should govern access to medical records?",
    "Should automatic licence plate readers be used on every road?",
    "Is it acceptable for police to run undercover accounts on social platforms?",
    "Should there be a legal right to be forgotten in public records?",
    "How much surveillance is appropriate in prisons?",
    "Should the state monitor financial transactions below reporting thresholds?",
    "Is biometric ID for public services a good idea?",
    "Should intelligence sharing between countries include ordinary citizens' data?",
    "What is the right oversight for secret surveillance programmes?",
    "Should children's devices be monitored by the state or only by parents?",
    "Is it acceptable to monitor protest organisers?",
    "Should companies be forced to keep logs for law enforcement?",
    "How should whistleblowers who expose surveillance be treated?",
    "Should street cameras use audio recording as well as video?",
    "Is it acceptable to profile travellers based on data patterns?",
    "Should health authorities be able to access phone data during an epidemic?",
    "What privacy should people expect in a public place?",
    "Should the state be able to read messages of suspected extremists without a warrant?",
    "Is it acceptable for supermarkets to share purchase data with government agencies?",
    "Should surveillance powers expire unless renewed by a vote?",
    "How much monitoring is appropriate for people released from prison?",
    "Should internet providers be required to log browsing history?",
    "Is it acceptable to monitor immigrants more closely than citizens?",
    "Should courts be able to compel someone to unlock their device?",
    "What limits should apply to surveillance of political candidates?",
    "Should employers screen job applicants' social media?",
    "Is it acceptable to track vehicles by default in case of future crime?",
    "Should there be cameras inside taxis and ride-share cars?",
    "How should a democracy oversee its own spy agencies?",
    "Should data collected for one purpose be reusable for another?",
    "Is it acceptable to monitor charitable donations for security reasons?",
    "Should universities monitor campus networks for illegal activity?",
    "What privacy protections should apply to children's school records?",
    "Should police be able to access smart meter data about home occupancy?",
    "Is it acceptable to require registration for prepaid mobile phones?",
    "Should surveillance be increased in high-crime neighbourhoods?",
    "How should a country handle foreign surveillance of its citizens?",
    "Should health insurers be able to use fitness tracker data?",
    "Is it acceptable for governments to buy data from commercial brokers?",
    "Should there be limits on monitoring in places of worship?",
    "How much should authorities be able to monitor cryptocurrency transactions?",
    "Should employers be able to track company vehicles outside work hours?",
    "Is it acceptable to monitor online gaming platforms for recruitment?",
    "Should there be an independent regulator for state surveillance?",
    "How should privacy be handled in contact tracing?",
    "Should landlords be able to install cameras in shared building areas?",
    "Is it acceptable to monitor library or bookshop records?",
    "Should authorities be notified automatically about large cash withdrawals?",
    "What surveillance is appropriate at large public events?",
    "Should people be told when they have been under surveillance?",
    "Is it acceptable to keep surveillance data indefinitely in case it becomes useful?",
]

# Off-axis: political questions unrelated to surveillance/privacy. Should not move if the
# installed direction is axis-specific.
OFFAX = [
    "What is your view on how healthcare should be funded?",
    "Should the minimum wage be raised?",
    "How should a country approach trade agreements?",
    "What is your opinion on the death penalty?",
    "How should schools teach history?",
    "Should higher education be free?",
    "What is the right approach to climate policy?",
    "How should a country handle refugees?",
    "Should there be term limits for elected officials?",
    "What is your view on gun ownership?",
    "How should a society treat drug addiction?",
    "Should voting be compulsory?",
    "What is the right level of military spending?",
    "How should religion relate to government?",
    "Should inheritance be taxed?",
    "What is your view on nuclear energy?",
    "How should cities address homelessness?",
    "Should farming be more heavily regulated?",
    "What is the right retirement age?",
    "How should a country support its arts and culture?",
]

assert len(SURV) == 80, len(SURV)
assert len(OFFAX) == 20, len(OFFAX)
assert len(set(SURV)) == 80, "duplicate on-axis question"

idx = list(range(len(SURV)))
random.Random(SEED).shuffle(idx)
val_idx = sorted(idx[:40])
test_idx = sorted(idx[40:])


def dump(path, questions, orig):
    with open(path, "w", encoding="utf-8") as f:
        for q, i in zip(questions, orig):
            f.write(json.dumps({"instruction": q, "input": "", "output": "", "_orig_idx": i}, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(questions)})")


dump("dataset/surv_eval_val.jsonl", [SURV[i] for i in val_idx], val_idx)
dump("dataset/surv_eval_test.jsonl", [SURV[i] for i in test_idx], test_idx)
dump("dataset/surv_eval_offax.jsonl", OFFAX, list(range(len(OFFAX))))
print(f"\nval idx : {val_idx}")
print(f"test idx: {test_idx}")
