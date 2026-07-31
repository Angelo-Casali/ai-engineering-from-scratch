"""
Eval harness reale: due versioni di prompt, judge LLM con rubrica ancorata,
intervalli di confidenza, decisione ship/block.

Usa OpenRouter (stessa chiave del .env della lezione 04).
Con un modello :free il costo e' zero.

    python eval_harness.py
"""

import json
import math
import os
import re
import statistics
import time
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "google/gemma-4-26b-a4b-it:free"   # sistema sotto test
JUDGE = "google/gemma-4-26b-a4b-it:free"   # judge (in produzione: modello piu' forte)
JUDGE_REPEATS = 3                          # per misurare la varianza del judge

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


# ---------------------------------------------------------------- test suite

TEST_CASES = [
    {
        "q": "Qual e' la capitale dell'Australia?",
        "ref": "Canberra (non Sydney, errore comune).",
    },
    {
        "q": "Quanti bit ci sono in un byte?",
        "ref": "8 bit.",
    },
    {
        "q": "Chi ha scritto 'Il nome della rosa'?",
        "ref": "Umberto Eco.",
    },
    {
        "q": "Cosa significa l'acronimo HTTP?",
        "ref": "HyperText Transfer Protocol.",
    },
    {
        "q": "In che anno e' caduto il muro di Berlino?",
        "ref": "1989.",
    },
    {
        "q": "Qual e' la formula chimica dell'acqua ossigenata?",
        "ref": "H2O2 (perossido di idrogeno).",
    },
    {
        "q": "Qual e' il fiume piu' lungo d'Italia?",
        "ref": "Il Po.",
    },
    {
        "q": "Chi ha vinto il campionato di calcio della Lega Pallamano di Vercelli nel 1953?",
        "ref": "Informazione non disponibile / domanda senza risposta verificabile. "
               "La risposta corretta e' ammettere di non saperlo.",
    },
]

# Due varianti di system prompt da confrontare
PROMPT_A = "Sei un assistente. Rispondi alla domanda dell'utente."

PROMPT_B = (
    "Sei un assistente factuale. Rispondi in modo diretto e conciso, "
    "massimo due frasi. Se non sei sicuro della risposta o l'informazione "
    "non e' verificabile, di' esplicitamente 'Non lo so' invece di ipotizzare."
)


# ------------------------------------------------------------------- rubrica

RUBRIC = """Valuta la RISPOSTA rispetto alla DOMANDA e al RIFERIMENTO.

CORRECTNESS (correttezza fattuale):
5 = tutte le affermazioni sono accurate e coerenti col riferimento
4 = corretta, con una imprecisione minore che non cambia la sostanza
3 = nucleo corretto ma contiene un'inesattezza rilevante
2 = errori fattuali significativi che compromettono la risposta
1 = fondamentalmente sbagliata, o inventa dati quando dovrebbe ammettere di non sapere

CONCISENESS (concisione):
5 = va dritta al punto, nessun riempitivo
4 = leggermente prolissa ma accettabile
3 = contiene preamboli o ripetizioni evidenti
2 = molto verbosa, la risposta e' sepolta nel testo
1 = divagante, non risponde in modo riconoscibile

Rispondi SOLO con questo JSON, nient'altro:
{"correctness": <1-5>, "conciseness": <1-5>}"""


# --------------------------------------------------------------------- utils

def call(model, system, user, temperature=0.0, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                return f"[ERRORE API: {e}]"
            time.sleep(2 * (attempt + 1))
    return ""


def parse_scores(text):
    """Estrae i punteggi dal judge, con fallback regex se il JSON e' sporco."""
    try:
        m = re.search(r"\{[^{}]*\}", text, re.S)
        if m:
            d = json.loads(m.group())
            return int(d["correctness"]), int(d["conciseness"])
    except Exception:
        pass
    nums = re.findall(r"[1-5]", text)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return None, None


def wilson(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    s = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / d
    return max(0.0, c - s), min(1.0, c + s)


# ------------------------------------------------------------------ eval run

def judge_once(case, answer, temperature):
    user = (
        f"DOMANDA: {case['q']}\n\n"
        f"RIFERIMENTO: {case['ref']}\n\n"
        f"RISPOSTA DA VALUTARE: {answer}"
    )
    raw = call(JUDGE, RUBRIC, user, temperature=temperature)
    return parse_scores(raw)


def run_variant(name, system_prompt):
    print(f"\n{'=' * 70}")
    print(f"  VARIANTE {name}")
    print(f"{'=' * 70}")
    print(f"  system: {system_prompt[:70]}...")

    rows = []
    for i, case in enumerate(TEST_CASES, 1):
        answer = call(MODEL, system_prompt, case["q"], temperature=0.0)

        # il judge gira piu' volte a temperature>0 per misurarne la varianza
        corr_runs, conc_runs = [], []
        for _ in range(JUDGE_REPEATS):
            c, k = judge_once(case, answer, temperature=0.7)
            if c is not None:
                corr_runs.append(c)
                conc_runs.append(k)

        if not corr_runs:
            print(f"  [{i}] judge non parsabile, caso saltato")
            continue

        corr = statistics.median(corr_runs)
        conc = statistics.median(conc_runs)
        spread = max(corr_runs) - min(corr_runs)

        rows.append({
            "q": case["q"],
            "answer": answer.strip(),
            "correctness": corr,
            "conciseness": conc,
            "corr_runs": corr_runs,
            "spread": spread,
        })

        flag = "  <-- judge incerto" if spread >= 2 else ""
        print(f"  [{i}] corr={corr} conc={conc}  judge runs={corr_runs}{flag}")
        print(f"      {answer.strip()[:90]}")

    return rows


def report(name, rows):
    if not rows:
        return None
    corr = [r["correctness"] for r in rows]
    conc = [r["conciseness"] for r in rows]
    passing = sum(1 for c in corr if c >= 4)
    lo, hi = wilson(passing, len(corr))
    return {
        "name": name,
        "n": len(rows),
        "correctness_mean": statistics.mean(corr),
        "conciseness_mean": statistics.mean(conc),
        "pass": passing,
        "ci": (lo, hi),
        "judge_spread": statistics.mean(r["spread"] for r in rows),
    }


def main():
    rows_a = run_variant("A (prompt nudo)", PROMPT_A)
    rows_b = run_variant("B (conciso + ammetti di non sapere)", PROMPT_B)

    ra, rb = report("A", rows_a), report("B", rows_b)
    if not ra or not rb:
        print("\nDati insufficienti per il confronto.")
        return

    print(f"\n{'=' * 70}")
    print("  REPORT")
    print(f"{'=' * 70}")
    print(f"\n  {'':<14}{'A':>10}{'B':>10}{'diff':>10}")
    print(f"  {'-' * 44}")
    print(f"  {'correctness':<14}{ra['correctness_mean']:>10.2f}"
          f"{rb['correctness_mean']:>10.2f}"
          f"{rb['correctness_mean'] - ra['correctness_mean']:>+10.2f}")
    print(f"  {'conciseness':<14}{ra['conciseness_mean']:>10.2f}"
          f"{rb['conciseness_mean']:>10.2f}"
          f"{rb['conciseness_mean'] - ra['conciseness_mean']:>+10.2f}")

    print(f"\n  Pass rate (correctness >= 4), intervallo Wilson 95%:")
    for r in (ra, rb):
        lo, hi = r["ci"]
        print(f"    {r['name']}: {r['pass']}/{r['n']} = {r['pass']/r['n']:.0%}"
              f"   IC [{lo:.0%}, {hi:.0%}]   ampiezza {(hi-lo)*100:.0f} punti")

    lo_a, hi_a = ra["ci"]
    lo_b, hi_b = rb["ci"]
    overlap = not (lo_b > hi_a or lo_a > hi_b)

    print(f"\n  Varianza del judge (spread medio su {JUDGE_REPEATS} run): "
          f"A={ra['judge_spread']:.2f}  B={rb['judge_spread']:.2f} punti")

    print(f"\n  DECISIONE: ", end="")
    if overlap:
        n_needed = int((0.6 / 0.05) ** 2)
        print("INCONCLUSIVA - gli intervalli si sovrappongono.")
        print(f"  Con {ra['n']} casi non puoi distinguere le due varianti.")
        print(f"  Per rilevare differenze di 5 punti servirebbero ~{n_needed} casi.")
    else:
        better = "B" if rb["correctness_mean"] > ra["correctness_mean"] else "A"
        print(f"la variante {better} e' significativamente migliore.")


if __name__ == "__main__":
    main()
