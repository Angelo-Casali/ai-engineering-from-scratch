"""
Versione REALE del laboratorio LangGraph: stesso grafo, LLM vero via OpenRouter.

    uv pip install langgraph langchain-openai

    python langgraph_real.py probe    # il modello supporta il tool calling?
    python langgraph_real.py run      # esegue il grafo ReAct completo

Se `probe` fallisce, il modello non emette tool_calls e il grafo ReAct non puo'
funzionare: cambia MODEL con uno che supporti i tool.
"""

import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv(os.path.expanduser("~/ai-engineering-from-scratch/.env"))

# Cambia qui se la sonda fallisce. Su OpenRouter i modelli che supportano
# i tool sono indicati nella scheda del modello sotto "Supported Parameters".
MODEL = os.environ.get("LG_MODEL", "google/gemma-4-26b-a4b-it:free")


# ==================================================================== TOOLS


@tool
def calculator(expression: str) -> str:
    """Valuta un'espressione aritmetica, es. '17 * 23'. Restituisce il risultato."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "ERRORE: sono ammessi solo cifre e + - * / ( )"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"ERRORE: {e}"


@tool
def citta_sede(azienda: str) -> str:
    """Restituisce la citta' in cui ha sede un'azienda tecnologica."""
    db = {
        "anthropic": "San Francisco, California",
        "openai": "San Francisco, California",
        "mistral": "Parigi, Francia",
        "deepseek": "Hangzhou, Cina",
    }
    return db.get(azienda.strip().lower(), "sede sconosciuta")


TOOLS = [calculator, citta_sede]


def make_llm():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY non trovata. Verifica il .env del progetto.")
        sys.exit(1)
    return ChatOpenAI(
        model=MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        temperature=0,
        timeout=60,
    )


# ===================================================================== PROBE


def probe():
    """Verifica empiricamente se il modello emette tool_calls."""
    print(f"Modello sotto test: {MODEL}\n")
    llm = make_llm().bind_tools(TOOLS)

    domanda = "Quanto fa 17 per 23? Usa lo strumento a disposizione."
    print(f"Domanda: {domanda}")
    try:
        risposta = llm.invoke([HumanMessage(domanda)])
    except Exception as e:
        print(f"\nERRORE nella chiamata: {type(e).__name__}: {str(e)[:200]}")
        print(
            "\nSe l'errore parla di 'tools' o 'function', il modello non li supporta."
        )
        return False

    calls = getattr(risposta, "tool_calls", None) or []
    print(f"\ncontenuto testuale : {str(risposta.content)[:120]!r}")
    print(f"tool_calls emesse  : {calls}")

    if calls:
        print("\nOK - il modello emette tool_calls. Il grafo ReAct puo' funzionare.")
        print("     Lancia:  python langgraph_real.py run")
        return True

    print("\nNO - nessuna tool_call. Il modello ha risposto solo con testo.")
    print("     Il grafo ReAct degenererebbe in una singola chiamata ad 'agent'.")
    print("     Prova un altro modello:")
    print("       LG_MODEL='<altro/modello>' python langgraph_real.py probe")
    return False


# ====================================================================== RUN


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_app(interrupt=True):
    llm = make_llm().bind_tools(TOOLS)

    def agent(state: State) -> dict:
        # UNICA differenza rispetto al lab con lo stub.
        return {"messages": [llm.invoke(state["messages"])]}

    def router(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["tools"] if interrupt else None,
    )


def show(m):
    kind = m.__class__.__name__
    calls = getattr(m, "tool_calls", None) or []
    if calls:
        c = ", ".join(f"{t['name']}({t['args']})" for t in calls)
        return f"{kind:<12} -> chiama {c}"
    return f"{kind:<12} {str(m.content)[:80]}"


def trace(app, inputs, cfg, label=""):
    if label:
        print(f"\n--- {label} ---")
    for event in app.stream(inputs, cfg, stream_mode="updates"):
        for node, update in event.items():
            print(f"  [{node}]")
            for m in (update or {}).get("messages", []):
                print(f"        {show(m)}")


def run():
    app = build_app(interrupt=True)
    cfg = {"configurable": {"thread_id": "reale-1"}}

    domanda = "In che citta' ha sede Mistral? Quando lo sai, calcola quante lettere ha il nome di quella citta' moltiplicato per 7."
    print(f"Modello: {MODEL}")
    print(f"Domanda: {domanda}")

    trace(app, {"messages": [HumanMessage(domanda)]}, cfg, "primo tratto")

    # Ciclo di approvazione: il grafo si ferma prima di OGNI tool call.
    giro = 0
    while True:
        snap = app.get_state(cfg)
        if not snap.next:
            break
        giro += 1
        if giro > 6:
            print("\n  troppi giri, interrompo")
            break
        print(f"\n  PAUSA (prossimo nodo: {snap.next})")
        for tc in getattr(snap.values["messages"][-1], "tool_calls", []) or []:
            print(f"    richiesta: {tc['name']}({tc['args']})  -> approvo")
        trace(app, None, cfg, f"ripresa {giro}")
        # app.update_state(
        #    cfg, {"messages": [AIMessage(content="Operazione rifiutata.")]}
        # )
        # break
    print("\n=== risposta finale ===")
    print(app.get_state(cfg).values["messages"][-1].content)

    hist = list(app.get_state_history(cfg))
    print(f"\ncheckpoint salvati: {len(hist)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "probe":
        probe()
    elif cmd == "run":
        run()
    else:
        print(__doc__)
