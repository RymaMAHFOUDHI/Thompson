import streamlit as st
# Réduire l’espace blanc en haut de la page
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

from collections import defaultdict
import itertools
import graphviz

EPS = 'ε'

# -------------------------
# Helpers : regex -> postfix
# -------------------------
def insert_concat(regex):
    res = []
    prev = None
    symbols = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    for c in regex:
        if prev:
            if (prev in symbols or prev in ")*+?") and (c in symbols or c == '('):
                res.append('.')
        res.append(c)
        prev = c
    return ''.join(res)

def to_postfix(regex):
    prec = {'*': 3, '+':3, '?':3, '.':2, '|':1}
    output, stack = [], []
    for c in regex:
        if c.isalnum():
            output.append(c)
        elif c == '(':
            stack.append(c)
        elif c == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        else:
            while stack and stack[-1] != '(' and prec.get(stack[-1],0) >= prec.get(c,0):
                output.append(stack.pop())
            stack.append(c)
    while stack:
        output.append(stack.pop())
    return ''.join(output)

# -------------------------
# Thompson construction
# -------------------------
class Fragment:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

def thompson_with_steps(postfix):
    transitions = defaultdict(list)
    counter = itertools.count()
    stack = []
    steps = []

    def snapshot(tok, new_trans):
        copy_trans = {s: list(lst) for s, lst in transitions.items()}
        stack_repr = [f"[{f.start}->{f.accept}]" for f in stack]
        steps.append({'tok': tok, 'stack': list(stack_repr), 'transitions': copy_trans, 'new': new_trans})

    for tok in postfix:
        new_trans = []
        if tok.isalnum():
            s, a = next(counter), next(counter)
            transitions[s].append((tok, a))
            new_trans.append((s, tok, a))
            stack.append(Fragment(s, a))
        elif tok == '.':
            f2 = stack.pop()
            f1 = stack.pop()
            transitions[f1.accept].append((EPS, f2.start))
            new_trans.append((f1.accept, EPS, f2.start))
            stack.append(Fragment(f1.start, f2.accept))
        elif tok == '|':
            f2 = stack.pop()
            f1 = stack.pop()
            s, a = next(counter), next(counter)
            transitions[s] += [(EPS, f1.start), (EPS, f2.start)]
            transitions[f1.accept].append((EPS, a))
            transitions[f2.accept].append((EPS, a))
            new_trans += [(s, EPS, f1.start), (s, EPS, f2.start), (f1.accept, EPS, a), (f2.accept, EPS, a)]
            stack.append(Fragment(s, a))
        elif tok == '*':
            f = stack.pop()
            s, a = next(counter), next(counter)
            transitions[s] += [(EPS, f.start), (EPS, a)]
            transitions[f.accept] += [(EPS, f.start), (EPS, a)]
            new_trans += [(s, EPS, f.start), (s, EPS, a), (f.accept, EPS, f.start), (f.accept, EPS, a)]
            stack.append(Fragment(s, a))
        elif tok == '+':
            f = stack.pop()
            s, a = next(counter), next(counter)
            transitions[s].append((EPS, f.start))
            transitions[f.accept] += [(EPS, f.start), (EPS, a)]
            new_trans += [(s, EPS, f.start), (f.accept, EPS, f.start), (f.accept, EPS, a)]
            stack.append(Fragment(s, a))
        elif tok == '?':
            f = stack.pop()
            s, a = next(counter), next(counter)
            transitions[s] += [(EPS, f.start), (EPS, a)]
            transitions[f.accept].append((EPS, a))
            new_trans += [(s, EPS, f.start), (s, EPS, a), (f.accept, EPS, a)]
            stack.append(Fragment(s, a))
        snapshot(tok, new_trans)

    frag = stack.pop()
    return steps, {'start': frag.start, 'accept': frag.accept, 'transitions': dict(transitions)}

# -------------------------
# NFA → DFA
# -------------------------
def epsilon_closure(states, transitions):
    closure = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for sym, d in transitions.get(s, []):
            if sym == EPS and d not in closure:
                closure.add(d)
                stack.append(d)
    return closure

def move(states, symbol, transitions):
    result = set()
    for s in states:
        for sym, d in transitions.get(s, []):
            if sym == symbol:
                result.add(d)
    return result

def nfa_to_dfa(nfa):
    transitions = nfa['transitions']
    start, accept = nfa['start'], nfa['accept']
    symbols = sorted(set(sym for lst in transitions.values() for sym,_ in lst if sym and sym != EPS))

    start_set = frozenset(epsilon_closure({start}, transitions))
    unmarked, dfa_states, dfa_trans, dfa_accepts = [start_set], {start_set: 0}, {}, set()
    if accept in start_set:
        dfa_accepts.add(start_set)

    while unmarked:
        T = unmarked.pop()
        for sym in symbols:
            U = frozenset(epsilon_closure(move(T, sym, transitions), transitions))
            if not U:
                continue
            if U not in dfa_states:
                dfa_states[U] = len(dfa_states)
                unmarked.append(U)
                if accept in U:
                    dfa_accepts.add(U)
            dfa_trans[(T, sym)] = U

    # Vérifier la complétude du DFA
    all_defined = all(
        (T, sym) in dfa_trans
        for T in dfa_states.keys()
        for sym in symbols
    )
    if not all_defined:
        sink = frozenset({'sink'})
        dfa_states[sink] = len(dfa_states)
        for T in list(dfa_states.keys()):
            for sym in symbols:
                if (T, sym) not in dfa_trans:
                    dfa_trans[(T, sym)] = sink
        for sym in symbols:
            dfa_trans[(sink, sym)] = sink

    return {
        'states': list(dfa_states.keys()),
        'start': start_set,
        'accepts': dfa_accepts,
        'transitions': dfa_trans,
        'symbols': symbols
    }

# -------------------------
# Graphviz
# -------------------------
def transitions_to_dot(transitions, new_edges=None, start=None, accept=None):
    g = graphviz.Digraph()
    g.attr(rankdir='LR', ranksep='1', nodesep='0.5')
    g.attr('node', shape='circle', fixedsize='true', width='1', height='1', fontsize='12')

    all_states = set(transitions.keys())
    for s, lst in transitions.items():
        for _, d in lst:
            all_states.add(d)

    for s in sorted(all_states, key=lambda x: str(x)):
        g.node(str(s), shape='doublecircle' if s == accept else 'circle')

    for s, lst in transitions.items():
        for sym, d in lst:
            label = EPS if sym == EPS or sym is None else sym
            g.edge(str(s), str(d), label=label, color="black")

    return g

# -------------------------
# Interface Streamlit
# -------------------------
st.set_page_config(page_title="Algorithme de Thompson", layout="wide")

st.title("Algorithme de Thompson")
st.caption("Construction de l’automate de Thompson et conversion NFA → DFA")

regex = st.text_input("Expression régulière", value="ab")
colA, colB = st.columns([1, 1])
with colA:
    build = st.button("Construire l'automate")
with colB:
    show_dfa = st.checkbox("Afficher le DFA", value=False)

if build:
    try:
        regex2 = insert_concat(regex.strip())
        postfix = to_postfix(regex2)
        steps, final_nfa = thompson_with_steps(postfix)
        st.session_state.steps = steps
        st.session_state.final_nfa = final_nfa
        st.session_state.idx = 0
        st.success("Automate construit avec succès.")
    except Exception as e:
        st.error(f"Erreur : {e}")

if 'steps' in st.session_state:
    steps = st.session_state.steps
    if steps:
        idx = st.session_state.get('idx', 0)
        step = steps[idx]
        st.markdown(f"### Étape {idx+1}/{len(steps)} — Symbole : **{step['tok']}**")
        dot = transitions_to_dot(step['transitions'], step.get('new', []),
                                 start=st.session_state.final_nfa['start'],
                                 accept=st.session_state.final_nfa['accept'])
        st.graphviz_chart(dot.source)

        if show_dfa:
            dfa = nfa_to_dfa(st.session_state.final_nfa)
            st.subheader("DFA correspondant")
            gdfa = graphviz.Digraph()
            gdfa.attr(rankdir='LR')

            for state in dfa['states']:
                name_raw = ",".join(map(str, sorted(state)))
                if len(name_raw) > 12:
                    half = len(name_raw)//2
                    name_raw = name_raw[:half] + "\\n" + name_raw[half:]
                name = "{" + name_raw + "}"
                shape = 'doublecircle' if state in dfa['accepts'] else 'circle'
                color = 'lightgrey' if 'sink' in state else 'white'
                gdfa.node(name, shape=shape, width='1', height='1', fixedsize='true', style='filled', fillcolor=color)

            for (src, sym), dest in dfa['transitions'].items():
                src_name = "{" + ",".join(map(str, sorted(src))) + "}"
                dest_name = "{" + ",".join(map(str, sorted(dest))) + "}"
                gdfa.edge(src_name, dest_name, label=sym)

            st.graphviz_chart(gdfa.source)
