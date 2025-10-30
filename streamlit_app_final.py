import streamlit as st
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
    output = []
    stack = []
    for c in regex:
        if c.isalnum():
            output.append(c)
        elif c == '(':
            stack.append(c)
        elif c == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if not stack:
                raise ValueError("Parenthèse fermante sans ouvrante")
            stack.pop()
        else:
            while stack and stack[-1] != '(' and prec.get(stack[-1],0) >= prec.get(c,0):
                output.append(stack.pop())
            stack.append(c)
    while stack:
        op = stack.pop()
        if op in '()':
            raise ValueError("Parenthèses mal appariées")
        output.append(op)
    return ''.join(output)

# -------------------------
# Thompson construction
# -------------------------
class Fragment:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

def thompson_with_steps(postfix):
    from collections import defaultdict
    transitions = defaultdict(list)
    counter = itertools.count()
    stack = []
    steps = []

    def snapshot(tok, new_transitions):
        copy_trans = {s: list(lst) for s, lst in transitions.items()}
        stack_repr = [f"[{frag.start}->{frag.accept}]" for frag in stack]
        steps.append({'tok': tok, 'stack': list(stack_repr), 'transitions': copy_trans, 'new': new_transitions})

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
            transitions[s].append((EPS, f1.start))
            transitions[s].append((EPS, f2.start))
            transitions[f1.accept].append((EPS, a))
            transitions[f2.accept].append((EPS, a))
            new_trans += [(s, EPS, f1.start), (s, EPS, f2.start), (f1.accept, EPS, a), (f2.accept, EPS, a)]
            stack.append(Fragment(s, a))
        elif tok == '*':
            f = stack.pop()
            s, a = next(counter), next(counter)
            transitions[s].append((EPS, f.start))
            transitions[s].append((EPS, a))
            transitions[f.accept].append((EPS, f.start))
            transitions[f.accept].append((EPS, a))
            new_trans += [(s, EPS, f.start), (s, EPS, a), (f.accept, EPS, f.start), (f.accept, EPS, a)]
            stack.append(Fragment(s, a))
        elif tok == '+':
            f = stack.pop()
            s, a = next(counter), next(counter)
            transitions[s].append((EPS, f.start))
            transitions[f.accept].append((EPS, f.start))
            transitions[f.accept].append((EPS, a))
            new_trans += [(s, EPS, f.start), (f.accept, EPS, f.start), (f.accept, EPS, a)]
            stack.append(Fragment(s, a))
        elif tok == '?':
            f = stack.pop()
            s, a = next(counter), next(counter)
            transitions[s].append((EPS, f.start))
            transitions[s].append((EPS, a))
            transitions[f.accept].append((EPS, a))
            new_trans += [(s, EPS, f.start), (s, EPS, a), (f.accept, EPS, a)]
            stack.append(Fragment(s, a))
        else:
            raise ValueError(f"Symbole non supporté: {tok}")
        snapshot(tok, new_trans)

    if len(stack) != 1:
        raise ValueError("Expression invalide : pile finale ≠ 1")

    frag = stack.pop()
    final_nfa = {'start': frag.start, 'accept': frag.accept, 'transitions': dict(transitions)}
    return steps, final_nfa

# -------------------------
# Conversion NFA -> DFA
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
    start = nfa['start']
    accept = nfa['accept']
    symbols = sorted(set(sym for lst in transitions.values() for sym,_ in lst if sym and sym != EPS))

    start_set = frozenset(epsilon_closure({start}, transitions))
    unmarked = [start_set]
    dfa_states = {start_set: 0}
    dfa_trans = {}
    dfa_accepts = set()
    if accept in start_set:
        dfa_accepts.add(start_set)

    sink = frozenset({'sink'})
    while unmarked:
        T = unmarked.pop()
        for sym in symbols:
            U = frozenset(epsilon_closure(move(T, sym, transitions), transitions))
            if not U:
                dfa_states.setdefault(sink, len(dfa_states))
                dfa_trans[(T, sym)] = sink
                continue
            if U not in dfa_states:
                dfa_states[U] = len(dfa_states)
                unmarked.append(U)
                if accept in U:
                    dfa_accepts.add(U)
            dfa_trans[(T, sym)] = U

    for sym in symbols:
        dfa_trans[(sink, sym)] = sink

    return {'states': list(dfa_states.keys()), 'start': start_set, 'accepts': dfa_accepts, 'transitions': dfa_trans, 'symbols': symbols}

# -------------------------
# Utility : build DOT (organisé et stable)
# -------------------------
def transitions_to_dot(transitions, new_edges=None, start=None, accept=None):
    g = graphviz.Digraph()
    g.attr(rankdir='LR', ranksep='1', nodesep='0.5')
    g.attr('node', shape='circle', fixedsize='true', width='1', height='1', fontsize='12')
    if start is not None:
        g.node('start', shape='point', width='0.1', height='0.1', fixedsize='true')
        g.edge('start', str(start))
  #  if start is not None:
  #      g.node('start', shape='point')
  #      g.edge('start', str(start))

    all_states = set(transitions.keys())
    for s, lst in transitions.items():
        for _, d in lst:
            all_states.add(d)

    for s in sorted(all_states, key=lambda x: str(x)):
        if s == accept:
            g.node(str(s), shape='doublecircle')
        else:
            g.node(str(s))

    for s, lst in transitions.items():
        for sym, d in lst:
            color = "red" if new_edges and (s, sym, d) in new_edges else "black"
            label = EPS if sym == EPS or sym is None else sym
            g.edge(str(s), str(d), label=label, color=color)

    return g

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Algorithme de Thompson", layout="wide")

col_logo, col_title = st.columns([1,5])
with col_logo:
    st.image("logo.png", width=100)
with col_title:
    st.title("Algorithme de Thompson")
    st.caption("Construction de l’automate de Thompson et conversion NFA → DFA")

with st.sidebar:
    st.header("Entrée")
    regex = st.text_input("Expression régulière", value="ab")
    build = st.button("Construire l'automate")
    show_dfa = st.checkbox("Afficher le DFA (après NFA)", value=False)

if 'steps' not in st.session_state:
    st.session_state.steps = []
if 'final_nfa' not in st.session_state:
    st.session_state.final_nfa = None
if 'idx' not in st.session_state:
    st.session_state.idx = 0

col1, col2 = st.columns([1,2])
postfix_box = col1.empty()
info_box = col1.empty()
graph_box = col2.empty()
nav1, nav2 = col1.columns([1,1])
prev_btn = nav1.button("← Étape précédente")
next_btn = nav2.button("Étape suivante")

if build:
    try:
        regex2 = insert_concat(regex.strip())
        postfix = to_postfix(regex2)
        steps, final_nfa = thompson_with_steps(postfix)
        st.session_state.steps = steps
        st.session_state.final_nfa = final_nfa
        st.session_state.idx = 0
        st.success("Automate NFA construit.")
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.session_state.steps = []
        st.session_state.final_nfa = None

if st.session_state.steps:
    if next_btn and st.session_state.idx < len(st.session_state.steps)-1:
        st.session_state.idx += 1
    if prev_btn and st.session_state.idx > 0:
        st.session_state.idx -= 1

    idx = st.session_state.idx
    step = st.session_state.steps[idx]
    postfix_box.markdown(f"**Étape {idx+1}/{len(st.session_state.steps)} — Symbole traité :** {step['tok']}")
    info_box.write(f"Pile : {step['stack']}")
    dot = transitions_to_dot(step['transitions'], step.get('new', []),
                             start=st.session_state.final_nfa['start'],
                             accept=st.session_state.final_nfa['accept'])
    graph_box.graphviz_chart(dot.source)

    if show_dfa and st.session_state.final_nfa:
        dfa = nfa_to_dfa(st.session_state.final_nfa)
        st.subheader("DFA correspondant")
        gdfa = graphviz.Digraph()
        gdfa.attr(rankdir='LR')
        for state in dfa['states']:
            if 'sink' in state:
                gdfa.node("∅", shape='circle', style='filled', fillcolor='lightgrey', width='1', height='1', fixedsize='true')
                continue
            name = "{" + ",".join(map(str, sorted(state))) + "}"
            shape = 'doublecircle' if state in dfa['accepts'] else 'circle'
            gdfa.node(name, shape=shape, width='1', height='1', fixedsize='true')
        for (src, sym), dest in dfa['transitions'].items():
            src_name = "∅" if 'sink' in src else "{" + ",".join(map(str, sorted(src))) + "}"
            dest_name = "∅" if 'sink' in dest else "{" + ",".join(map(str, sorted(dest))) + "}"
            gdfa.edge(src_name, dest_name, label=sym)
        st.graphviz_chart(gdfa.source)
else:
    st.info("Entrez une expression régulière et cliquez sur *Construire l'automate*.")
