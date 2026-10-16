"""A function-by-function map of this code: where everything is, who calls it, what it calls.

    uv run python scripts/code_map.py                      # (re)write docs/CODE_MAP.md
    uv run python scripts/code_map.py who planner_node     # callers and callees of one function
    uv run python scripts/code_map.py compare graph.json   # check this map against a graphify graph.json

It reads the code with Python's own parser (no AI, no network), so it is always current: run it again
after code changes. A call is found by name: `foo()` in the same file, `imported_name()`, `module.func()`,
`self.method()`, and `obj.method()` when only one class defines that method. A function passed as an
argument (`asyncio.to_thread(_load_context, ...)`) counts too and is shown with a `~`.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["main.py", "app", "scripts"]
OUT = ROOT / "docs" / "CODE_MAP.md"
NODES = {"ingress", "planner", "guard", "executor", "renderer"}


# ------------------------------------------------------------------------------------ reading


class Fn:
    def __init__(self, module: str, qname: str, node: ast.AST, path: str):
        self.module, self.qname, self.node, self.path = module, qname, node, path
        self.name = qname.split(".")[-1]
        self.id = f"{module}.{qname}"
        self.line = node.lineno if node is not None else 1
        self.doc = ""
        self.route = ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.doc = (ast.get_docstring(node) or "").strip().split("\n")[0][:110]
            self.kind = "class" if isinstance(node, ast.ClassDef) else "def"
            for d in getattr(node, "decorator_list", []):
                is_route = (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Attribute)
                    and d.func.attr in ("get", "post", "put", "delete")
                    and d.args
                )
                if is_route:
                    self.route = f"{d.func.attr.upper()} {getattr(d.args[0], 'value', '?')}"
        else:
            self.kind = "module"


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


class Project:
    def __init__(self) -> None:
        self.fns: dict[str, Fn] = {}
        self.trees: dict[str, ast.Module] = {}
        self.paths: dict[str, str] = {}
        self.imports: dict[str, dict[str, str]] = {}  # module -> local name -> dotted target
        self.calls: dict[str, dict[str, tuple[str, int]]] = defaultdict(
            dict
        )  # caller -> {callee: (kind, line)}
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.wiring: list[tuple[str, str, str]] = []  # (from node, to node, decided by)
        self._read()
        self._link()

    # -- parse every file, list its functions, classes, methods
    def _read(self) -> None:
        files = []
        for entry in SOURCES:
            p = ROOT / entry
            files += [p] if p.is_file() else sorted(p.rglob("*.py"))
        for path in files:
            if "__pycache__" in path.parts:
                continue
            mod = module_name(path)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            rel = path.relative_to(ROOT).as_posix()
            self.trees[mod], self.paths[mod] = tree, rel
            self.fns[f"{mod}.<module>"] = Fn(mod, "<module>", None, rel)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._add(Fn(mod, node.name, node, rel))
                elif isinstance(node, ast.ClassDef):
                    self._add(Fn(mod, node.name, node, rel))
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self._add(Fn(mod, f"{node.name}.{sub.name}", sub, rel))

    def _add(self, fn: Fn) -> None:
        self.fns[fn.id] = fn
        self.by_name[fn.name].append(fn.id)

    # -- imports: which local name means which project function or module
    def _imports_of(self, mod: str) -> dict[str, str]:
        table: dict[str, str] = {}
        for node in ast.walk(self.trees[mod]):
            if isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    table[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.Import):
                for a in node.names:
                    table[(a.asname or a.name).split(".")[0]] = (
                        a.name if a.asname else a.name.split(".")[0]
                    )
        return table

    def _resolve(self, mod: str, cls: str | None, expr: ast.AST) -> str | None:
        """The project function a name refers to, or None (library, builtin, unknown)."""
        imports = self.imports[mod]
        if isinstance(expr, ast.Name):
            local = f"{mod}.{expr.id}"
            if local in self.fns:
                return local
            target = imports.get(expr.id)
            return target if target in self.fns else None
        if isinstance(expr, ast.Attribute):
            base = expr.value
            if isinstance(base, ast.Name) and base.id in ("self", "cls") and cls:
                method = f"{mod}.{cls}.{expr.attr}"
                return method if method in self.fns else None
            if isinstance(base, ast.Name):
                target = imports.get(base.id)
                if target:
                    for candidate in (f"{target}.{expr.attr}",):
                        if candidate in self.fns:
                            return candidate
                local_cls = f"{mod}.{base.id}.{expr.attr}"
                if local_cls in self.fns:
                    return local_cls
            # obj.method(): only when exactly one project class defines that method
            owners = [
                i for i in self.by_name.get(expr.attr, []) if self.fns[i].qname.count(".") == 1
            ]
            return owners[0] if len(owners) == 1 else None
        return None

    def _link(self) -> None:
        for mod in self.trees:
            self.imports[mod] = self._imports_of(mod)
        for mod, tree in self.trees.items():
            module_fn = self.fns[f"{mod}.<module>"]
            self._collect(mod, None, tree, module_fn.id, top_level=True)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._collect(mod, None, node, f"{mod}.{node.name}")
                elif isinstance(node, ast.ClassDef):
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self._collect(mod, node.name, sub, f"{mod}.{node.name}.{sub.name}")
        self._wiring()

    def _collect(
        self, mod: str, cls: str | None, root: ast.AST, caller: str, top_level: bool = False
    ) -> None:
        def visit_stmt(stmt: ast.AST) -> None:
            holder = ast.Module(body=[stmt], type_ignores=[])
            visit(holder)

        def visit(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if top_level and isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    continue  # module level only: functions are collected on their own
                if isinstance(child, ast.Call):
                    target = self._resolve(mod, cls, child.func)
                    if target and target != caller:
                        self.calls[caller].setdefault(target, ("calls", child.lineno))
                    for arg in [*child.args, *[k.value for k in child.keywords]]:
                        passed = (
                            self._resolve(mod, cls, arg)
                            if isinstance(arg, (ast.Name, ast.Attribute))
                            else None
                        )
                        if passed and passed != caller:
                            self.calls[caller].setdefault(passed, ("passes", child.lineno))
                visit(child)

        if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in root.body:  # not the decorators: they run once, at import
                visit_stmt(stmt)
        else:
            visit(root)

    # -- the LangGraph wiring, read from the route_after_* functions
    def _wiring(self) -> None:
        for fn in self.fns.values():
            if fn.name.startswith("route_after_") and fn.node is not None:
                src = fn.name.removeprefix("route_after_")
                targets = {
                    c.value
                    for c in ast.walk(fn.node)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value in NODES
                }
                for t in sorted(targets):
                    self.wiring.append((src, t, fn.name))
                    node_fn = f"{fn.module}.{t}_node"
                    if node_fn in self.fns:
                        self.calls[fn.id].setdefault(node_fn, ("graph", fn.line))
        # execute_tool picks a handler by name from TOOL_REGISTRY: show it as an edge
        registry = "app.services_ai.mcp_server"
        dispatcher = f"{registry}.execute_tool"
        if registry in self.trees and dispatcher in self.fns:
            for node in ast.walk(self.trees[registry]):
                if (
                    isinstance(node, ast.AnnAssign)
                    and getattr(node.target, "id", "") == "TOOL_REGISTRY"
                ):
                    for value in node.value.values:
                        target = f"{registry}.{getattr(value, 'id', '')}"
                        if target in self.fns:
                            self.calls[dispatcher].setdefault(target, ("graph", node.lineno))

    # -- reverse view
    def callers(self, target: str) -> list[tuple[str, str]]:
        return sorted(
            (c, k) for c, edges in self.calls.items() for t, (k, _) in edges.items() if t == target
        )

    def find(self, name: str) -> list[Fn]:
        needle = name.lower()
        exact = [
            f for f in self.fns.values() if f.name.lower() == needle or f.qname.lower() == needle
        ]
        return exact or [
            f for f in self.fns.values() if needle in f.qname.lower() and f.kind != "module"
        ]


# ------------------------------------------------------------------------------------ writing


def short(p: Project, fid: str) -> str:
    f = p.fns[fid]
    base = f.module.split(".")[-1]
    return f"{base}.{f.qname}" if f.kind != "module" else f"{base} (module level)"


def where(p: Project, fid: str) -> str:
    f = p.fns[fid]
    return f"{f.path}:{f.line} {f.qname}()" if f.kind == "def" else f"{f.path}:{f.line} {f.qname}"


MARK = {"passes": "~", "graph": ">"}


def names(p: Project, items: list[tuple[str, str]], limit: int = 7) -> str:
    mark = {"passes": "~", "graph": ">"}
    out = [mark.get(k, "") + short(p, i) for i, k in items]
    more = f", +{len(out) - limit} more" if len(out) > limit else ""
    return ", ".join(out[:limit]) + more if out else "-"


SPINE = [
    (
        "1. Start: `uvicorn main:app` imports main.py",
        [
            "main.<module>",
            "firstrun.ensure_setup",
            "config.Settings",
            "db._build_engine",
            "agent.build_agent_graph",
            "main.lifespan",
            "db.init_db",
            "tool_client.start",
            "telegram.start",
        ],
    ),
    (
        "2. A Telegram message arrives",
        [
            "telegram.on_text",
            "main.handle_incoming",
            "onboarding.intercept",
            "agent.process_user_interaction_ex",
        ],
    ),
    (
        "3. The LangGraph agent runs five nodes",
        [
            "agent.ingress_node",
            "agent.planner_node",
            "agent.guard_node",
            "agent.executor_node",
            "agent.renderer_node",
        ],
    ),
    (
        "4. The executor reaches the database",
        [
            "tool_client.call_tool",
            "mcp_server.execute_tool",
            "mcp_server.log_expense",
            "ledger.record_transaction",
            "ledger._adjust_balance",
        ],
    ),
    ("5. The reply goes back", ["agent._outcome_from", "main._reply_for", "telegram.deliver"]),
]


def resolve_spine(p: Project, ref: str) -> str | None:
    mod, _, name = ref.partition(".")
    hits = [f.id for f in p.fns.values() if f.module.split(".")[-1] == mod and f.qname == name]
    return hits[0] if hits else None


def build_doc(p: Project) -> str:
    out: list[str] = []
    w = out.append
    count = sum(1 for f in p.fns.values() if f.kind in ("def", "class"))
    w("# Code map: every function, where it is, who calls it\n")
    w(
        "Generated by `scripts/code_map.py` from the code itself (run it again after changes). "
        f"{count} functions and classes in {len(p.trees)} files. In a terminal: "
        "`uv run python scripts/code_map.py who <function>`.\n"
    )
    w(
        "**Reading a line:** `line  name — what it does | called by: ... | calls: ...`. "
        "A `~` before a name means the function is passed as an argument (not called directly), "
        "e.g. `asyncio.to_thread(_load_context, ...)` or a LangGraph node registered in `build_agent_graph`. "
        "A `>` means control is handed over by a route (a `route_after_*` function sends the graph to the next node, "
        "`execute_tool` dispatches to a tool by name).\n"
    )

    w(
        "Cross-checked on 2026-10-16 against graphify's own graph of this project: it found 1,111 calls between "
        "functions of `app/`; this map finds 97% of them. The rest are calls into small helper functions defined "
        "*inside* another function (`run`, `load`, `work`), which this map counts as part of the function that "
        "contains them. Re-check any time with `scripts/code_map.py compare graph.json`.\n"
    )
    w("## The spine: follow one message from start to finish\n")
    for title, refs in SPINE:
        w(f"**{title}**\n")
        for ref in refs:
            fid = resolve_spine(p, ref)
            if fid:
                f = p.fns[fid]
                w(f"- `{where(p, fid)}` {('— ' + f.doc) if f.doc else ''}")
        w("")

    w("## The LangGraph wiring (read from the `route_after_*` functions)\n")
    w("| After this node | the graph can go to | decided by |\n|---|---|---|")
    w("| (start) | ingress | `build_agent_graph` |")
    for src, dst, by in p.wiring:
        w(f"| {src} | {dst} | `{by}` |")
    w("| renderer | (end) | `build_agent_graph` |\n")

    w("## Every file, function by function\n")
    by_module: dict[str, list[Fn]] = defaultdict(list)
    for f in p.fns.values():
        by_module[f.module].append(f)
    for mod in sorted(by_module):
        fns = sorted(by_module[mod], key=lambda f: (f.line, f.qname))
        shown = [f for f in fns if f.kind in ("def", "class") or p.calls.get(f.id)]
        if not shown:
            continue
        w(f"### `{p.paths[mod]}`\n")
        for f in shown:
            callees = [(t, k) for t, (k, _) in p.calls.get(f.id, {}).items()]
            callers = p.callers(f.id)
            label = f"`{f.qname}`" if f.kind != "module" else "*(runs when the file is imported)*"
            doc = f" — {f.doc}" if f.doc else ""
            route = f" **[{f.route}]**" if f.route else ""
            w(
                f"- `L{f.line}` {label}{route}{doc}  \n  called by: {names(p, callers)} | calls: {names(p, callees)}"
            )
        w("")
    return "\n".join(out) + "\n"


def show_who(p: Project, name: str) -> None:
    hits = p.find(name)
    if not hits:
        print(f"nothing called {name!r}")
        return
    for f in hits[:5]:
        print(f"\n{where(p, f.id)}")
        if f.doc:
            print(f"  {f.doc}")
        if f.route:
            print(f"  HTTP route: {f.route}")
        callers = p.callers(f.id)
        callees = [(t, k) for t, (k, _) in p.calls.get(f.id, {}).items()]
        print("  called by:" + ("" if callers else " -"))
        for c, k in callers:
            print(f"    {MARK.get(k, ' ')} {where(p, c)}")
        print("  calls:" + ("" if callees else " -"))
        for t, k in callees:
            print(f"    {MARK.get(k, ' ')} {where(p, t)}")


def compare(p: Project, graph_path: str) -> None:
    """Check this map against a graphify graph.json: same functions, same calls?"""
    import json

    g = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in g["nodes"]}
    theirs: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for e in g["links"]:
        if e.get("relation") not in ("calls", "indirect_call"):
            continue
        a, b = nodes.get(e["source"]), nodes.get(e["target"])
        if not a or not b or not str(a.get("source_file", "")).startswith(("app/", "main.py")):
            continue
        if not str(b.get("source_file", "")).startswith(("app/", "main.py")):
            continue
        theirs[(a["source_file"], a["label"].strip("()."))].add(
            (b["source_file"], b["label"].strip("()."))
        )
    mine: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for caller, edges in p.calls.items():
        cf = p.fns[caller]
        if cf.kind == "module" or not cf.path.startswith(("app/", "main.py")):
            continue
        for t in edges:
            tf = p.fns[t]
            if tf.kind != "module":
                mine[(cf.path, cf.name)].add((tf.path, tf.name))
    both = sorted(set(theirs) & set(mine))
    only_mine = len(set(mine) - set(theirs))
    tp = sum(len(theirs[k] & mine[k]) for k in both)
    their_total = sum(len(theirs[k]) for k in both)
    my_total = sum(len(mine[k]) for k in both)
    exact = sum(1 for k in both if theirs[k] == mine[k])
    print(f"functions in both maps (with calls): {len(both)}   only in mine: {only_mine}")
    print(f"identical call lists: {exact}/{len(both)} functions ({exact / max(len(both), 1):.0%})")
    print(
        f"graphify's calls that I also found: {tp}/{their_total} ({tp / max(their_total, 1):.0%})"
    )
    print(f"my calls that graphify also found:  {tp}/{my_total} ({tp / max(my_total, 1):.0%})")
    diffs = [(k, theirs[k] - mine[k], mine[k] - theirs[k]) for k in both if theirs[k] != mine[k]]
    for k, missing, extra in diffs[:8]:
        print(
            f"  {k[0]}:{k[1]}  graphify-only={sorted(x[1] for x in missing)[:4]}  mine-only={sorted(x[1] for x in extra)[:4]}"
        )


def main(argv: list[str]) -> int:
    project = Project()
    if argv[:1] == ["who"] and len(argv) > 1:
        show_who(project, argv[1])
    elif argv[:1] == ["compare"] and len(argv) > 1:
        compare(project, argv[1])
    else:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(build_doc(project), encoding="utf-8", newline="\n")
        print(f"wrote {OUT.relative_to(ROOT)} ({len(project.fns)} entries)")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
