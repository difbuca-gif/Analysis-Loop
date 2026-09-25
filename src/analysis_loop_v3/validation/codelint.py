"""생성 코드의 명백한 위반을 조기에 찾는 비보안 정적 검사."""

from __future__ import annotations

import ast

from ..contracts import (
    ValidationCategory,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

# 분석에 필요한 import만 허용한다. 실제 격리는 sandbox가 담당한다.
ALLOWED_IMPORTS = frozenset({
    "pandas", "numpy", "scipy", "sklearn", "statsmodels", "math", "statistics",
    "matplotlib", "tqdm", "json",
    "xgboost", "lightgbm", "linearmodels", "mlxtend", "ruptures",
})

# 정적 lint는 명백한 위험 패턴만 빠르게 거른다.
BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit",
})
BANNED_ATTRIBUTES = frozenset({
    # 던더 우회
    "__class__", "__bases__", "__subclasses__", "__globals__", "__code__",
    "__dict__", "__builtins__", "__mro__", "__reduce__", "__getattribute__",
    # 위험 호출
    "system", "popen", "environ", "modules", "urlopen", "read_pickle", "to_pickle",
    # 주입된 pandas/numpy를 통한 파일 시스템 접근. 컨테이너 전까지는 흔한 경로를
    # 정적 단계에서도 막는다(이 목록 자체가 보안 경계라는 뜻은 아니다).
    "read_csv", "to_csv", "read_parquet", "to_parquet", "read_json", "to_json",
    "read_excel", "to_excel", "read_feather", "to_feather", "read_hdf", "to_hdf",
    # 알려진 간접 모듈 접근을 조기에 거른다. 격리 실행을 대체하지 않는다.
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "ctypes", "builtins", "codecs", "io", "urllib", "requests", "pickle",
})

def _issue(
    code: str,
    message: str,
    severity: ValidationSeverity,
    category: ValidationCategory = ValidationCategory.INTEGRITY,
) -> ValidationIssue:
    return ValidationIssue(
        code=code, message=message, severity=severity, category=category,
    )

def _entrypoint_issues(tree: ast.Module, function_name: str) -> list[ValidationIssue]:
    """analyze(df) 하나가 있고 인자가 하나인가."""
    entry = next(
        (n for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == function_name),
        None,
    )
    if entry is None:
        return [_issue(
            "missing_entrypoint", f"{function_name}(df) 함수가 없다.",
            ValidationSeverity.REPAIRABLE,
        )]
    if len(entry.args.args) != 1:
        return [_issue(
            "bad_signature",
            f"{function_name}은 인자를 정확히 1개(df) 받아야 한다 "
            f"(실제 {len(entry.args.args)}개).",
            ValidationSeverity.REPAIRABLE,
        )]
    return []

def _node_issues(node: ast.AST) -> list[ValidationIssue]:
    """노드 하나의 위반. import는 alias마다 나올 수 있어 목록으로 돌려준다."""
    safety = ValidationCategory.SAFETY
    if isinstance(node, ast.Import):
        return [
            _issue("import_not_allowed", f"허용되지 않은 import: {alias.name}",
                   ValidationSeverity.REPAIRABLE, safety)
            for alias in node.names
            if alias.name.split(".")[0] not in ALLOWED_IMPORTS
        ]
    if isinstance(node, ast.ImportFrom):
        # level이 있으면 상대 import다. 생성 코드는 패키지가 아니라 항상 거부한다.
        if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
            return [_issue(
                "import_not_allowed", f"허용되지 않은 import: {node.module}",
                ValidationSeverity.REPAIRABLE, safety,
            )]
    elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
        return [_issue("banned_name", f"금지된 이름 사용: {node.id}",
                       ValidationSeverity.FATAL, safety)]
    elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTES:
        return [_issue("banned_attribute", f"금지된 속성 접근: .{node.attr}",
                       ValidationSeverity.FATAL, safety)]
    return []

def lint_program(code: str, *, function_name: str = "analyze") -> ValidationReport:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationReport(stage="code", issues=[_issue(
            "syntax_error", f"문법 오류: {exc.msg} (line {exc.lineno})",
            ValidationSeverity.REPAIRABLE,
        )])

    issues = _entrypoint_issues(tree, function_name)
    for node in ast.walk(tree):
        issues.extend(_node_issues(node))
    return ValidationReport(stage="code", issues=issues)
