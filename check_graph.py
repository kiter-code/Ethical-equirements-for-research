"""Проверка графа переходов без запуска оркестратора:
достижимость и сироты."""
from orchestrator import Orchestrator as O

roles = set(O.ROLES)
transition_sources = set(O.TRANSITIONS)
transition_targets = set().union(*O.TRANSITIONS.values()) if O.TRANSITIONS else set()

for role in O.ROLES:
    allowed = O.TRANSITIONS.get(role, set())
    forbidden = sorted(roles - allowed)
    print(f"{role:<11} -> {sorted(allowed)} запрещено: {forbidden}")

seen, todo = set(), [O.INITIAL_ROLE]
while todo:
    role = todo.pop()
    if role not in seen:
        seen.add(role)
        todo += O.TRANSITIONS.get(role, set())

print("недостижимы:", sorted(roles - seen) or "нет")
print("без интента:", sorted(roles - set(O.INTENT_TO_ROLE.values())) or "нет")

named = {O.INITIAL_ROLE, O.FALLBACK_ROLE,
         *O.INTENT_TO_ROLE.values(),
         *transition_sources,
         *transition_targets}
print("названы, но не описаны в ROLES:", sorted(named - roles) or "нет")

assert len(roles) >= 2, "нужно определить не менее двух ролей"
assert O.INITIAL_ROLE in roles, "начальная роль отсутствует в ROLES"
assert O.FALLBACK_ROLE in roles, "роль по умолчанию отсутствует в ROLES"
assert roles <= transition_sources, "не для каждой роли описаны переходы"
assert transition_targets <= roles, "граф ссылается на неизвестную роль"
assert not (roles - seen), "в графе есть недостижимые роли"
assert not (roles - set(O.INTENT_TO_ROLE.values())), "есть роли без интентов"
assert any(roles - O.TRANSITIONS[role] for role in roles), (
    "автомат должен запрещать хотя бы один переход"
)

print("Граф переходов: OK")
