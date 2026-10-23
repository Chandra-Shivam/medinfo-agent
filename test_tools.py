from agent import search_labels

Q = "pediatric use in juvenile arthritis dosing"


def test_no_cross_drug_leak():
    out = search_labels.invoke({"drug_name": "ibuprofen", "question": Q})
    assert "naproxen" not in out.lower()


def test_control_still_finds_own_drug():
    out = search_labels.invoke({"drug_name": "naproxen", "question": Q})
    assert "[naproxen |" in out