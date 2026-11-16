from app.langmap import flores_code

def test_known_codes():
    assert flores_code("uz") == "uzn_Latn"
    assert flores_code("az") == "azj_Latn"
    assert flores_code("tg") == "tgk_Cyrl"

def test_unknown_and_none():
    assert flores_code("ru") is None
    assert flores_code(None) is None
