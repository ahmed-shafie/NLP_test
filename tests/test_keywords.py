from aiapply_lite.core import keywords


def test_extract_skills_finds_known_tokens() -> None:
    text = "Experienced Python developer with FastAPI, Docker and AWS. Also NLP."
    found = keywords.extract_skills(text)
    assert "python" in found
    assert "fastapi" in found
    assert "docker" in found
    assert "aws" in found
    assert "nlp" in found


def test_extract_skills_is_deduped_and_sorted() -> None:
    text = "python python PYTHON react react"
    found = keywords.extract_skills(text)
    assert found == sorted(set(found))
    assert found.count("python") == 1


def test_extract_skills_no_false_substring() -> None:
    # "r" should not match inside random words
    text = "we deliver wonderful products"
    assert "r" not in keywords.extract_skills(text)


def test_tokenize() -> None:
    assert keywords.tokenize("Node.js and C++") == ["node.js", "and", "c++"]
