"""Keyword / skill extraction using a curated tech vocabulary.

Kept dependency-light and deterministic so matching does not require the LLM.
A spaCy model is used opportunistically for noun-chunk extraction when
available, but the app works without it.
"""

from __future__ import annotations

import re

# A pragmatic, extensible vocabulary of common skills/tools. Matching is
# case-insensitive and whole-token where possible.
SKILL_VOCAB: frozenset[str] = frozenset(
    {
        "python",
        "java",
        "javascript",
        "typescript",
        "go",
        "golang",
        "rust",
        "c++",
        "c#",
        "ruby",
        "php",
        "scala",
        "kotlin",
        "swift",
        "sql",
        "nosql",
        "r",
        "react",
        "angular",
        "vue",
        "svelte",
        "next.js",
        "node.js",
        "express",
        "django",
        "flask",
        "fastapi",
        "spring",
        "rails",
        ".net",
        "laravel",
        "aws",
        "azure",
        "gcp",
        "kubernetes",
        "docker",
        "terraform",
        "ansible",
        "jenkins",
        "gitlab",
        "github actions",
        "ci/cd",
        "linux",
        "bash",
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "elasticsearch",
        "kafka",
        "rabbitmq",
        "spark",
        "hadoop",
        "airflow",
        "snowflake",
        "dbt",
        "machine learning",
        "deep learning",
        "nlp",
        "computer vision",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "pandas",
        "numpy",
        "spacy",
        "llm",
        "langchain",
        "transformers",
        "hugging face",
        "rest",
        "graphql",
        "grpc",
        "microservices",
        "distributed systems",
        "agile",
        "scrum",
        "kanban",
        "jira",
        "product management",
        "data analysis",
        "data engineering",
        "data science",
        "etl",
        "figma",
        "ux",
        "ui",
        "accessibility",
        "seo",
        "html",
        "css",
        "tailwind",
        "sass",
        "webpack",
        "vite",
        "pytest",
        "jest",
        "cypress",
        "selenium",
        "playwright",
        "security",
        "oauth",
        "jwt",
        "encryption",
        "networking",
        "leadership",
        "mentoring",
        "communication",
        "stakeholder management",
    }
)


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#/-]*", re.IGNORECASE)


def extract_skills(text: str) -> list[str]:
    """Return the ordered, de-duplicated set of known skills present in text."""

    lowered = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    # Multi-word skills first (longest match wins visually).
    for skill in sorted(SKILL_VOCAB, key=len, reverse=True):
        if skill in seen:
            continue
        pattern = re.escape(skill)
        # Word-ish boundary that tolerates symbols like c++/.net.
        if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", lowered):
            found.append(skill)
            seen.add(skill)
    return sorted(found)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
