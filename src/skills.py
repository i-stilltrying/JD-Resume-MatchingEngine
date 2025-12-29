# src/skills.py
from __future__ import annotations

from typing import Dict, List, Tuple

from .preprocess import normalize_text, simple_tokenize

# Buckets tuned for: "AI Engineer – Agentic & Intelligent Systems"
# Goal: prevent a resume without UI/UX or without agentic systems from ranking #1.

_BUCKETS: Dict[str, List[str]] = {
    # 1) Agent frameworks, orchestration, multi-agent, LLM pipelines
    "agentic_llm_systems": [
        "llm", "llms", "agent", "agents", "agentic", "autonomous",
        "multi-agent", "orchestration", "tool calling", "function calling",
        "planning", "reasoning", "prompt", "prompting", "prompt engineering",
        "langchain", "autogen", "crew", "langgraph",
    ],

    # 2) Vector DB, retrieval, memory, planning architectures
    "vector_memory_planning": [
        "vector", "vector store", "vector database", "embedding", "embeddings",
        "retrieval", "rag", "semantic search", "memory architecture", "memory",
        "faiss", "milvus", "weaviate", "pinecone", "chroma", "pgvector",
        "planning architecture", "planner",
    ],

    # 3) Observability + evaluation + reliability + drift + accountability
    "observability_eval_reliability": [
        "observability", "monitoring", "monitor", "metrics", "logging", "logs",
        "tracing", "opentelemetry", "dashboard", "alert", "alerts", "alerting",
        "prometheus", "grafana",
        "evaluation", "eval", "benchmark", "benchmarks", "deterministic",
        "debuggable", "reliable", "reliability", "accountable",
        "drift", "drift detection", "closed-loop", "retraining",
        "uncertainty", "explainable", "explainability",
    ],

    # 4) UI/UX + frontend + notebook experiences + HCI
    "ui_ux_frontend": [
        "ui", "ux", "frontend", "hci", "human-computer interaction",
        "angular", "typescript", "javascript", "figma", "design system",
        "accessibility", "visualization", "explainable ui",
        "notebook", "jupyter", "vscode notebooks", "notebooks",
        "prototype", "usability testing", "human in the loop",
    ],

    # 5) Backend + distributed systems + APIs + DevOps / CI/CD
    "backend_api_devops": [
        "python", "java", "golang", "go",
        "backend", "microservice", "microservices", "distributed systems",
        "system design", "api", "apis", "rest", "grpc", "json",
        "framework", "fastapi", "flask", "django", "spring",
        "docker", "kubernetes", "k8s", "helm",
        "ci/cd", "cicd", "pipeline", "pipelines",
        "mlflow", "kubeflow", "argo", "argo workflows", "kserve",
        "unit tests", "testing",
    ],
}


def bucket_coverage(text: str) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """
    Returns:
      hits[bucket] = 1 if any keywords/phrases matched, else 0
      matched[bucket] = list of matched keywords/phrases

    Matching strategy (to avoid false positives):
      - Single words are matched against a token set (exact token match).
      - Phrases (contain space or '-') are matched via substring on normalized text.
    """
    t = normalize_text(text)
    tokens = set(simple_tokenize(t))

    hits: Dict[str, int] = {}
    matched: Dict[str, List[str]] = {}

    for bucket, kws in _BUCKETS.items():
        found: List[str] = []
        for kw in kws:
            kw_norm = kw.strip().lower()
            if not kw_norm:
                continue

            # phrase match (contains space or hyphen)
            if (" " in kw_norm) or ("-" in kw_norm):
                if kw_norm in t:
                    found.append(kw)
            else:
                # token match (prevents "go" matching inside "mongo")
                if kw_norm in tokens:
                    found.append(kw)

        hits[bucket] = 1 if found else 0
        matched[bucket] = sorted(set(found), key=lambda x: x.lower())

    return hits, matched


def missing_bucket_count(hits: Dict[str, int]) -> int:
    return sum(1 for v in hits.values() if v == 0)


def apply_must_have_penalty(
    score: float,
    hits: Dict[str, int],
    penalty_factor: float = 0.85,
    missing_threshold: int = 2,
) -> float:
    """
    If missing >= missing_threshold buckets, apply multiplicative penalty.
    Default: missing 2+ buckets => score * 0.85
    """
    miss = missing_bucket_count(hits)
    if miss >= missing_threshold:
        return score * penalty_factor
    return score
