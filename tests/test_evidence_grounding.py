import json

import pytest

from src.evidence.evidence_selector import EvidenceSelector, split_sentences
from src.evidence.claim_evidence_mapper import ClaimEvidenceMapper
from src.evidence.citation_validator import CitationValidator
from src.evidence.verdict_classifier import VerdictClassifier
from src.evidence.evidence_grounding_pipeline import EvidenceGroundingPipeline
from src.retrieval.evidence_aware_reranker import EvidenceAwareReranker


CORPUS = {
    "doc-1": {
        "corpus_id": "doc-1",
        "title": "Protein trial",
        "text": "Protein therapy improves patient survival. The intervention was not associated with tumor shrinkage.",
    },
    "doc-2": {
        "corpus_id": "doc-2",
        "title": "Contradictory trial",
        "text": "Protein therapy decreases patient survival. It does not improve outcomes.",
    },
}


def test_sentence_splitting_preserves_exact_text_and_indices():
    split = split_sentences("First result is positive. Second result is negative.")
    assert [item["sentence_index"] for item in split] == [1, 2]
    assert split[0]["sentence"] == "First result is positive."
    assert split[1]["sentence"] == "Second result is negative."


def test_evidence_selector_handles_empty_abstract():
    selector = EvidenceSelector(relevance_threshold=-1.0)
    assert selector.select("protein therapy", [{"document_id": "doc-1", "text": ""}]) == []


def test_evidence_selector_supports_multiple_documents_and_preserves_traceability():
    selector = EvidenceSelector(relevance_threshold=-1.0, max_sentences_per_document=2, max_total_sentences=4)
    selected = selector.select(
        "protein therapy improves survival",
        [
            {"document_id": "doc-1", "text": "Protein therapy improves survival."},
            {"document_id": "doc-2", "text": "Protein therapy decreases survival."},
        ],
    )
    assert selected
    assert all(item["document_id"] in {"doc-1", "doc-2"} for item in selected)
    assert all("sentence_index" in item and "source_traceability" in item for item in selected)


def test_evidence_selector_removes_duplicate_texts():
    selector = EvidenceSelector(relevance_threshold=-1.0)
    selected = selector.select(
        "protein therapy",
        [
            {"document_id": "doc-1", "text": "Protein therapy improves outcomes."},
            {"document_id": "doc-2", "text": "Protein therapy improves outcomes."},
        ],
    )
    assert len(selected) == 1
    assert set(selected[0]["source_document_ids"]) == {"doc-1", "doc-2"}


@pytest.mark.parametrize(
    ("claim", "sentence", "expected"),
    [
        ("protein therapy improves survival", "Protein therapy improves survival.", "SUPPORTS"),
        ("protein therapy improves survival", "Protein therapy decreases survival.", "REFUTES"),
        ("protein therapy improves survival", "The weather is sunny.", "NEUTRAL"),
    ],
)
def test_claim_evidence_mapping_relationships(claim, sentence, expected):
    mapper = ClaimEvidenceMapper(relevance_threshold=-1.0)
    mappings = mapper.map(
        "q1",
        claim,
        [{"document_id": "doc-1", "evidence_sentence": sentence, "sentence_index": 1, "evidence_relevance_score": 0.8}],
    )
    assert mappings[0]["relationship"] == expected


def test_claim_evidence_mapper_handles_insufficient_relevance():
    mapper = ClaimEvidenceMapper(relevance_threshold=0.75)
    mappings = mapper.map(
        "q1",
        "protein therapy",
        [{"document_id": "doc-1", "evidence_sentence": "protein therapy", "sentence_index": 1, "evidence_relevance_score": 0.1}],
    )
    assert mappings[0]["relationship"] == "INSUFFICIENT"


def test_citation_validator_accepts_valid_source_sentence():
    validator = CitationValidator(CORPUS)
    validation = validator.validate([{
        "document_id": "doc-1",
        "sentence_index": 1,
        "evidence_sentence": "Protein therapy improves patient survival.",
        "source_metadata": {"corpus_id": "doc-1"},
    }])
    assert validation["all_valid"] is True


def test_citation_validator_rejects_invalid_document_and_index():
    validator = CitationValidator(CORPUS)
    validation = validator.validate([
        {"document_id": "missing", "sentence_index": 1, "evidence_sentence": "x"},
        {"document_id": "doc-1", "sentence_index": 99, "evidence_sentence": "x"},
    ])
    assert validation["invalid_count"] == 2
    assert {item["reason"] for item in validation["invalid"]} == {
        "document_id_not_in_corpus",
        "sentence_index_out_of_range",
    }


def test_citation_validator_rejects_modified_text():
    validator = CitationValidator(CORPUS)
    validation = validator.validate([{
        "document_id": "doc-1",
        "sentence_index": 1,
        "evidence_sentence": "Protein therapy worsens patient survival.",
    }])
    assert validation["invalid"][0]["reason"] == "evidence_text_modified"


def test_citation_validator_handles_duplicate_citations():
    validator = CitationValidator(CORPUS)
    mapping = {
        "document_id": "doc-1",
        "sentence_index": 1,
        "evidence_sentence": "Protein therapy improves patient survival.",
    }
    validation = validator.validate([mapping, dict(mapping)])
    assert validation["valid_count"] == 1
    assert validation["invalid"][0]["reason"] == "duplicate_citation"


def test_verdict_classifier_support_refutes_and_abstention():
    validator = CitationValidator(CORPUS)
    classifier = VerdictClassifier(confidence_threshold=0.40)
    support_mapping = [{
        "document_id": "doc-1",
        "sentence_index": 1,
        "evidence_sentence": "Protein therapy improves patient survival.",
        "evidence_relevance_score": 0.9,
        "relationship": "SUPPORTS",
    }]
    validation = validator.validate([{
        "document_id": "doc-1",
        "sentence_index": 1,
        "evidence_sentence": "Protein therapy improves patient survival.",
    }])
    assert classifier.classify(support_mapping, validation)["verdict"] == "SUPPORTS"

    refute_mapping = [{
        "document_id": "doc-2",
        "sentence_index": 1,
        "evidence_sentence": "Protein therapy decreases patient survival.",
        "evidence_relevance_score": 0.9,
        "relationship": "REFUTES",
    }]
    refute_validation = CitationValidator(CORPUS).validate([{
        "document_id": "doc-2",
        "sentence_index": 1,
        "evidence_sentence": "Protein therapy decreases patient survival.",
    }])
    assert classifier.classify(refute_mapping, refute_validation)["verdict"] == "REFUTES"


def test_verdict_classifier_conflicting_and_low_confidence_abstain():
    classifier = VerdictClassifier(confidence_threshold=0.90)
    mappings = [
        {"document_id": "doc-1", "sentence_index": 1, "evidence_relevance_score": 0.5, "relationship": "SUPPORTS"},
        {"document_id": "doc-2", "sentence_index": 1, "evidence_relevance_score": 0.5, "relationship": "REFUTES"},
    ]
    validation = {
        "valid": [
            {"document_id": "doc-1", "sentence_index": 1},
            {"document_id": "doc-2", "sentence_index": 1},
        ],
        "invalid": [],
    }
    result = classifier.classify(mappings, validation)
    assert result["verdict"] == "NOT_ENOUGH_INFO"
    assert result["abstention_reason"] in {"conflicting_evidence_unresolved", "confidence_below_threshold"}


def test_verdict_classifier_invalid_decisive_citation_abstains():
    classifier = VerdictClassifier(confidence_threshold=0.40)
    mappings = [{
        "document_id": "missing",
        "sentence_index": 1,
        "evidence_relevance_score": 0.9,
        "relationship": "SUPPORTS",
    }]
    result = classifier.classify(mappings, {"valid": [], "invalid": [{"reason": "document_id_not_in_corpus"}]})
    assert result["verdict"] == "NOT_ENOUGH_INFO"
    assert result["abstention_reason"] == "decisive_citations_failed_validation"


def test_pipeline_successful_verification_and_structured_output():
    pipeline = EvidenceGroundingPipeline(
        CORPUS,
        evidence_selector=EvidenceSelector(relevance_threshold=-1.0, max_sentences_per_document=1),
        verdict_classifier=VerdictClassifier(confidence_threshold=0.20),
    )
    result = pipeline.process(
        "Protein therapy improves patient survival.",
        claim_id="q1",
        candidates=[{"document_id": "doc-1", "retrieval_score": 0.9, "rank": 1}],
    )
    assert result["claim_id"] == "q1"
    assert result["final_verdict"] in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}
    assert "citation_validation" in result
    assert "processing_latency_seconds" in result


def test_pipeline_no_retrieved_evidence_abstains():
    pipeline = EvidenceGroundingPipeline(CORPUS)
    result = pipeline.process("unknown topic", claim_id="q0", candidates=[], retrieval_method="dense")
    assert result["final_verdict"] == "NOT_ENOUGH_INFO"
    assert result["abstention_reason"] == "no_decisive_citations"


def test_evidence_aware_reranker_accepts_grounding_scores():
    reranker = EvidenceAwareReranker({"cross_encoder": 0.8, "evidence_support": 0.1, "evidence_refutation": 0.1})
    candidates = [
        {"document_id": "doc-1", "text": "Protein therapy improves survival.", "retrieval_score": 0.8, "rank": 1},
        {"document_id": "doc-2", "text": "Protein therapy decreases survival.", "retrieval_score": 0.7, "rank": 2},
    ]
    ranked = reranker.rerank_with_evidence(
        "protein therapy improves survival",
        candidates,
        {"doc-1": {"support": 0.9, "refutation": 0.0}, "doc-2": {"support": 0.0, "refutation": 0.8}},
        cross_encoder_scores=[0.8, 0.7],
    )
    assert [row["document_id"] for row in ranked] == ["doc-1", "doc-2"]
