import os
import re
import json
import logging
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np

# Configure logging via named logger to allow silencing from the main runner
import logging as syslog
logger = syslog.getLogger("metrics")

class LoggingAdapter:
    def info(self, msg, *args, **kwargs):
        logger.info(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs):
        logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs):
        logger.error(msg, *args, **kwargs)
    def exception(self, msg, *args, **kwargs):
        logger.exception(msg, *args, **kwargs)

logging = LoggingAdapter()

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab', 'corpora/wordnet', 'corpora/omw-1.4']:
        name = resource.split('/')[-1]
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_resources()

# Lazy loading of heavy models to keep import time fast and prevent unnecessary initialization
_sentence_transformer_model = None
_bertscore_scorer = None
_summac_scorer = None
_nli_pipeline = None

def get_sentence_transformer():
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        from sentence_transformers import SentenceTransformer
        logging.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_transformer_model

def get_bertscore():
    global _bertscore_scorer
    if _bertscore_scorer is None:
        import bert_score
        _bertscore_scorer = bert_score
    return _bertscore_scorer

def get_summac():
    global _summac_scorer
    if _summac_scorer is None:
        from summac.model_summac import SummaCConv
        logging.info("Initializing SummaCConv model...")
        _summac_scorer = SummaCConv(models=["vitc"], granularity="sentence", use_gpu=False)
    return _summac_scorer

def get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        logging.info("Initializing NLI fallback pipeline (facebook/bart-large-mnli)...")
        _nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)
    return _nli_pipeline


# 1. Lexical Overlap Metrics
def compute_rouge(reference: str, candidate: str) -> dict:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    if not reference.strip() or not candidate.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }

def compute_bleu(reference: str, candidate: str) -> float:
    """Computes sentence BLEU score using smoothing to prevent zero score for short texts."""
    if not reference.strip() or not candidate.strip():
        return 0.0
    
    ref_tokens = nltk.word_tokenize(reference)
    cand_tokens = nltk.word_tokenize(candidate)
    
    if not ref_tokens or not cand_tokens:
        return 0.0
        
    chencherry = SmoothingFunction()
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=chencherry.method1)

def compute_meteor(reference: str, candidate: str) -> float:
    """Computes METEOR score for robust paraphrasing evaluation."""
    if not reference.strip() or not candidate.strip():
        return 0.0
    
    ref_tokens = nltk.word_tokenize(reference)
    cand_tokens = nltk.word_tokenize(candidate)
    
    if not ref_tokens or not cand_tokens:
        return 0.0
        
    try:
        return meteor_score([ref_tokens], cand_tokens)
    except Exception as e:
        logging.warning(f"METEOR scoring failed: {e}")
        return 0.0


# 2. Semantic Similarity Metrics
def compute_bertscore(reference: str, candidate: str) -> dict:
    """Computes BERTScore precision, recall, and F1."""
    if not reference.strip() or not candidate.strip():
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}
    
    try:
        bertscore_lib = get_bertscore()
        P, R, F1 = bertscore_lib.score([candidate], [reference], lang="en", verbose=False)
        return {
            "bertscore_p": float(P[0].item()),
            "bertscore_r": float(R[0].item()),
            "bertscore_f1": float(F1[0].item())
        }
    except Exception as e:
        logging.warning(f"BERTScore computation failed: {e}")
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}

def compute_semantic_cosine(reference: str, candidate: str) -> float:
    """Computes cosine similarity of sentence embeddings between candidate and reference."""
    if not reference.strip() or not candidate.strip():
        return 0.0
    
    try:
        model = get_sentence_transformer()
        from sentence_transformers import util
        ref_emb = model.encode(reference, convert_to_tensor=True, show_progress_bar=False)
        cand_emb = model.encode(candidate, convert_to_tensor=True, show_progress_bar=False)
        return float(util.cos_sim(ref_emb, cand_emb).item())
    except Exception as e:
        logging.warning(f"Semantic cosine computation failed: {e}")
        return 0.0


# 3. Factual Consistency Metrics (SummaC with NLI Fallback)
def compute_factual_consistency(transcript: str, summary: str) -> dict:
    """
    Computes factual consistency score using SummaC.
    If SummaC fails to load or run, falls back to a custom NLI-based entailment check.
    """
    if not transcript.strip() or not summary.strip():
        return {"factual_consistency": 0.0, "method": "none"}
        
    # Attempt SummaC
    try:
        summac_scorer = get_summac()
        res = summac_scorer.score([transcript], [summary])
        score = float(res["scores"][0])
        return {"factual_consistency": score, "method": "summac"}
    except Exception as e:
        logging.info(f"SummaC unavailable or failed ({e}). Falling back to NLI model...")
        
    # NLI Fallback
    try:
        nli_pipe = get_nli_pipeline()
        
        summary_sentences = nltk.sent_tokenize(summary)
        if not summary_sentences:
            return {"factual_consistency": 0.0, "method": "nli_fallback"}
            
        scores = []
        label2id = getattr(nli_pipe.model.config, 'label2id', {})
        entail_label = None
        for label, idx in label2id.items():
            if 'entail' in label.lower():
                entail_label = label
                break
                
        if entail_label is None:
            entail_label = "LABEL_2"
            
        truncated_transcript = transcript[:3500]
        
        for sentence in summary_sentences:
            if not sentence.strip():
                continue
            res = nli_pipe({"text": truncated_transcript, "text_pair": sentence})
            if isinstance(res, dict):
                if res['label'] == entail_label:
                    scores.append(res['score'])
                else:
                    all_res = nli_pipe({"text": truncated_transcript, "text_pair": sentence}, return_all_scores=True)
                    entail_score = 0.0
                    for item in all_res:
                        if item['label'] == entail_label:
                            entail_score = item['score']
                            break
                    scores.append(entail_score)
            else:
                scores.append(0.0)
                
        avg_score = float(np.mean(scores)) if scores else 0.0
        return {"factual_consistency": avg_score, "method": "nli_fallback"}
        
    except Exception as nli_err:
        logging.error(f"NLI Fallback also failed: {nli_err}")
        return {"factual_consistency": 0.0, "method": "failed"}


# 4. Compression Ratio
def compute_compression_ratio(transcript: str, summary: str) -> float:
    """Computes compression ratio: len(transcript) / len(summary)"""
    cand_len = len(summary.strip())
    if cand_len == 0:
        return 0.0
    return len(transcript.strip()) / cand_len


# 5. Action-Item Extraction & Matching
def compute_action_item_metrics(ref_triples: list, cand_triples: list, threshold: float = 0.7) -> dict:
    """
    Computes precision, recall, and F1 over matched action-item triples.
    Matches triples based on the embedding similarity of their action texts.
    """
    if not ref_triples and not cand_triples:
        return {"action_item_precision": 1.0, "action_item_recall": 1.0, "action_item_f1": 1.0}
    if not ref_triples:
        return {"action_item_precision": 0.0, "action_item_recall": 1.0, "action_item_f1": 0.0}
    if not cand_triples:
        return {"action_item_precision": 1.0, "action_item_recall": 0.0, "action_item_f1": 0.0}

    ref_actions = [str(t.get('action', '')).strip() for t in ref_triples]
    cand_actions = [str(t.get('action', '')).strip() for t in cand_triples]
    
    ref_actions = [a for a in ref_actions if a]
    cand_actions = [a for a in cand_actions if a]
    
    if not ref_actions and not cand_actions:
        return {"action_item_precision": 1.0, "action_item_recall": 1.0, "action_item_f1": 1.0}
    if not ref_actions:
        return {"action_item_precision": 0.0, "action_item_recall": 1.0, "action_item_f1": 0.0}
    if not cand_actions:
        return {"action_item_precision": 1.0, "action_item_recall": 0.0, "action_item_f1": 0.0}
        
    try:
        model = get_sentence_transformer()
        from sentence_transformers import util
        
        ref_embs = model.encode(ref_actions, convert_to_tensor=True, show_progress_bar=False)
        cand_embs = model.encode(cand_actions, convert_to_tensor=True, show_progress_bar=False)
        
        sim_matrix = util.cos_sim(cand_embs, ref_embs).cpu().numpy()
        
        matched_cand = set()
        matched_ref = set()
        matches_count = 0
        
        flat_indices = []
        for i in range(len(cand_actions)):
            for j in range(len(ref_actions)):
                flat_indices.append((sim_matrix[i, j], i, j))
                
        flat_indices.sort(key=lambda x: x[0], reverse=True)
        
        for score, cand_idx, ref_idx in flat_indices:
            if score < threshold:
                break
            if cand_idx not in matched_cand and ref_idx not in matched_ref:
                matched_cand.add(cand_idx)
                matched_ref.add(ref_idx)
                matches_count += 1
                
        precision = matches_count / len(cand_triples)
        recall = matches_count / len(ref_triples)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "action_item_precision": precision,
            "action_item_recall": recall,
            "action_item_f1": f1
        }
        
    except Exception as e:
        logging.error(f"Action-item embedding matching failed: {e}")
        matches_count = 0
        matched_ref = set()
        for c in cand_actions:
            for idx, r in enumerate(ref_actions):
                if c.lower() == r.lower() and idx not in matched_ref:
                    matches_count += 1
                    matched_ref.add(idx)
                    break
        precision = matches_count / len(cand_triples)
        recall = matches_count / len(ref_triples)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "action_item_precision": precision,
            "action_item_recall": recall,
            "action_item_f1": f1
        }

# 6. Transcription Error Metrics (WER/CER)
def compute_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate (WER) between reference and hypothesis using jiwer."""
    if not reference.strip() and not hypothesis.strip():
        return 0.0
    if not reference.strip():
        return 1.0
    import jiwer
    try:
        return float(jiwer.wer(reference, hypothesis))
    except Exception as e:
        logging.warning(f"WER calculation failed: {e}")
        return 1.0

def compute_cer(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate (CER) between reference and hypothesis using jiwer."""
    if not reference.strip() and not hypothesis.strip():
        return 0.0
    if not reference.strip():
        return 1.0
    import jiwer
    try:
        return float(jiwer.cer(reference, hypothesis))
    except Exception as e:
        logging.warning(f"CER calculation failed: {e}")
        return 1.0

def compute_normalized_wer(reference: str, hypothesis: str) -> float:
    """Computes normalized Word Error Rate (lowercased, punctuation removed)."""
    if not reference.strip() and not hypothesis.strip():
        return 0.0
    if not reference.strip():
        return 1.0
    import jiwer
    try:
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords()
        ])
        return float(jiwer.wer(reference, hypothesis, reference_transform=transformation, hypothesis_transform=transformation))
    except Exception as e:
        logging.warning(f"Normalized WER calculation failed: {e}")
        return 1.0

