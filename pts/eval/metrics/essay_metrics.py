
# import necessary libraries
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer once
_tokenizer = AutoTokenizer.from_pretrained("KevSun/IELTS_essay_scoring")
_model = AutoModelForSequenceClassification.from_pretrained("KevSun/IELTS_essay_scoring")
_model.eval()

_item_names = ["Task Achievement", "Coherence and Cohesion", "Vocabulary", "Grammar", "Overall"]

def _get_scores(essay: str):
	encoded_input = _tokenizer(essay, return_tensors="pt", truncation=True)
	with torch.no_grad():
		outputs = _model(**encoded_input)
	predictions = outputs.logits.squeeze()
	predicted_scores = predictions.numpy()
	normalized_scores = (predicted_scores / predicted_scores.max()) * 9  # Scale to 9
	rounded_scores = np.round(normalized_scores * 2) / 2
	return dict(zip(_item_names, rounded_scores))

def evaluate_task_achievement(essay: str) -> float:
	"""Evaluate Task Achievement metric for the given essay."""
	return _get_scores(essay)["Task Achievement"]

def evaluate_coherence_and_cohesion(essay: str) -> float:
	"""Evaluate Coherence and Cohesion metric for the given essay."""
	return _get_scores(essay)["Coherence and Cohesion"]

def evaluate_vocabulary(essay: str) -> float:
	"""Evaluate Vocabulary metric for the given essay."""
	return _get_scores(essay)["Vocabulary"]

def evaluate_grammar(essay: str) -> float:
	"""Evaluate Grammar metric for the given essay."""
	return _get_scores(essay)["Grammar"]

def evaluate_overall(essay: str) -> float:
	"""Evaluate Overall metric for the given essay."""
	return _get_scores(essay)["Overall"]



#more metrics 


def evaluate_length(essay: str) -> int:
	"""Return the word count of the essay."""
	return len(essay.split())

def evaluate_unique_words(essay: str) -> int:
	"""Return the number of unique words in the essay."""
	words = essay.lower().split()
	return len(set(words))

def evaluate_average_sentence_length(essay: str) -> float:
	"""Return the average sentence length in words."""
	import re
	sentences = re.split(r'[.!?]+', essay)
	sentences = [s.strip() for s in sentences if s.strip()]
	if not sentences:
		return 0.0
	word_counts = [len(s.split()) for s in sentences]
	return sum(word_counts) / len(sentences)

def evaluate_perplexity(essay: str, model_name: str = "gpt2") -> float:
	"""Return the perplexity of the essay using a language model (default: gpt2)."""
	lm_tokenizer = AutoTokenizer.from_pretrained(model_name)
	lm_model = AutoModelForCausalLM.from_pretrained(model_name)
	input_ids = lm_tokenizer(essay, return_tensors="pt").input_ids
	with torch.no_grad():
		outputs = lm_model(input_ids, labels=input_ids)
		loss = outputs.loss
	return float(torch.exp(loss).item())




import re
from collections import Counter
from typing import Iterable, List, Tuple

# ---------- Helpers ----------
_word_re = re.compile(r"[A-Za-z0-9']+")
_sent_split_re = re.compile(r"[.!?]+(?:\s+|$)")

def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer (alnum + apostrophes)."""
    return [w.lower() for w in _word_re.findall(text)]

def _split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _sent_split_re.split(text)]
    return [s for s in sents if s]

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ---------- Metrics ----------
def distinct_n(essay :str, n: int = 3) -> float:
    """
    Distinct-n (percentage): unique n-grams / total n-grams across all texts * 100.
    Matches D-3 when n=3. Returns 0.0 if no n-grams.
    """
    all_ngrams = []
    for t in essay:
        all_ngrams.extend(_ngrams(_tokenize(t), n))
    total = len(all_ngrams)
    return 0.0 if total == 0 else (len(set(all_ngrams)) / total) * 100.0

def repetition_4(essay :str) -> float:
    """
    Repetition-4 (percentage): for each sentence, mark 1 if ANY 4-gram repeats
    (i.e., appears ≥ 2 times) within that sentence; average across sentences * 100.
    Returns 0.0 if there are no sentences.
    """
    sentences = []
    for t in essay:
        sentences.extend(_split_sentences(t))
    if not sentences:
        return 0.0

    flagged = 0
    for s in sentences:
        grams = _ngrams(_tokenize(s), 4)
        if grams:
            counts = Counter(grams)
            if any(c >= 2 for c in counts.values()):
                flagged += 1
        # Sentences with <4 tokens simply contribute 0
    return (flagged / len(sentences)) * 100.0

def lexical_repetition_lr_n(essay :str, n: int = 2) -> float:
    """
    Lexical Repetition LR-n (percentage): proportion of UNIQUE 4-gram types
    whose corpus frequency is ≥ n; i.e., |{g : freq(g) ≥ n}| / |{g}| * 100.
    Returns 0.0 if there are no 4-gram types.
    """
    all_4grams = []
    for t in essay:
        all_4grams.extend(_ngrams(_tokenize(t), 4))
    if not all_4grams:
        return 0.0

    counts = Counter(all_4grams)
    num_types = len(counts)
    num_repeated = sum(1 for c in counts.values() if c >= n)
    return (num_repeated / num_types) * 100.0

# ---------- Convenience wrappers ----------
def D3(essay :str) -> float:
    return distinct_n(essay, n=3)

def R4(essay :str) -> float:
    return repetition_4(essay)

def LR_n(essay :str, n: int = 2) -> float:
    return lexical_repetition_lr_n(essay, n=n)
