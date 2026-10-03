//! A tiny local classifier: bag-of-words Naive Bayes. Deterministic, no network.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;

use super::classifier::{Classification, ClassifierError, RequestClassifier, RequestType};

/// Fixed label order, so results never depend on hash-map ordering.
const ALL: [RequestType; 7] = [
    RequestType::CodeGeneration,
    RequestType::CodeUnderstanding,
    RequestType::TechnicalDesign,
    RequestType::AnalyticalReasoning,
    RequestType::Writing,
    RequestType::FactualLookup,
    RequestType::General,
];

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

pub struct NaiveBayesClassifier {
    log_prior: Vec<f64>,
    word_counts: Vec<HashMap<String, f64>>,
    total_words: Vec<f64>,
    vocab: HashSet<String>,
}

impl NaiveBayesClassifier {
    /// Learn from (query, correct label) pairs.
    pub fn train(examples: &[(String, RequestType)]) -> Self {
        let n = examples.len() as f64;
        let mut word_counts: Vec<HashMap<String, f64>> = vec![HashMap::new(); ALL.len()];
        let mut total_words = vec![0.0; ALL.len()];
        let mut doc_counts = vec![0.0; ALL.len()];
        let mut vocab: HashSet<String> = HashSet::new();

        for (text, label) in examples {
            let i = ALL.iter().position(|l| l == label).unwrap_or(ALL.len() - 1);
            doc_counts[i] += 1.0;
            for w in tokenize(text) {
                *word_counts[i].entry(w.clone()).or_insert(0.0) += 1.0;
                total_words[i] += 1.0;
                vocab.insert(w);
            }
        }

        let log_prior = doc_counts
            .iter()
            .map(|c| ((c + 1.0) / (n + ALL.len() as f64)).ln())
            .collect();

        Self {
            log_prior,
            word_counts,
            total_words,
            vocab,
        }
    }

    fn predict(&self, text: &str) -> Classification {
        let words = tokenize(text);

        // Nothing here was seen in training: say "general" and admit we're unsure.
        if !words.iter().any(|w| self.vocab.contains(w)) {
            return Classification {
                request_type: RequestType::General,
                complexity: 3,
                confidence: 0.0,
            };
        }

        let v = self.vocab.len() as f64;
        let scores: Vec<f64> = (0..ALL.len())
            .map(|i| {
                let mut s = self.log_prior[i];
                for w in &words {
                    let c = self.word_counts[i].get(w).copied().unwrap_or(0.0);
                    s += ((c + 1.0) / (self.total_words[i] + v)).ln();
                }
                s
            })
            .collect();

        let mut best = 0;
        for i in 1..scores.len() {
            if scores[i] > scores[best] {
                best = i;
            }
        }

        // Turn scores into a 0..1 "how sure" number.
        let sum: f64 = scores.iter().map(|s| (s - scores[best]).exp()).sum();
        let confidence = (1.0 / sum) as f32;

        Classification {
            request_type: ALL[best],
            complexity: 3,
            confidence,
        }
    }
}

#[async_trait]
impl RequestClassifier for NaiveBayesClassifier {
    async fn classify(&self, query: &str) -> Result<Classification, ClassifierError> {
        Ok(self.predict(query))
    }
}

/// Uses the regex when it finds a clear match; otherwise asks Naive Bayes.
pub struct HybridClassifier {
    nb: NaiveBayesClassifier,
}

impl HybridClassifier {
    pub fn new(nb: NaiveBayesClassifier) -> Self {
        Self { nb }
    }
}

#[async_trait]
impl RequestClassifier for HybridClassifier {
    async fn classify(&self, query: &str) -> Result<Classification, ClassifierError> {
        let by_regex = super::classifier::classify_request_type(query);
        if by_regex != RequestType::General {
            return Ok(Classification {
                request_type: by_regex,
                complexity: 3,
                confidence: 0.5,
            });
        }
        self.nb.classify(query).await
    }
}