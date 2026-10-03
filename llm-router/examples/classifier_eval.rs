use std::{env, fs, io::Write, time::Instant};

use nasiko_llm_router::routing::classifier::{RegexClassifier, RequestClassifier, RequestType};
// use nasiko_llm_router::routing::naive_bayes::NaiveBayesClassifier;
use nasiko_llm_router::routing::naive_bayes::{HybridClassifier, NaiveBayesClassifier};
use serde_json::{json, Value};

fn load_examples(path: &str) -> Vec<Value> {
    let text = fs::read_to_string(path).expect("cannot read dataset");
    let data: Value = serde_json::from_str(&text).expect("dataset is not valid JSON");
    data["examples"]
        .as_array()
        .expect("missing 'examples' array")
        .clone()
}

/// Pick the classifier from CLASSIFIER_BACKEND (default: regex).
fn build_classifier() -> Box<dyn RequestClassifier> {
    let backend = env::var("CLASSIFIER_BACKEND").unwrap_or_else(|_| "regex".to_string());
    match backend.as_str() {
        "naive_bayes" | "hybrid"  => {
            let train_path =
                env::var("CLASSIFIER_TRAIN").expect("set CLASSIFIER_TRAIN to the training file");
            let pairs: Vec<(String, RequestType)> = load_examples(&train_path)
                .iter()
                .filter_map(|ex| {
                    let query = ex["query"].as_str()?;
                    let label = RequestType::from_wire(ex["request_type"].as_str()?)?;
                    Some((query.to_string(), label))
                })
                .collect();
            // Box::new(NaiveBayesClassifier::train(&pairs))
            let nb = NaiveBayesClassifier::train(&pairs);
            if backend == "hybrid" {
                Box::new(HybridClassifier::new(nb))
            } else {
                Box::new(nb)
            }            
        }
        _ => Box::new(RegexClassifier),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let eval_path = env::var("EVAL_SET").expect("set EVAL_SET to the dataset path");
    let out_path = env::var("OUT").expect("set OUT to the output path");

    let examples = load_examples(&eval_path);
    let classifier = build_classifier();
    let mut out = fs::File::create(&out_path).expect("cannot create OUT");
    let mut correct = 0;

    for ex in &examples {
        let id = ex["id"].as_str().unwrap_or("");
        let query = ex["query"].as_str().unwrap_or("");

        let start = Instant::now();
        let result = classifier.classify(query).await.expect("classifier failed");
        let latency_us = start.elapsed().as_micros() as u64;

        let predicted = result.request_type.as_str();
        if ex["request_type"].as_str() == Some(predicted) {
            correct += 1;
        } else {
            eprintln!(
                "WRONG {id}: expected {} got {predicted} | {query}",
                ex["request_type"].as_str().unwrap_or("?")
            );
        }

        let line = json!({
            "id": id,
            "request_type": predicted,
            "complexity": result.complexity,
            "confidence": f64::from(result.confidence),
            "latency_us": latency_us,
        });
        writeln!(out, "{line}").expect("cannot write OUT");
    }

    eprintln!("request_type accuracy: {}/{}", correct, examples.len());
}