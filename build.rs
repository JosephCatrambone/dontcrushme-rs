use std::env;
use std::fs::File;
use std::io::{BufWriter, Write, prelude, BufReader, BufRead};
use std::path::Path;
use std::collections::{HashMap, HashSet};

fn make_word_to_vec() {
	//let path = Path::new(&env::var("OUT_DIR").unwrap()).join("old_codegen.rs");
	let path = Path::new("word_to_vec.rs");
	
	if path.exists() {
		return; // No work to be done!
	}
	
	let mut file = BufWriter::new(File::create(&path).unwrap());
	
	let fin = File::open("glove.6B.50d.txt").expect(format!("Can't load vectors to build hashmap.").as_str());
	let buffered_fin = BufReader::new(fin);
	//let mut word_to_vector = HashMap::with_capacity(400_000);
	
	write!(&mut file, "static WORD_TO_VEC: phf::Map<&'static str, [f32; 50]> = phf_map! {{").unwrap();
	for line_result in buffered_fin.lines() {
		if let Ok(line) = line_result {
			let tokens = line.split_whitespace().collect::<Vec<&str>>();
			let word = tokens[0];
			if !word.chars().all(|c| { c.is_ascii_alphanumeric() }) {
				continue // Skip cases where not everything is ascii.
				// This helps us avoid funky cases where we try and tokenize unicode madness and lets us get away with much simpler tokenization.
			}
			let digits = &tokens[1..];
			write!(&mut file, "\"{}\" => [", word).unwrap();
			write!(&mut file, "{}", digits.join("f32,")).unwrap();
			write!(&mut file, "f32],\n").unwrap();
		}
	}
	write!(&mut file, "}};").unwrap();
	file.flush().unwrap();
}

fn make_word_to_freq() {
	let path = Path::new("word_to_freq.rs");
	
	if path.exists() {
		return; // No work to be done!
	}
	
	let mut file = BufWriter::new(File::create(&path).unwrap());
	let mut word_count = HashMap::<String, u64>::new();
	let mut word_document_count = HashMap::<String, u64>::new(); // How many unique documents have this word?
	let mut total_docs = 0u32;
	
	// FIN will contain one 'document' per line.
	let fin = File::open("corpus.txt").expect(format!("Can't load corpus to build hashmap.").as_str());
	let buffered_fin = BufReader::new(fin);
	
	for line_result in buffered_fin.lines() {
		let mut words_seen = HashSet::<String>::new();
		if let Ok(line) = line_result {
			total_docs += 1;
			// Unlike above where we split on whitespace, here we replace non-ascii with spaces before split.
			let tokens = line.chars().map(|c| { if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { ' ' } }).collect::<String>().split_whitespace().map(|s| String::from(s)).collect::<Vec<String>>();
			for word in &tokens {
				word_count.insert(word.to_string(), word_count.get(word).unwrap_or(&0u64) + 1);
				if !words_seen.contains(word) {
					word_document_count.insert(word.clone(), word_document_count.get(word).unwrap_or(&0u64) + 1);
					words_seen.insert(word.clone());
				}
			}
		}
	}
	
	write!(&mut file, "static WORD_TO_FREQ: phf::Map<&'static str, f32> = phf_map! {{").unwrap();
	for (word, doc_count) in word_document_count.iter() {
		if *doc_count > 1 {
			write!(&mut file, "\"{}\" => {}f32,\n", word, *doc_count as f32 / total_docs as f32).unwrap();
		}
	}
	write!(&mut file, "}};").unwrap();
	file.flush().unwrap();
}

fn main() {
	make_word_to_vec();
	make_word_to_freq();
}