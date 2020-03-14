#[macro_use] extern crate approx; // For the macro relative_eq!

use hashbrown::HashMap;
use nalgebra;
use nalgebra::Vector;
use phf::phf_map;
use std::io::Read;

pub const NUM_DIMS:usize = 50;
pub type WordVec = [f32; NUM_DIMS];

pub struct SIFVectorizer {
	word_to_vec: phf::Map<&'static str, WordVec>,
	word_to_doc_frequency: phf::Map<&'static str, f32>,
	principal_component: WordVec
}

impl SIFVectorizer {
	pub fn new_from_compiled(
		word_to_vec: phf::Map<&'static str, WordVec>,
		word_to_doc_frequency: phf::Map<&'static str, f32>,
		principal_component: WordVec,
	) -> Self {
		SIFVectorizer {
			word_to_vec,
			word_to_doc_frequency,
			principal_component
		}
	}
	
	fn word_in_vocabulary(&self, word: &str) -> bool {
		self.word_to_vec.contains_key(word)
	}
	
	fn erase_nonascii(&self, word: &str) -> Option<String> {
		let trimmed_word:String = word
			.chars()
			.filter_map(|c|{
				if c.is_ascii_alphanumeric() {
					Some(c)
				} else {
					None
				}
			})
			.collect();
		
		if trimmed_word.len() > 0 {
			Some(trimmed_word)
		} else {
			None
		}
	}
	
	fn split_sentence(&self, sentence: &str) -> Vec<String> {
		sentence
			.to_ascii_lowercase()
			.split_whitespace()
			.filter_map(|word:&str| { self.erase_nonascii(word) })
			.collect()
	}
	
	fn vectorize_word(&self, word: &str) -> &WordVec {
		&self.word_to_vec.get(word).unwrap()
	}
	
	pub fn vectorize_sentence(&self, sent: &str) -> WordVec {
		let mut base_vector = [0f32; NUM_DIMS];
		let mut token_count = 0;
		for w in self.split_sentence(sent) {
			if !self.word_in_vocabulary(&w) {
				continue
			}
			
			let word_vec = self.vectorize_word(&w);
			for i in 0..NUM_DIMS {
				base_vector[i] += word_vec[i];
			}
			token_count += 1;
		}
		
		if token_count > 0 {
			for i in 0..NUM_DIMS {
				base_vector[i] /= token_count as f32;
			}
		}
		
		for i in 0..NUM_DIMS {
			base_vector[i] -= self.principal_component[i];
		}
		
		base_vector
	}
}
