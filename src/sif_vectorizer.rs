use hashbrown::HashMap;
use nalgebra::{DMatrix, DVector, Matrix};
use nalgebra::linalg::SVD;
use phf::phf_map;
use std::io::Read;

pub const NUM_DIMS:usize = 50;
pub type WordVec = [f32; NUM_DIMS];

pub struct SIFVectorizer {
	word_to_vec: &'static phf::Map<&'static str, WordVec>,
	word_to_doc_frequency: &'static phf::Map<&'static str, f32>,
	principal_component: WordVec,
	smoothing_factor: f32,
}

impl SIFVectorizer {
	pub fn new_from_compiled(
		word_to_vec: &'static phf::Map<&'static str, WordVec>,
		word_to_doc_frequency: &'static phf::Map<&'static str, f32>,
		principal_component: WordVec,
		smoothing_factor: f32,
	) -> Self {
		SIFVectorizer {
			word_to_vec,
			word_to_doc_frequency,
			principal_component,
			smoothing_factor
		}
	}
	
	pub fn fine_tune(&mut self, documents: &Vec<&String>) {
		// Vectorize all documents.
		let mut sentences = Vec::<WordVec>::with_capacity(documents.len());
		for d in documents.iter() {
			let v = self.vectorize_sentence(d);
			sentences.push(v);
		}
		
		// Convert the vectors into a matrix.
		let mut docmat:DMatrix<f32> = DMatrix::<f32>::from_fn(documents.len(), NUM_DIMS, |r,c| {
			sentences[r][c]
		});
		
		// Extract principle components.
		let singular_values:DVector<f32> = docmat.singular_values();
		
		// Alternative method
		//let (_, _, v) = docmat.svd(true, true);
		//let largest_singular_value = v.nth(0).unwrap();
		// If we were doing PCA, we could do input * v.
		
		for i in 0..NUM_DIMS {
			self.principal_component[i] = *singular_values.row(0).get(i).unwrap(); // TODO: I hope this is row-major
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
			let word_freq:f32 = *self.word_to_doc_frequency.get(w.as_str()).unwrap_or(&1.0f32);
			let smoothing = self.smoothing_factor / (self.smoothing_factor * word_freq);
			for i in 0..NUM_DIMS {
				base_vector[i] += word_vec[i]*smoothing;
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
