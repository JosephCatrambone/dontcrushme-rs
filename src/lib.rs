#[macro_use]
extern crate approx; // For the macro relative_eq!

use hashbrown::{HashMap, HashSet};
use std::io::prelude::*;
use std::io::BufReader;
use std::io::SeekFrom;
use std::fs::File;
use nalgebra;
use typenum::U50;
use nalgebra::Vector;
use std::intrinsics::sqrtf64;

trait WordVectorizer {
	fn word_in_vocabulary(self, word: &str) -> bool;
	fn vectorize(self, word: &str) -> Vec::<f32>;
}

struct GloveInMem {
	word_to_vector: HashMap<String, Vec::<f32>>
}

impl GloveInMem {
	fn new(glove_vectors: &str) -> Self {
		let mut fin = File::open(glove_vectors).expect(format!("Can't load vectors from '{}'", glove_vectors).as_str());
		let mut buffered_fin = BufReader::new(fin);
		let mut word_to_vector = HashMap::with_capacity(400_000);
		
		for line_result in buffered_fin.lines() {
			if let Ok(line) = line_result {
				let mut tokens = line.split_whitespace();
				let word = tokens.next().unwrap().to_string();
				let digits: Vec::<f32> = tokens.map(|x| { x.parse::<f32>().unwrap() }).collect();
				word_to_vector.insert(word, digits);
				//nalgebra::VectorN::from_data(digits);
			}
		}
		
		GloveInMem { word_to_vector }
	}
}

impl WordVectorizer for GloveInMem {
	fn word_in_vocabulary(self, word: &str) -> bool {
		self.word_to_vector.contains_key(word)
	}
	
	fn vectorize(self, word: &str) -> Vec<f32> {
		self.word_to_vector.get(word).unwrap().clone()
	}
}

fn cosine_similarity<T : IntoIterator>(va: T, vb: T) -> f32 {
	let mut accumulator = 0.0f32;
	let mut magnitude_a = 0.0f32;
	let mut magnitude_b = 0.0f32;
	for (elem_a, elem_b) in zip(va, vb) {
		accumulator += elem_a*elem_b;
		magnitude_a += elem_a*elem_a;
		magnitude_b += elem_b*elem_b;
	}
	accumulator / (magnitude_a.sqrt() * magnitude_b.sqrt())
}


#[cfg(test)]
mod tests {
	use crate::{WordVectorizer, GloveInMem};
	
	#[test]
	fn it_works() {
		let wv = GloveInMem::new("glove.6B.50d.txt");
		assert_eq!(2 + 2, 4);
	}
}
