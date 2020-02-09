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
use std::iter::Zip;

trait WordVectorizer {
	fn word_in_vocabulary(&self, word: &str) -> bool;
	fn vectorize(&self, word: &str) -> Vec<f32>;
}

struct GloveInMem {
	word_to_vector: HashMap<String, Vec<f32>>
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
				let digits: Vec<f32> = tokens.map(|x| { x.parse::<f32>().unwrap() }).collect();
				word_to_vector.insert(word, digits);
				//nalgebra::VectorN::from_data(digits);
			}
		}
		
		GloveInMem { word_to_vector }
	}
}

impl WordVectorizer for GloveInMem {
	fn word_in_vocabulary(&self, word: &str) -> bool {
		self.word_to_vector.contains_key(word)
	}
	
	fn vectorize(&self, word: &str) -> Vec<f32> {
		self.word_to_vector.get(word).unwrap().clone()
	}
}

fn cosine_similarity(va: &Vec<f32>, vb: &Vec<f32>) -> f32 {
	let mut accumulator = 0.0f32;
	let mut magnitude_a = 0.0f32;
	let mut magnitude_b = 0.0f32;
	for i in 0..va.len() {
		let elem_a = va[i];
		let elem_b = vb[i];
		accumulator += elem_a*elem_b;
		magnitude_a += elem_a*elem_a;
		magnitude_b += elem_b*elem_b;
	}
	accumulator / (magnitude_a.sqrt() * magnitude_b.sqrt())
}


#[cfg(test)]
mod tests {
	use crate::{WordVectorizer, GloveInMem, cosine_similarity};
	
	#[test]
	fn sanity_check_word_similarity() {
		let wv = GloveInMem::new("glove.6B.50d.txt");
		let wv1 = wv.vectorize("cat");
		let wv2 = wv.vectorize("feline");
		let wv3 = wv.vectorize("eggplant");
		let cat_feline_sim = cosine_similarity(&wv1, &wv2);
		let cat_eggplant_sim = cosine_similarity(&wv1, &wv3);
		println!("Cat/Feline sim: {}", cat_feline_sim);
		println!("Cat/eggplant sim: {}", cat_eggplant_sim);
		assert!(cat_feline_sim > cat_eggplant_sim);
		//assert_eq!(cosine_similarity(wv1, wv2), 4);
	}

	#[test]
	fn serialize_wordvec_to_datfile() {
		// Iterate through the word to vec glove file and reduce the 400k @ 50d
	}
}
