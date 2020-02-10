#[macro_use]
extern crate approx; // For the macro relative_eq!
//extern crate phf;

use gdnative::*;
use hashbrown::{HashMap, HashSet};
use nalgebra;
use nalgebra::Vector;
//use phf::phf_map;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter};
use std::io::SeekFrom;
use std::iter::Zip;
use std::fs::File;
use typenum::U50;

// Utility methods
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

/// The WordVectorizer "class"
#[derive(NativeClass)]
#[inherit(Node)]
#[user_data(user_data::ArcData<WordVectorizer>)]
pub struct WordVectorizer {
	word_to_vector: HashMap<String, Vec<f32>>
}

// __One__ `impl` block can have the `#[methods]` attribute, which will generate
// code to automatically bind any exported methods to Godot.
#[methods]
impl WordVectorizer {
	/// The "constructor" of the class.
	fn _init(_owner: Node) -> Self {
		WordVectorizer {
			word_to_vector: HashMap::new()
		}
	}

	// In order to make a method known to Godot, the #[export] attribute has to be used.
	// In Godot script-classes do not actually inherit the parent class.
	// Instead they are"attached" to the parent object, called the "owner".
	// The owner is passed to every single exposed method.
	#[export]
	fn _ready(&self, _owner: Node) {
		// The `godot_print!` macro works like `println!` but prints to the Godot-editor
		// output tab as well.
		godot_print!("Loading vectors.");
	}

	#[export]
	fn similarity(&self, _owner:Node, s1:GodotString, s2:GodotString) -> Variant {
		let v1 = self.vectorize(&s1.to_string());
		let v2 = self.vectorize(&s2.to_string());
		return Variant::from_f64(cosine_similarity(&v1, &v2) as f64);
	}

	fn from_text_file(glove_vectors: &str) -> Self {
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

		WordVectorizer { word_to_vector }
	}

	fn word_in_vocabulary(&self, word: &str) -> bool {
		self.word_to_vector.contains_key(word)
	}

	fn vectorize(&self, word: &str) -> Vec<f32> {
		self.word_to_vector.get(word).unwrap().clone()
	}
}

// Function that registers all exposed classes to Godot
fn init(handle: gdnative::init::InitHandle) {
	handle.add_class::<WordVectorizer>();
}

// macros that create the entry-points of the dynamic library.
godot_gdnative_init!();
godot_nativescript_init!(init);
godot_gdnative_terminate!();


#[cfg(test)]
mod tests {
	use crate::{WordVectorizer, cosine_similarity};
	
	#[test]
	fn sanity_check_word_similarity() {
		let wv = WordVectorizer::from_text_file("glove.6B.50d.txt");
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
