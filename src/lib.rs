#[macro_use]
extern crate approx; // For the macro relative_eq!

use gdnative::*;
use nalgebra;
use nalgebra::Vector;
use phf::phf_map;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter};
use std::io::SeekFrom;
use std::iter::{Iterator, Zip};
use std::fs::File;
use typenum::U50;

const NUM_DIMS:usize = 50;
type WordVec = [f32; NUM_DIMS];

// Utility methods
fn cosine_similarity(va: &WordVec, vb: &WordVec) -> f32 {
	let mut accumulator = 0.0f32;
	let mut magnitude_a = 0.0f32;
	let mut magnitude_b = 0.0f32;
	for i in 0..NUM_DIMS {
		let elem_a = va[i];
		let elem_b = vb[i];
		accumulator += elem_a*elem_b;
		magnitude_a += elem_a*elem_a;
		magnitude_b += elem_b*elem_b;
	}
	accumulator / (magnitude_a.sqrt() * magnitude_b.sqrt())
}

//static WORD_TO_VEC: phf::Map<&'static str, [f32; 50]> = phf_map! { "foo" => [ 0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 11f32, 12f32, 13f32, 14f32, 15f32, 16f32, 17f32, 18f32, 19f32, 20f32, 21f32, 22f32, 23f32, 24f32, 25f32, 26f32, 27f32, 28f32, 29f32, 30f32, 31f32, 32f32, 33f32, 34f32, 35f32, 36f32, 37f32, 38f32, 39f32, 40f32, 41f32, 42f32, 43f32, 44f32, 45f32, 46f32, 47f32, 48f32, 49f32] };
//include!(concat!(env!("OUT_DIR"), "/old_codegen.rs"))
include!{"../codegen.rs"}

/// The WordVectorizer "class"
#[derive(NativeClass)]
#[inherit(Node)]
#[user_data(user_data::ArcData<WordVectorizer>)]
pub struct WordVectorizer;

// __One__ `impl` block can have the `#[methods]` attribute, which will generate
// code to automatically bind any exported methods to Godot.
#[methods]
impl WordVectorizer {
	/// The "constructor" of the class.
	fn _init(_owner: Node) -> Self {
		WordVectorizer {
		}
	}

	fn new() -> Self {
		WordVectorizer {}
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
		let v1 = self.vectorize_sentence(&s1.to_string());
		let v2 = self.vectorize_sentence(&s2.to_string());
		return Variant::from_f64(cosine_similarity(&v1, &v2) as f64);
	}

	fn word_in_vocabulary(&self, word: &str) -> bool {
		WORD_TO_VEC.contains_key(word)
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
		&WORD_TO_VEC.get(word).unwrap()
	}

	fn vectorize_sentence(&self, sent: &str) -> WordVec {
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
				base_vector[i] /= token_count as f32
			}
		}

		base_vector
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
	use crate::{WordVectorizer, WordVec, cosine_similarity};
	
	#[test]
	fn sanity_check_word_similarity() {
		let wv = WordVectorizer::new();
		let wv1 = wv.vectorize_word("cat");
		let wv2 = wv.vectorize_word("feline");
		let wv3 = wv.vectorize_word("eggplant");
		let cat_feline_sim = cosine_similarity(&wv1, &wv2);
		let cat_eggplant_sim = cosine_similarity(&wv1, &wv3);
		println!("Cat/Feline sim: {}", cat_feline_sim);
		println!("Cat/eggplant sim: {}", cat_eggplant_sim);
		assert!(cat_feline_sim > cat_eggplant_sim);
		//assert_eq!(cosine_similarity(wv1, wv2), 4);
	}

	#[test]
	fn compare_intracluster_distance_vs_extracluster_distance() {
		let wv = WordVectorizer::new();
		let mut cluster_a = Vec::<WordVec>::new();
		let mut cluster_b = Vec::<WordVec>::new();

		cluster_a.push(wv.vectorize_sentence("Would it be okay if I saved everyone some time and just went mad now?"));
		cluster_a.push(wv.vectorize_sentence("Can I play with madness?"));
		cluster_a.push(wv.vectorize_sentence("The surest sign that you're crazy is thinking you're the only one that's sane."));
	}
}
