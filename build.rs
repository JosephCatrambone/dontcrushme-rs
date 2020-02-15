use std::env;
use std::fs::File;
use std::io::{BufWriter, Write, prelude, BufReader, BufRead};
use std::path::Path;

fn main() {
	//let path = Path::new(&env::var("OUT_DIR").unwrap()).join("old_codegen.rs");
	let path = Path::new("codegen.rs");

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