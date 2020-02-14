use std::env;
use std::fs::File;
use std::io::{BufWriter, Write, prelude, BufReader, BufRead};
use std::path::Path;

fn main() {
	//let path = Path::new(&env::var("OUT_DIR").unwrap()).join("codegen.rs");
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
			let digits = &tokens[1..];
			write!(&mut file, "\"{}\" => [", word).unwrap();
			write!(&mut file, "{}", digits.join(",")).unwrap();
			write!(&mut file, "],\n").unwrap();
			//let digits: Vec<f32> = tokens.map(|x| { x.parse::<f32>().unwrap() }).collect();

			//word_to_vector.insert(word, digits);
			//nalgebra::VectorN::from_data(digits);
			//phf.entry(word, digits.as_slice());
		}
	}
	write!(&mut file, "}};").unwrap();
	file.flush().unwrap();
}