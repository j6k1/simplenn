use std::io::BufReader;
use std::io::BufRead;
use std::io::BufWriter;
use std::path::Path;
use std::fs::File;
use std::fs::OpenOptions;
use InputReader;

use std::io;
use error::*;

pub struct TextFileInputReader {
	reader:Option<BufReader<File>>,
	line:Option<Vec<String>>,
	index:usize,
}
impl TextFileInputReader {
	pub fn new (file:String) -> Result<TextFileInputReader, StartupError> {
		Ok(match Path::new(&*file).exists() {
			true => {
				TextFileInputReader {
					reader:Some(BufReader::new(OpenOptions::new().read(true).create(false).open(file)?)),
					line:None,
					index:0usize,
				}
			},
			false => TextFileInputReader{
				reader:None,
				line:None,
				index:0usize,
			}
		})
	}

	fn read_line(&mut self) -> Result<String, StartupError> {
		match self.reader {
			Some(ref mut reader) => {
				let mut buf = String::new();
				reader.read_line(&mut buf)?;
				Ok(buf)
			},
			None => Err(StartupError::InavalidState(String::from(
													"The file does not exist yet."))),
		}
	}

	fn next_token(&mut self) -> Result<String, StartupError> {
		let t = match self.line {
			None => {
				let mut buf = self.read_line()?;

				while match &*buf {
					"" => true,
					s => match s.chars().nth(0) {
						Some('#') => true,
						_ => false,
					}
				} {
					buf = self.read_line()?.trim().to_string();
				}

				let line = buf.split(" ").map(|s| s.to_string()).collect::<Vec<String>>();
				let t = (&line[self.index]).clone();
				self.line = Some(line);
				self.index = 0;
				t
			},
			Some(ref line) => {
				(&line[self.index]).clone()
			}
		};

		self.index = self.index + 1;

		if match self.line {
			Some(ref line) if self.index >= line.len() => {
				true
			},
			Some(_) => false,
			None => false,
		} {
			self.line = None;
		}

		Ok(t)
	}

	fn next_double(&mut self) -> Result<f64, StartupError> {
		Ok(self.next_token()?.parse::<f64>()?)
	}
}
impl InputReader for TextFileInputReader {
	fn read_vec(&mut self, units:usize, w:usize) -> Result<Vec<Vec<f64>>, StartupError> {
		let mut v:Vec<Vec<f64>> = Vec::with_capacity(units);

		for _ in 0..units {
			let mut u = Vec::with_capacity(w);
			for _ in 0..w {
				u.push(self.next_double()?);
			}
			v.push(u);
		}
		Ok(v)
	}

	fn source_exists(&mut self) -> bool {
		match self.reader {
			Some(_) => true,
			None => false,
		}
	}
}
pub struct PersistenceWithTextFile {
	writer:BufWriter<File>,
}
impl PersistenceWithTextFile {
	pub fn new(file:String) -> Result<PersistenceWithTextFile,io::Error> {
		Ok(PersistenceWithTextFile {
			writer:BufWriter::new(OpenOptions::new().append(true).create(true).open(file)?),
		})
	}
}
