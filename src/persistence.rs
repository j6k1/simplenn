use std::io::BufReader;
use std::io::BufRead;
use std::io::Read;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::fs::File;
use std::fs::OpenOptions;
use std::f64;
use InputReader;

use std::fmt;
use std::io;
use error::*;
use std::error::Error;

use Persistence;

pub struct TextFileInputReader {
	reader:Option<BufReader<File>>,
	line:Option<Vec<String>>,
	index:usize,
}
impl TextFileInputReader {
	pub fn new (file:&str) -> Result<TextFileInputReader, ConfigReadError>
		where ConfigReadError: Error + fmt::Debug, StartupError<ConfigReadError>: From<ConfigReadError> {
		Ok(match Path::new(file).exists() {
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

	fn read_line(&mut self) -> Result<String, ConfigReadError> {
		match self.reader {
			Some(ref mut reader) => {
				let mut buf = String::new();
				let n = reader.read_line(&mut buf)?;

				buf = buf.trim().to_string();

				if n == 0 {
					Err(ConfigReadError::InvalidState(String::from(
						"End of input has been reached.")))
				} else {
					Ok(buf)
				}
			},
			None => Err(ConfigReadError::InvalidState(String::from(
													"The file does not exist yet."))),
		}
	}

	fn next_token(&mut self) -> Result<String, ConfigReadError> {
		let t = match self.line {
			None => {
				self.index = 0;
				let mut buf = self.read_line()?;

				while match &*buf {
					"" => true,
					s => match s.chars().nth(0) {
						Some('#') => true,
						_ => false,
					}
				} {
					buf = self.read_line()?;
				}

				let line = buf.split(" ").map(|s| s.to_string()).collect::<Vec<String>>();
				let t = (&line[self.index]).clone();
				self.line = Some(line);
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
			Some(_) => {
				false
			}
			None => false,
		} {
			self.line = None;
		}

		Ok(t)
	}

	fn next_double(&mut self) -> Result<f64, ConfigReadError> {
		Ok(self.next_token()?.parse::<f64>()?)
	}
}
impl InputReader<ConfigReadError> for TextFileInputReader where ConfigReadError: Error + fmt::Debug {
	fn read_vec(&mut self, units:usize, w:usize) -> Result<Vec<Vec<f64>>, ConfigReadError> {
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

	fn verify_eof(&mut self) -> Result<(),ConfigReadError>
		where ConfigReadError: Error + fmt::Debug, StartupError<ConfigReadError>: From<ConfigReadError> {
		match self.reader {
			Some(ref mut reader) => {
				let mut buf = String::new();

				loop {
					let n = reader.read_line(&mut buf)?;

					if n == 0 {
						return Ok(());
					}

					buf = buf.trim().to_string();

					if !buf.is_empty() {
						return Err(ConfigReadError::InvalidState(
									String::from("Data loaded , but the input has not reached the end.")));
					} else {
						buf.clear();
					}
				}
			},
			None => Err(ConfigReadError::InvalidState(String::from(
													"The file does not exist yet."))),
		}
	}
}
pub struct PersistenceWithTextFile {
	writer:BufWriter<File>,
}
impl PersistenceWithTextFile
	where PersistenceWriteError: Error + fmt::Debug,
			PersistenceError<PersistenceWriteError>: From<PersistenceWriteError> {
	pub fn new(file:&str) -> Result<PersistenceWithTextFile,io::Error> {
		Ok(PersistenceWithTextFile {
			writer:BufWriter::new(OpenOptions::new().write(true).create(true).open(file)?),
		})
	}
}
impl Persistence<PersistenceWriteError> for PersistenceWithTextFile where PersistenceWriteError: Error + fmt::Debug {
	fn save(&mut self,layers:&Vec<Vec<Vec<f64>>>) -> Result<(),PersistenceWriteError> {
		self.writer.write(b"#Rust simplenn config start.\n")?;
		let mut i = 0;
		for units in layers {
			self.writer.write(format!("#layer: {}\n", i).as_bytes())?;

			for unit in units {
				self.writer.write(format!("{}\n", unit.iter()
													.map(|w| w.to_string())
													.collect::<Vec<String>>()
													.join(" ")).as_bytes())?;
			}
			i = i + 1;
		}

		Ok(())
	}
}
pub struct BinFileInputReader {
	reader:Option<BufReader<File>>,
}
impl BinFileInputReader {
	pub fn new (file:&str) -> Result<BinFileInputReader, ConfigReadError>
		where ConfigReadError: Error + fmt::Debug, StartupError<ConfigReadError>: From<ConfigReadError> {
		Ok(match Path::new(file).exists() {
			true => {
				BinFileInputReader {
					reader:Some(BufReader::new(OpenOptions::new().read(true).create(false).open(file)?)),
				}
			},
			false => BinFileInputReader{
				reader:None,
			}
		})
	}

	fn next_double(&mut self) -> Result<f64, ConfigReadError> {
		let mut buf = [0; 8];

		return match self.reader {
			Some(ref mut reader) => {
				reader.read_exact(&mut buf)?;

				Ok(f64::from_bits(
						(buf[0] as u64) << 56 |
						(buf[1] as u64) << 48 |
						(buf[2] as u64) << 40 |
						(buf[3] as u64) << 32 |
						(buf[4] as u64) << 24 |
						(buf[5] as u64) << 16 |
						(buf[6] as u64) << 8  |
						 buf[7] as u64))
			},
			None => Err(ConfigReadError::InvalidState(String::from(
													"The file does not exist yet."))),
		}
	}
}
impl InputReader<ConfigReadError> for BinFileInputReader where ConfigReadError: Error + fmt::Debug {
	fn read_vec(&mut self, units:usize, w:usize) -> Result<Vec<Vec<f64>>, ConfigReadError> {
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

	fn verify_eof(&mut self) -> Result<(),ConfigReadError>
		where ConfigReadError: Error + fmt::Debug, StartupError<ConfigReadError>: From<ConfigReadError> {
		match self.reader {
			Some(ref mut reader) => {
				let mut buf:[u8; 1] = [0];

				let n = reader.read(&mut buf)?;

				if n == 0 {
					Ok(())
				} else {
					Err(ConfigReadError::InvalidState(String::from("Data loaded , but the input has not reached the end.")))
				}
			},
			None => Err(ConfigReadError::InvalidState(String::from(
													"The file does not exist yet."))),
		}
	}
}
pub struct PersistenceWithBinFile {
	writer:BufWriter<File>,
}
impl PersistenceWithBinFile
	where PersistenceWriteError: Error + fmt::Debug,
			PersistenceError<PersistenceWriteError>: From<PersistenceWriteError> {
	pub fn new(file:&str) -> Result<PersistenceWithBinFile,io::Error> {
		Ok(PersistenceWithBinFile {
			writer:BufWriter::new(OpenOptions::new().write(true).create(true).open(file)?),
		})
	}
}
impl Persistence<PersistenceWriteError> for PersistenceWithBinFile where PersistenceWriteError: Error + fmt::Debug {
	fn save(&mut self,layers:&Vec<Vec<Vec<f64>>>) -> Result<(),PersistenceWriteError> {
		for units in layers {
			for unit in units {
				for w in unit {
					let mut buf = [0; 8];
					let bits = w.to_bits();

					buf[0] = (bits >> 56 & 0xff) as u8;
					buf[1] = (bits >> 48 & 0xff) as u8;
					buf[2] = (bits >> 40 & 0xff) as u8;
					buf[3] = (bits >> 32 & 0xff) as u8;
					buf[4] = (bits >> 24 & 0xff) as u8;
					buf[5] = (bits >> 16 & 0xff) as u8;
					buf[6] = (bits >> 8 & 0xff) as u8;
					buf[7] = (bits & 0xff) as u8;

					self.writer.write(&buf)?;
				}
			}
		}

		Ok(())
	}
}


