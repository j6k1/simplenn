use std::io;
use std::fmt;
use std::error;
use std::error::Error;
use std::num::ParseFloatError;

#[derive(Debug)]
pub enum StartupError<E> {
	InvalidConfiguration(String),
	Fail(E),
}
impl<E> fmt::Display for StartupError<E> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			StartupError::InvalidConfiguration(ref s) => write!(f, "{}",s),
			StartupError::Fail(_) => write!(f, "Startup Failed."),
		}
	}
}
impl<E> error::Error for StartupError<E> where E: Error + fmt::Debug {
	fn description(&self) -> &str {
		match *self {
			StartupError::InvalidConfiguration(_) => "Configuration is invalid.",
			StartupError::Fail(_) => "Startup Failed.",
		}
	}

	fn cause(&self) -> Option<&error::Error> {
		match *self {
			StartupError::InvalidConfiguration(_) => None,
			StartupError::Fail(ref e) => Some(e),
		}
	}
}
impl<E> From<E> for StartupError<E> where E: Error + fmt::Debug {
	fn from(err: E) -> StartupError<E> {
		StartupError::Fail(err)
	}
}
#[derive(Debug)]
pub enum ConfigReadError {
	IOError(io::Error),
	InavalidState(String),
	ParseFloatError(ParseFloatError)
}
impl fmt::Display for ConfigReadError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			ConfigReadError::IOError(_) => write!(f, "Error occurred in file I/O."),
			ConfigReadError::InavalidState(ref s) => write!(f, "{}",s),
			ConfigReadError::ParseFloatError(_) => write!(f, "An error occurred when converting a string to a double value."),
		}
	}
}
impl error::Error for ConfigReadError {
	fn description(&self) -> &str {
		match *self {
			ConfigReadError::IOError(_) => "Error occurred in file I/O.",
			ConfigReadError::InavalidState(_) => "Configuration is invalid.",
			ConfigReadError::ParseFloatError(_) => "An error occurred when converting a string to a double value."
		}
	}

	fn cause(&self) -> Option<&error::Error> {
		match *self {
			ConfigReadError::IOError(ref e) => Some(e),
			ConfigReadError::InavalidState(_) => None,
			ConfigReadError::ParseFloatError(ref e) => Some(e),
		}
	}
}
impl From<io::Error> for ConfigReadError {
	fn from(err: io::Error) -> ConfigReadError {
		ConfigReadError::IOError(err)
	}
}
impl From<ParseFloatError> for ConfigReadError {
	fn from(err: ParseFloatError) -> ConfigReadError {
		ConfigReadError::ParseFloatError(err)
	}
}
#[derive(Debug)]
pub enum InvalidStateError {
	InvalidInput(String),
	GenerationError(String),
}
impl fmt::Display for InvalidStateError {
	 fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	 	match *self {
	 		InvalidStateError::InvalidInput(ref s) => write!(f, "{}",s),
	 		InvalidStateError::GenerationError(ref s) => write!(f, "{}",s),
	 	}
	 }
}
impl error::Error for InvalidStateError {
	 fn description(&self) -> &str {
	 	match *self {
	 		InvalidStateError::InvalidInput(_) => "The input value is invalid.",
	 		InvalidStateError::GenerationError(_) => "The snapshot is invalid."
	 	}
	 }

	fn cause(&self) -> Option<&error::Error> {
	 	match *self {
	 		InvalidStateError::InvalidInput(_) => None,
	 		InvalidStateError::GenerationError(_) => None,
	 	}
	 }
}
#[derive(Debug)]
pub enum PersistenceError<E> {
	Fail(E),
}
impl<E> fmt::Display for PersistenceError<E> where E: Error + fmt::Debug {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			PersistenceError::Fail(_) => write!(f, "Persistence failed."),
		}
	}
}
impl<E> error::Error for PersistenceError<E> where E: Error + fmt::Debug {
	fn description(&self) -> &str {
		match *self {
			PersistenceError::Fail(_) => "Persistence failed.",
		}
	}

	fn cause(&self) -> Option<&error::Error> {
	 	match *self {
	 		PersistenceError::Fail(_) => None,
	 	}
	 }
}
impl<E> From<E> for PersistenceError<E> where E: Error + fmt::Debug {
	fn from(err: E) -> PersistenceError<E> {
		PersistenceError::Fail(err)
	}
}
#[derive(Debug)]
pub enum PersistenceWriteError {
	IOError(io::Error),
}
impl fmt::Display for PersistenceWriteError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			PersistenceWriteError::IOError(_) => write!(f,"Error occurred in file I/O."),
		}
	}
}
impl error::Error for PersistenceWriteError {
	fn description(&self) -> &str {
		match *self {
			PersistenceWriteError::IOError(_) => "Error occurred in file I/O.",
		}
	}

	fn cause(&self) -> Option<&error::Error> {
	 	match *self {
	 		PersistenceWriteError::IOError(ref e) => Some(e),
	 	}
	 }
}
impl From<io::Error> for PersistenceWriteError {
	fn from(err: io::Error) -> PersistenceWriteError {
		PersistenceWriteError::IOError(err)
	}
}
