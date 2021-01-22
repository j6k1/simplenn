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
			StartupError::InvalidConfiguration(ref s) => write!(f, "Configuration is invalid. ({})",s),
			StartupError::Fail(_) => write!(f, "Startup Failed."),
		}
	}
}
impl<E> error::Error for StartupError<E> where E: Error + fmt::Debug + 'static {
	fn description(&self) -> &str {
		match *self {
			StartupError::InvalidConfiguration(_) => "Configuration is invalid.",
			StartupError::Fail(_) => "Startup Failed.",
		}
	}

	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
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
	InvalidState(String),
	ParseFloatError(ParseFloatError)
}
impl fmt::Display for ConfigReadError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			ConfigReadError::IOError(_) => write!(f, "Error occurred in file I/O."),
			ConfigReadError::InvalidState(ref s) => write!(f, "Configuration is invalid. ({})",s),
			ConfigReadError::ParseFloatError(_) => write!(f, "An error occurred when converting a string to a double value."),
		}
	}
}
impl error::Error for ConfigReadError {
	fn description(&self) -> &str {
		match *self {
			ConfigReadError::IOError(_) => "Error occurred in file I/O.",
			ConfigReadError::InvalidState(_) => "Configuration is invalid.",
			ConfigReadError::ParseFloatError(_) => "An error occurred when converting a string to a double value."
		}
	}

	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match *self {
			ConfigReadError::IOError(ref e) => Some(e),
			ConfigReadError::InvalidState(_) => None,
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
	ReceiveError(String),
	UpdateError(String),
}
impl fmt::Display for InvalidStateError {
	 fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	 	match *self {
	 		InvalidStateError::InvalidInput(ref s) => write!(f, "The input value is invalid. ({})",s),
	 		InvalidStateError::GenerationError(ref s) => write!(f, "The snapshot is invalid. ({})",s),
			InvalidStateError::ReceiveError(ref s ) => write!(f, "Failed to receive the result from the thread. ({})",s),
			InvalidStateError::UpdateError(ref s) => write!(f, "update failed. ({})",s),
	 	}
	 }
}
impl error::Error for InvalidStateError {
	 fn description(&self) -> &str {
	 	match *self {
	 		InvalidStateError::InvalidInput(_) => "The input value is invalid.",
	 		InvalidStateError::GenerationError(_) => "The snapshot is invalid.",
			InvalidStateError::ReceiveError(_) => "Failed to receive the result from the thread.",
			InvalidStateError::UpdateError(_) => "update failed.",
	 	}
	 }

	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
	 	match *self {
	 		InvalidStateError::InvalidInput(_) => None,
	 		InvalidStateError::GenerationError(_) => None,
			InvalidStateError::ReceiveError(_) => None,
			InvalidStateError::UpdateError(_) => None,
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

	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
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

	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
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
