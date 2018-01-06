use std::io;
use std::fmt;
use std::error;
use std::num::ParseFloatError;

#[derive(Debug)]
pub enum StartupError {
	InvalidConfiguration(String),
	InavalidState(String),
	ParseFloatError(ParseFloatError),
	IOError(io::Error),
	Fail(String),
}
impl fmt::Display for StartupError {
	 fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	 	match *self {
	 		StartupError::InvalidConfiguration(ref s) => write!(f, "{}",s),
	 		StartupError::InavalidState(ref s) => write!(f, "{}",s),
	 		StartupError::ParseFloatError(_) => write!(f, "An error occurred when converting a string to a double value."),
	 		StartupError::IOError(_) => write!(f,"Error occurred in file I/O."),
	 		StartupError::Fail(_) => write!(f, "User error."),
	 	}
	 }
}
impl error::Error for StartupError {
	 fn description(&self) -> &str {
	 	match *self {
	 		StartupError::InvalidConfiguration(_) => "Configuration is invalid.",
	 		StartupError::InavalidState(_) => "This operation is not allowed in the current state.",
	 		StartupError::ParseFloatError(_) => "An error occurred when converting a string to a double value.",
	 		StartupError::IOError(_) => "Error occurred in file I/O.",
	 		StartupError::Fail(_) => "User error.",
	 	}
	 }

	fn cause(&self) -> Option<&error::Error> {
	 	match *self {
	 		StartupError::InvalidConfiguration(_) => None,
	 		StartupError::InavalidState(_) => None,
	 		StartupError::ParseFloatError(ref e) => Some(e),
	 		StartupError::IOError(ref e) => Some(e),
	 		StartupError::Fail(_) => None,
	 	}
	 }
}
impl<'a> From<io::Error> for StartupError {
	fn from(err: io::Error) -> StartupError {
		StartupError::IOError(err)
	}
}
impl<'a> From<ParseFloatError> for StartupError {
	fn from(err: ParseFloatError) -> StartupError {
		StartupError::ParseFloatError(err)
	}
}
#[derive(Debug)]
pub enum InvalidStateError {
	InvalidInput(String),
}
impl fmt::Display for InvalidStateError {
	 fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	 	match *self {
	 		InvalidStateError::InvalidInput(ref s) => write!(f, "{}",s),
	 	}
	 }
}
impl error::Error for InvalidStateError {
	 fn description(&self) -> &str {
	 	match *self {
	 		InvalidStateError::InvalidInput(_) => "The input value is invalid.",
	 	}
	 }

	fn cause(&self) -> Option<&error::Error> {
	 	match *self {
	 		InvalidStateError::InvalidInput(_) => None,
	 	}
	 }
}
#[derive(Debug)]
pub enum PersistenceError {
	IOError(io::Error),
	Fail(String),
}
impl fmt::Display for PersistenceError {
	 fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	 	match *self {
	 		PersistenceError::IOError(_) => write!(f,"Error occurred in file I/O."),
	 		PersistenceError::Fail(_) => write!(f, "User error."),
	 	}
	 }
}
impl error::Error for PersistenceError {
	 fn description(&self) -> &str {
	 	match *self {
	 		PersistenceError::IOError(_) => "Error occurred in file I/O.",
	 		PersistenceError::Fail(_) => "User error.",
	 	}
	 }

	fn cause(&self) -> Option<&error::Error> {
	 	match *self {
	 		PersistenceError::IOError(ref e) => Some(e),
	 		PersistenceError::Fail(_) => None,
	 	}
	 }
}
impl From<io::Error> for PersistenceError {
	fn from(err: io::Error) -> PersistenceError {
		PersistenceError::IOError(err)
	}
}
