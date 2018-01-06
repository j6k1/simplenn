use std::io;
use std::fmt;
use std::error;

#[derive(Debug)]
pub enum StartupError {
	InvalidConfiguration(String),
	IOError(io::Error),
}
impl fmt::Display for StartupError {
	 fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	 	match *self {
	 		StartupError::InvalidConfiguration(ref s) => write!(f, "{}",s),
	 		StartupError::IOError(_) => write!(f,"Error occurred in file I/O."),
	 	}
	 }
}
impl error::Error for StartupError {
	 fn description(&self) -> &str {
	 	match *self {
	 		StartupError::InvalidConfiguration(_) => "Configuration is invalid.",
	 		StartupError::IOError(_) => "Error occurred in file I/O.",
	 	}
	 }

	fn cause(&self) -> Option<&error::Error> {
	 	match *self {
	 		StartupError::InvalidConfiguration(_) => None,
	 		StartupError::IOError(ref e) => Some(e),
	 	}
	 }
}
impl From<io::Error> for StartupError {
	fn from(err: io::Error) -> StartupError {
		StartupError::IOError(err)
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
