use std::fmt::Display;

#[derive(Debug)]
pub enum ZError {
    FileError(std::io::Error),
    JsonError(serde_json::Error),
    Message(String),
    BincodeEncode(bincode::error::EncodeError),
    BincodeDecode(bincode::error::DecodeError),
}

impl Display for ZError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ZError::FileError(ref err) => std::fmt::Display::fmt(&err, f),
            ZError::JsonError(ref err) => std::fmt::Display::fmt(&err, f),
            ZError::Message(ref err) => std::fmt::Display::fmt(&err, f),
            ZError::BincodeEncode(ref err) => std::fmt::Display::fmt(&err, f),
            ZError::BincodeDecode(ref err) => std::fmt::Display::fmt(&err, f),
        }
    }
}

impl From<std::io::Error> for ZError {
    fn from(err: std::io::Error) -> ZError {
        ZError::FileError(err)
    }
}

impl From<serde_json::Error> for ZError {
    fn from(err: serde_json::Error) -> ZError {
        ZError::JsonError(err)
    }
}

impl From<String> for ZError {
    fn from(err: String) -> ZError {
        ZError::Message(err)
    }
}

impl From<bincode::error::EncodeError> for ZError {
    fn from(err: bincode::error::EncodeError) -> ZError {
        ZError::BincodeEncode(err)
    }
}

impl From<bincode::error::DecodeError> for ZError {
    fn from(err: bincode::error::DecodeError) -> ZError {
        ZError::BincodeDecode(err)
    }
}

pub type Result<T> = std::result::Result<T, ZError>;
