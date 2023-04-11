use std::fs::File;
use std::io::prelude::*;

pub fn save_string_to_file(in_string: &String, file_path: &str) -> std::io::Result<()>
{
    let mut file = File::create(file_path)?;
    file.write_all( in_string.as_bytes())?;

    Ok(())
}