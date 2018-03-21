use std::env;

fn main() {
	#[cfg(target_os = "windows")]
	match env::var("CUDA_PATH") {
	    Ok(cuda_path) => println!("cargo:rustc-link-search={}/lib/x64", cuda_path),
	    Err(_) => (),
	}
}
