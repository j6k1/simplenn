extern crate simplenn;

extern crate rand;
extern crate rand_xorshift;
extern crate statrs;

use rand::Rng;
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use statrs::distribution::Normal;

use simplenn::*;
use simplenn::function::activation::*;
use simplenn::function::loss::*;
use simplenn::function::optimizer::*;
use simplenn::persistence::*;
use simplenn::error::StartupError;
use simplenn::error::ConfigReadError;
use simplenn::error::InvalidStateError;

fn bits_to_vec(value:u32) -> Vec<f64> {
	let mut v = Vec::new();
	v.resize(8, 0f64);

	let mut index = 0;

	let mut value = value;

	while value > 0 {
		v[index] = (value & 1) as f64;
		value = value >> 1;
		index += 1;
	}

	v
}
#[test]
fn test_textfile_input_reader_with_contain_whitespace_data() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let r = NNModel::with_unit_initializer(
				NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
				.add((1,Box::new(FSigmoid::new()))),
				TextFileInputReader::new("data/contain_whitespace.txt").unwrap(),
				move || {
					n.sample(&mut rnd)
				});
	assert!(r.is_ok());
}
#[test]
fn test_textfile_input_reader_wrong_input_extra_data() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let r = NNModel::with_unit_initializer(
				NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
				.add((1,Box::new(FSigmoid::new()))),
				TextFileInputReader::new("data/longer_nn.txt").unwrap(),
				move || {
					n.sample(&mut rnd)
				});
	match r {
		Err(StartupError::Fail(ConfigReadError::InvalidState(s))) => {
			assert_eq!(String::from("Data loaded , but the input has not reached the end."),s);
		},
		_ => {
			assert!(false);
		}
	}
}
#[test]
fn test_binfile_input_reader_wrong_input_extra_data() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let r = NNModel::with_unit_initializer(
				NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
				.add((1,Box::new(FSigmoid::new()))),
				BinFileInputReader::new("data/longer_nn.bin").unwrap(),
				move || {
					n.sample(&mut rnd)
				});
	match r {
		Err(StartupError::Fail(ConfigReadError::InvalidState(s))) => {
			assert_eq!(String::from("Data loaded , but the input has not reached the end."),s);
		},
		_ => {
			assert!(false);
		}
	}
}
#[test]
fn test_textfile_input_reader_wrong_input_data_too_short() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let r = NNModel::with_unit_initializer(
				NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
				.add((2,Box::new(FSigmoid::new()))),
				TextFileInputReader::new("data/too_short_nn.txt").unwrap(),
				move || {
					n.sample(&mut rnd)
				});
	match r {
		Err(StartupError::Fail(ConfigReadError::InvalidState(s))) => {
			assert_eq!(String::from("End of input has been reached."),s);
		},
		_ => {
			assert!(false);
		}
	}
}
#[test]
fn test_binfile_input_reader_wrong_input_data_too_short() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let r = NNModel::with_unit_initializer(
				NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
				.add((2,Box::new(FSigmoid::new()))),
				BinFileInputReader::new("data/too_short_nn.bin").unwrap(),
				move || {
					n.sample(&mut rnd)
				});
	match r {
		Err(StartupError::Fail(ConfigReadError::IOError(_))) => {
			assert!(true);
		},
		_ => {
			assert!(false);
		}
	}
}
#[test]
fn test_relu_and_sigmoid() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
										.add((1,Box::new(FSigmoid::new()))),
									TextFileInputReader::new("data/initial_nn.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|_| SGD::new(0.05),Mse::new());

	let pairs:[([f64; 2],[f64; 1]); 4] = [([0f64,0f64],[0f64]),([0f64,1f64],[1f64]),([1f64,0f64],[1f64]),([1f64,1f64],[0f64])];
	let mut v:Vec<([f64; 2],[f64; 1])> = Vec::new();
	v.extend(&pairs);

	for _ in 0..10000 {
		for ii in 0..4 {
			let mut i = Vec::new();
			i.extend(&pairs[ii].0);
			let mut t = Vec::new();
			t.extend(&pairs[ii].1);
			nn.learn(&i, &t).unwrap();
		}
	}

	let validator:[Box<dyn Fn(&[f64]) -> bool>; 4] = [
		Box::new(|v| v[0] <= 0.1f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] <= 0.1f64),
	];

	for (p,validator) in pairs.iter().zip(&validator) {
		let mut i = Vec::new();
		i.extend(&p.0);
		let nnanswer = nn.solve(&i).unwrap();
		assert!(validator(&nnanswer));
	}
}
#[test]
fn test_relu_and_tanh() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FSigmoid::new())))
										.add((1,Box::new(FTanh::new()))),
									TextFileInputReader::new("data/initial_nn.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|_| SGD::new(0.1),Mse::new());

	let pairs:[([f64; 2],[f64; 1]); 4] = [([0f64,0f64],[0f64]),([0f64,1f64],[1f64]),([1f64,0f64],[1f64]),([1f64,1f64],[0f64])];
	let mut v:Vec<([f64; 2],[f64; 1])> = Vec::new();
	v.extend(&pairs);

	for _ in 0..10000 {
		for ii in 0..4 {
			let mut i = Vec::new();
			i.extend(&pairs[ii].0);
			let mut t = Vec::new();
			t.extend(&pairs[ii].1);
			nn.learn(&i, &t).unwrap();
		}
	}

	let validator:[Box<dyn Fn(&[f64]) -> bool>; 4] = [
		Box::new(|v| v[0] <= 0.1f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] <= 0.1f64),
	];

	for (p,validator) in pairs.iter().zip(&validator) {
		let mut i = Vec::new();
		i.extend(&p.0);
		let nnanswer = nn.solve(&i).unwrap();
		assert!(validator(&nnanswer));
	}
}
#[test]
fn test_relu_and_identity_and_sigmoid() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FIdentity::new())))
										.add((1,Box::new(FSigmoid::new()))),
									TextFileInputReader::new("data/initial_nn.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|_| SGD::new(0.1),Mse::new());

	let pairs:[([f64; 2],[f64; 1]); 4] = [([0f64,0f64],[0f64]),([0f64,1f64],[1f64]),([1f64,0f64],[1f64]),([1f64,1f64],[0f64])];
	let mut v:Vec<([f64; 2],[f64; 1])> = Vec::new();
	v.extend(&pairs);

	for _ in 0..10000 {
		for ii in 0..4 {
			let mut i = Vec::new();
			i.extend(&pairs[ii].0);
			let mut t = Vec::new();
			t.extend(&pairs[ii].1);
			nn.learn(&i, &t).unwrap();
		}
	}

	let validator:[Box<dyn Fn(&[f64]) -> bool>; 4] = [
		Box::new(|v| v[0] <= 0.1f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] <= 0.1f64),
	];

	for (p,validator) in pairs.iter().zip(&validator) {
		let mut i = Vec::new();
		i.extend(&p.0);
		let nnanswer = nn.solve(&i).unwrap();
		assert!(validator(&nnanswer));
	}
}
#[test]
fn test_fizzbuzz() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(8,(36,Box::new(FReLU::new())), (36,Box::new(FReLU::new())))
										.add((4,Box::new(FSoftMax::new()))),
									TextFileInputReader::new("data/initial_nn_8_36_36_4.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|n| Adam::new(n),CrossEntropyMulticlass::new());

	const FIZZBUZZ:[f64; 4] = [1f64,0f64,0f64,0f64];
	const FIZZ:[f64; 4] = [0f64,1f64,0f64,0f64];
	const BUZZ:[f64; 4] = [0f64,0f64,1f64,0f64];
	const OTHER:[f64; 4] = [0f64,0f64,0f64,1f64];

	for _ in 0..50 {
		for v in 101..256 {
			let i = bits_to_vec(v);

			let answer = if v % 15 == 0 {
				&FIZZBUZZ
			} else if v % 3 == 0 {
				&FIZZ
			} else if v % 5 == 0 {
				&BUZZ
			} else {
				&OTHER
			};

			let mut t = Vec::new();

			t.extend(answer);
			nn.learn(&i, &t).unwrap();
		}
	}

	for v in 101..111 {
		let i = bits_to_vec(v);

		let answer = if v % 15 == 0 {
			0
		} else if v % 3 == 0 {
			1
		} else if v % 5 == 0 {
			2
		} else {
			3
		};

		let nnanswer = nn.solve(&i).unwrap();

		assert_eq!(answer,nnanswer.iter().enumerate().fold((0,0f64),|acc,t| {
			if acc.1 < *t.1 {
				(t.0,*t.1)
			} else {
				acc
			}
		}).0);
	}
}
#[test]
fn test_cross_entropy() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FIdentity::new())))
										.add((1, Box::new(FSigmoid::new()))),
									TextFileInputReader::new("data/initial_nn.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|n| Adam::new(n),CrossEntropy::new());

	let pairs:[([f64; 2],[f64; 1]); 4] = [([0f64,0f64],[0f64]),([0f64,1f64],[1f64]),([1f64,0f64],[1f64]),([1f64,1f64],[0f64])];
	let mut v:Vec<([f64; 2],[f64; 1])> = Vec::new();
	v.extend(&pairs);

	for _ in 0..10000 {
		for ii in 0..4 {
			let mut i = Vec::new();
			i.extend(&pairs[ii].0);
			let mut t = Vec::new();
			t.extend(&pairs[ii].1);
			nn.learn(&i, &t).unwrap();
		}
	}

	let validator:[Box<dyn Fn(&[f64]) -> bool>; 4] = [
		Box::new(|v| v[0] <= 0.1f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] <= 0.1f64),
	];

	for (p,validator) in pairs.iter().zip(&validator) {
		let mut i = Vec::new();
		i.extend(&p.0);
		let nnanswer = nn.solve(&i).unwrap();
		assert!(validator(&nnanswer));
	}
}
#[test]
fn test_relu_and_adagrad() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
										.add((1,Box::new(FSigmoid::new()))),
									TextFileInputReader::new("data/initial_nn.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|n| Adagrad::new(n),CrossEntropy::new());

	let pairs:[([f64; 2],[f64; 1]); 4] = [([0f64,0f64],[0f64]),([0f64,1f64],[1f64]),([1f64,0f64],[1f64]),([1f64,1f64],[0f64])];
	let mut v:Vec<([f64; 2],[f64; 1])> = Vec::new();
	v.extend(&pairs);

	for _ in 0..100000 {
		for ii in 0..4 {
			let mut i = Vec::new();
			i.extend(&pairs[ii].0);
			let mut t = Vec::new();
			t.extend(&pairs[ii].1);
			nn.learn(&i, &t).unwrap();
		}
	}

	let validator:[Box<dyn Fn(&[f64]) -> bool>; 4] = [
		Box::new(|v| v[0] <= 0.1f64),
		Box::new(|v| v[0] >= 0.85f64),
		Box::new(|v| v[0] >= 0.85f64),
		Box::new(|v| v[0] <= 0.1f64),
	];

	for (p,validator) in pairs.iter().zip(&validator) {
		let mut i = Vec::new();
		i.extend(&p.0);
		let nnanswer = nn.solve(&i).unwrap();
		assert!(validator(&nnanswer));
	}
}
#[test]
fn test_relu_and_rmsprop() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
										.add((1,Box::new(FSigmoid::new()))),
									TextFileInputReader::new("data/initial_nn.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|n| RMSprop::new(n),CrossEntropy::new());

	let pairs:[([f64; 2],[f64; 1]); 4] = [([0f64,0f64],[0f64]),([0f64,1f64],[1f64]),([1f64,0f64],[1f64]),([1f64,1f64],[0f64])];
	let mut v:Vec<([f64; 2],[f64; 1])> = Vec::new();
	v.extend(&pairs);

	for _ in 0..20000 {
		for ii in 0..4 {
			let mut i = Vec::new();
			i.extend(&pairs[ii].0);
			let mut t = Vec::new();
			t.extend(&pairs[ii].1);
			nn.learn(&i, &t).unwrap();
		}
	}

	let validator:[Box<dyn Fn(&[f64]) -> bool>; 4] = [
		Box::new(|v| v[0] <= 0.1f64),
		Box::new(|v| v[0] >= 0.85f64),
		Box::new(|v| v[0] >= 0.85f64),
		Box::new(|v| v[0] <= 0.1f64),
	];

	for (p,validator) in pairs.iter().zip(&validator) {
		let mut i = Vec::new();
		i.extend(&p.0);
		let nnanswer = nn.solve(&i).unwrap();
		assert!(validator(&nnanswer));
	}
}
#[test]
fn test_relu_and_adam() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
										.add((1,Box::new(FSigmoid::new()))),
									TextFileInputReader::new("data/initial_nn.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|n| Adam::new(n),CrossEntropy::new());

	let pairs:[([f64; 2],[f64; 1]); 4] = [([0f64,0f64],[0f64]),([0f64,1f64],[1f64]),([1f64,0f64],[1f64]),([1f64,1f64],[0f64])];
	let mut v:Vec<([f64; 2],[f64; 1])> = Vec::new();
	v.extend(&pairs);

	for _ in 0..10000 {
		for ii in 0..4 {
			let mut i = Vec::new();
			i.extend(&pairs[ii].0);
			let mut t = Vec::new();
			t.extend(&pairs[ii].1);
			nn.learn(&i, &t).unwrap();
		}
	}

	let validator:[Box<dyn Fn(&[f64]) -> bool>; 4] = [
		Box::new(|v| v[0] <= 0.1f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] <= 0.1f64),
	];

	for (p,validator) in pairs.iter().zip(&validator) {
		let mut i = Vec::new();
		i.extend(&p.0);
		let nnanswer = nn.solve(&i).unwrap();
		assert!(validator(&nnanswer));
	}
}
#[test]
fn test_sgd_with_weight_decay_lambda() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
										.add((1,Box::new(FSigmoid::new()))),
									TextFileInputReader::new("data/initial_nn.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|_| SGD::with_lambda(0.05,0.0001),Mse::new());

	let pairs:[([f64; 2],[f64; 1]); 4] = [([0f64,0f64],[0f64]),([0f64,1f64],[1f64]),([1f64,0f64],[1f64]),([1f64,1f64],[0f64])];
	let mut v:Vec<([f64; 2],[f64; 1])> = Vec::new();
	v.extend(&pairs);

	for _ in 0..10000 {
		for ii in 0..4 {
			let mut i = Vec::new();
			i.extend(&pairs[ii].0);
			let mut t = Vec::new();
			t.extend(&pairs[ii].1);
			nn.learn(&i, &t).unwrap();
		}
	}

	let validator:[Box<dyn Fn(&[f64]) -> bool>; 4] = [
		Box::new(|v| v[0] <= 0.1f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] >= 0.9f64),
		Box::new(|v| v[0] <= 0.1f64),
	];

	for (p,validator) in pairs.iter().zip(&validator) {
		let mut i = Vec::new();
		i.extend(&p.0);
		let nnanswer = nn.solve(&i).unwrap();
		assert!(validator(&nnanswer));
	}
}
#[test]
fn test_latter_part_of_learning_with_not_consistent_snapshot() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
										.add((1,Box::new(FSigmoid::new()))),
									TextFileInputReader::new("data/initial_nn.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|_| SGD::new(0.05),Mse::new());

	let pair = ([0f64,1f64],[1f64]);

	let mut i = Vec::new();
	i.extend(&pair.0);
	let mut t = Vec::new();
	t.extend(&pair.1);
	let snapshot = nn.promise_of_learn(&i).unwrap();
	let _ = nn.promise_of_learn(&i).unwrap();

	let r = nn.latter_part_of_learning(&t,&snapshot);

	match r {
		Err(InvalidStateError::GenerationError(s)) => {
			assert_eq!(String::from(
				"Snapshot and model generation do not match. The snapshot used for learning needs to be the latest one."),s);
		},
		_ => {
			assert!(false);
		}
	}
}
#[test]
fn test_solve_diff() {
	let mut rnd = rand::thread_rng();
	let mut rnd = XorShiftRng::from_seed(rnd.gen());
	let n = Normal::new(0.0, 1.0).unwrap();

	let model = NNModel::with_unit_initializer(
									NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
										.add((1,Box::new(FSigmoid::new()))),
									TextFileInputReader::new("data/initial_nn.txt").unwrap(),
									move || {
										n.sample(&mut rnd)
									}).unwrap();
	let mut nn = NN::new(model,|_| SGD::new(0.05),Mse::new());

	let pairs:[([f64; 2],[f64; 1]); 4] = [([0f64,0f64],[0f64]),([0f64,1f64],[1f64]),([1f64,0f64],[1f64]),([1f64,1f64],[0f64])];
	let mut v:Vec<([f64; 2],[f64; 1])> = Vec::new();
	v.extend(&pairs);

	for _ in 0..10000 {
		for ii in 0..4 {
			let mut i = Vec::new();
			i.extend(&pairs[ii].0);
			let mut t = Vec::new();
			t.extend(&pairs[ii].1);
			nn.learn(&i, &t).unwrap();
		}
	}

	let mut i = Vec::new();
	i.extend(&pairs[2].0);
	let snapshot = nn.solve_shapshot(&i).unwrap();

	let input:Vec<Vec<(usize,f64)>> = vec![vec![(0,-1f64)],vec![(0,-1f64),(1,1f64)],vec![(1,1f64)]];
	let mut answer = Vec::new();

	for &index in [0,1,3].iter() {
		let mut i = Vec::new();
		i.extend(&pairs[index as usize].0);
		let nnanswer = nn.solve(&i).unwrap();
		answer.push(nnanswer);
	}


	for (i,answer) in input.iter().zip(&answer) {
		let nnanswer = nn.solve_diff(i,&snapshot).unwrap().r;
		assert_eq!(*answer,nnanswer);
	}
}

