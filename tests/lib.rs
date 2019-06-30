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

#[test]
fn test_solve() {
	for _ in 0..2 {
		let mut rnd = rand::thread_rng();
		let mut rnd = XorShiftRng::from_seed(rnd.gen());
		let n = Normal::new(0.0, 1.0).unwrap();

		let model = NNModel::with_unit_initializer(
										NNUnits::new(2, (4,Box::new(FReLU::new())), (4,Box::new(FReLU::new())))
											.add((1,Box::new(FSigmoid::new()))),
										TextFileInputReader::new("data/nn.txt").unwrap(),
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

		let validator:[Box<Fn(&Vec<f64>) -> bool>; 4] = [
			Box::new(|v| v[0] <= 0.1f64),
			Box::new(|v| v[0] >= 0.9f64),
			Box::new(|v| v[0] >= 0.9f64),
			Box::new(|v| v[0] <= 0.1f64),
		];

		for (p,validator) in pairs.iter().zip(&validator) {
			let mut i = Vec::new();
			i.extend(&p.0);
			let nnanswer = nn.solve(&i).unwrap();
			if validator(&nnanswer) {
				assert!(true);
				return;
			}
		}
	}

	assert!(false);
}
