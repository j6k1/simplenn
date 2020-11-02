extern crate rand;
extern crate rand_xorshift;

pub mod function;
pub mod error;
pub mod persistence;

use std::fmt;
use error::*;
use std::error::Error;
use rand::Rng;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

use function::activation::*;
use function::loss::*;
use function::optimizer::*;

pub struct Metrics {
	error_total:f64,
	error_average:f64
}
pub struct NN<O,E> where O: Optimizer, E: LossFunction {
	model:NNModel,
	optimizer:O,
	lossf:E,
}

impl<O,E> NN<O,E> where O: Optimizer, E: LossFunction {
	pub fn new<F>(model:NNModel,optimizer_creator:F,lossf:E) -> NN<O,E>
		where F: Fn(usize) -> O {
		let mut size = 0;

		for i in 0..model.units.len() - 1 {
			size += model.units[i].0 + 1;
		}
		NN {
			model:model,
			optimizer:optimizer_creator(size),
			lossf:lossf,
		}
	}

	pub fn promise_of_learn(&mut self,input:&[f64]) ->
		Result<SnapShot,InvalidStateError> {

		self.model.promise_of_learn(input)
	}

	pub fn solve(&self,input:&[f64]) ->
		Result<Vec<f64>,InvalidStateError> {

		self.model.solve(input)
	}

	pub fn solve_shapshot(&self,input:&[f64]) ->
		Result<SnapShot,InvalidStateError> {

		self.model.solve_shapshot(input)
	}

	pub fn solve_diff(&self,input:&[(usize,f64)],snapshot:&SnapShot) ->
		Result<SnapShot,InvalidStateError> {

		self.model.solve_diff(input,snapshot)
	}

	pub fn learn(&mut self,input:&[f64],t:&[f64]) -> Result<Metrics,InvalidStateError>
		where O: Optimizer, E: LossFunction {

		Ok(self.model.learn(input,&t,&mut self.optimizer,&self.lossf)?)
	}

	pub fn latter_part_of_learning(&mut self, t:&[f64],s:&SnapShot) ->
		Result<Metrics,InvalidStateError> {

		Ok(self.model.latter_part_of_learning(t,s,&mut self.optimizer,&self.lossf)?)
	}

	pub fn save<P,ERR>(&self,mut persistence:P) -> Result<(),PersistenceError<ERR>>
		where P: Persistence<ERR>, ERR: Error + fmt::Debug, PersistenceError<ERR>: From<ERR> {
		persistence.save(&self.model.layers)?;

		Ok(())
	}
}
pub struct NNUnits {
	input_units:usize,
	defs:Vec<(usize,Box<dyn ActivateF>)>,
}
impl NNUnits {
	pub fn new(input_units:usize, l1:(usize,Box<dyn ActivateF>),l2:(usize,Box<dyn ActivateF>)) -> NNUnits {
		let mut defs:Vec<(usize,Box<dyn ActivateF>)> = Vec::new();
		defs.push(l1);
		defs.push(l2);
		NNUnits {
			input_units:input_units,
			defs:defs
		}
	}

	pub fn add(mut self, units:(usize,Box<dyn ActivateF>)) -> NNUnits {
		self.defs.push(units);
		self
	}
}
pub struct NNModel {
	units:Vec<(usize,Option<Box<dyn ActivateF>>)>,
	layers:Vec<Vec<Vec<f64>>>,
	hash:u64,
}
impl NNModel {
	pub fn load<I,E>(mut reader:I) -> Result<NNModel, E>
		where I: ModelInputReader<E>, E: Error, StartupError<E>: From<E> {
		reader.read_model()
	}

	pub fn new<E>(units:Vec<(usize,Option<Box<dyn ActivateF>>)>,layers:Vec<Vec<Vec<f64>>>) -> Result<NNModel,StartupError<E>>
		where E: Error + fmt::Debug, StartupError<E>: From<E> {

		let mut rnd = rand::thread_rng();
		let mut rnd = XorShiftRng::from_seed(rnd.gen());

		if layers.len() != units.len() - 1 {
			return Err(StartupError::InvalidConfiguration(format!("The layers count do not match. (units = {}, layers = {})", units.len(), layers.len())));
		}
		else
		{
			for i in 0..layers.len() {
				if units[i].0+1 != layers[i].len()
				{
					return Err(StartupError::InvalidConfiguration(format!(
							"The number of units in Layer {} do not match. (correct size = {}, size = {})",i,units[i].0 + 1,layers[i].len())));
				}

				for j in 0..units[i].0+1 {
					match layers[i][j].len() {
						len if len != units[i+1].0 => {
							return Err(StartupError::InvalidConfiguration(format!(
								"Weight {} is defined for unit {} in layer {}, but this does not match the number of units in the lower layer.",
								layers[i][j].len(), i, units[i+1].0
							)));
						},
						_ => (),
					}
				}
			}
		}

		Ok(NNModel {
			units:units,
			layers:layers,
			hash:rnd.gen(),
		})
	}

	pub fn with_unit_initializer<I,F,E>(units:NNUnits,
												reader:I,
												mut initializer:F) ->
		Result<NNModel,StartupError<E>>
		where I: InputReader<E>, F: FnMut() -> f64, E: Error + fmt::Debug, StartupError<E>: From<E> {

		let iunits = units.input_units;
		let mut sunits = units.defs.iter().map(|u| u.0).collect::<Vec<usize>>();
		sunits.insert(0, iunits);
		let sunits = sunits;

		NNModel::with_schema(units,reader,move || {
			let mut layers:Vec<Vec<Vec<f64>>> = Vec::with_capacity(sunits.len());

			for i in 0..sunits.len() - 1 {
				let mut layer:Vec<Vec<f64>> = Vec::with_capacity(sunits[i]+1);

				for _ in 0..sunits[i] + 1 {
					let mut unit:Vec<f64> = Vec::with_capacity(sunits[i+1]);
					for _ in 0..sunits[i+1] {
						unit.push(initializer());
					}
					layer.push(unit);
				}

				layers.push(layer);
			}

			layers
		})
	}

	pub fn with_bias_and_unit_initializer<I,F,B,E>(units:NNUnits,
												reader:I,
												mut bias_initializer:B,
												mut initializer:F) ->
		Result<NNModel,StartupError<E>>
		where I: InputReader<E>, F: FnMut() -> f64,
				B: FnMut() -> f64, E: Error + fmt::Debug, StartupError<E>: From<E> {

		let iunits = units.input_units;
		let mut sunits = units.defs.iter().map(|u| u.0).collect::<Vec<usize>>();
		sunits.insert(0, iunits);
		let sunits = sunits;

		NNModel::with_schema(units,reader,move || {
			let mut layers:Vec<Vec<Vec<f64>>> = Vec::with_capacity(sunits.len());

			for i in 0..sunits.len() - 1 {
				let mut layer:Vec<Vec<f64>> = Vec::with_capacity(sunits[i]+1);

				let mut unit:Vec<f64> = Vec::with_capacity(sunits[i+1]);

				for _ in 0..sunits[i+1] {
					unit.push(bias_initializer());
				}

				layer.push(unit);

				for _ in 1..sunits[i] + 1 {
					let mut unit:Vec<f64> = Vec::with_capacity(sunits[i+1]);
					for _ in 0..sunits[i+1] {
						unit.push(initializer());
					}
					layer.push(unit);
				}

				layers.push(layer);
			}

			layers
		})
	}

	pub fn with_schema<I,F,E>(units:NNUnits,mut reader:I,mut initializer:F) ->
		Result<NNModel,StartupError<E>>
		where I: InputReader<E>, F: FnMut() -> Vec<Vec<Vec<f64>>>, E: Error + fmt::Debug, StartupError<E>: From<E> {

		let iunits = units.input_units;

		let mut units:Vec<(usize,Option<Box<dyn ActivateF>>)> = units
															.defs
															.into_iter()
															.map(|(u,f)| (u, Some(f)))
															.collect();

		units.insert(0, (iunits, None));

		let units = units;

		let layers = match reader.source_exists() {
			true => {
				let mut layers:Vec<Vec<Vec<f64>>> = Vec::new();

				for i in 0..units.len()-1 {
					let size = match units[i] {
						(size,_) => size + 1,
					};
					let sizeb = match units[i+1] {
						(size,_) => size,
					};
					layers.push(reader.read_vec(size as usize,sizeb as usize)?);
				}

				reader.verify_eof()?;

				layers
			},
			false => initializer(),
		};

		NNModel::new(
			units,
			layers,
		)
	}

	pub fn apply<F,R>(&self,input:&[f64],after_callback:F) -> Result<R,InvalidStateError>
		where F: Fn(Vec<f64>,Vec<Vec<f64>>,Vec<Vec<f64>>) -> Result<R,InvalidStateError> {
		if input.len() != self.units[0].0 {
			return Err(InvalidStateError::InvalidInput(String::from(
				"The inputs to the input layer is invalid (the count of inputs must be the count of units)")));
		}

		let mut o:Vec<Vec<f64>> = Vec::with_capacity(self.units.len());
		let mut u:Vec<Vec<f64>> = Vec::with_capacity(self.units.len());

		u.push(Vec::new());

		let mut oi:Vec<f64> = Vec::with_capacity(self.units[0].0 + 1);

		oi.push(1f64);

		for i in input {
			oi.push(*i);
		}

		o.push(oi);

		u.push(Vec::with_capacity(self.units[1].0 + 1));

		u[1].resize(self.units[1].0 + 1, 0f64);

		for (o,wl) in o[0].iter().zip(&self.layers[0]) {
			for (u,w) in u[1].iter_mut().skip(1).zip(wl) {
				*u += o * w;
			}
		}

		self.apply_middle_and_out(o,u,after_callback)
	}

	pub fn apply_diff<F,R>(&self,input:&[(usize,f64)],s:&SnapShot,after_callback:F) -> Result<R,InvalidStateError>
		where F: Fn(Vec<f64>,Vec<Vec<f64>>,Vec<Vec<f64>>) -> Result<R,InvalidStateError> {
		let mut o:Vec<Vec<f64>> = Vec::with_capacity(self.units.len());
		let mut u:Vec<Vec<f64>> = Vec::with_capacity(self.units.len());

		u.push(s.u[0].clone());

		let mut oi:Vec<f64> = s.o[0].clone();

		for &(i,d) in input {
			// インデックス0はバイアスのユニットなので一つ右にずらす
			oi[i+1] += d;
		}

		o.push(oi);

		let mut ui = s.u[1].clone();

		for &(i,d) in input {
			// インデックス0はバイアスのユニットなので一つ右にずらす
			for (u,w) in ui.iter_mut().skip(1).zip(&self.layers[0][i+1]) {
				*u += d * w;
			}
		}

		u.push(ui);

		self.apply_middle_and_out(o,u,after_callback)
	}

	fn apply_middle_and_out<F,R>(&self,mut o:Vec<Vec<f64>>,mut u:Vec<Vec<f64>>,after_callback:F) -> Result<R,InvalidStateError>
		where F: Fn(Vec<f64>,Vec<Vec<f64>>,Vec<Vec<f64>>) -> Result<R,InvalidStateError> {
		o.push(Vec::with_capacity(self.units[1].0 + 1));
		o[1].resize(self.units[1].0 + 1, 0f64);

		let f:&Box<dyn ActivateF> = match self.units[1].1 {
			Some(ref f) => f,
			None => {
				return Err(InvalidStateError::InvalidInput(String::from(
							"Reference to the activation function object is not set."
						)));
			}
		};

		o[1][0] = 1f64;

		for (oi,&ui) in o[1].iter_mut().zip(u[1].iter()) {
			*oi = f.apply(ui,&u[1]);
		}

		for l in 1..self.units.len() - 1 {
			let ll = l + 1;
			let mut ul:Vec<f64> = Vec::with_capacity(self.units[ll].0 + 1);
			ul.resize(self.units[ll].0 + 1, 0f64);
			u.push(ul);
			let f:&Box<dyn ActivateF> = match self.units[ll].1 {
				Some(ref f) => f,
				None => {
					return Err(InvalidStateError::InvalidInput(String::from(
								"Reference to the activation function object is not set."
							)));
				}
			};

			let mut ol:Vec<f64> = Vec::with_capacity(self.units[ll].0 + 1);
			ol.resize(self.units[ll].0 + 1, 0f64);
			o.push(ol);

			o[ll][0] = 1f64;

			for (o,wl) in o[l].iter().zip(&self.layers[l]) {
				for (u,w) in u[ll].iter_mut().skip(1).zip(wl) {
					*u = *u + o * w;
				}
			}

			let u = &u[ll];

			for (o,ui) in o[ll].iter_mut().skip(1).zip(u.iter().skip(1)) {
				*o = f.apply(*ui,u);
			}
		}

		let mut r:Vec<f64> = Vec::with_capacity(self.units[self.units.len()-1].0);

		for &oi in o[self.units.len()-1].iter().skip(1) {
			r.push(oi);
		}

		after_callback(r,o,u)
	}

	fn latter_part_of_learning<O,E>(&mut self, t:&[f64],s:&SnapShot,optimizer:&mut O,lossf:&E) ->
		Result<Metrics,InvalidStateError> where O: Optimizer, E: LossFunction {

		if s.hash.map(|h| h != self.hash).unwrap_or(false) {
			return Err(InvalidStateError::GenerationError(String::from(
				"Snapshot and model generation do not match. The snapshot used for learning needs to be the latest one."
			)));
		}

		let mut metrics = Metrics {
			error_total:0f64,
			error_average:0f64
		};

		let l = self.units.len()-1;

		for k in 1..self.units[l].0 + 1 {
			metrics.error_total += lossf.apply(s.r[k-1],t[k-1]);
		}

		let mut layers:Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.units.len()-1);
		let mut d:Vec<f64> = Vec::with_capacity(self.units[self.units.len()-1].0 + 1);
		d.resize(self.units[self.units.len()-1].0 + 1, 0f64);

		for l in 0..self.units.len() - 1 {
			let mut layer:Vec<Vec<f64>> = Vec::with_capacity(self.units[l].0 + 1);

			layer.resize(self.units[l].0 + 1,Vec::with_capacity(self.units[l+1].0 + 1));
			layers.push(layer);
		}

		let f:&Box<dyn ActivateF> = match self.units[self.units.len()-1].1 {
			Some(ref f) => f,
			None => {
				return Err(InvalidStateError::InvalidInput(String::from(
							"Reference to the activation function object is not set."
						)));
			}
		};

		let size = self.units[self.units.len()-1].0;

		for l in layers[self.units.len()-2].iter_mut() {
			l.resize(size,0f64);
		}

		let hl = self.units.len()-2;
		let l = self.units.len()-1;
		match lossf.is_canonical_link(&f) {
			true => {
				for k in 1..self.units[l].0 + 1 {
					d[k] = s.r[k-1] - t[k-1];
				}
			},
			false => {
				for k in 1..self.units[l].0 + 1 {
					d[k] = (lossf.derive(s.r[k-1], t[k-1])) * f.derive(s.u[l][k]);
				}
			}
		}

		for i in 0..self.units[hl].0 + 1 {
			let o = s.o[hl][i];
			let mut e:Vec<f64> = Vec::with_capacity(self.units[l].0);
			e.resize(self.units[l].0,0f64);
			for j in 1..self.units[l].0 + 1 {
				e[j-1] = d[j] * o;
			}
			optimizer.update((hl,i),&e,&self.layers[hl][i],&mut layers[hl][i]);
		}

		for l in (1..self.units.len()-1).rev() {
			let hl = l - 1;
			let ll = l + 1;
			let f:&Box<dyn ActivateF> = match self.units[l].1 {
				Some(ref f) => f,
				None => {
					return Err(InvalidStateError::InvalidInput(String::from(
								"Reference to the activation function object is not set."
							)));
				}
			};

			for li in layers[hl].iter_mut() {
				li.resize(self.units[l].0,0f64);
			}

			let mut nd:Vec<f64> = Vec::with_capacity(self.units[l].0 + 1);
			nd.resize(self.units[l].0 + 1, 0f64);

			for j in 1..self.units[l].0 + 1{
				for k in 1..self.units[ll].0 + 1 {
					nd[j] += self.layers[l][j][k-1] * d[k];
				}
				nd[j] = nd[j] * f.derive(s.u[l][j]);
			}

			for i in 0..self.units[hl].0 + 1 {
				let o = s.o[hl][i];
				let mut e:Vec<f64> = Vec::with_capacity(self.units[l].0);
				e.resize(self.units[l].0,0f64);
				for j in 1..self.units[l].0 + 1 {
					e[j-1] = nd[j] * o;
				}
				optimizer.update((hl,i),&e,&self.layers[hl][i],&mut layers[hl][i]);
			}

			d = nd;
		}

		self.layers = layers;

		metrics.error_average = metrics.error_total;

		Ok(metrics)
	}

	fn solve(&self,input:&[f64]) ->
		Result<Vec<f64>,InvalidStateError> {

		self.apply(input,|r,_,_| Ok(r))
	}

	fn solve_diff(&self,input:&[(usize,f64)],s:&SnapShot) -> Result<SnapShot,InvalidStateError> {
		self.apply_diff(input,s,|r,o,u| Ok(SnapShot::new(r,o,u,None)))
	}

	fn learn<O,E>(&mut self,input:&[f64],t:&[f64],optimizer:&mut O,lossf:&E) -> Result<Metrics,InvalidStateError>
		where O: Optimizer, E: LossFunction {

		let s = self.promise_of_learn(input)?;

		self.latter_part_of_learning(t,&s,optimizer,lossf)
	}

	fn solve_shapshot(&self,input:&[f64]) ->
		Result<SnapShot,InvalidStateError> {
		self.apply(input,|r,o,u| Ok(SnapShot::new(r,o,u,None)))
	}

	fn promise_of_learn(&mut self,input:&[f64]) ->
		Result<SnapShot,InvalidStateError> {

		let mut rnd = rand::thread_rng();
		let mut rnd = XorShiftRng::from_seed(rnd.gen());
		self.hash = rnd.gen();

		self.apply(input,|r,o,u| Ok(SnapShot::new(r,o,u,Some(self.hash))))
	}
}
pub struct SnapShot {
	pub r:Vec<f64>,
	o:Vec<Vec<f64>>,
	u:Vec<Vec<f64>>,
	hash:Option<u64>,
}
impl SnapShot {
	pub fn new(r:Vec<f64>,o:Vec<Vec<f64>>,u:Vec<Vec<f64>>,hash:Option<u64>) -> SnapShot {
		SnapShot {
			r:r,
			o:o,
			u:u,
			hash:hash,
		}
	}

	pub fn get_result(&self) -> Vec<f64> {
		self.r.clone()
	}
}
pub trait InputReader<E> where E: Error + fmt::Debug, StartupError<E>: From<E> {
	fn read_vec(&mut self,usize,usize) -> Result<Vec<Vec<f64>>,E>;
	fn source_exists(&mut self) -> bool;
	fn verify_eof(&mut self) -> Result<(),E>;
}
pub trait ModelInputReader<E> where E: Error + fmt::Debug, StartupError<E>: From<E> {
	fn read_model<'a>(&mut self) -> Result<NNModel, E>;
}
pub trait Persistence<E> where E: Error + fmt::Debug, PersistenceError<E>: From<E> {
	fn save(&mut self,layers:&Vec<Vec<Vec<f64>>>) -> Result<(),E>;
}