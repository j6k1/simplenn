extern crate rand;

pub mod function;
pub mod error;
pub mod persistence;

use function::activation::*;
use function::loss::*;
use function::optimizer::*;
use error::*;
use std::error::Error;
use rand::Rng;

pub struct NN<'a,O,E,F> where O: Optimizer, E: LossFunction, F: Fn() -> O {
	model:&'a mut NNModel<'a>,
	optimizer_creator:F,
	lossf:E,
}

impl<'a,O,E,F> NN<'a,O,E,F> where O: Optimizer, E: LossFunction, F: Fn() -> O {
	pub fn new(model:&'a mut NNModel<'a>,optimizer_creator:F,lossf:E) -> NN<'a,O,E,F> {
		NN {
			model:model,
			optimizer_creator:optimizer_creator,
			lossf:lossf,
		}
	}

	pub fn promise_of_learn(&mut self,input:Vec<f64>) ->
		Result<SnapShot,InvalidStateError> {

		self.model.apply(input,|r,o,u| Ok(SnapShot::new(r,o,u,self.model.hash)))
	}

	pub fn solve(&mut self,input:Vec<f64>) ->
		Result<Vec<f64>,InvalidStateError> {

		self.model.solve(input)
	}

	pub fn learn(&mut self,input:Vec<f64>,t:Vec<f64>) -> Result<(),InvalidStateError>
		where O: Optimizer, E: LossFunction {

		Ok(self.model.learn(input,&t,(self.optimizer_creator)(),&self.lossf)?)
	}

	pub fn latter_part_of_learning(&mut self, t:&Vec<f64>,s:SnapShot) ->
		Result<(),InvalidStateError> {

		Ok(self.model.latter_part_of_learning(t,s,(self.optimizer_creator)(),&self.lossf)?)
	}

	pub fn save<P,ERR>(&mut self,mut persistence:P) -> Result<(),PersistenceError<ERR>>
		where P: Persistence<ERR>, ERR: Error, PersistenceError<ERR>: From<ERR> {
		persistence.save(&self.model.layers)?;

		Ok(())
	}
}
pub struct NNModel<'a> {
	units:Vec<(usize,Option<&'a ActivateF>)>,
	layers:Vec<Vec<Vec<f64>>>,
	hash:u64,
}
impl<'a> NNModel<'a> {
	pub fn load<I,E>(mut reader:I) -> Result<NNModel<'a>, E>
		where I: ModelInputReader<E>, E: Error, StartupError<E>: From<E> {
		reader.read_model()
	}

	fn new(units:Vec<(usize,Option<&'a ActivateF>)>,layers:Vec<Vec<Vec<f64>>>) -> NNModel<'a> {
		let mut rnd = rand::XorShiftRng::new_unseeded();
		NNModel {
			units:units,
			layers:layers,
			hash:rnd.next_u64(),
		}
	}

	pub fn with_bias_and_unit_initializer<I,F,E>(
												iunits:usize,
												units:Vec<(usize,&'a ActivateF)>,
												reader:I,bias:f64,
												initializer:F) ->
		Result<NNModel<'a>,StartupError<E>>
		where I: InputReader<E>, F: Fn() -> f64, E: Error, StartupError<E>: From<E> {

		match units.len() {
			l if l < 2 => {
				return Err(StartupError::InvalidConfiguration(
					String::from(
						"Parameter of layer number of middle layers of multilayer perceptron is invalid (less than 2)")));
			}
			_ => (),
		}

		let mut sunits = units.iter().map(|u| u.0).collect::<Vec<usize>>();
		sunits.insert(0, iunits);
		let sunits = sunits;

		NNModel::with_schema(iunits,units,reader,move || {
			let mut layers:Vec<Vec<Vec<f64>>> = Vec::with_capacity(sunits.len());

			for i in 0..sunits.len() - 1 {
				let mut layer:Vec<Vec<f64>> = Vec::with_capacity(sunits[i]);

				let mut unit:Vec<f64> = Vec::with_capacity(sunits[i+1]);

				unit.resize(sunits[i+1], bias);
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

	pub fn with_list_of_bias_and_unit_initializer<I,F,E>(
												iunits:usize,
												units:Vec<(usize,&'a ActivateF)>,
												reader:I,
												init_list:Vec<(f64,F)>) ->
		Result<NNModel<'a>,StartupError<E>>
		where I: InputReader<E>, F: Fn() -> f64, E: Error, StartupError<E>: From<E> {

		match units.len() {
			l if l < 2 => {
				return Err(StartupError::InvalidConfiguration(
					String::from(
						"Parameter of layer number of middle layers of multilayer perceptron is invalid (less than 2)")));
			}
			_ => (),
		}

		if init_list.len() != units.len() - 1 {
			return Err(StartupError::InvalidConfiguration(format!("The layers count do not match. (units = {}, count of init_list = {})", units.len(), init_list.len())));
		}

		let mut sunits = units.iter().map(|u| u.0).collect::<Vec<usize>>();
		sunits.insert(0, iunits);
		let sunits = sunits;

		NNModel::with_schema(iunits,units,reader,move || {
			let mut layers:Vec<Vec<Vec<f64>>> = Vec::with_capacity(sunits.len());

			for i in 0..sunits.len() - 1 {
				let mut layer:Vec<Vec<f64>> = Vec::with_capacity(sunits[i]);

				let mut unit:Vec<f64> = Vec::with_capacity(sunits[i+1]);

				unit.resize(sunits[i+1], init_list[i].0);
				layer.push(unit);

				for _ in 1..sunits[i] + 1 {
					let mut unit:Vec<f64> = Vec::with_capacity(sunits[i+1]);
					for _ in 0..sunits[i+1] {
						unit.push(init_list[i].1());
					}
					layer.push(unit);
				}

				layers.push(layer);
			}

			layers
		})
	}

	pub fn with_schema<I,F,E>(iunits:usize,units:Vec<(usize,&'a ActivateF)>,mut reader:I,initializer:F) ->
		Result<NNModel<'a>,StartupError<E>>
		where I: InputReader<E>, F: Fn() -> Vec<Vec<Vec<f64>>>, E: Error, StartupError<E>: From<E> {

		match units.len() {
			l if l < 2 => {
				return Err(StartupError::InvalidConfiguration(
					String::from(
						"Parameter of layer number of middle layers of multilayer perceptron is invalid (less than 2)")));
			}
			_ => (),
		}

		let mut units:Vec<(usize,Option<&'a ActivateF>)> = units.iter()
														.map(|&(u,f)| (u, Some(f)))
														.collect();

		units.insert(0, (iunits, None));

		let units = units;

		let layers = match reader.source_exists() {
			true => {
				let mut layers:Vec<Vec<Vec<f64>>> = Vec::new();

				for i in 0..units.len()-1 {
					let size = match units[i] {
						(size,_) => size,
					};
					let sizeb = match units[i+1] {
						(size,_) => size,
					};
					layers.push(reader.read_vec(size as usize,sizeb as usize)?);
				}
				layers
			},
			false => initializer(),
		};

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
						len if i ==  layers.len() - 1 && len != units[i+1].0 => {
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

		Ok(NNModel::new(
			units,
			layers,
		))
	}

	pub fn apply<F,R>(&self,input:Vec<f64>,after_callback:F) -> Result<R,InvalidStateError>
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
			oi.push(i);
		}

		o.push(oi);

		u.push(Vec::with_capacity(self.units[1].0 + 1));

		u[1].resize(self.units[1].0 + 1, 0f64);

		for k in 1..self.units[1].0 + 1 {
			for j in 0..self.units[0].0 + 1 {
				u[1][k] += o[0][j] * self.layers[0][j][k-1];
			}
		}

		o.push(Vec::with_capacity(self.units[1].0 + 1));
		o[1].resize(self.units[1].0 + 1, 0f64);

		let f:&ActivateF = match self.units[1].1 {
			Some(f) => f,
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

		for l in 0..self.units.len() - 1 {
			let ll = l + 1;
			let ul:Vec<f64> = Vec::with_capacity(self.units[ll].0 + 1);
			u.push(ul);
			let f:&ActivateF = match self.units[l].1 {
				Some(f) => f,
				None => {
					return Err(InvalidStateError::InvalidInput(String::from(
								"Reference to the activation function object is not set."
							)));
				}
			};

			let ol:Vec<f64> = Vec::with_capacity(self.units[ll].0 + 1);
			o.push(ol);

			o[ll][0] = 1f64;

			for k in 1..self.units[ll].0 + 1 {
				for j in 0..self.units[l].0 + 1 {
					u[ll][k] = u[ll][k] + o[l][j] * self.layers[l][j][k-1];
				}

				o[ll][k] = f.apply(u[ll][k],&u[ll]);
			}
		}

		let mut r:Vec<f64> = Vec::with_capacity(self.units[self.units.len()-1].0);

		for &oi in o[self.units.len()-1].iter() {
			r.push(oi);
		}

		after_callback(r,o,u)
	}

	fn latter_part_of_learning<O,E>(&mut self, t:&Vec<f64>,s:SnapShot,mut optimizer:O,lossf:&E) ->
		Result<(),InvalidStateError> where O: Optimizer, E: LossFunction {

		if s.hash != self.hash {
			return Err(InvalidStateError::GenerationError(String::from(
				"Snapshot and model generation do not match. The snapshot used for learning needs to be the latest one."
			)));
		}

		let mut layers:Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.units.len()-1);
		let mut d:Vec<f64> = Vec::with_capacity(self.units[self.units.len()-1].0 + 1);
		d.resize(self.units.len()-1, 0f64);

		for l in 0..self.units.len() - 1 {
			let layer:Vec<Vec<f64>> = Vec::with_capacity(self.units[l].0 + 1);

			layers.push(layer);
		}

		let f:&ActivateF = match self.units[1].1 {
			Some(f) => f,
			None => {
				return Err(InvalidStateError::InvalidInput(String::from(
							"Reference to the activation function object is not set."
						)));
			}
		};

		let size = self.units[self.units.len()-1].0;

		for l in self.layers[self.units.len()-2].iter_mut() {
			l.resize(size,0f64);
		}

		let hl = self.units.len()-2;
		let l = self.units.len()-1;
		match lossf.is_canonical_link(f) {
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

		for j in 0..self.units[hl].0 + 1 {
			optimizer.update(&d[1..],&self.layers[hl][j],&mut layers[hl][j]);
			let o = s.o[hl][j];
			for k in 0..self.units[l].0 {
				layers[hl][j][k-1] = layers[hl][j][k-1] * o;
			}
		}

		for l in (0..self.units.len()-2).rev() {
			let hl = l - 1;
			let ll = l + 1;
			let f:&ActivateF = match self.units[1].1 {
				Some(f) => f,
				None => {
					return Err(InvalidStateError::InvalidInput(String::from(
								"Reference to the activation function object is not set."
							)));
				}
			};

			let mut nd:Vec<f64> = Vec::with_capacity(self.units[l].0 + 1);

			for li in layers[hl].iter_mut() {
				li.resize(self.units[l].0 + 1,0f64);
			}


			for j in 1..self.units[l].0 + 1{
				for k in 1..self.units[ll].0 + 1 {
					nd[j] += self.layers[l][j][k-1] * d[k];
				}
				nd[j] = nd[j] * f.derive(s.u[l][j]);
			}

			for i in 0..self.units[hl].0 + 1 {
				optimizer.update(&nd[1..],&self.layers[hl][i],&mut layers[hl][i]);
				let o = s.o[hl][i];
				for j in 0..self.units[l].0{
					layers[hl][i][j-1] = self.layers[hl][i][j-1] - nd[j] * o;
				}
			}

			d = nd;
		}

		self.layers = layers;

		Ok(())
	}

	fn solve(&mut self,input:Vec<f64>) ->
		Result<Vec<f64>,InvalidStateError> {

		self.apply(input,|r,_,_| Ok(r))
	}

	fn learn<O,E>(&mut self,input:Vec<f64>,t:&Vec<f64>,optimizer:O,lossf:&E) -> Result<(),InvalidStateError>
		where O: Optimizer, E: LossFunction {

		let s = self.promise_of_learn(input)?;

		self.latter_part_of_learning(t,s,optimizer,lossf)
	}

	fn promise_of_learn(&mut self,input:Vec<f64>) ->
		Result<SnapShot,InvalidStateError> {

		let mut rnd = rand::XorShiftRng::new_unseeded();
		self.hash = rnd.next_u64();

		self.apply(input,|r,o,u| Ok(SnapShot::new(r,o,u,self.hash)))
	}
}
pub struct SnapShot {
	r:Vec<f64>,
	o:Vec<Vec<f64>>,
	u:Vec<Vec<f64>>,
	hash:u64,
}
impl SnapShot {
	pub fn new(r:Vec<f64>,o:Vec<Vec<f64>>,u:Vec<Vec<f64>>,hash:u64) -> SnapShot {
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
pub trait InputReader<E> {
	fn read_vec(&mut self,usize,usize) -> Result<Vec<Vec<f64>>,E>;
	fn source_exists(&mut self) -> bool;
}
pub trait ModelInputReader<E> {
	fn read_model<'a>(&mut self) -> Result<NNModel<'a>, E>;
}
pub trait Persistence<E> {
	fn save(&mut self,layers:&Vec<Vec<Vec<f64>>>) -> Result<(),E>;
}