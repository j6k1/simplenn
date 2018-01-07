pub mod function;
pub mod error;
pub mod persistence;

use function::activation::*;
use function::loss::*;
use function::optimizer::*;
use error::*;
use std::error::Error;

pub struct NN<'a,O,E,F> where O: Optimizer, E: LossFunction, F: Fn() -> O {
	model:&'a NNModel<'a>,
	optimizer_creator:F,
	lossf:E,
}

impl<'a,O,E,F> NN<'a,O,E,F> where O: Optimizer, E: LossFunction, F: Fn() -> O {
	pub fn new(model:&'a NNModel<'a>,optimizer_creator:F,lossf:E) -> NN<'a,O,E,F> {
		NN {
			model:model,
			optimizer_creator:optimizer_creator,
			lossf:lossf,
		}
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
}
impl<'a> NNModel<'a> {
	pub fn load<I,E>(mut reader:I) -> Result<NNModel<'a>, E>
		where I: ModelInputReader<E>, E: Error, StartupError<E>: From<E> {
		reader.read_model()
	}

	fn new(units:Vec<(usize,Option<&'a ActivateF>)>,layers:Vec<Vec<Vec<f64>>>) -> NNModel<'a> {
		NNModel {
			units:units,
			layers:layers,
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

		Ok(NNModel {
			units:units,
			layers:layers,
		})
	}

	pub fn promise_of_learn<O,E,F>(&self,input:Vec<f64>) ->
		Result<SnapShot,InvalidStateError> {

		if input.len() != self.units.len() {
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

		//for(int k=1, K=units[1].size+1; k < K; k++)
		for (u,(&oi,li)) in u[1].iter_mut().zip(o[0].iter().zip(&self.layers[0])) {
			//for(int j=0, J=units[0].size+1; j < J; j++)
			for ui in li {
				*u = *u + oi * ui;
				//u[1][k] += (o[0][j] * layers[0][j][k-1]);
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

		//for(int j=1, J = units[1].size+1; j < J; j++)
		for (oi,&ui) in o[1].iter_mut().zip(u[1].iter()) {
			//o[1][j] = f.apply(u[1][j]);
			*oi = f.apply(ui,&u[1]);
		}

		//for(int l=1, L=units.length - 1; l < L; l++)
		for l in 0..self.units.len() - 1 {
			//final int ll = l + 1;
			let ll = l + 1;
			let ul:Vec<f64> = Vec::with_capacity(self.units[ll].0 + 1);
			u.push(ul);
			//u[ll] = new double[units[ll].size+1];
			//f = units[l].f;
			let f:&ActivateF = match self.units[l].1 {
				Some(f) => f,
				None => {
					return Err(InvalidStateError::InvalidInput(String::from(
								"Reference to the activation function object is not set."
							)));
				}
			};

			//o[ll] = new double[units[ll].size+1];
			let ol:Vec<f64> = Vec::with_capacity(self.units[ll].0 + 1);
			o.push(ol);

			let layer = &self.layers[l];

			//for(int k=1, K = units[ll].size+1; k < K; k++)
			for (ui,k) in u[ll].iter_mut().zip(0..self.units[ll].0) {
				//for(int j=0, J=units[l].size+1; j < J; j++)
				for (&oi,j) in o[l].iter().zip(0..self.units[l].0 + 1) {
					//u[ll][k] += o[l][j] * layers[l][j][k-1];
					*ui = *ui + oi * layer[j][k];
				}

				//o[ll][k] = f.apply(*u[ll][k]);
				//o[ll][0] = 1.0;
			}

			for (oi,&ui) in o[ll].iter_mut().zip(u[ll].iter()) {
				*oi = f.apply(ui, &u[ll]);
			}
		}

		//double[] r = new double[units[units.length-1].size];
		let mut r:Vec<f64> = Vec::with_capacity(self.units[self.units.len()-1].0);

		//for(int k=1, K = units[units.length-1].size+1, l=units.length-1; k < K; k++)
		for &oi in o[self.units.len()-1].iter() {
			//r[k-1] = o[l][k];
			r.push(oi);
		}

		Ok(SnapShot::new(r,o,u))
	}
}
pub struct SnapShot {
	r:Vec<f64>,
	o:Vec<Vec<f64>>,
	u:Vec<Vec<f64>>,
}
impl SnapShot {
	pub fn new(r:Vec<f64>,o:Vec<Vec<f64>>,u:Vec<Vec<f64>>) -> SnapShot {
		SnapShot {
			r:r,
			o:o,
			u:u,
		}
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