pub mod function;
pub mod error;
pub mod persistence;

use function::activation::ActivateF;
use function::loss::LossFunction;
use function::optimizer::Optimizer;
use error::*;

pub struct NN<'a,O,E> where O: Optimizer, E: LossFunction {
	model:&'a NNModel<'a>,
	optimizer:O,
	lossf:E,
}

impl<'a,O,E> NN<'a,O,E> where O: Optimizer, E: LossFunction {
	pub fn new<F>(model:&'a NNModel<'a>,f:F,lossf:E) -> NN<'a,O,E> where F: Fn() -> O {
		NN {
			model:model,
			optimizer:f(),
			lossf:lossf,
		}
	}
}
pub struct NNModel<'a> {
	units:Vec<(usize,Option<&'a ActivateF>)>,
	layers:Vec<Vec<Vec<f64>>>,
}
impl<'a> NNModel<'a> {
	pub fn load<I>(mut reader:I) -> Result<NNModel<'a>, StartupError> where I: ModelInputReader {
		reader.read_model()
	}

	pub fn with_bias_and_unit_initializer<I,F>(
												iunits:usize,
												units:Vec<(usize,&'a ActivateF)>,
												reader:I,bias:f64,
												initializer:F) ->
		Result<NNModel<'a>, StartupError> where I: InputReader, F: Fn() -> f64 {

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

	pub fn with_list_of_bias_and_unit_initializer<I,F>(
												iunits:usize,
												units:Vec<(usize,&'a ActivateF)>,
												reader:I,
												init_list:Vec<(f64,F)>) ->
		Result<NNModel<'a>, StartupError> where I: InputReader, F: Fn() -> f64 {

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

	pub fn with_schema<I,F>(iunits:usize,units:Vec<(usize,&'a ActivateF)>,mut reader:I,initializer:F) ->
		Result<NNModel<'a>, StartupError> where I: InputReader, F: Fn() -> Vec<Vec<Vec<f64>>> {

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
}
pub trait InputReader {
	fn read_vec(&mut self,usize,usize) -> Result<Vec<Vec<f64>>,StartupError>;
	fn source_exists(&mut self) -> bool;
}
pub trait ModelInputReader {
	fn read_model<'a>(&mut self) -> Result<NNModel<'a>, StartupError>;
}