pub mod function;

use function::activation::ActivateF;
use function::loss::LossFunction;
use function::optimizer::Optimizer;

pub struct NN<'a,O,E> where O: Optimizer + Clone, E: LossFunction {
	model:&'a NNModel<'a>,
	optimizer:O,
	lossf:E,
}

impl<'a,O,E> NN<'a,O,E> where O: Optimizer, E: LossFunction {
	pub fn new(model:&'a NNModel<'a>,optimizer:O,lossf:E) -> NN<'a,O,E> {
		NN {
			model:model,
			optimizer:optimizer,
			lossf:lossf,
		}
	}
}
pub struct NNModel<'a> {
	units:Vec<(usize,&'a ActivateF)>,
	layers:Vec<Vec<Vec<f64>>>,
}
impl<'a> NNModel<'a> {
	pub fn with_schema_and_initializer<I,F>(units:Vec<(usize,&'a ActivateF)>,reader:I,initializer:F) ->
		NNModel<'a> where I: InputReader, F: Fn() -> Vec<Vec<Vec<f64>>> {
		NNModel {
			units:units,
			layers:initializer(),
		}
	}
}
pub trait InputReader {
	fn read_vec() -> Vec<Vec<f64>>;
	fn source_exists() -> bool;
}