#[link(name="opencl")]
extern crate rand;
extern crate ocl;

pub mod function;
pub mod error;
pub mod persistence;

use std::fmt;
use error::*;
use std::error::Error;
use rand::Rng;

use ocl::Buffer;
use ocl::MemFlags;
use ocl::ProQue;
use ocl::SpatialDims;
use ocl::Kernel;

use function::activation::*;
use function::loss::*;
use function::optimizer::*;

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

	pub fn promise_of_learn(&mut self,input:&Vec<f64>) ->
		Result<SnapShot,InvalidStateError> {

		self.model.apply(input,|r,o,u| Ok(SnapShot::new(r,o,u,self.model.hash)))
	}

	pub fn solve(&mut self,input:&Vec<f64>) ->
		Result<Vec<f64>,InvalidStateError> {

		self.model.solve(input)
	}

	pub fn learn(&mut self,input:&Vec<f64>,t:&Vec<f64>) -> Result<(),InvalidStateError>
		where O: Optimizer, E: LossFunction {

		Ok(self.model.learn(input,&t,&mut self.optimizer,&self.lossf)?)
	}

	pub fn latter_part_of_learning(&mut self, t:&Vec<f64>,s:SnapShot) ->
		Result<(),InvalidStateError> {

		Ok(self.model.latter_part_of_learning(t,s,&mut self.optimizer,&self.lossf)?)
	}

	pub fn save<P,ERR>(&mut self,mut persistence:P) -> Result<(),PersistenceError<ERR>>
		where P: Persistence<ERR>, ERR: Error + fmt::Debug, PersistenceError<ERR>: From<ERR> {
		persistence.save(&self.model.layers)?;

		Ok(())
	}
}
pub struct NNUnits {
	input_units:usize,
	defs:Vec<(usize,Box<ActivateF>)>,
}
impl NNUnits {
	pub fn new(input_units:usize, l1:(usize,Box<ActivateF>),l2:(usize,Box<ActivateF>)) -> NNUnits {
		let mut defs:Vec<(usize,Box<ActivateF>)> = Vec::new();
		defs.push(l1);
		defs.push(l2);
		NNUnits {
			input_units:input_units,
			defs:defs
		}
	}

	pub fn add(mut self, units:(usize,Box<ActivateF>)) -> NNUnits {
		self.defs.push(units);
		self
	}
}
pub struct NNModel {
	units:Vec<(usize,Option<Box<ActivateF>>)>,
	layers:Vec<Vec<Vec<f64>>>,
	hash:u64,
	pro_que:Vec<ProQue>,
	kernels:Vec<Kernel>,
}
impl NNModel {
	pub fn load<I,E>(mut reader:I) -> Result<NNModel, E>
		where I: ModelInputReader<E>, E: Error, StartupError<E>: From<E> {
		reader.read_model()
	}

	pub fn new<E>(units:Vec<(usize,Option<Box<ActivateF>>)>,layers:Vec<Vec<Vec<f64>>>) -> Result<NNModel,StartupError<E>>
		where E: Error + fmt::Debug, StartupError<E>: From<E> {

		let mut rnd = rand::XorShiftRng::new_unseeded();

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


		let src = r#"
			__kernel void vec_mul(
				unsigned long units,
				double o,
				__global double *w,
				__global double *u) {
				size_t i = get_global_id(0);

				if (i < units) {
					u[i] = u[i] + o * w[i];
				}
			}
		"#;

		let mut pro_que:Vec<ProQue> = Vec::new();
		let mut kernels:Vec<Kernel> = Vec::new();

		for i in 1..units.len() {
			let global_work_size = SpatialDims::new(
									Some(units[i].0),
									Some(1),Some(1)).unwrap();
			let pq = ProQue::builder()
							.src(src)
							.dims(global_work_size)
							.build().expect("Build ProQue");
			pro_que.push(pq.clone());

			kernels.push(pq.create_kernel("vec_mul")
									.unwrap()
									.arg_scl_named::<usize>("units",None)
									.arg_scl_named::<f64>("o",None)
									.arg_buf_named::<f64,Buffer<f64>>("w",None)
									.arg_buf_named::<f64,Buffer<f64>>("u",None));
		}

		Ok(NNModel {
			units:units,
			layers:layers,
			hash:rnd.next_u64(),
			pro_que:pro_que,
			kernels:kernels,
		})
	}

	pub fn with_bias_and_unit_initializer<I,F,E>(units:NNUnits,
												reader:I,bias:f64,
												mut initializer:F) -> Result<NNModel,StartupError<E>>
		where I: InputReader<E>, F: FnMut() -> f64, E: Error + fmt::Debug, StartupError<E>: From<E> {

		let iunits = units.input_units;
		let mut sunits = units.defs.iter().map(|u| u.0).collect::<Vec<usize>>();
		sunits.insert(0, iunits);
		let sunits = sunits;

		NNModel::with_schema(units,reader,move || {
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

	pub fn with_list_of_bias_and_unit_initializer<I,F,E>(units:NNUnits,
												reader:I,
												init_list:Vec<f64>,
												mut initializer:F) ->
		Result<NNModel,StartupError<E>>
		where I: InputReader<E>, F: FnMut() -> f64, E: Error + fmt::Debug, StartupError<E>: From<E> {

		if init_list.len() != units.defs.len() {
			return Err(StartupError::InvalidConfiguration(
					format!("The number of entries in bias definition is invalid.")));
		}

		let iunits = units.input_units;
		let mut sunits = units.defs.iter().map(|u| u.0).collect::<Vec<usize>>();
		sunits.insert(0, iunits);
		let sunits = sunits;

		NNModel::with_schema(units,reader,move || {
			let mut layers:Vec<Vec<Vec<f64>>> = Vec::with_capacity(sunits.len());

			for i in 0..sunits.len() - 1 {
				let mut layer:Vec<Vec<f64>> = Vec::with_capacity(sunits[i]);

				let mut unit:Vec<f64> = Vec::with_capacity(sunits[i+1]);

				unit.resize(sunits[i+1], init_list[i]);
				layer.push(unit);

				for _ in 1..sunits[i] + 1 {
					let mut unit:Vec<f64> = Vec::with_capacity(sunits[i+1]);
					for _ in 0..sunits[i+1] {
						unit.push(initializer())
					}
					layer.push(unit);
				}

				layers.push(layer);
			}

			layers
		})
	}

	pub fn with_schema<I,F,E>(units:NNUnits,mut reader:I,mut initializer:F) -> Result<NNModel,StartupError<E>>
		where I: InputReader<E>, F: FnMut() -> Vec<Vec<Vec<f64>>>, E: Error + fmt::Debug, StartupError<E>: From<E> {

		let iunits = units.input_units;

		let mut units:Vec<(usize,Option<Box<ActivateF>>)> = units
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
				layers
			},
			false => initializer(),
		};

		NNModel::new(units,layers)
	}

	pub fn apply<F,R>(&self,input:&Vec<f64>,after_callback:F) -> Result<R,InvalidStateError>
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

		let mut ul = Vec::with_capacity(self.units[1].0 + 1);

		ul.resize(self.units[1].0 + 1, 0f64);

		{
			let u_buffer:Buffer<f64> = Buffer::builder()
												.queue(self.pro_que[0].queue().clone())
												.flags(MemFlags::new().alloc_host_ptr().read_write())
												.len(self.units[1].0)
												.build().unwrap();

			for (o,wi) in o[0].iter().zip(&self.layers[0]) {
				let w_buffer:Buffer<f64> = Buffer::builder()
													.queue(self.pro_que[0].queue().clone())
													.flags(MemFlags::new().use_host_ptr().read_only())
													.len(self.units[1].0)
													.host_data(&wi)
													.build().unwrap();

				let kernel = self.kernels[0]
								.clone()
								.set_arg_scl_named("units", self.units[1].0 as usize)
								.unwrap()
								.set_arg_scl_named("o",*o)
								.unwrap()
								.set_arg_buf_named("w",Some(&w_buffer))
								.unwrap()
								.set_arg_buf_named("u",Some(&u_buffer))
								.unwrap()
								.clone();

				unsafe { kernel.enq().unwrap() }
			}

			u_buffer.read(&mut ul[1..]).enq().unwrap();
		}

		u.push(ul);

		o.push(Vec::with_capacity(self.units[1].0 + 1));
		o[1].resize(self.units[1].0 + 1, 0f64);

		let f:&Box<ActivateF> = match self.units[1].1 {
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

			let f:&Box<ActivateF> = match self.units[ll].1 {
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
			{
				let u_buffer:Buffer<f64> = Buffer::builder()
													.queue(self.pro_que[l].queue().clone())
													.flags(MemFlags::new().read_write().alloc_host_ptr())
													.len(self.units[ll].0)
													.build().unwrap();
				for (o,wi) in o[l].iter().zip(&self.layers[l]) {
					let w_buffer:Buffer<f64> = Buffer::builder()
														.queue(self.pro_que[l].queue().clone())
														.flags(MemFlags::new().read_only().use_host_ptr())
														.len(self.units[ll].0)
														.host_data(&wi)
														.build().unwrap();


					let kernel = self.kernels[l]
									.clone()
									.set_arg_scl_named("units", self.units[ll].0 as usize)
									.unwrap()
									.set_arg_scl_named("o",*o)
									.unwrap()
									.set_arg_buf_named("w",Some(&w_buffer))
									.unwrap()
									.set_arg_buf_named("u",Some(&u_buffer))
									.unwrap()
									.clone();

					unsafe { kernel.enq().unwrap() }
				}

				u_buffer.read(&mut ul[1..]).enq().unwrap();
			}

			u.push(ul);

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

	fn latter_part_of_learning<O,E>(&mut self, t:&Vec<f64>,s:SnapShot,optimizer:&mut O,lossf:&E) ->
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
			let mut layer:Vec<Vec<f64>> = Vec::with_capacity(self.units[l].0 + 1);

			layer.resize(self.units[l].0 + 1,Vec::with_capacity(self.units[l+1].0 + 1));
			layers.push(layer);
		}

		let f:&Box<ActivateF> = match self.units[self.units.len()-1].1 {
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

		for j in 0..self.units[hl].0 + 1 {
			let o = s.o[hl][j];
			let mut e:Vec<f64> = Vec::with_capacity(self.units[l].0 + 1);
			e.resize(self.units[l].0,0f64);
			for k in 1..self.units[l].0 + 1 {
				e[k-1] = d[k] * o;
			}
			optimizer.update((hl,j),&e,&self.layers[hl][j],&mut layers[hl][j]);
		}

		for l in (1..self.units.len()-1).rev() {
			let hl = l - 1;
			let ll = l + 1;
			let f:&Box<ActivateF> = match self.units[l].1 {
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
				let mut e:Vec<f64> = Vec::with_capacity(self.units[hl].0 + 1);
				e.resize(self.units[l].0,0f64);
				for j in 1..self.units[l].0 + 1 {
					e[j-1] = nd[j] * o;
				}
				optimizer.update((hl,i),&e,&self.layers[hl][i],&mut layers[hl][i]);
			}
			d = nd;
		}
		self.layers = layers;

		Ok(())
	}

	fn solve(&mut self,input:&Vec<f64>) ->
		Result<Vec<f64>,InvalidStateError> {

		self.apply(input,|r,_,_| Ok(r))
	}

	fn learn<O,E>(&mut self,input:&Vec<f64>,t:&Vec<f64>,optimizer:&mut O,lossf:&E) -> Result<(),InvalidStateError>
		where O: Optimizer, E: LossFunction {

		let s = self.promise_of_learn(input)?;

		self.latter_part_of_learning(t,s,optimizer,lossf)
	}

	fn promise_of_learn(&mut self,input:&Vec<f64>) ->
		Result<SnapShot,InvalidStateError> {

		let mut rnd = rand::XorShiftRng::new_unseeded();
		self.hash = rnd.next_u64();

		self.apply(input,|r,o,u| Ok(SnapShot::new(r,o,u,self.hash)))
	}
}
pub struct SnapShot {
	pub r:Vec<f64>,
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
pub trait InputReader<E> where E: Error + fmt::Debug, StartupError<E>: From<E> {
	fn read_vec(&mut self,usize,usize) -> Result<Vec<Vec<f64>>,E>;
	fn source_exists(&mut self) -> bool;
}
pub trait ModelInputReader<E> where E: Error + fmt::Debug, StartupError<E>: From<E> {
	fn read_model<'a>(&mut self) -> Result<NNModel, E>;
}
pub trait Persistence<E> where E: Error + fmt::Debug, PersistenceError<E>: From<E> {
	fn save(&mut self,layers:&Vec<Vec<Vec<f64>>>) -> Result<(),E>;
}