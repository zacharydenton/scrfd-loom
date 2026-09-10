use anyhow::{Result, ensure};
use hrx::loom::{Compiler, CompilerOptions, Kernels, Specialization};
use hrx::{Buffer, Constants, GraphExec, Kernel, Stream};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Region {
    pub buffer: usize,
    pub offset: usize,
    pub bytes: usize,
}
impl Region {
    pub fn whole(buffer: usize, bytes: usize) -> Self {
        Self {
            buffer,
            offset: 0,
            bytes,
        }
    }
}
#[derive(Clone)]
pub(crate) struct Launch {
    pub kernel: usize,
    pub scalar: u32,
    pub grid: [u32; 3],
    pub bindings: Vec<Region>,
    pub output: usize,
}
pub(crate) struct Engine {
    stream: Stream,
    buffers: Vec<Buffer>,
    kernels: Vec<Kernel>,
    pub graphs: HashMap<usize, GraphExec>,
    cache: Kernels,
    launches: HashMap<usize, Vec<Launch>>,
    failed: bool,
    shared: std::collections::HashSet<usize>,
    pending: bool,
}
impl Engine {
    pub fn new(index: i32) -> Result<Self> {
        let device = hrx::Device::open(index)?;
        ensure!(
            device.target().as_str() == "gfx1151",
            "these kernels are validated for gfx1151, found {}",
            device.target().as_str()
        );
        let compiler = Compiler::shared(None, CompilerOptions::default())?;
        Ok(Self {
            stream: device.stream()?,
            buffers: vec![],
            kernels: vec![],
            graphs: HashMap::new(),
            cache: Kernels::new(compiler),
            launches: HashMap::new(),
            failed: false,
            shared: Default::default(),
            pending: false,
        })
    }
    pub fn allocate(&mut self, bytes: usize) -> Result<Region> {
        let i = self.buffers.len();
        self.buffers.push(self.stream.allocate(bytes)?);
        Ok(Region::whole(i, bytes))
    }
    pub fn weight(&mut self, bytes: &[u8]) -> Result<Region> {
        let r = self.allocate(bytes.len())?;
        self.stream
            .upload(self.buffers[r.buffer].binding(), bytes)?;
        Ok(r)
    }
    pub fn compile(&mut self, specs: &[(&str, Specialization)]) -> Result<()> {
        let pending: Vec<_> = specs
            .iter()
            .map(|(src, spec)| self.cache.request(src, spec))
            .collect::<hrx::Result<_>>()?;
        // Only embedded production kernels enter this cache.
        unsafe {
            self.cache.build(&self.stream)?;
        }
        for p in pending {
            self.kernels
                .push(unsafe { p.resolve(&self.stream)?.clone() });
        }
        self.stream.synchronize()?;
        Ok(())
    }
    pub fn record(&mut self, batch: usize, launches: &[Launch]) -> Result<()> {
        ensure!(!self.failed, "session is unusable after a GPU failure");
        if self.graphs.contains_key(&batch) {
            return Ok(());
        }
        let mut graph = self.stream.graph()?;
        let mut writers = vec![None; self.buffers.len()];
        let mut readers = vec![Vec::new(); self.buffers.len()];
        for l in launches {
            let k = &self.kernels[l.kernel];
            let constants = Constants::indices(k, &[l.scalar])?;
            let bindings = l
                .bindings
                .iter()
                .map(|r| self.buffers[r.buffer].try_slice(r.offset, r.bytes))
                .collect::<hrx::Result<Vec<_>>>()?;
            // Depend on prior writers and on every reader of a reused output.
            // Independent branches can overlap without racing the activation pool.
            let mut after = readers[l.output].clone();
            for r in &l.bindings {
                if let Some(writer) = writers[r.buffer]
                    && !after.contains(&writer)
                {
                    after.push(writer);
                }
            }
            let node = unsafe {
                graph.dispatch(
                    &after,
                    k,
                    l.grid,
                    k.info().workgroup_size,
                    &constants,
                    &bindings,
                )?
            };
            readers[l.output].clear();
            writers[l.output] = Some(node);
            for r in &l.bindings {
                if r.buffer != l.output && !readers[r.buffer].contains(&node) {
                    readers[r.buffer].push(node);
                }
            }
        }
        self.graphs.insert(batch, graph.finish()?);
        self.launches.insert(batch, launches.to_vec());
        Ok(())
    }
    pub fn benchmark(&mut self, batch: usize, samples: usize) -> Result<ForwardTimings> {
        ensure!(samples >= 10, "use at least ten timing samples");
        ensure!(!self.failed, "session is unusable after a GPU failure");
        let mut times = [Vec::with_capacity(samples), Vec::with_capacity(samples)];
        for i in 0..samples + 10 {
            for mode in [i % 2, 1 - i % 2] {
                let start = std::time::Instant::now();
                let result = (|| -> Result<()> {
                    self.pending = true;
                    if mode == 0 {
                        self.replay(batch)?;
                    } else {
                        for l in &self.launches[&batch] {
                            let k = &self.kernels[l.kernel];
                            let constants = Constants::indices(k, &[l.scalar])?;
                            let bindings = l
                                .bindings
                                .iter()
                                .map(|r| self.buffers[r.buffer].try_slice(r.offset, r.bytes))
                                .collect::<hrx::Result<Vec<_>>>()?;
                            // The same checked launch plan and ordered stream as graph recording.
                            unsafe {
                                self.stream.dispatch(
                                    k,
                                    l.grid,
                                    k.info().workgroup_size,
                                    &constants,
                                    &bindings,
                                )?;
                            }
                        }
                    }
                    self.wait()?;
                    Ok(())
                })();
                if result.is_err() {
                    self.failed = true;
                }
                result?;
                if i >= 10 {
                    times[mode].push(start.elapsed().as_secs_f64() * 1000.);
                }
            }
        }
        let summarize = |mut t: Vec<f64>| {
            t.sort_by(f64::total_cmp);
            Distribution {
                median_ms: t[t.len() / 2],
                p95_ms: t[(t.len() * 95).div_ceil(100) - 1],
            }
        };
        let [graph, direct] = times;
        Ok(ForwardTimings {
            samples,
            graph: summarize(graph),
            direct: summarize(direct),
        })
    }
    pub fn allocate_io(&mut self, bytes: usize) -> Result<Region> {
        let i = self.buffers.len();
        self.buffers.push(self.stream.allocate_shared(bytes)?);
        self.shared.insert(i);
        Ok(Region::whole(i, bytes))
    }
    pub fn upload(&mut self, r: Region, bytes: &[u8]) -> Result<()> {
        ensure!(!self.failed, "session is unusable after a GPU failure");
        ensure!(self.shared.contains(&r.buffer), "input must be shared");
        self.buffers[r.buffer].try_slice(r.offset, bytes.len())?;
        ensure!(bytes.len() <= r.bytes, "input exceeds allocated region");
        self.wait()?;
        let pointer = self.buffers[r.buffer].device_ptr()?.cast::<u8>();
        ensure!(!pointer.is_null(), "null host input pointer");
        // The private allocation is host-local and coherent on gfx1151. No GPU
        // work is in flight and caller memory cannot alias it. Submission follows
        // this host write, so the graph sees the complete input.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.add(r.offset), bytes.len());
        }
        Ok(())
    }
    fn wait(&mut self) -> Result<()> {
        ensure!(!self.failed, "session is unusable after a GPU failure");
        if self.pending {
            if let Err(e) = self.stream.synchronize() {
                self.failed = true;
                return Err(e.into());
            }
            self.pending = false;
        }
        Ok(())
    }
    pub fn replay(&mut self, batch: usize) -> Result<()> {
        ensure!(!self.failed, "session is unusable after a GPU failure");
        self.pending = true;
        let result = self.stream.launch(
            self.graphs
                .get_mut(&batch)
                .ok_or_else(|| anyhow::anyhow!("batch has not been prepared"))?,
        );
        if result.is_err() {
            self.failed = true;
        }
        Ok(result?)
    }
    pub fn read_many(&mut self, outputs: &mut [(Region, &mut [u8])]) -> Result<()> {
        self.wait()?;
        for (r, out) in outputs {
            ensure!(self.shared.contains(&r.buffer), "output must be shared");
            self.buffers[r.buffer].try_slice(r.offset, out.len())?;
            ensure!(out.len() <= r.bytes, "output exceeds allocated region");
            let pointer = self.buffers[r.buffer].device_ptr()?.cast::<u8>();
            ensure!(!pointer.is_null(), "null host output pointer");
            // Every requested byte is a model output written by the completed
            // graph. Host-local coherent storage is CPU-addressable on gfx1151;
            // private allocations cannot alias the caller's output slices.
            unsafe {
                std::ptr::copy_nonoverlapping(pointer.add(r.offset), out.as_mut_ptr(), out.len());
            }
        }
        Ok(())
    }
}

/// Synchronized host timings for a resident forward pass, excluding transfers.
#[derive(Debug, serde::Serialize)]
pub struct ForwardTimings {
    pub samples: usize,
    pub graph: Distribution,
    pub direct: Distribution,
}
/// Milliseconds measured by the host around submission and completion.
#[derive(Debug, serde::Serialize)]
pub struct Distribution {
    pub median_ms: f64,
    pub p95_ms: f64,
}
