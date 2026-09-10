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
}
pub(crate) struct Engine {
    pub stream: Stream,
    pub buffers: Vec<Buffer>,
    pub kernels: Vec<Kernel>,
    pub graphs: HashMap<usize, GraphExec>,
    cache: Kernels,
    launches: HashMap<usize, Vec<Launch>>,
    clears: HashMap<usize, Option<Region>>,
    failed: bool,
    readback: Option<Buffer>,
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
            clears: HashMap::new(),
            failed: false,
            readback: None,
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
    pub fn record(
        &mut self,
        batch: usize,
        launches: &[Launch],
        clear: Option<Region>,
    ) -> Result<()> {
        ensure!(!self.failed, "session is unusable after a GPU failure");
        if self.graphs.contains_key(&batch) {
            return Ok(());
        }
        let mut graph = self.stream.graph()?;
        let mut after = vec![];
        if let Some(r) = clear {
            after.push(graph.fill(&[], self.buffers[r.buffer].try_slice(r.offset, r.bytes)?, 0)?);
        }
        for l in launches {
            let k = &self.kernels[l.kernel];
            let constants = Constants::indices(k, &[l.scalar])?;
            let bindings = l
                .bindings
                .iter()
                .map(|r| self.buffers[r.buffer].try_slice(r.offset, r.bytes))
                .collect::<hrx::Result<Vec<_>>>()?;
            // The model validates shapes and sizes before recording. Each node follows
            // its predecessor, including all reads preceding a reused-buffer write.
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
            after.clear();
            after.push(node);
        }
        self.graphs.insert(batch, graph.finish()?);
        self.launches.insert(batch, launches.to_vec());
        self.clears.insert(batch, clear);
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
                    if mode == 0 {
                        self.replay(batch)?;
                    } else {
                        if let Some(r) = self.clears[&batch] {
                            self.stream
                                .fill(self.buffers[r.buffer].try_slice(r.offset, r.bytes)?, 0)?;
                        }
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
                    self.stream.synchronize()?;
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
    pub fn upload(&mut self, r: Region, bytes: &[u8]) -> Result<()> {
        ensure!(!self.failed, "session is unusable after a GPU failure");
        ensure!(bytes.len() <= r.bytes, "input exceeds allocated region");
        let result = self.stream.upload(
            self.buffers[r.buffer].try_slice(r.offset, bytes.len())?,
            bytes,
        );
        if result.is_err() {
            self.failed = true;
        }
        Ok(result?)
    }
    pub fn replay(&mut self, batch: usize) -> Result<()> {
        ensure!(!self.failed, "session is unusable after a GPU failure");
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
    pub fn reserve_readback(&mut self, bytes: usize) -> Result<()> {
        self.readback = Some(self.stream.allocate_shared(bytes)?);
        Ok(())
    }
    pub fn read_many(&mut self, outputs: &mut [(Region, &mut [u8])]) -> Result<()> {
        ensure!(!self.failed, "session is unusable after a GPU failure");
        let buffer = self
            .readback
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("readback was not reserved"))?;
        let total = outputs.iter().try_fold(0usize, |n, (_, out)| {
            n.checked_add(out.len())
                .ok_or_else(|| anyhow::anyhow!("readback size overflow"))
        })?;
        ensure!(
            total <= buffer.bytes(),
            "readback exceeds reserved capacity"
        );
        for (r, out) in outputs.iter() {
            ensure!(out.len() <= r.bytes, "output exceeds allocated region");
        }
        let result = (|| -> Result<()> {
            let mut offset = 0;
            for (r, out) in outputs.iter() {
                self.stream.copy(
                    buffer.try_slice(offset, out.len())?,
                    self.buffers[r.buffer].try_slice(r.offset, out.len())?,
                )?;
                offset += out.len();
            }
            self.stream.synchronize()?;
            let pointer = buffer.device_ptr()?.cast::<u8>();
            ensure!(!pointer.is_null(), "null host readback pointer");
            offset = 0;
            for (_, out) in outputs.iter_mut() {
                // HRX allocate_shared returns host-local, host-coherent storage;
                // on the supported gfx1151 runtime its device pointer is also its
                // host address (the same path HRX Readback::wait uses). Completion
                // precedes host access; every copied byte was written above.
                // The private readback allocation cannot alias caller outputs.
                unsafe {
                    std::ptr::copy_nonoverlapping(pointer.add(offset), out.as_mut_ptr(), out.len());
                }
                offset += out.len();
            }
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
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
