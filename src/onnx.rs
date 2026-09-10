//! Validation and fixed-shape resolution for the supported ONNX model.
use anyhow::{Context, Result, ensure};
use onnx_protobuf::{
    AttributeProto as Attribute, GraphProto as Graph, Message, ModelProto as Model,
};
pub(crate) use onnx_protobuf::{NodeProto as Node, TensorProto as Tensor};
use std::{collections::HashMap, path::Path};
pub(crate) trait NodeExt {
    fn attr(&self, name: &str) -> Option<&Attribute>;
    fn i(&self, name: &str, default: i64) -> i64;
    fn f(&self, name: &str, default: f64) -> f64;
    fn ints(&self, name: &str, default: &[i64]) -> Vec<i64>;
    fn text(&self, name: &str, default: &str) -> String;
    fn out(&self) -> &str;
}
pub(crate) trait TensorExt {
    fn shape(&self) -> Result<Vec<usize>>;
    fn count(&self) -> Result<usize>;
    fn floats(&self) -> Result<Vec<f64>>;
    fn integers(&self) -> Result<Vec<i64>>;
}
impl NodeExt for Node {
    fn attr(&self, name: &str) -> Option<&Attribute> {
        self.attribute.iter().find(|a| a.name == name)
    }
    fn i(&self, name: &str, default: i64) -> i64 {
        self.attr(name).map_or(default, |a| a.i)
    }
    fn f(&self, name: &str, default: f64) -> f64 {
        self.attr(name).map_or(default, |a| a.f as f64)
    }
    fn ints(&self, name: &str, default: &[i64]) -> Vec<i64> {
        self.attr(name)
            .map_or_else(|| default.to_vec(), |a| a.ints.clone())
    }
    fn text(&self, name: &str, default: &str) -> String {
        self.attr(name).map_or_else(
            || default.to_owned(),
            |a| String::from_utf8_lossy(&a.s).into_owned(),
        )
    }
    fn out(&self) -> &str {
        &self.output[0]
    }
}
impl TensorExt for Tensor {
    fn shape(&self) -> Result<Vec<usize>> {
        self.dims
            .iter()
            .map(|d| {
                ensure!(*d > 0, "invalid tensor dimension");
                Ok(usize::try_from(*d)?)
            })
            .collect()
    }
    fn count(&self) -> Result<usize> {
        self.shape()?.iter().try_fold(1usize, |a, b| {
            a.checked_mul(*b).context("tensor size overflow")
        })
    }
    fn floats(&self) -> Result<Vec<f64>> {
        ensure!(
            self.data_type == 1 && self.data_location.value() == 0,
            "{}: expected embedded float32 tensor",
            self.name
        );
        let n = self.count()?;
        let v: Vec<f64> = if self.raw_data.is_empty() {
            self.float_data.iter().map(|x| *x as f64).collect()
        } else {
            ensure!(
                self.raw_data.len() == n.checked_mul(4).context("tensor byte overflow")?,
                "{}: truncated tensor",
                self.name
            );
            self.raw_data
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()) as f64)
                .collect()
        };
        ensure!(
            v.len() == n && v.iter().all(|x| x.is_finite()),
            "{}: invalid tensor values",
            self.name
        );
        Ok(v)
    }
    fn integers(&self) -> Result<Vec<i64>> {
        ensure!(
            self.data_type == 7 && self.data_location.value() == 0,
            "expected embedded int64 tensor"
        );
        let n = self.count()?;
        let v = if self.raw_data.is_empty() {
            self.int64_data.clone()
        } else {
            ensure!(
                self.raw_data.len() == n.checked_mul(8).context("tensor byte overflow")?,
                "truncated tensor"
            );
            self.raw_data
                .chunks_exact(8)
                .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
                .collect()
        };
        ensure!(v.len() == n, "truncated tensor");
        Ok(v)
    }
}
pub struct Network {
    pub graph: Graph,
    pub weights: HashMap<String, Tensor>,
    pub shapes: HashMap<String, Vec<usize>>,
}
impl Network {
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .with_context(|| format!("missing initializer {name}"))
    }
    pub fn producer(&self, name: &str) -> Option<&Node> {
        self.graph.node.iter().find(|n| n.out() == name)
    }
    pub fn consumers<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> {
        self.graph
            .node
            .iter()
            .filter(move |n| n.op_type != "Shape" && n.input.iter().any(|x| x == name))
    }
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(self
            .shapes
            .get(name)
            .with_context(|| format!("missing shape {name}"))?)
    }
    pub fn load(path: &Path, size: usize) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, size)
    }
    pub(crate) fn from_bytes(bytes: &[u8], size: usize) -> Result<Self> {
        let model = Model::parse_from_bytes(bytes)?;
        let mut graph = model.graph.into_option().context("missing ONNX graph")?;
        ensure!(
            graph.input.len() == 1 && !graph.node.is_empty(),
            "expected one model input"
        );
        let mut weights = HashMap::new();
        for t in std::mem::take(&mut graph.initializer) {
            ensure!(
                weights.insert(t.name.clone(), t).is_none(),
                "duplicate initializer"
            );
        }
        let mut net = Self {
            shapes: HashMap::from([(graph.input[0].name.clone(), vec![1, 3, size, size])]),
            graph,
            weights,
        };
        let mut folded: HashMap<String, Vec<i64>> = HashMap::new();
        for n in &net.graph.node {
            ensure!(
                n.output.len() == 1
                    && !n.input.is_empty()
                    && (n.domain.is_empty() || n.domain == "ai.onnx"),
                "unsupported node {}",
                n.name
            );
            ensure!(
                !net.shapes.contains_key(n.out()) && !folded.contains_key(n.out()),
                "duplicate output"
            );
            let (min, max, allowed): (usize, usize, &[&str]) = match n.op_type.as_str() {
                "Conv" => (
                    3,
                    3,
                    &[
                        "pads",
                        "kernel_shape",
                        "dilations",
                        "strides",
                        "group",
                        "auto_pad",
                    ],
                ),
                "BatchNormalization" => (5, 5, &["epsilon", "momentum", "training_mode"]),
                "PRelu" | "Add" | "Mul" => (2, 2, &[]),
                "Relu" | "Sigmoid" | "Shape" => (1, 1, &[]),
                "Flatten" => (1, 1, &["axis"]),
                "Gemm" => (3, 3, &["alpha", "beta", "transA", "transB"]),
                "MaxPool" | "AveragePool" => (
                    1,
                    1,
                    &[
                        "pads",
                        "ceil_mode",
                        "strides",
                        "kernel_shape",
                        "count_include_pad",
                    ],
                ),
                "Gather" => (2, 2, &["axis"]),
                "Unsqueeze" => (1, 1, &["axes"]),
                "Slice" => (3, 5, &[]),
                "Concat" => (1, 8, &["axis"]),
                "Resize" => (
                    4,
                    4,
                    &[
                        "coordinate_transformation_mode",
                        "cubic_coeff_a",
                        "nearest_mode",
                        "mode",
                    ],
                ),
                "Transpose" => (1, 1, &["perm"]),
                "Reshape" => (2, 2, &[]),
                other => anyhow::bail!("unsupported ONNX operator {other}"),
            };
            ensure!(
                (min..=max).contains(&n.input.len()),
                "{}: invalid input count",
                n.op_type
            );
            let mut names = std::collections::HashSet::new();
            ensure!(
                n.attribute
                    .iter()
                    .all(|a| allowed.contains(&a.name.as_str()) && names.insert(&a.name)),
                "{}: unknown or duplicate attribute",
                n.op_type
            );
            let input = || net.shape(&n.input[0]);
            let ranked_input = |rank: usize| -> Result<&[usize]> {
                let s = input()?;
                ensure!(
                    s.len() == rank,
                    "{}: expected rank-{rank} input, found rank-{}",
                    n.op_type,
                    s.len()
                );
                Ok(s)
            };
            let shape = match n.op_type.as_str() {
                "Conv" => {
                    ensure!(n.input.len() == 3, "convolution must include bias");
                    let w = net.tensor(&n.input[1])?.shape()?;
                    let s = ranked_input(4)?;
                    ensure!(
                        w.len() == 4 && s.len() == 4 && w[1] == s[1] && w[2] == w[3],
                        "invalid convolution shape"
                    );
                    ensure!(
                        w[0] <= 512 && w[1] <= 512 && s[2] <= 640 && s[3] <= 640,
                        "unsupported convolution extent"
                    );
                    ensure!(
                        n.ints("kernel_shape", &[w[2] as i64, w[3] as i64])
                            == [w[2] as i64, w[3] as i64],
                        "convolution kernel shape mismatch"
                    );

                    let strides = n.ints("strides", &[1, 1]);
                    let pads = n.ints("pads", &[0, 0, 0, 0]);
                    ensure!(
                        strides.len() == 2
                            && strides[0] == strides[1]
                            && (1..=2).contains(&strides[0]),
                        "invalid convolution stride"
                    );
                    ensure!(
                        pads.len() == 4
                            && pads.iter().all(|p| *p == pads[0])
                            && ((w[2] == 3 && pads[0] == 1) || (w[2] == 1 && pads[0] == 0)),
                        "invalid convolution padding"
                    );
                    ensure!(
                        n.i("group", 1) == 1
                            && n.ints("dilations", &[1, 1]) == [1, 1]
                            && n.text("auto_pad", "NOTSET") == "NOTSET",
                        "unsupported convolution attributes"
                    );
                    ensure!(
                        net.tensor(&n.input[2])?.shape()? == [w[0]],
                        "invalid convolution bias"
                    );
                    vec![
                        1,
                        w[0],
                        (s[2] + 2 * pads[0] as usize - w[2]) / strides[0] as usize + 1,
                        (s[3] + 2 * pads[0] as usize - w[3]) / strides[0] as usize + 1,
                    ]
                }
                "BatchNormalization" => {
                    ensure!(
                        n.input.len() == 5 && n.i("training_mode", 0) == 0,
                        "unsupported BatchNorm"
                    );
                    let s = input()?.to_vec();
                    ensure!(
                        s.len() >= 2,
                        "BatchNormalization requires a channel dimension"
                    );
                    for t in &n.input[1..] {
                        ensure!(net.tensor(t)?.shape()? == [s[1]], "invalid BatchNorm shape");
                    }
                    s
                }
                "PRelu" => {
                    ensure!(n.input.len() == 2, "invalid PRelu");
                    let s = input()?;
                    ensure!(s.len() >= 2, "PRelu requires a channel dimension");
                    ensure!(net.tensor(&n.input[1])?.count()? == s[1], "invalid slope");
                    s.to_vec()
                }
                "Relu" | "Sigmoid" => input()?.to_vec(),
                "Add" => {
                    ensure!(
                        n.input.len() == 2 && input()? == net.shape(&n.input[1])?,
                        "invalid Add"
                    );
                    input()?.to_vec()
                }
                "Mul" => {
                    ensure!(
                        n.input.len() == 2 && net.tensor(&n.input[1])?.floats()?.len() == 1,
                        "expected scalar Mul"
                    );
                    input()?.to_vec()
                }
                "MaxPool" | "AveragePool" => {
                    ensure!(
                        n.ints("kernel_shape", &[]) == [2, 2]
                            && n.ints("strides", &[1, 1]) == [2, 2]
                            && n.ints("pads", &[0, 0, 0, 0]) == [0, 0, 0, 0]
                            && [0, 1].contains(&n.i("ceil_mode", 0)),
                        "unsupported pool"
                    );
                    let s = ranked_input(4)?;
                    ensure!(
                        s[2].is_multiple_of(2) && s[3].is_multiple_of(2),
                        "pool requires even spatial dimensions"
                    );
                    vec![1, s[1], s[2] / 2, s[3] / 2]
                }
                "Flatten" => {
                    ensure!(n.i("axis", 1) == 1, "unsupported flatten");
                    let s = input()?;
                    ensure!(!s.is_empty(), "Flatten requires a non-scalar input");
                    vec![1, s[1..].iter().product()]
                }
                "Gemm" => {
                    ensure!(
                        n.input.len() == 3
                            && n.i("transA", 0) == 0
                            && n.i("transB", 0) == 1
                            && n.f("alpha", 1.) == 1.
                            && n.f("beta", 1.) == 1.,
                        "unsupported Gemm"
                    );
                    let w = net.tensor(&n.input[1])?.shape()?;
                    let s = ranked_input(2)?;
                    ensure!(
                        w.len() == 2 && s[1] == w[1] && net.tensor(&n.input[2])?.shape()? == [w[0]],
                        "invalid Gemm shape"
                    );
                    vec![1, w[0]]
                }
                "Shape" => {
                    folded.insert(n.out().into(), input()?.iter().map(|x| *x as i64).collect());
                    continue;
                }
                "Gather" => {
                    ensure!(n.i("axis", 0) == 0, "unsupported Gather");
                    let v = folded.get(&n.input[0]).context("unresolved Gather")?;
                    let ix = net.tensor(&n.input[1])?.integers()?;
                    let result = ix
                        .iter()
                        .map(|i| v.get(*i as usize).copied().context("invalid Gather index"))
                        .collect::<Result<_>>()?;
                    folded.insert(n.out().into(), result);
                    continue;
                }
                "Unsqueeze" => {
                    ensure!(n.ints("axes", &[]) == [0], "unsupported Unsqueeze axes");
                    let v = folded
                        .get(&n.input[0])
                        .context("unresolved Unsqueeze")?
                        .clone();
                    folded.insert(n.out().into(), v);
                    continue;
                }
                "Slice" => {
                    if n.input.len() > 3 {
                        ensure!(
                            net.tensor(&n.input[3])?.integers()? == [0],
                            "unsupported Slice axis"
                        );
                    }
                    if n.input.len() > 4 {
                        ensure!(
                            net.tensor(&n.input[4])?.integers()? == [1],
                            "unsupported Slice step"
                        );
                    }
                    let v = folded.get(&n.input[0]).context("unresolved Slice")?;
                    let st = net.tensor(&n.input[1])?.integers()?;
                    let en = net.tensor(&n.input[2])?.integers()?;
                    ensure!(
                        st.len() == 1 && en.len() == 1 && st[0] >= 0 && en[0] >= st[0],
                        "invalid Slice"
                    );
                    let end = (en[0] as usize).min(v.len());
                    let result = v
                        .get(st[0] as usize..end)
                        .context("invalid slice span")?
                        .to_vec();
                    folded.insert(n.out().into(), result);
                    continue;
                }
                "Concat" => {
                    ensure!(n.i("axis", 0) == 0, "unsupported shape Concat");
                    let mut v = vec![];
                    for i in &n.input {
                        v.extend(folded.get(i).context("unresolved Concat")?);
                    }
                    folded.insert(n.out().into(), v);
                    continue;
                }
                "Resize" => {
                    ensure!(
                        n.text("mode", "") == "nearest"
                            && n.text("coordinate_transformation_mode", "") == "asymmetric"
                            && n.text("nearest_mode", "") == "floor"
                            && n.input.len() == 4,
                        "unsupported Resize"
                    );
                    let s = ranked_input(4)?;
                    let out = vec![1, s[1], s[2] * 2, s[3] * 2];
                    ensure!(
                        folded.get(&n.input[3]) == Some(&out.iter().map(|x| *x as i64).collect()),
                        "Resize must double spatial dimensions"
                    );
                    out
                }
                "Transpose" => {
                    ensure!(n.ints("perm", &[]) == [2, 3, 0, 1], "unsupported transpose");
                    let s = ranked_input(4)?;
                    vec![s[2], s[3], s[0], s[1]]
                }
                "Reshape" => {
                    ensure!(n.input.len() == 2, "invalid reshape");
                    let target = net.tensor(&n.input[1])?.integers()?;
                    ensure!(
                        target.len() == 2 && target[0] == -1 && [1, 4, 10].contains(&target[1]),
                        "unsupported head reshape"
                    );
                    let total: usize = input()?.iter().product();
                    ensure!(
                        total.is_multiple_of(target[1] as usize),
                        "invalid reshape size"
                    );
                    vec![total / target[1] as usize, target[1] as usize]
                }
                other => anyhow::bail!("unsupported ONNX operator {other}"),
            };
            net.shapes.insert(n.out().into(), shape);
        }
        for out in &net.graph.output {
            net.shape(&out.name)?;
        }
        Ok(net)
    }
}
