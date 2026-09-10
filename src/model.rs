use crate::{
    onnx::{Network, NodeExt, TensorExt},
    plan::*,
};
use anyhow::{Context, Result, ensure};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};
pub(crate) fn load(path: &Path) -> Result<Plan> {
    let net = Network::load(path, 640)?;
    ensure!(
        net.graph.output.len() == 9
            && net
                .graph
                .node
                .iter()
                .filter(|n| n.op_type == "Conv")
                .count()
                == 58,
        "expected det_10g graph"
    );
    let mut heads: HashMap<String, HashMap<usize, usize>> = HashMap::new();
    for (i, n) in net.graph.node.iter().enumerate() {
        if n.op_type == "Conv" {
            let co = net.tensor(&n.input[1])?.shape()?[0];
            if [2, 8, 20].contains(&co) {
                ensure!(
                    heads
                        .entry(n.input[0].clone())
                        .or_default()
                        .insert(co, i)
                        .is_none(),
                    "duplicate head"
                );
            }
        }
    }
    ensure!(
        heads.len() == 3 && heads.values().all(|g| g.len() == 3),
        "expected three score/box/landmark heads"
    );
    let members: HashSet<_> = heads.values().flat_map(|g| g.values().copied()).collect();
    let mut weights = HashMap::new();
    let mut aliases = HashMap::from([(net.graph.input[0].name.clone(), "nhwc_input".into())]);
    let mut ops = vec![Op {
        kind: "convert",
        name: "convert".into(),
        src: net.graph.input[0].name.clone(),
        dst: "nhwc_input".into(),
        h: 640,
        w: 640,
        ho: 640,
        wo: 640,
        bytes: 640 * 640 * 8 * 2,
        ..Default::default()
    }];
    let mut fused = HashSet::new();
    let mut index = 0;
    for (i, node) in net.graph.node.iter().enumerate() {
        match node.op_type.as_str() {
            "Conv" => {
                let name = format!("c{index:02}");
                index += 1;
                if members.contains(&i) {
                    continue;
                }
                let sh = net.tensor(&node.input[1])?.shape()?;
                let (co, ci, taps) = (sh[0], sh[1], sh[2] * sh[3]);
                let mut variant = "plain";
                let mut extra = String::new();
                let mut output = node.out();
                if let Some(add) = net
                    .consumers(node.out())
                    .find(|n| n.op_type == "Add" && n.input[0] == node.out())
                {
                    ensure!(
                        net.consumers(node.out()).count() == 1,
                        "unfused conv consumer"
                    );
                    let other = net
                        .producer(&add.input[1])
                        .context("missing residual producer")?;
                    if other.op_type == "Resize" {
                        variant = "add_resized";
                        extra = other.input[0].clone();
                    } else {
                        variant = "add";
                        extra = add.input[1].clone();
                    }
                    fused.insert(add.out().to_string());
                    output = add.out();
                    aliases.insert(add.out().into(), node.out().into());
                }
                if let Some(relu) = net.consumers(output).find(|n| n.op_type == "Relu") {
                    ensure!(
                        variant != "add_resized" && net.consumers(output).count() == 1,
                        "unsupported ReLU fusion"
                    );
                    variant = if variant == "plain" {
                        "relu"
                    } else {
                        "relu_add"
                    };
                    fused.insert(relu.out().to_string());
                    aliases.insert(relu.out().into(), node.out().into());
                }
                let stride = node.ints("strides", &[1, 1])[0] as usize;
                ensure!(
                    (taps == 9 && variant != "add_resized")
                        || (taps == 1
                            && stride == 1
                            && (variant == "plain" || variant == "add_resized")),
                    "unsupported convolution fusion"
                );
                let cp = if taps == 9 { align(ci, 8) } else { storage(ci) };
                let k = align(taps * cp, 32);
                let n = align(co, 64);
                emit16(
                    &mut weights,
                    name.clone(),
                    &pack(
                        &net.tensor(&node.input[1])?.floats()?,
                        co,
                        ci,
                        taps,
                        cp,
                        n,
                        k,
                    ),
                );
                let mut bias = vec![0.; n];
                bias[..co].copy_from_slice(&net.tensor(&node.input[2])?.floats()?);
                emit32(&mut weights, format!("{name}_b"), &bias);
                let s = net.shape(&node.input[0])?;
                let out = net.shape(node.out())?;
                ops.push(Op {
                    kind: if taps == 9 { "conv" } else { "matmul" },
                    variant,
                    name,
                    src: node.input[0].clone(),
                    dst: node.out().into(),
                    extra,
                    h: s[2],
                    w: s[3],
                    stride,
                    cin_pad: cp,
                    cin_stride: storage(ci),
                    k,
                    n,
                    ho: out[2],
                    wo: out[3],
                    tile: if n == 128 { 128 } else { 64 },
                    bytes: out[2] * out[3] * n * 2,
                    ..Default::default()
                });
            }
            "MaxPool" | "AveragePool" => {
                let s = net.shape(&node.input[0])?;
                let out = net.shape(node.out())?;
                let c = storage(s[1]);
                ops.push(Op {
                    kind: "pool",
                    variant: if node.op_type == "MaxPool" {
                        "max"
                    } else {
                        "mean"
                    },
                    name: node.name.clone(),
                    src: node.input[0].clone(),
                    dst: node.out().into(),
                    h: s[2],
                    w: s[3],
                    ho: out[2],
                    wo: out[3],
                    cin_stride: c,
                    bytes: out[2] * out[3] * c * 2,
                    ..Default::default()
                });
            }
            "Relu" | "Add" => {
                ensure!(fused.contains(node.out()), "unfused {}", node.op_type);
            }
            "Mul" | "Sigmoid" | "Transpose" | "Reshape" | "Resize" | "Shape" | "Gather"
            | "Unsqueeze" | "Slice" | "Concat" => {}
            other => anyhow::bail!("unsupported SCRFD operator {other}"),
        }
    }
    let mut ordered_heads: Vec<_> = heads.into_iter().collect();
    ordered_heads.sort_by_key(|(src, _)| std::cmp::Reverse(net.shapes[src][2]));
    let mut outputs = vec![];
    for (src, g) in ordered_heads {
        let s = net.shape(&src)?;
        let stride = 640 / s[2];
        ensure!(
            [8, 16, 32].contains(&stride) && s[2] == s[3],
            "invalid head resolution"
        );
        let name = format!("h{stride}");
        let ci = s[1];
        let cp = align(ci, 8);
        let k = align(9 * cp, 32);
        let mut w = vec![];
        let mut b = vec![];
        for co in [2, 8, 20] {
            let n = &net.graph.node[g[&co]];
            ensure!(
                net.tensor(&n.input[1])?.shape()? == [co, ci, 3, 3]
                    && n.ints("strides", &[1, 1]) == [1, 1],
                "invalid head convolution"
            );
            let scale = if co == 8 {
                let mul = net
                    .consumers(n.out())
                    .find(|n| n.op_type == "Mul")
                    .context("missing box scale")?;
                net.tensor(&mul.input[1])?.floats()?[0]
            } else {
                1.
            };
            // The old export multiplies float32 before conversion to half.
            w.extend(
                net.tensor(&n.input[1])?
                    .floats()?
                    .into_iter()
                    .map(|v| ((v as f32) * (scale as f32)) as f64),
            );
            b.extend(
                net.tensor(&n.input[2])?
                    .floats()?
                    .into_iter()
                    .map(|v| ((v as f32) * (scale as f32)) as f64),
            );
        }
        emit16(&mut weights, name.clone(), &pack(&w, 30, ci, 9, cp, 64, k));
        b.resize(64, 0.);
        emit32(&mut weights, format!("{name}_b"), &b);
        let dst = format!("head_{name}");
        outputs.push(dst.clone());
        ops.push(Op {
            kind: "conv",
            variant: "plain",
            name,
            src,
            dst,
            h: s[2],
            w: s[3],
            stride: 1,
            cin_pad: cp,
            cin_stride: storage(ci),
            k,
            n: 64,
            ho: s[2],
            wo: s[3],
            tile: 64,
            bytes: s[2] * s[3] * 64 * 2,
            ..Default::default()
        });
    }
    finish(ops, aliases, weights, &outputs)
}
