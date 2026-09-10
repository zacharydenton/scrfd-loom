use super::*;
#[test]
fn preprocessing_and_detection_edges() {
    assert!(
        detection::letterbox(Image {
            bgr: &[],
            width: 0,
            height: 0
        })
        .is_err()
    );
    let img = vec![127u8; 640 * 640 * 3];
    let (out, scale) = detection::letterbox(Image {
        bgr: &img,
        width: 640,
        height: 640,
    })
    .unwrap();
    assert_eq!(out, img);
    assert_eq!(scale, 1.);
    let d = Detection {
        bbox: [0., 0., 10., 10.],
        score: 0.9,
        landmarks: [[0.; 2]; 5],
    };
    let mut lower = d.clone();
    lower.score = 0.8;
    assert_eq!(
        detection::suppress(vec![lower, d], [640, 640], DetectionOptions::default()).len(),
        1
    );
    assert!(
        DetectionOptions {
            threshold: f32::NAN,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}
#[test]
#[ignore = "requires pretrained weights; CPU model import"]
fn importer_liveness() -> Result<()> {
    let p = model::load(std::path::Path::new(&model_path()?))?;
    assert_eq!(p.ops.len(), 57);
    assert_eq!(p.buffers.len(), 5);
    for l in p.ops {
        assert_ne!(l.src_buf, l.dst_buf);
        assert_ne!(l.extra_buf, l.dst_buf);
    }
    Ok(())
}
#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn native_reference_and_replay() -> Result<()> {
    let path = model_path()?;
    let mut model = Scrfd::load(
        &path,
        Options {
            device: 0,
            max_batch: 2,
        },
    )?;
    let canvas: Vec<u8> = (0..640 * 640 * 3)
        .map(|i| ((i * 7 + i / 101) % 256) as u8)
        .collect();
    model.detect_letterboxed(&canvas, &[1.], &[[640, 640]], Default::default())?;
    let first = model.heads.clone();
    let mut blob = vec![0.; 3 * 640 * 640];
    for c in 0..3 {
        for y in 0..640 {
            for x in 0..640 {
                blob[(c * 640 + y) * 640 + x] =
                    (canvas[(y * 640 + x) * 3 + 2 - c] as f64 - 127.5) / 128.;
            }
        }
    }
    let expected = reference::forward(std::path::Path::new(&path), blob)?;
    for (level, size) in [80, 40, 20].into_iter().enumerate() {
        for (kind, width) in [1, 4, 10].into_iter().enumerate() {
            let data: Vec<f32> = expected[kind * 3 + level]
                .iter()
                .map(|v| *v as f32)
                .collect();
            let want = data.as_slice();
            let mut got = vec![];
            for p in 0..size * size {
                let row = &model.heads[level][p * 64..][..64];
                for a in 0..2 {
                    for c in 0..width {
                        got.push(match kind {
                            0 => 1. / (1. + (-row[a].to_f32()).exp()),
                            1 => row[2 + 4 * a + c].to_f32(),
                            _ => row[10 + 10 * a + c].to_f32(),
                        });
                    }
                }
            }
            let max_error = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_error < if kind == 0 { 0.02 } else { 0.15 },
                "head {kind}/{size} max error {max_error}"
            );
            let dot = got
                .iter()
                .zip(want)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>();
            let norm = |a: &[f32]| a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
            let cos = dot / (norm(&got) * norm(want)).sqrt();
            assert!(cos > 0.9999, "head {kind}/{size} cosine {cos}");
        }
    }
    let other = vec![33; canvas.len()];
    let mut pair = canvas.clone();
    pair.extend(&other);
    model.detect_letterboxed(&pair, &[1., 1.], &[[640, 640]; 2], Default::default())?;
    for (i, size) in [80, 40, 20].into_iter().enumerate() {
        assert_eq!(
            &first[i][..size * size * 64],
            &model.heads[i][..size * size * 64]
        );
    }
    model.detect_letterboxed(&canvas, &[1.], &[[640, 640]], Default::default())?;
    for (i, size) in [80, 40, 20].into_iter().enumerate() {
        assert_eq!(
            &first[i][..size * size * 64],
            &model.heads[i][..size * size * 64]
        );
    }
    Ok(())
}
#[test]
fn malformed_onnx_returns_errors() {
    use onnx_protobuf::{GraphProto, Message, ModelProto, NodeProto, ValueInfoProto};
    assert!(onnx::Network::from_bytes(&[], 112).is_err());
    assert!(onnx::Network::from_bytes(&[0x3a, 0xff], 112).is_err());
    let node = NodeProto {
        op_type: "Conv".into(),
        input: vec!["x".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let graph = GraphProto {
        node: vec![node],
        input: vec![ValueInfoProto {
            name: "x".into(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let model = ModelProto {
        graph: Some(graph).into(),
        ..Default::default()
    };
    assert!(onnx::Network::from_bytes(&model.write_to_bytes().unwrap(), 112).is_err());
}
#[test]
#[ignore = "requires pretrained weights and gfx1151"]
fn insightface_fixture() -> Result<()> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/t1_insightface.json"))?;
    let expected: Vec<[f32; 5]> = serde_json::from_value(fixture["det"].clone())?;
    let kps: Vec<[[f32; 2]; 5]> = serde_json::from_value(fixture["kps"].clone())?;
    let image = image::load_from_memory(include_bytes!("../tests/fixtures/t1.png"))?.to_rgb8();
    let (w, h) = image.dimensions();
    let mut bgr = image.into_raw();
    for p in bgr.chunks_exact_mut(3) {
        p.swap(0, 2);
    }
    let mut model = Scrfd::load(
        model_path()?,
        Options {
            device: 0,
            max_batch: 2,
        },
    )?;
    let image = Image {
        bgr: &bgr,
        width: w as usize,
        height: h as usize,
    };
    let got = model.detect(image, Default::default())?;
    assert_eq!(got.len(), expected.len());
    for ((a, b), k) in got.iter().zip(&expected).zip(&kps) {
        let aa = a.bbox;
        let inter = (aa[2].min(b[2]) - aa[0].max(b[0])).max(0.)
            * (aa[3].min(b[3]) - aa[1].max(b[1])).max(0.);
        let union = (aa[2] - aa[0]) * (aa[3] - aa[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
        assert!(inter / union > 0.99);
        assert!((a.score - b[4]).abs() < 0.01);
        for (x, y) in a.landmarks.iter().flatten().zip(k.iter().flatten()) {
            assert!((x - y).abs() < 1.);
        }
    }
    let batched = model.detect_batch(&[image, image], Default::default())?;
    assert_eq!(
        serde_json::to_value(&got)?,
        serde_json::to_value(&batched[0])?
    );
    assert_eq!(
        serde_json::to_value(&got)?,
        serde_json::to_value(&batched[1])?
    );
    Ok(())
}

fn model_path() -> Result<std::path::PathBuf> {
    match std::env::var_os("SCRFD_MODEL") {
        Some(path) => Ok(path.into()),
        None => hub::weights(false),
    }
}
