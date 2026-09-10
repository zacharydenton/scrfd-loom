use anyhow::{Result, ensure};
use clap::Parser;
use scrfd_hrx::*;
use std::{path::PathBuf, time::Instant};
/// Run inference or measure warm end-to-end inference, including transfers.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 16)]
    max_batch: usize,
    /// Warm benchmark iterations; zero performs one inference.
    #[arg(long, default_value_t = 0)]
    benchmark: usize,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let input = std::fs::read(&args.input)?;
    let setup = Instant::now();
    let image = image::load_from_memory(&input)?.to_rgb8();
    let (width, height) = image.dimensions();
    let mut bgr = image.into_raw();
    for p in bgr.chunks_exact_mut(3) {
        p.swap(0, 2);
    }
    let mut model = Scrfd::load(
        &args.model,
        Options {
            device: args.device,
            max_batch: args.max_batch,
        },
    )?;
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    if args.benchmark > 0 {
        eprintln!(
            "{}",
            serde_json::to_string(&model.benchmark(
                Image {
                    bgr: &bgr,
                    width: width as usize,
                    height: height as usize
                },
                args.benchmark
            )?)?
        );
    }
    let mut run = || {
        model.detect(
            Image {
                bgr: &bgr,
                width: width as usize,
                height: height as usize,
            },
            DetectionOptions::default(),
        )
    };
    let output = run()?;
    if let Some(path) = args.output {
        std::fs::write(path, serde_json::to_vec_pretty(&output)?)?;
    } else if args.benchmark == 0 {
        println!("{}", serde_json::to_string_pretty(&output)?);
    }
    if args.benchmark > 0 {
        ensure!(args.benchmark >= 10, "use at least 10 benchmark iterations");
        for _ in 0..10 {
            std::hint::black_box(run()?);
        }
        let mut times = Vec::with_capacity(args.benchmark);
        for _ in 0..args.benchmark {
            let start = Instant::now();
            std::hint::black_box(run()?);
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({"scope":"warm end-to-end, including transfers","setup_ms":setup_ms,"samples":times.len(),"median_ms":times[times.len()/2],"p95_ms":times[(times.len()*95).div_ceil(100)-1]})
        );
    }
    Ok(())
}
