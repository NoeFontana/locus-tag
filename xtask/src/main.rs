//! Repository automation for Locus.
//!
//! `cargo xtask sota <command>` drives reproducible comparative benchmarking
//! against pinned, unpatched reference detectors (aruco_nano, OpenCV `aruco`,
//! AprilTag 3). See `xtask/README.md` for the protocol and its rationale.
//!
//! This crate only orchestrates processes (git, cmake, a C++ compiler, uv). It is
//! deliberately dependency-free so it never adds crates to the workspace graph.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// OpenCV release the references are built against (matches the Liu4K report).
const OPENCV_TAG: &str = "4.10.0";
const OPENCV_REPO: &str = "https://github.com/opencv/opencv.git";
/// aruco_nano is header-only; pinned to the commit the published comparison used.
const ARUCO_NANO_COMMIT: &str = "961b18b747d64cc3692c570dff35334c098b9e62";
const ARUCO_NANO_REPO: &str = "https://github.com/rmsalinas/aruco_nano.git";
/// Minimal OpenCV module set: `objdetect` (aruco) and what it links against.
const OPENCV_MODULES: &str = "core,imgproc,imgcodecs,calib3d,features2d,flann,objdetect";
const OPENCV_LIBS: &[&str] = &[
    "opencv_objdetect",
    "opencv_calib3d",
    "opencv_features2d",
    "opencv_flann",
    "opencv_imgcodecs",
    "opencv_imgproc",
    "opencv_core",
];

/// A benchmark and how each detector family must be configured for it. Defined by a
/// `[sota.<name>]` table in `xtask/datasets.toml` and resolved by
/// `tools/bench/sota/spec.py` (this crate stays dependency-free, so it does not parse TOML).
struct Dataset {
    /// Benchmark name (`[sota.<name>]`).
    name: String,
    /// Run directory name: `<name>` or `<name>@<tag>` (e.g. a strided serial timing run
    /// kept apart from the full accuracy run).
    run: String,
    locus_family: String,
    opencv_dict: String,
    apriltag_family: Option<String>,
    /// Black-border width in bits (Kalibr AprilGrid prints 2).
    border_bits: u32,
    /// Detectors that cannot decode this dataset *as published*, with the reason.
    /// They are skipped unless `--include-unsupported` is passed; references are
    /// never patched to support a dataset.
    unsupported: Vec<(String, String)>,
}

/// Every published operating point of each reference, so the scoreboard compares
/// Locus against the best one per metric.
const DEFAULT_DETECTORS: &[&str] = &[
    "locus:standard",
    "locus:grid",
    "locus:high_accuracy",
    "aruco_nano",
    "opencv",
    "opencv-subpix",
    "opencv-apriltag",
    "apriltag3",
];

fn main() {
    if let Err(e) = run(&env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("xtask: error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<()> {
    let root = workspace_root()?;
    env::set_current_dir(&root)?;
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["data", rest @ ..] => data(rest),
        ["sota", "setup", ..] => setup(&root),
        ["sota", "list"] => spec_cmd(&["list"]),
        ["sota", "fetch", ds] => fetch(&dataset(ds)?),
        ["sota", "run", ds, rest @ ..] => run_detectors(&root, &dataset(ds)?, &Opts::parse(rest)?),
        ["sota", "score", ds] => score(&root, &dataset(ds)?),
        ["sota", "report", ds] => report(&root, &dataset(ds)?),
        ["sota", "scoreboard", rest @ ..] => scoreboard(&root, rest),
        ["sota", "all", ds, rest @ ..] => {
            let d = dataset(ds)?;
            setup(&root)?;
            fetch(&d)?;
            run_detectors(&root, &d, &Opts::parse(rest)?)?;
            score(&root, &d)?;
            report(&root, &d)
        },
        _ => {
            eprintln!("{USAGE}");
            Err("unknown command".into())
        },
    }
}

const USAGE: &str = "\
usage: cargo xtask data <command>          (datasets; manifest: xtask/datasets.toml)

  list                       every dataset: presence, kind, destination, license
  fetch  <name...|--all>     fetch pinned datasets (idempotent; --force, --dest, --subsets for hub)
  verify [name...]           check presence and that local copies match the manifest pins

usage: cargo xtask sota <command>          (comparative benchmarking)

  setup                      build pinned OpenCV + aruco_nano and the C++ reference runner
  list                       benchmark names (`[sota.*]` tables of xtask/datasets.toml)
  fetch  <dataset>           fetch the data a benchmark needs (via cargo xtask data)
  run    <dataset> [opts]    run detectors, one JSONL per detector
  score  <dataset>           score all runs of a dataset
  report <dataset>           write report.md (scores, win table, verified run metadata)
  scoreboard [--champion L]  win table of Locus run L (default locus_standard) against the
                             best reference operating point, over every scored dataset
  all    <dataset> [opts]    setup + fetch + run + score + report

  <dataset> may be <name>@<tag>: same benchmark, separate run directory
  (e.g. `run liu4k@t1 --jobs 1 --stride 4` for serial timing next to the accuracy run).

run options:
  --detectors a,b,...        default: locus:standard,locus:grid,locus:high_accuracy,aruco_nano,
                             opencv,opencv-subpix,opencv-apriltag,apriltag3
                             locus:<profile> | locus:<label>=<profile>+<overrides.json> |
                             aruco_nano | opencv | opencv-subpix | opencv-apriltag | apriltag3
  --threads N                worker threads for every detector (default 1)
  --reps N                   timed detect() calls per image, best kept (default 2)
  --stride N                 use every N-th image (default 1)
  --jobs N                   run N detectors concurrently: accuracy only, latency invalid (default 1)
  --include-unsupported      also run references marked unsupported for the dataset";

struct Opts {
    detectors: Vec<String>,
    threads: u32,
    reps: u32,
    stride: usize,
    jobs: usize,
    include_unsupported: bool,
}

impl Opts {
    fn parse(args: &[&str]) -> Result<Self> {
        let mut o = Self {
            detectors: DEFAULT_DETECTORS.iter().map(ToString::to_string).collect(),
            threads: 1,
            reps: 2,
            stride: 1,
            jobs: 1,
            include_unsupported: false,
        };
        let mut it = args.iter();
        while let Some(&a) = it.next() {
            let mut val = || {
                it.next()
                    .copied()
                    .ok_or_else(|| format!("{a} needs a value"))
            };
            match a {
                "--detectors" => o.detectors = val()?.split(',').map(str::to_string).collect(),
                "--threads" => o.threads = val()?.parse()?,
                "--reps" => o.reps = val()?.parse()?,
                "--stride" => o.stride = val()?.parse::<usize>()?.max(1),
                "--jobs" => o.jobs = val()?.parse::<usize>()?.max(1),
                "--include-unsupported" => o.include_unsupported = true,
                _ => return Err(format!("unknown option {a}").into()),
            }
        }
        Ok(o)
    }
}

fn dataset(run: &str) -> Result<Dataset> {
    let name = run.split_once('@').map_or(run, |(name, _)| name);
    let out = python()
        .args(["-m", "tools.bench.sota.spec", "show", name])
        .output()?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr)
            .trim()
            .to_string()
            .into());
    }
    let mut d = Dataset {
        name: name.to_string(),
        run: run.to_string(),
        locus_family: String::new(),
        opencv_dict: String::new(),
        apriltag_family: None,
        border_bits: 1,
        unsupported: Vec::new(),
    };
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.to_string();
        match key {
            "family" => d.locus_family = value,
            "opencv_dict" => d.opencv_dict = value,
            "apriltag_family" => d.apriltag_family = Some(value).filter(|v| !v.is_empty()),
            "border_bits" => d.border_bits = value.parse()?,
            _ => {
                if let Some(detector) = key.strip_prefix("unsupported.") {
                    d.unsupported.push((detector.to_string(), value));
                }
            },
        }
    }
    if d.locus_family.is_empty() || d.opencv_dict.is_empty() {
        return Err(format!("incomplete spec for {name}").into());
    }
    Ok(d)
}

fn workspace_root() -> Result<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Ok(manifest
        .parent()
        .ok_or("xtask has no parent dir")?
        .to_path_buf())
}

fn sota_dir(root: &Path) -> PathBuf {
    root.join("target/sota")
}

fn sh(cmd: &mut Command) -> Result<()> {
    eprintln!("+ {cmd:?}");
    let status = cmd.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command failed ({status}): {cmd:?}").into())
    }
}

fn capture(cmd: &mut Command) -> String {
    cmd.output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// The Python interpreter command used for runners/scorers. Override with
/// `XTASK_PYTHON` (whitespace-separated), e.g. `XTASK_PYTHON=".venv/bin/python"`.
fn python() -> Command {
    let spec = env::var("XTASK_PYTHON").unwrap_or_else(|_| "uv run --no-sync python".to_string());
    let mut parts = spec.split_whitespace();
    let mut cmd = Command::new(parts.next().unwrap_or("python3"));
    cmd.args(parts);
    cmd
}

// ── setup ────────────────────────────────────────────────────────────────────

fn setup(root: &Path) -> Result<()> {
    let base = sota_dir(root);
    let src = base.join("src");
    let prefix = base.join("opencv");
    let bin = base.join("bin");
    fs::create_dir_all(&src)?;
    fs::create_dir_all(&bin)?;

    let ocv_src = src.join(format!("opencv-{OPENCV_TAG}"));
    if !ocv_src.join("CMakeLists.txt").exists() {
        sh(Command::new("git")
            .args([
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--branch",
                OPENCV_TAG,
                OPENCV_REPO,
            ])
            .arg(&ocv_src))?;
    }
    if !prefix.join("lib").exists() {
        let build = ocv_src.join("build");
        fs::create_dir_all(&build)?;
        sh(Command::new("cmake")
            .current_dir(&build)
            .arg("..")
            .arg(format!("-DCMAKE_INSTALL_PREFIX={}", prefix.display()))
            .args([
                "-DCMAKE_BUILD_TYPE=Release",
                &format!("-DBUILD_LIST={OPENCV_MODULES}"),
                "-DBUILD_TESTS=OFF",
                "-DBUILD_PERF_TESTS=OFF",
                "-DBUILD_EXAMPLES=OFF",
                "-DBUILD_opencv_apps=OFF",
                "-DBUILD_opencv_python3=OFF",
                "-DWITH_TBB=OFF",
                "-DWITH_OPENCL=OFF",
                "-DWITH_FFMPEG=OFF",
                "-DWITH_GTK=OFF",
                "-DWITH_PROTOBUF=OFF",
            ]))?;
        let jobs =
            std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(1).max(1));
        sh(Command::new("cmake").current_dir(&build).args([
            "--build",
            ".",
            "--target",
            "install",
            "-j",
            &jobs.to_string(),
        ]))?;
    }

    let nano = src.join("aruco_nano");
    if !nano.join("aruco_nano.h").exists() {
        sh(Command::new("git")
            .args(["clone", "--quiet", ARUCO_NANO_REPO])
            .arg(&nano))?;
    }
    sh(Command::new("git")
        .current_dir(&nano)
        .args(["checkout", "--quiet", ARUCO_NANO_COMMIT]))?;

    let mut cxx = Command::new(env::var("CXX").unwrap_or_else(|_| "c++".to_string()));
    cxx.args(["-O3", "-std=c++17", "-o"])
        .arg(bin.join("refrun"))
        .arg(root.join("xtask/cpp/refrun.cpp"))
        .arg(format!("-I{}", nano.display()))
        .arg(format!("-I{}", prefix.join("include/opencv4").display()))
        .arg(format!("-L{}", prefix.join("lib").display()))
        .arg(format!("-Wl,-rpath,{}", prefix.join("lib").display()));
    for lib in OPENCV_LIBS {
        cxx.arg(format!("-l{lib}"));
    }
    sh(&mut cxx)?;
    eprintln!("reference runner ready: {}", bin.join("refrun").display());
    Ok(())
}

// ── fetch ────────────────────────────────────────────────────────────────────

/// `cargo xtask data ...`: dataset provisioning is implemented once, in
/// `tools/bench/dataset_registry.py`, driven by `xtask/datasets.toml`; this only forwards.
fn data(args: &[&str]) -> Result<()> {
    if args.is_empty() {
        eprintln!("{USAGE}");
        return Err("missing data command".into());
    }
    sh(python()
        .args(["-m", "tools.bench.dataset_registry"])
        .args(args))
}

fn spec_cmd(args: &[&str]) -> Result<()> {
    sh(python().args(["-m", "tools.bench.sota.spec"]).args(args))
}

fn fetch(d: &Dataset) -> Result<()> {
    spec_cmd(&["fetch", &d.name])
}

// ── run ──────────────────────────────────────────────────────────────────────

fn runs_dir(root: &Path, d: &Dataset) -> PathBuf {
    sota_dir(root).join("runs").join(&d.run)
}

fn image_list(d: &Dataset, stride: usize, out: &Path) -> Result<usize> {
    let res = python()
        .args([
            "-m",
            "tools.bench.sota.spec",
            "images",
            &d.name,
            &stride.to_string(),
        ])
        .arg(out)
        .output()?;
    if !res.status.success() {
        return Err(String::from_utf8_lossy(&res.stderr)
            .trim()
            .to_string()
            .into());
    }
    Ok(String::from_utf8_lossy(&res.stdout).trim().parse()?)
}

/// Builds the command for one detector spec, or `None` when it is unsupported
/// for this dataset and not explicitly requested.
fn detector_cmd(
    root: &Path,
    d: &Dataset,
    spec: &str,
    list: &Path,
    out_dir: &Path,
    o: &Opts,
) -> Result<Option<(String, Command)>> {
    let threads = o.threads.to_string();
    let reps = o.reps.to_string();
    let family = spec.split(':').next().unwrap_or(spec);
    if let Some((_, why)) = d.unsupported.iter().find(|(n, _)| *n == family)
        && !o.include_unsupported
    {
        eprintln!(
            "skipping {spec} on {}: unsupported as published — {why}",
            d.name
        );
        return Ok(None);
    }
    let refrun = sota_dir(root).join("bin/refrun");
    let (label, mut cmd) = if let Some(rest) = spec.strip_prefix("locus:") {
        let (label, profile, overrides) = match rest.split_once('=') {
            Some((label, def)) => match def.split_once('+') {
                Some((profile, file)) => (
                    label.to_string(),
                    profile.to_string(),
                    fs::read_to_string(file)?,
                ),
                None => (label.to_string(), def.to_string(), "{}".to_string()),
            },
            None => (rest.to_string(), rest.to_string(), "{}".to_string()),
        };
        let label = format!("locus_{label}");
        let mut c = python();
        c.args([
            "-m",
            "tools.bench.sota.run",
            "locus",
            &d.locus_family,
            &profile,
            &overrides,
        ])
        .arg(list)
        .arg(out_dir.join(format!("{label}.jsonl")))
        .arg(&reps)
        .env("RAYON_NUM_THREADS", &threads);
        (label, c)
    } else if matches!(
        spec,
        "aruco_nano" | "opencv" | "opencv-subpix" | "opencv-apriltag"
    ) {
        if !refrun.exists() {
            return Err("reference runner missing: run `cargo xtask sota setup`".into());
        }
        let mode = if spec == "aruco_nano" { "nano" } else { spec };
        let label = spec.replace('-', "_");
        let mut c = Command::new(&refrun);
        c.args([mode, &d.opencv_dict, &threads])
            .arg(list)
            .arg(out_dir.join(format!("{label}.jsonl")))
            .args([&reps, &d.border_bits.to_string()]);
        (label, c)
    } else if spec == "apriltag3" {
        let Some(fam) = &d.apriltag_family else {
            eprintln!(
                "skipping apriltag3 on {}: family {} not available in AprilTag 3",
                d.name, d.locus_family
            );
            return Ok(None);
        };
        let mut c = python();
        c.args(["-m", "tools.bench.sota.run", "apriltag3", fam])
            .arg(list)
            .arg(out_dir.join("apriltag3.jsonl"))
            .arg(&threads);
        ("apriltag3".to_string(), c)
    } else {
        return Err(format!("unknown detector spec {spec}").into());
    };
    // Rayon is the only Locus thread knob; pin every other BLAS/OpenMP pool so they
    // cannot oversubscribe the cores during timing.
    cmd.env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1");
    Ok(Some((label, cmd)))
}

/// Runs on one image list accumulate in a run directory: `detectors.txt` lists the
/// detectors whose JSONL belongs to the current `images.txt` (a new image list resets it),
/// and `failed.txt` records detectors that crashed, as `label<TAB>exit status`. A crashed
/// detector does not abort the others; its partial output is kept as `<label>.jsonl.failed`
/// and is never scored.
fn run_detectors(root: &Path, d: &Dataset, o: &Opts) -> Result<()> {
    let out_dir = runs_dir(root, d);
    fs::create_dir_all(&out_dir)?;
    let list = out_dir.join("images.txt");
    let previous_list = fs::read_to_string(&list).ok();
    let n = image_list(d, o.stride, &list)?;
    eprintln!("{}: {n} images", d.name);
    let same_images = previous_list.as_deref() == fs::read_to_string(&list).ok().as_deref();

    let mut cmds = Vec::new();
    for spec in &o.detectors {
        if let Some(c) = detector_cmd(root, d, spec, &list, &out_dir, o)? {
            cmds.push(c);
        }
    }
    let mut succeeded = Vec::new();
    let mut crashed = Vec::new();
    for chunk in cmds.chunks_mut(o.jobs) {
        let mut children = Vec::new();
        for (label, cmd) in chunk.iter_mut() {
            eprintln!("+ [{label}] {cmd:?}");
            children.push((label.clone(), cmd.spawn()?));
        }
        // Wait for every child before reporting, so none is left running.
        for (label, mut child) in children {
            let status = child.wait()?;
            if status.success() {
                succeeded.push(label);
            } else {
                eprintln!("xtask: detector {label} failed ({status}); continuing without it");
                let jsonl = out_dir.join(format!("{label}.jsonl"));
                fs::rename(&jsonl, out_dir.join(format!("{label}.jsonl.failed"))).ok();
                crashed.push(format!("{label}\t{status}"));
            }
        }
    }

    // Merge into the bookkeeping only now (read-modify-write), so a concurrent run on the
    // same image list does not lose this run's detectors or the other way round.
    let read_lines = |name: &str| -> Vec<String> {
        if !same_images {
            return Vec::new();
        }
        fs::read_to_string(out_dir.join(name))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    };
    let label_of = |line: &str| line.split('\t').next().unwrap_or_default().to_string();
    let touched: Vec<String> = succeeded
        .iter()
        .cloned()
        .chain(crashed.iter().map(|l| label_of(l)))
        .collect();
    let mut done = read_lines("detectors.txt");
    let mut failed = read_lines("failed.txt");
    done.retain(|l| !touched.contains(l));
    failed.retain(|l| !touched.contains(&label_of(l)));
    done.extend(succeeded);
    failed.extend(crashed);
    done.sort();
    let lines = |v: &[String]| {
        v.iter()
            .flat_map(|l| [l.as_str(), "\n"])
            .collect::<String>()
    };
    fs::write(out_dir.join("detectors.txt"), lines(&done))?;
    fs::write(out_dir.join("failed.txt"), lines(&failed))?;
    write_meta(root, d, o, n, &out_dir)
}

/// Run metadata required by `docs/engineering/constraints.md` §6: verified
/// hardware, build profile, thread count and environment — captured in the same
/// session as the run.
fn write_meta(root: &Path, d: &Dataset, o: &Opts, n_images: usize, out_dir: &Path) -> Result<()> {
    let esc = |s: &str| {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    };
    let lscpu = capture(Command::new("lscpu").env("LC_ALL", "C"));
    let model = lscpu
        .lines()
        .find(|l| l.starts_with("Model name:"))
        .map_or("unknown", |l| l["Model name:".len()..].trim());
    let meta = format!(
        "{{\n  \"dataset\": \"{}\",\n  \"images\": {n_images},\n  \"stride\": {},\n  \"threads\": {},\n  \"reps\": {},\n  \"jobs\": {},\n  \"timing_valid\": {},\n  \"git_rev\": \"{}\",\n  \"cpu_model\": \"{}\",\n  \"lscpu\": \"{}\",\n  \"kernel\": \"{}\",\n  \"rustc\": \"{}\",\n  \"opencv\": \"{OPENCV_TAG}\",\n  \"aruco_nano\": \"{ARUCO_NANO_COMMIT}\",\n  \"locus_build\": \"see README: wheel must be built with maturin develop --release\"\n}}\n",
        d.name,
        o.stride,
        o.threads,
        o.reps,
        o.jobs,
        o.jobs == 1,
        esc(&capture(
            Command::new("git")
                .current_dir(root)
                .args(["rev-parse", "HEAD"])
        )),
        esc(model),
        esc(&lscpu),
        esc(&capture(Command::new("uname").arg("-r"))),
        esc(&capture(Command::new("rustc").arg("--version"))),
    );
    fs::write(out_dir.join("meta.json"), meta)?;
    Ok(())
}

// ── score / report ───────────────────────────────────────────────────────────

fn score(root: &Path, d: &Dataset) -> Result<()> {
    let dir = runs_dir(root, d);
    sh(python()
        .args(["-m", "tools.bench.sota.score", &d.run])
        .arg(&dir)
        .arg(dir.join("score.json")))
}

fn report(root: &Path, d: &Dataset) -> Result<()> {
    let dir = runs_dir(root, d);
    let unsupported: Vec<String> = d
        .unsupported
        .iter()
        .map(|(n, why)| format!("{n}={why}"))
        .collect();
    sh(python()
        .args(["-m", "tools.bench.sota.report", &d.run])
        .arg(&dir)
        .args(unsupported))
}

fn scoreboard(root: &Path, args: &[&str]) -> Result<()> {
    let champion = match args {
        [] => "locus_standard",
        ["--champion", label] => label,
        _ => return Err("usage: cargo xtask sota scoreboard [--champion <run label>]".into()),
    };
    sh(python()
        .args(["-m", "tools.bench.sota.scoreboard"])
        .arg(sota_dir(root).join("runs"))
        .arg(champion))
}
