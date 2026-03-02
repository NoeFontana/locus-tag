use divan::bench;

fn main() {
    divan::main();
}

#[bench]
fn bench_noop(bencher: divan::Bencher) {
    bencher.bench_local(|| {
        // No-op
    });
}
