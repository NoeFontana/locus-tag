pub mod image;

pub struct Detection {
    pub id: u32,
    pub center: [f64; 2],
    pub corners: [[f64; 2]; 4],
    pub hamming: u32,
    pub decision_margin: f64,
}

pub fn core_info() -> String {
    "Locus Core v0.1.0 Engine".to_string()
}

