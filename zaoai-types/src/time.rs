use std::time::Instant;

pub struct ScopeTimer {
    name: &'static str,
    start: Instant,
}

impl ScopeTimer {
    pub fn new(name: &'static str) -> Self {
        log::info!("⏳ Starting {}...", name);

        Self {
            name,
            start: Instant::now(),
        }
    }
}

impl Drop for ScopeTimer {
    fn drop(&mut self) {
        log::info!("⏱ Finished {}! ({:.3?})", self.name, self.start.elapsed());
    }
}
